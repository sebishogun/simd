// Package objfile reads a compiled object file and extracts what is needed to
// re-emit one function as Plan 9 assembly.
//
// # Why this is simpler than it looks
//
// A function's internal branches are PC-relative and already encoded in its
// instruction bytes, so copying the bytes verbatim keeps every one of them
// valid. Nothing has to be decoded, re-spelled or re-linked for the code to
// work — the whole blob moves together.
//
// The only things that need attention are references *out* of the function:
// relocations. Measured across an eight-kernel suite there are very few, and
// they are concentrated:
//
//	arm64    0 relocations, 0 undefined symbols   — constants materialised inline
//	s390x    2 (R_390_PC32DBL)
//	riscv64  4, all R_RISCV_BRANCH                — internal, self-relative
//	amd64    12, all PC-relative into a local constant pool, and confined to
//	         the two kernels that use float constants
//	ppc64le  TOC-relative, plus an undefined .TOC. — the one awkward target
//
// So the reader's job is: find the symbol, take its bytes, and report any
// relocation that lands inside it along with the data it points at.
package objfile

import (
	"debug/elf"
	"encoding/binary"
	"fmt"
	"sort"
)

// Func is one extracted function.
type Func struct {
	Name string
	// Code is the function's instruction bytes, copied verbatim.
	Code []byte
	// Relocs are the relocations that land inside Code, sorted by offset and
	// with offsets relative to the start of Code.
	Relocs []Reloc
}

// Reloc is a relocation inside an extracted function.
type Reloc struct {
	// Off is the byte offset within Func.Code of the field to patch.
	Off uint64
	// Type is the architecture-specific relocation type.
	Type uint32
	// TypeName is the type spelled out, for diagnostics.
	TypeName string
	// Sym is the symbol being referenced.
	Sym string
	// Addend is the relocation addend.
	Addend int64
	// Target is the data the relocation points at, when the referenced symbol
	// is a local constant pool this package could resolve. Nil otherwise.
	Target []byte
	// TargetAlign is the alignment of the referenced section.
	TargetAlign uint64
}

// File is an opened object file.
type File struct {
	elf   *elf.File
	syms  []elf.Symbol
	close func() error
}

// Open reads an ELF object file.
func Open(path string) (*File, error) {
	f, err := elf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("objfile: %w", err)
	}
	syms, err := f.Symbols()
	if err != nil {
		// A stripped object has no symbol table, which means we cannot locate
		// the function. That is a hard error, not something to work around.
		f.Close()
		return nil, fmt.Errorf("objfile %s: no symbol table: %w", path, err)
	}
	return &File{elf: f, syms: syms, close: f.Close}, nil
}

// Close releases the file.
func (f *File) Close() error { return f.close() }

// UndefinedSymbols returns every symbol the object needs from elsewhere.
//
// This must be empty. Plan 9 assembly has no procedure linkage table, so a
// call to an undefined symbol cannot be resolved and the generated code would
// jump into nothing. It is the single most important precondition, and it is
// why the clang flags disable builtins: without -fno-builtin a scalar
// remainder loop can be pattern-matched into a call to memset.
func (f *File) UndefinedSymbols() []string {
	var out []string
	for _, s := range f.syms {
		if s.Section == elf.SHN_UNDEF && s.Name != "" {
			out = append(out, s.Name)
		}
	}
	sort.Strings(out)
	return out
}

// Func extracts the named function.
func (f *File) Func(name string) (*Func, error) {
	sym, ok := f.symbol(name)
	if !ok {
		return nil, fmt.Errorf("objfile: no symbol %q (have %v)", name, f.definedNames())
	}
	if sym.Size == 0 {
		return nil, fmt.Errorf("objfile: symbol %q has zero size", name)
	}
	if int(sym.Section) >= len(f.elf.Sections) {
		return nil, fmt.Errorf("objfile: symbol %q has an out-of-range section index", name)
	}
	sec := f.elf.Sections[sym.Section]
	data, err := sec.Data()
	if err != nil {
		return nil, fmt.Errorf("objfile: reading %s: %w", sec.Name, err)
	}
	if sym.Value+sym.Size > uint64(len(data)) {
		return nil, fmt.Errorf("objfile: symbol %q extends past section %s", name, sec.Name)
	}

	code := make([]byte, sym.Size)
	copy(code, data[sym.Value:sym.Value+sym.Size])

	relocs, err := f.relocsIn(sec, sym.Value, sym.Size)
	if err != nil {
		return nil, err
	}
	return &Func{Name: name, Code: code, Relocs: relocs}, nil
}

func (f *File) symbol(name string) (elf.Symbol, bool) {
	for _, s := range f.syms {
		if s.Name == name && s.Section != elf.SHN_UNDEF {
			return s, true
		}
	}
	return elf.Symbol{}, false
}

func (f *File) definedNames() []string {
	var out []string
	for _, s := range f.syms {
		if s.Section != elf.SHN_UNDEF && s.Name != "" && elf.ST_TYPE(s.Info) == elf.STT_FUNC {
			out = append(out, s.Name)
		}
	}
	sort.Strings(out)
	return out
}

// relocsIn collects the relocations applying to [start, start+size) of sec.
func (f *File) relocsIn(sec *elf.Section, start, size uint64) ([]Reloc, error) {
	var out []Reloc
	for _, rs := range f.elf.Sections {
		if rs.Type != elf.SHT_RELA || int(rs.Info) >= len(f.elf.Sections) {
			continue
		}
		if f.elf.Sections[rs.Info] != sec {
			continue
		}
		data, err := rs.Data()
		if err != nil {
			return nil, fmt.Errorf("objfile: reading %s: %w", rs.Name, err)
		}
		const relaSize = 24 // Elf64_Rela
		if len(data)%relaSize != 0 {
			return nil, fmt.Errorf("objfile: %s has a truncated entry", rs.Name)
		}
		bo := f.elf.ByteOrder
		for off := 0; off+relaSize <= len(data); off += relaSize {
			var (
				rOff    = bo.Uint64(data[off:])
				rInfo   = bo.Uint64(data[off+8:])
				rAddend = int64(bo.Uint64(data[off+16:]))
			)
			if rOff < start || rOff >= start+size {
				continue
			}
			symIdx := int(rInfo >> 32)
			rel := Reloc{
				Off:      rOff - start,
				Type:     uint32(rInfo),
				TypeName: relocTypeName(f.elf.Machine, uint32(rInfo)),
				Addend:   rAddend,
			}
			if symIdx > 0 && symIdx <= len(f.syms) {
				s := f.syms[symIdx-1] // Symbols() omits the null entry
				rel.Sym = s.Name
				if tgt, align, ok := f.symbolData(s); ok {
					rel.Target = tgt
					rel.TargetAlign = align
				}
			}
			out = append(out, rel)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Off < out[j].Off })
	return out, nil
}

// symbolData returns the bytes a data symbol refers to.
//
// clang names a constant pool with a section symbol plus an addend rather than
// a distinctly named one, so the whole referenced section is returned and the
// addend selects within it.
func (f *File) symbolData(s elf.Symbol) ([]byte, uint64, bool) {
	if s.Section == elf.SHN_UNDEF || int(s.Section) >= len(f.elf.Sections) {
		return nil, 0, false
	}
	sec := f.elf.Sections[s.Section]
	if sec.Type != elf.SHT_PROGBITS || sec.Flags&elf.SHF_EXECINSTR != 0 {
		return nil, 0, false
	}
	data, err := sec.Data()
	if err != nil {
		return nil, 0, false
	}
	return data, sec.Addralign, true
}

// Machine reports the object's architecture.
func (f *File) Machine() elf.Machine { return f.elf.Machine }

// ByteOrder reports the object's endianness.
func (f *File) ByteOrder() binary.ByteOrder { return f.elf.ByteOrder }

// relocTypeName spells a relocation type for diagnostics. Only the machines
// this generator targets are covered; anything else prints numerically, which
// is enough to act on.
func relocTypeName(m elf.Machine, t uint32) string {
	switch m {
	case elf.EM_X86_64:
		return elf.R_X86_64(t).String()
	case elf.EM_AARCH64:
		return elf.R_AARCH64(t).String()
	case elf.EM_RISCV:
		return elf.R_RISCV(t).String()
	case elf.EM_S390:
		return elf.R_390(t).String()
	case elf.EM_PPC64:
		return elf.R_PPC64(t).String()
	case elf.EM_LOONGARCH:
		return elf.R_LARCH(t).String()
	}
	return fmt.Sprintf("reloc(%d)", t)
}
