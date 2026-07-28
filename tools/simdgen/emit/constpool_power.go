package emit

// Constant pools on ppc64le.
//
// The file is constpool_power.go and not constpool_ppc64.go because _ppc64 is
// a GOARCH suffix; see the note in returns_power.go.
//
// # What clang emits
//
// Every kernel that reads a constant opens with the ELFv2 global-entry
// prologue, which computes the TOC pointer from the function's own address:
//
//	addis r2, r12, .TOC.@ha     R_PPC64_REL16_HA
//	addi  r2, r2,  .TOC.@l      R_PPC64_REL16_LO
//
// and then reaches each constant through it:
//
//	addis r5, r2, .rodata.cst8@toc@ha    R_PPC64_TOC16_HA
//	lfd   f0, .rodata.cst8@toc@l(r5)     R_PPC64_TOC16_LO_DS
//
// None of that works under Go, which neither sets r12 on entry nor maintains a
// TOC these objects were linked against. It is why 157 kernels are dropped
// here — the largest single coverage gap in the library.
//
// # Why this needs no PC-relative addressing
//
// Power9 has none, which for a long time looked like the obstacle: reaching an
// appended pool seemed to need either bcl/mflr — which clobbers the link
// register and so needs a save slot in a protected zone the kernels already use
// — or a dependency on whatever r2 held on entry.
//
// Neither is necessary. Go's own assembler materialises a symbol address in two
// instructions with no TOC involvement at all:
//
//	MOVD $·kernel(SB), R2   ->   lis  r2, hi
//	                             addi r2, r2, lo
//
// because Go builds non-PIE and the address is a link-time constant. That was
// verified by building and *running* a probe under emulated ppc64le, not
// reasoned about.
//
// # The rewrite
//
// So the pool is appended to the body exactly as on the other targets, the
// generator emits `MOVD $·kernel(SB), R2` in the prologue so r2 holds the
// function's own address, clang's two global-entry instructions are replaced
// with nops in place, and every TOC16 immediate is recomputed as a plain offset
// from the start of the function.
//
// Only immediates change inside the body, so nothing moves and every branch
// clang computed stays correct — which constpool.go's header explains is not
// optional.

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/objfile"
)

const (
	opAddi  = 14 // addi  rD, rA, SI
	opAddis = 15 // addis rD, rA, SI
	ppcNop  = 0x60000000
)

// tocRelocs are the relocation kinds that reach a constant through r2. The
// suffix decides how the value is encoded, not where it comes from.
func tocRelocKind(name string) (kind string, ok bool) {
	switch {
	case strings.HasSuffix(name, "TOC16_HA"), strings.HasSuffix(name, "TOC16_HI"):
		return "ha", true
	case strings.HasSuffix(name, "TOC16_LO_DS"):
		return "lods", true
	case strings.HasSuffix(name, "TOC16_LO"):
		return "lo", true
	case strings.HasSuffix(name, "TOC16_DS"):
		return "ds", true
	}
	return "", false
}

// isGlobalEntryReloc reports whether a relocation is one of the two that make
// up the global-entry prologue. They point at .TOC., which is a linker-defined
// symbol rather than a section this generator can copy, and they are the two
// this rewrite replaces outright.
func isGlobalEntryReloc(name string) bool {
	return strings.HasSuffix(name, "REL16_HA") || strings.HasSuffix(name, "REL16_LO") ||
		strings.HasSuffix(name, "REL16_HI") || strings.HasSuffix(name, "REL16")
}

// hasGlobalEntryPrologue reports whether the body opens with the two
// instructions that compute r2 from r12.
func hasGlobalEntryPrologue(code []byte) bool {
	if len(code) < 8 {
		return false
	}
	w0 := binary.LittleEndian.Uint32(code[0:])
	w1 := binary.LittleEndian.Uint32(code[4:])
	return w0>>26 == opAddis && (w0>>21)&31 == 2 && (w0>>16)&31 == 12 &&
		w1>>26 == opAddi && (w1>>21)&31 == 2 && (w1>>16)&31 == 2
}

// canLiftPPC64 reports whether a kernel's out-of-function references are ones
// this rewrite can resolve.
func canLiftPPC64(fn *objfile.Func) (bool, string) {
	if len(fn.Code)%4 != 0 {
		return false, "body is not a whole number of instructions"
	}
	sawGlobalEntry := false
	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) {
			continue
		}
		if isGlobalEntryReloc(rel.TypeName) {
			if rel.Off >= 8 {
				return false, fmt.Sprintf("computes the TOC pointer at +0x%x rather than "+
					"in the global entry prologue, so r2 is not settled by the time the "+
					"body uses it", rel.Off)
			}
			sawGlobalEntry = true
			continue
		}
		if _, ok := tocRelocKind(rel.TypeName); !ok {
			return false, fmt.Sprintf("has a %s relocation at +0x%x, which is not a TOC "+
				"reference this generator rewrites", rel.TypeName, rel.Off)
		}
		if len(rel.Target) == 0 {
			return false, fmt.Sprintf("relocation at +0x%x refers to %s, which is not a "+
				"constant pool this generator can copy", rel.Off, rel.Sym)
		}
	}
	if !sawGlobalEntry {
		return false, "reaches a constant through r2 without the global entry prologue " +
			"that would have set it, so there is nothing to repoint"
	}
	if !hasGlobalEntryPrologue(fn.Code) {
		return false, "does not open with addis r2,r12 / addi r2,r2, so the two " +
			"instructions this rewrite replaces are not where it expects them"
	}
	return true, ""
}

// resolvePoolPPC64 replaces the global-entry prologue with nops and rewrites
// each TOC16 immediate into an offset from the start of a separate pool symbol,
// which it returns alongside the patched code.
//
// The pool is a standalone GLOBL rather than bytes appended to the body, and
// that is load-bearing. The first version appended it and pointed r2 at
// `$·name(SB)` — the TEXT symbol — while computing offsets from the *body*.
// The generator-emitted prologue sits between the two, so every constant
// address was short by the prologue's length. It did not crash: it read
// neighbouring instruction bytes, which decode as perfectly plausible data, and
// surfaced as int8 multiply returning 9 where it should have returned 0.
//
// A separate symbol has no such gap. Offsets are from the pool's own base, and
// the prologue's length stops mattering.
func resolvePoolPPC64(fn *objfile.Func) ([]byte, []byte, error) {
	code := append([]byte(nil), fn.Code...)

	// Lay the referenced sections out in one pool, each 16-byte aligned so a
	// vector load of it is aligned as clang expected.
	bases := map[string]uint64{}
	var pool []byte
	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) || isGlobalEntryReloc(rel.TypeName) {
			continue
		}
		if _, ok := tocRelocKind(rel.TypeName); !ok {
			continue
		}
		if _, seen := bases[rel.TargetSection]; seen {
			continue
		}
		for len(pool)%16 != 0 {
			pool = append(pool, 0)
		}
		bases[rel.TargetSection] = uint64(len(pool))
		pool = append(pool, rel.Target...)
	}

	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) {
			continue
		}
		if int(rel.Off)+4 > len(code) {
			return nil, nil, fmt.Errorf("relocation at +0x%x is past the end of the body", rel.Off)
		}

		// The global-entry prologue is replaced outright: r2 is set by the
		// prologue the generator emits, so these two compute nothing anyone
		// reads. They are nopped rather than removed because removing them
		// would move every instruction after them.
		if isGlobalEntryReloc(rel.TypeName) {
			binary.LittleEndian.PutUint32(code[rel.Off:], ppcNop)
			continue
		}

		kind, ok := tocRelocKind(rel.TypeName)
		if !ok {
			return nil, nil, fmt.Errorf("relocation %s at +0x%x is not one this generator rewrites",
				rel.TypeName, rel.Off)
		}
		base, seen := bases[rel.TargetSection]
		if !seen {
			return nil, nil, fmt.Errorf("relocation at +0x%x targets %s, which was not laid out",
				rel.Off, rel.TargetSection)
		}
		// r2 will hold the pool symbol's address, so each half encodes the
		// constant's offset within the pool.
		v := int64(base) + rel.Addend

		w := binary.LittleEndian.Uint32(code[rel.Off:])
		switch kind {
		case "ha":
			// The high half is adjusted so that adding a sign-extended low
			// half gives the right result.
			w = w&0xffff0000 | uint32(((v+0x8000)>>16)&0xffff)
		case "lo":
			w = w&0xffff0000 | uint32(v&0xffff)
		case "lods", "ds":
			// DS-form: the displacement occupies bits 2..15 and the low two
			// bits belong to the opcode, so they are preserved.
			if v&3 != 0 {
				return nil, nil, fmt.Errorf("relocation at +0x%x needs a 4-byte aligned "+
					"displacement but the constant is at %d", rel.Off, v)
			}
			w = w&0xffff0003 | uint32(v&0xfffc)
		}
		binary.LittleEndian.PutUint32(code[rel.Off:], w)
	}

	return code, pool, nil
}
