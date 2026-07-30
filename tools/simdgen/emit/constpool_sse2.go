package emit

import (
	"fmt"
	"sort"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/target"

	"github.com/sebishogun/simd/tools/simdgen/objfile"
)

// Re-spelling the pool reads that a legacy SSE instruction makes.
//
// # The problem this solves
//
// constpool.go appends the constant pool to the function's own body and patches
// the displacement, which is length-preserving and therefore safe. It cannot
// help a legacy SSE arithmetic instruction — mulps, pand, pxor and the rest —
// because those require their memory operand to be 16-byte aligned and have no
// unaligned form to switch to, and nothing promises the alignment of bytes
// inside a TEXT symbol. 170 baseline-tier kernels were dropped for that reason,
// the largest single bucket in the repository.
//
// # Why the objection on file did not settle it
//
// The comment on checkPatchable rejects padding the appended pool to a 16-byte
// offset and relying on cmd/link aligning text symbols to 32, because -funcalign
// makes that not a guarantee. That is correct, and it is about appending the
// pool *inside the TEXT symbol*. It does not apply to a separate DATA/GLOBL
// symbol: -funcalign aligns text, and the linker aligns data by its size.
// Measured, with and without -ldflags=-funcalign=8, a 32-byte RODATA symbol
// landed 16-byte aligned both times. See docs/wrong.md entry 61.
//
// # Why re-spelling was abandoned before, and why it is safe here
//
// The header of constpool.go records the first attempt: rewriting the
// instruction as a Plan 9 mnemonic produced a *different length*, so every
// PC-relative branch spanning it jumped to the wrong place — silently, with a
// remainder loop that was simply never entered.
//
// That is a fact about particular instructions, not about the technique, and it
// is checkable. Every mnemonic in the table below was assembled both ways and
// compared:
//
//	Go     MULPS simdpool<>+0x20(SB), X3   ->  0f 59 1d <disp32>
//	clang  mulps 0x1234(%rip), %xmm3       ->  0f 59 1d <disp32>
//
// Byte-identical, including the mandatory 66 prefix and the REX byte that
// xmm8-xmm15 require, and in the same order. So the replacement occupies
// exactly the bytes it replaces and every branch displacement in the body
// remains correct. TestRespelledEncodingsMatchClang checks this rather than
// trusting it, because "should be the same encoding" is precisely the kind of
// claim this repository exists to disbelieve.
//
// # What is checked at emission time
//
// The table is not enough on its own — it says a mnemonic is safe, not that
// this particular instruction has the shape the table assumed. So each
// candidate is also verified structurally against its own bytes:
//
//   - the relocation occupies the last four bytes of the instruction,
//   - the ModRM byte selects a RIP-relative operand (mod=00, rm=101),
//   - the two-byte opcode escape 0F is where the encoding requires,
//   - the register is decoded from ModRM and REX.R rather than from text.
//
// Anything that does not match is an error, and an error drops the kernel back
// to the portable path exactly as before. The failure mode of this change is
// losing a kernel, not miscompiling one.

// poolForm is a mnemonic's Plan 9 spelling and, for the compare family, the
// predicate byte its name stands for.
type poolForm struct {
	plan9 string
	// imm is the trailing immediate the encoding carries, or -1 for none.
	// The compares are one instruction with eight names: cmpnltps is
	// CMPPS with predicate 5. That byte sits after the displacement, so it
	// also changes where the relocation is relative to the end.
	imm int
}

// respellable maps an x86 mnemonic to its Plan 9 form.
//
// Every entry was assembled both ways and compared byte for byte, including
// with a destination in xmm8-xmm15 where the encoding gains a REX prefix; see
// TestRespelledEncodingsMatchClang. Nothing goes in this table on the strength
// of looking obviously equivalent.
//
// Two spellings are not the uppercase form. Plan 9 names the doubleword
// integer operations after the operand width — PADDL for paddd, PCMPGTL for
// pcmpgtd, PCMPEQL for pcmpeqd — and the compares by their base mnemonic with
// the predicate as an operand.
var respellable = map[string]poolForm{
	"mulps":   {"MULPS", -1},
	"mulpd":   {"MULPD", -1},
	"addps":   {"ADDPS", -1},
	"addpd":   {"ADDPD", -1},
	"por":     {"POR", -1},
	"pand":    {"PAND", -1},
	"pxor":    {"PXOR", -1},
	"pcmpgtd": {"PCMPGTL", -1},
	"pcmpeqd": {"PCMPEQL", -1},
	"xorps":   {"XORPS", -1},
	"xorpd":   {"XORPD", -1},
	"andps":   {"ANDPS", -1},
	"andpd":   {"ANDPD", -1},
	"pminub":  {"PMINUB", -1},
	"pmaxub":  {"PMAXUB", -1},
	"paddb":   {"PADDB", -1},
	"paddw":   {"PADDW", -1},
	"paddd":   {"PADDL", -1},
	"paddq":   {"PADDQ", -1},
	"movq":    {"MOVQ", -1},
	"pandn":   {"PANDN", -1},
	"andnpd":  {"ANDNPD", -1},
	"andnps":  {"ANDNPS", -1},
	"psubb":   {"PSUBB", -1},
	"psubw":   {"PSUBW", -1},
	"psubd":   {"PSUBL", -1},
	"psubq":   {"PSUBQ", -1},
	"pmulhuw": {"PMULHUW", -1},
	"pmaddwd": {"PMADDWL", -1},
	"pminsw":  {"PMINSW", -1},
	"pmaxsw":  {"PMAXSW", -1},

	// The packed compare family. The predicate encoding is fixed by the ISA:
	// eq 0, lt 1, le 2, unord 3, neq 4, nlt 5, nle 6, ord 7.
	//
	// Only the packed forms. The scalar compares — cmpltss and friends — read
	// four or eight bytes and have no alignment requirement at all, so they
	// belong in unalignedLoads and keep the appended-pool treatment. Putting
	// them here instead demanded a 16-byte-aligned pool offset they had no
	// reason to satisfy, and two kernels failed the assertion that says so.
	"cmpeqps":  {"CMPPS", 0},
	"cmpltps":  {"CMPPS", 1},
	"cmpleps":  {"CMPPS", 2},
	"cmpneqps": {"CMPPS", 4},
	"cmpnltps": {"CMPPS", 5},
	"cmpnleps": {"CMPPS", 6},
	"cmpeqpd":  {"CMPPD", 0},
	"cmpltpd":  {"CMPPD", 1},
	"cmplepd":  {"CMPPD", 2},
	"cmpneqpd": {"CMPPD", 4},
	"cmpnltpd": {"CMPPD", 5},
	"cmpnlepd": {"CMPPD", 6},
}

// poolAlign is the alignment every section in the re-spelled pool is placed
// at. Sixteen is what a legacy SSE memory operand requires; the linker gives
// the symbol itself at least that, because it aligns a data symbol by its size
// and these pools are hundreds of bytes.
const poolAlign = 16

// respelled is one instruction replaced by a mnemonic, and the bytes it stands
// in for.
type poolInstr struct {
	off    int    // where the instruction starts in the body
	n      int    // how many bytes it occupies, which the replacement must match
	relOff uint64 // the relocation this stands in for, so it is not recomputed
	line   string // the Plan 9 form
}

// needsRespell reports whether any pool read in this function is a legacy SSE
// instruction that cannot be patched to tolerate an unaligned address.
func needsRespell(fn *objfile.Func, instrs []Instr) bool {
	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) || len(rel.Target) == 0 {
			continue
		}
		if in, ok := instrAt(instrs, rel.Off); ok {
			if checkPatchable(in) != nil {
				return true
			}
		}
	}
	return false
}

// respellSSE2 builds the replacement lines and the RODATA pool they read.
//
// Only the references that cannot be patched are re-spelled. The rest keep the
// appended-pool treatment, which is already correct for them; the two
// mechanisms coexist in one function, and a section needed by both is simply
// copied twice. A few duplicated bytes of read-only data is a better trade than
// a second set of rules about which pool a reference belongs to.
func respellSSE2(fn *objfile.Func, instrs []Instr, code []byte, sym string) ([]poolInstr, []byte, error) {
	var (
		refs     []poolInstr
		pool     []byte
		base     = map[string]int{}
		respells = map[uint64]bool{}
	)
	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) || len(rel.Target) == 0 {
			continue
		}
		in, ok := instrAt(instrs, rel.Off)
		if !ok {
			return nil, nil, fmt.Errorf("no disassembly for the instruction at +0x%x", rel.Off)
		}
		if checkPatchable(in) == nil {
			continue // the appended pool and an opcode patch already handle it
		}
		form, ok := respellable[in.Mnemonic]
		if !ok {
			return nil, nil, fmt.Errorf("reads a constant pool with %q, which requires "+
				"16-byte alignment and has no verified Plan 9 spelling of the same "+
				"length; see respellable in constpool_sse2.go", in.Mnemonic)
		}

		n, err := instrLen(instrs, in, len(code))
		if err != nil {
			return nil, nil, err
		}
		start := int(in.Offset)
		// The displacement is followed by the immediate, where there is one,
		// so the tail is four bytes or five. Anything else means the operand
		// is not the plain RIP-relative form this rewrite assumes.
		tail := 4
		if form.imm >= 0 {
			tail = 5
		}
		if int(rel.Off)+tail != start+n {
			return nil, nil, fmt.Errorf("%s at +0x%x: the relocation is %d bytes from "+
				"the end of the instruction, not %d, so the operand is not the plain "+
				"RIP-relative form this rewrite assumes",
				in.Mnemonic, rel.Off, start+n-int(rel.Off), tail)
		}
		if form.imm >= 0 {
			// The predicate in the bytes must be the one the mnemonic names.
			// If clang ever spells a predicate differently from this table,
			// that disagreement is caught here rather than emitted.
			if got := int(code[int(rel.Off)+4]); got != form.imm {
				return nil, nil, fmt.Errorf("%s at +0x%x carries predicate %d, but the "+
					"mnemonic means %d", in.Mnemonic, rel.Off, got, form.imm)
			}
		}
		reg, err := ripRegister(code, int(rel.Off), start)
		if err != nil {
			return nil, nil, fmt.Errorf("%s at +0x%x: %w", in.Mnemonic, rel.Off, err)
		}

		off, seen := base[rel.TargetSection]
		if !seen {
			// Pad to the section's own alignment before appending it. The
			// symbol as a whole is aligned by the linker, but a second section
			// laid down straight after a first whose length is not a multiple
			// of sixteen starts misaligned — and the whole point of this path
			// is that these instructions fault on a misaligned operand. The
			// first version did not pad and produced a reference to
			// pool+0xa9f, which segfaulted in fastSigmoidFloat64SSE2 the
			// moment the differential suite ran it.
			align := int(rel.TargetAlign)
			if align < poolAlign {
				align = poolAlign
			}
			off = (len(pool) + align - 1) &^ (align - 1)
			for len(pool) < off {
				pool = append(pool, 0)
			}
			pool = append(pool, rel.Target...)
			base[rel.TargetSection] = off
		}
		// The immediate is also why the target offset needs correcting.
		// A PC-relative displacement is measured from the end of the whole
		// instruction, so on a form with a trailing imm8 the assembler emits
		// an addend of -5 rather than -4. objfile adds back a fixed +4, which
		// leaves TargetOff one byte short — and one byte short of a 16-byte
		// boundary is exactly the misalignment these instructions fault on.
		// It cost a segfault in fastSigmoidFloat64SSE2 pointing at pool+0xa9f
		// where the constant is at 0xaa0.
		target := off + int(rel.TargetOff)
		line := fmt.Sprintf("%s\t%s<>+0x%x(SB), X%d", form.plan9, sym, target, reg)
		if form.imm >= 0 {
			target++
			line = fmt.Sprintf("%s\t%s<>+0x%x(SB), X%d, $%d", form.plan9, sym,
				target, reg, form.imm)
		}
		if target%poolAlign != 0 {
			return nil, nil, fmt.Errorf("%s at +0x%x would read the pool at +0x%x, "+
				"which is not %d-byte aligned; a legacy SSE operand faults there",
				in.Mnemonic, rel.Off, target, poolAlign)
		}
		refs = append(refs, poolInstr{off: start, n: n, relOff: rel.Off, line: line})
		respells[rel.Off] = true
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].off < refs[j].off })
	for i := 1; i < len(refs); i++ {
		if refs[i-1].off+refs[i-1].n > refs[i].off {
			return nil, nil, fmt.Errorf("re-spelled instructions at +0x%x and +0x%x overlap",
				refs[i-1].off, refs[i].off)
		}
	}
	return refs, pool, nil
}

// instrLen returns how many bytes an instruction occupies, from the offset of
// the one after it.
func instrLen(instrs []Instr, in Instr, codeLen int) (int, error) {
	for i, cur := range instrs {
		if cur.Offset != in.Offset {
			continue
		}
		end := codeLen
		if i+1 < len(instrs) {
			end = int(instrs[i+1].Offset)
		}
		if n := end - int(in.Offset); n > 0 {
			return n, nil
		}
		return 0, fmt.Errorf("instruction at +0x%x has a non-positive length", in.Offset)
	}
	return 0, fmt.Errorf("instruction at +0x%x is not in the disassembly", in.Offset)
}

// ripRegister decodes the register operand of a RIP-relative instruction from
// its own bytes.
//
// The encoding is fixed: [legacy prefixes] [REX] 0F opcode ModRM disp32, with
// the displacement last. So from the relocation offset, ModRM is one byte back,
// the opcode two, and the 0F escape three. ModRM must be mod=00 rm=101, which
// is what makes the operand RIP-relative rather than a register or a base
// register; the register field is bits 3-5, extended to 8-15 by REX.R.
//
// Decoded from bytes rather than parsed from the disassembly text, because the
// bytes are what the replacement has to agree with.
func ripRegister(code []byte, relOff, start int) (int, error) {
	if relOff-3 < start {
		return 0, fmt.Errorf("instruction is too short to carry a two-byte opcode")
	}
	if code[relOff-3] != 0x0f {
		return 0, fmt.Errorf("expected the 0F opcode escape at +0x%x, found 0x%02x",
			relOff-3, code[relOff-3])
	}
	modrm := code[relOff-1]
	if modrm&0xc7 != 0x05 {
		return 0, fmt.Errorf("ModRM 0x%02x is not the RIP-relative form (mod=00, rm=101)", modrm)
	}
	reg := int(modrm>>3) & 7
	// REX, if present, is the last prefix before the opcode escape.
	if relOff-4 >= start {
		if b := code[relOff-4]; b&0xf0 == 0x40 {
			if b&0x04 != 0 { // REX.R
				reg += 8
			}
		}
	}
	return reg, nil
}

// emitBodyRespelled writes the body with the re-spelled instructions as
// mnemonics and everything else as raw bytes.
//
// The mnemonic stands in for exactly the bytes it replaces, so the offsets of
// everything after it are unchanged and every branch displacement computed by
// clang is still correct. That equality is the whole safety argument, and it is
// why respellSSE2 refuses anything whose encoding it has not verified.
func emitBodyRespelled(b *strings.Builder, code []byte, refs []poolInstr, tgt target.Target) error {
	at := 0
	for _, r := range refs {
		if r.off < at || r.off+r.n > len(code) {
			return fmt.Errorf("re-spelled instruction at +0x%x is out of range", r.off)
		}
		if err := emitBody(b, code[at:r.off], tgt); err != nil {
			return err
		}
		fmt.Fprintf(b, "\t%s\n", r.line)
		at = r.off + r.n
	}
	return emitBody(b, code[at:], tgt)
}
