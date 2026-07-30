package emit

import (
	"encoding/binary"
	"fmt"

	"github.com/sebishogun/simd/tools/simdgen/objfile"
)

// Constant pools on LoongArch, by the same move as on AArch64: replace a
// page-relative address computation with one that is not page-relative.
//
// clang emits
//
//	pcalau12i $a3, .LCPI0_0     // (pc & ~0xFFF) + (imm20 << 12)
//	fld.d     $fa1, $a3, lo12
//
// and the page term cannot be computed until the linker has placed the code.
// pcaddu12i is the same size and the same shape but computes pc + (imm20 << 12)
// with the low bits of pc left in place, so with the pool appended to the
// function's body the whole address is known here. The two differ only in the
// seven-bit primary opcode.
//
// The split is the usual one: imm20 rounds the distance to the nearest 4096
// and the load's signed 12-bit offset carries the remainder, which is why
// imm20 is computed with a +0x800 bias rather than by truncation.
const (
	relLarchPcalaHi20 = 71 // R_LARCH_PCALA_HI20
	relLarchPcalaLo12 = 72 // R_LARCH_PCALA_LO12

	// Primary opcodes, bits 31:25.
	opPcalau12i = 0x0D
	opPcaddu12i = 0x0E
)

// resolvePoolLoong64 appends every referenced pool to the body and rewrites the
// instructions that reach it.
func resolvePoolLoong64(fn *objfile.Func) ([]byte, error) {
	code := append([]byte(nil), fn.Code...)

	bases := map[string]uint64{}
	var tail []byte
	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) {
			continue
		}
		if len(rel.Target) == 0 {
			return nil, fmt.Errorf("relocation at +0x%x refers to %s, which is not a "+
				"constant pool this generator can copy", rel.Off, rel.Sym)
		}
		if _, seen := bases[rel.TargetSection]; seen {
			continue
		}
		for (uint64(len(code))+uint64(len(tail)))%16 != 0 {
			tail = append(tail, 0)
		}
		bases[rel.TargetSection] = uint64(len(code)) + uint64(len(tail))
		tail = append(tail, rel.Target...)
	}

	// The high and low halves of one address are two separate relocations at
	// two separate instructions, and both need the same split of the same
	// distance. The high one is rewritten first and its choice recorded, so
	// the low one does not have to recompute it and cannot disagree.
	//
	// They are paired by register, not by proximity. clang interleaves them —
	// two pcalau12i into different registers, then the two loads — so pairing
	// a low half with the nearest preceding high half picks the wrong one and
	// reads a constant from somewhere else in the pool. That failure is
	// silent: any eight bytes decode as a perfectly good double.
	type hiHalf struct {
		at    uint64
		imm20 int64
	}
	live := map[uint32]hiHalf{} // destination register -> its high half

	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) {
			continue
		}
		if rel.Off+4 > uint64(len(code)) {
			return nil, fmt.Errorf("relocation at +0x%x runs past the end of the body", rel.Off)
		}
		word := binary.LittleEndian.Uint32(code[rel.Off:])
		target := int64(bases[rel.TargetSection] + rel.TargetOff)

		switch rel.Type {
		case relLarchPcalaHi20:
			if word>>25 != opPcalau12i {
				return nil, fmt.Errorf("the instruction at +0x%x carries a PCALA_HI20 "+
					"relocation but is not a pcalau12i (word 0x%08x)", rel.Off, word)
			}
			delta := target - int64(rel.Off)
			imm20 := (delta + 0x800) >> 12
			if imm20 < -(1<<19) || imm20 >= 1<<19 {
				return nil, fmt.Errorf("constant pool is %d bytes away, out of "+
					"pcaddu12i's reach", delta)
			}
			rd := word & 0x1F
			live[rd] = hiHalf{at: rel.Off, imm20: imm20}
			word = uint32(opPcaddu12i)<<25 | (uint32(imm20)&0xFFFFF)<<5 | rd

		case relLarchPcalaLo12:
			// The base register of the load names which high half this belongs
			// to. rj is bits 9:5.
			rj := (word >> 5) & 0x1F
			hi, ok := live[rj]
			if !ok {
				return nil, fmt.Errorf("the PCALA_LO12 at +0x%x uses r%d, which no "+
					"preceding PCALA_HI20 wrote", rel.Off, rj)
			}
			lo := target - (int64(hi.at) + (hi.imm20 << 12))
			if lo < -2048 || lo > 2047 {
				return nil, fmt.Errorf("constant at +0x%x is %d bytes from what the "+
					"high half computed, outside a signed 12-bit offset", target, lo)
			}
			word = (word &^ (0xFFF << 10)) | (uint32(lo)&0xFFF)<<10

		default:
			return nil, fmt.Errorf("relocation %s at +0x%x is not one this generator "+
				"knows how to rewrite", rel.TypeName, rel.Off)
		}
		binary.LittleEndian.PutUint32(code[rel.Off:], word)
	}
	return append(code, tail...), nil
}

func canLiftLoong64(fn *objfile.Func) (bool, string) {
	for _, r := range fn.Relocs {
		if isSelfRelative(r.TypeName) {
			continue
		}
		// A branch relocation reaches here rather than being skipped above,
		// because on LoongArch the displacement is not in the instruction yet
		// — see isSelfRelative. It falls into the len(r.Target) == 0 case
		// below and is reported as ".L0, which is not a constant pool", which
		// names the symbol accurately even if the phrasing is about pools.
		if len(r.Target) == 0 {
			return false, "references " + r.Sym + ", which is not a constant pool"
		}
		switch r.Type {
		case relLarchPcalaHi20, relLarchPcalaLo12:
		default:
			return false, "uses relocation " + r.TypeName + ", which this generator " +
				"does not rewrite"
		}
	}
	return true, ""
}
