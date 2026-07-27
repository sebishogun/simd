package emit

import (
	"encoding/binary"
	"fmt"

	"github.com/sebishogun/simd/tools/simdgen/objfile"
)

// Constant pools on RISC-V.
//
// clang emits the usual two-instruction address:
//
//	auipc a5, %pcrel_hi(.LCPI0_0)    // a5 = pc + (hi20 << 12)
//	fld   fa5, %pcrel_lo(.Lpcrel_hi0)(a5)
//
// auipc is already PC-relative in bytes rather than pages, so unlike AArch64
// and LoongArch nothing has to be turned into a different instruction — the
// two immediates simply have to be filled in, which is possible once the pool
// sits at a known offset in the function's own body.
//
// The awkward part is the pairing. R_RISCV_PCREL_LO12's symbol is not the
// constant: it is a label placed on the auipc, because the low half's value
// depends on where the high half was, not on where the low half is. So the
// relocation carries no usable target of its own and the low half has to find
// its high half some other way.
//
// It finds it by register. The load's base register is the auipc's
// destination, and that is exact — where pairing by proximity is not, because
// clang interleaves the pairs: two auipc into different registers, then the
// two loads. Taking the nearest preceding high half there reads a constant
// from somewhere else in the pool, and any eight bytes decode as a perfectly
// good double, so the mistake would surface as a wrong number rather than a
// crash.
const (
	relRiscvPcrelHi20  = 23 // R_RISCV_PCREL_HI20
	relRiscvPcrelLo12I = 24 // R_RISCV_PCREL_LO12_I
	relRiscvPcrelLo12S = 25 // R_RISCV_PCREL_LO12_S
)

// resolvePoolRISCV64 appends every referenced pool to the body and fills in the
// two halves of each address.
func resolvePoolRISCV64(fn *objfile.Func) ([]byte, error) {
	code := append([]byte(nil), fn.Code...)

	bases := map[string]uint64{}
	var tail []byte
	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) || rel.Type != relRiscvPcrelHi20 {
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

	// The high half carries its resolved target forward, because the low half
	// cannot look it up. A PCREL_LO12's symbol is a label on the auipc, so its
	// TargetSection is .text — asking bases for it returns the zero value, and
	// the address computed from that lands a few bytes *before* the function
	// rather than in the pool. Every constant read that way is instruction
	// bytes, which decode as a perfectly plausible double, so the symptom is a
	// transcendental returning +Inf rather than anything that looks like a
	// pointer bug.
	type hiHalf struct {
		at     uint64
		hi20   int64
		target int64 // absolute offset of the constant in the emitted body
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

		switch rel.Type {
		case relRiscvPcrelHi20:
			if word&0x7F != 0x17 { // AUIPC opcode
				return nil, fmt.Errorf("the instruction at +0x%x carries a PCREL_HI20 "+
					"relocation but is not an auipc (word 0x%08x)", rel.Off, word)
			}
			target := int64(bases[rel.TargetSection] + rel.TargetOff)
			delta := target - int64(rel.Off)
			// The low half is a signed 12-bit field, so the high half rounds to
			// the nearest 4096 rather than truncating toward zero.
			hi20 := (delta + 0x800) >> 12
			if hi20 < -(1<<19) || hi20 >= 1<<19 {
				return nil, fmt.Errorf("constant pool is %d bytes away, out of auipc's reach",
					delta)
			}
			rd := (word >> 7) & 0x1F
			live[rd] = hiHalf{at: rel.Off, hi20: hi20, target: target}
			word = (word & 0xFFF) | uint32(hi20)<<12

		case relRiscvPcrelLo12I, relRiscvPcrelLo12S:
			rs1 := (word >> 15) & 0x1F
			hi, ok := live[rs1]
			if !ok {
				return nil, fmt.Errorf("the PCREL_LO12 at +0x%x uses x%d, which no "+
					"preceding PCREL_HI20 wrote", rel.Off, rs1)
			}
			// auipc left rs1 holding hi.at + (hi20 << 12); the load adds lo to
			// it, so lo is whatever closes the gap to the constant the high
			// half was pointed at.
			lo := hi.target - (int64(hi.at) + (hi.hi20 << 12))
			if lo < -2048 || lo > 2047 {
				return nil, fmt.Errorf("constant is %d bytes from what the high half at "+
					"+0x%x computed, outside a signed 12-bit offset", lo, hi.at)
			}
			if rel.Type == relRiscvPcrelLo12I {
				word = (word & 0x000FFFFF) | uint32(lo&0xFFF)<<20
			} else {
				// S-type splits the immediate: bits 11:5 at 31:25, bits 4:0 at 11:7.
				word = (word &^ 0xFE000F80) |
					uint32((lo>>5)&0x7F)<<25 | uint32(lo&0x1F)<<7
			}

		default:
			return nil, fmt.Errorf("relocation %s at +0x%x is not one this generator "+
				"knows how to rewrite", rel.TypeName, rel.Off)
		}
		binary.LittleEndian.PutUint32(code[rel.Off:], word)
	}
	return append(code, tail...), nil
}

func canLiftRISCV64(fn *objfile.Func) (bool, string) {
	for _, r := range fn.Relocs {
		if isSelfRelative(r.TypeName) {
			continue
		}
		switch r.Type {
		case relRiscvPcrelHi20:
			if len(r.Target) == 0 {
				return false, "references " + r.Sym + ", which is not a constant pool"
			}
		case relRiscvPcrelLo12I, relRiscvPcrelLo12S:
			// Its symbol is a label on the auipc, so there is nothing to check
			// here; the pairing is verified when it is resolved.
		default:
			return false, "uses relocation " + r.TypeName + ", which this generator " +
				"does not rewrite"
		}
	}
	return true, ""
}
