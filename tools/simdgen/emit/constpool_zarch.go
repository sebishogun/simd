package emit

import (
	"debug/elf"
	"encoding/binary"
	"fmt"

	"github.com/sebishogun/simd/tools/simdgen/objfile"
)

// Constant pools on s390x, which are the easiest of the set.
//
// clang reaches a pool with a single instruction:
//
//	larl %r5, .LCPI0_0      // load address relative long
//	vl   %v0, 0(%r5), 3
//
// larl takes a signed 32-bit displacement counted in halfwords from the start
// of the instruction, giving it a reach of +-4GB with no page alignment
// anywhere in the encoding. Once the pool is appended to the function's own
// body the distance is a number this generator knows, so the field can simply
// be written. Nothing changes length and no instruction needs re-spelling.
//
// The relocation's addend is +2 rather than 0, because the four-byte field
// sits two bytes into the six-byte instruction while the displacement is
// measured from the instruction's start. objfile.pcAdjust takes that back out,
// so TargetOff here is the constant's own position in the pool.

const relS390PC32DBL = uint32(elf.R_390_PC32DBL)

// resolvePoolS390X appends every referenced pool to the body and writes the
// halfword displacement that reaches it.
func resolvePoolS390X(fn *objfile.Func) ([]byte, error) {
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
		// Eight-byte aligned within the body. vl tolerates any address, so this
		// is for the load unit rather than for correctness.
		for (uint64(len(code))+uint64(len(tail)))%8 != 0 {
			tail = append(tail, 0)
		}
		bases[rel.TargetSection] = uint64(len(code)) + uint64(len(tail))
		tail = append(tail, rel.Target...)
	}

	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) {
			continue
		}
		if rel.Type != relS390PC32DBL {
			return nil, fmt.Errorf("relocation %s at +0x%x is not one this generator "+
				"knows how to rewrite", rel.TypeName, rel.Off)
		}
		if rel.Off+4 > uint64(len(code)) {
			return nil, fmt.Errorf("relocation at +0x%x runs past the end of the body", rel.Off)
		}
		target := int64(bases[rel.TargetSection] + rel.TargetOff)
		// The displacement is from the instruction, which starts two bytes
		// before the field, and is counted in halfwords.
		delta := target - (int64(rel.Off) - 2)
		if delta%2 != 0 {
			return nil, fmt.Errorf("constant at +0x%x is at an odd address, which larl "+
				"cannot reach", target)
		}
		half := delta / 2
		if half < -(1<<31) || half >= 1<<31 {
			return nil, fmt.Errorf("constant pool is %d bytes away, out of larl's reach", delta)
		}
		// s390x is big-endian, and so is its instruction encoding.
		binary.BigEndian.PutUint32(code[rel.Off:], uint32(int32(half)))
	}
	return append(code, tail...), nil
}

func canLiftS390X(fn *objfile.Func) (bool, string) {
	for _, r := range fn.Relocs {
		if isSelfRelative(r.TypeName) {
			continue
		}
		if len(r.Target) == 0 {
			return false, "references " + r.Sym + ", which is not a constant pool"
		}
		if r.Type != relS390PC32DBL {
			return false, "uses relocation " + r.TypeName + ", which this generator " +
				"does not rewrite"
		}
	}
	return true, ""
}
