package emit

// The file is named _aarch64 rather than _arm64 on purpose: Go reads a trailing
// _arm64 as a build constraint, and this code has to compile on whatever
// machine runs the generator, not on the machine it generates for.

import (
	"debug/elf"
	"encoding/binary"
	"fmt"

	"github.com/sebishogun/simd/tools/simdgen/objfile"
)

// Constant pools on AArch64, resolved by turning a page-relative address into
// a byte-relative one.
//
// # The problem
//
// clang reaches a constant pool with a pair:
//
//	adrp x11, .LCPI0_0          // page containing the pool
//	ldr  d1, [x11, :lo12:...]   // offset within that page
//
// adrp computes page(target) - page(pc), which depends on where the linker
// finally places the code. Nothing this generator can compute at build time
// will produce the right 21-bit immediate, and re-spelling the pair as Plan 9
// instructions that Go's own linker would relocate does not fit either: the
// shortest correct spelling is three instructions where the original is two,
// and a body that changes length invalidates every PC-relative branch across
// it. That is the failure that made a reciprocal kernel's remainder loop
// unreachable, and it is why nothing here is ever re-spelled.
//
// # The way through
//
// ADR is the same four bytes as ADRP and differs in one bit, but its immediate
// is a byte offset from the instruction rather than a page count. Once the
// pool is appended to the function's own body — which is what happens on
// amd64 too — the distance from the instruction to the constant is a number
// this generator knows exactly. So:
//
//	adrp xN, sym            ->  adr xN, #(pool - here)
//	ldr  dM, [xN, :lo12:s]  ->  ldr dM, [xN, #off]
//
// Both edits are in place and neither changes a length, so every branch that
// was correct in the object file is still correct. ADR reaches +-1MB, which is
// three orders of magnitude more than the largest kernel here needs.
//
// The ADR points at the base of the copied section rather than at the
// individual constant, and each load carries its own offset. That matters
// because LLVM shares one adrp between several loads at different offsets in
// the same pool; pointing at one of them would silently give the others the
// wrong number.
//
// Unaligned loads are architecturally fine on AArch64 for normal memory, so
// nothing here depends on where inside the TEXT symbol the pool lands. That is
// deliberate: Go's linker takes a -funcalign flag, so text alignment is not
// something a library may rely on.

// AArch64 relocation types this understands. Anything else is a hard error:
// the kernel is dropped and the portable implementation kept, which is always
// safe, whereas guessing at an encoding would produce a wrong number rather
// than a crash.
const (
	relAdrPrelPgHi21 = uint32(elf.R_AARCH64_ADR_PREL_PG_HI21)
	relAddAbsLo12Nc  = uint32(elf.R_AARCH64_ADD_ABS_LO12_NC)
	relLdst8Lo12Nc   = uint32(elf.R_AARCH64_LDST8_ABS_LO12_NC)
	relLdst16Lo12Nc  = uint32(elf.R_AARCH64_LDST16_ABS_LO12_NC)
	relLdst32Lo12Nc  = uint32(elf.R_AARCH64_LDST32_ABS_LO12_NC)
	relLdst64Lo12Nc  = uint32(elf.R_AARCH64_LDST64_ABS_LO12_NC)
	relLdst128Lo12Nc = uint32(elf.R_AARCH64_LDST128_ABS_LO12_NC)
)

// lo12Scale is the shift applied to the 12-bit immediate of the instruction a
// relocation patches. A load scales its offset by the access width; ADD does
// not scale at all.
func lo12Scale(t uint32) (shift uint, ok bool) {
	switch t {
	case relAddAbsLo12Nc, relLdst8Lo12Nc:
		return 0, true
	case relLdst16Lo12Nc:
		return 1, true
	case relLdst32Lo12Nc:
		return 2, true
	case relLdst64Lo12Nc:
		return 3, true
	case relLdst128Lo12Nc:
		return 4, true
	}
	return 0, false
}

// resolvePoolARM64 appends every referenced constant pool to the body and
// rewrites the instructions that reach it.
func resolvePoolARM64(fn *objfile.Func) ([]byte, error) {
	code := append([]byte(nil), fn.Code...)

	// Lay out the pools first, so a relocation patched in the second pass
	// already knows where its section landed. Each is 16-byte aligned within
	// the body, which is not needed for correctness but keeps every load on
	// its natural boundary when the enclosing symbol happens to be aligned.
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

	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) {
			continue
		}
		if rel.Off+4 > uint64(len(code)) {
			return nil, fmt.Errorf("relocation at +0x%x runs past the end of the body", rel.Off)
		}
		base := bases[rel.TargetSection]
		word := binary.LittleEndian.Uint32(code[rel.Off:])

		switch rel.Type {
		case relAdrPrelPgHi21:
			delta := int64(base) - int64(rel.Off)
			if delta < -(1<<20) || delta >= 1<<20 {
				return nil, fmt.Errorf("constant pool is %d bytes from the adrp at +0x%x, "+
					"outside ADR's +-1MB reach", delta, rel.Off)
			}
			if word&0x9F000000 != 0x90000000 {
				return nil, fmt.Errorf("the instruction at +0x%x carries an "+
					"ADR_PREL_PG_HI21 relocation but is not an adrp (word 0x%08x)",
					rel.Off, word)
			}
			// Keep Rd, clear the op bit to make it an ADR, and write the byte
			// offset as immlo:immhi.
			rd := word & 0x1F
			imm := uint32(delta) & 0x1FFFFF
			word = 0x10000000 | ((imm & 3) << 29) | ((imm >> 2) << 5) | rd

		case relAddAbsLo12Nc, relLdst8Lo12Nc, relLdst16Lo12Nc,
			relLdst32Lo12Nc, relLdst64Lo12Nc, relLdst128Lo12Nc:
			shift, _ := lo12Scale(rel.Type)
			off := rel.TargetOff
			if off&((1<<shift)-1) != 0 {
				return nil, fmt.Errorf("constant at +0x%x in %s is not %d-byte aligned, "+
					"so it cannot be reached by a scaled offset",
					off, rel.TargetSection, 1<<shift)
			}
			imm := off >> shift
			if imm >= 1<<12 {
				return nil, fmt.Errorf("constant at +0x%x in %s is beyond the 12-bit "+
					"offset the load can encode", off, rel.TargetSection)
			}
			word = (word &^ (0xFFF << 10)) | uint32(imm)<<10

		default:
			return nil, fmt.Errorf("relocation %s at +0x%x is not one this generator "+
				"knows how to rewrite", rel.TypeName, rel.Off)
		}
		binary.LittleEndian.PutUint32(code[rel.Off:], word)
	}
	return append(code, tail...), nil
}

// canLiftARM64 reports whether every reference out of the function is one
// resolvePoolARM64 can rewrite.
func canLiftARM64(fn *objfile.Func) (bool, string) {
	for _, r := range fn.Relocs {
		if isSelfRelative(r.TypeName) {
			continue
		}
		if len(r.Target) == 0 {
			return false, "references " + r.Sym + ", which is not a constant pool"
		}
		switch r.Type {
		case relAdrPrelPgHi21:
		case relAddAbsLo12Nc, relLdst8Lo12Nc, relLdst16Lo12Nc,
			relLdst32Lo12Nc, relLdst64Lo12Nc, relLdst128Lo12Nc:
			shift, _ := lo12Scale(r.Type)
			if r.TargetOff>>shift >= 1<<12 {
				return false, fmt.Sprintf("constant pool offset 0x%x is beyond a "+
					"12-bit scaled load offset", r.TargetOff)
			}
		default:
			return false, "uses relocation " + r.TypeName + ", which this generator " +
				"does not rewrite"
		}
	}
	return true, ""
}
