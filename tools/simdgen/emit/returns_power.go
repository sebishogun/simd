package emit

// Making a ppc64le kernel end somewhere the generator controls.
//
// The file is returns_power.go and not returns_ppc64.go because _ppc64 is a
// GOARCH suffix: Go would build this file only when compiling *for* PowerPC,
// and the generator runs on the host. Its sibling constant-pool files are
// named for the same reason.
//
// Go's ppc64 ABI reserves r0 as a *value*, not just as a name. From
// cmd/compile/abi-internal.md: "R0 | Zero value | Same | Same" — the register
// holds zero on call, on return and throughout a function body, and the
// compiler relies on it. PPC64Ops.go lowers a struct zeroing to a run of
// "MOVD R0,(R3)", storing the register on the understanding that its contents
// are the zero it is writing.
//
// clang knows none of this. r0 is volatile scratch in ELFv2 and LLVM allocates
// it freely — 212 writes to it across the kernel sources here. There is no way
// to stop it: unlike arm64 and riscv64, clang's PowerPC target has no
// -ffixed-rN at all, and a global register variable for it is accepted and
// then ignored, exactly as on SystemZ.
//
// The symptom is not a wrong answer from the kernel. The kernel is correct and
// returns; the *next* allocation in Go comes back non-zero, and the process
// dies somewhere inside the runtime with an index out of range in the
// allocator or a nil dereference in the garbage collector, several calls away
// from anything to do with this library. One "li r0, -1" in an otherwise empty
// assembly function is enough to reproduce it.
//
// So r0 is restored on the way out. That needs a place that always runs, and a
// compiled body has none: LLVM lays basic blocks out after the return, so
// there is no position at the end that is guaranteed to be reached. What is
// guaranteed is that control leaves through one of the body's own returns, and
// on this architecture every one of them is a bclr — a single instruction with
// a known encoding. Rewriting each into a branch to the end of the body makes
// the end of the body the single exit, and the two-instruction epilogue after
// it (target.Target.Epilogue) then always runs.

import (
	"encoding/binary"
	"fmt"

	"github.com/sebishogun/simd/tools/simdgen/target"
)

// PowerPC primary opcodes and extended opcodes this file rewrites or refuses.
const (
	opBC     = 16 // bc: conditional branch, 14-bit signed word displacement
	opB      = 18 // b: unconditional branch, 24-bit signed word displacement
	opBCLR   = 19 // primary opcode of the branch-to-register family
	xoBCLR   = 16 // bclr:  branch to the link register
	xoBCCTR  = 528
	xoBCTAR  = 560
	boAlways = 0x14 // BO0 and BO2 set: "branch always", ignoring the condition
)

// retargetReturns rewrites every return in a compiled body into a branch to
// the first instruction past it, so that the epilogue emitted there runs on
// every path out.
//
// It is a no-op on every architecture but ppc64le, which is the only one whose
// ABI reserves a general register by value rather than by name; see the file
// comment. The rewrite is exact: bclr and bc take the same BO and BI fields in
// the same bit positions, so a conditional return becomes the same condition
// branching forward, and an unconditional one becomes a plain b.
func retargetReturns(code []byte, tgt target.Target) ([]byte, error) {
	if tgt.Arch != target.PPC64LE || len(tgt.Epilogue) == 0 {
		return code, nil
	}
	if len(code)%4 != 0 {
		return nil, fmt.Errorf("emit: ppc64le body is %d bytes, not a whole "+
			"number of instructions", len(code))
	}
	out := append([]byte(nil), code...)
	n := len(out) / 4
	for i := range n {
		w := binary.LittleEndian.Uint32(out[i*4:])
		if w>>26 != opBCLR || (w>>1)&0x3ff != xoBCLR || w&1 != 0 {
			continue
		}
		bo, bi := (w>>21)&0x1f, (w>>16)&0x1f
		// Distance from this instruction to the byte after the body, which is
		// where the epilogue starts. Positive and small: the largest kernel
		// here is under 1500 bytes.
		d := uint32((n - i) * 4)
		var rewritten uint32
		if bo&boAlways == boAlways {
			if d > 0x01fffffc {
				return nil, fmt.Errorf("emit: ppc64le return at +%d is %d bytes "+
					"from the end of the body, past the range of b", i*4, d)
			}
			rewritten = opB<<26 | d&0x03fffffc
		} else {
			if d > 0x7ffc {
				return nil, fmt.Errorf("emit: ppc64le return at +%d is %d bytes "+
					"from the end of the body, past the range of bc", i*4, d)
			}
			rewritten = opBC<<26 | bo<<21 | bi<<16 | d&0xfffc
		}
		binary.LittleEndian.PutUint32(out[i*4:], rewritten)
	}
	return out, nil
}

// canRetargetPPC64 reports whether every way out of a ppc64le body is a return
// this package can rewrite.
//
// Three encodings would defeat the rewrite and each is refused rather than
// mishandled. A blrl calls through the link register; a bcctr or bctar leaves
// through a register this package cannot follow, which is what a jump table
// compiles to. None has ever appeared — jump tables are off and there are no
// calls — so this is a guard against a future kernel, not a known case.
func canRetargetPPC64(code []byte) (bool, string) {
	if len(code)%4 != 0 {
		return false, fmt.Sprintf("body is %d bytes, not a whole number of instructions",
			len(code))
	}
	for i := 0; i < len(code); i += 4 {
		w := binary.LittleEndian.Uint32(code[i:])
		if w>>26 != opBCLR {
			continue
		}
		switch xo := (w >> 1) & 0x3ff; {
		case xo == xoBCLR && w&1 == 1:
			return false, "calls through the link register, so its return cannot be retargeted"
		case xo == xoBCCTR || xo == xoBCTAR:
			return false, "branches through a register, so its exits cannot be enumerated"
		}
	}
	return true, ""
}
