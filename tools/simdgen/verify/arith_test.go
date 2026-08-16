package verify

import (
	"testing"

	"github.com/sebishogun/simd/tools/simdgen/target"
)

// arithKind separates work from moves, and packed work from scalar work.
//
// The mnemonics below are copied from llvm-objdump output for kernels in this
// repository, not invented: the amd64 scalar cases are what simd_convolve_f32
// actually contains at sse2 and avx2, and the "neither" cases are the two
// instructions that made it pass as vectorized.
func TestArithKindSeparatesPackedFromScalarAndFromMoves(t *testing.T) {
	for _, tc := range []struct {
		arch     target.Arch
		mnemonic string
		operands string
		want     arithForm
	}{
		// amd64, scalar: one lane, in an %xmm register, which is why
		// vectorWidth counts them as vector.
		{target.AMD64, "mulss", "(%rsi), %xmm0", arithScalar},
		{target.AMD64, "addss", "%xmm1, %xmm0", arithScalar},
		{target.AMD64, "vmulss", "%xmm2, %xmm1, %xmm0", arithScalar},
		{target.AMD64, "vaddss", "%xmm2, %xmm1, %xmm0", arithScalar},
		{target.AMD64, "mulsd", "%xmm1, %xmm0", arithScalar},
		{target.AMD64, "vfmadd213sd", "%xmm2, %xmm1, %xmm0", arithScalar},
		{target.AMD64, "divss", "%xmm1, %xmm0", arithScalar},
		{target.AMD64, "maxsd", "%xmm1, %xmm0", arithScalar},

		// amd64, packed.
		{target.AMD64, "mulps", "%xmm1, %xmm0", arithPacked},
		{target.AMD64, "vaddps", "%ymm2, %ymm1, %ymm0", arithPacked},
		{target.AMD64, "vfmadd213ps", "%zmm2, %zmm1, %zmm0", arithPacked},
		{target.AMD64, "vmulpd", "%ymm2, %ymm1, %ymm0", arithPacked},
		{target.AMD64, "paddd", "%xmm1, %xmm0", arithPacked},
		{target.AMD64, "vpmulld", "%ymm2, %ymm1, %ymm0", arithPacked},
		{target.AMD64, "vpmaxsd", "%ymm2, %ymm1, %ymm0", arithPacked},
		{target.AMD64, "vpsubb", "%xmm2, %xmm1, %xmm0", arithPacked},

		// amd64, NEITHER. These two are the whole finding: they are the only
		// vector-register instructions in simd_convolve_f32 at sse2, and
		// counting them is what let a scalar kernel ship as accelerated.
		{target.AMD64, "movups", "(%rdi), %xmm0", arithNone},
		{target.AMD64, "xorps", "%xmm0, %xmm0", arithNone},
		{target.AMD64, "vmovups", "(%rdi), %ymm0", arithNone},
		{target.AMD64, "vmovss", "(%rsi), %xmm0", arithNone},
		{target.AMD64, "movq", "%rax, %rcx", arithNone},
		{target.AMD64, "vshufps", "$0, %xmm0, %xmm0, %xmm0", arithNone},

		// arm64: the OPERAND decides, because `fmul` spells both forms.
		{target.ARM64, "fmul", "v0.4s, v1.4s, v2.4s", arithPacked},
		{target.ARM64, "fadd", "v0.2d, v1.2d, v2.2d", arithPacked},
		{target.ARM64, "fmul", "s0, s1, s2", arithScalar},
		{target.ARM64, "fadd", "d0, d1, d2", arithScalar},
		{target.ARM64, "fmul", "z0.s, p0/m, z0.s, z1.s", arithPacked},
		{target.ARM64, "ldr", "q0, [x0]", arithNone},
		{target.ARM64, "mov", "v0.16b, v1.16b", arithNone},

		// An architecture the classifier does not claim: everything is
		// unclassified, and ArithKnown is what says so.
		{target.PPC64LE, "xvmuldp", "vs0, vs1, vs2", arithNone},
		{target.S390X, "vfmdb", "%v0, %v1, %v2", arithNone},
	} {
		got := arithKind(Instr{Mnemonic: tc.mnemonic, Operands: tc.operands}, tc.arch)
		if got != tc.want {
			t.Errorf("%s %s %q = %v, want %v", tc.arch, tc.mnemonic, tc.operands, got, tc.want)
		}
	}
}

// ScalarOnly is false where the classifier does not run, so a zero count is
// never read as a finding.
//
// Without this, adding an architecture to the report and forgetting to add it
// to arithKind would mark every one of its kernels scalar-only -- a check that
// fires everywhere is as useless as one that fires nowhere, and this one would
// have been noise on four of the six architectures.
func TestScalarOnlyIsSilentWhereTheClassifierDoesNotRun(t *testing.T) {
	for _, arch := range []target.Arch{target.AMD64, target.ARM64, target.PPC64LE,
		target.S390X, target.RISCV64, target.LOONG64} {
		known := arithClassified(arch)
		r := Report{ArithKnown: known, ScalarArith: 10, PackedArith: 0}
		if got := r.ScalarOnly(); got != known {
			t.Errorf("%s: ScalarOnly()=%v with ArithKnown=%v; an unclassified "+
				"architecture must not report a finding it did not measure", arch, got, known)
		}
	}
	// And where it does run, the three states are distinct.
	for _, tc := range []struct {
		name           string
		packed, scalar int
		want           bool
	}{
		{"vectorized", 12, 0, false},
		{"mixed", 12, 50, false},
		{"scalar only", 0, 10, true},
		{"a permute kernel: no arithmetic at all", 0, 0, false},
	} {
		r := Report{ArithKnown: true, PackedArith: tc.packed, ScalarArith: tc.scalar}
		if got := r.ScalarOnly(); got != tc.want {
			t.Errorf("%s (packed=%d scalar=%d) = %v, want %v",
				tc.name, tc.packed, tc.scalar, got, tc.want)
		}
	}
}
