//go:build arm64

package cpu

import "golang.org/x/sys/cpu"

// detect reports the tiers this CPU supports, weakest first.
//
// NEON is unconditional: ASIMD is architecturally mandatory on AArch64, which
// is why internal/cpu has no HasASIMD field and why the standard library's
// index_arm64.go sets its tuning constant without any check.
//
// SVE2 is scalable: the vector length is a boot-time property of the machine,
// not a compile-time constant. Kernels in this tier must be written and tested
// against every length; see internal/testutil.
//
// HasSVE2 is only reachable through x/sys/cpu. The standard library's
// internal/cpu ARM64 struct has neither HasSVE nor HasSVE2, and Go's assembler
// cannot encode a single SVE instruction, so this tier exists solely because
// the codegen pipeline emits raw instruction words.
func detect() []Tier {
	t := []Tier{Scalar, NEON}
	if cpu.ARM64.HasSVE2 {
		t = append(t, SVE2)
	}
	return t
}
