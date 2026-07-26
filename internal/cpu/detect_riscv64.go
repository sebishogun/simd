//go:build riscv64

package cpu

import "golang.org/x/sys/cpu"

// detect reports the tiers this CPU supports, weakest first.
//
// RVV 1.0 is scalable, like SVE2: VLEN is a property of the implementation and
// kernels select an element count at runtime with vsetvli. The same multi
// vector-length test discipline applies.
func detect() []Tier {
	t := []Tier{Scalar}
	if cpu.RISCV64.HasV {
		t = append(t, RVV)
	}
	return t
}
