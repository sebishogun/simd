//go:build ppc64 || ppc64le

package cpu

import "golang.org/x/sys/cpu"

// detect reports the tiers this CPU supports, weakest first.
//
// There is no per-feature VSX bit: the ISA level implies it, which is why
// x/sys/cpu exposes IsPOWER8/IsPOWER9 rather than HasVSX. POWER8 (ISA 2.07) is
// the floor for the VSX kernels this library generates.
func detect() []Tier {
	t := []Tier{Scalar}
	if cpu.PPC64.IsPOWER8 || cpu.PPC64.IsPOWER9 {
		t = append(t, VSX)
	}
	return t
}
