//go:build s390x

package cpu

import "golang.org/x/sys/cpu"

// detect reports the tiers this CPU supports, weakest first.
//
// VX is the z13 vector facility; VXE adds the vector-enhancements facility.
// VXE implies VX, so the tiers nest.
func detect() []Tier {
	t := []Tier{Scalar}
	if cpu.S390X.HasVX {
		t = append(t, VX)
	}
	if cpu.S390X.HasVXE {
		t = append(t, VXE)
	}
	return t
}
