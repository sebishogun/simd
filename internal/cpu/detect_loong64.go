//go:build loong64

package cpu

import "golang.org/x/sys/cpu"

// detect reports the tiers this CPU supports, weakest first.
//
// LSX is 128-bit, LASX is 256-bit. LASX implies LSX on every shipped
// implementation, so the tiers nest.
func detect() []Tier {
	t := []Tier{Scalar}
	if cpu.Loong64.HasLSX {
		t = append(t, LSX)
	}
	if cpu.Loong64.HasLASX {
		t = append(t, LASX)
	}
	return t
}
