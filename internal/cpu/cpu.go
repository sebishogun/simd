// Package cpu detects which SIMD instruction-set tier the running CPU supports.
//
// Detection uses golang.org/x/sys/cpu rather than the standard library's
// internal/cpu, which is not importable from third-party modules. x/sys/cpu is
// also the only source of ARM64.HasSVE and HasSVE2; internal/cpu exposes
// neither.
//
// Nothing in this package executes a SIMD instruction. That is deliberate: a
// package-level variable initializer that runs a vector instruction executes
// before the dispatch check that is supposed to guard it, which is how
// go-highway#69 produced SIGILL on non-AVX2 CPUs.
package cpu

import (
	"os"
	"runtime"
	"strings"
)

// Tier identifies a SIMD instruction-set level. Tiers are architecture
// specific; only those belonging to the running GOARCH are ever reported.
// Within an architecture the constants are ordered weakest to strongest, so
// the highest available tier is the numerically largest one.
type Tier uint8

const (
	// Scalar is the portable Go fallback. Available on every architecture.
	Scalar Tier = iota

	// amd64
	SSE2   // baseline on amd64; no runtime check needed
	AVX2   // requires AVX2 + FMA
	AVX512 // requires the F+CD+BW+DQ+VL bundle

	// arm64
	NEON // ASIMD, architecturally mandatory on AArch64
	SVE2 // scalable; vector length is a runtime property

	// riscv64
	RVV // RVV 1.0; scalable, like SVE2

	// s390x
	VX  // z13 vector facility
	VXE // vector enhancements

	// loong64
	LSX  // 128-bit
	LASX // 256-bit

	// ppc64/ppc64le
	VSX // VMX/VSX, POWER8+

	numTiers
)

var tierNames = [numTiers]string{
	Scalar: "scalar",
	SSE2:   "sse2",
	AVX2:   "avx2",
	AVX512: "avx512",
	NEON:   "neon",
	SVE2:   "sve2",
	RVV:    "rvv",
	VX:     "vx",
	VXE:    "vxe",
	LSX:    "lsx",
	LASX:   "lasx",
	VSX:    "vsx",
}

func (t Tier) String() string {
	if t < numTiers && tierNames[t] != "" {
		return tierNames[t]
	}
	return "tier(" + string(rune('0'+int(t))) + ")"
}

// ParseTier maps a tier name to a Tier. Names are the lowercase strings
// returned by Tier.String.
func ParseTier(s string) (Tier, bool) {
	s = strings.ToLower(strings.TrimSpace(s))
	for i, n := range tierNames {
		if n != "" && n == s {
			return Tier(i), true
		}
	}
	return Scalar, false
}

// Selection records the outcome of tier selection, including why a tier was
// chosen. Tests and diagnostics use it; the hot path does not.
type Selection struct {
	// Tier is the selected tier.
	Tier Tier
	// Available lists every tier this CPU supports, weakest first. It always
	// begins with Scalar.
	Available []Tier
	// Forced is true if GOSIMD selected the tier rather than detection.
	Forced bool
	// Disabled lists tiers masked out by SIMD_DISABLE.
	Disabled []Tier
	// Reason explains a fallback to Scalar caused by an override. Empty when
	// selection proceeded normally.
	Reason string
}

var selection Selection

func init() { selection = selectTier(os.Getenv("GOSIMD"), os.Getenv("SIMD_DISABLE")) }

// Selected returns the tier this process will dispatch to.
func Selected() Tier { return selection.Tier }

// Detail returns the full selection outcome, for diagnostics and tests.
func Detail() Selection { return selection }

// selectTier is the pure core of selection, split out so tests can drive it
// with arbitrary environment values.
func selectTier(gosimd, disable string) Selection {
	sel := Selection{Available: detect()}

	// SIMD_DISABLE masks tiers out of consideration. Scalar can never be
	// masked; there would be nothing left to run.
	if disable != "" {
		for _, name := range strings.Split(disable, ",") {
			t, ok := ParseTier(name)
			if !ok || t == Scalar {
				continue
			}
			if remove(&sel.Available, t) {
				sel.Disabled = append(sel.Disabled, t)
			}
		}
	}

	// GOSIMD pins an exact tier. It can only select down: naming a tier this
	// CPU does not support would mean executing instructions it cannot decode,
	// so that falls back to Scalar with a reason rather than crashing.
	if gosimd != "" {
		sel.Forced = true
		t, ok := ParseTier(gosimd)
		switch {
		case !ok:
			sel.Tier, sel.Reason = Scalar, "GOSIMD="+gosimd+": unknown tier"
		case !contains(sel.Available, t):
			sel.Tier, sel.Reason = Scalar, "GOSIMD="+gosimd+": not available on this "+runtime.GOARCH+" CPU"
		default:
			sel.Tier = t
		}
		return sel
	}

	sel.Tier = sel.Available[len(sel.Available)-1]
	return sel
}

func contains(ts []Tier, t Tier) bool {
	for _, x := range ts {
		if x == t {
			return true
		}
	}
	return false
}

func remove(ts *[]Tier, t Tier) bool {
	for i, x := range *ts {
		if x == t {
			*ts = append((*ts)[:i], (*ts)[i+1:]...)
			return true
		}
	}
	return false
}

// Describe renders the selection for logs and test failure messages.
func Describe() string {
	s := selection
	var b strings.Builder
	b.WriteString(runtime.GOARCH)
	b.WriteString(" tier=")
	b.WriteString(s.Tier.String())
	b.WriteString(" available=[")
	for i, t := range s.Available {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(t.String())
	}
	b.WriteByte(']')
	if s.Forced {
		b.WriteString(" forced")
	}
	if len(s.Disabled) > 0 {
		b.WriteString(" disabled=[")
		for i, t := range s.Disabled {
			if i > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(t.String())
		}
		b.WriteByte(']')
	}
	if s.Reason != "" {
		b.WriteString(" reason=")
		b.WriteString(s.Reason)
	}
	return b.String()
}
