package conformance

// The narrow floating-point conversions, kernel against reference.
//
// The inputs are not random. Both domains are small enough to enumerate — a
// float16 and a bfloat16 each have 65536 values — so the widening direction is
// checked completely rather than sampled, and the narrowing direction is fed
// every float32 that a widening produced plus the boundaries where its
// behaviour changes: the largest representable value, the first that
// overflows, the denormal edge, and exact halfway cases that must round to
// even.
//
// That matters here more than elsewhere. A wrong magic constant in the
// denormal path is right for every normal value and wrong only for the
// smallest few thousand, which no sampled test would find; one did survive
// into a first draft of the kernel and only enumeration caught it.

import (
	"math"
	"testing"

	"github.com/sebishogun/simd/internal/ref"
)

func TestConvertKernels(t *testing.T) {
	want := ref.Set()
	const n = 1 << 16
	all16 := make([]uint16, n)
	for i := range all16 {
		all16[i] = uint16(i)
	}

	// Every float32 the widenings produce, plus the narrowing boundaries.
	wide := make([]float32, n)
	want.Convert.F16ToF32(wide, all16)
	edges := []float32{
		0, float32(math.Copysign(0, -1)),
		65504, 65519, 65520, 65535, 131008,
		6.1035156e-05, 5.9604645e-08, 2.9802322e-08, 8.940697e-08,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
		1e-45, 3.4028235e+38, -3.4028235e+38,
	}
	f32in := append(append([]float32(nil), wide...), edges...)

	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			for _, c := range []struct {
				op        string
				got, want func(dst []float32, a []uint16)
			}{
				{"BF16ToF32", got.Convert.BF16ToF32, want.Convert.BF16ToF32},
				{"F16ToF32", got.Convert.F16ToF32, want.Convert.F16ToF32},
			} {
				if c.got == nil || c.want == nil {
					continue
				}
				g, w := make([]float32, n), make([]float32, n)
				c.got(g, all16)
				c.want(w, all16)
				for i := range g {
					if !sameF32Bits(g[i], w[i]) {
						t.Fatalf("%s/Convert.%s(%#04x): got %v (%#08x) want %v (%#08x)",
							tier, c.op, all16[i], g[i], math.Float32bits(g[i]),
							w[i], math.Float32bits(w[i]))
					}
				}
			}
			for _, c := range []struct {
				op        string
				got, want func(dst []uint16, a []float32)
			}{
				{"F32ToBF16", got.Convert.F32ToBF16, want.Convert.F32ToBF16},
				{"F32ToF16", got.Convert.F32ToF16, want.Convert.F32ToF16},
			} {
				if c.got == nil || c.want == nil {
					continue
				}
				m := len(f32in)
				g, w := make([]uint16, m), make([]uint16, m)
				c.got(g, f32in)
				c.want(w, f32in)
				for i := range g {
					if g[i] != w[i] {
						t.Fatalf("%s/Convert.%s(%v): got %#04x want %#04x",
							tier, c.op, f32in[i], g[i], w[i])
					}
				}
			}
		})
	}
}

// sameF32Bits compares bit patterns, so a lost sign on a zero is a failure.
// NaN is the one exception, for the reason given on `same`: which NaN survives
// is not specified and hardware differs.
func sameF32Bits(a, b float32) bool {
	af, bf := float64(a), float64(b)
	if math.IsNaN(af) || math.IsNaN(bf) {
		return math.IsNaN(af) && math.IsNaN(bf)
	}
	return math.Float32bits(a) == math.Float32bits(b)
}
