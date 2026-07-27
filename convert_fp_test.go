package simd_test

// float16 and bfloat16, checked against an independent definition rather than
// against internal/ref.
//
// The reference and the kernel are two different pieces of arithmetic that are
// supposed to agree; a test that compared them to each other would pass if
// both shared a mistake. So the expected values here come from a third route:
// math/big for the exact real value, and exhaustive enumeration where the
// domain allows it — every one of the 65536 float16 values, and every one of
// the 65536 bfloat16 values, which is small enough to check completely.

import (
	"math"
	"testing"

	"github.com/sebishogun/simd"
)

// TestBFloat16RoundTripsExhaustively checks all 65536 bit patterns. A bfloat16
// widened to float32 and narrowed back must be itself, for every value that is
// not a NaN — and every NaN must stay a NaN.
func TestBFloat16RoundTripsExhaustively(t *testing.T) {
	const n = 1 << 16
	src := make([]uint16, n)
	for i := range src {
		src[i] = uint16(i)
	}
	wide := make([]float32, n)
	back := make([]uint16, n)
	simd.BFloat16ToFloat32Into(wide, src)
	simd.Float32ToBFloat16Into(back, wide)

	for i := range src {
		// Widening is a pure shift, so this is checkable directly.
		if got, want := math.Float32bits(wide[i]), uint32(src[i])<<16; got != want {
			t.Fatalf("BFloat16ToFloat32(%#04x) = %#08x, want %#08x", src[i], got, want)
		}
		if math.IsNaN(float64(wide[i])) {
			if back[i]&0x7f80 != 0x7f80 || back[i]&0x007f == 0 {
				t.Fatalf("a NaN did not survive the round trip: %#04x -> %#04x", src[i], back[i])
			}
			continue
		}
		if back[i] != src[i] {
			t.Fatalf("round trip %#04x -> %v -> %#04x", src[i], wide[i], back[i])
		}
	}
}

// TestFloat16RoundTripsExhaustively is the same for float16.
func TestFloat16RoundTripsExhaustively(t *testing.T) {
	const n = 1 << 16
	src := make([]uint16, n)
	for i := range src {
		src[i] = uint16(i)
	}
	wide := make([]float32, n)
	back := make([]uint16, n)
	simd.Float16ToFloat32Into(wide, src)
	simd.Float32ToFloat16Into(back, wide)

	for i := range src {
		h := src[i]
		want := halfToFloat(h)
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(wide[i])) {
				t.Fatalf("Float16ToFloat32(%#04x) = %v, want NaN", h, wide[i])
			}
			if back[i]&0x7c00 != 0x7c00 || back[i]&0x03ff == 0 {
				t.Fatalf("a NaN did not survive: %#04x -> %#04x", h, back[i])
			}
			continue
		}
		if math.Float32bits(wide[i]) != math.Float32bits(want) {
			t.Fatalf("Float16ToFloat32(%#04x) = %v (%#08x), want %v (%#08x)",
				h, wide[i], math.Float32bits(wide[i]), want, math.Float32bits(want))
		}
		if back[i] != h {
			t.Fatalf("round trip %#04x -> %v -> %#04x", h, wide[i], back[i])
		}
	}
}

// halfToFloat is the definition, written the slow obvious way from the format
// description so that it shares nothing with either implementation.
func halfToFloat(h uint16) float32 {
	sign := float64(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	exp := int(h>>10) & 0x1f
	man := int(h & 0x3ff)
	switch {
	case exp == 0x1f && man == 0:
		return float32(sign * math.Inf(1))
	case exp == 0x1f:
		return float32(math.NaN())
	case exp == 0:
		return float32(sign * float64(man) * math.Pow(2, -24))
	default:
		return float32(sign * (1 + float64(man)/1024) * math.Pow(2, float64(exp-15)))
	}
}

// TestFloat32ToFloat16Rounding covers the narrowing direction at the points
// where it is decided: the largest representable value, the first that
// overflows, the denormal boundary, and exact halfway cases that must round to
// even.
func TestFloat32ToFloat16Rounding(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{0, 0x0000},
		{float32(math.Copysign(0, -1)), 0x8000},
		{1, 0x3c00},
		{-1, 0xbc00},
		{65504, 0x7bff},  // the largest float16
		{65520, 0x7c00},  // the first that rounds to infinity
		{65519, 0x7bff},  // and the last that does not
		{131008, 0x7c00}, // well past it
		{float32(math.Inf(1)), 0x7c00},
		{float32(math.Inf(-1)), 0xfc00},
		{6.1035156e-05, 0x0400}, // the smallest normal
		{5.9604645e-08, 0x0001}, // the smallest denormal
		{2.9802322e-08, 0x0000}, // half of it: rounds to even, which is zero
		{8.940697e-08, 0x0002},  // 1.5 denormals: rounds to even, which is two
	}
	dst := make([]uint16, len(cases))
	in := make([]float32, len(cases))
	for i, c := range cases {
		in[i] = c.in
	}
	simd.Float32ToFloat16Into(dst, in)
	for i, c := range cases {
		if dst[i] != c.want {
			t.Errorf("Float32ToFloat16(%v) = %#04x, want %#04x", c.in, dst[i], c.want)
		}
	}
}

func TestConversionsDoNotAllocate(t *testing.T) {
	const n = 1024
	h := make([]uint16, n)
	f := make([]float32, n)
	for _, c := range []struct {
		name string
		fn   func()
	}{
		{"BFloat16ToFloat32Into", func() { simd.BFloat16ToFloat32Into(f, h) }},
		{"Float32ToBFloat16Into", func() { simd.Float32ToBFloat16Into(h, f) }},
		{"Float16ToFloat32Into", func() { simd.Float16ToFloat32Into(f, h) }},
		{"Float32ToFloat16Into", func() { simd.Float32ToFloat16Into(h, f) }},
	} {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %.0f times, want 0", c.name, a)
		}
	}
}
