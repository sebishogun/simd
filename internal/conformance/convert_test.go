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

// TestZigzagKernels drives every zigzag kernel against ref over the whole
// domain where the domain is small enough to enumerate, and over the boundary
// values where it is not.
//
// Zigzag is exact — shifts and exclusive ors — so this is bit identity and not
// a bound, and the interesting inputs are the extremes: the most negative
// value has no positive counterpart, so any formulation that negates it is
// wrong there and nowhere else.
func TestZigzagKernels(t *testing.T) {
	want := ref.Set()

	i8in := make([]int8, 256)
	for i := range i8in {
		i8in[i] = int8(i + math.MinInt8)
	}
	i16in := make([]int16, 65536)
	for i := range i16in {
		i16in[i] = int16(i + math.MinInt16)
	}
	i32in := []int32{0, -1, 1, -2, 2, math.MinInt32, math.MaxInt32,
		math.MinInt32 + 1, math.MaxInt32 - 1, -65536, 65536}
	i64in := []int64{0, -1, 1, -2, 2, math.MinInt64, math.MaxInt64,
		math.MinInt64 + 1, math.MaxInt64 - 1, -1 << 40, 1 << 40}

	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			zigzag(t, tier, "I8", i8in, got.Convert.ZigzagEncodeI8, want.Convert.ZigzagEncodeI8,
				got.Convert.ZigzagDecodeI8, want.Convert.ZigzagDecodeI8)
			zigzag(t, tier, "I16", i16in, got.Convert.ZigzagEncodeI16, want.Convert.ZigzagEncodeI16,
				got.Convert.ZigzagDecodeI16, want.Convert.ZigzagDecodeI16)
			zigzag(t, tier, "I32", i32in, got.Convert.ZigzagEncodeI32, want.Convert.ZigzagEncodeI32,
				got.Convert.ZigzagDecodeI32, want.Convert.ZigzagDecodeI32)
			zigzag(t, tier, "I64", i64in, got.Convert.ZigzagEncodeI64, want.Convert.ZigzagEncodeI64,
				got.Convert.ZigzagDecodeI64, want.Convert.ZigzagDecodeI64)
		})
	}
}

// zigzag checks one width: the kernel agrees with ref on the encoding, and the
// kernel's own decode inverts the kernel's own encode. The round trip is
// checked against the input rather than against ref's decode, which makes it a
// statement about the pair rather than about either half.
func zigzag[S, U comparable](t *testing.T, tier, name string, in []S,
	encGot, encWant func(dst []U, a []S), decGot, decWant func(dst []S, a []U)) {
	t.Helper()
	if encGot == nil || decGot == nil {
		return
	}
	g, w := make([]U, len(in)), make([]U, len(in))
	encGot(g, in)
	encWant(w, in)
	for i := range g {
		if g[i] != w[i] {
			t.Fatalf("%s/Convert.ZigzagEncode%s(%v): got %v want %v", tier, name, in[i], g[i], w[i])
		}
	}

	back, backWant := make([]S, len(in)), make([]S, len(in))
	decGot(back, g)
	decWant(backWant, g)
	for i := range back {
		if back[i] != in[i] {
			t.Fatalf("%s/Convert.Zigzag%s round trip: %v -> %v -> %v", tier, name, in[i], g[i], back[i])
		}
		if back[i] != backWant[i] {
			t.Fatalf("%s/Convert.ZigzagDecode%s(%v): got %v want %v", tier, name, g[i], back[i], backWant[i])
		}
	}
}
