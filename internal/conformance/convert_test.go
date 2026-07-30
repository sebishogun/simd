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
	"math/rand/v2"
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

// TestVarintWidthKernels drives the LEB128 width kernels against ref.
//
// The inputs are every width boundary rather than random values: the kernel
// computes the width as a sum of unsigned comparisons against 2^7, 2^14 and so
// on, and an off-by-one in any threshold is wrong for exactly one value on
// each side of it and right everywhere else. Random inputs would almost never
// land there.
func TestVarintWidthKernels(t *testing.T) {
	want := ref.Set()

	var in64 []uint64
	for shift := 0; shift < 64; shift += 7 {
		b := uint64(1) << shift
		in64 = append(in64, b-1, b, b+1)
	}
	in64 = append(in64, 0, 1, ^uint64(0), ^uint64(0)-1, 1<<63, 1<<63-1)
	in32 := make([]uint32, len(in64))
	for i, v := range in64 {
		in32[i] = uint32(v)
	}

	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			if f := got.Convert.VarintLenU32; f != nil {
				g, w := make([]int32, len(in32)), make([]int32, len(in32))
				f(g, in32)
				want.Convert.VarintLenU32(w, in32)
				for i := range g {
					if g[i] != w[i] {
						t.Fatalf("%s/Convert.VarintLenU32(%#x): got %d want %d",
							tier, in32[i], g[i], w[i])
					}
				}
			}
			if f := got.Convert.VarintLenU64; f != nil {
				g, w := make([]int32, len(in64)), make([]int32, len(in64))
				f(g, in64)
				want.Convert.VarintLenU64(w, in64)
				for i := range g {
					if g[i] != w[i] {
						t.Fatalf("%s/Convert.VarintLenU64(%#x): got %d want %d",
							tier, in64[i], g[i], w[i])
					}
				}
			}
			// The totals are checked at lengths around the kernel's eight-lane
			// fold, since a fold that dropped its tail would agree on a length
			// that happens to be a multiple of eight and nowhere else.
			for n := range len(in64) {
				if f := got.Convert.VarintSizeU32; f != nil {
					if g, w := f(in32[:n]), want.Convert.VarintSizeU32(in32[:n]); g != w {
						t.Fatalf("%s/Convert.VarintSizeU32 n=%d: got %d want %d",
							tier, n, g, w)
					}
				}
				if f := got.Convert.VarintSizeU64; f != nil {
					if g, w := f(in64[:n]), want.Convert.VarintSizeU64(in64[:n]); g != w {
						t.Fatalf("%s/Convert.VarintSizeU64 n=%d: got %d want %d",
							tier, n, g, w)
					}
				}
			}
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

// TestQMatMulKernels drives the quantized matrix multiply against ref across
// every tier, at sizes either side of the register tile in both dimensions so
// the edge paths run rather than only the tile.
//
// Exact, not bounded: integer addition is associative, so unlike the float
// matmul there is no accumulation order to preserve and the two must agree bit
// for bit.
func TestQMatMulKernels(t *testing.T) {
	want := ref.Set()
	r := rand.New(rand.NewPCG(211, 223))

	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			if got.Convert.QMatMulI8 == nil {
				return
			}
			for _, d := range []struct{ m, k, n int }{
				{1, 1, 1}, {3, 5, 7}, {8, 8, 8}, {8, 16, 16},
				{9, 17, 33}, {6, 6, 6}, {13, 29, 31},
			} {
				a := make([]int8, d.m*d.k)
				b := make([]int8, d.k*d.n)
				for i := range a {
					a[i] = int8(r.Uint32())
				}
				for i := range b {
					b[i] = int8(r.Uint32())
				}
				g := make([]int32, d.m*d.n)
				w := make([]int32, d.m*d.n)
				got.Convert.QMatMulI8(g, a, b, d.m, d.k, d.n)
				want.Convert.QMatMulI8(w, a, b, d.m, d.k, d.n)
				if i, ok := same(g, w); !ok {
					t.Fatalf("%s/Convert.QMatMulI8 m=%d k=%d n=%d at %d: got %d want %d",
						tier, d.m, d.k, d.n, i, g[i], w[i])
				}
			}

			if got.Convert.RequantizeI8 == nil {
				return
			}
			for _, n := range []int{0, 1, 15, 16, 17, 100} {
				acc := make([]int32, n)
				for i := range acc {
					acc[i] = int32(r.Uint32()) / 512
				}
				g := make([]int8, n)
				w := make([]int8, n)
				got.Convert.RequantizeI8(g, acc, 0.0625, -7)
				want.Convert.RequantizeI8(w, acc, 0.0625, -7)
				if i, ok := same(g, w); !ok {
					t.Fatalf("%s/Convert.RequantizeI8 n=%d at %d: got %d want %d",
						tier, n, i, g[i], w[i])
				}
			}
		})
	}
}

// TestPerChannelQuantKernels drives the per-channel quantizers against ref at
// shapes either side of a vector width in the inner dimension, which is the
// one the vectorizer sees.
func TestPerChannelQuantKernels(t *testing.T) {
	want := ref.Set()
	r := rand.New(rand.NewPCG(401, 409))

	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			if got.Convert.QuantizePerChannelI8 == nil {
				return
			}
			for _, d := range []struct{ channels, inner int }{
				{1, 1}, {1, 15}, {1, 16}, {1, 17}, {3, 15}, {4, 16}, {5, 33}, {8, 64},
			} {
				n := d.channels * d.inner
				a := make([]float32, n)
				scale := make([]float32, d.channels)
				zp := make([]int32, d.channels)
				for i := range a {
					a[i] = float32(r.NormFloat64()) * 30
				}
				for c := range scale {
					scale[c] = float32(r.Float64())*0.5 + 0.05
					zp[c] = int32(r.IntN(41) - 20)
				}

				g8, w8 := make([]int8, n), make([]int8, n)
				got.Convert.QuantizePerChannelI8(g8, a, scale, zp, d.channels, d.inner)
				want.Convert.QuantizePerChannelI8(w8, a, scale, zp, d.channels, d.inner)
				if i, ok := same(g8, w8); !ok {
					t.Fatalf("%s/QuantizePerChannelI8 c=%d inner=%d at %d: got %d want %d",
						tier, d.channels, d.inner, i, g8[i], w8[i])
				}

				gf, wf := make([]float32, n), make([]float32, n)
				got.Convert.DequantizePerChannelI8(gf, w8, scale, zp, d.channels, d.inner)
				want.Convert.DequantizePerChannelI8(wf, w8, scale, zp, d.channels, d.inner)
				for i := range gf {
					if !sameF32Bits(gf[i], wf[i]) {
						t.Fatalf("%s/DequantizePerChannelI8 at %d: got %v want %v",
							tier, i, gf[i], wf[i])
					}
				}

				gu, wu := make([]uint8, n), make([]uint8, n)
				got.Convert.QuantizePerChannelU8(gu, a, scale, zp, d.channels, d.inner)
				want.Convert.QuantizePerChannelU8(wu, a, scale, zp, d.channels, d.inner)
				if i, ok := same(gu, wu); !ok {
					t.Fatalf("%s/QuantizePerChannelU8 at %d: got %d want %d", tier, i, gu[i], wu[i])
				}
			}
		})
	}
}

// TestFP8Kernels drives both fp8 formats against ref over their entire domain
// in one direction and over the float32 edge cases in the other.
//
// Exhaustive is possible here and worth taking: 256 encodings is the whole
// input space of the widening direction, so this is a proof rather than a
// sample.
func TestFP8Kernels(t *testing.T) {
	want := ref.Set()
	all := make([]byte, 256)
	for i := range all {
		all[i] = byte(i)
	}
	// For narrowing: every value the widening produces, plus the boundaries
	// that decide saturation, denormal and NaN.
	wideRef := make([]float32, 256)
	want.Convert.F8E4M3ToF32(wideRef, all)
	edges := []float32{
		0, float32(math.Copysign(0, -1)), 448, -448, 449, 500, 57344, 57345,
		1.0 / 512, 1.0 / 1024, 1.0 / 2048, 0.015625, 0.0078125,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	}
	narrowIn := append(append([]float32(nil), wideRef...), edges...)

	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			for _, c := range []struct {
				op        string
				got, want func(dst []float32, a []byte)
			}{
				{"F8E4M3ToF32", got.Convert.F8E4M3ToF32, want.Convert.F8E4M3ToF32},
				{"F8E5M2ToF32", got.Convert.F8E5M2ToF32, want.Convert.F8E5M2ToF32},
			} {
				if c.got == nil {
					continue
				}
				g, w := make([]float32, 256), make([]float32, 256)
				c.got(g, all)
				c.want(w, all)
				for i := range g {
					if !sameF32Bits(g[i], w[i]) {
						t.Fatalf("%s/Convert.%s(%#02x): got %v want %v",
							tier, c.op, i, g[i], w[i])
					}
				}
			}
			for _, c := range []struct {
				op        string
				got, want func(dst []byte, a []float32)
			}{
				{"F32ToF8E4M3", got.Convert.F32ToF8E4M3, want.Convert.F32ToF8E4M3},
				{"F32ToF8E5M2", got.Convert.F32ToF8E5M2, want.Convert.F32ToF8E5M2},
			} {
				if c.got == nil {
					continue
				}
				m := len(narrowIn)
				g, w := make([]byte, m), make([]byte, m)
				c.got(g, narrowIn)
				c.want(w, narrowIn)
				for i := range g {
					if g[i] != w[i] {
						t.Fatalf("%s/Convert.%s(%v): got %#02x want %#02x",
							tier, c.op, narrowIn[i], g[i], w[i])
					}
				}
			}
		})
	}
}
