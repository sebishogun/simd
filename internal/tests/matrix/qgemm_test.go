package matrix

import (
	"math"
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

func qmatmulRef(dst []int32, a, b []int8, m, k, n int) {
	for i := range dst[:m*n] {
		dst[i] = 0
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s int32
			for p := 0; p < k; p++ {
				s += int32(a[i*k+p]) * int32(b[p*n+j])
			}
			dst[i*n+j] = s
		}
	}
}

func TestQMatMulInt8(t *testing.T) {
	r := rand.New(rand.NewPCG(101, 103))
	// Sizes either side of the tile in both dimensions, so the edge paths run.
	for _, d := range []struct{ m, k, n int }{
		{1, 1, 1}, {1, 8, 1}, {3, 5, 7}, {8, 8, 8}, {8, 16, 16},
		{9, 17, 33}, {16, 4, 16}, {6, 6, 6}, {13, 29, 31}, {32, 32, 32},
	} {
		a := make([]int8, d.m*d.k)
		b := make([]int8, d.k*d.n)
		for i := range a {
			a[i] = int8(r.Uint32())
		}
		for i := range b {
			b[i] = int8(r.Uint32())
		}
		got := make([]int32, d.m*d.n)
		want := make([]int32, d.m*d.n)
		simd.QMatMulInt8Into(got, a, b, d.m, d.k, d.n)
		qmatmulRef(want, a, b, d.m, d.k, d.n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("m=%d k=%d n=%d at %d: got %d want %d",
					d.m, d.k, d.n, i, got[i], want[i])
			}
		}
	}
}

// The reason this is a separate kernel rather than MatMulInto[int8]: the
// accumulator has to be wider than the inputs. Every element -128 or 127 makes
// the products maximal, so this is the case an int8 accumulator gets wrong
// after two terms.
func TestQMatMulInt8DoesNotOverflow(t *testing.T) {
	const k = 512
	a := make([]int8, k)
	b := make([]int8, k)
	for i := range a {
		a[i], b[i] = 127, 127
	}
	got := make([]int32, 1)
	simd.QMatMulInt8Into(got, a, b, 1, k, 1)
	want := int32(k) * 127 * 127
	if got[0] != want {
		t.Fatalf("got %d, want %d — the accumulator is too narrow", got[0], want)
	}

	// And the most negative product, which is where a naive negation breaks.
	for i := range a {
		a[i], b[i] = -128, 127
	}
	simd.QMatMulInt8Into(got, a, b, 1, k, 1)
	if w := int32(k) * -128 * 127; got[0] != w {
		t.Fatalf("got %d, want %d", got[0], w)
	}
}

func TestQMatMulInt8RejectsBadSizes(t *testing.T) {
	dst := make([]int32, 4)
	for i := range dst {
		dst[i] = -1
	}
	a := make([]int8, 2)
	b := make([]int8, 2)
	// Too short for the stated dimensions: it must do nothing rather than
	// read out of bounds.
	simd.QMatMulInt8Into(dst, a, b, 4, 4, 4)
	for i := range dst {
		if dst[i] != -1 {
			t.Fatalf("a badly sized call wrote to dst[%d]", i)
		}
	}
}

func TestRequantizeInt8(t *testing.T) {
	// Halfway cases, which is where round-half-to-even differs from the naive
	// +0.5, and the saturation boundaries.
	in := []int32{0, 1, 2, 3, 4, 5, -1, -2, -3, 1000, -1000}
	dst := make([]int8, len(in))
	simd.RequantizeInt8Into(dst, in, 0.5, 0)
	want := []int8{0, 0, 1, 2, 2, 2, 0, -1, -2, 127, -128}
	for i := range want {
		if dst[i] != want[i] {
			t.Errorf("Requantize(%d, 0.5) = %d, want %d", in[i], dst[i], want[i])
		}
	}
}

func TestRequantizeInt8Random(t *testing.T) {
	r := rand.New(rand.NewPCG(107, 109))
	for _, n := range []int{0, 1, 15, 16, 17, 100, 1000} {
		a := make([]int32, n)
		for i := range a {
			a[i] = int32(r.Uint32()) / 1024
		}
		dst := make([]int8, n)
		simd.RequantizeInt8Into(dst, a, 0.125, 3)
		for i := range a {
			q := math.RoundToEven(float64(float32(a[i])*0.125)) + 3
			if q < -128 {
				q = -128
			}
			if q > 127 {
				q = 127
			}
			if dst[i] != int8(q) {
				t.Fatalf("n=%d i=%d: a=%d got %d want %d", n, i, a[i], dst[i], int8(q))
			}
		}
	}
}

// The end-to-end shape a caller actually writes: quantize both operands,
// multiply in int8, requantize the result.
//
// The inputs are bounded and the scale is chosen to cover them exactly. That
// is not tidiness — the first version of this test drew from NormFloat64 and
// used scale 0.02, which represents only ±2.54, so a third of the values
// saturated and the test was measuring clipping rather than rounding. A scale
// that does not cover the data is a modelling error and no kernel can rescue
// it; picking it correctly is the caller's job and the reason QuantizeInt8
// takes the scale rather than deriving one.
func TestQuantizedLayerRoundTrip(t *testing.T) {
	const m, k, n = 4, 32, 8
	// 1.0 is the largest magnitude below, and 1/127 is the scale that puts it
	// exactly at full scale with nothing clipped.
	const scale = 1.0 / 127.0
	r := rand.New(rand.NewPCG(113, 127))
	af := make([]float32, m*k)
	bf := make([]float32, k*n)
	for i := range af {
		af[i] = float32(r.Float64()*2 - 1)
	}
	for i := range bf {
		bf[i] = float32(r.Float64()*2 - 1)
	}

	aq := make([]int8, len(af))
	bq := make([]int8, len(bf))
	simd.QuantizeInt8(aq, af, scale, 0)
	simd.QuantizeInt8(bq, bf, scale, 0)

	acc := make([]int32, m*n)
	simd.QMatMulInt8Into(acc, aq, bq, m, k, n)

	// Each operand carries at most scale/2 of error, so a k-term dot product
	// carries about k*scale on the worst case and much less in practice. The
	// bound below is that worst case, so this fails on a real defect rather
	// than on an unlucky draw.
	const tol = float32(k) * scale
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var want float32
			for p := 0; p < k; p++ {
				want += af[i*k+p] * bf[p*n+j]
			}
			got := float32(acc[i*n+j]) * scale * scale
			if d := got - want; d > tol || d < -tol {
				t.Errorf("[%d,%d]: quantized %v, float %v, off by %v (tolerance %v)",
					i, j, got, want, d, tol)
			}
		}
	}
}

func TestQMatMulNoAlloc(t *testing.T) {
	const m, k, n = 8, 32, 16
	a := make([]int8, m*k)
	b := make([]int8, k*n)
	dst := make([]int32, m*n)
	if x := testing.AllocsPerRun(20, func() { simd.QMatMulInt8Into(dst, a, b, m, k, n) }); x != 0 {
		t.Errorf("QMatMulInt8Into allocated %v times per run, want 0", x)
	}
}
