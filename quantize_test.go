package simd_test

// Quantization is specified, not invented: ONNX, PyTorch and TFLite all define
// q = clamp(round(x/scale) + zeroPoint, lo, hi) with round-half-to-EVEN. So
// the oracle here is that formula written out in float64, and the cases that
// matter are the exact .5 values where half-even and half-away-from-zero
// disagree — which a symmetric scale produces in quantity.

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

func wantQ(x, scale float32, zp int32, lo, hi float64) float64 {
	q := math.RoundToEven(float64(x/scale)) + float64(zp)
	return math.Min(math.Max(q, lo), hi)
}

func TestQuantizeInt8(t *testing.T) {
	// Every exact .5 in a scale of 1 — the values half-even and half-away
	// disagree on. A naive int8(x+0.5) fails every one of the negatives here.
	t.Run("halfway", func(t *testing.T) {
		var a []float32
		for v := -8.5; v <= 8.5; v += 0.5 {
			a = append(a, float32(v))
		}
		got := make([]int8, len(a))
		simd.QuantizeInt8(got, a, 1, 0)
		for i, x := range a {
			want := int8(wantQ(x, 1, 0, -128, 127))
			if got[i] != want {
				t.Errorf("x=%g: got %d, want %d (round half to EVEN)", x, got[i], want)
			}
		}
	})

	// Saturation in both directions must clamp, never wrap. A wrapped int8 in
	// an inference pipeline is a sign flip, not a small error.
	t.Run("saturation", func(t *testing.T) {
		a := []float32{-1e9, -200, -129, -128.5, 127.5, 128, 1e9,
			float32(math.Inf(-1)), float32(math.Inf(1))}
		got := make([]int8, len(a))
		simd.QuantizeInt8(got, a, 1, 0)
		for i, x := range a {
			if x < 0 && got[i] != -128 {
				t.Errorf("x=%g: got %d, want -128", x, got[i])
			}
			if x > 0 && got[i] != 127 {
				t.Errorf("x=%g: got %d, want 127", x, got[i])
			}
		}
	})

	// Random values against the oracle, across scales and zero points, at
	// lengths that straddle every vector width.
	r := rand.New(rand.NewPCG(97, 101))
	for _, n := range []int{0, 1, 7, 8, 15, 16, 17, 31, 33, 1000} {
		for _, sc := range []float32{1, 0.5, 0.007874016, 3.2} {
			for _, zp := range []int32{0, -128, 42, 127} {
				a := make([]float32, n)
				for i := range a {
					a[i] = float32(r.NormFloat64() * 64)
				}
				got := make([]int8, n)
				simd.QuantizeInt8(got, a, sc, zp)
				for i, x := range a {
					if want := int8(wantQ(x, sc, zp, -128, 127)); got[i] != want {
						t.Fatalf("n=%d scale=%g zp=%d i=%d x=%g: got %d want %d",
							n, sc, zp, i, x, got[i], want)
					}
				}
			}
		}
	}
}

func TestQuantizeUint8(t *testing.T) {
	r := rand.New(rand.NewPCG(103, 107))
	for _, n := range []int{0, 1, 16, 17, 1000} {
		for _, sc := range []float32{1, 0.0039215686, 2.5} {
			for _, zp := range []int32{0, 128, 255} {
				a := make([]float32, n)
				for i := range a {
					a[i] = float32(r.NormFloat64() * 64)
				}
				got := make([]uint8, n)
				simd.QuantizeUint8(got, a, sc, zp)
				for i, x := range a {
					if want := uint8(wantQ(x, sc, zp, 0, 255)); got[i] != want {
						t.Fatalf("n=%d scale=%g zp=%d i=%d x=%g: got %d want %d",
							n, sc, zp, i, x, got[i], want)
					}
				}
			}
		}
	}
}

func TestDequantizeRoundTrip(t *testing.T) {
	// Dequantize is exact: (q - zp) * scale in float32. Check it directly, and
	// then that a value already on the grid survives a round trip unchanged,
	// which is the property a calibrated model depends on.
	for _, zp := range []int32{0, -128, 64} {
		const scale float32 = 0.25
		q := make([]int8, 256)
		for i := range q {
			q[i] = int8(i - 128)
		}
		x := make([]float32, len(q))
		simd.DequantizeInt8(x, q, scale, zp)
		for i, v := range q {
			if want := float32(int32(v)-zp) * scale; x[i] != want {
				t.Fatalf("zp=%d q=%d: got %v want %v", zp, v, x[i], want)
			}
		}
		back := make([]int8, len(q))
		simd.QuantizeInt8(back, x, scale, zp)
		for i := range q {
			if back[i] != q[i] {
				t.Fatalf("zp=%d round trip at %d: %d -> %v -> %d", zp, i, q[i], x[i], back[i])
			}
		}
	}
}

func ExampleQuantizeInt8() {
	// A symmetric per-tensor scale, the common case for weights.
	w := []float32{-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0}
	scale := float32(1.0 / 127)
	q := make([]int8, len(w))
	simd.QuantizeInt8(q, w, scale, 0)
	fmt.Println(q)
	// Output: [-127 -64 -32 0 32 64 127]
}
