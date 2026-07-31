package matrix

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// The contract is bit-identity with MatMulInto, so that is the oracle — not a
// tolerance, equality. Shapes deliberately miss the tile width (16/8) and the
// MR row block on both sides, and straddle the packing cliff.
func TestMatMulPacked(t *testing.T) {
	r := rand.New(rand.NewPCG(211, 223))
	shapes := [][3]int{
		{1, 1, 1}, {3, 5, 7}, {8, 8, 8}, {6, 17, 9}, {13, 13, 16},
		{33, 7, 65}, {64, 64, 64}, {100, 3, 50}, {17, 400, 23},
		{130, 130, 130}, {512, 512, 512},
	}
	for _, sh := range shapes {
		m, k, n := sh[0], sh[1], sh[2]
		t.Run(fmt.Sprintf("f64/%dx%dx%d", m, k, n), func(t *testing.T) {
			a := make([]float64, m*k)
			b := make([]float64, k*n)
			for i := range a {
				a[i] = r.NormFloat64()
			}
			for i := range b {
				b[i] = r.NormFloat64()
			}
			want := make([]float64, m*n)
			simd.MatMulInto(want, a, b, m, k, n)

			bp := make([]float64, simd.GemmPackLen[float64](k, n))
			simd.PackBInto(bp, b, k, n)
			got := make([]float64, m*n)
			simd.MatMulIntoPacked(got, a, bp, m, k, n)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("packed[%d] = %v, want %v (bit-identity)", i, got[i], want[i])
				}
			}

			// The scratch wrapper must agree on both of its paths.
			viaScratch := make([]float64, m*n)
			simd.MatMulIntoScratch(viaScratch, a, b, bp, m, k, n)
			for i := range want {
				if viaScratch[i] != want[i] {
					t.Fatalf("scratch[%d] = %v, want %v", i, viaScratch[i], want[i])
				}
			}
			// And with scratch too short it falls back, same bits.
			simd.MatMulIntoScratch(viaScratch, a, b, nil, m, k, n)
			for i := range want {
				if viaScratch[i] != want[i] {
					t.Fatalf("fallback[%d] differs", i)
				}
			}
		})
	}

	// float32, whose tile width is 16 rather than 8.
	t.Run("f32", func(t *testing.T) {
		m, k, n := 37, 19, 41
		a := make([]float32, m*k)
		b := make([]float32, k*n)
		for i := range a {
			a[i] = float32(r.NormFloat64())
		}
		for i := range b {
			b[i] = float32(r.NormFloat64())
		}
		want := make([]float32, m*n)
		simd.MatMulInto(want, a, b, m, k, n)
		bp := make([]float32, simd.GemmPackLen[float32](k, n))
		simd.PackBInto(bp, b, k, n)
		got := make([]float32, m*n)
		simd.MatMulIntoPacked(got, a, bp, m, k, n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("f32 packed[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})

	// Degenerate shapes do nothing rather than panicking or reading short.
	t.Run("degenerate", func(t *testing.T) {
		if simd.GemmPackLen[float64](0, 5) != 0 || simd.GemmPackLen[float64](5, 0) != 0 {
			t.Error("GemmPackLen of empty shape should be 0")
		}
		d := []float64{7}
		simd.MatMulIntoPacked(d, []float64{1}, []float64{}, 1, 1, 1)
		if d[0] != 7 {
			t.Error("short bp should leave dst untouched")
		}
	})
}
