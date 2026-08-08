package simd_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

// The columnar family's contract, differentially: kernel against the
// scalar truth at every size that crosses a block or byte edge, with NaN
// under null bits -- Arrow leaves those slots undefined, and the sum must
// never read them.
func TestColumnarDifferential(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	sizes := []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 200, 1000, 4096}
	for _, n := range sizes {
		a := make([]float64, n)
		bm := make([]byte, (n+7)/8)
		rng.Read(bm)
		for i := range a {
			if bm[i>>3]>>(i&7)&1 == 0 {
				a[i] = math.NaN() // never to be read
			} else {
				a[i] = rng.NormFloat64()
			}
		}
		var wantSum float64
		var kept []float64
		for i := range a {
			if bm[i>>3]>>(i&7)&1 != 0 {
				kept = append(kept, a[i])
			}
		}
		wantSum = simd.Sum(maskedZero(a, bm))
		got := simd.SumValid(a, bm)
		if math.IsNaN(got) || got != wantSum {
			t.Fatalf("n=%d SumValid=%v want %v", n, got, wantSum)
		}
		dst := make([]float64, n)
		k := simd.CompressBitsInto(dst, a, bm)
		if k != len(kept) {
			t.Fatalf("n=%d CompressBits count %d want %d", n, k, len(kept))
		}
		for i := range kept {
			if dst[i] != kept[i] {
				t.Fatalf("n=%d dst[%d]=%v want %v", n, i, dst[i], kept[i])
			}
		}
		if c := simd.CountValid(bm, n); c != len(kept) {
			t.Fatalf("n=%d CountValid=%d want %d", n, c, len(kept))
		}
	}
	// The int64 side, and the all-set bit-identity promise.
	xs := make([]float64, 999)
	ones := make([]byte, 125)
	for i := range xs {
		xs[i] = rng.NormFloat64()
	}
	for i := range ones {
		ones[i] = 0xFF
	}
	if simd.SumValid(xs, ones) != simd.Sum(xs) {
		t.Fatal("SumValid over all-ones is not bit-identical to Sum")
	}
}

func maskedZero(a []float64, bm []byte) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		if bm[i>>3]>>(i&7)&1 != 0 {
			out[i] = a[i]
		}
	}
	return out
}
