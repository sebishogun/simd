package simd_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// Both functions have two paths — a plain loop below 512 elements and a
// four-table one above — so every case is checked on both sides of that
// threshold. It is the same trap the Median tests fell into: a suite that runs
// only short inputs tests the fallback and reports success.
func TestBincount(t *testing.T) {
	r := rand.New(rand.NewPCG(29, 31))
	for _, n := range []int{0, 1, 7, 511, 512, 513, 4096} {
		for _, nb := range []int{1, 3, 16, 256} {
			a := make([]int32, n)
			for i := range a {
				// Deliberately includes out-of-range values on both sides,
				// which must be skipped rather than wrap or panic.
				a[i] = int32(r.IntN(nb+4)) - 2
			}
			want := make([]int32, nb)
			for _, v := range a {
				if v >= 0 && int(v) < nb {
					want[v]++
				}
			}
			t.Run(fmt.Sprintf("n=%d/bins=%d", n, nb), func(t *testing.T) {
				got := simd.Bincount(a, nb)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("bin %d = %d, want %d", i, got[i], want[i])
					}
				}
				// The Into form must agree, and must accumulate rather than
				// overwrite.
				acc := make([]int32, nb)
				simd.BincountInto(acc, a)
				simd.BincountInto(acc, a)
				for i := range want {
					if acc[i] != 2*want[i] {
						t.Fatalf("accumulated bin %d = %d, want %d", i, acc[i], 2*want[i])
					}
				}
				// A short scratch costs speed, not correctness.
				short := make([]int32, nb)
				simd.BincountInto(short, a)
				for i := range want {
					if short[i] != want[i] {
						t.Fatalf("nil scratch bin %d = %d, want %d", i, short[i], want[i])
					}
				}
			})
		}
	}
}

func TestHistogram(t *testing.T) {
	r := rand.New(rand.NewPCG(37, 41))
	for _, n := range []int{0, 1, 7, 511, 512, 513, 4096} {
		for _, nb := range []int{1, 4, 64} {
			a := make([]float64, n)
			for i := range a {
				switch i % 11 {
				case 0:
					a[i] = -5 // below lo
				case 1:
					a[i] = 15 // above hi
				case 2:
					a[i] = 0 // exactly lo, counted
				case 3:
					a[i] = 10 // exactly hi, excluded
				case 4:
					a[i] = math.NaN() // excluded, compares false both ways
				default:
					a[i] = r.Float64() * 10
				}
			}
			want := make([]int32, nb)
			for _, v := range a {
				if !(v >= 0) || !(v < 10) {
					continue
				}
				k := int(v / 10 * float64(nb))
				if k >= nb {
					k = nb - 1
				}
				want[k]++
			}
			t.Run(fmt.Sprintf("n=%d/bins=%d", n, nb), func(t *testing.T) {
				got := simd.Histogram(a, nb, 0.0, 10.0)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("bin %d = %d, want %d", i, got[i], want[i])
					}
				}
				// Total counted must equal the in-range count exactly — this
				// catches an off-by-one at either edge that per-bin equality
				// might absorb.
				var sum, wsum int32
				for i := range want {
					sum += got[i]
					wsum += want[i]
				}
				if sum != wsum {
					t.Fatalf("total = %d, want %d", sum, wsum)
				}
			})
		}
	}

	// Degenerate ranges do nothing rather than divide by zero.
	t.Run("emptyRange", func(t *testing.T) {
		c := make([]int32, 4)
		simd.HistogramInto(c, []float64{1, 2, 3}, 5, 5)
		simd.HistogramInto(c, []float64{1, 2, 3}, 5, 1)
		for i, v := range c {
			if v != 0 {
				t.Fatalf("bin %d = %d, want 0", i, v)
			}
		}
	})

	// Integers, where the scale must not truncate to zero.
	t.Run("int", func(t *testing.T) {
		a := make([]int32, 1000)
		for i := range a {
			a[i] = int32(i % 100)
		}
		got := simd.Histogram(a, 10, int32(0), int32(100))
		for i, v := range got {
			if v != 100 {
				t.Fatalf("int bin %d = %d, want 100", i, v)
			}
		}
	})
}
