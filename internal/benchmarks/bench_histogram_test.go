package benchmarks

// This benchmark exists to keep a negative result honest.
//
// Splitting the accumulation across four private tables, to break the
// store-to-load dependency on counts[k], was implemented and measured 22% to
// 39% SLOWER than the plain loop at every size and both distributions —
// including the skewed data it was meant to rescue. It was removed; see the
// comment at the top of histogram.go for the numbers.
//
// What remains here is the comparison against the loop a caller writes without
// this package, so that if BincountInto ever grows a cleverer implementation
// there is something to hold it to. Anything at parity is doing its job; the
// operation is a scatter with an increment and no architecture but AVX-512 has
// an instruction for the conflict.
//
//	go test -run '^$' -bench Histogram -count 8 | benchstat -col /impl -

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkHist []int32

// The distribution is the variable that matters. Uniform data spreads the
// increments over the whole table and rarely conflicts; skewed data hits the
// same counter repeatedly, which is exactly the dependency the split is meant
// to break.
func histInput(n, bins int, skewed bool, r *rand.Rand) []int32 {
	a := make([]int32, n)
	for i := range a {
		if skewed {
			a[i] = int32(r.IntN(4)) // four hot bins
		} else {
			a[i] = int32(r.IntN(bins))
		}
	}
	return a
}

func BenchmarkHistogramBincount(b *testing.B) {
	r := rand.New(rand.NewPCG(43, 47))
	for _, n := range []int{4096, 65536, 1 << 20} {
		for _, bins := range []int{16, 256} {
			for _, skew := range []bool{false, true} {
				name := "uniform"
				if skew {
					name = "skewed"
				}
				a := histInput(n, bins, skew, r)
				counts := make([]int32, bins)

				b.Run(fmt.Sprintf("n=%d/bins=%d/%s/impl=simd", n, bins, name), func(b *testing.B) {
					b.SetBytes(int64(n) * 4)
					for b.Loop() {
						clear(counts)
						simd.BincountInto(counts, a)
					}
					sinkHist = counts
				})
				b.Run(fmt.Sprintf("n=%d/bins=%d/%s/impl=naive", n, bins, name), func(b *testing.B) {
					b.SetBytes(int64(n) * 4)
					for b.Loop() {
						clear(counts)
						for _, v := range a {
							if v >= 0 && int(v) < bins {
								counts[v]++
							}
						}
					}
					sinkHist = counts
				})
			}
		}
	}
}

// BenchmarkTranspose checks the claim that blocking is what matters here.
//
// The naive arm is the loop a caller writes: read a row, write a column. If
// the blocked kernel is not clearly ahead at sizes past the cache, the
// blocking is not earning its complexity and should go.
func BenchmarkTranspose(b *testing.B) {
	for _, dims := range [][2]int{{64, 64}, {512, 512}, {2048, 2048}, {4096, 512}} {
		m, n := dims[0], dims[1]
		a := make([]float64, m*n)
		for i := range a {
			a[i] = float64(i)
		}
		dst := make([]float64, m*n)
		b.Run(fmt.Sprintf("%dx%d/impl=simd", m, n), func(b *testing.B) {
			b.SetBytes(int64(m) * int64(n) * 8)
			for b.Loop() {
				simd.TransposeInto(dst, a, m, n)
			}
		})
		b.Run(fmt.Sprintf("%dx%d/impl=naive", m, n), func(b *testing.B) {
			b.SetBytes(int64(m) * int64(n) * 8)
			for b.Loop() {
				for i := range m {
					for j := range n {
						dst[j*m+i] = a[i*n+j]
					}
				}
			}
		})
	}
}
