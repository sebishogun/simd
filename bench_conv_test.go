package simd_test

// The crossover between direct and frequency-domain convolution, measured
// rather than quoted. Folklore says about 64 taps; that figure comes from
// scalar implementations, and this package's direct path is accelerated.

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkConv []float64

func BenchmarkConvolveFull(b *testing.B) {
	r := rand.New(rand.NewPCG(113, 127))
	const n = 65536
	sig := make([]float64, n)
	for i := range sig {
		sig[i] = r.NormFloat64()
	}
	for _, taps := range []int{16, 64, 256, 1024, 4096} {
		ker := make([]float64, taps)
		for i := range ker {
			ker[i] = r.NormFloat64()
		}
		dst := make([]float64, n+taps-1)
		for _, useFFT := range []bool{false, true} {
			name := "direct"
			if useFFT {
				name = "fft"
			}
			b.Run(fmt.Sprintf("taps=%d/impl=%s", taps, name), func(b *testing.B) {
				for b.Loop() {
					simd.ConvolveForTest(dst, sig, ker, useFFT)
				}
				sinkConv = dst
			})
		}
	}
}
