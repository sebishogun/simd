package benchmarks

// The FFT against the definition it replaces. The naive DFT is O(n^2) so it is
// only run at small sizes; past a few hundred points the comparison stops
// being interesting and the absolute throughput is what matters.

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkFFT []complex128

func BenchmarkFFT(b *testing.B) {
	r := rand.New(rand.NewPCG(101, 103))
	for _, n := range []int{64, 256, 1024, 4096, 65536} {
		a := make([]complex128, n)
		for i := range a {
			a[i] = complex(r.NormFloat64(), r.NormFloat64())
		}
		p := simd.NewFFTPlan(n)
		dst := make([]complex128, n)

		b.Run(fmt.Sprintf("n=%d/impl=fft", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 16)
			for b.Loop() {
				simd.FFTInto(p, dst, a)
			}
			sinkFFT = dst
		})
		// The plan is the expensive part, so a caller that rebuilds it per
		// transform gets a very different number. Worth showing.
		b.Run(fmt.Sprintf("n=%d/impl=fft+plan", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 16)
			for b.Loop() {
				sinkFFT = simd.FFT(a)
			}
		})
		if n <= 1024 {
			b.Run(fmt.Sprintf("n=%d/impl=naive", n), func(b *testing.B) {
				b.SetBytes(int64(n) * 16)
				for b.Loop() {
					sinkFFT = naiveDFT(a, false)
				}
			})
		}
	}
}

// RFFT against transforming the same real signal as complex, which is what a
// caller does without it.
func BenchmarkRFFT(b *testing.B) {
	r := rand.New(rand.NewPCG(151, 157))
	for _, n := range []int{1024, 65536} {
		x := make([]float64, n)
		cx := make([]complex128, n)
		for i := range x {
			x[i] = r.NormFloat64()
			cx[i] = complex(x[i], 0)
		}
		rp := simd.NewRFFTPlan(n)
		cp := simd.NewFFTPlan(n)
		rd := make([]complex128, rp.OutLen())
		cd := make([]complex128, n)
		scr := make([]complex128, n/2)
		b.Run(fmt.Sprintf("n=%d/impl=real", n), func(b *testing.B) {
			for b.Loop() {
				simd.RFFTInto(rp, rd, x, scr)
			}
			sinkFFT = rd
		})
		b.Run(fmt.Sprintf("n=%d/impl=complex", n), func(b *testing.B) {
			for b.Loop() {
				simd.FFTInto(cp, cd, cx)
			}
			sinkFFT = cd
		})
	}
}
