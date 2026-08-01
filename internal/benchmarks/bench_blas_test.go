package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

func BenchmarkRankOne(b *testing.B) {
	for _, n := range []int{64, 512, 2048} {
		r := rand.New(rand.NewPCG(1, 2))
		a := make([]float64, n*n)
		x, y := make([]float64, n), make([]float64, n)
		for i := range a {
			a[i] = r.NormFloat64()
		}
		for i := range x {
			x[i], y[i] = r.NormFloat64(), r.NormFloat64()
		}
		b.Run(fmt.Sprintf("n=%d/portable", n), func(b *testing.B) {
			for b.Loop() {
				ref.RankOneFloat(a, x, y, 1.5, n, n)
			}
		})
		b.Run(fmt.Sprintf("n=%d/simd", n), func(b *testing.B) {
			for b.Loop() {
				simd.RankOneInto(a, x, y, 1.5, n, n)
			}
		})
	}
}

func BenchmarkRotate(b *testing.B) {
	for _, n := range []int{1000, 100000} {
		r := rand.New(rand.NewPCG(3, 5))
		x, y := make([]float64, n), make([]float64, n)
		for i := range x {
			x[i], y[i] = r.NormFloat64(), r.NormFloat64()
		}
		b.Run(fmt.Sprintf("n=%d/portable", n), func(b *testing.B) {
			for b.Loop() {
				ref.RotateFloat(x, y, 0.6, 0.8)
			}
		})
		b.Run(fmt.Sprintf("n=%d/simd", n), func(b *testing.B) {
			for b.Loop() {
				simd.Rotate(x, y, 0.6, 0.8)
			}
		})
	}
}

func BenchmarkSwapSlices(b *testing.B) {
	for _, n := range []int{1000, 100000} {
		x, y := make([]float64, n), make([]float64, n)
		b.Run(fmt.Sprintf("n=%d/portable", n), func(b *testing.B) {
			for b.Loop() {
				ref.SwapFloat(x, y)
			}
		})
		b.Run(fmt.Sprintf("n=%d/simd", n), func(b *testing.B) {
			for b.Loop() {
				simd.Swap(x, y)
			}
		})
	}
}
