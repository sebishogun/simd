package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

func BenchmarkMatMulParallel(b *testing.B) {
	for _, n := range []int{128, 256, 512, 1024} {
		r := rand.New(rand.NewPCG(3, 5))
		a := make([]float64, n*n)
		m := make([]float64, n*n)
		for i := range a {
			a[i], m[i] = r.NormFloat64(), r.NormFloat64()
		}
		c := make([]float64, n*n)
		b.Run(fmt.Sprintf("n=%d/serial", n), func(b *testing.B) {
			for b.Loop() {
				simd.MatMulInto(c, a, m, n, n, n)
			}
		})
		b.Run(fmt.Sprintf("n=%d/parallel", n), func(b *testing.B) {
			for b.Loop() {
				simd.MatMulParallelInto(c, a, m, n, n, n)
			}
		})
	}
}

func BenchmarkGemvParallel(b *testing.B) {
	for _, n := range []int{512, 1024, 2048, 4096} {
		r := rand.New(rand.NewPCG(7, 9))
		a := make([]float64, n*n)
		for i := range a {
			a[i] = r.NormFloat64()
		}
		x, y := make([]float64, n), make([]float64, n)
		b.Run(fmt.Sprintf("n=%d/serial", n), func(b *testing.B) {
			for b.Loop() {
				simd.GemvInto(y, a, x, n, n)
			}
		})
		b.Run(fmt.Sprintf("n=%d/parallel", n), func(b *testing.B) {
			for b.Loop() {
				simd.GemvParallelInto(y, a, x, n, n)
			}
		})
	}
}
