package simd_test

import (
	"fmt"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkGemm []float64

func BenchmarkMatMulScratch(b *testing.B) {
	for _, n := range []int{256, 512, 1024} {
		a := make([]float64, n*n)
		bb := make([]float64, n*n)
		d := make([]float64, n*n)
		for i := range a {
			a[i] = float64(i%97) * 0.5
			bb[i] = float64(i%89) * 0.25
		}
		scratch := make([]float64, simd.GemmPackLen[float64](n, n))
		flops := 2 * int64(n) * int64(n) * int64(n)
		b.Run(fmt.Sprintf("n=%d/impl=plain", n), func(b *testing.B) {
			b.SetBytes(flops)
			for b.Loop() {
				simd.MatMulInto(d, a, bb, n, n, n)
			}
			sinkGemm = d
		})
		b.Run(fmt.Sprintf("n=%d/impl=packed", n), func(b *testing.B) {
			b.SetBytes(flops)
			for b.Loop() {
				simd.MatMulIntoScratch(d, a, bb, scratch, n, n, n)
			}
			sinkGemm = d
		})
	}
}
