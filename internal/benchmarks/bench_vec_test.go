//go:build goexperiment.simd && amd64

package benchmarks

// The band the vector type exists for: below the dispatch threshold, where a
// non-inlinable call into assembly costs more than the arithmetic saves.
//
// Three implementations of the same expression, so the comparison is the one a
// caller actually faces: a plain Go loop, this library's slice API, and the
// vector type inline.

import (
	"fmt"
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

var sinkVec []float32

func addGo(dst, a, b []float32) {
	n := min(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func BenchmarkSmallAdd(b *testing.B) {
	r := rand.New(rand.NewPCG(2, 3))
	for _, n := range []int{4, 8, 16, 32, 64, 256} {
		x := make([]float32, n)
		y := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64())
			y[i] = float32(r.NormFloat64())
		}
		dst := make([]float32, n)

		b.Run(fmt.Sprintf("n=%d/impl=go", n), func(b *testing.B) {
			for b.Loop() {
				addGo(dst, x, y)
			}
			sinkVec = dst
		})
		b.Run(fmt.Sprintf("n=%d/impl=slice", n), func(b *testing.B) {
			for b.Loop() {
				simd.AddInto(dst, x, y)
			}
			sinkVec = dst
		})
		b.Run(fmt.Sprintf("n=%d/impl=vec", n), func(b *testing.B) {
			for b.Loop() {
				simd.ZipFloat32x8(dst, x, y, func(p, q simd.F32x8) simd.F32x8 { return p.Add(q) })
			}
			sinkVec = dst
		})
	}
}
