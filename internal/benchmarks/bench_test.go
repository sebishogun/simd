package benchmarks

import (
	"fmt"
	"testing"

	"github.com/sebishogun/simd"
)

// Benchmarks of the public API.
//
// Run them twice to see what the assembly is worth:
//
//	go test -run '^$' -bench . -count 10 > asm.txt
//	go test -run '^$' -bench . -count 10 -tags purego > pure.txt
//	benchstat pure.txt asm.txt
//
// The sizes bracket the crossover. Below the per-kernel threshold the
// dispatcher runs the portable path deliberately, because a Go-to-assembly
// call is a fixed cost that cannot be inlined away and small inputs cannot
// amortize it; above it the vector width takes over.
var benchSizes = []int{8, 32, 128, 1024, 16384, 262144}

func benchF64(n int) (a, b, dst []float64) {
	a, b, dst = make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range a {
		a[i], b[i] = float64(i)*0.5, float64(i)*0.25
	}
	return
}

func BenchmarkAdd(b *testing.B) {
	for _, n := range benchSizes {
		x, y, dst := benchF64(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for b.Loop() {
				simd.AddInto(dst, x, y)
			}
		})
	}
}

func BenchmarkSum(b *testing.B) {
	for _, n := range benchSizes {
		x, _, _ := benchF64(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for b.Loop() {
				sinkF = simd.Sum(x)
			}
		})
	}
}

func BenchmarkDot(b *testing.B) {
	for _, n := range benchSizes {
		x, y, _ := benchF64(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for b.Loop() {
				sinkF = simd.Dot(x, y)
			}
		})
	}
}

// AddScaled is the fused AXPY, and the reason a fused catalogue exists: the
// same result via separate Mul and Add calls costs two passes over memory.
func BenchmarkAddScaled(b *testing.B) {
	for _, n := range benchSizes {
		x, y, _ := benchF64(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for b.Loop() {
				simd.AddScaled(x, y, 1.5)
			}
		})
	}
}

// A whole task, to show the primitives compose without giving the speedup back.
func BenchmarkCosineSimilarity(b *testing.B) {
	for _, n := range []int{128, 1024, 16384} {
		x, y, _ := benchF64(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for b.Loop() {
				sinkF = simd.CosineSimilarity(x, y)
			}
		})
	}
}
