//go:build amd64 && !purego

package amd64

import (
	"fmt"
	"testing"
)

// Benchmarks comparing the generated kernels against a plain Go loop.
//
// The sizes span the range where the answer changes. A Go-to-assembly call is
// a fixed cost of roughly 1.4ns and can never be inlined, so at small n the
// call dominates and the scalar loop wins; by 4K the call is noise and the
// vector width decides. Where exactly the lines cross is what sets the
// per-kernel thresholds in the manifest, and it has to be measured on real
// hardware rather than guessed.
var sizes = []int{4, 16, 64, 256, 4096, 65536}

func goAdd(dst, a, b []float64) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func goSum(a []float64) float64 {
	var acc [sumLanes]float64
	i := 0
	for ; i+sumLanes <= len(a); i += sumLanes {
		for j := range sumLanes {
			acc[j] += a[i+j]
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		acc[j] += a[i]
	}
	for w := sumLanes / 2; w >= 1; w /= 2 {
		for j := range w {
			acc[j] += acc[j+w]
		}
	}
	return acc[0]
}

func goDot(a, b []float64) float64 {
	var acc [sumLanes]float64
	i := 0
	for ; i+sumLanes <= len(a); i += sumLanes {
		for j := range sumLanes {
			acc[j] += a[i+j] * b[i+j]
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		acc[j] += a[i] * b[i]
	}
	for w := sumLanes / 2; w >= 1; w /= 2 {
		for j := range w {
			acc[j] += acc[j+w]
		}
	}
	return acc[0]
}

func benchInputs(n int) (a, b, dst []float64) {
	a, b, dst = make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range a {
		a[i], b[i] = float64(i)*0.5, float64(i)*0.25
	}
	return
}

func BenchmarkAddFloat64(b *testing.B) {
	for _, n := range sizes {
		x, y, dst := benchInputs(n)
		b.Run(fmt.Sprintf("n=%d/go", n), func(b *testing.B) {
			for b.Loop() {
				goAdd(dst, x, y)
			}
		})
		b.Run(fmt.Sprintf("n=%d/sse2", n), func(b *testing.B) {
			for b.Loop() {
				addFloat64SSE2(dst, x, y)
			}
		})
		if hasAVX2() {
			b.Run(fmt.Sprintf("n=%d/avx2", n), func(b *testing.B) {
				for b.Loop() {
					addFloat64AVX2(dst, x, y)
				}
			})
		}
	}
}

func BenchmarkSumFloat64(b *testing.B) {
	for _, n := range sizes {
		x, _, _ := benchInputs(n)
		b.Run(fmt.Sprintf("n=%d/go", n), func(b *testing.B) {
			for b.Loop() {
				sinkF = goSum(x)
			}
		})
		b.Run(fmt.Sprintf("n=%d/sse2", n), func(b *testing.B) {
			for b.Loop() {
				sinkF = sumFloat64SSE2(x)
			}
		})
		if hasAVX2() {
			b.Run(fmt.Sprintf("n=%d/avx2", n), func(b *testing.B) {
				for b.Loop() {
					sinkF = sumFloat64AVX2(x)
				}
			})
		}
	}
}

func BenchmarkDotFloat64(b *testing.B) {
	for _, n := range sizes {
		x, y, _ := benchInputs(n)
		b.Run(fmt.Sprintf("n=%d/go", n), func(b *testing.B) {
			for b.Loop() {
				sinkF = goDot(x, y)
			}
		})
		b.Run(fmt.Sprintf("n=%d/sse2", n), func(b *testing.B) {
			for b.Loop() {
				sinkF = dotFloat64SSE2(x, y)
			}
		})
		if hasAVX2() {
			b.Run(fmt.Sprintf("n=%d/avx2", n), func(b *testing.B) {
				for b.Loop() {
					sinkF = dotFloat64AVX2(x, y)
				}
			})
		}
	}
}
