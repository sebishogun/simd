package simd_test

// Benchmarks for the complex reductions, against the portable path they used
// to be.
//
// These are the three complex operations with a numerical contract — the lane
// an element lands in and the tree that folds the lanes are fixed, so the
// answer does not change with vector width — which is why they were the last
// complex operations to get a kernel.
//
//	go test -run '^$' -bench Complex -count 8 | benchstat -col /impl -

import (
	"fmt"
	"math/cmplx"
	"testing"

	"github.com/sebishogun/simd"
)

var (
	sinkC64  complex64
	sinkC128 complex128
)

func complexInput128(n int) []complex128 {
	a := make([]complex128, n)
	for i := range a {
		a[i] = complex(float64(i%97)*0.5-24, float64(i%53)*0.25-6)
	}
	return a
}

func complexInput64(n int) []complex64 {
	a := make([]complex64, n)
	for i := range a {
		a[i] = complex(float32(i%97)*0.5-24, float32(i%53)*0.25-6)
	}
	return a
}

func BenchmarkComplexReduce(b *testing.B) {
	for _, n := range []int{1024, 65536, 1 << 20} {
		a128, b128 := complexInput128(n), complexInput128(n)
		a64, b64 := complexInput64(n), complexInput64(n)

		// 16 bytes per complex128 element, two slices for the dots.
		b.Run(fmt.Sprintf("Sum/c128/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 16)
			for b.Loop() {
				sinkC128 = simd.SumComplex(a128)
			}
		})
		b.Run(fmt.Sprintf("Dot/c128/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 32)
			for b.Loop() {
				sinkC128 = simd.DotComplex(a128, b128)
			}
		})
		b.Run(fmt.Sprintf("DotConj/c128/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 32)
			for b.Loop() {
				sinkC128 = simd.DotComplexConj(a128, b128)
			}
		})
		b.Run(fmt.Sprintf("DotConj/c64/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 16)
			for b.Loop() {
				sinkC64 = simd.DotComplexConj(a64, b64)
			}
		})

		// The naive loop a caller writes without this package. It is not the
		// reference — the reference keeps the sixteen-lane discipline — but it
		// is what the comparison is actually against in practice.
		b.Run(fmt.Sprintf("DotConj/c128/n=%d/impl=naive", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 32)
			for b.Loop() {
				var s complex128
				for i := range a128 {
					s += a128[i] * cmplx.Conj(b128[i])
				}
				sinkC128 = s
			}
		})
	}
}
