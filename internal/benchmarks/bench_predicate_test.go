package benchmarks

// predicate.go is entirely compositions of already-accelerated primitives, and
// none of it was measured when it shipped. The question each benchmark answers
// is whether the composition beats the loop a caller writes without this
// package — if it does not, the operation wants a real kernel.

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

var (
	sinkPredI int
	sinkPredF float64
)

func predInput(n int, r *rand.Rand) []float64 {
	a := make([]float64, n)
	for i := range a {
		switch i % 16 {
		case 0:
			a[i] = math.NaN()
		case 1:
			a[i] = math.Inf(1)
		default:
			a[i] = r.NormFloat64()
		}
	}
	return a
}

func BenchmarkPredicates(b *testing.B) {
	r := rand.New(rand.NewPCG(163, 167))
	for _, n := range []int{4096, 65536, 1 << 20} {
		a := predInput(n, r)
		mask := make([]bool, n)
		scratch := make([]float64, n)
		dst := make([]float64, n)

		b.Run(fmt.Sprintf("CountNaN/n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 8)
			for b.Loop() {
				sinkPredI = simd.CountNaN(a, mask)
			}
		})
		b.Run(fmt.Sprintf("CountNaN/n=%d/impl=naive", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 8)
			for b.Loop() {
				c := 0
				for _, v := range a {
					if v != v {
						c++
					}
				}
				sinkPredI = c
			}
		})
		b.Run(fmt.Sprintf("NanSum/n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 8)
			for b.Loop() {
				sinkPredF = simd.NanSum(a, scratch, mask)
			}
		})
		b.Run(fmt.Sprintf("NanSum/n=%d/impl=naive", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 8)
			for b.Loop() {
				var s float64
				for _, v := range a {
					if v == v {
						s += v
					}
				}
				sinkPredF = s
			}
		})
		b.Run(fmt.Sprintf("Sign/n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 8)
			for b.Loop() {
				simd.SignInto(dst, a, scratch, mask)
			}
			sinkPredF = dst[0]
		})
		b.Run(fmt.Sprintf("Sign/n=%d/impl=naive", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 8)
			for b.Loop() {
				for i, v := range a {
					switch {
					case v != v:
						dst[i] = v
					case v > 0:
						dst[i] = 1
					case v < 0:
						dst[i] = -1
					default:
						dst[i] = 0
					}
				}
			}
			sinkPredF = dst[0]
		})
	}
}
