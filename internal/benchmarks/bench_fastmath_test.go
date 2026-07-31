package benchmarks

// What the Fast tier is worth, against its own accurate twin.
//
// This is the only comparison that means anything for it. Against a scalar Go
// loop both tiers win by the same enormous factor and the number says nothing
// about whether the trade is worth making; against each other it says exactly
// that.
//
//	go test -run '^$' -bench FastMath -count 10 | benchstat -col /tier -
//
// The sizes start above the threshold, because below it both tiers run the
// same portable code and the comparison is empty.

import (
	"fmt"
	"testing"

	"github.com/sebishogun/simd"
)

var fastSizes = []int{256, 4096, 65536}

func benchPair(b *testing.B, name string, fast, accurate func()) {
	for _, tc := range []struct {
		tier string
		fn   func()
	}{{"accurate", accurate}, {"fast", fast}} {
		b.Run(name+"/tier="+tc.tier, func(b *testing.B) {
			for b.Loop() {
				tc.fn()
			}
		})
	}
}

func fastInputs(n int) (a, dst []float64) {
	a, dst = make([]float64, n), make([]float64, n)
	for i := range a {
		// Spread over a range that exercises the reduction rather than
		// sitting in the polynomial's own interval, which is where a real
		// caller's data is and where the reduction cost is visible.
		a[i] = float64(i%2001)*0.01 - 10
	}
	return
}

func fastInputs32(n int) (a, dst []float32) {
	a64, _ := fastInputs(n)
	a, dst = make([]float32, n), make([]float32, n)
	for i := range a {
		a[i] = float32(a64[i])
	}
	return
}

func BenchmarkFastMathF64(b *testing.B) {
	for _, n := range fastSizes {
		a, dst := fastInputs(n)
		for _, c := range []struct {
			name           string
			fast, accurate func(dst, a []float64)
		}{
			{"Exp", simd.FastExpInto[float64], simd.ExpInto[float64]},
			{"Log", simd.FastLogInto[float64], simd.LogInto[float64]},
			{"Sin", simd.FastSinInto[float64], simd.SinInto[float64]},
			{"Tanh", simd.FastTanhInto[float64], simd.TanhInto[float64]},
			{"Sigmoid", simd.FastSigmoidInto[float64], simd.SigmoidInto[float64]},
			{"Atan", simd.FastAtanInto[float64], simd.AtanInto[float64]},
		} {
			benchPair(b, fmt.Sprintf("%s/n=%d", c.name, n),
				func() { c.fast(dst, a) },
				func() { c.accurate(dst, a) })
		}
	}
}

func BenchmarkFastMathF32(b *testing.B) {
	for _, n := range fastSizes {
		a, dst := fastInputs32(n)
		for _, c := range []struct {
			name           string
			fast, accurate func(dst, a []float32)
		}{
			{"Exp", simd.FastExpInto[float32], simd.ExpInto[float32]},
			{"Log", simd.FastLogInto[float32], simd.LogInto[float32]},
			{"Sin", simd.FastSinInto[float32], simd.SinInto[float32]},
			{"Tanh", simd.FastTanhInto[float32], simd.TanhInto[float32]},
		} {
			benchPair(b, fmt.Sprintf("%s/n=%d", c.name, n),
				func() { c.fast(dst, a) },
				func() { c.accurate(dst, a) })
		}
	}
}

// BenchmarkFastMathPow covers the binary shape, which has the largest kernel
// of the set — pow is exp(y*log(x)) with both reductions and both special-case
// trees, so it is where the tier has the most to save and the least of it is
// polynomial.
func BenchmarkFastMathPow(b *testing.B) {
	for _, n := range fastSizes {
		a, dst := fastInputs(n)
		e := make([]float64, n)
		for i := range e {
			a[i] = float64(i%1000)*0.01 + 0.5
			e[i] = float64(i%7) - 3
		}
		benchPair(b, fmt.Sprintf("Pow/n=%d", n),
			func() { simd.FastPowInto(dst, a, e) },
			func() { simd.PowInto(dst, a, e) })
	}
}
