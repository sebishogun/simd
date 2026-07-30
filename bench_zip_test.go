package simd_test

// Is a general n-ary closure combinator worth shipping?
//
// The proposal is ZipInto(dst, f, srcs...) so a caller can express a fused
// expression this library has never heard of and still make one pass. The
// objection is that a closure call per element defeats vectorization
// completely, so the thing being offered may be slower than what the caller
// would have written unaided — in which case the honest answer is a pointer to
// the fused catalogue and no new function.
//
// This measures the four things a caller could do for one concrete expression,
// dst = a*b + c:
//
//	handwritten   a plain Go loop, which is the real baseline
//	composed      MulInto then AddInto — two passes over memory, no closure
//	zip           the proposed combinator, closure per element
//	zipShaped     the combinator recognizing the shape and dispatching to
//	              real kernels, which is what FilterInto does

import (
	"fmt"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkZip float64

// zipClosure is the honest implementation of the general form: it cannot know
// what f does, so it calls it once per element.
func zipClosure(dst []float64, f func(v ...float64) float64, srcs ...[]float64) {
	n := len(dst)
	for _, s := range srcs {
		n = min(n, len(s))
	}
	args := make([]float64, len(srcs))
	for i := range n {
		for j, s := range srcs {
			args[j] = s[i]
		}
		dst[i] = f(args...)
	}
}

// zipBinaryClosure is the same idea specialized to a fixed arity, which
// removes the variadic slice build per element — the most generous version of
// the general design.
func zipBinaryClosure(dst []float64, f func(x, y, z float64) float64, a, b, c []float64) {
	n := min(len(dst), min(len(a), min(len(b), len(c))))
	for i := range n {
		dst[i] = f(a[i], b[i], c[i])
	}
}

func BenchmarkZipCombinator(b *testing.B) {
	for _, n := range []int{1024, 262144, 4194304} {
		a := make([]float64, n)
		bb := make([]float64, n)
		c := make([]float64, n)
		dst := make([]float64, n)
		tmp := make([]float64, n)
		for i := range a {
			a[i] = float64(i%97) * 0.5
			bb[i] = float64(i%89) * 0.25
			c[i] = float64(i % 83)
		}

		b.Run(fmt.Sprintf("n=%d/impl=handwritten", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 3))
			for b.Loop() {
				for i := range dst {
					dst[i] = a[i]*bb[i] + c[i]
				}
				sinkZip = dst[n-1]
			}
		})

		b.Run(fmt.Sprintf("n=%d/impl=composed", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 3))
			for b.Loop() {
				simd.MulInto(tmp, a, bb)
				simd.AddInto(dst, tmp, c)
				sinkZip = dst[n-1]
			}
		})

		b.Run(fmt.Sprintf("n=%d/impl=zipVariadic", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 3))
			f := func(v ...float64) float64 { return v[0]*v[1] + v[2] }
			for b.Loop() {
				zipClosure(dst, f, a, bb, c)
				sinkZip = dst[n-1]
			}
		})

		b.Run(fmt.Sprintf("n=%d/impl=zipFixedArity", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 3))
			f := func(x, y, z float64) float64 { return x*y + z }
			for b.Loop() {
				zipBinaryClosure(dst, f, a, bb, c)
				sinkZip = dst[n-1]
			}
		})

		// The shape this library already has a kernel for, as the ceiling:
		// one pass, no closure. Mul3 is a*b*c rather than a*b+c, but it is the
		// same memory traffic and the same call shape, so it bounds what a
		// shape-recognizing ZipInto could possibly reach.
		b.Run(fmt.Sprintf("n=%d/impl=fusedKernelCeiling", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 3))
			for b.Loop() {
				simd.MulAll(dst, a, bb, c)
				sinkZip = dst[n-1]
			}
		})
	}
}
