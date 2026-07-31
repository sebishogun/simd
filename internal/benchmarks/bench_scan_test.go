package benchmarks

// The baseline that decides whether a scan kernel is worth building is the Go
// loop this library actually ships today, not the same loop compiled by clang.
// Go does not auto-vectorize, so a C probe's "serial" arm is a different and
// much faster program than the one a caller is running.

import (
	"fmt"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkScanF []float64

func BenchmarkScanBaseline(b *testing.B) {
	for _, n := range []int{1024, 262144, 4194304} {
		a32 := make([]float32, n)
		a64 := make([]float64, n)
		i32 := make([]int32, n)
		i64 := make([]int64, n)
		for i := range a32 {
			a32[i] = float32((i%97)-48) * 0.25
			a64[i] = float64(a32[i])
			i32[i] = int32(i%97) - 48
			i64[i] = int64(i32[i])
		}
		d32 := make([]float32, n)
		d64 := make([]float64, n)
		e32 := make([]int32, n)
		e64 := make([]int64, n)

		b.Run(fmt.Sprintf("n=%d/type=f32/op=cumsum", n), func(b *testing.B) {
			b.SetBytes(int64(n * 4))
			for b.Loop() {
				simd.CumSumInto(d32, a32)
			}
		})
		b.Run(fmt.Sprintf("n=%d/type=f64/op=cumsum", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for b.Loop() {
				simd.CumSumInto(d64, a64)
			}
		})
		b.Run(fmt.Sprintf("n=%d/type=i32/op=cumsum", n), func(b *testing.B) {
			b.SetBytes(int64(n * 4))
			for b.Loop() {
				simd.CumSumInto(e32, i32)
			}
		})
		b.Run(fmt.Sprintf("n=%d/type=i64/op=cumsum", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for b.Loop() {
				simd.CumSumInto(e64, i64)
			}
		})
		b.Run(fmt.Sprintf("n=%d/type=i32/op=cumprod", n), func(b *testing.B) {
			b.SetBytes(int64(n * 4))
			for b.Loop() {
				simd.CumProdInto(e32, i32)
			}
		})
	}
}

// The shipped Fast forms against the serial CumSum they opt out of, through
// the exported API rather than the kernel, so dispatch overhead is included.
func BenchmarkFastScan(b *testing.B) {
	for _, n := range []int{1024, 262144, 4194304} {
		a32 := make([]float32, n)
		a64 := make([]float64, n)
		p32 := make([]float32, n)
		p64 := make([]float64, n)
		i32 := make([]int32, n)
		for i := range a32 {
			a32[i] = float32((i%97)-48) * 0.25
			a64[i] = float64(a32[i])
			p32[i] = 1 + float32(i%7-3)*1e-6
			p64[i] = float64(p32[i])
			i32[i] = int32(i%7) + 2
		}
		d32 := make([]float32, n)
		d64 := make([]float64, n)
		e32 := make([]int32, n)

		for _, c := range []struct {
			name string
			fn   func()
			sz   int
		}{
			{"f32/op=cumsum/impl=serial", func() { simd.CumSumInto(d32, a32) }, 4},
			{"f32/op=cumsum/impl=fast", func() { simd.FastCumSumInto(d32, a32) }, 4},
			{"f64/op=cumsum/impl=serial", func() { simd.CumSumInto(d64, a64) }, 8},
			{"f64/op=cumsum/impl=fast", func() { simd.FastCumSumInto(d64, a64) }, 8},
			{"f32/op=cumprod/impl=serial", func() { simd.CumProdInto(d32, p32) }, 4},
			{"f32/op=cumprod/impl=fast", func() { simd.FastCumProdInto(d32, p32) }, 4},
			{"f64/op=cumprod/impl=serial", func() { simd.CumProdInto(d64, p64) }, 8},
			{"f64/op=cumprod/impl=fast", func() { simd.FastCumProdInto(d64, p64) }, 8},
			// The integer product needs no Fast form; this is the exact
			// kernel against the number this file recorded before it existed.
			{"i32/op=cumprod/impl=exact_kernel", func() { simd.CumProdInto(e32, i32) }, 4},
		} {
			b.Run(fmt.Sprintf("n=%d/%s", n, c.name), func(b *testing.B) {
				b.SetBytes(int64(n * c.sz))
				for b.Loop() {
					c.fn()
				}
			})
		}
	}
}
