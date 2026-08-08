package simd_test

import (
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

// The comparison a caller would actually make: the scalar bit-test loop
// these kernels replace, at the densities that decide the crossover.
func BenchmarkColumnar(b *testing.B) {
	for _, n := range []int{64, 256, 4096, 1 << 20} {
		vals := make([]float64, n)
		bm := make([]byte, (n+7)/8)
		rng := rand.New(rand.NewSource(1))
		for i := range vals {
			vals[i] = rng.NormFloat64()
		}
		rng.Read(bm) // ~50% density
		dst := make([]float64, n)

		b.Run(benchName("CompressBits", n), func(b *testing.B) {
			for b.Loop() {
				simd.CompressBitsInto(dst, vals, bm)
			}
		})
		b.Run(benchName("CompressBits-scalar", n), func(b *testing.B) {
			for b.Loop() {
				k := 0
				for i, v := range vals {
					if bm[i>>3]>>(i&7)&1 != 0 {
						dst[k] = v
						k++
					}
				}
			}
		})
		b.Run(benchName("SumValid", n), func(b *testing.B) {
			for b.Loop() {
				sinkF64 = simd.SumValid(vals, bm)
			}
		})
		b.Run(benchName("SumValid-scalar", n), func(b *testing.B) {
			for b.Loop() {
				var s float64
				for i, v := range vals {
					if bm[i>>3]>>(i&7)&1 != 0 {
						s += v
					}
				}
				sinkF64 = s
			}
		})
	}
}

var sinkF64 float64

func benchName(op string, n int) string {
	switch {
	case n >= 1<<20:
		return op + "/n=1M"
	default:
		return op + "/n=" + itoa(n)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [8]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
