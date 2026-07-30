package simd_test

// Zigzag, affine quantization and Hamming distance, against the plain Go loop
// each replaces.
//
// Both are the shape this library is best at — one pass, no branches, no
// dependence between elements — so the interesting number is not that the
// kernel wins but by how much, and from what size. The scalar loop is written
// the way anyone would write it, because a rigged baseline measures nothing.

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

var (
	sinkZigU32 []uint32
	sinkZigI32 []int32
	sinkQuant  []int8
)

func zigzagGo(dst []uint32, a []int32) {
	n := min(len(dst), len(a))
	for i := 0; i < n; i++ {
		dst[i] = uint32(a[i]<<1) ^ uint32(a[i]>>31)
	}
}

func BenchmarkZigzagEncode(b *testing.B) {
	r := rand.New(rand.NewPCG(41, 43))
	for _, n := range []int{16, 64, 1024, 65536} {
		a := make([]int32, n)
		for i := range a {
			a[i] = int32(r.Uint32())
		}
		dst := make([]uint32, n)
		b.Run(fmt.Sprintf("n=%d/impl=go", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				zigzagGo(dst, a)
			}
			sinkZigU32 = dst
		})
		b.Run(fmt.Sprintf("n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				simd.ZigzagEncodeInt32Into(dst, a)
			}
			sinkZigU32 = dst
		})
	}
}

func BenchmarkZigzagDecode(b *testing.B) {
	r := rand.New(rand.NewPCG(47, 53))
	for _, n := range []int{1024, 65536} {
		a := make([]uint32, n)
		for i := range a {
			a[i] = r.Uint32()
		}
		dst := make([]int32, n)
		b.Run(fmt.Sprintf("n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				simd.ZigzagDecodeInt32Into(dst, a)
			}
			sinkZigI32 = dst
		})
	}
}

func quantizeGo(dst []int8, a []float32, scale float32, zeroPoint int32) {
	n := min(len(dst), len(a))
	for i := 0; i < n; i++ {
		// Deliberately the naive rounding, since that is what the loop this
		// replaces usually does; the kernel rounds half to even, so the two
		// disagree on exact halves. Compared here for speed only.
		v := float64(a[i]/scale) + float64(zeroPoint)
		switch {
		case v < -128:
			dst[i] = -128
		case v > 127:
			dst[i] = 127
		default:
			dst[i] = int8(v + 0.5)
		}
	}
}

func BenchmarkQuantizeInt8(b *testing.B) {
	r := rand.New(rand.NewPCG(59, 61))
	for _, n := range []int{1024, 65536} {
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(r.NormFloat64()) * 40
		}
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("n=%d/impl=go", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				quantizeGo(dst, a, 0.5, 0)
			}
			sinkQuant = dst
		})
		b.Run(fmt.Sprintf("n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				simd.QuantizeInt8(dst, a, 0.5, 0)
			}
			sinkQuant = dst
		})
	}
}

var sinkHamming int

// The comparison that matters for Hamming is not against a scalar loop but
// against the chain it replaces — Xor into a buffer, then PopCount — because
// that is what a caller would write with the operations already in the
// catalogue. The fused kernel should win on the memory traffic alone.
func BenchmarkHammingDistance(b *testing.B) {
	r := rand.New(rand.NewPCG(67, 71))
	for _, n := range []int{64, 1024, 65536} {
		x := make([]byte, n)
		y := make([]byte, n)
		for i := range x {
			x[i] = byte(r.Uint32())
			y[i] = byte(r.Uint32())
		}
		scratch := make([]byte, n)
		b.Run(fmt.Sprintf("n=%d/impl=xor+popcount", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				simd.XorInto(scratch, x, y)
				sinkHamming = simd.PopCount(scratch)
			}
		})
		b.Run(fmt.Sprintf("n=%d/impl=fused", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkHamming = simd.HammingDistance(x, y)
			}
		})
	}
}
