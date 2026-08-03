package text

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

func jsonish(n int) []byte {
	b := make([]byte, 0, n)
	r := rand.New(rand.NewSource(7))
	for len(b) < n {
		b = append(b, fmt.Sprintf(`{"key%d":"value %d","n":%d,"a":[1,2,3]},`,
			r.Intn(1000), r.Intn(1000), r.Intn(100000))...)
	}
	return b[:n]
}

func BenchmarkMasksSeparate(b *testing.B) {
	for _, n := range []int{1 << 10, 1 << 16, 1 << 20, 2 << 20} {
		data := jsonish(n)
		m := (n + 7) / 8
		q, e, st, ct, ws := make([]byte, m), make([]byte, m), make([]byte, m), make([]byte, m), make([]byte, m)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				simd.MaskBits(q, data, '"')
				simd.MaskBits(e, data, '\\')
				simd.MaskBitsAny(st, data, "{}[]")
				simd.MaskBitsLess(ct, data, 0x20)
				simd.MaskBitsAny(ws, data, " \t\n\r")
			}
		})
	}
}

func BenchmarkMasksFused(b *testing.B) {
	for _, n := range []int{1 << 10, 1 << 16, 1 << 20, 2 << 20} {
		data := jsonish(n)
		dst := make([]byte, 5*((n+7)/8))
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				simd.JSONMasks(dst, data, simd.JSONMaskAll)
			}
		})
	}
}

// Three masks, which is what Scan asks for.
func BenchmarkMasksSeparate3(b *testing.B) {
	for _, n := range []int{1 << 16, 1 << 20} {
		data := jsonish(n)
		m := (n + 7) / 8
		q, e, st := make([]byte, m), make([]byte, m), make([]byte, m)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				simd.MaskBits(q, data, '"')
				simd.MaskBits(e, data, '\\')
				simd.MaskBitsAny(st, data, "{}[]")
			}
		})
	}
}

func BenchmarkMasksFused3(b *testing.B) {
	for _, n := range []int{1 << 16, 1 << 20} {
		data := jsonish(n)
		dst := make([]byte, 5*((n+7)/8))
		want := uint32(simd.JSONMaskQuote | simd.JSONMaskEscape | simd.JSONMaskStructural)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				simd.JSONMasks(dst, data, want)
			}
		})
	}
}
