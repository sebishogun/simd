package simd_test

import (
	"encoding/binary"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// Against the loop each one replaces, at a size where the call overhead has
// long since disappeared.

func BenchmarkCommonPrefixLen(b *testing.B) {
	// A long shared prefix is the case worth measuring: it is what suffix
	// arrays and trie descent produce, and it is the case the blocked scan is
	// built for. A pair that differs in the first byte is a threshold check,
	// not a throughput one.
	const n = 1 << 20
	x := make([]byte, n)
	r := rand.New(rand.NewPCG(1, 2))
	for i := range x {
		x[i] = byte(r.UintN(256))
	}
	y := append([]byte(nil), x...)
	y[n-1] ^= 0xff

	b.Run("scalar", func(b *testing.B) {
		b.SetBytes(n)
		for b.Loop() {
			k := 0
			for k < len(x) && x[k] == y[k] {
				k++
			}
			_ = k
		}
	})
	b.Run("simd", func(b *testing.B) {
		b.SetBytes(n)
		for b.Loop() {
			_ = simd.CommonPrefixLen(x, y)
		}
	})
}

func BenchmarkRollingMin(b *testing.B) {
	const n = 1 << 20
	a := make([]float64, n)
	r := rand.New(rand.NewPCG(3, 4))
	for i := range a {
		a[i] = r.NormFloat64()
	}

	// The window is the axis that matters, not n: the kernel does window-1
	// vectorized passes where the deque does two comparisons per element
	// regardless. Measuring one window would misreport it in either direction.
	for _, w := range []int{4, 8, 16, 32, 64, 256} {
		dst := make([]float64, n-w+1)
		b.Run("deque/w="+itoa(w), func(b *testing.B) {
			idx := make([]int, 0, w)
			for b.Loop() {
				idx = idx[:0]
				for i := range a {
					for len(idx) > 0 && a[idx[len(idx)-1]] >= a[i] {
						idx = idx[:len(idx)-1]
					}
					idx = append(idx, i)
					if idx[0] <= i-w {
						idx = idx[1:]
					}
					if i >= w-1 {
						dst[i-w+1] = a[idx[0]]
					}
				}
			}
		})
		b.Run("simd/w="+itoa(w), func(b *testing.B) {
			for b.Loop() {
				simd.RollingMinInto(dst, a, w)
			}
		})
	}
}

func BenchmarkVarintSize(b *testing.B) {
	const n = 1 << 20
	a := make([]uint64, n)
	r := rand.New(rand.NewPCG(5, 6))
	for i := range a {
		a[i] = r.Uint64() >> (r.UintN(64))
	}

	b.Run("scalar", func(b *testing.B) {
		b.SetBytes(n * 8)
		for b.Loop() {
			t := 0
			for _, v := range a {
				k := 1
				for v >= 0x80 {
					v >>= 7
					k++
				}
				t += k
			}
			_ = t
		}
	})
	b.Run("simd", func(b *testing.B) {
		b.SetBytes(n * 8)
		for b.Loop() {
			_ = simd.VarintSize(a)
		}
	})
}

// The encoder comparison is the one that answers "is this worth using": the
// baseline is what an ordinary Go encoder does, appending as it goes, and the
// difference is that the vectorized size pass lets the buffer be allocated
// once at exactly the right size.
func BenchmarkAppendVarints(b *testing.B) {
	const n = 1 << 18
	a := make([]uint64, n)
	r := rand.New(rand.NewPCG(7, 8))
	for i := range a {
		a[i] = r.Uint64() >> (r.UintN(64))
	}

	b.Run("append-and-grow", func(b *testing.B) {
		buf := make([]byte, binary.MaxVarintLen64)
		for b.Loop() {
			var out []byte
			for _, v := range a {
				out = append(out, buf[:binary.PutUvarint(buf, v)]...)
			}
			_ = out
		}
	})
	b.Run("simd", func(b *testing.B) {
		for b.Loop() {
			_ = simd.AppendVarints(nil, a)
		}
	})
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d [8]byte
	i := len(d)
	for n > 0 {
		i--
		d[i] = byte('0' + n%10)
		n /= 10
	}
	return string(d[i:])
}
