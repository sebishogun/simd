package simd_test

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

func TestVarintDecode(t *testing.T) {
	rng := rand.New(rand.NewSource(15))
	for trial := 0; trial < 300; trial++ {
		n := rng.Intn(500)
		vals := make([]uint64, n)
		var src []byte
		for i := range vals {
			vals[i] = rng.Uint64() >> uint(rng.Intn(64))
			src = binary.AppendUvarint(src, vals[i])
		}
		d1 := make([]uint64, n)
		d2 := make([]uint64, n)
		n1, c1 := ref.VarintDecodeU64(d1, src)
		n2, c2 := simd.VarintDecode(d2, src)
		if n1 != n2 || c1 != c2 || n1 != n || c1 != len(src) {
			t.Fatalf("trial %d: ref (%d,%d) kernel (%d,%d) want (%d,%d)",
				trial, n1, c1, n2, c2, n, len(src))
		}
		for i := range d1 {
			if d1[i] != vals[i] || d2[i] != vals[i] {
				t.Fatalf("trial %d: [%d] ref %d kernel %d want %d", trial, i, d1[i], d2[i], vals[i])
			}
		}
		// Truncations and mutations: both sides must agree exactly.
		for m := 0; m < 30 && len(src) > 0; m++ {
			cut := src[:rng.Intn(len(src))]
			n1, c1 = ref.VarintDecodeU64(d1, cut)
			n2, c2 = simd.VarintDecode(d2, cut)
			if n1 != n2 || c1 != c2 {
				t.Fatalf("trial %d cut: ref (%d,%d) kernel (%d,%d)", trial, n1, c1, n2, c2)
			}
			mut := append([]byte(nil), src...)
			mut[rng.Intn(len(mut))] |= 0x80
			n1, c1 = ref.VarintDecodeU64(d1, mut)
			n2, c2 = simd.VarintDecode(d2, mut)
			if n1 != n2 || c1 != c2 {
				t.Fatalf("trial %d mut: ref (%d,%d) kernel (%d,%d)", trial, n1, c1, n2, c2)
			}
			for i := 0; i < n1; i++ {
				if d1[i] != d2[i] {
					t.Fatalf("trial %d mut: [%d] differs", trial, i)
				}
			}
		}
	}
}

func BenchmarkVarintDecode(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	var src []byte
	n := 1 << 16
	for i := 0; i < n; i++ {
		src = binary.AppendUvarint(src, rng.Uint64()>>uint(rng.Intn(48)+16))
	}
	dst := make([]uint64, n)
	b.Run("kernel", func(b *testing.B) {
		b.SetBytes(int64(len(src)))
		for b.Loop() {
			sinkInt, _ = simd.VarintDecode(dst, src)
		}
	})
	b.Run("stdlib-loop", func(b *testing.B) {
		b.SetBytes(int64(len(src)))
		for b.Loop() {
			i, d := 0, 0
			for i < len(src) && d < len(dst) {
				v, k := binary.Uvarint(src[i:])
				if k <= 0 {
					break
				}
				dst[d] = v
				d++
				i += k
			}
			sinkInt = d
		}
	})
}
