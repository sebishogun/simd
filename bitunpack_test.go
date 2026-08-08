package simd_test

import (
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

func TestBitUnpackFast(t *testing.T) {
	rng := rand.New(rand.NewSource(14))
	for bits := int32(1); bits <= 32; bits++ {
		for _, n := range []int{0, 1, 31, 32, 33, 64, 96, 1000} {
			vals := make([]uint32, n)
			mask := uint32(1)<<uint(bits) - 1
			if bits == 32 {
				mask = ^uint32(0)
			}
			for i := range vals {
				vals[i] = rng.Uint32() & mask
			}
			packed := make([]uint32, (n*int(bits)+31)/32+1)
			simd.BitPackInto(packed, vals, bits)
			got := make([]uint32, n)
			simd.BitUnpackInto(got, packed, bits)
			for i := range vals {
				if got[i] != vals[i] {
					t.Fatalf("bits=%d n=%d: [%d] got %d want %d", bits, n, i, got[i], vals[i])
				}
			}
		}
	}
}

func BenchmarkBitUnpack(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	n := 1 << 16
	for _, bits := range []int32{3, 8, 13} {
		vals := make([]uint32, n)
		mask := uint32(1)<<uint(bits) - 1
		for i := range vals {
			vals[i] = rng.Uint32() & mask
		}
		packed := make([]uint32, (n*int(bits)+31)/32+1)
		simd.BitPackInto(packed, vals, bits)
		dst := make([]uint32, n)
		b.Run("fast/w="+string(rune('0'+bits%10)), func(b *testing.B) {
			b.SetBytes(int64(n * 4))
			for b.Loop() {
				simd.BitUnpackInto(dst, packed, bits)
			}
		})
		b.Run("general/w="+string(rune('0'+bits%10)), func(b *testing.B) {
			b.SetBytes(int64(n * 4))
			for b.Loop() {
				ref.BitUnpackU32(dst, packed, bits)
			}
		})
	}
}
