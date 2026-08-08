package simd_test

import (
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

func TestShuffleFamily(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for _, n := range []int{0, 1, 15, 16, 17, 100, 4096} {
		a := make([]byte, n)
		b := make([]byte, n)
		rng.Read(a)
		rng.Read(b)
		d1 := make([]byte, 2*n)
		d2 := make([]byte, 2*n)
		ref.Interleave2U8(d1, a, b)
		simd.Interleave2(d2, a, b)
		for i := range d1 {
			if d1[i] != d2[i] {
				t.Fatalf("interleave n=%d differs at %d", n, i)
			}
		}
		a2 := make([]byte, n)
		b2 := make([]byte, n)
		simd.Deinterleave2(a2, b2, d2)
		for i := 0; i < n; i++ {
			if a2[i] != a[i] || b2[i] != b[i] {
				t.Fatalf("deinterleave round trip n=%d at %d", n, i)
			}
		}
	}
	for _, tiles := range []int{0, 1, 3, 64} {
		src := make([]byte, tiles*64)
		rng.Read(src)
		d1 := make([]byte, len(src))
		// The kernel's destination lives inside a larger canary buffer: the
		// first transpose shipped iterating bytes as tiles, writing 64x past
		// the slice while its own comparison stayed in bounds. The canary
		// turns that class of stomp into a failure here, not two tests later.
		canary := make([]byte, len(src)+4096)
		for i := range canary {
			canary[i] = 0xEE
		}
		d2 := canary[:len(src)]
		ref.Transpose8x8U8(d1, src)
		simd.Transpose8x8Bytes(d2, src)
		for i := range d1 {
			if d1[i] != d2[i] {
				t.Fatalf("transpose tiles=%d differs at %d", tiles, i)
			}
		}
		for i := len(src); i < len(canary); i++ {
			if canary[i] != 0xEE {
				t.Fatalf("tiles=%d: kernel wrote past dst at +%d", tiles, i-len(src))
			}
		}
		// Transposing twice is the identity.
		d3 := make([]byte, len(src))
		simd.Transpose8x8Bytes(d3, d2)
		for i := range src {
			if d3[i] != src[i] {
				t.Fatalf("double transpose tiles=%d at %d", tiles, i)
			}
		}
	}
}

func TestBitshuffle(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	for _, tiles := range []int{0, 1, 5, 32} {
		src := make([]byte, tiles*64)
		rng.Read(src)
		d1 := make([]byte, len(src))
		d2 := make([]byte, len(src))
		ref.BitshuffleU8(d1, src, 0)
		simd.Bitshuffle(d2, src)
		for i := range d1 {
			if d1[i] != d2[i] {
				t.Fatalf("tiles=%d shuffle differs at %d", tiles, i)
			}
		}
		back := make([]byte, len(src))
		simd.Unbitshuffle(back, d2)
		for i := range src {
			if back[i] != src[i] {
				t.Fatalf("tiles=%d round trip at %d", tiles, i)
			}
		}
		// The semantic spot check: plane 0, byte 0 is the low bits of the
		// first eight input bytes.
		if tiles > 0 {
			var want byte
			for k := 0; k < 8; k++ {
				want |= (src[k] & 1) << uint(k)
			}
			if d2[0] != want {
				t.Fatalf("plane semantics: got %08b want %08b", d2[0], want)
			}
		}
	}
}
