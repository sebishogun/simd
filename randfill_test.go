package simd_test

import (
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

func TestRandFillU64(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 63, 64, 100, 4096} {
		a := make([]uint64, n)
		b := make([]uint64, n)
		simd.RandFillU64(a, 42)
		ref.RandFillU64(b, 42)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("n=%d differs at %d: %x vs %x", n, i, a[i], b[i])
			}
		}
	}
	// Same seed, same slice; different seed, different slice; and a crude
	// bit balance says the stream is not degenerate.
	a := make([]uint64, 1<<16)
	b := make([]uint64, 1<<16)
	simd.RandFillU64(a, 7)
	simd.RandFillU64(b, 7)
	ones := 0
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("same seed disagrees")
		}
		for x := a[i]; x != 0; x &= x - 1 {
			ones++
		}
	}
	mean := float64(ones) / float64(len(a)*64)
	if mean < 0.49 || mean > 0.51 {
		t.Fatalf("bit balance %f", mean)
	}
	simd.RandFillU64(b, 8)
	if a[0] == b[0] && a[1] == b[1] && a[2] == b[2] {
		t.Fatal("different seeds agree suspiciously")
	}
}

func BenchmarkRandFillU64(b *testing.B) {
	dst := make([]uint64, 1<<16)
	b.Run("kernel", func(b *testing.B) {
		b.SetBytes(int64(len(dst) * 8))
		for b.Loop() {
			simd.RandFillU64(dst, 1)
		}
	})
	b.Run("mathrand", func(b *testing.B) {
		b.SetBytes(int64(len(dst) * 8))
		rng := rand.New(rand.NewSource(1))
		for b.Loop() {
			for i := range dst {
				dst[i] = rng.Uint64()
			}
		}
	})
}

func TestHashUint64(t *testing.T) {
	keys := make([]uint64, 1000)
	for i := range keys {
		keys[i] = uint64(i) * 0x123456789
	}
	d1 := make([]uint64, len(keys))
	d2 := make([]uint64, len(keys))
	ref.HashU64(d1, keys, 7)
	simd.HashUint64(d2, keys, 7)
	seen := map[uint64]bool{}
	for i := range d1 {
		if d1[i] != d2[i] {
			t.Fatalf("[%d] ref %x kernel %x", i, d1[i], d2[i])
		}
		if seen[d1[i]] {
			t.Fatalf("collision at %d for sequential keys", i)
		}
		seen[d1[i]] = true
	}
	simd.HashUint64(d2, keys, 8)
	if d1[0] == d2[0] && d1[1] == d2[1] {
		t.Fatal("seed does not move the hashes")
	}
}

func BenchmarkHashUint64(b *testing.B) {
	keys := make([]uint64, 1<<16)
	for i := range keys {
		keys[i] = uint64(i)
	}
	dst := make([]uint64, len(keys))
	b.Run("kernel", func(b *testing.B) {
		b.SetBytes(int64(len(keys) * 8))
		for b.Loop() {
			simd.HashUint64(dst, keys, 1)
		}
	})
	b.Run("goloop", func(b *testing.B) {
		b.SetBytes(int64(len(keys) * 8))
		for b.Loop() {
			ref.HashU64(dst, keys, 1)
		}
	})
}
