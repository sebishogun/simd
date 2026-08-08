package simd_test

import (
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

func TestRunLengthDecodeInt32Kernel(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for trial := 0; trial < 300; trial++ {
		n := rng.Intn(200)
		values := make([]int32, n)
		counts := make([]int32, n)
		total := 0
		for i := range values {
			values[i] = rng.Int31()
			switch rng.Intn(12) {
			case 0:
				counts[i] = int32(rng.Intn(2000)) // a long run
			case 1:
				counts[i] = -int32(rng.Intn(5)) // skipped by contract
			default:
				counts[i] = int32(rng.Intn(40))
			}
			if counts[i] > 0 {
				total += int(counts[i])
			}
		}
		// Full, short, and empty destinations: truncation is part of the
		// contract and both sides must agree on the partial output too.
		for _, cap := range []int{total, total * 3 / 4, 0} {
			dst1 := make([]int32, cap)
			dst2 := make([]int32, cap)
			g1 := ref.RLEDecodeInt32(dst1, values, counts)
			g2 := simd.RunLengthDecodeInt32(dst2, values, counts)
			if g1 != g2 {
				t.Fatalf("trial %d cap %d: ref %d kernel %d", trial, cap, g1, g2)
			}
			for i := 0; i < g1; i++ {
				if dst1[i] != dst2[i] {
					t.Fatalf("trial %d cap %d: differs at %d", trial, cap, i)
				}
			}
		}
	}
	// Round trip with the encoder.
	a := make([]int32, 5000)
	v := int32(0)
	for i := range a {
		if rng.Intn(6) == 0 {
			v = rng.Int31n(8)
		}
		a[i] = v
	}
	values := make([]int32, len(a))
	counts := make([]int32, len(a))
	scratch := make([]bool, len(a))
	runs := simd.RunLengthEncodeInt32(values, counts, a, scratch)
	values, counts = values[:runs], counts[:runs]
	dst := make([]int32, len(a))
	if n := simd.RunLengthDecodeInt32(dst, values, counts); n != len(a) {
		t.Fatalf("round trip length %d want %d", n, len(a))
	}
	for i := range a {
		if dst[i] != a[i] {
			t.Fatalf("round trip differs at %d", i)
		}
	}
}

func BenchmarkRunLengthDecodeInt32(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	values := make([]int32, 10000)
	counts := make([]int32, 10000)
	total := 0
	for i := range values {
		values[i] = rng.Int31()
		counts[i] = int32(5 + rng.Intn(60))
		total += int(counts[i])
	}
	dst := make([]int32, total)
	b.Run("kernel", func(b *testing.B) {
		b.SetBytes(int64(total * 4))
		for b.Loop() {
			sinkInt = simd.RunLengthDecodeInt32(dst, values, counts)
		}
	})
	b.Run("ref", func(b *testing.B) {
		b.SetBytes(int64(total * 4))
		for b.Loop() {
			sinkInt = ref.RLEDecodeInt32(dst, values, counts)
		}
	})
}
