package text

import (
	"math/bits"
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

func hammingRef(a, b []byte) int {
	n := min(len(a), len(b))
	c := 0
	for i := 0; i < n; i++ {
		c += bits.OnesCount8(a[i] ^ b[i])
	}
	return c
}

func TestHammingDistance(t *testing.T) {
	t.Run("known", func(t *testing.T) {
		for _, c := range []struct {
			a, b []byte
			want int
		}{
			{nil, nil, 0},
			{[]byte{0x00}, []byte{0x00}, 0},
			{[]byte{0x00}, []byte{0xFF}, 8},
			{[]byte{0xFF}, []byte{0xFF}, 0},
			{[]byte{0b1010}, []byte{0b0101}, 4},
			{[]byte{0x01, 0x02, 0x04}, []byte{0x00, 0x00, 0x00}, 3},
			// Unequal lengths take the shorter, like every other operation here.
			{[]byte{0xFF, 0xFF}, []byte{0x00}, 8},
		} {
			if got := simd.HammingDistance(c.a, c.b); got != c.want {
				t.Errorf("HammingDistance(%x, %x) = %d, want %d", c.a, c.b, got, c.want)
			}
		}
	})

	t.Run("random", func(t *testing.T) {
		r := rand.New(rand.NewPCG(71, 73))
		// Sizes either side of every vector width, and large enough that the
		// accumulator would overflow anything narrower than the count type:
		// 65536 bytes is up to 524,288 set bits.
		for _, n := range []int{0, 1, 15, 16, 17, 31, 33, 63, 65, 127, 1000, 65536} {
			a := make([]byte, n)
			b := make([]byte, n)
			for i := range a {
				a[i] = byte(r.Uint32())
				b[i] = byte(r.Uint32())
			}
			if got, want := simd.HammingDistance(a, b), hammingRef(a, b); got != want {
				t.Fatalf("n=%d: got %d want %d", n, got, want)
			}
			// All bits differing is the maximum, and the case where a signed
			// byte accumulator would have wrapped long ago.
			ones := make([]byte, n)
			zeros := make([]byte, n)
			for i := range ones {
				ones[i] = 0xFF
			}
			if got := simd.HammingDistance(ones, zeros); got != n*8 {
				t.Fatalf("n=%d all-differing: got %d want %d", n, got, n*8)
			}
		}
	})

	t.Run("words", func(t *testing.T) {
		r := rand.New(rand.NewPCG(79, 83))
		for _, n := range []int{0, 1, 3, 4, 5, 8, 9, 128, 8192} {
			a := make([]uint64, n)
			b := make([]uint64, n)
			want := 0
			for i := range a {
				a[i], b[i] = r.Uint64(), r.Uint64()
				want += bits.OnesCount64(a[i] ^ b[i])
			}
			if got := simd.HammingDistanceWords(a, b); got != want {
				t.Fatalf("n=%d: got %d want %d", n, got, want)
			}
		}
	})

	// The identity that makes this worth having as one kernel: it is exactly
	// PopCount of the Xor, which is what a caller would otherwise chain.
	t.Run("agrees with popcount of xor", func(t *testing.T) {
		r := rand.New(rand.NewPCG(89, 97))
		a := make([]byte, 4096)
		b := make([]byte, 4096)
		for i := range a {
			a[i] = byte(r.Uint32())
			b[i] = byte(r.Uint32())
		}
		x := make([]byte, len(a))
		simd.XorInto(x, a, b)
		if got, want := simd.HammingDistance(a, b), simd.PopCount(x); got != want {
			t.Errorf("HammingDistance = %d but PopCount(Xor) = %d", got, want)
		}
	})
}

func TestHammingDistanceNoAlloc(t *testing.T) {
	a := make([]byte, 8192)
	b := make([]byte, 8192)
	if n := testing.AllocsPerRun(100, func() { _ = simd.HammingDistance(a, b) }); n != 0 {
		t.Errorf("HammingDistance allocated %v times per run, want 0", n)
	}
}
