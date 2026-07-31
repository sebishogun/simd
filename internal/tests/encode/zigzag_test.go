package encode

import (
	"math"
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

// The identity zigzag exists for: a small magnitude of either sign becomes a
// small unsigned value, so a varint of it is short.
func TestZigzagKnownValues(t *testing.T) {
	in := []int32{0, -1, 1, -2, 2, -3, 3, 2147483647, -2147483648}
	want := []uint32{0, 1, 2, 3, 4, 5, 6, 4294967294, 4294967295}

	got := make([]uint32, len(in))
	simd.ZigzagEncodeInt32Into(got, in)
	for i := range in {
		if got[i] != want[i] {
			t.Errorf("ZigzagEncode(%d) = %d, want %d", in[i], got[i], want[i])
		}
	}
}

// The property that matters is bijectivity, and the value that breaks a naive
// implementation is the most negative one: it has no positive counterpart, so
// any formulation that negates it overflows.
func TestZigzagRoundTrip(t *testing.T) {
	t.Run("int32", func(t *testing.T) {
		// Sizes either side of every vector width, so the tail is exercised.
		for _, n := range []int{0, 1, 7, 15, 16, 17, 31, 33, 64, 1000} {
			r := rand.New(rand.NewPCG(1, uint64(n)))
			a := make([]int32, n)
			for i := range a {
				a[i] = int32(r.Uint32())
			}
			if n > 3 {
				a[0], a[1], a[2], a[3] = math.MinInt32, math.MaxInt32, 0, -1
			}
			enc := make([]uint32, n)
			dec := make([]int32, n)
			simd.ZigzagEncodeInt32Into(enc, a)
			simd.ZigzagDecodeInt32Into(dec, enc)
			for i := range a {
				if dec[i] != a[i] {
					t.Fatalf("n=%d i=%d: %d -> %d -> %d", n, i, a[i], enc[i], dec[i])
				}
			}
		}
	})

	t.Run("int64", func(t *testing.T) {
		for _, n := range []int{0, 1, 7, 16, 17, 33, 1000} {
			r := rand.New(rand.NewPCG(2, uint64(n)))
			a := make([]int64, n)
			for i := range a {
				a[i] = int64(r.Uint64())
			}
			if n > 3 {
				a[0], a[1], a[2], a[3] = math.MinInt64, math.MaxInt64, 0, -1
			}
			enc := make([]uint64, n)
			dec := make([]int64, n)
			simd.ZigzagEncodeInt64Into(enc, a)
			simd.ZigzagDecodeInt64Into(dec, enc)
			for i := range a {
				if dec[i] != a[i] {
					t.Fatalf("n=%d i=%d: %d -> %d -> %d", n, i, a[i], enc[i], dec[i])
				}
			}
		}
	})

	t.Run("int16", func(t *testing.T) {
		// Every int16 there is, which makes the round trip exhaustive rather
		// than sampled — 65,536 values is cheap enough to be total.
		a := make([]int16, 65536)
		for i := range a {
			a[i] = int16(i + math.MinInt16)
		}
		enc := make([]uint16, len(a))
		dec := make([]int16, len(a))
		simd.ZigzagEncodeInt16Into(enc, a)
		simd.ZigzagDecodeInt16Into(dec, enc)
		for i := range a {
			if dec[i] != a[i] {
				t.Fatalf("i=%d: %d -> %d -> %d", i, a[i], enc[i], dec[i])
			}
		}
	})

	t.Run("int8", func(t *testing.T) {
		// Exhaustive, and additionally checks that the encoding is a
		// permutation of 0..255 rather than merely reversible.
		a := make([]int8, 256)
		for i := range a {
			a[i] = int8(i + math.MinInt8)
		}
		enc := make([]byte, len(a))
		dec := make([]int8, len(a))
		simd.ZigzagEncodeInt8Into(enc, a)
		simd.ZigzagDecodeInt8Into(dec, enc)

		var seen [256]bool
		for i := range a {
			if dec[i] != a[i] {
				t.Fatalf("i=%d: %d -> %d -> %d", i, a[i], enc[i], dec[i])
			}
			if seen[enc[i]] {
				t.Fatalf("encoding is not injective: %d repeats", enc[i])
			}
			seen[enc[i]] = true
		}
	})
}

// Ordering by magnitude is the whole point, so it is worth asserting rather
// than assuming: |x| < |y| must imply encode(x) < encode(y).
func TestZigzagOrdersByMagnitude(t *testing.T) {
	a := []int32{0, -1, 1, -2, 2, -100, 100, -1000, 1000}
	enc := make([]uint32, len(a))
	simd.ZigzagEncodeInt32Into(enc, a)
	for i := 1; i < len(a); i++ {
		if enc[i] <= enc[i-1] {
			t.Errorf("encode(%d)=%d is not above encode(%d)=%d, but its magnitude is",
				a[i], enc[i], a[i-1], enc[i-1])
		}
	}
}
