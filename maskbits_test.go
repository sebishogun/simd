package simd_test

import (
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// maskBitsRef is the definition MaskBits has to meet, written the slow obvious
// way so that the fast one has something to disagree with.
func maskBitsRef(b []byte, in func(byte) bool) []byte {
	dst := make([]byte, simd.MaskLen(len(b)))
	for i, c := range b {
		if in(c) {
			dst[i/8] |= 1 << (i % 8)
		}
	}
	return dst
}

// The lengths that matter are the ones either side of a vector block: the
// kernel handles sixty-four bytes at a time and the remainder byte by byte, so
// a bug in the tail shows up at 63 and 65 and nowhere else. Lengths that are
// not a multiple of eight also check that the last byte's unused bits are
// cleared rather than left over from whatever was in the buffer.
var maskLens = []int{0, 1, 7, 8, 9, 15, 63, 64, 65, 127, 128, 129, 255, 1000, 4096, 4097}

func TestMaskBits(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 11))
	for _, n := range maskLens {
		b := make([]byte, n)
		for i := range b {
			// A small alphabet so matches are dense; a sparse one would pass
			// even if whole blocks were dropped.
			b[i] = byte("ab{}\"\\,:"[r.IntN(8)])
		}
		for _, c := range []byte{'a', '{', '"', '\\', 'z'} {
			want := maskBitsRef(b, func(x byte) bool { return x == c })
			got := make([]byte, simd.MaskLen(n))
			for i := range got {
				got[i] = 0xAA // must be overwritten, including the tail
			}
			simd.MaskBits(got, b, c)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("MaskBits(n=%d, %q): byte %d = %08b, want %08b",
						n, c, i, got[i], want[i])
				}
			}
		}
	}
}

func TestMaskBitsAny(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 5))
	sets := []string{"{", "{}", "{}[]:,", "{}[]:,\"\\", "abcdefghij"}
	for _, n := range maskLens {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte("ab{}\"\\,:[]j"[r.IntN(11)])
		}
		for _, set := range sets {
			want := maskBitsRef(b, func(x byte) bool {
				for i := 0; i < len(set); i++ {
					if set[i] == x {
						return true
					}
				}
				return false
			})
			got := make([]byte, simd.MaskLen(n))
			for i := range got {
				got[i] = 0xAA
			}
			simd.MaskBitsAny(got, b, set)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("MaskBitsAny(n=%d, %q): byte %d = %08b, want %08b",
						n, set, i, got[i], want[i])
				}
			}
		}
	}
}

// A mask and an index list are two spellings of the same answer, so they have
// to agree. This is the check that would catch a bit-order mistake, which the
// reference above shares with the kernel and so cannot catch on its own.
func TestMaskBitsAgreesWithIndexAll(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 17))
	b := make([]byte, 5000)
	for i := range b {
		b[i] = byte("ab{}\"\\,:"[r.IntN(8)])
	}
	idx := make([]int32, len(b))
	n := simd.IndexAllAny(idx, b, "{}[]:,")
	mask := make([]byte, simd.MaskLen(len(b)))
	simd.MaskBitsAny(mask, b, "{}[]:,")

	k := 0
	for i := 0; i < len(b); i++ {
		if mask[i/8]&(1<<(i%8)) == 0 {
			continue
		}
		if k >= n {
			t.Fatalf("mask has a bit at %d that IndexAllAny did not report", i)
		}
		if int(idx[k]) != i {
			t.Fatalf("bit %d of the mask, but IndexAllAny's %dth offset is %d", i, k, idx[k])
		}
		k++
	}
	if k != n {
		t.Fatalf("mask has %d bits set, IndexAllAny found %d", k, n)
	}
}

func TestMaskBitsShortDstPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("MaskBits with a short dst must panic, not write a partial answer")
		}
	}()
	simd.MaskBits(make([]byte, 1), make([]byte, 100), 'a')
}

func TestMaskBitsLess(t *testing.T) {
	r := rand.New(rand.NewPCG(29, 31))
	for _, n := range maskLens {
		b := make([]byte, n)
		for i := range b {
			// Spread across the boundary so the answer is neither all set nor
			// all clear, which a broken compare would still pass.
			b[i] = byte(r.IntN(64))
		}
		for _, c := range []byte{0, 1, 0x20, 0x40, 0xff} {
			want := maskBitsRef(b, func(x byte) bool { return x < c })
			got := make([]byte, simd.MaskLen(n))
			for i := range got {
				got[i] = 0xAA
			}
			simd.MaskBitsLess(got, b, c)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("MaskBitsLess(n=%d, %d): byte %d = %08b, want %08b",
						n, c, i, got[i], want[i])
				}
			}
		}
	}
}

// The kernel must write exactly MaskLen(n) bytes and not one more.
//
// simdjson depends on this: it allocates whole 64-bit words, zeroes the bytes
// past the document's mask, and reads the last word entire. A kernel that wrote
// a full vector's worth of output regardless of the input length would put
// nonzero bits past the end of the document, and those bits become structural
// characters that are not there.
func TestMaskBitsWritesNoFurther(t *testing.T) {
	for _, n := range maskLens {
		b := make([]byte, n)
		for i := range b {
			b[i] = 'a'
		}
		const guard = 0x5A
		for _, pad := range []int{1, 8, 64} {
			dst := make([]byte, simd.MaskLen(n)+pad)
			for i := range dst {
				dst[i] = guard
			}
			simd.MaskBits(dst, b, 'a')
			for i := simd.MaskLen(n); i < len(dst); i++ {
				if dst[i] != guard {
					t.Fatalf("MaskBits(n=%d) wrote %#02x at %d, past the %d bytes it owns",
						n, dst[i], i, simd.MaskLen(n))
				}
			}

			for i := range dst {
				dst[i] = guard
			}
			simd.MaskBitsAny(dst, b, "abc")
			for i := simd.MaskLen(n); i < len(dst); i++ {
				if dst[i] != guard {
					t.Fatalf("MaskBitsAny(n=%d) wrote %#02x at %d, past the %d bytes it owns",
						n, dst[i], i, simd.MaskLen(n))
				}
			}

			for i := range dst {
				dst[i] = guard
			}
			simd.MaskBitsLess(dst, b, 'b')
			for i := simd.MaskLen(n); i < len(dst); i++ {
				if dst[i] != guard {
					t.Fatalf("MaskBitsLess(n=%d) wrote %#02x at %d, past the %d bytes it owns",
						n, dst[i], i, simd.MaskLen(n))
				}
			}
		}
	}
}
