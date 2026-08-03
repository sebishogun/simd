package text

// The mask builders write one bit per byte, and a bit in the wrong place is a
// wrong answer rather than a slow one. That distinguishes them from most of
// this library, where an expression that answers "is any byte set" is enough.
//
// Every length from 0 to 40 is checked, because the interesting cases are the
// partial word at the end and the bytes just before it, and every byte value
// 0x00 to 0xff, because the ones above ASCII are where a word-at-a-time
// comparison goes wrong: they borrow.
//
// These sizes are all below the dispatch threshold, so this is the portable
// path. That is deliberate and it is not a lesser case — a small input goes
// through it entirely, and so does every caller on a target with no kernel for
// these operations.

import (
	"testing"

	"github.com/sebishogun/simd"
)

func wantMask(b []byte, match func(byte) bool) []byte {
	d := make([]byte, (len(b)+7)/8)
	for i, c := range b {
		if match(c) {
			d[i/8] |= 1 << (i % 8)
		}
	}
	return d
}

func checkMask(t *testing.T, what string, got, want []byte) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: word %d is %08b, want %08b", what, i, got[i], want[i])
		}
	}
}

func TestMaskBitsPositions(t *testing.T) {
	// Every byte value, so the bytes above ASCII are in there.
	all := make([]byte, 256)
	for i := range all {
		all[i] = byte(i)
	}
	for n := 0; n <= 40; n++ {
		for _, in := range [][]byte{all[:n], all[256-n:]} {
			dst := make([]byte, (n+7)/8+1)

			for _, c := range []byte{0, 1, 0x0e, 0x20, 0x7f, 0x80, 0xff} {
				simd.MaskBits(dst, in, c)
				checkMask(t, "MaskBits", dst, wantMask(in, func(x byte) bool { return x == c }))

				simd.MaskBitsLess(dst, in, c)
				checkMask(t, "MaskBitsLess", dst, wantMask(in, func(x byte) bool { return x < c }))
			}

			for _, set := range []string{"", "\x00", " \t\n\r", `"\`, "{}[]", "abcdefgh", "\x80\xff"} {
				simd.MaskBitsAny(dst, in, set)
				checkMask(t, "MaskBitsAny "+set, dst, wantMask(in, func(x byte) bool {
					for i := 0; i < len(set); i++ {
						if set[i] == x {
							return true
						}
					}
					return false
				}))
			}
		}
	}
}
