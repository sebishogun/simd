package text

// UTF-8 validation against unicode/utf8, which defines the answer.
//
// The kernel checks each byte against the three before it, so every rule it
// implements can be broken by an input that is well-formed everywhere except
// across a block boundary. A random corpus finds the ordinary mistakes and
// misses exactly those, so the sequences are placed at every offset that puts
// them astride the boundary rather than at a sample of offsets.
//
// Two- and three-byte prefixes are enumerated rather than sampled. There are
// only 65536 of the first, and the leaders with special rules — C0, C1, E0,
// ED, F0, F4, F5 — are where an implementation is wrong rather than slow.

import (
	"bytes"
	"math/rand/v2"
	"testing"
	"unicode/utf8"

	"github.com/sebishogun/simd"
)

// blockOffsets straddle the 32-byte block the kernel works in, and the 64-byte
// one the dispatcher's threshold cares about.
var blockOffsets = []int{0, 1, 29, 30, 31, 32, 33, 34, 61, 62, 63, 64, 65, 66}

func TestValidUTF8AgainstStdlib(t *testing.T) {
	pad := bytes.Repeat([]byte("x"), 96)
	check := func(t *testing.T, b []byte) {
		t.Helper()
		if got, want := simd.ValidUTF8(b), utf8.Valid(b); got != want {
			t.Fatalf("ValidUTF8(%x) = %v, want %v", b, got, want)
		}
	}
	// Every single byte, at every offset up to two blocks, both with trailing
	// context and as the final byte — truncation is checked by a different
	// path than a bad byte in the middle.
	for c := 0; c < 256; c++ {
		seq := []byte{byte(c)}
		for off := 0; off <= 72; off++ {
			check(t, concat(pad[:off], seq, pad[:48]))
			check(t, concat(pad[:off], seq))
		}
	}
	// Every two-byte prefix. This covers the overlong, surrogate and
	// out-of-range rules, all of which are decided by a leader and the byte
	// after it.
	for a := 0; a < 256; a++ {
		for b := 0; b < 256; b++ {
			seq := []byte{byte(a), byte(b)}
			for _, off := range blockOffsets {
				check(t, concat(pad[:off], seq, pad[:48]))
				check(t, concat(pad[:off], seq))
			}
		}
	}
	// Three-byte prefixes, restricted to the leaders that have a rule of their
	// own plus the ones on either side of them.
	for _, a := range []byte{0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1, 0xEC, 0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF3, 0xF4, 0xF5, 0xFF} {
		for b := 0; b < 256; b++ {
			for c := 0; c < 256; c++ {
				seq := []byte{a, byte(b), byte(c)}
				for _, off := range []int{0, 30, 31, 32, 62, 63} {
					check(t, concat(pad[:off], seq, pad[:48]))
					check(t, concat(pad[:off], seq))
				}
			}
		}
	}
}

func TestValidUTF8Random(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 7, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 1000, 4096, 65537} {
		for trial := 0; trial < 2000; trial++ {
			b := make([]byte, n)
			switch trial % 4 {
			case 0: // uniformly random bytes: almost always invalid
				for i := range b {
					b[i] = byte(r.UintN(256))
				}
			case 1: // mostly ASCII with occasional high bytes
				for i := range b {
					if r.UintN(8) == 0 {
						b[i] = byte(0x80 + r.UintN(0x80))
					} else {
						b[i] = byte(r.UintN(0x80))
					}
				}
			case 2, 3: // well-formed text, half of it then corrupted
				var w bytes.Buffer
				for w.Len() < n {
					w.WriteRune(rune(r.UintN(0x110000)))
				}
				copy(b, w.Bytes())
				if n > 0 && trial%4 == 3 {
					b[r.IntN(n)] = byte(r.UintN(256))
				}
			}
			if got, want := simd.ValidUTF8(b), utf8.Valid(b); got != want {
				t.Fatalf("n=%d ValidUTF8(%x) = %v, want %v", n, b, got, want)
			}
		}
	}
}

func concat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}
