package simd_test

// Differential tests for the four operations TestEveryThresholdHasCoverage
// found dispatching above any length the rest of the suite reaches.
//
//	index          threshold 256
//	countSeq       threshold 256
//	indexByte      threshold 1024
//	compareBytes   threshold 2048
//
// Below those lengths the dispatcher calls the reference, so every existing
// test of these proves something about the portable path and nothing about the
// kernel. The oracle here is the standard library, which is the right one:
// bytes.Index and bytes.IndexByte are themselves assembly, so agreeing with
// them is a real constraint rather than agreeing with a loop written to match.

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// The lengths straddle every threshold above, and continue far enough past the
// largest that the tail is exercised at several remainders modulo the vector
// width.
var textThresholdLens = []int{255, 256, 257, 1023, 1024, 1025, 2047, 2048,
	2049, 4099, 20000}

func textThresholdCorpus(n int, r *rand.Rand) []byte {
	b := make([]byte, n)
	for i := range b {
		// A small alphabet, so needles occur often and the candidate filter is
		// exercised rather than skipping to the end.
		b[i] = byte('a' + r.IntN(4))
	}
	return b
}

func TestTextAboveThreshold(t *testing.T) {
	r := rand.New(rand.NewPCG(173, 179))
	for _, n := range textThresholdLens {
		hay := textThresholdCorpus(n, r)

		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			// IndexByte and CountByte, over every byte value that occurs and
			// one that does not.
			for _, c := range []byte{'a', 'b', 'c', 'd', 'z'} {
				if got, want := simd.IndexByte(hay, c), bytes.IndexByte(hay, c); got != want {
					t.Fatalf("IndexByte(%q) = %d, want %d", c, got, want)
				}
				if got, want := simd.CountByte(hay, c), bytes.Count(hay, []byte{c}); got != want {
					t.Fatalf("CountByte(%q) = %d, want %d", c, got, want)
				}
			}

			// Index and Count with multi-byte needles: one that occurs, one
			// that does not, one at the very end, and one longer than any run.
			needles := [][]byte{
				[]byte("ab"), []byte("abc"), []byte("zzz"),
				[]byte("abcd"), hay[max(0, n-5):],
			}
			for _, nd := range needles {
				if len(nd) == 0 {
					continue
				}
				if got, want := simd.Index(hay, nd), bytes.Index(hay, nd); got != want {
					t.Fatalf("Index(%q) = %d, want %d", nd, got, want)
				}
				if got, want := simd.Count(hay, nd), bytes.Count(hay, nd); got != want {
					t.Fatalf("Count(%q) = %d, want %d", nd, got, want)
				}
			}

			// Compare, against copies differing at the start, the middle, the
			// end, and not at all — the last being the case that runs the
			// whole length and so most exercises the kernel.
			same := append([]byte(nil), hay...)
			if got := simd.Compare(hay, same); got != 0 {
				t.Fatalf("Compare(equal) = %d, want 0", got)
			}
			if got, want := simd.Equal(hay, same), true; got != want {
				t.Fatalf("Equal(equal) = %v", got)
			}
			for _, at := range []int{0, n / 2, n - 1} {
				diff := append([]byte(nil), hay...)
				diff[at] ^= 0x20
				if got, want := simd.Compare(hay, diff), bytes.Compare(hay, diff); got != want {
					t.Fatalf("Compare(differ at %d) = %d, want %d", at, got, want)
				}
				if simd.Equal(hay, diff) {
					t.Fatalf("Equal said true for slices differing at %d", at)
				}
			}
			// Different lengths, where the shorter is a prefix.
			if got, want := simd.Compare(hay, hay[:n-1]), bytes.Compare(hay, hay[:n-1]); got != want {
				t.Fatalf("Compare(prefix) = %d, want %d", got, want)
			}
		})
	}
}
