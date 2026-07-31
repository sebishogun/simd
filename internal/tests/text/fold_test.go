package text

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

// The oracle is the standard library's own composition — fold both sides,
// search. Sizes straddle Index's dispatch threshold of 256, since below it the
// underlying search runs the reference and proves nothing about the kernel.
func TestIndexFoldASCII(t *testing.T) {
	r := rand.New(rand.NewPCG(181, 191))
	for _, n := range []int{0, 1, 5, 255, 256, 257, 1000, 5000} {
		hay := make([]byte, n)
		for i := range hay {
			// Mixed-case letters, so folding matters everywhere.
			c := byte('a' + r.IntN(4))
			if r.IntN(2) == 1 {
				c -= 32
			}
			hay[i] = c
		}
		scratch := make([]byte, n)
		needles := []string{"", "ab", "AB", "aB", "abcd", "ABCD", "zzz",
			"AbCaB", strings.Repeat("aB", 40)}
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			lowHay := strings.ToLower(string(hay))
			for _, nd := range needles {
				want := strings.Index(lowHay, strings.ToLower(nd))
				if got := simd.IndexFoldASCII(hay, nd, scratch); got != want {
					t.Fatalf("IndexFoldASCII(%q) = %d, want %d", nd, got, want)
				}
				if got, want := simd.ContainsFoldASCII(hay, nd, scratch), want >= 0; got != want {
					t.Fatalf("ContainsFoldASCII(%q) = %v, want %v", nd, got, want)
				}
				wantC := bytes.Count([]byte(lowHay), []byte(strings.ToLower(nd)))
				if got := simd.CountFoldASCII(hay, nd, scratch); got != wantC {
					t.Fatalf("CountFoldASCII(%q) = %d, want %d", nd, got, wantC)
				}
			}
		})
	}

	// The offsets must be positions in the ORIGINAL haystack.
	t.Run("offsets", func(t *testing.T) {
		hay := "xxxxHeLLoxxxx"
		if got := simd.IndexFoldASCII(hay, "hello", nil); got != 4 {
			t.Fatalf("got %d, want 4", got)
		}
	})

	// Bytes 0x80 and up pass through untouched, so UTF-8 neither breaks nor
	// folds. Two directions to pin: identical non-ASCII bytes still match
	// (é == é, with the ASCII letters around them folding), and unequal
	// non-ASCII bytes never fold together (É is not é, e is not é).
	t.Run("utf8", func(t *testing.T) {
		hay := "caFÉ und CAFé"
		// "CAFé" at byte 10 matches: C/A/F fold, é is byte-identical.
		if got := simd.IndexFoldASCII(hay, "café", nil); got != 10 {
			t.Fatalf("café = %d, want 10 (CAFé, with é matching é)", got)
		}
		// Plain e must not match é in either occurrence.
		if got := simd.IndexFoldASCII(hay, "cafe", nil); got != -1 {
			t.Fatalf("plain e matched é under ASCII folding: %d", got)
		}
		// But the ASCII letters around it still fold.
		if got := simd.IndexFoldASCII(hay, "caf", nil); got != 0 {
			t.Fatalf("caf = %d, want 0", got)
		}
		if got := simd.IndexFoldASCII(hay, "UND", nil); got != 4+len("É") {
			t.Fatalf("UND = %d, want %d", got, 4+len("É"))
		}
	})

	// Needle longer than the haystack, and both empty.
	t.Run("edges", func(t *testing.T) {
		if got := simd.IndexFoldASCII("ab", "abc", nil); got != -1 {
			t.Fatalf("long needle = %d, want -1", got)
		}
		if got := simd.IndexFoldASCII("", "", nil); got != 0 {
			t.Fatalf("empty/empty = %d, want 0", got)
		}
		if got := simd.CountFoldASCII("abc", "", nil); got != 4 {
			t.Fatalf("empty-needle count = %d, want 4", got)
		}
	})

	// With scratch supplied and a short needle, no allocation.
	t.Run("allocFree", func(t *testing.T) {
		hay := bytes.Repeat([]byte("aBcD"), 1000)
		// len(hay)+len(needle), which is the documented size for zero
		// allocations — the needle's fold lives in the same scratch.
		scratch := make([]byte, len(hay)+8)
		if got := testing.AllocsPerRun(20, func() {
			sinkTextInt = simd.IndexFoldASCII(hay, "CdAb", scratch)
		}); got != 0 {
			t.Errorf("allocated %.1f times with scratch supplied, want 0", got)
		}
	})
}
