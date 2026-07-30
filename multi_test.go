package simd_test

// The oracle is the k-loop itself: for any needle set and any haystack, the
// answer must equal "call Index on each needle, keep the earliest, break ties
// by needle order". That is a total specification, so the test is a
// randomized comparison against it rather than a list of cases.
//
// The interesting axis is the needle count, because the implementation changes
// algorithm at 8, and the byte alphabet, because the rare-byte anchor is a
// heuristic that must not change any answer when it guesses badly.

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

func multiOracle(hay string, needles []string) (int, int) {
	best, which := -1, -1
	for i, nd := range needles {
		p := strings.Index(hay, nd)
		if p < 0 {
			continue
		}
		if best < 0 || p < best || (p == best && i < which) {
			best, which = p, i
		}
	}
	return best, which
}

func TestMultiSearcher(t *testing.T) {
	t.Run("cases", func(t *testing.T) {
		cases := []struct {
			hay     string
			needles []string
		}{
			{"", nil},
			{"", []string{"a"}},
			{"abc", nil},
			{"abc", []string{""}},          // empty needle matches at 0
			{"abc", []string{"zzz", ""}},   // and wins even listed second
			{"hello world", []string{"world", "hello"}}, // earliest, not first-listed
			{"hello world", []string{"lo w", "o w"}},
			{"aaaa", []string{"aa", "aaa"}}, // same position, earlier needle wins
			{"abc", []string{"abcd"}},       // needle longer than haystack
			{"abcabcabc", []string{"cab"}},
		}
		for i, c := range cases {
			wantPos, wantWhich := multiOracle(c.hay, c.needles)
			m := simd.NewMultiSearcher(c.needles)
			gotPos, gotWhich := m.IndexString(c.hay)
			if gotPos != wantPos || gotWhich != wantWhich {
				t.Errorf("case %d %q in %q: got (%d,%d), want (%d,%d)",
					i, c.needles, c.hay, gotPos, gotWhich, wantPos, wantWhich)
			}
			if got, want := m.ContainsString(c.hay), wantPos >= 0; got != want {
				t.Errorf("case %d: Contains = %v, want %v", i, got, want)
			}
		}
	})

	// Random sets and haystacks, sweeping k across the algorithm switch at 8
	// and the alphabet across the anchor heuristic's best and worst cases.
	r := rand.New(rand.NewPCG(241, 251))
	alphabets := map[string]string{
		"narrow":   "abcdefgh",  // every needle byte is common here
		"wide":     "abcdefghijklmnopqrstuvwxyz0123456789 .,",
		"binary":   "\x00\x01\xfe\xff\x7f",
		"repeated": "aab", // heavy repetition, many false candidates
	}
	for aname, alpha := range alphabets {
		for _, k := range []int{1, 2, 7, 8, 9, 16, 64} {
			for _, hn := range []int{0, 1, 32, 1000, 20000} {
				t.Run(fmt.Sprintf("%s/k=%d/n=%d", aname, k, hn), func(t *testing.T) {
					hay := make([]byte, hn)
					for i := range hay {
						hay[i] = alpha[r.IntN(len(alpha))]
					}
					needles := make([]string, k)
					for i := range needles {
						l := 1 + r.IntN(6)
						var sb strings.Builder
						for j := 0; j < l; j++ {
							sb.WriteByte(alpha[r.IntN(len(alpha))])
						}
						needles[i] = sb.String()
					}
					wantPos, wantWhich := multiOracle(string(hay), needles)
					m := simd.NewMultiSearcher(needles)
					gotPos, gotWhich := m.Index(hay)
					if gotPos != wantPos || gotWhich != wantWhich {
						t.Fatalf("got (%d,%d), want (%d,%d)\nneedles %q\nhay %q",
							gotPos, gotWhich, wantPos, wantWhich, needles, hay)
					}
				})
			}
		}
	}

	// Needles that are substrings of each other, and needles whose rare byte
	// sits at different offsets — the case where scanning in anchor order is
	// not scanning in match order, and the reason the loop keeps going past
	// the first hit.
	t.Run("anchorOrder", func(t *testing.T) {
		// "qqqqqA" anchors late (q is rare, at offset 0 though) — build
		// explicit staggered anchors instead.
		needles := []string{
			"aaaaaz", // rare byte 'z' at offset 5
			"zaaaaa", // rare byte 'z' at offset 0
			"aaazaa", // rare byte 'z' at offset 3
			"aazaaa", "azaaaa", "aaaazb", "baaaaz", "zbaaaa", "aaaabz",
		}
		hay := "aaaaaaaazbaaaaaaaaaaaz" + strings.Repeat("a", 50) + "zaaaaa"
		wantPos, wantWhich := multiOracle(hay, needles)
		m := simd.NewMultiSearcher(needles)
		gotPos, gotWhich := m.IndexString(hay)
		if gotPos != wantPos || gotWhich != wantWhich {
			t.Fatalf("got (%d,%d), want (%d,%d)", gotPos, gotWhich, wantPos, wantWhich)
		}
	})

	// Searching does not allocate: the set is compiled once.
	t.Run("allocFree", func(t *testing.T) {
		needles := make([]string, 32)
		for i := range needles {
			needles[i] = fmt.Sprintf("keyword%02dq", i)
		}
		m := simd.NewMultiSearcher(needles)
		hay := []byte(strings.Repeat("the quick brown fox jumps over the lazy dog ", 200))
		if got := testing.AllocsPerRun(20, func() {
			sinkMultiPos, sinkMultiWhich = m.Index(hay)
		}); got != 0 {
			t.Errorf("allocated %.1f times per search, want 0", got)
		}
	})
}
