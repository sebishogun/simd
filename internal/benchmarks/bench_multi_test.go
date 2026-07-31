package benchmarks

// Multi-pattern search: is one pass with a candidate filter actually better
// than k passes with Index?
//
// The task that asked for this said to measure the two-needle case before
// building anything, because "call Index twice" is a strong baseline and the
// obvious algorithms (Aho-Corasick, shift-or) carry real per-byte cost. This
// file is that measurement. The baseline is exactly what a caller writes: loop
// the needles, take the earliest hit.
//
// The variable that decides it is not k alone but how often the filter fires.
// A set of needles whose first bytes are rare gives one cheap pass; a set whose
// first bytes are common gives one pass plus a verification at every other
// position, which is worse than k clean passes.

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkMultiPos, sinkMultiWhich int

// multiCorpus builds n bytes of lowercase text plus a needle set. rare says
// whether the needles start with bytes that are uncommon in the corpus.
func multiCorpus(n, k int, rare bool) ([]byte, []string) {
	r := rand.New(rand.NewPCG(233, 239))
	hay := make([]byte, n)
	for i := range hay {
		hay[i] = byte('a' + r.IntN(8)) // only a..h occur
	}
	needles := make([]string, k)
	for i := range needles {
		if rare {
			// Starts with a byte that never occurs, so the filter never fires.
			needles[i] = string(rune('q'+i%6)) + "xyzzy"
		} else {
			// Starts with a byte that occurs one time in eight.
			needles[i] = string(rune('a'+i%8)) + "zzzzq"
		}
	}
	return hay, needles
}

// indexAnyOfLoop is the baseline: k passes, earliest wins.
func indexAnyOfLoop(hay []byte, needles []string) (int, int) {
	best, which := -1, -1
	for i, nd := range needles {
		if p := simd.Index(hay, nd); p >= 0 && (best < 0 || p < best) {
			best, which = p, i
		}
	}
	return best, which
}

// indexAnyOfFilter is the candidate-filter shape a kernel-backed version would
// have: one IndexAny pass over the set of first bytes, verify each hit.
func indexAnyOfFilter(hay []byte, needles []string, firsts []byte) (int, int) {
	at := 0
	for at < len(hay) {
		rel := simd.IndexAny(hay[at:], firsts)
		if rel < 0 {
			return -1, -1
		}
		at += rel
		for i, nd := range needles {
			if len(hay)-at >= len(nd) && string(hay[at:at+len(nd)]) == nd {
				return at, i
			}
		}
		at++
	}
	return -1, -1
}

// BenchmarkMultiPatternWorst is the case the rare-byte anchor cannot help
// with: every byte of every needle is common in the haystack, so there is no
// rare byte to pick and the filter fires constantly. This is the honest
// ceiling on the cost of choosing the one-pass algorithm, and the reason the
// k-loop is kept below eight needles rather than replaced.
func BenchmarkMultiPatternWorst(b *testing.B) {
	r := rand.New(rand.NewPCG(251, 257))
	hay := make([]byte, 1<<16)
	for i := range hay {
		hay[i] = byte('a' + r.IntN(4)) // a..d only
	}
	for _, k := range []int{16, 64} {
		needles := make([]string, k)
		for i := range needles {
			needles[i] = string([]byte{
				byte('a' + i%4), byte('a' + (i/4)%4), byte('a' + (i/16)%4),
				byte('a' + i%3), byte('a' + i%2), 'd', 'c', 'b', 'a', 'd',
			})
		}
		b.Run(fmt.Sprintf("k=%d/impl=loop", k), func(b *testing.B) {
			b.SetBytes(int64(len(hay)))
			for b.Loop() {
				sinkMultiPos, sinkMultiWhich = indexAnyOfLoop(hay, needles)
			}
		})
		m := simd.NewMultiSearcher(needles)
		b.Run(fmt.Sprintf("k=%d/impl=simd", k), func(b *testing.B) {
			b.SetBytes(int64(len(hay)))
			for b.Loop() {
				sinkMultiPos, sinkMultiWhich = m.Index(hay)
			}
		})
	}
}

func BenchmarkMultiPattern(b *testing.B) {
	for _, rare := range []bool{true, false} {
		for _, k := range []int{2, 4, 16, 64} {
			hay, needles := multiCorpus(1<<16, k, rare)
			firsts := make([]byte, 0, k)
			for _, nd := range needles {
				if !bytes.ContainsRune(firsts, rune(nd[0])) {
					firsts = append(firsts, nd[0])
				}
			}
			name := fmt.Sprintf("firstbyte=%s/k=%d", map[bool]string{true: "rare", false: "common"}[rare], k)

			b.Run(name+"/impl=loop", func(b *testing.B) {
				b.SetBytes(int64(len(hay)))
				for b.Loop() {
					sinkMultiPos, sinkMultiWhich = indexAnyOfLoop(hay, needles)
				}
			})
			b.Run(name+"/impl=filter", func(b *testing.B) {
				b.SetBytes(int64(len(hay)))
				for b.Loop() {
					sinkMultiPos, sinkMultiWhich = indexAnyOfFilter(hay, needles, firsts)
				}
			})
			// What shipped: the same one-pass shape, anchored on each
			// needle's rarest byte instead of its first.
			m := simd.NewMultiSearcher(needles)
			b.Run(name+"/impl=simd", func(b *testing.B) {
				b.SetBytes(int64(len(hay)))
				for b.Loop() {
					sinkMultiPos, sinkMultiWhich = m.Index(hay)
				}
			})
		}
	}
}
