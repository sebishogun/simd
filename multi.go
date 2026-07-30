package simd

// Multi-pattern search.
//
// # Why the obvious filter is unshippable
//
// The natural way to search for several needles at once is to scan for the set
// of their first bytes with [IndexAny] and verify each hit. Measured against
// the baseline — call [Index] once per needle and keep the earliest — over a
// 64 KiB haystack:
//
//	needles start with a byte that is rare in the haystack
//	    k=2    filter 1.3x slower      k=16   filter 3.4x faster
//	    k=4    filter 1.1x slower      k=64   filter 11.8x faster
//
//	needles start with a byte that is common in the haystack
//	    k=2    filter 252x SLOWER      k=16   filter 136x slower
//	    k=4    filter 248x slower      k=64   filter 239x slower
//
// One pass beats k passes only when the pass is actually one pass. When the
// filter fires every few bytes the scan restarts constantly and every hit
// costs a comparison against every needle, which is quadratic in disguise. A
// caller cannot predict which case they are in, and being 250x slower on data
// you did not choose is not a tradeoff worth offering.
//
// # What this does instead
//
// Anchor on each needle's *rarest* byte rather than its first. "azzzzq" in a
// haystack of a..h is hopeless anchored at 'a' and free anchored at 'z'. The
// scan is still one [IndexAny] pass, over the set of anchor bytes, and the
// verification is now against only the needles whose anchor that byte is.
//
// Rarity comes from a fixed table, so compiling a needle set costs nothing and
// two runs on the same input behave the same. The table is a heuristic and is
// allowed to be wrong: it changes which byte is scanned for, never which
// matches are found. A pathological set on a pathological haystack degrades
// toward the filter numbers above, which is why [MultiSearcher.Index] keeps
// the k-loop for small sets, where it is faster anyway.

import "sort"

// byteRank orders bytes from most common to least common in the kind of data
// people search: text, source, logs and protocol dumps. Lower is more common.
//
// It only decides which byte the scan anchors on, so an inaccuracy costs speed
// and never correctness. The shape is what matters, not the exact ordering:
// space and lowercase letters are everywhere, digits and punctuation are
// frequent, uppercase less so, control bytes and the high half are rare —
// except NUL and 0xFF, which are common in binary and are demoted accordingly.
var byteRank = func() [256]uint8 {
	// Start with everything rare, then promote the common classes.
	var freq [256]int
	for i := range freq {
		freq[i] = 1
	}
	for _, c := range []byte(" \n\t") {
		freq[c] = 1000
	}
	for c := byte('a'); c <= 'z'; c++ {
		freq[c] = 500
	}
	for c := byte('0'); c <= '9'; c++ {
		freq[c] = 300
	}
	for _, c := range []byte(",.:;/-_=\"'()[]{}<>") {
		freq[c] = 250
	}
	for c := byte('A'); c <= 'Z'; c++ {
		freq[c] = 200
	}
	// Common in binary and structured data, so not actually rare.
	freq[0x00] = 400
	freq[0xff] = 150
	// A few letters that are genuinely uncommon even in English.
	for _, c := range []byte("jqxzJQXZ") {
		freq[c] = 40
	}

	idx := make([]int, 256)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return freq[idx[a]] > freq[idx[b]] })
	var rank [256]uint8
	for r, c := range idx {
		rank[c] = uint8(r)
	}
	return rank
}()

// multiLoopMax is the needle count below which the k-loop is kept.
//
// Measured: at k=2 and k=4 the loop is 1.1x to 1.3x ahead even in the case
// that most favours a filter, because k passes of a tuned [Index] are cheaper
// than one pass plus per-hit dispatch. The crossover is between 4 and 16, and
// 8 is the midpoint that never loses by much on either side.
const multiLoopMax = 8

// MultiSearcher searches a haystack for any of a fixed set of needles.
//
// Compile the set once and search many haystacks with it:
//
//	m := simd.NewMultiSearcher(keywords)
//	for _, line := range lines {
//	    if pos, which := m.Index(line); pos >= 0 {
//	        ...
//	    }
//	}
//
// A MultiSearcher is immutable after construction and safe for concurrent use.
type MultiSearcher struct {
	needles []string
	// anchors[i] is the offset within needles[i] of the byte the scan looks
	// for, chosen as the rarest by byteRank.
	anchors []int
	// set is the distinct anchor bytes, and byAnchor[c] lists the needle
	// indices anchored on byte c.
	set      []byte
	byAnchor [256][]int32
	// maxAnchor is the largest anchor offset, and bounds how far past a
	// candidate the scan must continue before the earliest match is settled.
	maxAnchor int
	// minLen is the shortest needle, used only to reject hopeless haystacks.
	minLen int
	// empty records that some needle is empty, which matches at 0.
	empty int
}

// NewMultiSearcher compiles a needle set. An empty needle matches at position
// 0, as [Index] does; duplicate needles are kept, and the earliest-listed one
// wins a tie.
func NewMultiSearcher[T Text](needles []T) *MultiSearcher {
	m := &MultiSearcher{
		needles: make([]string, len(needles)),
		anchors: make([]int, len(needles)),
		empty:   -1,
		minLen:  -1,
	}
	for i, nd := range needles {
		s := string(textBytes(nd))
		m.needles[i] = s
		if s == "" {
			if m.empty < 0 {
				m.empty = i
			}
			continue
		}
		if m.minLen < 0 || len(s) < m.minLen {
			m.minLen = len(s)
		}
		// The rarest byte, earliest one winning so the anchor is as close to
		// the start as the ranking allows.
		best := 0
		for j := 1; j < len(s); j++ {
			if byteRank[s[j]] > byteRank[s[best]] {
				best = j
			}
		}
		m.anchors[i] = best
		if best > m.maxAnchor {
			m.maxAnchor = best
		}
		c := s[best]
		if len(m.byAnchor[c]) == 0 {
			m.set = append(m.set, c)
		}
		m.byAnchor[c] = append(m.byAnchor[c], int32(i))
	}
	return m
}

// Index returns the position of the earliest match of any needle in haystack
// and which needle matched, or (-1, -1) if none does.
//
// Earliest is by position first and by needle order second, so the result does
// not depend on which needle the scan happened to find first.
//
// A method cannot take a type parameter in Go, so unlike the free functions in
// this package these come in []byte and string pairs, as the standard
// library's own bytes and strings packages do.
func (m *MultiSearcher) Index(haystack []byte) (pos, which int) {
	return m.index(haystack)
}

// IndexString is [MultiSearcher.Index] over a string, without copying it.
func (m *MultiSearcher) IndexString(haystack string) (pos, which int) {
	return m.index(textBytes(haystack))
}

func (m *MultiSearcher) index(h []byte) (pos, which int) {
	if m.empty >= 0 {
		return 0, m.empty
	}
	if len(m.needles) == 0 || m.minLen < 0 || len(h) < m.minLen {
		return -1, -1
	}
	if len(m.needles) <= multiLoopMax {
		return m.indexLoop(h)
	}

	best, bestWhich := -1, -1
	at := 0
	for at < len(h) {
		rel := active.Bytes.IndexAny(h[at:], m.set)
		if rel < 0 {
			break
		}
		at += rel
		for _, ni := range m.byAnchor[h[at]] {
			i := int(ni)
			start := at - m.anchors[i]
			if start < 0 || start+len(m.needles[i]) > len(h) {
				continue
			}
			if string(h[start:start+len(m.needles[i])]) != m.needles[i] {
				continue
			}
			// Position first, needle order second.
			if best < 0 || start < best || (start == best && i < bestWhich) {
				best, bestWhich = start, i
			}
		}
		at++
		// A match found at start `best` was anchored somewhere at or after it.
		// An earlier-starting match can only be anchored before best+maxAnchor,
		// so once the scan passes that point nothing earlier remains.
		if best >= 0 && at > best+m.maxAnchor {
			break
		}
	}
	return best, bestWhich
}

// indexLoop is the k-pass baseline, which wins for small sets.
func (m *MultiSearcher) indexLoop(h []byte) (int, int) {
	best, bestWhich := -1, -1
	for i, nd := range m.needles {
		p := Index(h, nd)
		if p < 0 {
			continue
		}
		if best < 0 || p < best || (p == best && i < bestWhich) {
			best, bestWhich = p, i
		}
	}
	return best, bestWhich
}

// Contains reports whether any needle occurs in haystack.
func (m *MultiSearcher) Contains(haystack []byte) bool {
	pos, _ := m.index(haystack)
	return pos >= 0
}

// ContainsString is [MultiSearcher.Contains] over a string.
func (m *MultiSearcher) ContainsString(haystack string) bool {
	pos, _ := m.index(textBytes(haystack))
	return pos >= 0
}
