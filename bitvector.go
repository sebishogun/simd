package simd

import "math/bits"

// Rank and select over a bit vector.
//
// These are the two primitives every succinct data structure is built on — the
// wavelet trees, the FM-indexes, the compressed tries — and they are the pair
// people most often get subtly wrong, because the useful definition of rank is
// exclusive and the obvious implementation is inclusive.
//
// # What is vectorized and what is not
//
// Building the table is a population count per word followed by a prefix sum.
// The popcount half is elementwise and runs on the accelerated path through
// [OnesCountInto]. The prefix sum is a scan, and this library measured integer
// scans and found them slower than the serial loop — a one-cycle dependency is
// not one worth breaking, which is recorded on csrc/scan.c — so it runs
// serially through [CumSumInto], and that is a measured choice rather than a
// gap.
//
// So this is a composition, and it says so. What it adds over writing the two
// calls yourself is the off-by-one: [RankTableInto] wants len(v)+1 entries and
// produces an *exclusive* prefix, which is what makes [Rank] a single addition
// with no special case at a word boundary.
//
// # The table is built once and answers every query afterwards
//
// Rank is O(1) and Select is O(log n) once it exists. Neither touches the
// vector unit and neither should: a single query reads two words. All the
// vector work is in building the table, which happens once per bit vector.
//
// The bit at position p lives in v[p/64] at offset p%64, counting from the
// least significant bit — the same layout [OnesCountInto] and the rest of this
// package's bit operations use.

// RankTableInto fills dst with the exclusive prefix population count of v:
// dst[i] is the number of set bits in v[:i], so dst[0] is 0 and the last entry
// is the total.
//
// dst must have len(v)+1 entries. It panics otherwise, because a table one
// short answers queries at the end of the vector wrongly rather than visibly.
func RankTableInto(dst, v []uint64) {
	if len(dst) != len(v)+1 {
		panic("simd: RankTableInto: dst must have len(v)+1 entries")
	}
	dst[0] = 0
	OnesCountInto(dst[1:], v)
	CumSumInto(dst, dst)
}

// Rank returns the number of set bits in v at positions strictly below p, using
// a table from [RankTableInto].
//
// Exclusive, so Rank(v, t, 0) is 0 and Rank(v, t, len(v)*64) is the total. That
// is the definition [Select] inverts: Select(v, t, Rank(v, t, p)) is the first
// set bit at or after p.
//
// A p past the end returns the total rather than panicking, which is what makes
// the usual Rank(hi) - Rank(lo) idiom safe at both ends.
func Rank(v, table []uint64, p int) int {
	if p <= 0 {
		return 0
	}
	w, off := p/64, p%64
	if w >= len(v) {
		return int(table[len(v)])
	}
	if off == 0 {
		return int(table[w])
	}
	// The mask is built from off, which is in 1..63 here, so the shift is
	// always in range. Writing it as v[w]<<(64-off) instead would shift by 64
	// when off is 0 and rely on Go defining that to zero.
	return int(table[w]) + bits.OnesCount64(v[w]&(1<<uint(off)-1))
}

// Select returns the position of the k-th set bit of v, counting from zero, or
// -1 if there are fewer than k+1 set bits.
//
// Binary search over the table for the word, then a walk inside it: O(log n) on
// the words and at most 64 steps within one. There is no vectorized form
// because a query reads two words and the search is the whole cost.
func Select(v, table []uint64, k int) int {
	if k < 0 || len(v) == 0 || k >= int(table[len(v)]) {
		return -1
	}
	// The largest w with table[w] <= k. table is non-decreasing and
	// table[len(v)] > k was just checked, so the answer is a valid word index.
	lo, hi := 0, len(v)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if int(table[mid]) <= k {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	word := v[lo]
	for range k - int(table[lo]) {
		word &= word - 1 // clear the lowest set bit
	}
	return lo*64 + bits.TrailingZeros64(word)
}
