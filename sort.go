package simd

import (
	"cmp"
	"slices"
)

// Sorting.
//
// Only the inner half of a sort vectorizes. The recursion, the pivot choice and
// the small-case handling are control flow and belong in Go; what a vector unit
// helps with is the partition — walking a block and splitting it about a pivot,
// which the compress instruction does without a branch per element.
//
// So these functions are a quicksort written in Go around an accelerated
// partition, and they fall back to the standard library wherever that is the
// better answer. Which it often is: Go's pdqsort is good, and this package's
// rule is that a measurement decides, not a preference. See the notes on each
// function for where the line falls.

// PartitionInto splits src about a pivot: every element strictly less than the
// pivot is written to the front of dst, everything else after them, and the
// number that went to the front is returned.
//
// Both sides keep their relative order, so this is a stable partition.
//
// dst must be at least as long as src and must not overlap it. This is the
// primitive [SortInto] is built on, exposed because a partition is useful on
// its own — selecting a quantile, splitting a batch by threshold, bucketing
// before a scatter — and because it is the part that is accelerated.
//
//	n := simd.PartitionInto(dst, src, 0)
//	negatives, rest := dst[:n], dst[n:]
func PartitionInto[T Number](dst, src []T, pivot T) int {
	return ops[T]().Partition(dst, src, pivot)
}

// sortCutoff is where the quicksort stops recursing and hands the range to the
// standard library.
//
// Each level of this recursion costs three passes over the range — the
// partition reads the source twice, once per side, and the result is copied
// back out of scratch — against pdqsort's one, so the recursion has to be worth
// paying for. Raising this to 16384 was tried and made everything slower:
// stopping early removes the very levels that were doing the useful work.
const sortCutoff = 2048

// sortSkewLimit is how lopsided a split may be before the range is handed to
// pdqsort instead.
//
// This exists because of duplicates. With few distinct values, a
// median-of-three pivot is very often *equal* to a large fraction of the range,
// and since the partition splits on "strictly less than", all of those land on
// the high side. The recursion then makes almost no progress while paying three
// passes a level, and on two million elements drawn from three distinct values
// this was 37% slower than pdqsort — which has a dedicated path for exactly
// that shape.
//
// One wasted partition is an acceptable price for detecting it; continuing is
// not.
const sortSkewLimit = 16

// # Measured against slices.Sort
//
// float64, Zen 5, this against Go's pdqsort:
//
//	n = 1024     random         6.80µs   6.82µs    even
//	n = 16384    random          645µs    791µs    19% faster
//	n = 262144   random         13.3ms   16.9ms    21% faster
//	n = 2097152  random          125ms    156ms    19% faster
//	n = 262144   few-distinct   1.19ms   1.55ms    24% faster
//	n = 2097152  few-distinct   11.1ms   13.4ms    17% faster
//	n = 16384    few-distinct   34.5µs   22.8µs    34% SLOWER
//
// The last row is the one case that loses and it is worth saying why rather
// than hiding it. With few distinct values a median-of-three pivot is often
// equal to much of the range, the split comes out lopsided, and the skew guard
// hands the range to pdqsort — but only after paying for one partition. At
// larger sizes that partition earns its cost back; at 16384 it does not. The
// proper fix is a three-way partition that consumes the equal elements at each
// level instead of pushing them all to one side, which needs a second kernel.

// Sort sorts a in ascending order, in place.
//
// For floating-point slices NaN sorts to the end, consistently with [Median]
// and [Quantile] and with the IEEE-754-2019 ordering the rest of this package
// uses; note that this differs from [slices.Sort], which orders NaN first.
//
// Sort allocates. That is unusual for this package and unavoidable here: the
// accelerated partition is out of place, so it needs somewhere to write. If
// that matters, use [SortInto] and supply the scratch yourself.
func Sort[T Number](a []T) {
	if len(a) < sortCutoff || ops[T]().Partition == nil {
		sortFallback(a)
		return
	}
	SortInto(a, make([]T, len(a)))
}

// SortInto sorts a in place using scratch as working space, allocating nothing.
//
// scratch must be at least as long as a. Its contents afterwards are
// unspecified. This is the form to use in a loop or on a hot path:
//
//	scratch := make([]T, len(a))   // once
//	for _, batch := range batches {
//	    simd.SortInto(batch, scratch)
//	}
func SortInto[T Number](a, scratch []T) {
	if len(scratch) < len(a) || ops[T]().Partition == nil {
		sortFallback(a)
		return
	}
	quicksort(a, scratch, 2*bitsLen(len(a)))
}

// quicksort recurses on the smaller side and loops on the larger, which bounds
// the stack at log2(n) frames. depth counts down to a heapsort-free fallback:
// on a pathological pivot sequence it hands the range to the standard library
// rather than degrading to quadratic, which is the same protection pdqsort
// gives itself.
func quicksort[T Number](a, scratch []T, depth int) {
	for len(a) >= sortCutoff && depth > 0 {
		pivot := medianOfThree(a)
		n := PartitionInto(scratch[:len(a)], a, pivot)
		copy(a, scratch[:len(a)])

		// A badly skewed split means the pivot is a poor one — most often
		// because it equals many elements, which duplicates make likely. Hand
		// the range over rather than pay three more passes to learn the same
		// thing again.
		lo, hi := n, len(a)-n
		if lo > hi {
			lo, hi = hi, lo
		}
		if lo == 0 || hi/lo >= sortSkewLimit {
			sortFallback(a)
			return
		}

		depth--
		if n < len(a)-n {
			quicksort(a[:n], scratch, depth)
			a = a[n:]
		} else {
			quicksort(a[n:], scratch, depth)
			a = a[:n]
		}
	}
	sortFallback(a)
}

// medianOfThree picks a pivot from the first, middle and last elements, which
// is enough to make the common already-sorted and reverse-sorted inputs behave.
func medianOfThree[T Number](a []T) T {
	lo, mid, hi := a[0], a[len(a)/2], a[len(a)-1]
	if mid < lo {
		lo, mid = mid, lo
	}
	if hi < mid {
		mid = hi
	}
	if mid < lo {
		mid = lo
	}
	return mid
}

// sortFallback is the standard library's pdqsort, with NaN moved from the front
// to the end to match this package's ordering.
//
// It must be slices.Sort and not slices.SortFunc. A closure comparator is
// called once per comparison and cannot be inlined, where slices.Sort inlines a
// bare `<`; using SortFunc to express the NaN rule made sorting 1024 float64
// take 17.5µs against pdqsort's 6.9µs, and that 2.5x was mistaken for a
// weakness in the vectorized partition. It was the comparator.
//
// slices.Sort orders NaN first, because cmp.Less says so. Rotating that block
// to the end costs one pass and only when NaN is present at all — which for
// every integer type is never, and the check is a single comparison.
func sortFallback[T Number](a []T) {
	slices.Sort(a)

	// NaN is the only value not equal to itself, and after sorting they are
	// all at the front. This spelling avoids constraining T to a float type.
	n := 0
	for n < len(a) && a[n] != a[n] {
		n++
	}
	if n == 0 || n == len(a) {
		return
	}
	rotateLeft(a, n)
}

// rotateLeft moves the first k elements to the end, in place, by three
// reversals. Reverse is already an accelerated operation in this package.
func rotateLeft[T Number](a []T, k int) {
	Reverse(a[:k])
	Reverse(a[k:])
	Reverse(a)
}

func bitsLen(n int) int {
	b := 0
	for n > 0 {
		b++
		n >>= 1
	}
	return b
}

// Argsort writes into idx a permutation of 0..len(a)-1 that would sort a in
// ascending order, leaving a untouched.
//
// This is the operation Go has no good answer for: sorting one slice by the
// values of another usually means building a slice of structs or a slice of
// indices closed over the data, and both allocate and both are slow. Here idx
// is yours and nothing else is allocated.
//
// It writes min(len(idx), len(a)) entries and returns that count. Ties keep
// their original relative order, so the permutation is stable. NaN sorts to the
// end, as in [Sort].
//
// Applying the result is [GatherInto]:
//
//	n := simd.Argsort(idx, a)
//	simd.GatherInto(sorted, a, idx[:n])
func Argsort[T Number](idx []int32, a []T) int {
	n := min(len(idx), len(a))
	idx = idx[:n]
	a = a[:n]
	for i := range idx {
		idx[i] = int32(i)
	}
	slices.SortStableFunc(idx, func(x, y int32) int {
		p, q := a[x], a[y]
		switch {
		case p != p:
			if q != q {
				return 0
			}
			return 1
		case q != q:
			return -1
		}
		return cmp.Compare(p, q)
	})
	return n
}

// SortedIndex reports where v would be inserted into the sorted slice a to keep
// it sorted, and whether v is already present. It is a plain binary search,
// here because it is the natural companion to [Sort] and callers should not
// have to reach for a second package to use one with the other.
func SortedIndex[T Number](a []T, v T) (int, bool) {
	return slices.BinarySearchFunc(a, v, func(x, y T) int {
		switch {
		case x != x:
			if y != y {
				return 0
			}
			return 1
		case y != y:
			return -1
		}
		return cmp.Compare(x, y)
	})
}
