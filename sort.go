package simd

import (
	"cmp"
	"slices"

	"github.com/sebishogun/simd/internal/ref"
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
	// The nil check is not belt-and-braces. Partition is one of the slots that
	// only exists where a hardware compress does, so on most architectures the
	// backend leaves it unset and the portable implementation has to stand in.
	// Calling through it unguarded panicked on s390x — see the note in
	// CompressInto, which is the same arrangement.
	if f := ops[T]().Partition; f != nil {
		return f(dst, src, pivot)
	}
	return ref.Partition(dst, src, pivot)
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
//	n = 1024     random         6.76us   6.81us    even
//	n = 16384    random          649us    804us    24% faster
//	n = 262144   random         13.5ms   16.9ms    26% faster
//	n = 2097152  random          127ms    156ms    24% faster
//	n = 262144   few-distinct   1.10ms   1.58ms    43% faster
//	n = 2097152  few-distinct   10.1ms   13.7ms    36% faster
//	n = 16384    few-distinct   28.7us   22.9us    20% SLOWER
//
// The last row is the one case that still loses, and it is worth saying what
// changed. It used to lose by 34%, because a median-of-three pivot equal to
// much of the range sent every copy of itself to the high side and the
// recursion made no progress against them. extractEqual now takes that run out
// when the split comes back skewed, which turned the two larger few-distinct
// sizes from 24% and 17% ahead into 43% and 36%, and cut this row from 34%
// behind to 20%.
//
// What is left at 16384 is the fixed cost: three passes to detect and remove
// the equal run, against a range small enough that pdqsort's own
// duplicate-handling finishes before they pay for themselves. Making that back
// needs the extraction folded into the partition kernel itself rather than run
// as separate passes over the result.
//
// Sort sorts a in ascending order, in place.
//
// For floating-point slices NaN sorts to the end, consistently with [Median]
// and [Quantile] and with the IEEE-754-2019 ordering the rest of this package
// uses; note that this differs from [slices.Sort], which orders NaN first.
//
// # Negative zero
//
// This is the one place in the package where the accelerated and portable
// paths may produce different bits for the same input, and it is worth being
// exact about why. The order is defined by `<`, exactly as [slices.Sort] and
// [cmp.Less] define it, and under `<` negative zero and positive zero compare
// equal. Which of two equal-comparing values ends up in a given position is
// then a property of the algorithm, and the two paths run different
// algorithms — a stable out-of-place partition feeding pdqsort against
// pdqsort alone. On a 4096-element slice containing both zeros they differed
// in 848 positions.
//
// Every one of those outputs is a correct ascending sort, and every pair of
// differing elements is == to the other. Making them agree means giving the
// zeros a total order, which means a comparator function rather than a bare
// `<` — measured at 2.5x slower, for a distinction that only [math.Signbit]
// can observe. [Median] and [Quantile] inherit the same caveat.
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

// SortInto sorts a in place using scratch as working space. It avoids [Sort]'s
// unconditional element-scratch allocation. If a partition is badly skewed by
// many copies of its pivot, the duplicate-recovery path allocates a temporary
// boolean mask for each equal run it extracts.
//
// scratch must be at least as long as a. Its contents afterwards are
// unspecified. Reuse it to keep the element scratch out of a loop:
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

		// A badly skewed split means the pivot is a poor one, and with
		// duplicates it is almost always because the pivot EQUALS a large part
		// of the range: the partition splits on strictly-less-than, so every
		// copy of the pivot lands on the high side and the recursion makes no
		// progress against them.
		//
		// Rather than hand the range to pdqsort, take the equal elements out.
		// They are already in their final position once the two sides are
		// sorted, so removing them from the recursion is exact and turns the
		// case the two-way split handles worst into the one it handles best.
		lo, hi := n, len(a)-n
		if lo > hi {
			lo, hi = hi, lo
		}
		if lo == 0 || hi/lo >= sortSkewLimit {
			if eq := extractEqual(a[n:], pivot, scratch); eq > 0 {
				// a[n:n+eq] is now the run of pivots and needs no sorting.
				quicksort(a[:n], scratch, depth-1)
				quicksort(a[n+eq:], scratch, depth-1)
				return
			}
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

// extractEqual moves every element of a equal to pivot to the front of a,
// leaving the rest after them in their original relative order, and returns
// how many it moved.
//
// a must contain no element less than pivot, which is what the partition above
// guarantees for its high side. It returns 0 when the equal run is too small
// to be worth the passes, so the caller can fall back.
//
// Every step is an operation this package already accelerates — a scalar
// comparison, a mask negation, a count and a compress — so this needs no new
// kernel. It costs three passes over the high side, against the three per
// level the recursion would otherwise spend making no progress.
func extractEqual[T Number](a []T, pivot T, scratch []T) int {
	n := len(a)
	if n == 0 || len(scratch) < n {
		return 0
	}
	// The mask is bytes, so it needs its own space rather than sharing the
	// element scratch.
	mask := make([]bool, n)
	EqualScalarInto(mask, a, pivot)
	eq := CountTrue(mask)
	// Below a quarter the equal run is not what is skewing the split, and
	// three more passes would not pay.
	if eq == 0 || eq*4 < n {
		return 0
	}
	if eq == n {
		return n // all pivots; nothing to move and nothing left to sort
	}
	NotMask(mask)
	rest := scratch[:n-eq]
	if got := CompressInto(rest, a, mask); got != n-eq {
		return 0
	}
	Fill(a[:eq], pivot)
	copy(a[eq:], rest)
	return eq
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

// ---------- selection ----------

// selectCutoff is where the quickselect stops narrowing and hands the live
// window to sortFallback.
//
// It is far lower than sortCutoff because a select is a much cheaper
// recursion: it descends into one side only, so a level costs two passes over
// the window rather than three over the whole range, and the window halves
// each time. Inheriting sortCutoff's 2048 was tried first and was the wrong
// shape entirely — at 1024 elements it left a single partition followed by a
// 512-element pdqsort, and ran 2.7x slower than stopping at 256.
//
// Swept on float64, Zen 5, taking the median of nine runs:
//
//	cutoff        64        128        256       (ns)
//	n=512        777       1704        881
//	n=4096      9752      11366      10552
//	n=65536   155708     156044     156893      random
//	n=65536   428581     231201     244509      few-distinct
//	n=1M     3073876    3120471    3121807      random
//	n=1M     5064240    4875455    4893450      few-distinct
//
// On random data the three are within 8% of each other and 64 is marginally
// ahead. The decision is made by the few-distinct row at 65536, where 64 is
// 75% slower: a small window reaches the no-progress guard often, and each
// time it does the partition that discovered it was wasted. 256 is the
// smallest value that does not pay that.
const selectCutoff = 256

// selectMinLen is the length at which Median and Quantile switch to the
// accelerated path at all, as opposed to selectCutoff which is where that path
// stops recursing.
//
// Below two cutoffs the first partition already yields a window under the
// cutoff, so the whole thing degenerates to one partition followed by a sort —
// paying for the vector unit and getting a sort. It is also the point below
// which Median's own allocation stops being worth it.
const selectMinLen = 2 * selectCutoff

// selectKthInto narrows a until a[k] holds the k-th smallest value under the
// NaN-last order this package uses, writing partitions through scratch.
//
// The invariant it maintains is stronger than "a[k] is correct", and both
// callers depend on the stronger form: everything left of the live window is
// less than or equal to every element in it, and everything right is greater
// than or equal. So once the window is sorted, a[:k] are all <= a[k] and
// a[k+1:] are all >= a[k] — exactly the state a scalar quickselect leaves, and
// what lets Median find the lower middle and Quantile find the next order
// statistic with one linear scan instead of a second select.
func selectKthInto[T Number](a, scratch []T, k int) {
	lo, hi := 0, len(a)
	depth := 2 * bitsLen(len(a))
	for hi-lo >= selectCutoff && depth > 0 {
		w := a[lo:hi]
		n := PartitionInto(scratch[:len(w)], w, medianOfThree(w))
		copy(w, scratch[:len(w)])

		// No progress. The pivot is either the window's minimum or greater
		// than everything in it, which both duplicates and a NaN pivot
		// produce — medianOfThree compares with a bare `<`, so a NaN can come
		// back from it and then nothing is strictly less. One wasted pass is
		// what it costs to find out; a second would learn the same thing.
		if n == 0 || n == len(w) {
			break
		}
		if k-lo < n {
			hi = lo + n
		} else {
			lo += n
		}
		depth--
	}
	sortFallback(a[lo:hi])
}

// maxNaNLast returns the largest element of a under the NaN-last order, and
// minNaNLast the smallest. Both are spelled without constraining T to a float:
// x != x is true only for NaN and is a constant false for every integer type,
// so the integer instantiations compile down to a bare comparison.
func maxNaNLast[T Number](a []T) T {
	m := a[0]
	for _, v := range a[1:] {
		if m == m && (v != v || m < v) {
			m = v
		}
	}
	return m
}

func minNaNLast[T Number](a []T) T {
	m := a[0]
	for _, v := range a[1:] {
		if v == v && (m != m || v < m) {
			m = v
		}
	}
	return m
}

// average returns the midpoint of two values, and has to know whether T is a
// floating-point type to compute it the way the reference does.
//
// For floats that is (lo+hi)/2. For integers it overflows, so the reference
// halves the gap instead — lo + (hi-lo)/2, which cannot overflow because hi is
// never less than lo. The two are not interchangeable in the other direction
// either: halving the gap in floating point rounds twice and differs from
// (lo+hi)/2 in the last bit, and the tiers have to agree bit for bit.
//
// T(1)/T(2) is 0.5 for every float type here and 0 for every integer one, so
// the test is a constant the compiler folds away per instantiation.
func average[T Number](lo, hi T) T {
	if T(1)/T(2) != 0 {
		return (lo + hi) / 2
	}
	return lo + (hi-lo)/2
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
