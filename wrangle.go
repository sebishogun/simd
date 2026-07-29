package simd

import "github.com/sebishogun/simd/internal/ref"

// Data wrangling: top-k selection and interpolation.

// TopKInto writes the k largest elements of a to dst, in descending order, and
// returns how many it wrote — min(k, len(a), len(dst)).
//
// It does not sort a, and it does not sort more of a than it has to: the
// quickselect that [Median] uses partitions around the k-th largest in linear
// time, and only the k elements to its right are then sorted. Sorting the
// whole slice to take the top ten of a million is the thing this exists to
// avoid.
//
// NaN sorts to the end, as in [Sort], so NaNs are the last thing returned and
// only when k reaches them.
//
// a is reordered. scratch is working space of at least len(a); a shorter one
// falls back to a sort rather than failing, which costs speed and not
// correctness.
func TopKInto[T Number](dst []T, a []T, k int, scratch []T) int {
	k = min(k, len(a), len(dst))
	if k <= 0 {
		return 0
	}
	if k == len(a) || len(a) < selectMinLen || len(scratch) < len(a) ||
		ops[T]().Partition == nil {
		sortFallback(a)
		for i := range k {
			dst[i] = a[len(a)-1-i]
		}
		return k
	}
	// The k largest are the elements at or after index len(a)-k once that
	// position holds its order statistic.
	selectKthInto(a, scratch, len(a)-k)
	tail := a[len(a)-k:]
	sortFallback(tail)
	for i := range k {
		dst[i] = tail[k-1-i]
	}
	return k
}

// TopK is [TopKInto] allocating both the destination and the scratch.
func TopK[T Number](a []T, k int) []T {
	k = min(k, len(a))
	if k <= 0 {
		return nil
	}
	dst := make([]T, k)
	TopKInto(dst, a, k, make([]T, len(a)))
	return dst
}

// BottomKInto is [TopKInto] for the k smallest, in ascending order.
func BottomKInto[T Number](dst []T, a []T, k int, scratch []T) int {
	k = min(k, len(a), len(dst))
	if k <= 0 {
		return 0
	}
	if k == len(a) || len(a) < selectMinLen || len(scratch) < len(a) ||
		ops[T]().Partition == nil {
		sortFallback(a)
		copy(dst[:k], a[:k])
		return k
	}
	selectKthInto(a, scratch, k-1)
	head := a[:k]
	sortFallback(head)
	copy(dst[:k], head)
	return k
}

// BottomK is [BottomKInto] allocating both the destination and the scratch.
func BottomK[T Number](a []T, k int) []T {
	k = min(k, len(a))
	if k <= 0 {
		return nil
	}
	dst := make([]T, k)
	BottomKInto(dst, a, k, make([]T, len(a)))
	return dst
}

// InterpInto writes to dst the piecewise-linear interpolation of (xp, fp)
// evaluated at each x, matching numpy's interp.
//
// xp must be increasing; the result is undefined if it is not, and that is not
// checked, because checking costs a pass over xp on every call and the caller
// almost always knows. Values of x below xp[0] give fp[0] and values above the
// last xp give the last fp, which is numpy's clamping default rather than
// extrapolation.
//
// The search for each x is a binary search over xp, which is where the time
// goes and is not vectorisable — the branch depends on the loaded value. What
// is vectorisable is the interpolation itself, and it is left as a plain
// expression because the search dominates: at 64 knots the search is six
// dependent loads against three arithmetic operations.
func InterpInto[T Float](dst []T, x, xp, fp []T) {
	n := min(len(dst), len(x))
	m := min(len(xp), len(fp))
	if m == 0 {
		return
	}
	if m == 1 {
		for i := range n {
			dst[i] = fp[0]
		}
		return
	}
	for i := range n {
		v := x[i]
		switch {
		case !(v >= xp[0]): // catches NaN too, which numpy propagates
			if v != v {
				dst[i] = v
				continue
			}
			dst[i] = fp[0]
			continue
		case v >= xp[m-1]:
			dst[i] = fp[m-1]
			continue
		}
		// Largest j with xp[j] <= v, which exists because v >= xp[0].
		lo, hi := 0, m-1
		for hi-lo > 1 {
			mid := int(uint(lo+hi) >> 1)
			if xp[mid] <= v {
				lo = mid
			} else {
				hi = mid
			}
		}
		w := xp[lo+1] - xp[lo]
		if w == 0 {
			dst[i] = fp[lo]
			continue
		}
		dst[i] = fp[lo] + (v-xp[lo])*(fp[lo+1]-fp[lo])/w
	}
}

// Interp is [InterpInto] allocating the destination.
func Interp[T Float](x, xp, fp []T) []T {
	dst := make([]T, len(x))
	InterpInto(dst, x, xp, fp)
	return dst
}

// TransposeInto writes the m*n row-major matrix a as an n*m row-major matrix
// into dst.
//
// It does nothing if either slice is too short for the stated dimensions, so
// check your sizes — the same contract [MatMulInto] has.
//
// dst must not overlap a. An in-place transpose of a non-square matrix is a
// different algorithm entirely, a permutation into cycles, and it is not what
// this is.
//
// The kernel walks the matrix in square blocks rather than row by row. Written
// the obvious way the loop reads a contiguously and writes dst with a stride
// of m, so every write lands in a different cache line and a matrix wider than
// the cache evicts each line before its next element arrives. Blocking keeps a
// block's rows and columns resident together, so each line is filled before it
// leaves.
func TransposeInto[T Number](dst, a []T, m, n int) {
	if f := ops[T]().Transpose; f != nil {
		f(dst, a, m, n)
		return
	}
	ref.Transpose(dst, a, m, n)
}

// Transpose is [TransposeInto] allocating the destination.
func Transpose[T Number](a []T, m, n int) []T {
	if m < 0 || n < 0 || len(a) < m*n {
		return nil
	}
	dst := make([]T, m*n)
	TransposeInto(dst, a, m, n)
	return dst
}
