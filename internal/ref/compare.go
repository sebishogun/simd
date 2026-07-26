package ref

import (
	"math"

	"github.com/sebishogun/simd/internal/kernel"
)

// Comparison, selection, gather/scatter and construction kernels.
//
// A comparison on a vector unit produces a mask register, one bit or one lane
// per element. []bool is the portable spelling of that: one byte per element,
// which is exactly what storing such a mask produces.
//
// Float comparisons follow IEEE 754 without exception, which has one
// consequence worth stating because it surprises people: every comparison
// involving NaN is false, so NotEqual is *not* the negation of Equal. For
// x = NaN, both Equal(x,x) and Less(x,x) are false, while NotEqual(x,x) is
// true. Writing these as `!equal` would be wrong, so each is written out.

func cmp2[T number](dst []bool, a, b []T, f func(x, y T) bool) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = f(a[i], b[i])
	}
}

func cmp1[T number](dst []bool, a []T, v T, f func(x, y T) bool) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = f(a[i], v)
	}
}

func eq[T number](x, y T) bool { return x == y }
func ne[T number](x, y T) bool { return x != y }
func lt[T number](x, y T) bool { return x < y }
func le[T number](x, y T) bool { return x <= y }
func gt[T number](x, y T) bool { return x > y }
func ge[T number](x, y T) bool { return x >= y }

// selectBlend is the vector blend: pick from yes or no per lane.
func selectBlend[T number](dst []T, mask []bool, yes, no []T) {
	n := min(len(dst), len(mask), len(yes), len(no))
	dst, mask, yes, no = dst[:n], mask[:n], yes[:n], no[:n]
	for i := range dst {
		if mask[i] {
			dst[i] = yes[i]
		} else {
			dst[i] = no[i]
		}
	}
}

// gather reads src at the given indices. Out-of-range indices are skipped
// rather than panicking, because a gather is usually driven by a computed
// index vector where a stray value should not take the process down.
func gather[T number](dst, src []T, idx []int32) {
	n := min(len(dst), len(idx))
	dst, idx = dst[:n], idx[:n]
	for i := range dst {
		if j := int(idx[i]); j >= 0 && j < len(src) {
			dst[i] = src[j]
		}
	}
}

// scatter writes src to the given indices of dst, skipping out-of-range ones.
// Where two indices collide the later write wins, matching the hardware.
func scatter[T number](dst []T, idx []int32, src []T) {
	n := min(len(idx), len(src))
	idx, src = idx[:n], src[:n]
	for i := range idx {
		if j := int(idx[i]); j >= 0 && j < len(dst) {
			dst[j] = src[i]
		}
	}
}

// ramp fills dst with an arithmetic progression.
//
// It multiplies rather than accumulating, so that element i is exactly
// start + i*step and rounding error does not build up along the slice. It is
// also what lets a vector unit compute a whole register at once.
func ramp[T number](dst []T, start, step T) {
	for i := range dst {
		dst[i] = start + T(T(i)*step)
	}
}

// tile repeats pattern across dst, truncating the final copy if it does not
// fit evenly.
func tile[T number](dst, pattern []T) {
	if len(pattern) == 0 {
		return
	}
	for i := range dst {
		dst[i] = pattern[i%len(pattern)]
	}
}

// selectKth partially reorders a so that a[k] holds the value that would sit
// at index k in sorted order, with everything before it no greater and
// everything after no less. It returns that value.
//
// This is quickselect, not a sort. A median needs one order statistic, not a
// total order, so this is O(n) on average where sorting is O(n log n) — and
// unlike sort.Slice it neither reflects nor allocates.
//
// The pivot is the median of the first, middle and last elements, which keeps
// already-sorted and reverse-sorted input — by far the most common shapes in
// practice — off the quadratic path.
func selectKth[T number](a []T, k int, less func(x, y T) bool) T {
	lo, hi := 0, len(a)-1
	for lo < hi {
		// partition parks its pivot at hi-1, which needs at least three
		// elements in the range. Settle one- and two-element ranges directly.
		if hi-lo < 2 {
			if less(a[hi], a[lo]) {
				a[lo], a[hi] = a[hi], a[lo]
			}
			return a[k]
		}
		p := partition(a, lo, hi, less)
		switch {
		case k == p:
			return a[k]
		case k < p:
			hi = p - 1
		default:
			lo = p + 1
		}
	}
	return a[k]
}

// partition places the median-of-three pivot at its final position and returns
// that index, using Hoare's scheme adapted to report a settled pivot.
func partition[T number](a []T, lo, hi int, less func(x, y T) bool) int {
	mid := lo + (hi-lo)/2
	// Order lo, mid, hi so the median of the three lands at mid.
	if less(a[mid], a[lo]) {
		a[lo], a[mid] = a[mid], a[lo]
	}
	if less(a[hi], a[lo]) {
		a[lo], a[hi] = a[hi], a[lo]
	}
	if less(a[hi], a[mid]) {
		a[mid], a[hi] = a[hi], a[mid]
	}
	pivot := a[mid]
	// Park the pivot at hi-1 so the loop cannot run off either end.
	a[mid], a[hi-1] = a[hi-1], a[mid]

	i, j := lo, hi-1
	for {
		i++
		for less(a[i], pivot) {
			i++
		}
		j--
		for less(pivot, a[j]) {
			j--
		}
		if i >= j {
			break
		}
		a[i], a[j] = a[j], a[i]
	}
	a[i], a[hi-1] = a[hi-1], a[i]
	return i
}

// medianFloat reorders a and returns its median.
//
// For an even count it averages the two middle elements, which is the usual
// definition and means the result need not be an element of the input.
func medianFloat[T float](a []T) T {
	n := len(a)
	if n == 0 {
		panic("simd: Median of empty slice")
	}
	if n < 3 {
		if n == 1 {
			return a[0]
		}
		lo, hi := min2Float(a[0], a[1]), max2Float(a[0], a[1])
		return (lo + hi) / 2
	}
	hi := selectKth(a, n/2, lessNaNLast[T])
	if n%2 == 1 {
		return hi
	}
	// The lower middle is the largest value left of k, which quickselect has
	// already placed there — no second pass over the whole slice.
	lo := a[0]
	for _, v := range a[1 : n/2] {
		if lessNaNLast(lo, v) {
			lo = v
		}
	}
	return (lo + hi) / 2
}

func medianInt[T integer](a []T) T {
	n := len(a)
	if n == 0 {
		panic("simd: Median of empty slice")
	}
	if n == 1 {
		return a[0]
	}
	if n == 2 {
		lo, hi := min(a[0], a[1]), max(a[0], a[1])
		return lo + (hi-lo)/2
	}
	hi := selectKth(a, n/2, ltOrder[T])
	if n%2 == 1 {
		return hi
	}
	lo := a[0]
	for _, v := range a[1 : n/2] {
		if v > lo {
			lo = v
		}
	}
	// Averaging without overflowing: halve the gap rather than the sum.
	return lo + (hi-lo)/2
}

func ltOrder[T integer](x, y T) bool { return x < y }

// quantile returns the q-th quantile of a, reordering a in the process.
//
// It interpolates linearly between the two bracketing order statistics, which
// is what numpy, R type 7 and most other tools mean by "quantile", so results
// are comparable with them. q is clamped to [0,1] rather than rejected.
//
// It needs two order statistics but only runs one quickselect: having placed
// the lower index, the next value up is by construction the smallest element
// to its right, which is one linear scan of the already-partitioned tail.
func quantile[T number](a []T, q float64, less func(x, y T) bool) T {
	n := len(a)
	if n == 0 {
		panic("simd: Quantile of empty slice")
	}
	if n == 1 {
		return a[0]
	}
	q = math.Min(math.Max(q, 0), 1)

	pos := q * float64(n-1)
	lo := int(pos)
	frac := pos - float64(lo)
	if lo >= n-1 {
		return selectKth(a, n-1, less)
	}

	x := selectKth(a, lo, less)
	if frac == 0 {
		return x
	}
	// Everything right of lo is >= x after the select, so the next order
	// statistic is the smallest of them.
	y := a[lo+1]
	for _, v := range a[lo+2:] {
		if less(v, y) {
			y = v
		}
	}
	return x + T(frac*float64(y-x))
}

// lessNaNLast gives sort a total order. NaN compares greater than everything,
// so it sorts to the end and cannot corrupt the middle of the slice.
func lessNaNLast[T float](x, y T) bool {
	if math.IsNaN(float64(x)) {
		return false
	}
	if math.IsNaN(float64(y)) {
		return true
	}
	return x < y
}

// matMul multiplies an m*k matrix by a k*n matrix, row-major.
//
// The loop order is i, then p, then j — not the textbook i, j, p. It hoists
// a[i*k+p] out of the inner loop and walks both b and dst along contiguous
// rows, so the inner loop is a scaled vector add over n elements. That is both
// cache-friendly and the shape a vector unit wants; the textbook order strides
// down a column of b and defeats both.
func matMul[T number](dst, a, b []T, m, k, n int) {
	if m <= 0 || k <= 0 || n <= 0 ||
		len(dst) < m*n || len(a) < m*k || len(b) < k*n {
		return
	}
	for i := range dst[:m*n] {
		dst[i] = 0
	}
	for i := range m {
		row := dst[i*n : (i+1)*n : (i+1)*n]
		for p := range k {
			s := a[i*k+p]
			if s == 0 {
				continue
			}
			br := b[p*n : (p+1)*n : (p+1)*n]
			for j := range row {
				row[j] += T(s * br[j])
			}
		}
	}
}

// ---------- boolean vectors ----------

func maskAll(m []bool) bool {
	for i := range m {
		if !m[i] {
			return false
		}
	}
	return true
}

func maskAny(m []bool) bool {
	for i := range m {
		if m[i] {
			return true
		}
	}
	return false
}

func maskCount(m []bool) int {
	n := 0
	for i := range m {
		if m[i] {
			n++
		}
	}
	return n
}

func maskAnd(dst, a, b []bool) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] && b[i]
	}
}

func maskOr(dst, a, b []bool) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] || b[i]
	}
}

func maskXor(dst, a, b []bool) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] != b[i]
	}
}

func maskNot(dst, a []bool) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = !a[i]
	}
}

// maskOps is the boolean-vector kernel group.
func maskOps() kernel.Mask {
	return kernel.Mask{
		All: maskAll, Any: maskAny, Count: maskCount,
		And: maskAnd, Or: maskOr, Xor: maskXor, Not: maskNot,
	}
}

// compareOps fills in the comparison, selection and construction portion of a
// kernel group. Median is supplied by the caller because its tie-breaking
// differs between floats and integers.
func compareOps[T number](o *kernel.Ops[T], median func(a []T) T, less func(x, y T) bool) {
	o.Quantile = func(a []T, q float64) T { return quantile(a, q, less) }
	o.EqualMask = func(d []bool, a, b []T) { cmp2(d, a, b, eq[T]) }
	o.NotEqualMask = func(d []bool, a, b []T) { cmp2(d, a, b, ne[T]) }
	o.LessMask = func(d []bool, a, b []T) { cmp2(d, a, b, lt[T]) }
	o.LessEqualMask = func(d []bool, a, b []T) { cmp2(d, a, b, le[T]) }
	o.GreaterMask = func(d []bool, a, b []T) { cmp2(d, a, b, gt[T]) }
	o.GreaterEqualMask = func(d []bool, a, b []T) { cmp2(d, a, b, ge[T]) }

	o.EqualScalarMask = func(d []bool, a []T, v T) { cmp1(d, a, v, eq[T]) }
	o.NotEqualScalarMask = func(d []bool, a []T, v T) { cmp1(d, a, v, ne[T]) }
	o.LessScalarMask = func(d []bool, a []T, v T) { cmp1(d, a, v, lt[T]) }
	o.LessEqualScalarMask = func(d []bool, a []T, v T) { cmp1(d, a, v, le[T]) }
	o.GreaterScalarMask = func(d []bool, a []T, v T) { cmp1(d, a, v, gt[T]) }
	o.GreaterEqualScalarMask = func(d []bool, a []T, v T) { cmp1(d, a, v, ge[T]) }

	o.Select = selectBlend[T]
	o.Gather = gather[T]
	o.Scatter = scatter[T]
	o.Ramp = ramp[T]
	o.Tile = tile[T]
	o.Median = median
	o.MatMul = matMul[T]
}

// Exported entry points for generated code; see the note on the exports in
// ref.go for why the threshold guards call these directly.

func EqualMask[T number](dst []bool, a, b []T)        { cmp2(dst, a, b, eq[T]) }
func NotEqualMask[T number](dst []bool, a, b []T)     { cmp2(dst, a, b, ne[T]) }
func LessMask[T number](dst []bool, a, b []T)         { cmp2(dst, a, b, lt[T]) }
func LessEqualMask[T number](dst []bool, a, b []T)    { cmp2(dst, a, b, le[T]) }
func GreaterMask[T number](dst []bool, a, b []T)      { cmp2(dst, a, b, gt[T]) }
func GreaterEqualMask[T number](dst []bool, a, b []T) { cmp2(dst, a, b, ge[T]) }

func EqualScalarMask[T number](dst []bool, a []T, v T)        { cmp1(dst, a, v, eq[T]) }
func NotEqualScalarMask[T number](dst []bool, a []T, v T)     { cmp1(dst, a, v, ne[T]) }
func LessScalarMask[T number](dst []bool, a []T, v T)         { cmp1(dst, a, v, lt[T]) }
func LessEqualScalarMask[T number](dst []bool, a []T, v T)    { cmp1(dst, a, v, le[T]) }
func GreaterScalarMask[T number](dst []bool, a []T, v T)      { cmp1(dst, a, v, gt[T]) }
func GreaterEqualScalarMask[T number](dst []bool, a []T, v T) { cmp1(dst, a, v, ge[T]) }

func Select[T number](dst []T, mask []bool, yes, no []T) { selectBlend(dst, mask, yes, no) }

func MaskAll(m []bool) bool    { return maskAll(m) }
func MaskAny(m []bool) bool    { return maskAny(m) }
func MaskCount(m []bool) int   { return maskCount(m) }
func MaskAnd(dst, a, b []bool) { maskAnd(dst, a, b) }
func MaskOr(dst, a, b []bool)  { maskOr(dst, a, b) }
func MaskXor(dst, a, b []bool) { maskXor(dst, a, b) }
func MaskNot(dst, a []bool)    { maskNot(dst, a) }
