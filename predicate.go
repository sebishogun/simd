package simd

import "math"

// Predicates over floating-point slices.
//
// None of these has a kernel of its own, and none needs one. Every question
// they ask reduces to a comparison the library already vectorizes, so they are
// compositions rather than new code paths — and they are accelerated exactly
// where the comparison underneath them is.
//
// The one that carries the trick is [IsNaNInto]. NaN is the only value that is
// not equal to itself, so "is this a NaN" is `a != a` and nothing more: no
// bit-masking, no exponent test, no special handling for the payload, and it
// works for float32 and float64 through the same generic comparison.
//
// # These are measured, and the compositions win
//
// The obvious worry about building an operation out of three passes is that
// three passes over memory beat one pass of arithmetic. Measured against the
// loop a caller writes without this package, float64, median of six:
//
//	                 n=4096      n=65536     n=1048576
//	CountNaN     215ns/1149ns   3.2us/18us   81us/275us
//	NanSum       750ns/9103ns   16us/145us   515us/2335us
//	SignInto     1.9us/2.5us    45us/211us   1.40ms/4.16ms
//
// Between 1.3x and 12x ahead everywhere, and the worst case is SignInto at
// 4096, which is the one that costs the most passes. The extra memory traffic
// is real but the accelerated comparison and select are enough faster than a
// branch per element to pay for it several times over.
//
// So none of these needs a kernel, and the numbers are here so that conclusion
// does not get re-litigated by someone assuming a composition must be slow.

// IsNaNInto writes to dst whether each element of a is a NaN.
//
// It is [NotEqualInto] of a against itself, which is exactly what the IEEE
// definition of NaN says: the only value not equal to itself. Every NaN
// payload, quiet and signalling alike, answers true.
//
// dst and a must be the same length; the shorter bounds the work.
func IsNaNInto[T Float](dst []bool, a []T) { NotEqualInto(dst, a, a) }

// IsInfInto writes to dst whether each element of a is an infinity of either
// sign.
//
// The magnitude of an infinity is +Inf and the magnitude of everything else is
// finite or NaN, so one absolute value and one comparison against +Inf answers
// it. scratch is working space of at least len(a); if it is shorter this
// allocates one, so pass it on a hot path and omit it otherwise.
func IsInfInto[T Float](dst []bool, a []T, scratch []T) {
	n := min(len(dst), len(a))
	if len(scratch) < n {
		scratch = make([]T, n)
	}
	AbsInto(scratch[:n], a[:n])
	EqualScalarInto(dst[:n], scratch[:n], T(math.Inf(1)))
}

// IsFiniteInto writes to dst whether each element of a is neither infinite nor
// NaN.
//
// A NaN fails the comparison against +Inf as unordered rather than as less
// than, so this cannot be spelled as "not infinite" — the magnitude has to be
// strictly less than infinity, which excludes both.
func IsFiniteInto[T Float](dst []bool, a []T, scratch []T) {
	n := min(len(dst), len(a))
	if len(scratch) < n {
		scratch = make([]T, n)
	}
	AbsInto(scratch[:n], a[:n])
	LessScalarInto(dst[:n], scratch[:n], T(math.Inf(1)))
}

// CountNaN reports how many elements of a are NaN.
//
// mask is working space of at least len(a); a shorter one is replaced by an
// allocation, so pass one in a loop.
func CountNaN[T Float](a []T, mask []bool) int {
	if len(mask) < len(a) {
		mask = make([]bool, len(a))
	}
	IsNaNInto(mask[:len(a)], a)
	return CountTrue(mask[:len(a)])
}

// AnyNaN reports whether a contains a NaN.
//
// This is the cheap question and it has a cheaper answer than counting: a
// single reduction that stops at the first one. It still needs the mask,
// because the comparison and the reduction are separate kernels.
func AnyNaN[T Float](a []T, mask []bool) bool {
	if len(mask) < len(a) {
		mask = make([]bool, len(a))
	}
	IsNaNInto(mask[:len(a)], a)
	return Any(mask[:len(a)])
}

// NanSum returns the sum of the non-NaN elements of a, treating NaN as absent
// rather than as poison.
//
// [Sum] propagates: one NaN anywhere makes the whole answer NaN, which is what
// IEEE says and usually what you want. This is the other convention — numpy's
// nansum, R's sum(na.rm=TRUE) — for data with gaps in it.
//
// It is a select and a sum, both accelerated: the NaN lanes are replaced by
// zero and the ordinary reduction runs over the result. Adding zero is exact,
// so the answer is the sum of the surviving elements and nothing has been
// perturbed by the substitution.
//
// scratch and mask are working space of at least len(a); short ones are
// replaced by allocations. An empty slice, or one that is entirely NaN, sums
// to zero — the identity, consistently with [Sum] of an empty slice.
func NanSum[T Float](a []T, scratch []T, mask []bool) T {
	n := len(a)
	if len(scratch) < n {
		scratch = make([]T, n)
	}
	if len(mask) < n {
		mask = make([]bool, n)
	}
	scratch, mask = scratch[:n], mask[:n]
	IsNaNInto(mask, a)
	// SelectInto takes from yes where the mask is true, so the NaN lanes want
	// a zeroed slice as yes and the input as no. scratch serves as both the
	// destination and yes, which is why it is cleared first — and aliasing
	// those two is safe because the select is elementwise with no dependency
	// between lanes.
	clear(scratch)
	SelectInto(scratch, mask, scratch, a)
	return Sum(scratch)
}

// NanMean returns the mean of the non-NaN elements of a, and how many there
// were.
//
// The count is returned rather than discarded because it is the thing a caller
// needs to know whether the mean means anything: a mean over three surviving
// points out of a thousand is not a mean. A slice with no non-NaN elements
// returns NaN and zero rather than dividing by zero.
func NanMean[T Float](a []T, scratch []T, mask []bool) (T, int) {
	n := len(a)
	if len(mask) < n {
		mask = make([]bool, n)
	}
	mask = mask[:n]
	IsNaNInto(mask, a)
	k := n - CountTrue(mask)
	if k == 0 {
		return T(math.NaN()), 0
	}
	return NanSum(a, scratch, mask) / T(k), k
}

// SignInto writes the sign of each element of a to dst: -1 for a negative, +1
// for a positive, and zero for either zero.
//
// NaN propagates. sign(NaN) is NaN rather than zero, which is the rule the rest
// of this package keeps and the one numpy chose; the alternative — treating a
// NaN as unsigned and so as zero — quietly turns missing data into a real
// value. Both zeros give +0, because zero has no sign to report even when its
// bit pattern has one.
//
// scratch is working space of at least len(a) and mask of at least len(a);
// short ones are replaced by allocations.
//
// # This costs more passes than a kernel would, and still wins
//
// It is four passes over the data — two comparisons and two selects — where a
// kernel would be one. Measured against the branch-per-element loop it
// replaces: 1.9us against 2.5us at 4096, 45us against 211us at 65536, and
// 1.40ms against 4.16ms at a million. The extra traffic costs least where the
// data is largest, because both versions are then memory bound and only one of
// them branches.
//
// A single-pass kernel would still be faster and is worth writing if a
// measurement asks for it. That measurement has now been taken and it does not
// ask.
func SignInto[T Float](dst []T, a []T, scratch []T, mask []bool) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	if len(scratch) < n {
		scratch = make([]T, n)
	}
	if len(mask) < n {
		mask = make([]bool, n)
	}
	dst, a, scratch, mask = dst[:n], a[:n], scratch[:n], mask[:n]

	// Positive lanes take +1 and everything else zero, then negative lanes
	// take -1. A NaN is neither greater nor less than zero, so it survives
	// both selects as zero and the last one puts it back.
	clear(dst)
	Fill(scratch, 1)
	GreaterScalarInto(mask, a, 0)
	SelectInto(dst, mask, scratch, dst)

	Fill(scratch, -1)
	LessScalarInto(mask, a, 0)
	SelectInto(dst, mask, scratch, dst)

	IsNaNInto(mask, a)
	SelectInto(dst, mask, a, dst)
}
