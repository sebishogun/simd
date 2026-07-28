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
