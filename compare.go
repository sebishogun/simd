package simd

// Comparisons, boolean vectors, selection, gather and scatter.
//
// # Comparisons produce []bool
//
// A comparison on a vector unit yields a mask, one lane per element. []bool is
// the portable spelling of that — one byte per element, which is exactly what
// storing such a mask gives you — so comparison results are ordinary Go slices
// you can index, count and combine.
//
//	mask := make([]bool, len(a))
//	simd.GreaterScalarInto(mask, a, 0)     // which elements are positive
//	n := simd.CountTrue(mask)          // how many
//	simd.SelectInto(out, mask, a, zero) // keep those, zero the rest
//
// # NaN
//
// Float comparisons follow IEEE 754 exactly, which has one consequence that
// surprises people: every comparison involving NaN is false. So for x = NaN,
// [EqualInto] reports false and [NotEqualInto] reports true, and NotEqual is therefore
// *not* the negation of Equal. That is the correct behaviour, and matching Go's
// own == and != operators, but it means [Not] applied to an Equal mask is not
// the same thing as a NotEqual mask.
//
// All of these write min of their argument lengths and allocate nothing.

// ---------- comparisons against another slice ----------

// EqualInto writes whether a[i] == b[i]. NaN compares equal to nothing, itself
// included.
func EqualInto[T Number](dst []bool, a, b []T) { ops[T]().EqualMask(dst, a, b) }

// NotEqualInto writes whether a[i] != b[i]. This is true when either side is NaN,
// so it is not the negation of [EqualInto].
func NotEqualInto[T Number](dst []bool, a, b []T) { ops[T]().NotEqualMask(dst, a, b) }

// LessInto writes whether a[i] < b[i].
func LessInto[T Number](dst []bool, a, b []T) { ops[T]().LessMask(dst, a, b) }

// LessEqualInto writes whether a[i] <= b[i].
func LessEqualInto[T Number](dst []bool, a, b []T) { ops[T]().LessEqualMask(dst, a, b) }

// GreaterInto writes whether a[i] > b[i].
func GreaterInto[T Number](dst []bool, a, b []T) { ops[T]().GreaterMask(dst, a, b) }

// GreaterEqualInto writes whether a[i] >= b[i].
func GreaterEqualInto[T Number](dst []bool, a, b []T) { ops[T]().GreaterEqualMask(dst, a, b) }

// ---------- comparisons against a scalar ----------

// EqualScalarInto writes whether a[i] == v.
func EqualScalarInto[T Number](dst []bool, a []T, v T) { ops[T]().EqualScalarMask(dst, a, v) }

// NotEqualScalarInto writes whether a[i] != v.
func NotEqualScalarInto[T Number](dst []bool, a []T, v T) { ops[T]().NotEqualScalarMask(dst, a, v) }

// LessScalarInto writes whether a[i] < v.
func LessScalarInto[T Number](dst []bool, a []T, v T) { ops[T]().LessScalarMask(dst, a, v) }

// LessEqualScalarInto writes whether a[i] <= v.
func LessEqualScalarInto[T Number](dst []bool, a []T, v T) { ops[T]().LessEqualScalarMask(dst, a, v) }

// GreaterScalarInto writes whether a[i] > v.
func GreaterScalarInto[T Number](dst []bool, a []T, v T) { ops[T]().GreaterScalarMask(dst, a, v) }

// GreaterEqualScalarInto writes whether a[i] >= v.
func GreaterEqualScalarInto[T Number](dst []bool, a []T, v T) {
	ops[T]().GreaterEqualScalarMask(dst, a, v)
}

// ---------- boolean vectors ----------

// All reports whether every element is true. An empty mask is vacuously true.
func All(m []bool) bool { return active.Mask.All(m) }

// Any reports whether any element is true. An empty mask is false.
func Any(m []bool) bool { return active.Mask.Any(m) }

// CountTrue returns how many elements are true.
func CountTrue(m []bool) int { return active.Mask.Count(m) }

// AndMask sets a[i] = a[i] && b[i], in place.
func AndMask(a, b []bool) { active.Mask.And(a, a, b) }

// OrMask sets a[i] = a[i] || b[i], in place.
func OrMask(a, b []bool) { active.Mask.Or(a, a, b) }

// XorMask sets a[i] = a[i] != b[i], in place.
func XorMask(a, b []bool) { active.Mask.Xor(a, a, b) }

// NotMask inverts every element, in place.
func NotMask(a []bool) { active.Mask.Not(a, a) }

// AndMaskInto sets dst[i] = a[i] && b[i]. dst may alias a or b.
func AndMaskInto(dst, a, b []bool) { active.Mask.And(dst, a, b) }

// OrMaskInto sets dst[i] = a[i] || b[i]. dst may alias a or b.
func OrMaskInto(dst, a, b []bool) { active.Mask.Or(dst, a, b) }

// XorMaskInto sets dst[i] = a[i] != b[i]. dst may alias a or b.
func XorMaskInto(dst, a, b []bool) { active.Mask.Xor(dst, a, b) }

// NotMaskInto sets dst[i] = !a[i]. dst may alias a.
func NotMaskInto(dst, a []bool) { active.Mask.Not(dst, a) }

// ---------- selection ----------

// SelectInto blends two slices under a mask: dst[i] = mask[i] ? yes[i] : no[i].
//
// This is the branch-free way to apply a condition across a slice, and the
// operation a vector unit calls a blend. dst may alias yes or no.
func SelectInto[T Number](dst []T, mask []bool, yes, no []T) {
	ops[T]().Select(dst, mask, yes, no)
}

// ---------- gather and scatter ----------

// GatherInto reads src at each index in idx: dst[i] = src[idx[i]].
//
// Indices outside src are skipped, leaving dst untouched at that position,
// rather than panicking. A gather is usually driven by computed indices where
// a stray value should not take the process down; check the indices yourself
// if you need strictness.
func GatherInto[T Number](dst, src []T, idx []int32) { ops[T]().Gather(dst, src, idx) }

// ScatterInto writes src to the given indices of dst: dst[idx[i]] = src[i].
//
// Indices outside dst are skipped. Where two indices collide the later write
// wins, matching what the hardware instruction does.
func ScatterInto[T Number](dst []T, idx []int32, src []T) { ops[T]().Scatter(dst, idx, src) }

// ---------- construction ----------

// Ramp fills a with an arithmetic progression: a[i] = start + i*step.
//
// Each element is computed from its own index rather than by accumulating, so
// rounding error does not build up along the slice and the whole thing
// vectorizes.
func Ramp[T Number](a []T, start, step T) { ops[T]().Ramp(a, start, step) }

// Tile fills a by repeating pattern, truncating the last copy if it does not
// fit evenly. A empty pattern leaves a unchanged.
func Tile[T Number](a, pattern []T) { ops[T]().Tile(a, pattern) }

// ---------- order statistics ----------

// Median returns the median of a, **reordering a in the process**.
//
// The reordering is what keeps this allocation-free; copy the slice first if
// you need to keep its order. For an even number of elements the two middle
// values are averaged, so the result need not be an element of the input.
// NaN sorts to the end and so does not corrupt the middle.
//
// It panics on an empty slice.
func Median[T Number](a []T) T { return ops[T]().Median(a) }

// ---------- linear algebra ----------

// MatMulInto multiplies an m*k matrix by a k*n matrix into an m*n one, all in
// row-major order. dst is zeroed first.
//
// It does nothing if the slices are too short for the stated dimensions, so
// check your sizes.
//
//	// 2x3 times 3x2 into 2x2
//	simd.MatMulInto(dst, a, b, 2, 3, 2)
//
// Each output element is one accumulator summed over the shared dimension in
// ascending order, which is what makes every instruction set here agree bit for
// bit. Zeros in a are not treated specially: a zero times an infinity is a NaN,
// as IEEE 754 says and as BLAS and numpy both produce.
func MatMulInto[T Number](dst, a, b []T, m, k, n int) { ops[T]().MatMul(dst, a, b, m, k, n) }

// GemvInto multiplies an m*k row-major matrix by a k-vector into m results:
// dst[i] = sum over p of a[i*k+p] * x[p].
//
// This is the shape most callers actually want — a matrix applied to a vector,
// once per row — and it is much cheaper than going through MatMulInto with
// n=1, which would treat the vector as a one-column matrix and give up the
// contiguous reads that make the reduction fast.
//
// Row i is bit-identical to [Dot] of that row against x. That is by
// construction rather than by coincidence, so a caller can freely mix the two.
//
// It does nothing if the slices are too short for the stated dimensions.
//
//	// a 1000x256 matrix applied to a 256-vector
//	simd.GemvInto(dst, a, x, 1000, 256)
func GemvInto[T Number](dst, a, x []T, m, k int) { ops[T]().Gemv(dst, a, x, m, k) }

// Quantile returns the q-th quantile of a, **reordering a in the process**.
//
// q is clamped to [0, 1]: Quantile(a, 0) is the minimum, 0.5 the median, 1 the
// maximum. Values between order statistics are interpolated linearly, which is
// what numpy and R type 7 do, so results are comparable with those tools.
//
// Like [Median] it reorders rather than copying, which is what keeps it
// allocation-free. Copy the slice first if you need to keep its order. For
// integer types the interpolated value truncates toward zero.
//
// It panics on an empty slice.
func Quantile[T Number](a []T, q float64) T { return ops[T]().Quantile(a, q) }
