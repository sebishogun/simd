// Package ref is the portable Go reference implementation of every kernel.
//
// It defines the semantics. Every generated backend is differential-tested
// against it, and it is also the live fallback: the dispatcher runs these
// functions on architectures with no backend, in builds made with the purego
// tag, and below the per-kernel element threshold where crossing into assembly
// costs more than it saves.
//
// It is therefore not throwaway code. The loops are written for bounds-check
// elimination because they run in the small-n hot path.
//
// The numerical contract these functions define is documented on package
// kernel. The parts that are easy to get wrong:
//
//   - Floating-point reductions use exactly kernel.SumLanes accumulators and
//     kernel.CombineTree, so every vector width reproduces them bit for bit.
//   - Dot multiplies and adds with separate roundings; it does not fuse.
//   - Minimum, Maximum, Min and Max implement IEEE-754-2019 minimum/maximum:
//     NaN propagates, and +0 compares greater than -0.
//   - Integer Abs and Neg wrap, so Abs(MinInt32) is MinInt32, matching PABSD.
package ref

import (
	"math"
	"math/bits"

	"github.com/sebishogun/simd/internal/kernel"
)

type float interface{ ~float32 | ~float64 }
type integer interface{ ~int32 | ~int64 }
type number interface{ float | integer }

// ---------- elementwise, two inputs ----------
//
// Bit-identical on every tier by construction: no reassociation is possible.

func add[T number](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func sub[T number](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mul[T number](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func div[T float](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func minimumFloat[T float](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = min2Float(a[i], b[i])
	}
}

func maximumFloat[T float](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = max2Float(a[i], b[i])
	}
}

func minimumInt[T integer](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = min(a[i], b[i])
	}
}

func maximumInt[T integer](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = max(a[i], b[i])
	}
}

// min2Float and max2Float are IEEE-754-2019 minimum/maximum: NaN propagates,
// and -0 sorts below +0. The hardware MINPS/MAXPS do neither, so kernels
// implement these with a compare and a blend. Getting this wrong is how
// accelerated and scalar paths end up disagreeing.
func min2Float[T float](x, y T) T {
	switch {
	case x != x:
		return x
	case y != y:
		return y
	case x < y:
		return x
	case y < x:
		return y
	case x == 0 && signbit(x):
		return x // -0 beats +0
	default:
		return y
	}
}

func max2Float[T float](x, y T) T {
	switch {
	case x != x:
		return x
	case y != y:
		return y
	case x > y:
		return x
	case y > x:
		return y
	case x == 0 && !signbit(x):
		return x // +0 beats -0
	default:
		return y
	}
}

// ---------- elementwise, one input ----------

// absFloat clears the sign bit, exactly as the vector instructions do (ANDPS
// against a 0x7fffffff mask). Doing it by bit manipulation rather than
// compare-and-negate keeps -0 mapping to +0 and preserves NaN payloads, so the
// reference and the kernels agree on every input.
//
// The type switch runs once per call, not per element.
func absFloat[T float](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	switch d := any(dst).(type) {
	case []float32:
		s := any(a).([]float32)[:n]
		for i := range d {
			d[i] = math.Float32frombits(math.Float32bits(s[i]) &^ (1 << 31))
		}
	case []float64:
		s := any(a).([]float64)[:n]
		for i := range d {
			d[i] = math.Float64frombits(math.Float64bits(s[i]) &^ (1 << 63))
		}
	}
}

// negFloat flips the sign bit, which unlike 0-x is correct for ±0 and NaN.
func negFloat[T float](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	switch d := any(dst).(type) {
	case []float32:
		s := any(a).([]float32)[:n]
		for i := range d {
			d[i] = math.Float32frombits(math.Float32bits(s[i]) ^ (1 << 31))
		}
	case []float64:
		s := any(a).([]float64)[:n]
		for i := range d {
			d[i] = math.Float64frombits(math.Float64bits(s[i]) ^ (1 << 63))
		}
	}
}

// signbit reports whether v is negative, including for -0 and negative NaN.
// Converting float32 to float64 preserves the sign bit exactly, so one
// implementation covers both widths.
func signbit[T float](v T) bool { return math.Signbit(float64(v)) }

// absInt wraps on the most negative value, matching PABSD/PABSQ.
func absInt[T integer](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		if v := a[i]; v < 0 {
			dst[i] = -v
		} else {
			dst[i] = v
		}
	}
}

func negInt[T integer](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = -a[i]
	}
}

func sqrt[T float](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = T(math.Sqrt(float64(a[i])))
	}
}

func reciprocal[T float](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = 1 / a[i]
	}
}

// reverse works both in place (dst and a identical) and between disjoint
// slices. Partially overlapping slices are not supported.
func reverse[T number](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
		dst[i], dst[j] = a[j], a[i]
	}
	if n%2 == 1 {
		dst[n/2] = a[n/2]
	}
}

// ---------- elementwise with a scalar ----------

func scale[T number](dst, a []T, s T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func addScalar[T number](dst, a []T, s T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = a[i] + s
	}
}

func subScalar[T number](dst, a []T, s T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = a[i] - s
	}
}

// divScalar divides rather than multiplying by a reciprocal, which costs
// throughput and buys exactness; see the note on kernel.Ops.
func divScalar[T number](dst, a []T, s T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = a[i] / s
	}
}

func clampFloat[T float](dst, a []T, lo, hi T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = min2Float(max2Float(a[i], lo), hi)
	}
}

func clampInt[T integer](dst, a []T, lo, hi T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = min(max(a[i], lo), hi)
	}
}

func fill[T number](dst []T, v T) {
	for i := range dst {
		dst[i] = v
	}
}

// addScaled is AXPY: dst[i] = a[i] + b[i]*s, in one pass over memory. It does
// not fuse the multiply and add, for the same reason Dot does not.
func addScaled[T number](dst, a, b []T, s T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] + b[i]*s
	}
}

// cumSum is a running total. It is safe in place because each output depends
// only on inputs at or before its own index.
func cumSum[T number](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	var run T
	for i := range dst {
		run += a[i]
		dst[i] = run
	}
}

// ---------- reductions ----------

// sumFloat accumulates into exactly kernel.SumLanes lanes and combines them
// with kernel.CombineTree. Element i contributes to accumulator i%SumLanes.
// Every tier reproduces this shape whatever its vector width; see package
// kernel for why.
func sumFloat[T float](a []T) T {
	var acc [kernel.SumLanes]T
	i := 0
	for ; i+kernel.SumLanes <= len(a); i += kernel.SumLanes {
		blk := a[i : i+kernel.SumLanes : i+kernel.SumLanes]
		for j := range kernel.SumLanes {
			acc[j] += blk[j]
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		acc[j] += a[i]
	}
	return kernel.CombineTree(&acc)
}

// sumInt needs no fixed tree: integer addition is associative, so no
// accumulation order is observable. It wraps on overflow, like the hardware.
func sumInt[T integer](a []T) T {
	var s T
	for _, v := range a {
		s += v
	}
	return s
}

// dotFloat multiplies and adds with separate roundings; see kernel rule 4.
func dotFloat[T float](a, b []T) T {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var acc [kernel.SumLanes]T
	i := 0
	for ; i+kernel.SumLanes <= n; i += kernel.SumLanes {
		x := a[i : i+kernel.SumLanes : i+kernel.SumLanes]
		y := b[i : i+kernel.SumLanes : i+kernel.SumLanes]
		for j := range kernel.SumLanes {
			acc[j] += x[j] * y[j]
		}
	}
	for j := 0; i < n; i, j = i+1, j+1 {
		acc[j] += a[i] * b[i]
	}
	return kernel.CombineTree(&acc)
}

func dotInt[T integer](a, b []T) T {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var s T
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func sumSquaresFloat[T float](a []T) T { return dotFloat(a, a) }

func sumSquaresInt[T integer](a []T) T { return dotInt(a, a) }

func l1NormFloat[T float](a []T) T {
	var acc [kernel.SumLanes]T
	i := 0
	for ; i+kernel.SumLanes <= len(a); i += kernel.SumLanes {
		blk := a[i : i+kernel.SumLanes : i+kernel.SumLanes]
		for j := range kernel.SumLanes {
			acc[j] += absScalar(blk[j])
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		acc[j] += absScalar(a[i])
	}
	return kernel.CombineTree(&acc)
}

func absScalar[T float](v T) T {
	if signbit(v) {
		return -v
	}
	return v
}

func l1NormInt[T integer](a []T) T {
	var s T
	for _, v := range a {
		if v < 0 {
			s -= v
		} else {
			s += v
		}
	}
	return s
}

func normFloat[T float](a []T) T { return T(math.Sqrt(float64(sumSquaresFloat(a)))) }

// sumSqDevFloat is sum((a[i]-c)^2), the second pass of a two-pass variance.
// Computing it directly rather than as SumSquares - n*mean^2 avoids the
// catastrophic cancellation that formula suffers when the variance is small
// relative to the mean.
func sumSqDevFloat[T float](a []T, c T) T {
	var acc [kernel.SumLanes]T
	i := 0
	for ; i+kernel.SumLanes <= len(a); i += kernel.SumLanes {
		blk := a[i : i+kernel.SumLanes : i+kernel.SumLanes]
		for j := range kernel.SumLanes {
			d := blk[j] - c
			acc[j] += d * d
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		d := a[i] - c
		acc[j] += d * d
	}
	return kernel.CombineTree(&acc)
}

func sumSqDiffFloat[T float](a, b []T) T {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var acc [kernel.SumLanes]T
	i := 0
	for ; i+kernel.SumLanes <= n; i += kernel.SumLanes {
		x := a[i : i+kernel.SumLanes : i+kernel.SumLanes]
		y := b[i : i+kernel.SumLanes : i+kernel.SumLanes]
		for j := range kernel.SumLanes {
			d := x[j] - y[j]
			acc[j] += d * d
		}
	}
	for j := 0; i < n; i, j = i+1, j+1 {
		d := a[i] - b[i]
		acc[j] += d * d
	}
	return kernel.CombineTree(&acc)
}

func l1DiffFloat[T float](a, b []T) T {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var acc [kernel.SumLanes]T
	i := 0
	for ; i+kernel.SumLanes <= n; i += kernel.SumLanes {
		x := a[i : i+kernel.SumLanes : i+kernel.SumLanes]
		y := b[i : i+kernel.SumLanes : i+kernel.SumLanes]
		for j := range kernel.SumLanes {
			acc[j] += absScalar(x[j] - y[j])
		}
	}
	for j := 0; i < n; i, j = i+1, j+1 {
		acc[j] += absScalar(a[i] - b[i])
	}
	return kernel.CombineTree(&acc)
}

func sumSqDevInt[T integer](a []T, c T) T {
	var s T
	for _, v := range a {
		d := v - c
		s += d * d
	}
	return s
}

func sumSqDiffInt[T integer](a, b []T) T {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var s T
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func l1DiffInt[T integer](a, b []T) T {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var s T
	for i := range a {
		if d := a[i] - b[i]; d < 0 {
			s -= d
		} else {
			s += d
		}
	}
	return s
}

func maxFloat[T float](a []T) T {
	if len(a) == 0 {
		panic("simd: Max of empty slice")
	}
	m := a[0]
	for _, v := range a[1:] {
		m = max2Float(m, v)
	}
	return m
}

func minFloat[T float](a []T) T {
	if len(a) == 0 {
		panic("simd: Min of empty slice")
	}
	m := a[0]
	for _, v := range a[1:] {
		m = min2Float(m, v)
	}
	return m
}

func maxInt[T integer](a []T) T {
	if len(a) == 0 {
		panic("simd: Max of empty slice")
	}
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func minInt[T integer](a []T) T {
	if len(a) == 0 {
		panic("simd: Min of empty slice")
	}
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func minMaxFloat[T float](a []T) (T, T) { return minFloat(a), maxFloat(a) }

func minMaxInt[T integer](a []T) (T, T) { return minInt(a), maxInt(a) }

// argMinFloat returns the index of the first minimal element. A NaN anywhere
// makes the result the index of the first NaN, matching the value semantics.
func argMinFloat[T float](a []T) int {
	if len(a) == 0 {
		panic("simd: ArgMin of empty slice")
	}
	best, bi := a[0], 0
	for i := 1; i < len(a); i++ {
		if m := min2Float(best, a[i]); m != best || (best != best) {
			if best != best {
				return bi
			}
			best, bi = a[i], i
		}
	}
	if best != best {
		return bi
	}
	return bi
}

func argMaxFloat[T float](a []T) int {
	if len(a) == 0 {
		panic("simd: ArgMax of empty slice")
	}
	best, bi := a[0], 0
	for i := 1; i < len(a); i++ {
		if best != best {
			return bi
		}
		if m := max2Float(best, a[i]); m != best {
			best, bi = a[i], i
		}
	}
	return bi
}

func argMinInt[T integer](a []T) int {
	if len(a) == 0 {
		panic("simd: ArgMin of empty slice")
	}
	bi := 0
	for i := 1; i < len(a); i++ {
		if a[i] < a[bi] {
			bi = i
		}
	}
	return bi
}

func argMaxInt[T integer](a []T) int {
	if len(a) == 0 {
		panic("simd: ArgMax of empty slice")
	}
	bi := 0
	for i := 1; i < len(a); i++ {
		if a[i] > a[bi] {
			bi = i
		}
	}
	return bi
}

// ---------- bytes and bits ----------

func indexByte(b []byte, c byte) int {
	for i := range b {
		if b[i] == c {
			return i
		}
	}
	return -1
}

func lastIndexByte(b []byte, c byte) int {
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] == c {
			return i
		}
	}
	return -1
}

func countByte(b []byte, c byte) int {
	n := 0
	for i := range b {
		if b[i] == c {
			n++
		}
	}
	return n
}

func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	b = b[:len(a)]
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// compareBytes orders lexicographically on content, then on length, matching
// bytes.Compare.
func compareBytes(a, b []byte) int {
	n := min(len(a), len(b))
	for i := range n {
		switch {
		case a[i] < b[i]:
			return -1
		case a[i] > b[i]:
			return +1
		}
	}
	switch {
	case len(a) < len(b):
		return -1
	case len(a) > len(b):
		return +1
	}
	return 0
}

func popCount(b []byte) int {
	n := 0
	for _, v := range b {
		n += bits.OnesCount8(v)
	}
	return n
}

func bitAnd(dst, a, b []byte) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] & b[i]
	}
}

func bitOr(dst, a, b []byte) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] | b[i]
	}
}

func bitXor(dst, a, b []byte) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] ^ b[i]
	}
}

func bitAndNot(dst, a, b []byte) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] &^ b[i]
	}
}

func bitNot(dst, a []byte) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = ^a[i]
	}
}

func fillBytes(dst []byte, v byte) {
	for i := range dst {
		dst[i] = v
	}
}

// ---------- assembly ----------

// floatOps builds the kernel group for a float element type.
func floatOps[T float]() kernel.Ops[T] {
	o := kernel.Ops[T]{
		Add: add[T], Sub: sub[T], Mul: mul[T], Div: div[T],
		Minimum: minimumFloat[T], Maximum: maximumFloat[T],
		Abs: absFloat[T], Neg: negFloat[T], Sqrt: sqrt[T], Reciprocal: reciprocal[T],
		Reverse: reverse[T],
		Scale:   scale[T], AddScalar: addScalar[T], SubScalar: subScalar[T], DivScalar: divScalar[T],
		Clamp: clampFloat[T], Fill: fill[T],
		AddScaled: addScaled[T],
		CumSum:    cumSum[T],
		Sum:       sumFloat[T], Min: minFloat[T], Max: maxFloat[T],
		SumSquares: sumSquaresFloat[T], L1Norm: l1NormFloat[T], Norm: normFloat[T],
		Dot:      dotFloat[T],
		SumSqDev: sumSqDevFloat[T], SumSqDiff: sumSqDiffFloat[T], L1Diff: l1DiffFloat[T],
		ArgMin: argMinFloat[T], ArgMax: argMaxFloat[T], MinMax: minMaxFloat[T],
	}
	floatMathOps(&o)
	signalOps(&o)
	compareOps(&o, medianFloat[T], lessNaNLast[T])
	return o
}

// intOps builds the kernel group for an integer element type. Div, Sqrt,
// Reciprocal and Norm stay nil; the exported API constrains those to floats.
func intOps[T integer]() kernel.Ops[T] {
	o := kernel.Ops[T]{
		Add: add[T], Sub: sub[T], Mul: mul[T],
		Minimum: minimumInt[T], Maximum: maximumInt[T],
		Abs: absInt[T], Neg: negInt[T],
		Reverse: reverse[T],
		Scale:   scale[T], AddScalar: addScalar[T], SubScalar: subScalar[T], DivScalar: divScalar[T],
		Clamp: clampInt[T], Fill: fill[T],
		AddScaled: addScaled[T],
		CumSum:    cumSum[T],
		Sum:       sumInt[T], Min: minInt[T], Max: maxInt[T],
		SumSquares: sumSquaresInt[T], L1Norm: l1NormInt[T],
		Dot:      dotInt[T],
		SumSqDev: sumSqDevInt[T], SumSqDiff: sumSqDiffInt[T], L1Diff: l1DiffInt[T],
		ArgMin: argMinInt[T], ArgMax: argMaxInt[T], MinMax: minMaxInt[T],
	}
	intMathOps(&o)
	signalOps(&o)
	compareOps(&o, medianInt[T], ltOrder[T])
	return o
}

// Set returns the reference backend: every kernel, portable Go.
func Set() kernel.Set {
	return kernel.Set{
		Name: "scalar",
		F32:  floatOps[float32](),
		F64:  floatOps[float64](),
		I32:  intOps[int32](),
		I64:  intOps[int64](),
		Bytes: kernel.Bytes{
			IndexByte: indexByte, LastIndexByte: lastIndexByte, Count: countByte,
			Equal: equalBytes, Compare: compareBytes, PopCount: popCount,
			And: bitAnd, Or: bitOr, Xor: bitXor, AndNot: bitAndNot, Not: bitNot,
			Fill: fillBytes,

			IndexAll: indexAll, IndexAny: indexAny, CountAny: countAny, Index: index,
			IsASCII: isASCII, ValidUTF8: validUTF8,
			ToUpperASCII: toUpperASCII, ToLowerASCII: toLowerASCII,
			EqualFoldASCII: equalFoldASCII, ReplaceByte: replaceByte,
			HexEncode: hexEncode, HexDecode: hexDecode,
		},
		Mask: maskOps(),
	}
}
