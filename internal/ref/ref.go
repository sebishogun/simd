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
	gomath "math"

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
			d[i] = gomath.Float32frombits(gomath.Float32bits(s[i]) &^ (1 << 31))
		}
	case []float64:
		s := any(a).([]float64)[:n]
		for i := range d {
			d[i] = gomath.Float64frombits(gomath.Float64bits(s[i]) &^ (1 << 63))
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
			d[i] = gomath.Float32frombits(gomath.Float32bits(s[i]) ^ (1 << 31))
		}
	case []float64:
		s := any(a).([]float64)[:n]
		for i := range d {
			d[i] = gomath.Float64frombits(gomath.Float64bits(s[i]) ^ (1 << 63))
		}
	}
}

// signbit reports whether v is negative, including for -0 and negative NaN.
// Converting float32 to float64 preserves the sign bit exactly, so one
// implementation covers both widths.
func signbit[T float](v T) bool { return gomath.Signbit(float64(v)) }

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
		dst[i] = T(gomath.Sqrt(float64(a[i])))
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

// addScaled is AXPY: dst[i] = a[i] + b[i]*s, in one pass over memory.
//
// The conversion around the product is not decoration and is not a no-op.
// Go's spec lets an implementation fuse a multiply and a following add into a
// single rounding, and gc takes that licence on arm64, ppc64, s390x and
// riscv64 but not on amd64 — so the obvious spelling returns different bits on
// different machines, which is the one thing this library promises it will
// not do. An explicit floating-point conversion rounds, and the spec says a
// fusion that would discard that rounding is then forbidden.
//
// The same conversion appears at every multiply that feeds an add in this
// package, for the same reason; TestNoFusedMultiplyAdd pins it down. The
// kernels are compiled with -ffp-contract=off and so never fuse either, which
// is what lets the two paths be compared bit for bit.
func addScaled[T number](dst, a, b []T, s T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] + T(b[i]*s)
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
			acc[j] += T(x[j] * y[j])
		}
	}
	for j := 0; i < n; i, j = i+1, j+1 {
		acc[j] += T(a[i] * b[i])
	}
	return kernel.CombineTree(&acc)
}

func dotInt[T integer](a, b []T) T {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var s T
	for i := range a {
		s += T(a[i] * b[i])
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

func normFloat[T float](a []T) T { return T(gomath.Sqrt(float64(sumSquaresFloat(a)))) }

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
			acc[j] += T(d * d)
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		d := a[i] - c
		acc[j] += T(d * d)
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
			acc[j] += T(d * d)
		}
	}
	for j := 0; i < n; i, j = i+1, j+1 {
		d := a[i] - b[i]
		acc[j] += T(d * d)
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
		s += T(d * d)
	}
	return s
}

func sumSqDiffInt[T integer](a, b []T) T {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var s T
	for i := range a {
		d := a[i] - b[i]
		s += T(d * d)
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

// Exported entry points for generated code.
//
// A generated kernel guards itself with a length threshold and runs the
// portable implementation below it. Reaching that implementation through the
// kernel.Set function pointer costs an indirect call the purego build never
// pays, which showed up as the accelerated build being *slower* than the
// portable one on short slices — 0.85x at n=8 — even though the assembly
// itself was faster.
//
// Calling these directly instead lets the compiler monomorphize and inline the
// loop into the guard, so the short-slice path costs no more than it does in a
// purego build. They are the same functions the kernel set is built from, so
// there is one definition of the semantics and no way for the two paths to
// disagree.
type (
	// Number is any element type the kernels handle.
	Number interface {
		~float32 | ~float64 | ~int32 | ~int64
	}
	// Float is the element types with a fixed-tree reduction.
	Float interface{ ~float32 | ~float64 }
)

func Add[T Number](dst, a, b []T) { add(dst, a, b) }
func Sub[T Number](dst, a, b []T) { sub(dst, a, b) }
func Mul[T Number](dst, a, b []T) { mul(dst, a, b) }

func Scale[T Number](dst, a []T, s T)        { scale(dst, a, s) }
func AddScaled[T Number](dst, a, b []T, s T) { addScaled(dst, a, b, s) }

func SumFloat[T Float](a []T) T    { return sumFloat(a) }
func DotFloat[T Float](a, b []T) T { return dotFloat(a, b) }

func Div[T Float](dst, a, b []T)                  { div(dst, a, b) }
func MinimumFloat[T Float](dst, a, b []T)         { minimumFloat(dst, a, b) }
func MaximumFloat[T Float](dst, a, b []T)         { maximumFloat(dst, a, b) }
func MinimumInt[T ~int32 | ~int64](dst, a, b []T) { minimumInt(dst, a, b) }
func MaximumInt[T ~int32 | ~int64](dst, a, b []T) { maximumInt(dst, a, b) }

func AbsFloat[T Float](dst, a []T)         { absFloat(dst, a) }
func NegFloat[T Float](dst, a []T)         { negFloat(dst, a) }
func AbsInt[T ~int32 | ~int64](dst, a []T) { absInt(dst, a) }
func NegInt[T ~int32 | ~int64](dst, a []T) { negInt(dst, a) }
func Sqrt[T Float](dst, a []T)             { sqrt(dst, a) }
func Reciprocal[T Float](dst, a []T)       { reciprocal(dst, a) }
func Reverse[T Number](dst, a []T)         { reverse(dst, a) }

func AddScalar[T Number](dst, a []T, s T) { addScalar(dst, a, s) }
func SubScalar[T Number](dst, a []T, s T) { subScalar(dst, a, s) }
func DivScalar[T Number](dst, a []T, s T) { divScalar(dst, a, s) }

func ClampFloat[T Float](dst, a []T, lo, hi T)         { clampFloat(dst, a, lo, hi) }
func ClampInt[T ~int32 | ~int64](dst, a []T, lo, hi T) { clampInt(dst, a, lo, hi) }
func Fill[T Number](dst []T, v T)                      { fill(dst, v) }
func Ramp[T Number](dst []T, start, step T)            { ramp(dst, start, step) }
func Lerp[T Number](dst, a, b []T, t T)                { lerp(dst, a, b, t) }

func Floor[T Float](dst, a []T)       { unary[T](gomath.Floor)(dst, a) }
func Ceil[T Float](dst, a []T)        { unary[T](gomath.Ceil)(dst, a) }
func Trunc[T Float](dst, a []T)       { unary[T](gomath.Trunc)(dst, a) }
func Round[T Float](dst, a []T)       { unary[T](gomath.Round)(dst, a) }
func RoundToEven[T Float](dst, a []T) { unary[T](gomath.RoundToEven)(dst, a) }

func MinReduceFloat[T Float](a []T) T         { return minFloat(a) }
func MaxReduceFloat[T Float](a []T) T         { return maxFloat(a) }
func MinReduceInt[T ~int32 | ~int64](a []T) T { return minInt(a) }
func MaxReduceInt[T ~int32 | ~int64](a []T) T { return maxInt(a) }

func SumSquaresFloat[T Float](a []T) T         { return sumSquaresFloat(a) }
func SumSquaresInt[T ~int32 | ~int64](a []T) T { return sumSquaresInt(a) }
func L1NormFloat[T Float](a []T) T             { return l1NormFloat(a) }
func L1DiffFloat[T Float](a, b []T) T          { return l1DiffFloat(a, b) }

func SumSqDevFloat[T Float](a []T, c T) T         { return sumSqDevFloat(a, c) }
func SumSqDevInt[T ~int32 | ~int64](a []T, c T) T { return sumSqDevInt(a, c) }
func SumSqDiffFloat[T Float](a, b []T) T          { return sumSqDiffFloat(a, b) }
func SumSqDiffInt[T ~int32 | ~int64](a, b []T) T  { return sumSqDiffInt(a, b) }

func SumInt[T ~int32 | ~int64](a []T) T    { return sumInt(a) }
func DotInt[T ~int32 | ~int64](a, b []T) T { return dotInt(a, b) }
func ProdInt[T ~int32 | ~int64](a []T) T   { return prod(a) }
func Diff[T Number](dst, a []T)            { diff(dst, a) }

func L1NormInt[T ~int32 | ~int64](a []T) T    { return l1NormInt(a) }
func L1DiffInt[T ~int32 | ~int64](a, b []T) T { return l1DiffInt(a, b) }
func CompareBytes(a, b []byte) int            { return compareBytes(a, b) }
func EqualFoldASCII(a, b []byte) bool         { return equalFoldASCII(a, b) }
func IndexAny(b, chars []byte) int            { return indexAny(b, chars) }
func CountAny(b, chars []byte) int            { return countAny(b, chars) }
func HexEncode(dst, src []byte) int           { return hexEncode(dst, src) }

func NormFloat[T Float](a []T) T { return normFloat(a) }
