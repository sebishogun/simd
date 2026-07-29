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
	"bytes"
	gomath "math"
	"unsafe"

	"math/bits"

	"github.com/sebishogun/simd/internal/kernel"
)

type float interface{ ~float32 | ~float64 }
type integer interface {
	~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64
}

// satInteger is the subset with a wider type to detect overflow in, which is
// what the saturating operations need. The 64-bit types are excluded because
// there is nothing wider to widen into.
type satInteger interface {
	~int8 | ~int16 | ~int32 | ~uint8 | ~uint16 | ~uint32
}
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

// shl, shr, rotl and rotr define the shift contract the kernels must match.
//
// Go's own operators already do the right thing for counts at or above the
// width — zero for a left shift and for an unsigned right shift, and the sign
// for a signed one — so the reference is simply the operator, and it is the C
// side that has to be taught not to invoke undefined behaviour.
func shl[T integer](dst, a []T, s uint64) {
	n := min(len(dst), len(a))
	for i := range n {
		dst[i] = a[i] << s
	}
}

func shr[T integer](dst, a []T, s uint64) {
	n := min(len(dst), len(a))
	for i := range n {
		dst[i] = a[i] >> s
	}
}

// bitsOf is the element width, from the type rather than from a parameter.
func bitsOf[T integer]() uint64 { var z T; return uint64(unsafe.Sizeof(z)) * 8 }

func rotl[T integer](dst, a []T, s uint64) {
	w := bitsOf[T]()
	s %= w
	n := min(len(dst), len(a))
	for i := range n {
		if s == 0 {
			dst[i] = a[i]
			continue
		}
		// The shifts are done on the unsigned view so the right half brings in
		// zeros rather than sign bits; T(x) narrows it back.
		u := uint64(a[i]) & (1<<w - 1)
		dst[i] = T(u<<s | u>>(w-s))
	}
}

func rotr[T integer](dst, a []T, s uint64) {
	w := bitsOf[T]()
	s %= w
	n := min(len(dst), len(a))
	for i := range n {
		if s == 0 {
			dst[i] = a[i]
			continue
		}
		u := uint64(a[i]) & (1<<w - 1)
		dst[i] = T(u>>s | u<<(w-s))
	}
}

// The per-element bit operations, defined by math/bits rather than by C. The
// zero case is the one that matters: LeadingZeros and TrailingZeros of zero
// are the element width, where __builtin_clz and __builtin_ctz are undefined
// and x86, arm64 and LLVM each do something different with it.
func onesCount[T integer](dst, a []T) {
	n, w := min(len(dst), len(a)), bitsOf[T]()
	for i := range n {
		dst[i] = T(bits.OnesCount64(uint64(a[i]) & (1<<w - 1)))
	}
}

func leadingZeros[T integer](dst, a []T) {
	n, w := min(len(dst), len(a)), bitsOf[T]()
	for i := range n {
		u := uint64(a[i]) & (1<<w - 1)
		dst[i] = T(uint64(bits.LeadingZeros64(u)) - (64 - w))
	}
}

func trailingZeros[T integer](dst, a []T) {
	n, w := min(len(dst), len(a)), bitsOf[T]()
	for i := range n {
		u := uint64(a[i]) & (1<<w - 1)
		if u == 0 {
			dst[i] = T(w)
			continue
		}
		dst[i] = T(bits.TrailingZeros64(u))
	}
}

func reverseBits[T integer](dst, a []T) {
	n, w := min(len(dst), len(a)), bitsOf[T]()
	for i := range n {
		u := uint64(a[i]) & (1<<w - 1)
		dst[i] = T(bits.Reverse64(u) >> (64 - w))
	}
}

func byteSwap[T integer](dst, a []T) {
	n, w := min(len(dst), len(a)), bitsOf[T]()
	for i := range n {
		u := uint64(a[i]) & (1<<w - 1)
		dst[i] = T(bits.ReverseBytes64(u) >> (64 - w))
	}
}

// transpose writes the m*n row-major matrix a as an n*m one. It does nothing
// if either slice is too short for the stated dimensions, matching MatMul.
func transpose[T number](dst, a []T, m, n int) {
	if m < 0 || n < 0 || len(a) < m*n || len(dst) < m*n {
		return
	}
	for i := range m {
		for j := range n {
			dst[j*m+i] = a[i*n+j]
		}
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

// The byte scanners the standard library already defines exactly.
//
// These delegate rather than reimplement, which is a departure from the rest
// of this package and is deliberate in both directions.
//
// It makes the reference stronger. The documented contract for each of these
// is "matches bytes.IndexByte and is a drop-in replacement", so the standard
// library is the specification, and comparing the kernels against a
// reimplementation of it only proves the kernels agree with our reading.
//
// It also makes the small-n path fast. The dispatcher runs the reference below
// each kernel's length threshold, and `bytes` is not naive Go there: IndexByte
// and Equal are assembly on the architectures that matter and can be inlined,
// where a call into this package cannot be. Written as a plain loop, this
// package *lost* to the standard library on a 64-byte input — which would make
// it a worse choice than the thing it replaces for every short string, however
// good the kernels are on a long one.
//
// Only the functions where the two agree exactly are here. IndexAny, CountAny
// and their complements are not: those are byte-set operations and the
// standard library's are rune-set operations, which is a different answer for
// any non-ASCII set and is documented as such on the exported wrappers.

func indexByte(b []byte, c byte) int     { return bytes.IndexByte(b, c) }
func lastIndexByte(b []byte, c byte) int { return bytes.LastIndexByte(b, c) }
func countByte(b []byte, c byte) int     { return bytes.Count(b, []byte{c}) }
func equalBytes(a, b []byte) bool        { return bytes.Equal(a, b) }
func compareBytes(a, b []byte) int       { return bytes.Compare(a, b) }

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
		Compress:  compress[T],
		Add3:      add3[T], Add4: add4[T], Mul3: mul3[T], Mul4: mul4[T],
		Partition: partitionOut[T],
		Gemv:      gemvFloat[T],
		Transpose: transpose[T],
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
		Compress:  compress[T],
		Add3:      add3[T], Add4: add4[T], Mul3: mul3[T], Mul4: mul4[T],
		Partition: partitionOut[T],
		Gemv:      gemvInt[T],
		Transpose: transpose[T],
		Sum:       sumInt[T], Min: minInt[T], Max: maxInt[T],
		SumSquares: sumSquaresInt[T], L1Norm: l1NormInt[T],
		Dot:      dotInt[T],
		SumSqDev: sumSqDevInt[T], SumSqDiff: sumSqDiffInt[T], L1Diff: l1DiffInt[T],
		ArgMin: argMinInt[T], ArgMax: argMaxInt[T], MinMax: minMaxInt[T],
		Shl: shl[T], Shr: shr[T], Rotl: rotl[T], Rotr: rotr[T],
		OnesCount: onesCount[T], LeadingZeros: leadingZeros[T],
		TrailingZeros: trailingZeros[T], ReverseBits: reverseBits[T],
		ByteSwap: byteSwap[T],
	}
	intMathOps(&o)
	signalOps(&o)
	compareOps(&o, medianInt[T], ltOrder[T])
	return o
}

// satOps is intOps plus the two saturating operations, for the types narrow
// enough to have them.
func satOps[T satInteger]() kernel.Ops[T] {
	o := intOps[T]()
	o.SatAdd = satAdd[T]
	o.SatSub = satSub[T]
	return o
}

// satAdd and satSub clamp at the element type's limits instead of wrapping.
//
// Both work in int64, which every type here fits in with room for the sum, so
// the clamp sees the true result rather than one that has already wrapped.
// That is also how the kernels are written, and it is why the 64-bit element
// types are absent: there is nothing wider to compute in.
func satAdd[T satInteger](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	lo, hi := satRange[T]()
	for i := range dst {
		dst[i] = T(clamp64(int64(a[i])+int64(b[i]), lo, hi))
	}
}

func satSub[T satInteger](dst, a, b []T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	lo, hi := satRange[T]()
	for i := range dst {
		dst[i] = T(clamp64(int64(a[i])-int64(b[i]), lo, hi))
	}
}

func clamp64(v, lo, hi int64) int64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// satRange is the closed interval T saturates to. It is derived from the type
// rather than tabulated, so a type added to satInteger cannot be given the
// wrong bounds by hand: the shift gives the width and the zero comparison the
// signedness.
func satRange[T satInteger]() (lo, hi int64) {
	var zero T
	bits := 8 * int64(unsafe.Sizeof(zero))
	if zero-1 > zero { // unsigned: 0-1 wraps to the maximum
		return 0, 1<<bits - 1
	}
	return -(1 << (bits - 1)), 1<<(bits-1) - 1
}

// Set returns the reference backend: every kernel, portable Go.
func Set() kernel.Set {
	return kernel.Set{
		Name: "scalar",
		F32:  floatOps[float32](),
		F64:  floatOps[float64](),
		I32:  satOps[int32](),
		I64:  intOps[int64](),
		I8:   satOps[int8](),
		I16:  satOps[int16](),
		U8:   satOps[uint8](),
		U16:  satOps[uint16](),
		U32:  satOps[uint32](),
		U64:  intOps[uint64](),
		Bytes: kernel.Bytes{
			IndexByte: indexByte, LastIndexByte: lastIndexByte, Count: countByte,
			Equal: equalBytes, Compare: compareBytes, PopCount: popCount,
			And: bitAnd, Or: bitOr, Xor: bitXor, AndNot: bitAndNot, Not: bitNot,
			Fill: fillBytes,

			IndexAll: indexAll, IndexAny: indexAny, CountAny: countAny, Index: index,
			IndexNotAny: indexNotAny, LastIndexNotAny: lastIndexNotAny, LastIndex: lastIndex, CountSeq: countSeq,
			IsASCII: isASCII, ValidUTF8: validUTF8,
			ToUpperASCII: toUpperASCII, ToLowerASCII: toLowerASCII,
			EqualFoldASCII: equalFoldASCII, ReplaceByte: replaceByte,
			HexEncode: hexEncode, HexDecode: hexDecode,
			ParseInts: ParseInts,
			B64Encode: b64Encode, B64Decode: b64Decode,
		},
		Convert:   convertOps(),
		Mask:      maskOps(),
		C64:       complexGroup[complex64, float32](),
		C128:      complexGroup[complex128, float64](),
		C64Parts:  complexPartsGroup[complex64, float32](),
		C128Parts: complexPartsGroup[complex128, float64](),
	}
}

// complexGroup builds one complex kernel group. It exists because the
// composite literal above cannot call a function that fills through a
// pointer, and filling through a pointer is what keeps complexOps readable.
func complexGroup[C complexT, R float]() kernel.Complex[C] {
	var o kernel.Complex[C]
	complexOps[C, R](&o)
	return o
}

func complexPartsGroup[C complexT, R float]() kernel.ComplexParts[C, R] {
	var o kernel.ComplexParts[C, R]
	complexPartsOps(&o)
	return o
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
		~float32 | ~float64 |
			~int8 | ~int16 | ~int32 | ~int64 |
			~uint8 | ~uint16 | ~uint32 | ~uint64
	}
	// Integer is the integer half of it, for the operations that are not the
	// same on floats: integer minimum is not IEEE minimum, and integer Abs
	// wraps where float Abs clears a sign bit.
	Integer = integer
	// Saturating is the integer types that have saturating add and subtract.
	// The 64-bit ones are absent for the reason kernels.saturating gives:
	// there is nothing wider to detect the overflow in.
	Saturating = satInteger
	// Float is the element types with a fixed-tree reduction.
	Float interface{ ~float32 | ~float64 }
)

func Add[T Number](dst, a, b []T) { add(dst, a, b) }
func Sub[T Number](dst, a, b []T) { sub(dst, a, b) }
func Mul[T Number](dst, a, b []T) { mul(dst, a, b) }

func SatAdd[T Saturating](dst, a, b []T) { satAdd(dst, a, b) }
func SatSub[T Saturating](dst, a, b []T) { satSub(dst, a, b) }

func Scale[T Number](dst, a []T, s T)        { scale(dst, a, s) }
func AddScaled[T Number](dst, a, b []T, s T) { addScaled(dst, a, b, s) }

func SumFloat[T Float](a []T) T    { return sumFloat(a) }
func DotFloat[T Float](a, b []T) T { return dotFloat(a, b) }

func Div[T Float](dst, a, b []T)          { div(dst, a, b) }
func MinimumFloat[T Float](dst, a, b []T) { minimumFloat(dst, a, b) }
func MaximumFloat[T Float](dst, a, b []T) { maximumFloat(dst, a, b) }
func MinimumInt[T Integer](dst, a, b []T) { minimumInt(dst, a, b) }
func MaximumInt[T Integer](dst, a, b []T) { maximumInt(dst, a, b) }

func AbsFloat[T Float](dst, a []T)   { absFloat(dst, a) }
func NegFloat[T Float](dst, a []T)   { negFloat(dst, a) }
func AbsInt[T Integer](dst, a []T)   { absInt(dst, a) }
func NegInt[T Integer](dst, a []T)   { negInt(dst, a) }
func Sqrt[T Float](dst, a []T)       { sqrt(dst, a) }
func Reciprocal[T Float](dst, a []T) { reciprocal(dst, a) }
func Reverse[T Number](dst, a []T)   { reverse(dst, a) }

func AddScalar[T Number](dst, a []T, s T) { addScalar(dst, a, s) }
func SubScalar[T Number](dst, a []T, s T) { subScalar(dst, a, s) }
func DivScalar[T Number](dst, a []T, s T) { divScalar(dst, a, s) }

func ClampFloat[T Float](dst, a []T, lo, hi T) { clampFloat(dst, a, lo, hi) }
func ClampInt[T Integer](dst, a []T, lo, hi T) { clampInt(dst, a, lo, hi) }
func Fill[T Number](dst []T, v T)              { fill(dst, v) }
func Ramp[T Number](dst []T, start, step T)    { ramp(dst, start, step) }
func Lerp[T Number](dst, a, b []T, t T)        { lerp(dst, a, b, t) }

func Floor[T Float](dst, a []T)       { unary[T](gomath.Floor)(dst, a) }
func Ceil[T Float](dst, a []T)        { unary[T](gomath.Ceil)(dst, a) }
func Trunc[T Float](dst, a []T)       { unary[T](gomath.Trunc)(dst, a) }
func Round[T Float](dst, a []T)       { unary[T](gomath.Round)(dst, a) }
func RoundToEven[T Float](dst, a []T) { unary[T](gomath.RoundToEven)(dst, a) }

func MinReduceFloat[T Float](a []T) T { return minFloat(a) }
func MaxReduceFloat[T Float](a []T) T { return maxFloat(a) }
func MinReduceInt[T Integer](a []T) T { return minInt(a) }
func MaxReduceInt[T Integer](a []T) T { return maxInt(a) }

func SumSquaresFloat[T Float](a []T) T { return sumSquaresFloat(a) }
func SumSquaresInt[T Integer](a []T) T { return sumSquaresInt(a) }
func L1NormFloat[T Float](a []T) T     { return l1NormFloat(a) }
func L1DiffFloat[T Float](a, b []T) T  { return l1DiffFloat(a, b) }

func SumSqDevFloat[T Float](a []T, c T) T { return sumSqDevFloat(a, c) }
func SumSqDevInt[T Integer](a []T, c T) T { return sumSqDevInt(a, c) }
func SumSqDiffFloat[T Float](a, b []T) T  { return sumSqDiffFloat(a, b) }
func SumSqDiffInt[T Integer](a, b []T) T  { return sumSqDiffInt(a, b) }

func OnesCount[T Integer](dst, a []T)     { onesCount(dst, a) }
func LeadingZeros[T Integer](dst, a []T)  { leadingZeros(dst, a) }
func TrailingZeros[T Integer](dst, a []T) { trailingZeros(dst, a) }
func ReverseBits[T Integer](dst, a []T)   { reverseBits(dst, a) }
func ByteSwap[T Integer](dst, a []T)      { byteSwap(dst, a) }

func Transpose[T Number](dst, a []T, m, n int) { transpose(dst, a, m, n) }

func Shl[T Integer](dst, a []T, s uint64)  { shl(dst, a, s) }
func Shr[T Integer](dst, a []T, s uint64)  { shr(dst, a, s) }
func Rotl[T Integer](dst, a []T, s uint64) { rotl(dst, a, s) }
func Rotr[T Integer](dst, a []T, s uint64) { rotr(dst, a, s) }

func SumInt[T Integer](a []T) T    { return sumInt(a) }
func DotInt[T Integer](a, b []T) T { return dotInt(a, b) }
func ProdInt[T Integer](a []T) T   { return prod(a) }
func Diff[T Number](dst, a []T)    { diff(dst, a) }

func L1NormInt[T Integer](a []T) T               { return l1NormInt(a) }
func L1DiffInt[T Integer](a, b []T) T            { return l1DiffInt(a, b) }
func CompareBytes(a, b []byte) int               { return compareBytes(a, b) }
func EqualFoldASCII(a, b []byte) bool            { return equalFoldASCII(a, b) }
func IndexAny(b, chars []byte) int               { return indexAny(b, chars) }
func CountAny(b, chars []byte) int               { return countAny(b, chars) }
func HexEncode(dst, src []byte) int              { return hexEncode(dst, src) }
func IndexAll(dst []int32, b []byte, c byte) int { return indexAll(dst, b, c) }

// ArgMin and ArgMax are exported for the generated threshold guards. The float
// and integer forms differ because only one of them has NaN to reckon with.

func MinMaxFloat[T Float](a []T) (T, T) { return minMaxFloat(a) }
func MinMaxInt[T Integer](a []T) (T, T) { return minMaxInt(a) }

func CumMinFloat[T Float](dst, a []T) { cumMinFloat(dst, a) }
func CumMaxFloat[T Float](dst, a []T) { cumMaxFloat(dst, a) }
func CumMinInt[T Integer](dst, a []T) { cumMinInt(dst, a) }
func CumMaxInt[T Integer](dst, a []T) { cumMaxInt(dst, a) }

func ArgMinFloat[T Float](a []T) int { return argMinFloat(a) }
func ArgMaxFloat[T Float](a []T) int { return argMaxFloat(a) }
func ArgMinInt[T Integer](a []T) int { return argMinInt(a) }
func ArgMaxInt[T Integer](a []T) int { return argMaxInt(a) }

func NormFloat[T Float](a []T) T { return normFloat(a) }
