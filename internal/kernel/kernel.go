// Package kernel defines the contract every backend implements.
//
// A [Set] is one complete implementation of every kernel, for one instruction
// set tier. Backends are swapped wholesale rather than per function, which
// keeps a tier internally consistent and lets the differential tests exercise
// every tier the host supports inside a single process, instead of re-running
// the suite once per GOSIMD setting.
//
// # The numerical contract
//
// This is the part that must not drift. The failure mode it exists to prevent
// is viterin/vek#11, where the vectorized body and the scalar remainder loop
// disagree on NaN, so the result of a reduction changes with the length of the
// input.
//
// Rule 1 — elementwise operations are bit-identical on every tier, for every
// input, including ±Inf, ±0 and denormals. No reassociation is possible in an
// elementwise operation, so this is free. It is not negotiable.
//
// The single exception is the payload of a NaN result. IEEE 754 does not say
// which NaN survives an operation whose operands are NaN, and hardware
// genuinely differs — x86 returns the first source operand, other
// architectures choose otherwise. Promising identical payloads would be
// promising something no implementation can deliver. What is promised is that
// a NaN in yields a NaN out, on every tier, which is what callers actually
// depend on.
//
// Rule 2 — integer reductions are bit-identical on every tier. Integer
// addition is associative, so accumulation order cannot be observed.
//
// Rule 3 — floating-point reductions accumulate into exactly [SumLanes]
// independent accumulators, then combine them with the fixed binary tree in
// [CombineTree]. Element i contributes to accumulator i%SumLanes. Every tier
// reproduces this exact shape regardless of its hardware vector width:
//
//	SSE2    float32: 4 xmm registers × 4 lanes
//	AVX2    float32: 2 ymm registers × 8 lanes
//	AVX-512 float32: 1 zmm register  × 16 lanes
//	NEON    float32: 4 q registers   × 4 lanes
//
// Scalable tiers (SVE2, RVV) whose hardware width may exceed SumLanes must
// still present exactly SumLanes accumulators, using predication to clamp the
// active lane count. That costs throughput on wide implementations. It is the
// price of the contract, and it is deliberate.
//
// Rule 4 — Dot does not contract into fused multiply-add. The multiply and the
// add round separately, which is what a naive scalar loop does. Kernels for
// Dot compile with -ffp-contract=off.
//
// Rule 5 — operations named Fast* are exempt from rules 3 and 4 and must
// document their accumulation order and error bound. They are never the
// default and never silently substituted.
//
// Rule 6 — transcendental functions (Exp, Log, Sin, Pow and the rest of
// [Ops.Exp] through [Ops.Atan2]) guarantee a stated ULP bound rather than bit
// identity. They are polynomial approximations, and the polynomial that is
// correct to 1 ULP in float32 is not the one that is correct to 1 ULP in
// float64, so no single evaluation order reproduces both. The default
// variants target 1.0 ULP; the Fast* variants target 3.5 ULP, matching
// SLEEF's u10 and u35 families.
//
// This rule is deliberately narrow. It covers only the transcendentals.
// Rounding (Floor, Ceil, Trunc, Round, RoundToEven) is exact and stays
// bit-identical under rule 1, and so does everything algebraic.
//
// A consequence worth stating: this library never compiles kernels with
// -ffast-math or -Ofast. vek does, which is why its NaN and Inf behavior is
// undefined and why the caveat had to be added to its README retroactively.
//
// # Thresholds
//
// Below a per-kernel element count, crossing into assembly costs more than the
// arithmetic saves — a Go-to-assembly call is a fixed ~1.4ns and can never be
// inlined. Each generated backend therefore guards its own entry point and
// defers to the reference implementation for small inputs. The threshold
// belongs to the kernel, not to this dispatch layer, because it depends on
// both the operation and the element type and must be measured rather than
// guessed.
package kernel

// SumLanes is the number of independent accumulators every floating-point
// reduction uses, on every tier, regardless of hardware vector width.
//
// Sixteen is chosen because it is enough to hide FMA latency (typically 4
// cycles at 2 per cycle, so ~8 are needed to saturate) and because it divides
// evenly into every fixed vector width this library targets, from 128-bit
// SSE2 through 512-bit AVX-512.
const SumLanes = 16

// CombineTree reduces the SumLanes accumulators to a single value using a
// fixed pairwise binary tree: 16→8→4→2→1, always in that shape, always in
// that order.
//
// Every tier must implement this exact tree. On AVX-512 float32 the sixteen
// accumulators occupy one zmm register and the tree is the standard
// horizontal-reduce shuffle sequence; on SSE2 they occupy four xmm registers
// and the first step is three vector adds. Both produce the same bits.
func CombineTree[T ~float32 | ~float64](acc *[SumLanes]T) T {
	for w := SumLanes / 2; w >= 1; w /= 2 {
		for j := range w {
			acc[j] += acc[j+w]
		}
	}
	return acc[0]
}

// Ops is the kernel group for one element type.
//
// Fields that are meaningless for an element type are nil: Div, Sqrt,
// Reciprocal and Norm are populated only for float types. The exported API
// constrains those operations to floats, so a nil field is never reached.
type Ops[T any] struct {
	// Elementwise, two inputs. Each writes min of the three lengths.
	Add, Sub, Mul, Div func(dst, a, b []T)
	Minimum, Maximum   func(dst, a, b []T)

	// Elementwise, one input.
	Abs, Neg, Sqrt, Reciprocal func(dst, a []T)
	Reverse                    func(dst, a []T)

	// Rounding. Exact, and so bit-identical under rule 1. Round is half away
	// from zero, matching math.Round; RoundToEven is half to even, matching
	// math.RoundToEven and the default IEEE 754 rounding mode.
	Floor, Ceil, Trunc, Round, RoundToEven func(dst, a []T)

	// Transcendentals. Float only, and governed by rule 6: a stated ULP
	// bound, not bit identity.
	Exp, Exp2, Expm1     func(dst, a []T)
	Log, Log2, Log10     func(dst, a []T)
	Log1p, Cbrt, Sigmoid func(dst, a []T)
	Sin, Cos, Tan        func(dst, a []T)
	Asin, Acos, Atan     func(dst, a []T)
	Sinh, Cosh, Tanh     func(dst, a []T)
	Pow, Atan2, Hypot    func(dst, a, b []T)

	// Elementwise with a scalar operand.
	//
	// DivScalar really divides rather than multiplying by a precomputed
	// reciprocal. Multiplying is faster but loses a bit: 3*(1/5) is
	// 0.6000000000000001 where 3/5 is exactly 0.6. Accuracy is the default
	// here; a reciprocal-multiply variant belongs under a Fast name.
	Scale, AddScalar, SubScalar, DivScalar func(dst, a []T, s T)
	Clamp                                  func(dst, a []T, lo, hi T)
	Fill                                   func(dst []T, v T)

	// Fused. AddScaled is AXPY: dst[i] = a[i] + b[i]*s in one pass over
	// memory, which is the whole reason a fused catalogue exists.
	AddScaled func(dst, a, b []T, s T)

	// Fused. Lerp is dst[i] = a[i] + (b[i]-a[i])*t, which is monotonic and
	// lands exactly on b at t=1, unlike the algebraically equal
	// a*(1-t) + b*t.
	Lerp func(dst, a, b []T, t T)

	// Scan. Diff writes successive differences, so len(dst) is one less than
	// len(a) worth of useful output.
	CumSum, CumProd, CumMin, CumMax, Diff func(dst, a []T)

	// Signal and polynomial kernels. These are their own kernels rather than
	// compositions because composing them would cost one pass over memory per
	// coefficient or per tap, which is exactly the memory-bound trap a
	// slice-at-a-time library falls into.
	//
	// PolyEval evaluates a polynomial at every point of x by Horner's method,
	// coefficients lowest order first, in a single pass.
	PolyEval func(dst, x, coeffs []T)
	// Convolve is the direct form: dst[i] = sum_j sig[i+j]*ker[j], so it
	// writes len(sig)-len(ker)+1 elements. Correlate is the same without
	// reversing the kernel.
	Convolve, Correlate func(dst, sig, ker []T)
	// MovingAverage writes the mean of each window of the given width,
	// producing len(a)-width+1 elements.
	MovingAverage func(dst, a []T, width int)
	// EMA is the exponentially weighted moving average, which is inherently
	// sequential: dst[i] = alpha*a[i] + (1-alpha)*dst[i-1].
	EMA func(dst, a []T, alpha T)

	// Reductions to a scalar. Prod wraps on integer overflow, like Sum.
	Sum, Prod, Min, Max func(a []T) T
	SumSquares, L1Norm  func(a []T) T
	Norm                func(a []T) T
	Dot                 func(a, b []T) T

	// Two-input and centred reductions. These exist as kernels rather than
	// being composed in the caller so that variance, distance and similarity
	// need one pass over memory and no scratch buffer.
	SumSqDev  func(a []T, c T) T // sum((a[i]-c)^2), for variance about a mean
	SumSqDiff func(a, b []T) T   // sum((a[i]-b[i])^2), squared Euclidean distance
	L1Diff    func(a, b []T) T   // sum(|a[i]-b[i]|), Manhattan distance

	// Reductions to an index or a pair.
	ArgMin, ArgMax func(a []T) int
	MinMax         func(a []T) (T, T)

	// Comparisons, writing a boolean per element. A vector unit produces a
	// mask register here; []bool is the portable spelling of that, one byte
	// per lane, which is also what a store of the mask produces.
	//
	// Float comparisons follow IEEE 754: any comparison involving NaN is
	// false, including NaN == NaN, so NotEqual is not the negation of Equal.
	EqualMask, NotEqualMask                   func(dst []bool, a, b []T)
	LessMask, LessEqualMask                   func(dst []bool, a, b []T)
	GreaterMask, GreaterEqualMask             func(dst []bool, a, b []T)
	EqualScalarMask, NotEqualScalarMask       func(dst []bool, a []T, v T)
	LessScalarMask, LessEqualScalarMask       func(dst []bool, a []T, v T)
	GreaterScalarMask, GreaterEqualScalarMask func(dst []bool, a []T, v T)

	// SatAdd and SatSub clamp at the element type's limits instead of
	// wrapping. They are the reason the narrow integer types are worth having:
	// a single instruction on every vector unit here, and the operation image
	// and audio code actually wants, where wrapping turns a bright pixel dark.
	//
	// Nil for the floating-point and 64-bit integer instantiations: floats
	// saturate at infinity by themselves, and no wider integer exists to
	// detect a 64-bit overflow in.
	SatAdd, SatSub func(dst, a, b []T)

	// Select is the blend: dst[i] = mask[i] ? yes[i] : no[i].
	Select func(dst []T, mask []bool, yes, no []T)

	// Gather reads src at the given indices; Scatter writes them. Both are
	// single instructions on AVX-512 and SVE2, and loops everywhere else.
	Gather  func(dst, src []T, idx []int32)
	Scatter func(dst []T, idx []int32, src []T)

	// Construction.
	Ramp   func(dst []T, start, step T) // dst[i] = start + i*step
	Tile   func(dst, pattern []T)       // repeat pattern across dst
	Median func(a []T) T                // reorders a; see the exported doc
	// Quantile takes q in [0,1] and interpolates between order statistics.
	Quantile func(a []T, q float64) T
	// MatMul multiplies an m*k matrix by a k*n matrix into an m*n one, all
	// in row-major order.
	MatMul func(dst, a, b []T, m, k, n int)
}

// Mask is the kernel group for boolean vectors, the output of the comparison
// kernels above.
type Mask struct {
	All, Any     func(m []bool) bool
	Count        func(m []bool) int
	And, Or, Xor func(dst, a, b []bool)
	Not          func(dst, a []bool)
}

// Bytes is the byte, bit and text kernel group.
type Bytes struct {
	IndexByte, LastIndexByte func(b []byte, c byte) int
	Count                    func(b []byte, c byte) int
	Equal                    func(a, b []byte) bool
	Compare                  func(a, b []byte) int
	PopCount                 func(b []byte) int

	And, Or, Xor, AndNot func(dst, a, b []byte)
	Not                  func(dst, a []byte)
	Fill                 func(dst []byte, v byte)

	// Text scanning. These are the primitives a tokenizer is built from, and
	// they are the part of a parser that actually benefits from vectors: a
	// whole register of bytes classified per instruction.
	//
	// IndexAll writes the offset of every occurrence of c and returns how many
	// it found. It is the structural-index step that a JSON or CSV parser
	// spends most of its time in, and the reason it is a kernel rather than a
	// loop over IndexByte is that the vector form classifies 16 to 64 bytes
	// per instruction and only touches memory once.
	IndexAll func(dst []int32, b []byte, c byte) int
	// IndexAny and CountAny match against a set of bytes rather than one.
	IndexAny, CountAny func(b, chars []byte) int
	// IndexNotAny is the complement: the first byte that is *not* in the set.
	// It is the primitive under trimming and under skipping a run of
	// whitespace, which is where a tokenizer spends the time it is not
	// spending in IndexAny.
	// IndexNotAny and LastIndexNotAny are trimming, from each end.
	IndexNotAny, LastIndexNotAny func(b, chars []byte) int
	// Index and LastIndex are substring search, forward and backward.
	// CountSeq counts non-overlapping occurrences.
	Index, LastIndex, CountSeq func(haystack, needle []byte) int
	// IsASCII reports whether every byte is below 0x80; ValidUTF8 reports
	// whether the whole slice is well-formed UTF-8.
	IsASCII, ValidUTF8 func(b []byte) bool
	// ASCII case folding, which unlike the Unicode kind is a branch-free
	// range compare and maps one byte to one byte.
	ToUpperASCII, ToLowerASCII func(dst, b []byte)
	EqualFoldASCII             func(a, b []byte) bool
	// ReplaceByte substitutes every occurrence of old with new.
	ReplaceByte func(dst, b []byte, old, new byte)
	// Hex encoding, lowercase. Decode returns the number of bytes written and
	// whether the input was valid.
	HexEncode func(dst, src []byte) int
	HexDecode func(dst, src []byte) (int, bool)
}

// Complex is the kernel group for complex numbers, parameterised by the
// complex type C and its real component type R.
//
// It is a separate group rather than another Ops instantiation because most
// of Ops does not apply: there is no ordering on the complex numbers, so
// Minimum, the comparisons and the sorts are all meaningless, and the ones
// that do apply often return a real rather than a complex.
//
// Go stores a complex as its two components adjacent in memory, so a slice of
// them is the interleaved layout, and that is what these kernels read. It is
// not the layout that vectorizes best — a multiply needs the real and
// imaginary parts in separate registers, which from interleaved data costs a
// shuffle — but it is the layout the caller already has, and converting would
// cost more than the shuffle does.
// The group is split in two by whether an operation's signature mentions the
// real component type. That is not tidiness, it is what lets the public API
// stay generic: a function over complex alone, like Add, cannot name the
// matching real type, because Go has no way to derive float32 from complex64
// in a type parameter list. Splitting means Add needs one parameter and Abs
// needs two, and each is inferable from its own arguments.
type Complex[C any] struct {
	Add, Sub, Mul, Div func(dst, a, b []C)
	Neg, Conj          func(dst, a []C)

	// Sum accumulates into the fixed lane count the real reductions use, so
	// that a complex sum does not change value with the vector width either.
	Sum func(a []C) C
	// Dot is the bilinear product, sum(a[i]*b[i]). DotConj is the Hermitian
	// one, sum(conj(a[i])*b[i]), which is the inner product most linear
	// algebra means; both are offered because both are wanted and neither is
	// obviously the default.
	Dot, DotConj func(a, b []C) C
}

// ComplexParts holds the operations that cross between a complex slice and a
// real one.
type ComplexParts[C any, R any] struct {
	// Scale multiplies by a real, which is the common case and avoids the
	// four multiplies a full complex product needs.
	Scale func(dst, a []C, s R)

	// Abs is the magnitude, computed through the scaled form that cannot
	// overflow for a representable answer, the same as Hypot.
	Abs func(dst []R, a []C)

	// Real and Imag extract a component; FromParts is the inverse.
	Real, Imag func(dst []R, a []C)
	FromParts  func(dst []C, re, im []R)
}

// Set is one complete backend: every kernel, for one tier.
type Set struct {
	// Name identifies the tier, matching cpu.Tier.String.
	Name string
	F32  Ops[float32]
	F64  Ops[float64]
	I32  Ops[int32]
	I64  Ops[int64]

	// The narrow and unsigned integers. They carry the same Ops shape, with
	// the operations that do not apply left nil, so that a caller reaching
	// one through the generic API gets the same surface as any other type.
	I8        Ops[int8]
	I16       Ops[int16]
	U8        Ops[uint8]
	U16       Ops[uint16]
	U32       Ops[uint32]
	U64       Ops[uint64]
	Bytes     Bytes
	Mask      Mask
	C64       Complex[complex64]
	C128      Complex[complex128]
	C64Parts  ComplexParts[complex64, float32]
	C128Parts ComplexParts[complex128, float64]
}
