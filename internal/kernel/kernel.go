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
// input, including NaN, ±Inf, ±0 and denormals. No reassociation is possible
// in an elementwise operation, so this is free. It is not negotiable.
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

	// Scan.
	CumSum func(dst, a []T)

	// Reductions to a scalar.
	Sum, Min, Max      func(a []T) T
	SumSquares, L1Norm func(a []T) T
	Norm               func(a []T) T
	Dot                func(a, b []T) T

	// Two-input and centred reductions. These exist as kernels rather than
	// being composed in the caller so that variance, distance and similarity
	// need one pass over memory and no scratch buffer.
	SumSqDev  func(a []T, c T) T // sum((a[i]-c)^2), for variance about a mean
	SumSqDiff func(a, b []T) T   // sum((a[i]-b[i])^2), squared Euclidean distance
	L1Diff    func(a, b []T) T   // sum(|a[i]-b[i]|), Manhattan distance

	// Reductions to an index or a pair.
	ArgMin, ArgMax func(a []T) int
	MinMax         func(a []T) (T, T)
}

// Bytes is the byte and bit kernel group.
type Bytes struct {
	IndexByte, LastIndexByte func(b []byte, c byte) int
	Count                    func(b []byte, c byte) int
	Equal                    func(a, b []byte) bool
	Compare                  func(a, b []byte) int
	PopCount                 func(b []byte) int

	And, Or, Xor, AndNot func(dst, a, b []byte)
	Not                  func(dst, a []byte)
	Fill                 func(dst []byte, v byte)
}

// Set is one complete backend: every kernel, for one tier.
type Set struct {
	// Name identifies the tier, matching cpu.Tier.String.
	Name  string
	F32   Ops[float32]
	F64   Ops[float64]
	I32   Ops[int32]
	I64   Ops[int64]
	Bytes Bytes
}
