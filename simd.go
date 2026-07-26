// Package simd provides SIMD-accelerated operations over slices, on every
// architecture that has a vector unit, without cgo.
//
// You call ordinary Go functions on ordinary Go slices. There is no vector
// type, no lane count, no target selection, and nothing to initialize.
//
// # In place by default
//
// The plain name operates in place on its first argument, which allocates
// nothing and reuses memory you already have:
//
//	simd.Add(a, b)        // a[i] += b[i]
//	simd.Scale(a, 2.5)    // a[i] *= 2.5
//	simd.Abs(a)           // a[i] = |a[i]|
//
// When the result belongs somewhere else, the same name with an Into suffix
// takes a destination first:
//
//	simd.AddInto(dst, a, b)      // dst[i] = a[i] + b[i]
//	simd.ScaleInto(dst, a, 2.5)
//
// Reductions return a single value and write nothing:
//
//	total := simd.Sum(a)
//	d := simd.Dot(a, b)
//	lo, hi := simd.MinMax(a)
//
// This package never allocates. Every function writes only into memory the
// caller supplied, and no variant returns a freshly made slice. If you want
// one, make it yourself and use the Into form.
//
// The functions are generic over float32, float64, int32 and int64, so the
// element type is inferred and there are no per-type name suffixes.
//
// # The right instructions are chosen for you
//
// The instruction set is selected once, at process start, from the CPU the
// program is actually running on. A binary built on a laptop with AVX-512 runs
// correctly on a server without it, using AVX2 or SSE2 instead; on an
// architecture with no backend it uses portable Go. Nothing to configure,
// nothing to build twice.
//
// # Results do not depend on the hardware
//
// Every operation returns bit-identical results on every instruction set,
// including for NaN, ±Inf, ±0 and denormals. Reductions such as [Sum] and
// [Dot] use a fixed accumulation order that a 128-bit and a 512-bit machine
// both reproduce exactly, so a computation cannot change answer because it
// moved to a different server. Operations that trade this away for speed are
// named Fast* and say so.
//
// This costs some throughput and is deliberate: the alternative is a library
// where results shift with slice length or CPU model, which is the most common
// bug in this class of package.
//
// # Sizing
//
// Lengths need not match. Every operation processes the minimum length of its
// slice arguments, so slicing is how you bound work:
//
//	simd.AddInto(dst[:n], a, b)
//
// Small inputs use an inlined scalar path rather than a call into assembly,
// because below roughly sixteen elements the call costs more than the
// arithmetic saves. You do not need to check for this.
//
// # Environment
//
// GOSIMD names an instruction-set tier to use instead of the detected one, for
// benchmarking and debugging: GOSIMD=sse2, GOSIMD=avx2, GOSIMD=scalar. It can
// only select down; naming a tier the CPU lacks falls back to portable Go
// rather than crashing.
//
// SIMD_DISABLE masks tiers out of consideration, as a comma-separated list:
// SIMD_DISABLE=avx512. Useful on CPUs where wide vectors cause frequency
// throttling.
//
// [Tier] and [Describe] report what was selected.
package simd

// Number is the element type of the arithmetic operations.
//
// The types are listed exactly rather than as approximations, so a defined
// type such as `type Celsius float32` is not accepted. Supporting those would
// require reinterpreting the slice, which this package does not do.
type Number interface {
	float32 | float64 | int32 | int64
}

// Float is the element type of operations that only make sense for floats.
type Float interface {
	float32 | float64
}

// ---------- elementwise, two inputs, in place ----------

// Add adds b into a, elementwise: a[i] += b[i].
//
// It processes min(len(a), len(b)) elements and allocates nothing.
// Use [AddInto] to write the result elsewhere.
func Add[T Number](a, b []T) { ops[T]().Add(a, a, b) }

// Sub subtracts b from a, elementwise: a[i] -= b[i].
func Sub[T Number](a, b []T) { ops[T]().Sub(a, a, b) }

// Mul multiplies a by b, elementwise: a[i] *= b[i].
//
// Integer multiplication wraps on overflow, matching the hardware.
func Mul[T Number](a, b []T) { ops[T]().Mul(a, a, b) }

// Div divides a by b, elementwise: a[i] /= b[i].
//
// Division by zero follows IEEE 754 and yields ±Inf or NaN; it does not panic.
func Div[T Float](a, b []T) { ops[T]().Div(a, a, b) }

// Minimum keeps the smaller of each pair: a[i] = min(a[i], b[i]).
//
// This is the elementwise operation. For the smallest element of one slice,
// see [Min].
func Minimum[T Number](a, b []T) { ops[T]().Minimum(a, a, b) }

// Maximum keeps the larger of each pair: a[i] = max(a[i], b[i]).
//
// This is the elementwise operation. For the largest element of one slice,
// see [Max].
func Maximum[T Number](a, b []T) { ops[T]().Maximum(a, a, b) }

// ---------- elementwise, two inputs, with a destination ----------

// AddInto sets dst[i] = a[i] + b[i].
//
// It processes min(len(dst), len(a), len(b)) elements. dst may alias a or b.
func AddInto[T Number](dst, a, b []T) { ops[T]().Add(dst, a, b) }

// SubInto sets dst[i] = a[i] - b[i]. dst may alias a or b.
func SubInto[T Number](dst, a, b []T) { ops[T]().Sub(dst, a, b) }

// MulInto sets dst[i] = a[i] * b[i]. dst may alias a or b.
func MulInto[T Number](dst, a, b []T) { ops[T]().Mul(dst, a, b) }

// DivInto sets dst[i] = a[i] / b[i]. dst may alias a or b.
func DivInto[T Float](dst, a, b []T) { ops[T]().Div(dst, a, b) }

// MinimumInto sets dst[i] = min(a[i], b[i]). dst may alias a or b.
func MinimumInto[T Number](dst, a, b []T) { ops[T]().Minimum(dst, a, b) }

// MaximumInto sets dst[i] = max(a[i], b[i]). dst may alias a or b.
func MaximumInto[T Number](dst, a, b []T) { ops[T]().Maximum(dst, a, b) }

// ---------- elementwise, one input, in place ----------

// Abs replaces every element with its absolute value.
//
// For floats this clears the sign bit, so -0 becomes +0 and NaN keeps its
// payload. For integers it wraps: the absolute value of the most negative
// value is itself, matching the hardware instruction.
func Abs[T Number](a []T) { ops[T]().Abs(a, a) }

// Neg negates every element: a[i] = -a[i].
//
// For floats this flips the sign bit, which unlike 0-x is correct for ±0 and
// NaN. For integers it wraps.
func Neg[T Number](a []T) { ops[T]().Neg(a, a) }

// Sqrt replaces every element with its square root.
//
// A negative input yields NaN, following IEEE 754.
func Sqrt[T Float](a []T) { ops[T]().Sqrt(a, a) }

// Reciprocal replaces every element with 1/x.
//
// This is a correctly rounded division, not a fast approximation.
func Reciprocal[T Float](a []T) { ops[T]().Reciprocal(a, a) }

// Reverse reverses the order of the elements.
func Reverse[T Number](a []T) { ops[T]().Reverse(a, a) }

// ---------- elementwise, one input, with a destination ----------

// AbsInto sets dst[i] to the absolute value of a[i]. See [Abs] for semantics.
func AbsInto[T Number](dst, a []T) { ops[T]().Abs(dst, a) }

// NegInto sets dst[i] = -a[i]. See [Neg] for semantics.
func NegInto[T Number](dst, a []T) { ops[T]().Neg(dst, a) }

// SqrtInto sets dst[i] to the square root of a[i].
func SqrtInto[T Float](dst, a []T) { ops[T]().Sqrt(dst, a) }

// ReciprocalInto sets dst[i] = 1/a[i].
func ReciprocalInto[T Float](dst, a []T) { ops[T]().Reciprocal(dst, a) }

// ReverseInto writes a into dst in reverse order.
//
// dst may be a itself, but partially overlapping slices are not supported.
func ReverseInto[T Number](dst, a []T) { ops[T]().Reverse(dst, a) }

// ---------- with a scalar operand ----------

// Scale multiplies every element by s: a[i] *= s.
func Scale[T Number](a []T, s T) { ops[T]().Scale(a, a, s) }

// AddScalar adds s to every element: a[i] += s.
func AddScalar[T Number](a []T, s T) { ops[T]().AddScalar(a, a, s) }

// Clamp limits every element to the range [lo, hi].
//
// For floats, NaN inputs propagate rather than being clamped.
func Clamp[T Number](a []T, lo, hi T) { ops[T]().Clamp(a, a, lo, hi) }

// Fill sets every element to v.
func Fill[T Number](a []T, v T) { ops[T]().Fill(a, v) }

// Zero sets every element to the zero value.
func Zero[T Number](a []T) { var z T; ops[T]().Fill(a, z) }

// ScaleInto sets dst[i] = a[i] * s. dst may alias a.
func ScaleInto[T Number](dst, a []T, s T) { ops[T]().Scale(dst, a, s) }

// AddScalarInto sets dst[i] = a[i] + s. dst may alias a.
func AddScalarInto[T Number](dst, a []T, s T) { ops[T]().AddScalar(dst, a, s) }

// ClampInto sets dst[i] to a[i] limited to [lo, hi]. dst may alias a.
func ClampInto[T Number](dst, a []T, lo, hi T) { ops[T]().Clamp(dst, a, lo, hi) }

// ---------- fused ----------

// AddScaled adds a scaled b into a: a[i] += b[i] * s.
//
// This is the AXPY of BLAS, and it is the reason a fused catalogue exists:
// written as separate Mul and Add calls the same work costs two passes over
// memory, which is what makes naive slice libraries memory-bound.
func AddScaled[T Number](a, b []T, s T) { ops[T]().AddScaled(a, a, b, s) }

// AddScaledInto sets dst[i] = a[i] + b[i]*s in a single pass over memory.
func AddScaledInto[T Number](dst, a, b []T, s T) { ops[T]().AddScaled(dst, a, b, s) }

// ---------- scan ----------

// CumSum replaces every element with the running total up to and including it.
func CumSum[T Number](a []T) { ops[T]().CumSum(a, a) }

// CumSumInto writes the running totals of a into dst. dst may alias a.
func CumSumInto[T Number](dst, a []T) { ops[T]().CumSum(dst, a) }

// SubScalar subtracts s from every element: a[i] -= s.
func SubScalar[T Number](a []T, s T) { ops[T]().SubScalar(a, a, s) }

// DivScalar divides every element by s: a[i] /= s.
//
// It really divides rather than multiplying by a precomputed reciprocal, which
// is slower but exact: 3*(1/5) is 0.6000000000000001 where 3/5 is 0.6.
// Integer division truncates toward zero and panics on a zero divisor.
func DivScalar[T Number](a []T, s T) { ops[T]().DivScalar(a, a, s) }

// SubScalarInto sets dst[i] = a[i] - s. dst may alias a.
func SubScalarInto[T Number](dst, a []T, s T) { ops[T]().SubScalar(dst, a, s) }

// DivScalarInto sets dst[i] = a[i] / s. dst may alias a. See [DivScalar].
func DivScalarInto[T Number](dst, a []T, s T) { ops[T]().DivScalar(dst, a, s) }

// CumProd replaces every element with the running product up to and including
// it.
func CumProd[T Number](a []T) { ops[T]().CumProd(a, a) }

// CumProdInto writes the running products of a into dst. dst may alias a.
func CumProdInto[T Number](dst, a []T) { ops[T]().CumProd(dst, a) }

// CumMin replaces every element with the smallest value seen so far.
func CumMin[T Number](a []T) { ops[T]().CumMin(a, a) }

// CumMinInto writes the running minimum of a into dst. dst may alias a.
func CumMinInto[T Number](dst, a []T) { ops[T]().CumMin(dst, a) }

// CumMax replaces every element with the largest value seen so far.
func CumMax[T Number](a []T) { ops[T]().CumMax(a, a) }

// CumMaxInto writes the running maximum of a into dst. dst may alias a.
func CumMaxInto[T Number](dst, a []T) { ops[T]().CumMax(dst, a) }

// DiffInto writes the successive differences of a into dst:
// dst[i] = a[i+1] - a[i].
//
// It produces one fewer element than a has, so size dst accordingly.
// dst may alias a.
func DiffInto[T Number](dst, a []T) { ops[T]().Diff(dst, a) }
