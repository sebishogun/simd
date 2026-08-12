// Package simd provides runtime-dispatched SIMD operations over ordinary Go
// slices, without cgo.
//
// The package detects the current CPU once at startup and selects generated
// assembly for the available instruction-set tier. Operations without a
// generated kernel on that tier use the portable Go implementation.
//
// # Common call shapes
//
// Arithmetic operations commonly use a plain name for an in-place update:
//
//	simd.Add(a, b)        // a[i] += b[i]
//	simd.Scale(a, 2.5)    // a[i] *= 2.5
//	simd.Abs(a)           // a[i] = |a[i]|
//
// The corresponding Into form commonly takes a destination first:
//
//	simd.AddInto(dst, a, b)      // dst[i] = a[i] + b[i]
//	simd.ScaleInto(dst, a, 2.5)
//
// Reductions return a value and write nothing:
//
//	total := simd.Sum(a)
//	d := simd.Dot(a, b)
//	lo, hi := simd.MinMax(a)
//
// These are conventions, not a grammar for the entire package. Decoders return
// progress counts, append-style functions may grow a destination, and some
// Into functions take workspace while modifying another argument. Each
// function documents its output length and overlap rules.
//
// Generated kernels do not allocate. Into forms let hot paths supply and reuse
// output or workspace. Convenience functions that return slices or construct
// plans and workspaces can allocate and say so in their documentation.
//
// Generic constraints cover the numeric types supported by each operation, so
// element types are inferred and names do not carry per-type suffixes.
//
// # The right instructions are chosen for you
//
// The instruction set is selected once, at process start, from the CPU the
// program is actually running on. A binary built on a laptop with AVX-512 runs
// correctly on a server without it, using AVX2 or SSE2 instead; on an
// architecture with no backend it uses portable Go. Nothing to configure,
// nothing to build twice.
//
// # Numerical contracts
//
// Core operations agree bit for bit across generated tiers and the portable
// path. Reductions such as [Sum] and [Dot] use a fixed accumulation order that
// a 128-bit and a 512-bit machine both reproduce; Dot does not contract into
// fused multiply-add.
//
// Transcendentals instead document an error bound against the standard
// library. Fast* transcendental functions allow a wider bound and may vary by
// architecture. Sort and order-statistic functions treat -0 and +0 as equal,
// so their order inside that tie may differ. Function comments state narrower
// exceptions.
//
// # Sizing
//
// Elementwise operations generally process the minimum length of their slice
// arguments, so slicing is how you bound that work:
//
//	simd.AddInto(dst[:n], a, b)
//
// Decoders, matrices, compaction, packed layouts, and other output-shaped
// operations have their own sizing contracts. Small inputs generally use a Go
// path because below roughly 16 to 64 elements the assembly call can cost more
// than the work it saves.
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
	float32 | float64 |
		int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64
}

// Float is the element type of operations that only make sense for floats.
type Float interface {
	float32 | float64
}

// Integer is the element type of operations that only make sense for
// integers: the bitwise ones, and the two saturating ones.
type Integer interface {
	int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64
}

// Saturating is the element type of [SatAdd] and [SatSub].
//
// The 64-bit types are absent, and it is a limit of the implementation rather
// than of the idea. A saturating add is written as a widening add followed by
// a clamp, which is the form that compiles to the single instruction every
// vector unit has for it — and there is no integer wider than 64 bits to
// widen into. Clamping a 64-bit sum needs an overflow test, which does not
// vectorize into anything worth crossing the call boundary for.
type Saturating interface {
	int8 | int16 | int32 | uint8 | uint16 | uint32
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

// ---------- saturating arithmetic ----------
//
// [Add] and [Sub] wrap on overflow, which is what Go's operators do and what
// the hardware does. Saturating is the other useful answer, and the one that
// image, audio and fixed-point code wants: brightening an already-bright pixel
// should leave it white, not wrap it to black. It is a single instruction on
// every vector unit here.

// SatAdd adds b into a with saturation: a[i] = clamp(a[i] + b[i]).
//
// A sum past the element type's maximum gives the maximum, and one past its
// minimum gives the minimum, instead of wrapping.
//
// It processes min(len(a), len(b)) elements and allocates nothing.
// Use [SatAddInto] to write the result elsewhere.
func SatAdd[T Saturating](a, b []T) { ops[T]().SatAdd(a, a, b) }

// SatSub subtracts b from a with saturation: a[i] = clamp(a[i] - b[i]).
//
// For an unsigned type this is the useful one of the pair: the difference
// clamps at zero rather than wrapping to a huge value.
func SatSub[T Saturating](a, b []T) { ops[T]().SatSub(a, a, b) }

// SatAddInto sets dst[i] to the saturating sum of a[i] and b[i].
// dst may alias a or b.
func SatAddInto[T Saturating](dst, a, b []T) { ops[T]().SatAdd(dst, a, b) }

// SatSubInto sets dst[i] to the saturating difference of a[i] and b[i].
// dst may alias a or b.
func SatSubInto[T Saturating](dst, a, b []T) { ops[T]().SatSub(dst, a, b) }

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
//
// # This one is permanently portable, and that is a decision rather than a gap
//
// There is no accelerated CumSum on any architecture and there will not be.
// Each output depends on the one before it, so the only way to vectorize a
// scan is to regroup the arithmetic — and unlike a reduction, every partial
// result is written to dst, so the regrouping is visible in the answer. This
// package's contract is that the accelerated and portable paths agree bit for
// bit, which forbids exactly that.
//
// For integers the contract is not the obstacle, since integer addition is
// associative; the measurement is. A blocked scan replaces a chain of dependent
// adds with a shuffle chain, which only pays when the serial chain is
// latency-bound, and a one-cycle add is not: int32 measured 980µs serial
// against 1082µs blocked, and int64 1441µs against 2165µs.
//
// [FastCumSum] is the opt-in float version, which drops agreement with a naive
// serial loop — and is, measurably, closer to the true sum. [CumMin] and
// [CumMax] are accelerated for integers, because minimum and maximum are
// associative and regrouping them changes nothing at all.
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
//
// Accelerated for int32 and portable everywhere else, which is the one place
// in this family where the two halves of the rule pull apart. Two's-complement
// multiplication is associative — wrapping does not change that, since the
// arithmetic is exact in Z/2^32 — so the blocked scan computes bit for bit
// what the serial loop does, and a three-cycle multiply leaves enough latency
// to be worth hiding: 2.04x, verified over four million deliberately
// overflowing values.
//
// int64 is portable because there is no 64-bit vector multiply below AVX-512DQ
// (0.67x), and float is portable for the reason [CumSum] gives. [FastCumProd]
// is the opt-in float version.
func CumProd[T Number](a []T) { ops[T]().CumProd(a, a) }

// CumProdInto writes the running products of a into dst. dst may alias a.
func CumProdInto[T Number](dst, a []T) { ops[T]().CumProd(dst, a) }

// FastCumSum is [CumSum] with the prefix sums grouped for the vector unit.
//
// # What is traded
//
// A prefix scan writes every partial result, so its grouping is observable:
// the serial loop computes ((a0+a1)+a2) where the vector form computes
// (a0+(a1+a2)), and floating-point addition is not associative, so the two
// differ in the last place. [CumSum] therefore stays serial, permanently, and
// this is the opt-in that does not.
//
// What is NOT traded is agreement between machines. The block is sixteen
// elements for float32 and eight for float64, on every tier and in the
// portable path, so this returns identical bits on a Graviton, an AVX-512 box
// and `-tags purego`. Only agreement with a naive loop is given up.
//
// # And it is not less accurate
//
// The obvious reading of Fast* — as it is for the transcendentals, which
// really do trade 1.0 ULP for 3.5 — is wrong here. Blocked summation has
// O(log n) error growth where a running accumulator has O(n), so measured
// against a long-double scan of a million values this is *closer* to the true
// result than [CumSum] on every corpus tried: 680,000 of a million elements
// closer on uniform positive input, and on the case that breaks serial
// accumulation — 1e16 followed by a million ones, where the running total
// cannot represent the increment — a mean absolute error of 1.0 against
// [CumSum]'s 5.0e+05.
//
// # float64 is the serial loop, deliberately
//
// Measured at four million elements, against [CumSum]:
//
//	float32   1669 us -> 1347 us   1.24x   on avx512, and 2.59x on sse2
//	float64   1745 us -> 1909 us   0.91x   SLOWER, so it is not used
//
// Eight doubles fill one AVX-512 register, so the shift steps become
// cross-lane permutes and there is not enough serial latency to hide behind
// them. float32 has sixteen lanes and wins. Rather than offer a Fast function
// that is slower than the plain one on the tier most machines select,
// FastCumSum on float64 IS [CumSum] — same bits, same speed, no surprise. The
// float32 path is the blocked scan and everything above applies to it.
//
// The wider the vector, the less this helps, which is why sse2 is the fastest
// tier for it. That is unusual enough to be worth stating: the cost of a
// log-shift scan is the shuffle, and shuffles get more expensive with width
// while the work per element does not.
func FastCumSum[T Float](a []T) { ops[T]().FastCumSum(a, a) }

// FastCumSumInto writes the running totals of a into dst. dst may alias a.
// See [FastCumSum] for what it trades.
func FastCumSumInto[T Float](dst, a []T) { ops[T]().FastCumSum(dst, a) }

// FastCumProd is [CumProd] grouped for the vector unit, trading agreement with
// a serial loop exactly as [FastCumSum] does.
//
// Measured at four million elements against [CumProd]: 3.65x on float32 and
// 1.76x on float64 — the largest speedup in this family, and unlike
// [FastCumSum] it is worth having for both types. A floating-point multiply
// has the longest serial latency of the four combines, so there is the most to
// hide behind the shuffles.
//
// Integer CumProd needs no Fast form: two's-complement multiplication is
// associative, so [CumProd] on int32 is already this algorithm and already
// bit-identical to the serial loop, at 2.17x.
func FastCumProd[T Float](a []T) { ops[T]().FastCumProd(a, a) }

// FastCumProdInto writes the running products of a into dst. dst may alias a.
func FastCumProdInto[T Float](dst, a []T) { ops[T]().FastCumProd(dst, a) }

// CumMin replaces every element with the smallest value seen so far.
func CumMin[T Number](a []T) { ops[T]().CumMin(a, a) }

// CumMinInto writes the running minimum of a into dst. dst may alias a.
func CumMinInto[T Number](dst, a []T) { ops[T]().CumMin(dst, a) }

// CumMax replaces every element with the largest value seen so far.
func CumMax[T Number](a []T) { ops[T]().CumMax(a, a) }

// CumMaxInto writes the running maximum of a into dst. dst may alias a.
func CumMaxInto[T Number](dst, a []T) { ops[T]().CumMax(dst, a) }

// RollingMinInto writes the minimum of every window of the given size into
// dst: dst[i] is the smallest element of a[i : i+window].
//
// There are len(a)-window+1 outputs, so dst must have room for that many. A
// window that is not positive, or is longer than a, writes nothing.
//
// The extreme is IEEE 754-2019 minimum, the same one Min and Minimum use: a
// window containing a NaN yields NaN, and -0 is smaller than +0.
//
// dst must not overlap a.
//
// # Use this below a window of about 48, and a deque above it
//
// The textbook sliding-window minimum is a monotonic deque: two amortized
// comparisons per element, whatever the window. This does window-1 elementwise
// passes, which is more arithmetic — but each pass is a plain contiguous
// minimum, the shape a vector unit is fastest at, and they are tiled so the
// block being accumulated stays in L1 across all of them. So it does sixteen
// windows at a time where the deque does one, and the comparison turns on the
// *window*, not on n. Measured on a Zen 5 at one million float64:
//
//	window     this     hand-written deque
//	     4     0.65 ms   8.35 ms   12.8x
//	     8     1.35      8.90       6.6x
//	    16     2.79      8.62       3.1x
//	    32     5.65      8.44       1.5x
//	    64    11.2       8.33       0.75x
//	   256    44.7       8.21       0.18x
//
// The crossover is just above 32, which is four times the eight float64 lanes
// an AVX-512 register holds. Above roughly 48, write the deque.
//
// This function does not switch to a deque itself. A deque needs an index ring
// proportional to the window, which would add hidden workspace to this
// caller-owned-output operation; and
// getting IEEE minimum out of one is subtle in a way that would not show up in
// testing — "pop the back while it is worse" does nothing when neither operand
// orders, so a plain deque holds a NaN without ever reporting it. A third
// implementation of these semantics is a liability, so the function states
// where this implementation stops paying. See docs/wrong.md entry 64.
func RollingMinInto[T Number](dst, a []T, window int) { ops[T]().RollingMin(dst, a, window) }

// RollingMaxInto writes the maximum of every window of the given size into dst.
// See [RollingMinInto], whose contract it shares.
func RollingMaxInto[T Number](dst, a []T, window int) { ops[T]().RollingMax(dst, a, window) }

// DiffInto writes the successive differences of a into dst:
// dst[i] = a[i+1] - a[i].
//
// It produces one fewer element than a has, so size dst accordingly.
// dst may alias a.
func DiffInto[T Number](dst, a []T) { ops[T]().Diff(dst, a) }
