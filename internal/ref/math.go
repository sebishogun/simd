package ref

import (
	"math"

	"github.com/sebishogun/simd/internal/kernel"
)

// Rounding, transcendental and scan kernels.
//
// # Why the transcendentals go through float64
//
// The reference evaluates every transcendental in float64 and rounds to T.
// For float32 that is a double rounding, so the result is not always the
// correctly rounded float32 answer.
//
// That is acceptable here, and deliberate, because transcendentals are
// governed by rule 6 of the kernel contract: they promise a stated ULP bound,
// not bit identity. A polynomial correct to 1 ULP in float32 is not the one
// correct to 1 ULP in float64, so no evaluation order could make a float32
// kernel and a float64 kernel agree bit for bit anyway. Pretending otherwise
// would be the kind of contract that quietly breaks.
//
// Rounding is a different matter. Floor, Ceil, Trunc, Round and RoundToEven
// are exact for every representable input, so they stay bit-identical under
// rule 1 and the float64 detour costs nothing.

// unary lifts a float64 scalar function to a kernel over a slice.
//
// One line per operation beats thirty near-identical loops, and it makes the
// list of what each backend has to replace obvious at a glance.
func unary[T float](f func(float64) float64) func(dst, a []T) {
	return func(dst, a []T) {
		n := min(len(dst), len(a))
		dst, a = dst[:n], a[:n]
		for i := range dst {
			dst[i] = T(f(float64(a[i])))
		}
	}
}

// binary lifts a two-argument float64 scalar function to a kernel.
func binary[T float](f func(x, y float64) float64) func(dst, a, b []T) {
	return func(dst, a, b []T) {
		n := min(len(dst), len(a), len(b))
		dst, a, b = dst[:n], a[:n], b[:n]
		for i := range dst {
			dst[i] = T(f(float64(a[i]), float64(b[i])))
		}
	}
}

// Three of the standard library's functions lose most of their significant
// digits in part of their range, because each is written as an identity that
// cancels. This library promises a ULP bound that holds on every tier, and the
// portable path is a tier — the one a -tags purego build uses everywhere and
// the one the baseline x86-64 backend falls back to — so it has to meet the
// bound too.
//
// Each replacement below is the same function computed through an identity
// that does not cancel, using the standard library only where it is accurate.
// Checked against glibc: at the points where these and math disagree, these
// are glibc's answer to the bit.

// log2Accurate avoids math.Log2's cancellation.
//
// math.Log2 splits x with Frexp and returns log(frac)*(1/Ln2) + float64(exp).
// For x a little above 1 that is exp=1 and frac just above 1/2, so it adds two
// numbers near 1 to get an answer near 0: at x = 1.0139 it is 26 ULP out.
// Dividing the logarithm has no such step.
func log2Accurate(x float64) float64 {
	if x == 1 {
		return 0
	}
	return math.Log(x) / math.Ln2
}

// asinAccurate avoids math.Asin's cancellation near the ends of its domain.
//
// math.Asin computes Atan(x/Sqrt(1-x*x)), and 1-x*x loses half its digits as
// |x| approaches 1. The half-angle identity moves the argument to where the
// standard library is accurate, and 1-|x| is exact there by Sterbenz's lemma.
func asinAccurate(x float64) float64 {
	ax := math.Abs(x)
	if ax <= 0.5 || ax > 1 || math.IsNaN(x) {
		return math.Asin(x)
	}
	v := math.Pi/2 - 2*math.Asin(math.Sqrt((1-ax)/2))
	if x < 0 {
		return -v
	}
	return v
}

// acosAccurate avoids math.Acos's cancellation near 1.
//
// math.Acos is pi/2 - Asin(x). Near x=1 both terms are near pi/2 and the
// answer is near 0, so the subtraction throws away about seven digits — 971
// ULP at x = 0.9999. The half-angle form has no subtraction of like-sized
// quantities.
func acosAccurate(x float64) float64 {
	ax := math.Abs(x)
	if ax <= 0.5 || ax > 1 || math.IsNaN(x) {
		return math.Acos(x)
	}
	a := 2 * math.Asin(math.Sqrt((1-ax)/2))
	if x < 0 {
		return math.Pi - a
	}
	return a
}

// sigmoid is the logistic function, 1/(1+exp(-x)).
//
// It is computed in the branch that avoids overflow: exp(x) overflows for
// large positive x, exp(-x) for large negative x, so pick whichever exponent
// is negative. The naive form returns NaN at around x = -746 in float64.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// lerp interpolates from a to b by t.
//
// The form a + (b-a)*t is used rather than the algebraically equal
// a*(1-t) + b*t because it is monotonic in t and lands exactly on b at t=1.
// The other form can overshoot by an ULP, which matters when the result feeds
// a clamp or an index.
func lerp[T number](dst, a, b []T, t T) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] + T((b[i]-a[i])*t)
	}
}

// ---------- scans ----------

// cumProd is a running product. Safe in place: each output depends only on
// inputs at or before its own index.
func cumProd[T number](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	if n == 0 {
		return
	}
	run := T(1)
	for i := range dst {
		run *= a[i]
		dst[i] = run
	}
}

func cumMinFloat[T float](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	if n == 0 {
		return
	}
	run := a[0]
	for i := range dst {
		run = min2Float(run, a[i])
		dst[i] = run
	}
}

func cumMaxFloat[T float](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	if n == 0 {
		return
	}
	run := a[0]
	for i := range dst {
		run = max2Float(run, a[i])
		dst[i] = run
	}
}

func cumMinInt[T integer](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	if n == 0 {
		return
	}
	run := a[0]
	for i := range dst {
		run = min(run, a[i])
		dst[i] = run
	}
}

func cumMaxInt[T integer](dst, a []T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	if n == 0 {
		return
	}
	run := a[0]
	for i := range dst {
		run = max(run, a[i])
		dst[i] = run
	}
}

// rolling* is the sliding-window extreme: dst[i] is the minimum or maximum of
// a[i : i+window], so there are len(a)-window+1 outputs.
//
// The obvious reference is a monotonic deque, which is O(n) rather than the
// O(n·window) written here. It is not used, and the reason is NaN: IEEE
// minimum propagates it, and "pop the back while it is worse than the new
// element" has no defined behaviour when neither operand orders. The deque and
// the kernel would then disagree on any window containing a NaN, and a
// reference that disagrees with the kernel is worse than a slow one.
//
// This does the same window-1 combines in the same order with the same
// minimum, so the two agree bit for bit on every input.
func rollingMinFloat[T float](dst, a []T, window int) {
	rollingFloat(dst, a, window, min2Float[T])
}

func rollingMaxFloat[T float](dst, a []T, window int) {
	rollingFloat(dst, a, window, max2Float[T])
}

func rollingFloat[T float](dst, a []T, window int, comb func(T, T) T) {
	for i, out := range rollingWindows(dst, a, window) {
		v := a[i]
		for _, x := range a[i+1 : i+window] {
			v = comb(v, x)
		}
		out[0] = v
	}
}

func rollingMinInt[T integer](dst, a []T, window int) {
	for i, out := range rollingWindows(dst, a, window) {
		v := a[i]
		for _, x := range a[i+1 : i+window] {
			v = min(v, x)
		}
		out[0] = v
	}
}

func rollingMaxInt[T integer](dst, a []T, window int) {
	for i, out := range rollingWindows(dst, a, window) {
		v := a[i]
		for _, x := range a[i+1 : i+window] {
			v = max(v, x)
		}
		out[0] = v
	}
}

// intersectSorted and differenceSorted are the two-cursor merge, which is the
// specification the blocked kernel is checked against.
//
// Both assume a and b are sorted and duplicate-free. That is the caller's
// contract rather than something checked here: verifying it costs a pass over
// both slices, which is what the operation itself costs.
func intersectSorted[T number](dst, a, b []T) int {
	k := 0
	for i, j := 0, 0; i < len(a) && j < len(b) && k < len(dst); {
		switch {
		case a[i] < b[j]:
			i++
		case b[j] < a[i]:
			j++
		default:
			dst[k] = a[i]
			k++
			i++
			j++
		}
	}
	return k
}

func differenceSorted[T number](dst, a, b []T) int {
	k := 0
	i, j := 0, 0
	for i < len(a) && k < len(dst) {
		for j < len(b) && b[j] < a[i] {
			j++
		}
		if j < len(b) && b[j] == a[i] {
			i++
			j++
			continue
		}
		dst[k] = a[i]
		k++
		i++
	}
	return k
}

// rollingWindows iterates the valid output positions, yielding each index and a
// one-element view of dst to write through.
//
// It exists so the four bodies above cannot disagree about the bound. The
// output count is len(a)-window+1, which is not len(dst) and not len(a) — the
// mistake the generated guard's usual min(len(dst), len(a)) would make, and the
// reason the kernel declares two lengths.
func rollingWindows[T any](dst, a []T, window int) func(func(int, []T) bool) {
	return func(yield func(int, []T) bool) {
		if window <= 0 || window > len(a) {
			return
		}
		n := min(len(dst), len(a)-window+1)
		for i := range n {
			if !yield(i, dst[i:i+1]) {
				return
			}
		}
	}
}

// diff writes successive differences: dst[i] = a[i+1] - a[i], producing one
// fewer element than a has. Writing forwards keeps it safe in place.
func diff[T number](dst, a []T) {
	if len(a) < 2 {
		return
	}
	n := min(len(dst), len(a)-1)
	dst = dst[:n]
	for i := range dst {
		dst[i] = a[i+1] - a[i]
	}
}

// ---------- reductions ----------

// prod wraps on integer overflow, like sum, and is left-to-right for floats.
// Unlike summation there is no accumulator tree here: products of
// floating-point values overflow and underflow far more readily, and splitting
// them across lanes changes which intermediate blows up rather than merely
// changing rounding.
func prod[T number](a []T) T {
	p := T(1)
	for _, v := range a {
		p *= v
	}
	return p
}

// ---------- wiring ----------

// floatMathOps fills in the rounding, transcendental and scan portion of a
// float kernel group.
func floatMathOps[T float](o *kernel.Ops[T]) {
	o.Floor = unary[T](math.Floor)
	o.Ceil = unary[T](math.Ceil)
	o.Trunc = unary[T](math.Trunc)
	o.Round = unary[T](math.Round)
	o.RoundToEven = unary[T](math.RoundToEven)

	o.Exp = unary[T](math.Exp)
	o.Exp2 = unary[T](math.Exp2)
	o.Expm1 = unary[T](math.Expm1)
	o.Log = unary[T](math.Log)
	o.Log2 = unary[T](log2Accurate)
	o.Log10 = unary[T](math.Log10)
	o.Log1p = unary[T](math.Log1p)
	o.Cbrt = unary[T](math.Cbrt)
	o.Sigmoid = unary[T](sigmoid)

	o.Sin = unary[T](math.Sin)
	o.Cos = unary[T](math.Cos)
	o.Tan = unary[T](math.Tan)
	o.Asin = unary[T](asinAccurate)
	o.Acos = unary[T](acosAccurate)
	o.Atan = unary[T](math.Atan)
	o.Sinh = unary[T](math.Sinh)
	o.Asinh = unary[T](math.Asinh)
	o.Acosh = unary[T](math.Acosh)
	o.Atanh = unary[T](math.Atanh)
	o.Erf = unary[T](math.Erf)
	o.Erfc = unary[T](math.Erfc)
	o.Cosh = unary[T](math.Cosh)
	o.Tanh = unary[T](math.Tanh)

	o.Pow = binary[T](math.Pow)
	o.Atan2 = binary[T](math.Atan2)
	o.Hypot = binary[T](math.Hypot)

	// The Fast slots are deliberately left nil here. They are filled by
	// FillFastFallbacks once a backend has finished installing its kernels,
	// because until then "no Fast kernel" and "the accurate kernel" cannot be
	// told apart — and the difference matters: the fallback has to be the
	// accurate *kernel*, not the portable loop this file provides.

	o.Lerp = lerp[T]
	o.CumProd = cumProd[T]
	o.CumMin = cumMinFloat[T]
	o.CumMax = cumMaxFloat[T]
	o.RollingMin = rollingMinFloat[T]
	o.RollingMax = rollingMaxFloat[T]
	o.Diff = diff[T]
	o.Prod = prod[T]
}

// fastFrom points every Fast slot that is still nil at its accurate twin.
func fastFrom[T any](o *kernel.Ops[T]) {
	pairs := []struct{ fast, accurate *func(dst, a []T) }{
		{&o.FastExp, &o.Exp}, {&o.FastExp2, &o.Exp2}, {&o.FastExpm1, &o.Expm1},
		{&o.FastLog, &o.Log}, {&o.FastLog2, &o.Log2}, {&o.FastLog10, &o.Log10},
		{&o.FastLog1p, &o.Log1p}, {&o.FastCbrt, &o.Cbrt},
		{&o.FastSigmoid, &o.Sigmoid},
		{&o.FastSin, &o.Sin}, {&o.FastCos, &o.Cos}, {&o.FastTan, &o.Tan},
		{&o.FastAsin, &o.Asin}, {&o.FastAcos, &o.Acos}, {&o.FastAtan, &o.Atan},
		{&o.FastSinh, &o.Sinh}, {&o.FastCosh, &o.Cosh}, {&o.FastTanh, &o.Tanh},
		{&o.FastAsinh, &o.Asinh}, {&o.FastAcosh, &o.Acosh},
		{&o.FastAtanh, &o.Atanh}, {&o.FastErf, &o.Erf}, {&o.FastErfc, &o.Erfc},
	}
	for _, p := range pairs {
		if *p.fast == nil {
			*p.fast = *p.accurate
		}
	}
	bin := []struct{ fast, accurate *func(dst, a, b []T) }{
		{&o.FastPow, &o.Pow}, {&o.FastAtan2, &o.Atan2}, {&o.FastHypot, &o.Hypot},
	}
	for _, p := range bin {
		if *p.fast == nil {
			*p.fast = *p.accurate
		}
	}
}

// FillFastFallbacks points any Fast slot with no generated kernel at the
// accurate one, for both float groups of a backend.
//
// It runs after a backend is fully assembled, which is the only moment the two
// cases can be distinguished, and it is what lets a caller use FastExp on every
// architecture without asking whether that architecture has it. A target where
// the Fast tier did not measure faster simply computes a more accurate answer,
// which an upper bound on error permits.
func FillFastFallbacks(s *kernel.Set) {
	fastFrom(&s.F32)
	fastFrom(&s.F64)
}

// intMathOps fills in the portion an integer kernel group can support.
// Rounding and the transcendentals stay nil; the exported API constrains them
// to floats.
func intMathOps[T integer](o *kernel.Ops[T]) {
	o.Lerp = lerp[T]
	o.CumProd = cumProd[T]
	o.CumMin = cumMinInt[T]
	o.CumMax = cumMaxInt[T]
	o.RollingMin = rollingMinInt[T]
	o.RollingMax = rollingMaxInt[T]
	o.Intersect = intersectSorted[T]
	o.Difference = differenceSorted[T]
	o.Diff = diff[T]
	o.Prod = prod[T]
}

// Exported entry points for generated code.
//
// The threshold guard in each generated backend calls these directly rather
// than reaching through the kernel set, so a short slice costs no indirect
// call through a function-pointer table. They are the same computation the
// closures above perform.

func mapUnary[T float](dst, a []T, f func(float64) float64) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = T(f(float64(a[i])))
	}
}

func mapBinary[T float](dst, a, b []T, f func(x, y float64) float64) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = T(f(float64(a[i]), float64(b[i])))
	}
}

func Exp[T float](dst, a []T)     { mapUnary(dst, a, math.Exp) }
func Exp2[T float](dst, a []T)    { mapUnary(dst, a, math.Exp2) }
func Expm1[T float](dst, a []T)   { mapUnary(dst, a, math.Expm1) }
func Log[T float](dst, a []T)     { mapUnary(dst, a, math.Log) }
func Log2[T float](dst, a []T)    { mapUnary(dst, a, log2Accurate) }
func Log10[T float](dst, a []T)   { mapUnary(dst, a, math.Log10) }
func Log1p[T float](dst, a []T)   { mapUnary(dst, a, math.Log1p) }
func Cbrt[T float](dst, a []T)    { mapUnary(dst, a, math.Cbrt) }
func Sigmoid[T float](dst, a []T) { mapUnary(dst, a, sigmoid) }
func Sin[T float](dst, a []T)     { mapUnary(dst, a, math.Sin) }
func Cos[T float](dst, a []T)     { mapUnary(dst, a, math.Cos) }
func Tan[T float](dst, a []T)     { mapUnary(dst, a, math.Tan) }
func Asin[T float](dst, a []T)    { mapUnary(dst, a, asinAccurate) }
func Acos[T float](dst, a []T)    { mapUnary(dst, a, acosAccurate) }
func Atan[T float](dst, a []T)    { mapUnary(dst, a, math.Atan) }
func Sinh[T float](dst, a []T)    { mapUnary(dst, a, math.Sinh) }
func Asinh[T float](dst, a []T)   { mapUnary(dst, a, math.Asinh) }
func Acosh[T float](dst, a []T)   { mapUnary(dst, a, math.Acosh) }
func Atanh[T float](dst, a []T)   { mapUnary(dst, a, math.Atanh) }
func Erf[T float](dst, a []T)     { mapUnary(dst, a, math.Erf) }
func Erfc[T float](dst, a []T)    { mapUnary(dst, a, math.Erfc) }
func Cosh[T float](dst, a []T)    { mapUnary(dst, a, math.Cosh) }
func Tanh[T float](dst, a []T)    { mapUnary(dst, a, math.Tanh) }

func Pow[T float](dst, a, b []T)   { mapBinary(dst, a, b, math.Pow) }
func Atan2[T float](dst, a, b []T) { mapBinary(dst, a, b, math.Atan2) }
func Hypot[T float](dst, a, b []T) { mapBinary(dst, a, b, math.Hypot) }
