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
		dst[i] = a[i] + (b[i]-a[i])*t
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
	o.Log2 = unary[T](math.Log2)
	o.Log10 = unary[T](math.Log10)
	o.Log1p = unary[T](math.Log1p)
	o.Cbrt = unary[T](math.Cbrt)
	o.Sigmoid = unary[T](sigmoid)

	o.Sin = unary[T](math.Sin)
	o.Cos = unary[T](math.Cos)
	o.Tan = unary[T](math.Tan)
	o.Asin = unary[T](math.Asin)
	o.Acos = unary[T](math.Acos)
	o.Atan = unary[T](math.Atan)
	o.Sinh = unary[T](math.Sinh)
	o.Cosh = unary[T](math.Cosh)
	o.Tanh = unary[T](math.Tanh)

	o.Pow = binary[T](math.Pow)
	o.Atan2 = binary[T](math.Atan2)
	o.Hypot = binary[T](math.Hypot)

	o.Lerp = lerp[T]
	o.CumProd = cumProd[T]
	o.CumMin = cumMinFloat[T]
	o.CumMax = cumMaxFloat[T]
	o.Diff = diff[T]
	o.Prod = prod[T]
}

// intMathOps fills in the portion an integer kernel group can support.
// Rounding and the transcendentals stay nil; the exported API constrains them
// to floats.
func intMathOps[T integer](o *kernel.Ops[T]) {
	o.Lerp = lerp[T]
	o.CumProd = cumProd[T]
	o.CumMin = cumMinInt[T]
	o.CumMax = cumMaxInt[T]
	o.Diff = diff[T]
	o.Prod = prod[T]
}
