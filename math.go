package simd

// Rounding, transcendental and elementwise math functions.
//
// # Accuracy
//
// Rounding is exact, and therefore bit-identical on every instruction set like
// everything else in this package.
//
// The transcendentals are the one documented exception. They are polynomial
// approximations, and the polynomial that is correct to 1 ULP in float32 is
// not the one correct to 1 ULP in float64, so no implementation could make the
// two agree bit for bit. They instead guarantee a bound: the functions here
// target **1.0 ULP**. Where a cheaper approximation is worth having, it is
// exposed separately under a Fast name and documents its own bound.
//
// Every function comes in two forms, as elsewhere: the plain name rewrites its
// argument in place and allocates nothing, and the Into form takes a
// destination.

// ---------- rounding, in place ----------

// Floor rounds every element down, towards negative infinity.
func Floor[T Float](a []T) { ops[T]().Floor(a, a) }

// Ceil rounds every element up, towards positive infinity.
func Ceil[T Float](a []T) { ops[T]().Ceil(a, a) }

// Trunc rounds every element towards zero, discarding the fractional part.
func Trunc[T Float](a []T) { ops[T]().Trunc(a, a) }

// Round rounds every element to the nearest integer, with halves going away
// from zero. This matches math.Round.
func Round[T Float](a []T) { ops[T]().Round(a, a) }

// RoundToEven rounds every element to the nearest integer, with halves going
// to the even neighbour. This matches math.RoundToEven and is the default
// IEEE 754 rounding mode, which makes it the right choice when rounding
// repeatedly, since it does not accumulate the upward bias that [Round] does.
func RoundToEven[T Float](a []T) { ops[T]().RoundToEven(a, a) }

// ---------- rounding, with a destination ----------

// FloorInto sets dst[i] to a[i] rounded down. dst may alias a.
func FloorInto[T Float](dst, a []T) { ops[T]().Floor(dst, a) }

// CeilInto sets dst[i] to a[i] rounded up. dst may alias a.
func CeilInto[T Float](dst, a []T) { ops[T]().Ceil(dst, a) }

// TruncInto sets dst[i] to a[i] rounded towards zero. dst may alias a.
func TruncInto[T Float](dst, a []T) { ops[T]().Trunc(dst, a) }

// RoundInto sets dst[i] to a[i] rounded to nearest, halves away from zero.
func RoundInto[T Float](dst, a []T) { ops[T]().Round(dst, a) }

// RoundToEvenInto sets dst[i] to a[i] rounded to nearest, halves to even.
func RoundToEvenInto[T Float](dst, a []T) { ops[T]().RoundToEven(dst, a) }

// ---------- exponential and logarithmic, in place ----------

// Exp replaces every element with e raised to it.
func Exp[T Float](a []T) { ops[T]().Exp(a, a) }

// Exp2 replaces every element with 2 raised to it.
func Exp2[T Float](a []T) { ops[T]().Exp2(a, a) }

// Expm1 replaces every element x with e**x - 1.
//
// Use it instead of Exp followed by subtracting one when x is near zero, where
// that form loses almost all its significant digits.
func Expm1[T Float](a []T) { ops[T]().Expm1(a, a) }

// Log replaces every element with its natural logarithm.
//
// Zero yields -Inf and a negative input yields NaN, following IEEE 754.
func Log[T Float](a []T) { ops[T]().Log(a, a) }

// Log2 replaces every element with its base-2 logarithm.
func Log2[T Float](a []T) { ops[T]().Log2(a, a) }

// Log10 replaces every element with its base-10 logarithm.
func Log10[T Float](a []T) { ops[T]().Log10(a, a) }

// Log1p replaces every element x with log(1+x).
//
// Use it instead of adding one and calling Log when x is near zero, where that
// form loses almost all its significant digits.
func Log1p[T Float](a []T) { ops[T]().Log1p(a, a) }

// Cbrt replaces every element with its cube root. Negative inputs are fine.
func Cbrt[T Float](a []T) { ops[T]().Cbrt(a, a) }

// Sigmoid replaces every element x with the logistic function 1/(1+e**-x).
//
// It is evaluated in whichever of the two algebraically equal forms keeps the
// exponent negative, so it does not overflow to NaN for large negative inputs
// the way the naive expression does.
func Sigmoid[T Float](a []T) { ops[T]().Sigmoid(a, a) }

// ---------- trigonometric and hyperbolic, in place ----------

// Sin replaces every element with its sine, in radians.
func Sin[T Float](a []T) { ops[T]().Sin(a, a) }

// Cos replaces every element with its cosine, in radians.
func Cos[T Float](a []T) { ops[T]().Cos(a, a) }

// Tan replaces every element with its tangent, in radians.
func Tan[T Float](a []T) { ops[T]().Tan(a, a) }

// Asin replaces every element with its arcsine, in radians.
// Inputs outside [-1, 1] yield NaN.
func Asin[T Float](a []T) { ops[T]().Asin(a, a) }

// Acos replaces every element with its arccosine, in radians.
// Inputs outside [-1, 1] yield NaN.
func Acos[T Float](a []T) { ops[T]().Acos(a, a) }

// Atan replaces every element with its arctangent, in radians.
func Atan[T Float](a []T) { ops[T]().Atan(a, a) }

// Sinh replaces every element with its hyperbolic sine.
func Sinh[T Float](a []T) { ops[T]().Sinh(a, a) }

// Cosh replaces every element with its hyperbolic cosine.
func Cosh[T Float](a []T) { ops[T]().Cosh(a, a) }

// Tanh replaces every element with its hyperbolic tangent.
func Tanh[T Float](a []T) { ops[T]().Tanh(a, a) }

// ---------- exponential, logarithmic, trigonometric, with a destination ----------

// ExpInto sets dst[i] = e**a[i]. dst may alias a.
func ExpInto[T Float](dst, a []T) { ops[T]().Exp(dst, a) }

// Exp2Into sets dst[i] = 2**a[i]. dst may alias a.
func Exp2Into[T Float](dst, a []T) { ops[T]().Exp2(dst, a) }

// Expm1Into sets dst[i] = e**a[i] - 1, accurately near zero. dst may alias a.
func Expm1Into[T Float](dst, a []T) { ops[T]().Expm1(dst, a) }

// LogInto sets dst[i] to the natural logarithm of a[i]. dst may alias a.
func LogInto[T Float](dst, a []T) { ops[T]().Log(dst, a) }

// Log2Into sets dst[i] to the base-2 logarithm of a[i]. dst may alias a.
func Log2Into[T Float](dst, a []T) { ops[T]().Log2(dst, a) }

// Log10Into sets dst[i] to the base-10 logarithm of a[i]. dst may alias a.
func Log10Into[T Float](dst, a []T) { ops[T]().Log10(dst, a) }

// Log1pInto sets dst[i] = log(1+a[i]), accurately near zero. dst may alias a.
func Log1pInto[T Float](dst, a []T) { ops[T]().Log1p(dst, a) }

// CbrtInto sets dst[i] to the cube root of a[i]. dst may alias a.
func CbrtInto[T Float](dst, a []T) { ops[T]().Cbrt(dst, a) }

// SigmoidInto sets dst[i] = 1/(1+e**-a[i]). dst may alias a.
func SigmoidInto[T Float](dst, a []T) { ops[T]().Sigmoid(dst, a) }

// SinInto sets dst[i] to the sine of a[i]. dst may alias a.
func SinInto[T Float](dst, a []T) { ops[T]().Sin(dst, a) }

// CosInto sets dst[i] to the cosine of a[i]. dst may alias a.
func CosInto[T Float](dst, a []T) { ops[T]().Cos(dst, a) }

// TanInto sets dst[i] to the tangent of a[i]. dst may alias a.
func TanInto[T Float](dst, a []T) { ops[T]().Tan(dst, a) }

// AsinInto sets dst[i] to the arcsine of a[i]. dst may alias a.
func AsinInto[T Float](dst, a []T) { ops[T]().Asin(dst, a) }

// AcosInto sets dst[i] to the arccosine of a[i]. dst may alias a.
func AcosInto[T Float](dst, a []T) { ops[T]().Acos(dst, a) }

// AtanInto sets dst[i] to the arctangent of a[i]. dst may alias a.
func AtanInto[T Float](dst, a []T) { ops[T]().Atan(dst, a) }

// SinhInto sets dst[i] to the hyperbolic sine of a[i]. dst may alias a.
func SinhInto[T Float](dst, a []T) { ops[T]().Sinh(dst, a) }

// CoshInto sets dst[i] to the hyperbolic cosine of a[i]. dst may alias a.
func CoshInto[T Float](dst, a []T) { ops[T]().Cosh(dst, a) }

// TanhInto sets dst[i] to the hyperbolic tangent of a[i]. dst may alias a.
func TanhInto[T Float](dst, a []T) { ops[T]().Tanh(dst, a) }

// ---------- two-argument math ----------

// Pow raises each element of a to the corresponding power in b: a[i] **= b[i].
func Pow[T Float](a, b []T) { ops[T]().Pow(a, a, b) }

// PowInto sets dst[i] = a[i] ** b[i]. dst may alias a or b.
func PowInto[T Float](dst, a, b []T) { ops[T]().Pow(dst, a, b) }

// Atan2 replaces each element of a with the angle, in radians, of the point
// (b[i], a[i]) — that is, atan(a[i]/b[i]) resolved to the correct quadrant.
//
// The argument order follows math.Atan2: y first, then x.
func Atan2[T Float](a, b []T) { ops[T]().Atan2(a, a, b) }

// Atan2Into sets dst[i] to the angle of the point (b[i], a[i]) in radians.
func Atan2Into[T Float](dst, a, b []T) { ops[T]().Atan2(dst, a, b) }

// Hypot replaces each element of a with sqrt(a[i]**2 + b[i]**2), computed
// without the intermediate overflow that squaring would cause.
func Hypot[T Float](a, b []T) { ops[T]().Hypot(a, a, b) }

// HypotInto sets dst[i] = sqrt(a[i]**2 + b[i]**2), avoiding overflow.
func HypotInto[T Float](dst, a, b []T) { ops[T]().Hypot(dst, a, b) }

// ---------- interpolation ----------

// Lerp interpolates each element of a towards b by t: a[i] += (b[i]-a[i]) * t.
//
// t is not clamped, so values outside [0, 1] extrapolate. The evaluation form
// is monotonic in t and lands exactly on b at t=1, which the algebraically
// equal a*(1-t) + b*t does not.
func Lerp[T Number](a, b []T, t T) { ops[T]().Lerp(a, a, b, t) }

// LerpInto sets dst[i] = a[i] + (b[i]-a[i])*t. dst may alias a or b.
func LerpInto[T Number](dst, a, b []T, t T) { ops[T]().Lerp(dst, a, b, t) }
