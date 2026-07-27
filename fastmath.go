package simd

// The Fast tier: the same transcendentals, cheaper and less exact.
//
// Every function here computes what its accurate twin computes and guarantees
// a looser bound: **3.5 ULP** instead of 1.0. Use them where the input is
// already approximate — activations in a neural network, a colour transform,
// a physical simulation whose state carries more error than this does — and
// use the accurate ones anywhere a result is compared, hashed, or stored as a
// key.
//
// Two things they give up, and it is worth being exact about which.
//
// Accuracy, by a stated amount: 3.5 ULP is the bound, measured against the
// standard library by TestFastTranscendentalULP and reported there rather than
// asserted from theory.
//
// And reproducibility across architectures. These are compiled with fused
// multiply-add enabled, so a machine with an FMA gives a different answer from
// one without — which is precisely the promise every other function in this
// package keeps and this one does not. That is the whole reason for the
// separate name.
//
// What they do *not* give up is meaning. NaN in gives NaN out, the infinities
// go where IEEE 754 says, and the signed zeros survive. -ffast-math would buy
// more and is refused: it makes a function wrong on those inputs rather than
// merely less precise, which is the mistake this package exists partly to
// avoid.
//
// On an architecture where the Fast tier did not measure faster, these call
// the accurate kernel instead. A bound is an upper bound, so a more accurate
// answer satisfies it, and a caller never has to ask which architecture it is
// on. See kernel.Ops and ref.FillFastFallbacks.

// FastExp replaces every element with e raised to it, to within 3.5 ULP.
// See [Exp] for the accurate form.
func FastExp[T Float](a []T) { ops[T]().FastExp(a, a) }

// FastExpInto writes the result into dst. dst may alias a.
func FastExpInto[T Float](dst, a []T) { ops[T]().FastExp(dst, a) }

// FastExp2 replaces every element with 2 raised to it, to within 3.5 ULP.
// See [Exp2] for the accurate form.
func FastExp2[T Float](a []T) { ops[T]().FastExp2(a, a) }

// FastExp2Into writes the result into dst. dst may alias a.
func FastExp2Into[T Float](dst, a []T) { ops[T]().FastExp2(dst, a) }

// FastExpm1 replaces every element with e raised to it, minus one, to within 3.5 ULP.
// See [Expm1] for the accurate form.
func FastExpm1[T Float](a []T) { ops[T]().FastExpm1(a, a) }

// FastExpm1Into writes the result into dst. dst may alias a.
func FastExpm1Into[T Float](dst, a []T) { ops[T]().FastExpm1(dst, a) }

// FastLog replaces every element with its natural logarithm, to within 3.5 ULP.
// See [Log] for the accurate form.
func FastLog[T Float](a []T) { ops[T]().FastLog(a, a) }

// FastLogInto writes the result into dst. dst may alias a.
func FastLogInto[T Float](dst, a []T) { ops[T]().FastLog(dst, a) }

// FastLog2 replaces every element with its base-2 logarithm, to within 3.5 ULP.
// See [Log2] for the accurate form.
func FastLog2[T Float](a []T) { ops[T]().FastLog2(a, a) }

// FastLog2Into writes the result into dst. dst may alias a.
func FastLog2Into[T Float](dst, a []T) { ops[T]().FastLog2(dst, a) }

// FastLog10 replaces every element with its base-10 logarithm, to within 3.5 ULP.
// See [Log10] for the accurate form.
func FastLog10[T Float](a []T) { ops[T]().FastLog10(a, a) }

// FastLog10Into writes the result into dst. dst may alias a.
func FastLog10Into[T Float](dst, a []T) { ops[T]().FastLog10(dst, a) }

// FastLog1p replaces every element with the natural logarithm of one plus it, to within 3.5 ULP.
// See [Log1p] for the accurate form.
func FastLog1p[T Float](a []T) { ops[T]().FastLog1p(a, a) }

// FastLog1pInto writes the result into dst. dst may alias a.
func FastLog1pInto[T Float](dst, a []T) { ops[T]().FastLog1p(dst, a) }

// FastCbrt replaces every element with its cube root, to within 3.5 ULP.
// See [Cbrt] for the accurate form.
func FastCbrt[T Float](a []T) { ops[T]().FastCbrt(a, a) }

// FastCbrtInto writes the result into dst. dst may alias a.
func FastCbrtInto[T Float](dst, a []T) { ops[T]().FastCbrt(dst, a) }

// FastSigmoid replaces every element with its logistic sigmoid, to within 3.5 ULP.
// See [Sigmoid] for the accurate form.
func FastSigmoid[T Float](a []T) { ops[T]().FastSigmoid(a, a) }

// FastSigmoidInto writes the result into dst. dst may alias a.
func FastSigmoidInto[T Float](dst, a []T) { ops[T]().FastSigmoid(dst, a) }

// FastSin replaces every element with its sine, to within 3.5 ULP.
// See [Sin] for the accurate form.
func FastSin[T Float](a []T) { ops[T]().FastSin(a, a) }

// FastSinInto writes the result into dst. dst may alias a.
func FastSinInto[T Float](dst, a []T) { ops[T]().FastSin(dst, a) }

// FastCos replaces every element with its cosine, to within 3.5 ULP.
// See [Cos] for the accurate form.
func FastCos[T Float](a []T) { ops[T]().FastCos(a, a) }

// FastCosInto writes the result into dst. dst may alias a.
func FastCosInto[T Float](dst, a []T) { ops[T]().FastCos(dst, a) }

// FastTan replaces every element with its tangent, to within 3.5 ULP.
// See [Tan] for the accurate form.
func FastTan[T Float](a []T) { ops[T]().FastTan(a, a) }

// FastTanInto writes the result into dst. dst may alias a.
func FastTanInto[T Float](dst, a []T) { ops[T]().FastTan(dst, a) }

// FastAsin replaces every element with its arcsine, to within 3.5 ULP.
// See [Asin] for the accurate form.
func FastAsin[T Float](a []T) { ops[T]().FastAsin(a, a) }

// FastAsinInto writes the result into dst. dst may alias a.
func FastAsinInto[T Float](dst, a []T) { ops[T]().FastAsin(dst, a) }

// FastAcos replaces every element with its arccosine, to within 3.5 ULP.
// See [Acos] for the accurate form.
func FastAcos[T Float](a []T) { ops[T]().FastAcos(a, a) }

// FastAcosInto writes the result into dst. dst may alias a.
func FastAcosInto[T Float](dst, a []T) { ops[T]().FastAcos(dst, a) }

// FastAtan replaces every element with its arctangent, to within 3.5 ULP.
// See [Atan] for the accurate form.
func FastAtan[T Float](a []T) { ops[T]().FastAtan(a, a) }

// FastAtanInto writes the result into dst. dst may alias a.
func FastAtanInto[T Float](dst, a []T) { ops[T]().FastAtan(dst, a) }

// FastSinh replaces every element with its hyperbolic sine, to within 3.5 ULP.
// See [Sinh] for the accurate form.
func FastSinh[T Float](a []T) { ops[T]().FastSinh(a, a) }

// FastSinhInto writes the result into dst. dst may alias a.
func FastSinhInto[T Float](dst, a []T) { ops[T]().FastSinh(dst, a) }

// FastCosh replaces every element with its hyperbolic cosine, to within 3.5 ULP.
// See [Cosh] for the accurate form.
func FastCosh[T Float](a []T) { ops[T]().FastCosh(a, a) }

// FastCoshInto writes the result into dst. dst may alias a.
func FastCoshInto[T Float](dst, a []T) { ops[T]().FastCosh(dst, a) }

// FastTanh replaces every element with its hyperbolic tangent, to within 3.5 ULP.
// See [Tanh] for the accurate form.
func FastTanh[T Float](a []T) { ops[T]().FastTanh(a, a) }

// FastTanhInto writes the result into dst. dst may alias a.
func FastTanhInto[T Float](dst, a []T) { ops[T]().FastTanh(dst, a) }

// FastPow sets a[i] to a[i] raised to the power b[i], to within 3.5 ULP.
// See [Pow] for the accurate form.
func FastPow[T Float](a, b []T) { ops[T]().FastPow(a, a, b) }

// FastPowInto writes the result into dst. dst may alias a or b.
func FastPowInto[T Float](dst, a, b []T) { ops[T]().FastPow(dst, a, b) }

// FastAtan2 sets a[i] to atan(a[i]/b[i]) in the correct quadrant, to within 3.5 ULP.
// See [Atan2] for the accurate form.
func FastAtan2[T Float](a, b []T) { ops[T]().FastAtan2(a, a, b) }

// FastAtan2Into writes the result into dst. dst may alias a or b.
func FastAtan2Into[T Float](dst, a, b []T) { ops[T]().FastAtan2(dst, a, b) }

// FastHypot sets a[i] to the length of the hypotenuse with sides a[i] and b[i], to within 3.5 ULP.
// See [Hypot] for the accurate form.
func FastHypot[T Float](a, b []T) { ops[T]().FastHypot(a, a, b) }

// FastHypotInto writes the result into dst. dst may alias a or b.
func FastHypotInto[T Float](dst, a, b []T) { ops[T]().FastHypot(dst, a, b) }
