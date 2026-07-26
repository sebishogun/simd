package simd

import "math"

// Numerical routines built from the accelerated primitives: polynomial and
// signal evaluation, quadrature, ODE integration, regression, and the
// activation and normalization functions used in machine learning.
//
// Everything here is chosen because it is dominated by whole-slice arithmetic
// and therefore gets faster for free as backends land. Routines that are
// inherently sequential are marked as such, so you know what you are getting.
//
// Like the rest of the package, none of these allocate. Where a routine needs
// scratch space across many calls — [RK4Step] is the case — it takes a
// workspace you allocate once and reuse.

// ---------- polynomials and signals ----------

// PolyEvalInto evaluates a polynomial at every point of x, writing the results
// to dst. Coefficients are lowest order first, so coeffs[0] is the constant
// term: coeffs = {1, 2, 3} means 1 + 2x + 3x².
//
// It uses Horner's method and reads x once, rather than the pass per
// coefficient that chaining [Mul] and [AddScalar] would cost.
func PolyEvalInto[T Number](dst, x, coeffs []T) { ops[T]().PolyEval(dst, x, coeffs) }

// PolyEval evaluates a polynomial in place, replacing each x with p(x).
// Coefficients are lowest order first.
func PolyEval[T Number](x, coeffs []T) { ops[T]().PolyEval(x, x, coeffs) }

// ConvolveInto writes the discrete convolution of sig with ker.
//
// It produces len(sig)-len(ker)+1 elements: the region where the kernel fully
// overlaps the signal, with no edge padding. Size dst accordingly, or slice it.
func ConvolveInto[T Number](dst, sig, ker []T) { ops[T]().Convolve(dst, sig, ker) }

// CorrelateInto writes the cross-correlation of sig with ker.
//
// This is [ConvolveInto] without reversing the kernel, which is what you want
// for template matching and for filters whose taps are already in signal
// order. It also produces len(sig)-len(ker)+1 elements.
func CorrelateInto[T Number](dst, sig, ker []T) { ops[T]().Correlate(dst, sig, ker) }

// MovingAverageInto writes the mean of each window of the given width,
// producing len(a)-width+1 elements.
//
// Each window is computed independently rather than by sliding a running
// total. That is more arithmetic, but a running total accumulates rounding
// error without bound over a long series and cannot be vectorized, since every
// output would depend on the one before it.
func MovingAverageInto[T Number](dst, a []T, width int) { ops[T]().MovingAverage(dst, a, width) }

// EMAInto writes the exponentially weighted moving average of a, where alpha
// is the weight given to each new sample: dst[i] = alpha*a[i] + (1-alpha)*dst[i-1].
//
// This one is inherently sequential — each output depends on the previous —
// so it does not vectorize and will not get faster as backends land. It is
// here because it belongs next to [MovingAverageInto], not for speed.
func EMAInto[T Number](dst, a []T, alpha T) { ops[T]().EMA(dst, a, alpha) }

// ---------- quadrature ----------

// Trapezoid approximates the integral of a function sampled at n evenly
// spaced points h apart, using the trapezoidal rule.
//
// Its error falls as h². For smooth functions [Simpson] is usually far better
// for the same samples.
func Trapezoid[T Float](y []T, h T) T {
	if len(y) < 2 {
		return 0
	}
	// The rule weights the interior points 1 and the endpoints 1/2, so the
	// whole thing is one Sum with a correction rather than a weighted pass.
	return h * (Sum(y) - (y[0]+y[len(y)-1])/2)
}

// Simpson approximates the integral of a function sampled at n evenly spaced
// points h apart, using Simpson's rule.
//
// It requires an odd number of samples, that is an even number of intervals,
// and returns 0 otherwise. Its error falls as h⁴.
func Simpson[T Float](y []T, h T) T {
	n := len(y)
	if n < 3 || n%2 == 0 {
		return 0
	}
	// Weights run 1, 4, 2, 4, ..., 2, 4, 1. Summing the odd and even indexed
	// interior points separately keeps this to a single pass.
	var odd, even T
	for i := 1; i < n-1; i += 2 {
		odd += y[i]
	}
	for i := 2; i < n-1; i += 2 {
		even += y[i]
	}
	return h / 3 * (y[0] + y[n-1] + 4*odd + 2*even)
}

// ---------- ordinary differential equations ----------
//
// The integrators below advance a whole state vector at once. That is where
// the speed comes from: the derivative function is called a fixed number of
// times per step, and everything between those calls is slice arithmetic.

// Derivative computes dy/dt at time t for state y, writing into dydt.
//
// It must not resize or reallocate dydt, and must write every element.
type Derivative[T Float] func(t T, y, dydt []T)

// EulerStep advances y by one step of size h using the forward Euler method.
//
// It is first order, so its error per step is proportional to h². Use it when
// the derivative is expensive and accuracy is not critical; prefer [RK4Step]
// otherwise.
//
// dydt is scratch the caller owns, the same length as y.
func EulerStep[T Float](y, dydt []T, t, h T, f Derivative[T]) {
	f(t, y, dydt)
	AddScaled(y, dydt, h) // y += dydt*h
}

// RK4Workspace holds the scratch that [RK4Step] needs, so that stepping a
// system forward allocates nothing however many steps you take.
//
// Allocate it once with [NewRK4Workspace] and reuse it for the whole
// integration.
type RK4Workspace[T Float] struct {
	k1, k2, k3, k4, tmp []T
}

// NewRK4Workspace allocates the scratch for integrating a system of n
// equations. This is the only function in the package that allocates, and it
// is called once per system rather than once per step.
func NewRK4Workspace[T Float](n int) *RK4Workspace[T] {
	buf := make([]T, 5*n)
	return &RK4Workspace[T]{
		k1:  buf[0*n : 1*n : 1*n],
		k2:  buf[1*n : 2*n : 2*n],
		k3:  buf[2*n : 3*n : 3*n],
		k4:  buf[3*n : 4*n : 4*n],
		tmp: buf[4*n : 5*n : 5*n],
	}
}

// RK4Step advances y by one step of size h using the classical fourth-order
// Runge-Kutta method, calling f four times.
//
// It is fourth order, so halving the step size cuts the error per step by
// about sixteen. w must have been made by [NewRK4Workspace] for a system at
// least as large as y.
//
//	w := simd.NewRK4Workspace[float64](len(y))
//	for t := 0.0; t < 10; t += h {
//	    simd.RK4Step(y, t, h, deriv, w)
//	}
func RK4Step[T Float](y []T, t, h T, f Derivative[T], w *RK4Workspace[T]) {
	n := len(y)
	k1, k2, k3, k4, tmp := w.k1[:n], w.k2[:n], w.k3[:n], w.k4[:n], w.tmp[:n]
	half := h / 2

	f(t, y, k1)

	copyInto(tmp, y)
	AddScaled(tmp, k1, half) // tmp = y + k1*h/2
	f(t+half, tmp, k2)

	copyInto(tmp, y)
	AddScaled(tmp, k2, half) // tmp = y + k2*h/2
	f(t+half, tmp, k3)

	copyInto(tmp, y)
	AddScaled(tmp, k3, h) // tmp = y + k3*h
	f(t+h, tmp, k4)

	// y += h/6 * (k1 + 2*k2 + 2*k3 + k4), accumulated in k1 to avoid a fifth
	// buffer.
	AddScaled(k1, k2, 2)
	AddScaled(k1, k3, 2)
	Add(k1, k4)
	AddScaled(y, k1, h/6)
}

// VerletStep advances position and velocity by one step of size h using
// velocity Verlet, given the acceleration at the current and next positions.
//
// It is the standard integrator for molecular dynamics and games because it
// conserves energy over long runs far better than Euler or even RK4 does, and
// because it needs only one force evaluation per step.
//
// accel is called with the position to evaluate and the buffer to write the
// acceleration into. acc holds the acceleration at the current position on
// entry and at the new position on return, so pass the same slice back on the
// next step rather than recomputing it.
func VerletStep[T Float](pos, vel, acc []T, h T, accel func(pos, out []T)) {
	// pos += vel*h + acc*h²/2
	AddScaled(pos, vel, h)
	AddScaled(pos, acc, h*h/2)

	// vel += (acc_old + acc_new) * h/2, done in two halves so the old
	// acceleration can be overwritten in place.
	AddScaled(vel, acc, h/2)
	accel(pos, acc)
	AddScaled(vel, acc, h/2)
}

// copyInto copies src into dst without allocating. It exists so the
// integrators read as arithmetic rather than as buffer management.
func copyInto[T Number](dst, src []T) { copy(dst, src) }

// ---------- regression and correlation ----------

// Covariance returns the population covariance of x and y over
// min(len(x), len(y)) elements, or zero for fewer than two.
func Covariance[T Float](x, y []T) T {
	n := min(len(x), len(y))
	if n < 2 {
		return 0
	}
	x, y = x[:n], y[:n]
	mx, my := Mean(x), Mean(y)
	// sum((x-mx)*(y-my)) expands to Dot(x,y) - n*mx*my, and unlike the
	// equivalent trick for variance this form is well conditioned as long as
	// the means are not enormous relative to the spread.
	return (Dot(x, y) - T(n)*mx*my) / T(n)
}

// Correlation returns the Pearson correlation coefficient of x and y, in
// [-1, 1]. It returns zero if either input has no variation, since the
// coefficient is undefined there.
func Correlation[T Float](x, y []T) T {
	sx, sy := StdDev(x), StdDev(y)
	if sx == 0 || sy == 0 {
		return 0
	}
	return Covariance(x, y) / (sx * sy)
}

// LinearRegression fits y = slope*x + intercept by ordinary least squares.
//
// It returns zeros if there are fewer than two points or if every x is the
// same, since no line is determined in either case.
func LinearRegression[T Float](x, y []T) (slope, intercept T) {
	n := min(len(x), len(y))
	if n < 2 {
		return 0, 0
	}
	x, y = x[:n], y[:n]
	mx, my := Mean(x), Mean(y)
	varX := ops[T]().SumSqDev(x, mx)
	if varX == 0 {
		return 0, 0
	}
	slope = (Dot(x, y) - T(n)*mx*my) / varX
	return slope, my - slope*mx
}

// ---------- machine learning ----------

// Softmax converts a to a probability distribution: each element becomes
// exp(a[i]) divided by the sum of all the exponentials, so the result is
// non-negative and sums to one.
//
// The maximum is subtracted before exponentiating. That does not change the
// result mathematically but it is what stops exp overflowing to Inf for large
// inputs, which is the usual way a naive implementation produces NaN.
func Softmax[T Float](a []T) {
	if len(a) == 0 {
		return
	}
	SubScalar(a, Max(a))
	Exp(a)
	if s := Sum(a); s != 0 {
		DivScalar(a, s)
	}
}

// LogSumExp returns log(sum(exp(a))) without overflowing, by factoring out the
// maximum. It is the normalizing constant of [Softmax], and appears wherever
// log-probabilities are combined.
func LogSumExp[T Float](a []T) T {
	if len(a) == 0 {
		return T(math.Inf(-1))
	}
	m := Max(a)
	if math.IsInf(float64(m), -1) {
		return m
	}
	var s T
	for _, v := range a {
		s += T(math.Exp(float64(v - m)))
	}
	return m + T(math.Log(float64(s)))
}

// ReLU clamps every negative element to zero, the rectified linear unit.
//
// It is expressed as a clamp against a scalar rather than an elementwise
// maximum against a zero slice, so it needs no second buffer. NaN propagates.
func ReLU[T Float](a []T) { Clamp(a, 0, T(math.Inf(1))) }

// LeakyReLU scales negative elements by slope instead of zeroing them, which
// keeps a gradient flowing where [ReLU] would stop it.
func LeakyReLU[T Float](a []T, slope T) {
	for i, v := range a {
		if v < 0 {
			a[i] = v * slope
		}
	}
}

// Softplus applies log(1+exp(x)), a smooth approximation to [ReLU].
//
// It is evaluated as max(x,0) + log1p(exp(-|x|)), which is exact for large
// inputs of either sign where the direct form overflows or underflows.
func Softplus[T Float](a []T) {
	for i, v := range a {
		x := float64(v)
		a[i] = T(math.Max(x, 0) + math.Log1p(math.Exp(-math.Abs(x))))
	}
}

// SiLU applies x*sigmoid(x), also called swish.
func SiLU[T Float](a []T) {
	for i, v := range a {
		x := float64(v)
		a[i] = T(x / (1 + math.Exp(-x)))
	}
}

// GELU applies the Gaussian error linear unit, using the tanh approximation
// that transformer implementations conventionally use:
//
//	0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x³)))
//
// The exact form uses the error function; this one is within about 1e-3 of it
// and is what published model weights were trained against, so it is the
// compatible choice rather than merely the fast one.
func GELU[T Float](a []T) {
	const c = 0.7978845608028654 // sqrt(2/pi)
	for i, v := range a {
		x := float64(v)
		a[i] = T(0.5 * x * (1 + math.Tanh(c*(x+0.044715*x*x*x))))
	}
}

// LayerNorm rescales a to zero mean and unit variance, with eps added to the
// variance before the square root to keep the division stable when the input
// is nearly constant.
//
// This is [Standardize] with the epsilon that neural network layers require.
func LayerNorm[T Float](a []T, eps T) {
	if len(a) == 0 {
		return
	}
	m := Mean(a)
	v := ops[T]().SumSqDev(a, m) / T(len(a))
	SubScalar(a, m)
	DivScalar(a, T(math.Sqrt(float64(v+eps))))
}

// RMSNorm divides a by the root mean square of its elements, with eps added
// for stability.
//
// Unlike [LayerNorm] it does not subtract the mean, which makes it cheaper and
// is what several recent transformer architectures use.
func RMSNorm[T Float](a []T, eps T) {
	if len(a) == 0 {
		return
	}
	ms := SumSquares(a) / T(len(a))
	DivScalar(a, T(math.Sqrt(float64(ms+eps))))
}
