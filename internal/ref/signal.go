package ref

import "github.com/sebishogun/simd/internal/kernel"

// Polynomial and signal kernels.
//
// These are separate kernels rather than compositions of the elementwise ones
// because composing them costs a full pass over memory per coefficient or per
// filter tap. Evaluating a degree-8 polynomial as a chain of Mul and AddScalar
// calls reads and writes the whole slice nine times; done properly it reads it
// once. That difference is the entire reason a fused catalogue exists.

// polyEval evaluates a polynomial at every point of x by Horner's method,
// coefficients lowest order first.
//
// Horner is used rather than accumulating powers of x because it needs one
// multiply and one add per coefficient, and because computing x**k directly
// loses accuracy quickly as k grows.
func polyEval[T number](dst, x, coeffs []T) {
	n := min(len(dst), len(x))
	dst, x = dst[:n], x[:n]
	if len(coeffs) == 0 {
		var zero T
		for i := range dst {
			dst[i] = zero
		}
		return
	}
	last := len(coeffs) - 1
	for i := range dst {
		acc := coeffs[last]
		for k := last - 1; k >= 0; k-- {
			acc = T(acc*x[i]) + coeffs[k]
		}
		dst[i] = acc
	}
}

// convolve is the direct form, kernel reversed: dst[i] = sum_j sig[i+j]*ker[m-1-j].
// It writes len(sig)-len(ker)+1 elements, the "valid" region where the kernel
// fully overlaps the signal.
func convolve[T number](dst, sig, ker []T) {
	m := len(ker)
	if m == 0 || len(sig) < m {
		return
	}
	n := min(len(dst), len(sig)-m+1)
	dst = dst[:n]
	for i := range dst {
		var acc T
		w := sig[i : i+m : i+m]
		for j := range m {
			acc += T(w[j] * ker[m-1-j])
		}
		dst[i] = acc
	}
}

// correlate is convolve without reversing the kernel, which is what you want
// for template matching and for filters whose taps are already in signal
// order.
func correlate[T number](dst, sig, ker []T) {
	m := len(ker)
	if m == 0 || len(sig) < m {
		return
	}
	n := min(len(dst), len(sig)-m+1)
	dst = dst[:n]
	for i := range dst {
		var acc T
		w := sig[i : i+m : i+m]
		for j := range m {
			acc += T(w[j] * ker[j])
		}
		dst[i] = acc
	}
}

// movingAverage writes the mean of each window of the given width, producing
// len(a)-width+1 elements.
//
// It recomputes each window rather than sliding a running total. A running
// total is O(n) instead of O(n*width), but it accumulates rounding error
// without bound over a long series and it cannot be vectorized, since each
// output depends on the previous one. Recomputing keeps every window
// independent, which is both accurate and the shape a vector unit wants.
func movingAverage[T number](dst, a []T, width int) {
	if width <= 0 || len(a) < width {
		return
	}
	n := min(len(dst), len(a)-width+1)
	dst = dst[:n]
	inv := T(1) / T(width)
	_ = inv
	for i := range dst {
		var acc T
		w := a[i : i+width : i+width]
		for j := range width {
			acc += w[j]
		}
		dst[i] = acc / T(width)
	}
}

// ema is the exponentially weighted moving average. It is inherently
// sequential — each output depends on the one before it — so it does not
// vectorize, and it is here for completeness rather than for speed.
func ema[T number](dst, a []T, alpha T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	if n == 0 {
		return
	}
	prev := a[0]
	dst[0] = prev
	for i := 1; i < n; i++ {
		prev = T(alpha*a[i]) + T((1-alpha)*prev)
		dst[i] = prev
	}
}

// signalOps fills in the polynomial and signal portion of a kernel group.
func signalOps[T number](o *kernel.Ops[T]) {
	o.PolyEval = polyEval[T]
	o.Convolve = convolve[T]
	o.Correlate = correlate[T]
	o.MovingAverage = movingAverage[T]
	o.EMA = ema[T]
}
