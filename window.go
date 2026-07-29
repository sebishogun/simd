package simd

import "math"

// Window functions, and a note on the rest of classical DSP.
//
// These are elementwise functions of the sample index, so they compose out of
// primitives this package already accelerates: [Ramp] fills the index, [Cos]
// transforms it, and the rest is scalar arithmetic over slices. None of them
// needs a kernel of its own, and each is accelerated wherever Cos is.
//
// The convention is the symmetric one, dividing by N-1, which is what
// scipy.signal.get_window(..., fftbins=False) and MATLAB give and what you
// want for filter design. The periodic form, dividing by N, is what you want
// for spectral analysis; it is the symmetric window of length N+1 with the
// last sample dropped, and [HannPeriodic] is provided because that difference
// is a common and silent source of scalloping error.
//
// # Why there is no biquad or IIR filter here
//
// A biquad is y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2].
// Every output depends on the two before it, so the recurrence is serial
// through its own output — the same reason [EMAInto] and [CumSum] are
// permanently portable, and the same contract forbids the reassociation that
// would break it. A block-parallel IIR exists (the state-space form raised to
// a power) but it changes the arithmetic and so the answer, which puts it
// behind the Fast* convention rather than in the ordinary API.
//
// FIR filtering is the opposite case and is not serial at all: it is a
// correlation, which is [DotInto] over a sliding window, or a multiply in the
// frequency domain using [FFTInto] once the transform is worth its cost.

// window fills dst using a raised-cosine of the given coefficients, evaluated
// on the symmetric grid. It is the shared body of Hann, Hamming and Blackman.
func window[T Float](dst []T, a0, a1, a2 T, denom float64) {
	n := len(dst)
	if n == 0 {
		return
	}
	if n == 1 {
		dst[0] = 1
		return
	}
	// The index ramp and the cosine are both accelerated; the combination is
	// two more elementwise passes.
	step := T(2 * math.Pi / denom)
	Ramp(dst, 0, step)
	Cos(dst)
	// a0 - a1*cos(x) + a2*cos(2x). cos(2x) = 2cos^2(x) - 1, so the second
	// harmonic needs no second transcendental pass — one multiply and one
	// add, which is much cheaper than another Cos over the slice.
	for i := range dst {
		c := dst[i]
		dst[i] = a0 - a1*c + a2*(2*c*c-1)
	}
}

// Hann fills dst with a symmetric Hann window.
func Hann[T Float](dst []T) { window(dst, 0.5, 0.5, 0, float64(len(dst)-1)) }

// HannPeriodic fills dst with a periodic Hann window, which is the symmetric
// window of length len(dst)+1 with its last sample dropped. This is the form
// spectral analysis wants; [Hann] is the form filter design wants.
func HannPeriodic[T Float](dst []T) { window(dst, 0.5, 0.5, 0, float64(len(dst))) }

// Hamming fills dst with a symmetric Hamming window.
func Hamming[T Float](dst []T) { window(dst, 0.54, 0.46, 0, float64(len(dst)-1)) }

// Blackman fills dst with a symmetric Blackman window.
func Blackman[T Float](dst []T) {
	window(dst, 0.42, 0.5, 0.08, float64(len(dst)-1))
}

// Bartlett fills dst with a symmetric Bartlett (triangular) window.
//
// This one is not a raised cosine, so it is a ramp and an absolute value
// rather than a transcendental — much the cheapest of the set.
func Bartlett[T Float](dst []T) {
	n := len(dst)
	if n == 0 {
		return
	}
	if n == 1 {
		dst[0] = 1
		return
	}
	half := T(float64(n-1) / 2)
	Ramp(dst, -half, 1)
	Abs(dst)
	// 1 - |i - (n-1)/2| / ((n-1)/2)
	Scale(dst, -1/half)
	AddScalar(dst, 1)
}

// ApplyWindowInto multiplies a by a window into dst, which is what a caller
// actually does with one:
//
//	w := make([]float64, n)
//	simd.Hann(w)
//	simd.ApplyWindowInto(dst, samples, w)
//
// It is [MulInto] under a clearer name, and exists so the pairing is
// discoverable from the window functions.
func ApplyWindowInto[T Float](dst, a, w []T) { MulInto(dst, a, w) }
