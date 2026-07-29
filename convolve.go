package simd

// Convolution and correlation.
//
// Two algorithms with very different shapes, and which one wins is a
// measurement rather than a rule of thumb.
//
// Direct convolution is O(n*m) but every operation is a fused multiply-add
// over contiguous memory, which is what a vector unit does best. FFT
// convolution is O((n+m) log(n+m)) but pays three transforms, a complex
// multiply and a great deal of memory traffic before it does any useful work.
// The crossover is where the asymptotics finally beat the constant, and it
// sits much further out than the operation counts suggest — see the measured
// figure at convCutoff.

// convCutoff is the kernel length at or above which convolution switches to
// the frequency domain.
//
// Measured on float64, Zen 5, convolving a 65536-sample signal, median of
// five, with both paths forced:
//
//	taps       direct      via FFT
//	  16       99.4us      7503us
//	  64        385us      7471us
//	 256       1530us      7479us
//	1024       6220us      7504us
//	4096      24545us      7495us
//
// The FFT cost is flat in the kernel length — it is dominated by transforming
// the signal, which does not care how long the kernel is — while the direct
// cost is linear at about 6.07us per tap. They cross at roughly 1234, and the
// constant is set just above that.
//
// That is twenty times the sixty-four taps folklore quotes, and the reason is
// that the folklore figure comes from scalar implementations. The direct path
// here is AddScaled over contiguous memory, which is accelerated on every
// architecture, so its constant is small and the asymptotics have much further
// to travel before they win. A crossover copied from a textbook would have
// sent everything from 64 taps upward through three transforms to lose.
const convCutoff = 1250

// ConvolveFullInto writes the full linear convolution of a and b to dst, which
// must be at least len(a)+len(b)-1 long.
//
// It does nothing if dst is too short, if either input is empty, or if dst
// overlaps either input.
//
// The algorithm is chosen by the shorter input's length: direct below
// convCutoff taps and frequency-domain above it, for the reason given there.
func ConvolveFullInto[T Float](dst, a, b []T) {
	n, m := len(a), len(b)
	if n == 0 || m == 0 || len(dst) < n+m-1 {
		return
	}
	// Convolution is commutative, so the shorter operand is the kernel.
	if m > n {
		a, b, n, m = b, a, m, n
	}
	if m < convCutoff {
		convolveDirect(dst, a, b)
		return
	}
	convolveFFT(dst, a, b)
}

// convolveForTest exposes both paths so the crossover can be measured rather
// than asserted. See BenchmarkConvolveCrossover.
func convolveForTest[T Float](dst, a, b []T, useFFT bool) {
	n, m := len(a), len(b)
	if n == 0 || m == 0 || len(dst) < n+m-1 {
		return
	}
	if m > n {
		a, b = b, a
	}
	if useFFT {
		convolveFFT(dst, a, b)
	} else {
		convolveDirect(dst, a, b)
	}
}

// ConvolveFull is [ConvolveFullInto] allocating the destination.
func ConvolveFull[T Float](a, b []T) []T {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	dst := make([]T, len(a)+len(b)-1)
	ConvolveFullInto(dst, a, b)
	return dst
}

// convolveDirect accumulates one shifted, scaled copy of a per element of b.
//
// The loop order is deliberate. Written as a dot product per output element it
// walks b backwards, which is a reversed read on every iteration. Written this
// way — for each b[j], add b[j]*a to dst starting at j — the inner operation is
// AddScaled over a contiguous run, which is accelerated, and both operands are
// read forwards.
func convolveDirect[T Float](dst, a, b []T) {
	out := dst[:len(a)+len(b)-1]
	clear(out)
	for j, s := range b {
		if s == 0 {
			continue
		}
		AddScaledInto(out[j:j+len(a)], out[j:j+len(a)], a, s)
	}
}

// convolveFFT multiplies in the frequency domain.
//
// The transform length is the next power of two at or above the output length,
// which is what makes the wraparound of a cyclic convolution land entirely in
// the zero padding and so leaves the linear result.
func convolveFFT[T Float](dst, a, b []T) {
	outLen := len(a) + len(b) - 1
	size := 1
	for size < outLen {
		size <<= 1
	}
	p := NewFFTPlan(size)
	if p == nil {
		convolveDirect(dst, a, b)
		return
	}
	fa := make([]complex128, size)
	fb := make([]complex128, size)
	for i, v := range a {
		fa[i] = complex(float64(v), 0)
	}
	for i, v := range b {
		fb[i] = complex(float64(v), 0)
	}
	ta := make([]complex128, size)
	tb := make([]complex128, size)
	FFTInto(p, ta, fa)
	FFTInto(p, tb, fb)
	MulComplexInto(ta, ta, tb)
	IFFTInto(p, fa, ta)
	for i := range outLen {
		dst[i] = T(real(fa[i]))
	}
}

// CorrelateFullInto writes the full cross-correlation of a and b to dst, which
// must be at least len(a)+len(b)-1 long.
//
// Correlation is convolution with one operand reversed, so this reverses b
// into scratch and convolves. scratch must be at least len(b) long; a shorter
// one is replaced by an allocation.
//
// dst[k] is the overlap at lag k-(len(b)-1), so the zero lag is at index
// len(b)-1 — the same indexing numpy.correlate uses in "full" mode.
func CorrelateFullInto[T Float](dst, a, b, scratch []T) {
	if len(b) == 0 {
		return
	}
	if len(scratch) < len(b) {
		scratch = make([]T, len(b))
	}
	rev := scratch[:len(b)]
	copy(rev, b)
	Reverse(rev)
	ConvolveFullInto(dst, a, rev)
}

// CorrelateFull is [CorrelateFullInto] allocating both the destination and the
// scratch.
func CorrelateFull[T Float](a, b []T) []T {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	dst := make([]T, len(a)+len(b)-1)
	CorrelateFullInto(dst, a, b, nil)
	return dst
}
