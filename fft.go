package simd

import "math"

// The discrete Fourier transform.
//
// # Shape
//
// [FFTInto] is out of place and takes a plan. The plan holds the twiddle
// factors and the bit-reversal permutation, both of which depend only on the
// length, and computing them costs more than the transform itself — so a
// caller transforming many blocks of the same size builds one plan and reuses
// it. That is the arrangement FFTW, pocketfft and every serious library uses,
// and it is the reason this is not a single function taking a slice.
//
// # Radix-2, and why that is the right first answer
//
// This is iterative Cooley-Tukey, radix-2, for power-of-two lengths. Radix-4
// and split-radix do fewer multiplies — split-radix is about 20% below radix-2
// on operation count — but the gap on a machine with a vector unit is much
// smaller than the operation count suggests, because the butterfly is memory
// bound at the sizes anyone cares about. Radix-2 is also the one whose
// correctness is easy to establish, and an FFT that is 20% off optimal is
// worth far more than no FFT, which is what this package had.
//
// Measured against the definition, float64, median of five:
//
//	n = 64        254ns        60.3us naive      237x
//	n = 256      1.30us         971us naive      746x
//	n = 1024     6.20us        15.2ms naive     2456x
//	n = 4096     29.5us
//	n = 65536     808us
//
// and the plan costs about as much again as the transform it serves — 703ns
// against 254 at n=64, 1.23ms against 808us at 65536 — which is the whole
// reason it is a separate object rather than rebuilt per call.
//
// The inner loop is a complex multiply and two complex adds per butterfly.
// Those are the operations [MulComplexInto] and [AddComplexInto] already
// accelerate, but calling through them per butterfly would cross the assembly
// boundary once per four floats, which the notes on this package's design
// explain is exactly the way to lose. The butterfly is therefore written in Go
// over a contiguous stride, where the Go compiler keeps it in registers, and a
// kernel for the whole stage is left as measured future work.

// FFTPlan holds the tables for transforms of one length. It is safe for
// concurrent use by multiple goroutines, since transforming does not modify
// it.
type FFTPlan struct {
	n       int
	twiddle []complex128 // n/2 factors, exp(-2*pi*i*k/n)
	rev     []int32      // bit-reversal permutation
}

// NewFFTPlan builds a plan for transforms of length n, which must be a power
// of two and at least 1. It returns nil for any other n.
func NewFFTPlan(n int) *FFTPlan {
	if n < 1 || n&(n-1) != 0 {
		return nil
	}
	p := &FFTPlan{n: n, twiddle: make([]complex128, n/2), rev: make([]int32, n)}
	for k := range p.twiddle {
		s, c := math.Sincos(-2 * math.Pi * float64(k) / float64(n))
		p.twiddle[k] = complex(c, s)
	}
	// The bit reversal of i in log2(n) bits, built incrementally: rev[i] is
	// rev[i>>1]>>1 with the low bit of i moved to the top.
	bits := 0
	for 1<<bits < n {
		bits++
	}
	for i := 1; i < n; i++ {
		p.rev[i] = p.rev[i>>1]>>1 | int32((i&1)<<(bits-1))
	}
	return p
}

// Len returns the transform length the plan was built for.
func (p *FFTPlan) Len() int { return p.n }

// FFTInto writes the discrete Fourier transform of src to dst.
//
// dst and src must both be at least p.Len() long and must not overlap. src is
// not modified.
//
// The sign convention is the usual one for a forward transform, exp(-2*pi*i*k*
// n/N), matching numpy.fft and gonum. [IFFTInto] inverts it exactly, including
// the 1/N scaling.
func FFTInto(p *FFTPlan, dst, src []complex128) {
	if p == nil || len(dst) < p.n || len(src) < p.n {
		return
	}
	fftCore(p, dst, src, false)
}

// IFFTInto writes the inverse discrete Fourier transform of src to dst,
// including the 1/N scaling, so IFFTInto(FFTInto(x)) recovers x.
func IFFTInto(p *FFTPlan, dst, src []complex128) {
	if p == nil || len(dst) < p.n || len(src) < p.n {
		return
	}
	fftCore(p, dst, src, true)
}

func fftCore(p *FFTPlan, dst, src []complex128, inverse bool) {
	n := p.n
	dst, src = dst[:n], src[:n]

	// Permute into bit-reversed order, which is what lets the butterflies run
	// over contiguous strides afterwards.
	for i := range n {
		dst[i] = src[p.rev[i]]
	}

	// Stage by stage, doubling the block length. At each stage a block of
	// length m is built from two of length m/2, with the twiddle factor
	// stepping by n/m so one table serves every stage.
	for m := 2; m <= n; m <<= 1 {
		half := m >> 1
		step := n / m
		for start := 0; start < n; start += m {
			k := 0
			for j := start; j < start+half; j++ {
				w := p.twiddle[k]
				if inverse {
					w = complex(real(w), -imag(w))
				}
				t := w * dst[j+half]
				u := dst[j]
				dst[j] = u + t
				dst[j+half] = u - t
				k += step
			}
		}
	}

	if inverse {
		s := complex(1/float64(n), 0)
		for i := range dst {
			dst[i] *= s
		}
	}
}

// FFT returns the discrete Fourier transform of a, allocating both the plan
// and the result. len(a) must be a power of two; it returns nil otherwise.
//
// Use [NewFFTPlan] and [FFTInto] in a loop — building the plan is the
// expensive part and it depends only on the length.
func FFT(a []complex128) []complex128 {
	p := NewFFTPlan(len(a))
	if p == nil {
		return nil
	}
	dst := make([]complex128, len(a))
	FFTInto(p, dst, a)
	return dst
}

// IFFT returns the inverse transform of a, including the 1/N scaling.
func IFFT(a []complex128) []complex128 {
	p := NewFFTPlan(len(a))
	if p == nil {
		return nil
	}
	dst := make([]complex128, len(a))
	IFFTInto(p, dst, a)
	return dst
}

// HilbertInto writes the analytic signal of the real sequence src to dst: a
// complex sequence whose real part is src and whose imaginary part is its
// Hilbert transform.
//
// len(src) must be p.Len(), a power of two, and dst must be at least as long.
//
// The construction is the frequency-domain one, which is why this lives beside
// the FFT rather than in window.go: transform, discard the negative
// frequencies and double the positive ones, transform back. The time-domain
// alternative is convolution with a filter whose ideal impulse response decays
// as 1/n and so has to be truncated, which is both slower and less accurate.
//
// The direct-current and Nyquist bins are left alone rather than doubled. They
// have no negative-frequency partner to fold in — bin 0 and bin n/2 are their
// own conjugates for real input — and doubling them is the classic error,
// showing up as a constant offset and an alternating ripple.
//
// The envelope of a signal is the magnitude of its analytic signal, which is
// what most callers want this for:
//
//	simd.HilbertInto(p, analytic, samples)
//	simd.AbsComplexInto(env, analytic)
func HilbertInto(p *FFTPlan, dst []complex128, src []float64) {
	if p == nil || len(src) < p.n || len(dst) < p.n {
		return
	}
	n := p.n
	for i := range n {
		dst[i] = complex(src[i], 0)
	}
	spec := make([]complex128, n)
	FFTInto(p, spec, dst)

	// Bin 0 and, for even n, bin n/2 keep their value; bins 1..n/2-1 double;
	// the rest are zeroed. For n == 1 there is nothing but the DC bin.
	if n > 1 {
		for k := 1; k < n/2; k++ {
			spec[k] *= 2
		}
		for k := n/2 + 1; k < n; k++ {
			spec[k] = 0
		}
	}
	IFFTInto(p, dst, spec)
}

// Hilbert returns the analytic signal of src, allocating the plan and the
// result. len(src) must be a power of two; it returns nil otherwise.
func Hilbert(src []float64) []complex128 {
	p := NewFFTPlan(len(src))
	if p == nil {
		return nil
	}
	dst := make([]complex128, len(src))
	HilbertInto(p, dst, src)
	return dst
}

// ---------- real input ----------

// RFFTPlan transforms real sequences of a fixed even length.
//
// A real signal's spectrum is conjugate-symmetric, so half of it is redundant
// and half the work is wasted computing it. This does the standard trick: pack
// the n real samples into n/2 complex ones by putting the even-indexed samples
// in the real parts and the odd-indexed in the imaginary parts, run a complex
// transform of half the length, then untangle. The result is the n/2+1
// non-redundant bins, from DC to Nyquist.
//
// Measured against transforming the same real signal as complex, float64,
// median of five:
//
//	n = 1024     5.85us    vs   6.19us      6% faster
//	n = 65536     532us    vs    818us     54% faster
//
// Short of the factor of two the operation count promises, because the
// untangling pass is real work and is not free — at 1024 it very nearly eats
// the whole saving. It also halves the output: n/2+1 bins instead of n, which
// at 65536 is 512 KiB rather than 1 MiB, and that matters more than the time
// for anything that keeps spectra around.
type RFFTPlan struct {
	half  *FFTPlan     // the n/2-point complex plan
	n     int          // the real length
	twist []complex128 // exp(-2*pi*i*k/n) for k in [0, n/2)
	scr   int          // scratch length the caller must supply
}

// NewRFFTPlan builds a plan for real transforms of length n, which must be
// even and n/2 must be a power of two — so n is 2, 4, 8, 16 and so on. It
// returns nil otherwise.
func NewRFFTPlan(n int) *RFFTPlan {
	if n < 2 || n%2 != 0 {
		return nil
	}
	h := NewFFTPlan(n / 2)
	if h == nil {
		return nil
	}
	p := &RFFTPlan{half: h, n: n, twist: make([]complex128, n/2), scr: n / 2}
	for k := range p.twist {
		s, c := math.Sincos(-2 * math.Pi * float64(k) / float64(n))
		p.twist[k] = complex(c, s)
	}
	return p
}

// Len returns the real transform length.
func (p *RFFTPlan) Len() int { return p.n }

// OutLen returns the number of bins RFFTInto writes, which is p.Len()/2+1 —
// DC through Nyquist inclusive. The rest of the spectrum is the conjugate
// mirror of these and is not written.
func (p *RFFTPlan) OutLen() int { return p.n/2 + 1 }

// RFFTInto writes the non-redundant half of the Fourier transform of the real
// sequence src to dst.
//
// src must be at least p.Len() long and dst at least p.OutLen(). scratch must
// be at least p.Len()/2 long; a shorter one is replaced by an allocation.
//
// dst[k] equals the k-th bin of the full complex transform, for k from 0 to
// p.Len()/2. The remaining bins are conj(dst[Len()-k]) and are not written.
func RFFTInto(p *RFFTPlan, dst []complex128, src []float64, scratch []complex128) {
	if p == nil || len(src) < p.n || len(dst) < p.OutLen() {
		return
	}
	h := p.n / 2
	if len(scratch) < h {
		scratch = make([]complex128, h)
	}
	z := scratch[:h]
	// Even samples to the real parts, odd to the imaginary.
	for k := range h {
		z[k] = complex(src[2*k], src[2*k+1])
	}
	Z := dst[:h] // the half transform lands in the first h bins
	FFTInto(p.half, Z, z)

	// Untangle. Ze and Zo are the transforms of the even and odd subsequences,
	// recovered from the conjugate symmetry of the packed transform, and the
	// twiddle recombines them. k and h-k are read together, so the loop walks
	// inward from both ends and each pair is computed before either is
	// overwritten.
	z0 := Z[0]
	for k := 0; k <= h/2; k++ {
		j := h - k
		var zk, zj complex128
		if k == 0 {
			zk, zj = z0, z0
		} else {
			zk, zj = Z[k], Z[j]
		}
		ck := complex(real(zj), -imag(zj))
		cj := complex(real(zk), -imag(zk))

		ek := (zk + ck) / 2
		ok := (zk - ck) / complex(0, 2)
		ej := (zj + cj) / 2
		oj := (zj - cj) / complex(0, 2)

		Z[k] = ek + p.twist[k]*ok
		if j < h {
			Z[j] = ej + p.twist[j]*oj
		}
	}
	// Nyquist. Ze(0) - Zo(0) is the bin at h, and z0 supplies both.
	dst[h] = complex(real(z0)-imag(z0), 0)
	dst[0] = complex(real(z0)+imag(z0), 0)
}

// RFFT returns the non-redundant half of the transform of the real sequence
// src, allocating the plan, the scratch and the result. len(src) must be even
// with len(src)/2 a power of two; it returns nil otherwise.
func RFFT(src []float64) []complex128 {
	p := NewRFFTPlan(len(src))
	if p == nil {
		return nil
	}
	dst := make([]complex128, p.OutLen())
	RFFTInto(p, dst, src, nil)
	return dst
}
