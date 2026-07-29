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
