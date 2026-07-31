package simd_test

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// Linear algebra, signal processing and the neural-network pieces.

// TransposeInto is blocked rather than a naive double loop, because the naive
// one strides the whole row length on every element and misses the cache on
// each.
func ExampleTransposeInto() {
	// 2x3 -> 3x2
	a := []float64{1, 2, 3, 4, 5, 6}
	dst := make([]float64, 6)
	simd.TransposeInto(dst, a, 2, 3)
	fmt.Println(dst)
	// Output: [1 4 2 5 3 6]
}

// InterpInto is numpy's interp: piecewise-linear lookup in a table, clamping
// outside it.
func ExampleInterpInto() {
	xp := []float64{0, 10, 20} // table positions
	fp := []float64{0, 100, 0} // table values

	dst := make([]float64, 3)
	simd.InterpInto(dst, []float64{5, 15, 99}, xp, fp)
	fmt.Println(dst)
	// Output: [50 50 0]
}

// ConvolveFullInto picks a direct or FFT convolution by a measured crossover,
// so the caller does not have to know where it is.
func ExampleConvolveFullInto() {
	a := []float64{1, 2, 3}
	b := []float64{1, 1}

	dst := make([]float64, len(a)+len(b)-1)
	simd.ConvolveFullInto(dst, a, b)
	fmt.Println(dst)
	// Output: [1 3 5 3]
}

// A real-input FFT returns n/2+1 complex bins, because the rest are conjugates
// and carry no new information.
func ExampleRFFT() {
	// A constant signal has all its energy in bin 0.
	spectrum := simd.RFFT([]float64{1, 1, 1, 1})
	fmt.Println(len(spectrum), real(spectrum[0]), real(spectrum[1]))
	// Output: 3 4 0
}

// The plan holds the twiddle factors, so a loop over many transforms of the
// same size computes them once.
func ExampleFFTInto() {
	p := simd.NewFFTPlan(4)

	src := []complex128{1, 1, 1, 1}
	dst := make([]complex128, 4)
	simd.FFTInto(p, dst, src)
	fmt.Println(real(dst[0]), real(dst[1]))
	// Output: 4 0
}

// A window is generated once and applied to every frame.
func ExampleHann() {
	w := make([]float64, 4)
	simd.Hann(w)
	fmt.Printf("%.2f %.2f %.2f %.2f\n", w[0], w[1], w[2], w[3])
	// Output: 0.00 0.75 0.75 0.00
}

func ExampleBlackman() {
	w := make([]float64, 3)
	simd.Blackman(w)
	fmt.Printf("%.4f %.4f %.4f\n", w[0], w[1], w[2])
	// Output: -0.0000 1.0000 -0.0000
}

// Hamming is the window function. The bit-counting operation is
// [HammingDistance] — the two are unrelated and the names collide, which is
// why one of them is spelled out.
func ExampleHamming() {
	w := make([]float64, 3)
	simd.Hamming(w)
	fmt.Printf("%.2f %.2f %.2f\n", w[0], w[1], w[2])
	// Output: 0.08 1.00 0.08
}

func ExampleApplyWindowInto() {
	w := make([]float64, 4)
	simd.Hann(w)

	dst := make([]float64, 4)
	simd.ApplyWindowInto(dst, []float64{1, 1, 1, 1}, w)
	fmt.Printf("%.2f %.2f\n", dst[0], dst[1])
	// Output: 0.00 0.75
}

// The analytic signal: the Hilbert transform gives a complex signal whose
// magnitude is the envelope.
func ExampleHilbertInto() {
	src := []float64{0, 1, 0, -1}
	p := simd.NewFFTPlan(len(src))

	analytic := make([]complex128, len(src))
	simd.HilbertInto(p, analytic, src)

	env := make([]float64, len(src))
	simd.AbsComplexInto(env, analytic)
	fmt.Printf("%.2f %.2f %.2f %.2f\n", env[0], env[1], env[2], env[3])
	// Output: 1.00 1.00 1.00 1.00
}

func ExampleAbsComplexInto() {
	dst := make([]float64, 2)
	simd.AbsComplexInto(dst, []complex128{complex(3, 4), complex(0, 1)})
	fmt.Println(dst)
	// Output: [5 1]
}

// LayerNorm centres and scales a vector to zero mean and unit variance — the
// normalization a transformer block does between every sublayer. One pass,
// where Mean followed by StdDev followed by the arithmetic is four.
func ExampleLayerNorm() {
	a := []float64{1, 2, 3, 4}
	simd.LayerNorm(a, 1e-5)
	fmt.Printf("%.3f %.3f %.3f %.3f\n", a[0], a[1], a[2], a[3])
	// Output: -1.342 -0.447 0.447 1.342
}

// The Into form applies the learned gamma and beta in the same pass.
func ExampleLayerNormInto() {
	a := []float64{1, 2, 3, 4}
	gamma := []float64{2, 2, 2, 2}
	beta := []float64{1, 1, 1, 1}

	dst := make([]float64, 4)
	simd.LayerNormInto(dst, a, gamma, beta, 1e-5)
	fmt.Printf("%.3f %.3f\n", dst[0], dst[3])
	// Output: -1.683 3.683
}

// MatMulParallelInto is MatMulInto spread across goroutines. It is opt-in
// because a library that fans out on its own steals cores from a caller that
// may already be using them — reach for it when the multiply is the whole job,
// and stay with MatMulInto when you are running a batch of them in parallel
// yourself.
//
// The result is bit-identical to MatMulInto: the work divides by output row, so
// no element's accumulation order changes. Below a few million
// multiply-accumulates it runs the serial kernel anyway, which is why this
// small example prints the same thing either way.
func ExampleMatMulParallelInto() {
	//  [1 2]   [5 6]   [19 22]
	//  [3 4] * [7 8] = [43 50]
	a := []float64{1, 2, 3, 4} // 2x2, row-major
	b := []float64{5, 6, 7, 8} // 2x2

	dst := make([]float64, 4)
	simd.MatMulParallelInto(dst, a, b, 2, 2, 2)

	fmt.Println(dst)
	// Output: [19 22 43 50]
}

// GemvParallelInto is GemvInto across goroutines, for when one matrix-vector
// product is the whole job. Like MatMulParallelInto it divides by output row,
// so the result is bit-identical to the serial version, and below a few million
// multiply-accumulates it runs the serial kernel anyway.
func ExampleGemvParallelInto() {
	//  [1 2]   [1]   [ 5]
	//  [3 4] * [2] = [11]
	a := []float64{1, 2, 3, 4} // 2x2, row-major
	x := []float64{1, 2}

	dst := make([]float64, 2)
	simd.GemvParallelInto(dst, a, x, 2, 2)

	fmt.Println(dst)
	// Output: [5 11]
}
