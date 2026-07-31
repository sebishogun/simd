# Signal and matrices

Fourier transforms, windows, convolution, and matrix multiplication. These are
the operations where the library stops being a faster loop and starts being an
algorithm — several of them pick a strategy for you, and it is worth knowing
which one and why.

## Matrix times vector

```go
//  [1 2]   [1]   [ 5]
//  [3 4] * [2] = [11]
a := []float64{1, 2, 3, 4} // row-major, 2x2
x := []float64{1, 2}

dst := make([]float64, 2)
simd.GemvInto(dst, a, x, 2, 2)
```

The matrix is row-major and the dimensions are passed explicitly — there is no
matrix type here, only slices and the two numbers that say how to read them.
That is a deliberate limit: a matrix type would need an opinion about
ownership, striding and views, and this library does not want one.

`GemvInto` reaches about 172 GB/s while the matrix is cache-resident and 49
GB/s at 4096×4096, where it is bound by memory rather than arithmetic. If you
are in the second regime, no amount of vector width helps; the fix is to touch
less memory.

## Matrix times matrix

```go
dst := make([]float64, m*n)
simd.MatMulInto(dst, a, b, m, k, n)
```

Register-blocked rather than a triple loop. The measured difference against the
naive kernel is large enough to quote, along with the number that actually
matters — single-core AVX-512 float32 peak on the machine this was measured on
is about 290 GFLOP/s:

| f32, square | naive | blocked | | GFLOP/s |
|---|---|---|---|---|
| n=64 | 9.51 µs | 2.13 µs | −78% | 246 |
| n=128 | 51.7 µs | 16.9 µs | −67% | 249 |
| n=256 | 331 µs | 129 µs | −61% | **260** |
| n=512 | 3.25 ms | 1.31 ms | −60% | 204 |

At n=256 that is 90% of the machine's theoretical peak, which is about as close
as a portable kernel gets. The drop at 512 is the working set leaving L2.

### When the same B is used repeatedly

Blocking has to rearrange B into a packed layout before it can stream it. If
you multiply many matrices by the *same* B — a batch of inputs through one
weight matrix, which is most of inference — pack it once:

```go
bp := make([]float32, simd.GemmPackLen[float32](k, n))
simd.PackBInto(bp, b, k, n)

for _, a := range batch {
    simd.MatMulIntoPacked(dst, a, bp, m, k, n)
}
```

`MatMulIntoScratch` is the middle ground: it takes the scratch buffer rather
than allocating one internally, for when B changes every time but you still do
not want an allocation per call.

## Transposing

```go
dst := make([]float64, 6)
simd.TransposeInto(dst, a, 2, 3) // 2x3 -> 3x2
```

Blocked, for the reason the naive version is bad rather than for the reason you
might expect: the naive double loop reads down a column, striding the whole row
length on every element, and misses the cache on each one. Working in tiles
means both the read and the write stay within a few cache lines. About 3.6× the
naive loop.

## Fourier transforms

For a one-off, the convenience form allocates and gets on with it:

```go
spectrum := simd.FFT(signal) // []complex128 -> []complex128
```

In a loop, build a plan. The plan holds the twiddle factors — the roots of
unity the transform multiplies by — and computing those is a meaningful
fraction of a small transform:

```go
p := simd.NewFFTPlan(1024)

dst := make([]complex128, 1024)
for _, frame := range frames {
    simd.FFTInto(p, dst, frame)
    // ...
}
```

A plan is bound to one size. `IFFTInto` uses the same plan for the inverse.

### Real input

If your signal is real — audio, sensor data, anything measured — half the
output is redundant. Bins above the midpoint are conjugates of the ones below
and carry no new information:

```go
spectrum := simd.RFFT(signal) // n/2+1 bins, not n
```

```go
p := simd.NewRFFTPlan(n)
dst := make([]complex128, n/2+1)
scratch := make([]complex128, n/2)
simd.RFFTInto(p, dst, signal, scratch)
```

Roughly half the work and half the memory of running a complex FFT over a
signal whose imaginary part you know is zero. Use it whenever the input is
real; there is no downside.

## Windows

A window tapers a frame to zero at both ends, so that treating a finite chunk
of signal as if it repeated forever does not introduce a discontinuity — which
would smear energy across every frequency bin.

```go
const n = 1024

w := make([]float64, n)
simd.Hann(w) // built once, outside the loop

p := simd.NewRFFTPlan(n)
windowed := make([]float64, n)
spectrum := make([]complex128, n/2+1)
scratch := make([]complex128, n/2)

for _, frame := range frames {
    simd.ApplyWindowInto(windowed, frame, w)
    simd.RFFTInto(p, spectrum, windowed, scratch)
    // ...
}
```

That is the whole short-time Fourier transform inner loop, and nothing in it
allocates: the window, the plan and all three buffers are built once.

`Hann`, `Hamming`, `Blackman` and `Bartlett` are available. They differ in how
much they trade main-lobe width for sidelobe suppression, which is a signal
processing decision rather than a performance one.

Two notes on names. `HannPeriodic` is the variant for spectral analysis —
symmetric windows are for filter design, periodic ones for the FFT, and using
the wrong one is a small systematic error rather than an obvious break.

And `Hamming` here is the **window function**. The bit-counting operation is
`HammingDistance`, in [text and bytes](text.md). The two are unrelated; the
collision is real and is why one of them is spelled out.

## Convolution and correlation

```go
dst := make([]float64, len(a)+len(b)-1)
simd.ConvolveFullInto(dst, a, b)
```

This one picks its own algorithm. Direct convolution is O(nm); going through
the FFT is O((n+m) log(n+m)) but with a large constant, so which wins depends
on the kernel length. `ConvolveFullInto` switches at a measured crossover
rather than making you know where it is.

`ConvolveInto` is the "same" mode — output the length of the signal, kernel
centred — which is what you want for filtering. `CorrelateInto` and
`CorrelateFullInto` are the same operation without the kernel reversal, for
template matching and lag estimation. `CorrelateFullInto` takes a scratch
slice, because reversing without allocating needs somewhere to put the result.

`Correlation` is the different thing with the similar name: the Pearson
correlation coefficient of two slices, one number.

## The analytic signal

The Hilbert transform gives a complex signal whose magnitude is the envelope of
the original — the amplitude modulation, with the carrier removed:

```go
p := simd.NewFFTPlan(len(src))

analytic := make([]complex128, len(src))
simd.HilbertInto(p, analytic, src)

env := make([]float64, len(src))
simd.AbsComplexInto(env, analytic)
```

That is the standard way to get an amplitude envelope, and it is three calls
because each of them is useful separately.

## Complex slices

Go stores a complex as its two components adjacent in memory, so a
`[]complex128` is already interleaved real/imaginary pairs — which is the
layout a vector unit wants for arithmetic and the wrong one for anything that
treats the parts separately. Both views are available:

```go
re := make([]float64, len(z))
im := make([]float64, len(z))
simd.RealInto(re, z)
simd.ImagInto(im, z)

simd.FromPartsInto(z, re, im) // and back
```

`DotComplex` is the complex dot product; `DotComplexConj` conjugates the first
argument, which is the one that gives you energy rather than a signed
projection.

## Interpolating a table

```go
xp := []float64{0, 10, 20} // table positions, ascending
fp := []float64{0, 100, 0} // table values

dst := make([]float64, len(x))
simd.InterpInto(dst, x, xp, fp)
```

numpy's `interp`: piecewise-linear lookup, clamping to the end values outside
the table rather than extrapolating. Useful for lookup tables, gamma curves and
resampling.

`PolyEvalInto` is the neighbouring operation — evaluate one polynomial at many
points, by Horner's method, which is both the accurate order and the one that
vectorizes.
