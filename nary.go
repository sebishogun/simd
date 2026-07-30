package simd

// Combining several slices at once.
//
// The binary forms already exist, so these look redundant until you count
// memory traffic. `dst = a+b+c+d` written as three calls to AddInto is three
// passes: read two and write one, read that back with a third and write again,
// and once more. The arithmetic is one instruction per element and is not what
// costs; above the last level of cache those extra round trips are the whole
// runtime.
//
// Done in one pass, every source is read once and dst written once.
//
// # What the compiler checks for you
//
// The variadic parameter is `...[]T`, not `...any` and not `[][]T` of some
// looser type. Every slice must have the same element type as dst, and that is
// checked when you compile rather than when you run:
//
//	simd.AddAll(dst, a, b, c)     // all []float64 — fine
//	simd.AddAll(dst, a, ints)     // does not compile
//
// That is the whole of the type safety, and it is deliberate. Lengths are still
// a run-time property in Go — a []T carries its length in its header, not in
// its type — so the length rule is the same as everywhere else in this library:
// the work is bounded by the shortest slice involved.

// # When it actually helps
//
// Measured against the binary calls it replaces, on a Zen 5 with AVX-512, the
// win is not monotonic in size and it is worth knowing where it is not:
//
//	slices × n     n-ary      binary
//	4 × 256        16.2 ns    28.0 ns    n-ary -42%
//	8 × 256        35.9 ns    64.8 ns    n-ary -45%
//	4 × 4096        349 ns     317 ns    binary -9%
//	8 × 4096        729 ns     647 ns    binary -11%
//	4 × 1M          153 µs     217 µs    n-ary -30%
//	8 × 1M          477 µs     650 µs    n-ary -27%
//
// Two different things are being saved at the two ends. Below about a thousand
// elements it is call overhead — one crossing into assembly instead of n-1.
// Above about thirty thousand it is memory traffic, which is the reason these
// exist at all. In between it is neither: the whole working set fits in L2, so
// the repeated binary passes hit cache and cost almost nothing, while a
// four-source kernel is keeping five streams in flight against a much smaller
// L1 and loses about ten percent.
//
// That band is left alone rather than papered over with a cache-size
// heuristic, which would be a guess about the machine rather than a
// measurement of it. If your slices live there and the last ten percent
// matters, write the binary calls out.

// AddAll sets dst[i] to the sum of the corresponding element of every source,
// making a single pass over memory rather than one per source.
//
// The sum is accumulated left to right, so the result is bit-identical to
// writing the binary calls out by hand. Floating-point addition is not
// associative, so that is a real guarantee and not a formality: reordering the
// sources changes the answer, and this function will not reorder them.
//
// The work is bounded by the shortest slice, dst included. With no sources dst
// is zeroed; with one it is copied.
//
// Sources beyond the fourth are folded in groups, so a sixteen-way sum is four
// passes rather than fifteen.
//
// # Why there is no general ZipInto(dst, f, srcs...)
//
// Because it would be slower than the loop you would write without it. A
// closure cannot be vectorized, so the combinator's only advantage is making
// one pass instead of several — and that is not nearly enough. Measured on
// dst = a*b + c, nanoseconds:
//
//	n           your own loop   this catalogue   ZipInto with a closure
//	1024                728.7             52.6                     1163
//	262144             195722            60707                   310461
//	4194304           4377370          3597582                  5446108
//
// The ZipInto column is the *generous* version, specialized to a fixed arity
// so there is no per-element argument slice; the honest variadic form is
// another 1.6x to 2.6x worse again. It loses to a plain Go loop at every size.
//
// So the guidance is: use these, and where your expression is not here, write
// the loop. [AddScaled] covers a*s + b, [MulAll] and [AddAll] cover the n-ary
// products and sums, and a loop you write yourself will beat any closure this
// package could call on your behalf. See entry 47 of docs/wrong.md.
func AddAll[T Number](dst []T, srcs ...[]T) {
	naryFold(dst, srcs, ops[T]().Add3, ops[T]().Add4, ops[T]().Add, zeroValue[T])
}

// MulAll is AddAll with multiplication: dst[i] is the product of the
// corresponding element of every source, left to right, in one pass.
//
// With no sources dst is filled with ones, which is the identity for the
// operation being folded.
func MulAll[T Number](dst []T, srcs ...[]T) {
	naryFold(dst, srcs, ops[T]().Mul3, ops[T]().Mul4, ops[T]().Mul, oneValue[T])
}

func zeroValue[T Number]() T { return 0 }
func oneValue[T Number]() T  { return 1 }

// naryFold is the shared body. It consumes sources four at a time where it can
// and three or two otherwise, accumulating into dst.
//
// The first group writes dst; every later group reads dst back as one of its
// inputs, which is why the fold is not simply len(srcs)/4 independent calls.
// That still leaves the traffic at one pass per four sources rather than one
// per source, which is the point.
func naryFold[T Number](
	dst []T, srcs [][]T,
	f3 func(dst, a, b, c []T),
	f4 func(dst, a, b, c, d []T),
	f2 func(dst, a, b []T),
	identity func() T,
) {
	switch len(srcs) {
	case 0:
		v := identity()
		for i := range dst {
			dst[i] = v
		}
		return
	case 1:
		copy(dst, srcs[0])
		return
	case 2:
		f2(dst, srcs[0], srcs[1])
		return
	case 3:
		f3(dst, srcs[0], srcs[1], srcs[2])
		return
	case 4:
		f4(dst, srcs[0], srcs[1], srcs[2], srcs[3])
		return
	}

	// More than four. Take the first four into dst, then fold the rest three
	// at a time with dst as the fourth input.
	f4(dst, srcs[0], srcs[1], srcs[2], srcs[3])
	rest := srcs[4:]
	for len(rest) >= 3 {
		f4(dst, dst, rest[0], rest[1], rest[2])
		rest = rest[3:]
	}
	switch len(rest) {
	case 2:
		f3(dst, dst, rest[0], rest[1])
	case 1:
		f2(dst, dst, rest[0])
	}
}
