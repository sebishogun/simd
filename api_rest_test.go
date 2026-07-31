package simd

// The public wrappers that the tables in api_test.go do not reach.
//
// api_test.go covers the operations whose shape is "float64 slice in, float64
// slice out", which is most of them. What was left over was 90 exported
// functions that no test called through the public API at all — the complex
// family, the bool masks, the integer-only operations, and a scattering of
// Into forms whose in-place partner was covered but which were not.
//
// The distinction matters because the conformance suite tests kernels, not
// wrappers. A wrapper reaching for the wrong kernel field — MulComplex wired
// to the complex *add*, say — is invisible to every kernel test, because the
// kernel it happens to call is perfectly correct.
//
// Finding this also turned up four operations that had kernels generated,
// wired and conformance-tested with no way to call them at all: FastAsinh,
// FastAcosh, FastAtanh and FastErf. They are exported now.

import (
	"math"
	"math/rand/v2"
	"testing"
)

func rndComplex(n int, r *rand.Rand) []complex128 {
	out := make([]complex128, n)
	for i := range out {
		out[i] = complex(r.NormFloat64()*10, r.NormFloat64()*10)
	}
	return out
}

func complexBitsEqual(t *testing.T, name string, got, want []complex128) {
	t.Helper()
	for i := range got {
		if math.Float64bits(real(got[i])) != math.Float64bits(real(want[i])) ||
			math.Float64bits(imag(got[i])) != math.Float64bits(imag(want[i])) {
			t.Fatalf("%s: at %d got %v want %v", name, i, got[i], want[i])
		}
	}
}

// TestComplexInPlaceMatchesInto is the same identity api_test.go checks for
// the real operations, over the complex ones.
//
// It also checks the arithmetic against Go's own complex operators. That is
// not redundant with the kernel tests: complex multiply is (ac-bd, ad+bc), and
// a wrapper that reached the wrong kernel would produce a perfectly
// self-consistent wrong answer that only a comparison against the definition
// catches.
func TestComplexInPlaceMatchesInto(t *testing.T) {
	r := rand.New(rand.NewPCG(71, 72))

	binary := []struct {
		name    string
		inPlace func(a, b []complex128)
		into    func(dst, a, b []complex128)
		want    func(a, b complex128) complex128
	}{
		{"AddComplex", AddComplex[complex128], AddComplexInto[complex128],
			func(a, b complex128) complex128 { return a + b }},
		{"SubComplex", SubComplex[complex128], SubComplexInto[complex128],
			func(a, b complex128) complex128 { return a - b }},
		// Not `a * b`. Complex multiply is (ac-bd, ad+bc), and Go fuses each
		// multiply into the following add or subtract on arm64, riscv64,
		// loong64 and ppc64 — but not on amd64 — while the kernels compile
		// -ffp-contract=off and never fuse. The explicit conversions force
		// each product to round first, which is what the kernel does.
		//
		// Written as `a * b` this passed on the development machine and failed
		// on arm64 by one bit, which is the second time this exact trap has
		// been walked into; see docs/wrong.md on the LayerNorm and SparseDot
		// cases.
		{"MulComplex", MulComplex[complex128], MulComplexInto[complex128],
			func(a, b complex128) complex128 {
				re := float64(real(a)*real(b)) - float64(imag(a)*imag(b))
				im := float64(real(a)*imag(b)) + float64(imag(a)*real(b))
				return complex(re, im)
			}},
		{"DivComplex", DivComplex[complex128], DivComplexInto[complex128], nil},
	}
	for _, c := range binary {
		for _, n := range apiLens {
			a, b := rndComplex(n, r), rndComplex(n, r)

			inPlace := append([]complex128(nil), a...)
			c.inPlace(inPlace, b)

			dst := make([]complex128, n)
			c.into(dst, a, b)
			complexBitsEqual(t, c.name+" in place vs Into", inPlace, dst)

			// Division is left out of the value check on purpose: Go's / uses
			// Smith's algorithm to avoid overflowing the intermediate, and the
			// kernel does the algebraic form, so they legitimately differ in
			// the last bits. The in-place/Into identity above still holds.
			if c.want == nil {
				continue
			}
			for i := range dst {
				if w := c.want(a[i], b[i]); dst[i] != w {
					t.Fatalf("%s: at %d got %v, want %v from the operator",
						c.name, i, dst[i], w)
				}
			}
		}
	}

	unary := []struct {
		name    string
		inPlace func(a []complex128)
		into    func(dst, a []complex128)
		want    func(complex128) complex128
	}{
		{"ConjComplex", ConjComplex[complex128], ConjComplexInto[complex128],
			func(z complex128) complex128 { return complex(real(z), -imag(z)) }},
		{"NegComplex", NegComplex[complex128], NegComplexInto[complex128],
			func(z complex128) complex128 { return -z }},
	}
	for _, c := range unary {
		for _, n := range apiLens {
			a := rndComplex(n, r)

			inPlace := append([]complex128(nil), a...)
			c.inPlace(inPlace)

			dst := make([]complex128, n)
			c.into(dst, a)
			complexBitsEqual(t, c.name+" in place vs Into", inPlace, dst)

			for i := range dst {
				if w := c.want(a[i]); dst[i] != w {
					t.Fatalf("%s: at %d got %v, want %v", c.name, i, dst[i], w)
				}
			}
		}
	}

	// ScaleComplex takes a real scalar, so it has its own shape.
	for _, n := range apiLens {
		a := rndComplex(n, r)
		inPlace := append([]complex128(nil), a...)
		ScaleComplex(inPlace, 2.5)

		dst := make([]complex128, n)
		ScaleComplexInto(dst, a, 2.5)
		complexBitsEqual(t, "ScaleComplex in place vs Into", inPlace, dst)
	}
}

// The three views of a complex slice have to round-trip. Go stores a complex
// as its two components adjacent in memory, so splitting and rejoining is
// mostly a striding question, and a stride that is wrong by one produces
// plausible nonsense.
func TestComplexPartsRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(73, 74))
	for _, n := range apiLens {
		z := rndComplex(n, r)

		re := make([]float64, n)
		im := make([]float64, n)
		RealInto(re, z)
		ImagInto(im, z)

		for i := range z {
			if re[i] != real(z[i]) || im[i] != imag(z[i]) {
				t.Fatalf("at %d: split gave (%v,%v), want (%v,%v)",
					i, re[i], im[i], real(z[i]), imag(z[i]))
			}
		}

		back := make([]complex128, n)
		FromPartsInto(back, re, im)
		complexBitsEqual(t, "RealInto/ImagInto then FromPartsInto", back, z)
	}
}

// The bool masks are the output of every comparison, so the boolean algebra
// over them is on the path of anything that filters.
func TestMaskOperations(t *testing.T) {
	r := rand.New(rand.NewPCG(75, 76))
	rndMask := func(n int) []bool {
		out := make([]bool, n)
		for i := range out {
			out[i] = r.IntN(2) == 1
		}
		return out
	}
	for _, n := range apiLens {
		a, b := rndMask(n), rndMask(n)

		for _, c := range []struct {
			name string
			op   func(a, b []bool)
			want func(x, y bool) bool
		}{
			{"AndMask", AndMask, func(x, y bool) bool { return x && y }},
			{"OrMask", OrMask, func(x, y bool) bool { return x || y }},
			{"XorMask", XorMask, func(x, y bool) bool { return x != y }},
		} {
			got := append([]bool(nil), a...)
			c.op(got, b)
			for i := range got {
				if w := c.want(a[i], b[i]); got[i] != w {
					t.Fatalf("%s: at %d got %v want %v", c.name, i, got[i], w)
				}
			}
		}

		got := append([]bool(nil), a...)
		NotMask(got)
		for i := range got {
			if got[i] == a[i] {
				t.Fatalf("NotMask: at %d the value did not change", i)
			}
		}
	}
}

// The integer-only operations, against the Go operators they mirror.
//
// Shr and Rotr are here because their left-handed partners were covered and
// they were not, which is exactly the asymmetry a copy-paste introduces.
func TestIntegerWrappers(t *testing.T) {
	r := rand.New(rand.NewPCG(77, 78))
	for _, n := range apiLens {
		a := make([]uint32, n)
		for i := range a {
			a[i] = r.Uint32()
		}

		shr := append([]uint32(nil), a...)
		Shr(shr, 5)
		for i := range shr {
			if w := a[i] >> 5; shr[i] != w {
				t.Fatalf("Shr: at %d got %d want %d", i, shr[i], w)
			}
		}

		rotr := append([]uint32(nil), a...)
		Rotr(rotr, 5)
		for i := range rotr {
			if w := a[i]>>5 | a[i]<<27; rotr[i] != w {
				t.Fatalf("Rotr: at %d got %d want %d", i, rotr[i], w)
			}
		}

		or := append([]uint8(nil), make([]uint8, n)...)
		for i := range or {
			or[i] = uint8(a[i])
		}
		want := append([]uint8(nil), or...)
		rhs := make([]uint8, n)
		for i := range rhs {
			rhs[i] = uint8(r.Uint32())
		}
		Or(or, rhs)
		for i := range or {
			if w := want[i] | rhs[i]; or[i] != w {
				t.Fatalf("Or: at %d got %d want %d", i, or[i], w)
			}
		}

		z := append([]uint32(nil), a...)
		Zero(z)
		for i := range z {
			if z[i] != 0 {
				t.Fatalf("Zero: at %d the value survived", i)
			}
		}
	}

	// Saturating arithmetic clamps rather than wrapping, which is the whole
	// point of it and the thing a plain Add would get wrong.
	hi := []int8{127, 127, -128}
	SatAdd(hi, []int8{1, 100, -100})
	if hi[0] != 127 || hi[1] != 127 || hi[2] != -128 {
		t.Fatalf("SatAdd did not saturate: %v", hi)
	}
	lo := []int8{-128, 127}
	SatSub(lo, []int8{100, -100})
	if lo[0] != -128 || lo[1] != 127 {
		t.Fatalf("SatSub did not saturate: %v", lo)
	}
}

// The statistics and shapes that no other table reaches.
func TestRemainingWrappers(t *testing.T) {
	r := rand.New(rand.NewPCG(79, 80))

	// Covariance of a slice with itself is its population variance, and the
	// sample standard deviation is the population one scaled by n/(n-1) under
	// the square root. Checking the relationships rather than a literal keeps
	// this a test of the wrapper rather than of arithmetic.
	x := rnd(64, r)
	if got, want := Covariance(x, x), Variance(x); math.Abs(got-want) > 1e-9*math.Abs(want) {
		t.Errorf("Covariance(x, x) = %v, want Variance(x) = %v", got, want)
	}
	if got, want := SampleStdDev(x), StdDev(x)*math.Sqrt(64.0/63.0); math.Abs(got-want) > 1e-9*math.Abs(want) {
		t.Errorf("SampleStdDev = %v, want %v", got, want)
	}

	// Lerp lands exactly on the endpoints, which the algebraically equal
	// a*(1-t) + b*t does not always do.
	a, b := []float64{0, 10}, []float64{100, 20}
	end := append([]float64(nil), a...)
	Lerp(end, b, 1)
	if end[0] != 100 || end[1] != 20 {
		t.Errorf("Lerp at t=1 gave %v, want exactly b", end)
	}

	// PolyEval is Horner's method, in place, coefficients from the constant
	// term up. 1 + 2x + 3x^2 at x=2 is 17.
	poly := []float64{2}
	PolyEval(poly, []float64{1, 2, 3})
	if poly[0] != 17 {
		t.Errorf("PolyEval = %v, want 17", poly[0])
	}

	if !ContainsAny("hello world", "xyzw") {
		t.Error(`ContainsAny("hello world", "xyzw") should find the w`)
	}
	if ContainsAny("hello", "xyz") {
		t.Error(`ContainsAny("hello", "xyz") should find nothing`)
	}

	// ClampInto is the Into partner of a covered in-place form.
	src := []float64{-5, 0.5, 99}
	dst := make([]float64, 3)
	ClampInto(dst, src, 0, 1)
	if dst[0] != 0 || dst[1] != 0.5 || dst[2] != 1 {
		t.Errorf("ClampInto gave %v", dst)
	}
	if src[0] != -5 {
		t.Error("ClampInto modified its source")
	}

	// The tier report is the only introspection this library offers, and a
	// caller may well branch on it.
	if Tier() == "" {
		t.Error("Tier() returned an empty string")
	}
	if len(AvailableTiers()) == 0 {
		t.Error("AvailableTiers() returned nothing; scalar is always available")
	}
}

// The last of the exported surface: the uint8 quantization family, the
// remaining Into partners, and the run-start variants for the other two
// element widths.
//
// The uint8 family is the one worth having here for its own sake rather than
// for completeness. Asymmetric quantization uses an unsigned type with a
// non-zero zero point — that is what lets it represent a range that does not
// straddle zero, such as a post-ReLU activation — and the arithmetic differs
// from the int8 path in exactly the place the zero point enters.
func TestRemainingExportedSurface(t *testing.T) {
	// Round-tripping through uint8 with a zero point recovers the input to
	// within the quantization step, and lands exactly on values the grid can
	// represent.
	a := []float32{0, 1, 2, 3}
	q := make([]uint8, len(a))
	QuantizeUint8(q, a, 0.5, 128)
	if q[0] != 128 || q[1] != 130 {
		t.Errorf("QuantizeUint8 gave %v; 0 should land on the zero point", q)
	}
	back := make([]float32, len(a))
	DequantizeUint8(back, q, 0.5, 128)
	for i := range a {
		if back[i] != a[i] {
			t.Errorf("uint8 round trip: at %d got %v want %v", i, back[i], a[i])
		}
	}

	// Per channel, two channels of two values with different scales.
	pa := []float32{1, 2, 30, 40}
	pq := make([]uint8, len(pa))
	QuantizePerChannelUint8(pq, pa, []float32{0.5, 10}, []int32{0, 0}, 2, 2)
	pback := make([]float32, len(pa))
	DequantizePerChannelUint8(pback, pq, []float32{0.5, 10}, []int32{0, 0}, 2, 2)
	for i := range pa {
		if math.Abs(float64(pback[i]-pa[i])) > float64(10) {
			t.Errorf("per-channel uint8 round trip: at %d got %v want near %v",
				i, pback[i], pa[i])
		}
	}

	// The scalar comparisons that produce masks. Each is the negation or
	// mirror of one already covered, which is exactly where a copy-paste
	// between them would land.
	x := []float64{1, 2, 3}
	ge := make([]bool, 3)
	GreaterEqualScalarInto(ge, x, 2)
	lt := make([]bool, 3)
	LessScalarInto(lt, x, 2)
	ne := make([]bool, 3)
	NotEqualScalarInto(ne, x, 2)
	for i := range x {
		if ge[i] == lt[i] {
			t.Errorf(">= and < disagree about being complements at %d", i)
		}
		if want := x[i] != 2; ne[i] != want {
			t.Errorf("NotEqualScalarInto at %d: got %v want %v", i, ne[i], want)
		}
	}

	// The byte Into forms, against their in-place partners.
	src := "Hello, World"
	lower := make([]byte, len(src))
	ToLowerASCIIInto(lower, src)
	if string(lower) != "hello, world" {
		t.Errorf("ToLowerASCIIInto gave %q", lower)
	}
	repl := make([]byte, len(src))
	ReplaceByteInto(repl, src, 'l', 'L')
	if string(repl) != "HeLLo, WorLd" {
		t.Errorf("ReplaceByteInto gave %q", repl)
	}

	// Run starts for the two element widths the int32 form does not cover.
	b8 := make([]bool, 5)
	RunStartsBytesInto(b8, []byte{7, 7, 9, 9, 4})
	b64 := make([]bool, 5)
	RunStartsInt64Into(b64, []int64{7, 7, 9, 9, 4})
	want := []bool{true, false, true, false, true}
	for i := range want {
		if b8[i] != want[i] || b64[i] != want[i] {
			t.Errorf("run starts at %d: byte %v int64 %v, want %v",
				i, b8[i], b64[i], want[i])
		}
	}

	// BottomKInto is the allocation-free BottomK.
	vals := []float64{5, 1, 4, 2, 3}
	dst := make([]float64, 3)
	n := BottomKInto(dst, vals, 3, make([]float64, len(vals)))
	got := append([]float64(nil), dst[:n]...)
	Sort(got)
	if len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Errorf("BottomKInto gave %v, want the three smallest", got)
	}

	// An exponential moving average is a first-order recurrence and therefore
	// cannot vectorize; it is here because it is exported, and because its
	// first element is the seed rather than a smoothed value.
	ema := make([]float64, 4)
	EMAInto(ema, []float64{10, 10, 10, 10}, 0.5)
	if ema[0] != 10 || ema[3] != 10 {
		t.Errorf("EMA of a constant should stay constant, got %v", ema)
	}

	// The inverse transform undoes the forward one.
	p := NewFFTPlan(4)
	sig := []complex128{1, 2, 3, 4}
	spec := make([]complex128, 4)
	FFTInto(p, spec, sig)
	roundTrip := make([]complex128, 4)
	IFFTInto(p, roundTrip, spec)
	for i := range sig {
		if math.Abs(real(roundTrip[i])-real(sig[i])) > 1e-9 {
			t.Errorf("FFT/IFFT round trip at %d: got %v want %v",
				i, roundTrip[i], sig[i])
		}
	}

	// CorrelateFullInto takes scratch because reversing without allocating
	// needs somewhere to put the result.
	ca := []float64{1, 2, 3}
	cb := []float64{1, 1}
	cdst := make([]float64, len(ca)+len(cb)-1)
	CorrelateFullInto(cdst, ca, cb, make([]float64, len(cb)))
	var sum float64
	for _, v := range cdst {
		sum += v
	}
	if sum == 0 {
		t.Error("CorrelateFullInto produced all zeros")
	}
}
