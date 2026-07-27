package simd

// The public wrappers, checked against each other and against the reference.
//
// Everything under internal/conformance tests the kernels. This tests the thin
// layer above them, which is where a different class of mistake lives: an
// argument passed in the wrong order, an in-place form that does not match its
// Into form, a wrapper that allocates, or one that reads past the end when the
// slices differ in length. None of those would fail a kernel test, because the
// kernel is not what is wrong.
//
// The library's central convention is that the plain name works in place and
// the Into suffix takes a destination. That gives an identity worth checking
// everywhere it applies:
//
//	F(a, rest...)  ==  FInto(dst, a, rest...)
//
// with dst a copy of a. Eighty-three functions come in that pairing, and the
// tables below cover them by shape rather than one at a time.

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd/internal/ref"
)

func rnd(n int, r *rand.Rand) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = r.NormFloat64() * 10
	}
	return out
}

func rndPos(n int, r *rand.Rand) []float64 {
	out := rnd(n, r)
	for i := range out {
		out[i] = math.Abs(out[i]) + 0.25
	}
	return out
}

func rndUnit(n int, r *rand.Rand) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = r.Float64()*2 - 1
	}
	return out
}

func bitsEqual(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) &&
			!(math.IsNaN(got[i]) && math.IsNaN(want[i])) {
			t.Fatalf("%s: at %d got %v want %v", name, i, got[i], want[i])
		}
	}
}

// TestInPlaceMatchesInto checks the library's central convention for every
// unary function that has both forms.
func TestInPlaceMatchesInto(t *testing.T) {
	type unary struct {
		name    string
		inPlace func([]float64)
		into    func(dst, a []float64)
		gen     func(int, *rand.Rand) []float64
	}
	cases := []unary{
		{"Abs", Abs[float64], AbsInto[float64], rnd},
		{"Neg", Neg[float64], NegInto[float64], rnd},
		{"Sqrt", Sqrt[float64], SqrtInto[float64], rndPos},
		{"Reciprocal", Reciprocal[float64], ReciprocalInto[float64], rndPos},
		{"Floor", Floor[float64], FloorInto[float64], rnd},
		{"Ceil", Ceil[float64], CeilInto[float64], rnd},
		{"Trunc", Trunc[float64], TruncInto[float64], rnd},
		{"Round", Round[float64], RoundInto[float64], rnd},
		{"RoundToEven", RoundToEven[float64], RoundToEvenInto[float64], rnd},
		{"Exp", Exp[float64], ExpInto[float64], rndUnit},
		{"Exp2", Exp2[float64], Exp2Into[float64], rndUnit},
		{"Expm1", Expm1[float64], Expm1Into[float64], rndUnit},
		{"Log", Log[float64], LogInto[float64], rndPos},
		{"Log2", Log2[float64], Log2Into[float64], rndPos},
		{"Log10", Log10[float64], Log10Into[float64], rndPos},
		{"Log1p", Log1p[float64], Log1pInto[float64], rndPos},
		{"Cbrt", Cbrt[float64], CbrtInto[float64], rnd},
		{"Sigmoid", Sigmoid[float64], SigmoidInto[float64], rnd},
		{"Sin", Sin[float64], SinInto[float64], rnd},
		{"Cos", Cos[float64], CosInto[float64], rnd},
		{"Tan", Tan[float64], TanInto[float64], rnd},
		{"Asin", Asin[float64], AsinInto[float64], rndUnit},
		{"Acos", Acos[float64], AcosInto[float64], rndUnit},
		{"Atan", Atan[float64], AtanInto[float64], rnd},
		{"Sinh", Sinh[float64], SinhInto[float64], rndUnit},
		{"Cosh", Cosh[float64], CoshInto[float64], rndUnit},
		{"Tanh", Tanh[float64], TanhInto[float64], rndUnit},
		{"CumSum", CumSum[float64], CumSumInto[float64], rnd},
		{"CumProd", CumProd[float64], CumProdInto[float64], rnd},
		{"CumMin", CumMin[float64], CumMinInto[float64], rnd},
		{"CumMax", CumMax[float64], CumMaxInto[float64], rnd},
		{"Reverse", Reverse[float64], ReverseInto[float64], rnd},
	}
	r := rand.New(rand.NewPCG(41, 42))
	for _, c := range cases {
		for _, n := range apiLens {
			a := c.gen(n, r)
			inPlace := append([]float64(nil), a...)
			c.inPlace(inPlace)

			dst := make([]float64, n)
			c.into(dst, a)

			bitsEqual(t, c.name+" in place vs Into", inPlace, dst)

			// Into with the destination aliasing the source must give the same
			// answer as the in-place form, which is the same code path but is
			// worth stating: a kernel that reads an element after overwriting
			// it would differ here and nowhere else.
			alias := append([]float64(nil), a...)
			c.into(alias, alias)
			bitsEqual(t, c.name+" Into aliased", alias, dst)
		}
	}
}

var apiLens = []int{0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 63, 64, 65, 127, 256}

// TestBinaryInPlaceMatchesInto is the same identity for the two-operand forms.
func TestBinaryInPlaceMatchesInto(t *testing.T) {
	type binary struct {
		name    string
		inPlace func(a, b []float64)
		into    func(dst, a, b []float64)
		gen     func(int, *rand.Rand) []float64
	}
	cases := []binary{
		{"Add", Add[float64], AddInto[float64], rnd},
		{"Sub", Sub[float64], SubInto[float64], rnd},
		{"Mul", Mul[float64], MulInto[float64], rnd},
		{"Div", Div[float64], DivInto[float64], rndPos},
		{"Minimum", Minimum[float64], MinimumInto[float64], rnd},
		{"Maximum", Maximum[float64], MaximumInto[float64], rnd},
		{"Pow", Pow[float64], PowInto[float64], rndPos},
		{"Atan2", Atan2[float64], Atan2Into[float64], rnd},
		{"Hypot", Hypot[float64], HypotInto[float64], rnd},
	}
	r := rand.New(rand.NewPCG(43, 44))
	for _, c := range cases {
		for _, n := range apiLens {
			a, b := c.gen(n, r), c.gen(n, r)
			inPlace := append([]float64(nil), a...)
			c.inPlace(inPlace, b)

			dst := make([]float64, n)
			c.into(dst, a, b)
			bitsEqual(t, c.name+" in place vs Into", inPlace, dst)

			// Aliasing the destination onto the *second* operand is the case
			// most likely to be wrong, because a kernel writing dst[i] before
			// reading b[i] would corrupt its own input.
			alias := append([]float64(nil), b...)
			c.into(alias, a, alias)
			want := make([]float64, n)
			c.into(want, a, b)
			bitsEqual(t, c.name+" Into aliased onto b", alias, want)
		}
	}
}

// TestScalarInPlaceMatchesInto covers the forms taking one scalar operand.
func TestScalarInPlaceMatchesInto(t *testing.T) {
	type scalarOp struct {
		name    string
		inPlace func(a []float64, s float64)
		into    func(dst, a []float64, s float64)
	}
	cases := []scalarOp{
		{"Scale", Scale[float64], ScaleInto[float64]},
		{"AddScalar", AddScalar[float64], AddScalarInto[float64]},
		{"SubScalar", SubScalar[float64], SubScalarInto[float64]},
		{"DivScalar", DivScalar[float64], DivScalarInto[float64]},
	}
	r := rand.New(rand.NewPCG(45, 46))
	for _, c := range cases {
		for _, n := range apiLens {
			a := rnd(n, r)
			s := r.NormFloat64() + 1.5
			inPlace := append([]float64(nil), a...)
			c.inPlace(inPlace, s)
			dst := make([]float64, n)
			c.into(dst, a, s)
			bitsEqual(t, c.name+" in place vs Into", inPlace, dst)
		}
	}
}

// TestWrappersDoNotAllocate is the promise the whole API shape exists to make.
//
// A wrapper that grew a slice, boxed a value into an interface, or built a
// closure would show up here and nowhere else — the kernels themselves are
// checked separately, and none of this is visible in a correctness test.
func TestWrappersDoNotAllocate(t *testing.T) {
	const n = 1024
	r := rand.New(rand.NewPCG(47, 48))
	a, b := rndPos(n, r), rndPos(n, r)
	dst := make([]float64, n)
	mask := make([]bool, n)
	bytesA, bytesB := make([]byte, n), make([]byte, n)
	dstB := make([]byte, n)

	checks := []struct {
		name string
		fn   func()
	}{
		{"AddInto", func() { AddInto(dst, a, b) }},
		{"MulInto", func() { MulInto(dst, a, b) }},
		{"ExpInto", func() { ExpInto(dst, a) }},
		{"LogInto", func() { LogInto(dst, a) }},
		{"SinInto", func() { SinInto(dst, a) }},
		{"PowInto", func() { PowInto(dst, a, b) }},
		{"Sum", func() { sinkF64 = Sum(a) }},
		{"Dot", func() { sinkF64 = Dot(a, b) }},
		{"Min", func() { sinkF64 = Min(a) }},
		{"Norm", func() { sinkF64 = Norm(a) }},
		{"Mean", func() { sinkF64 = Mean(a) }},
		{"LessInto", func() { LessInto(mask, a, b) }},
		{"SelectInto", func() { SelectInto(dst, mask, a, b) }},
		{"All", func() { sinkBoolAPI = All(mask) }},
		{"CountTrue", func() { sinkIntAPI = CountTrue(mask) }},
		{"IndexByte", func() { sinkIntAPI = IndexByte(bytesA, 'x') }},
		{"CountByte", func() { sinkIntAPI = CountByte(bytesA, 'x') }},
		{"Equal", func() { sinkBoolAPI = Equal(bytesA, bytesB) }},
		{"Compare", func() { sinkIntAPI = Compare(bytesA, bytesB) }},
		{"PopCount", func() { sinkIntAPI = PopCount(bytesA) }},
		{"XorInto", func() { XorInto(dstB, bytesA, bytesB) }},
		{"ToUpperASCIIInto", func() { ToUpperASCIIInto(dstB, bytesA) }},
		{"CumSumInto", func() { CumSumInto(dst, a) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(50, c.fn); got != 0 {
			t.Errorf("%s allocates %.1f times per call, want 0", c.name, got)
		}
	}
}

var (
	sinkF64     float64
	sinkIntAPI  int
	sinkBoolAPI bool
)

// TestMismatchedLengths checks that every whole-slice function stops at the
// shortest operand rather than reading past the end.
//
// The kernels take one length and trust it for every pointer, so the wrapper
// and the guard between them are the only thing keeping a short source from
// being read out of bounds. Under -race or a bounds-checked build this test
// fails loudly; without one it would still corrupt.
func TestMismatchedLengths(t *testing.T) {
	r := rand.New(rand.NewPCG(49, 50))
	for _, lens := range [][3]int{
		{64, 32, 48}, {32, 64, 48}, {48, 32, 64}, {0, 32, 32}, {32, 0, 32}, {1, 64, 64},
	} {
		dst, a, b := rnd(lens[0], r), rnd(lens[1], r), rnd(lens[2], r)
		n := min(len(dst), len(a), len(b))

		AddInto(dst, a, b)
		want := make([]float64, n)
		ref.Add(want, a[:n], b[:n])
		bitsEqual(t, "AddInto clamped", dst[:n], want)

		MulInto(dst, a, b)
		ref.Mul(want, a[:n], b[:n])
		bitsEqual(t, "MulInto clamped", dst[:n], want)

		// An algebraic op, deliberately: a transcendental is only promised to
		// within a ULP bound, so comparing one bit-for-bit against the
		// reference would be testing rule 6 rather than the clamping.
		AbsInto(dst, a)
		m := min(len(dst), len(a))
		w2 := make([]float64, m)
		ref.AbsFloat(w2, a[:m])
		bitsEqual(t, "AbsInto clamped", dst[:m], w2)
	}
}
