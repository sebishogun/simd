package arrays

import (
	"bytes"
	"encoding/hex"
	"math"
	"math/rand/v2"
	"testing"
	"unicode/utf8"

	"github.com/sebishogun/simd"
)

// Correctness tests for the whole public surface.
//
// Two principles run through this file.
//
// First, every operation is checked against an independent oracle rather than
// against a second copy of its own logic — the standard library where one
// exists (math, bytes, utf8, encoding/hex), an obvious scalar loop otherwise,
// or a closed-form answer where the mathematics gives one. Testing an
// implementation against a paraphrase of itself finds nothing.
//
// Second, lengths are swept from 0 to 70 rather than sampled. The vector width
// this library targets tops out at 64 bytes, and reductions use 16
// accumulators, so 0..70 crosses every block boundary and every remainder in
// between. Nearly every correctness bug in the libraries surveyed in
// docs/research/03-competitive-analysis.md was a tail-handling bug that a
// single convenient length would have missed.
const maxLen = 70

func randF64(n int, r *rand.Rand) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64() * 10
	}
	return s
}

func randF32(n int, r *rand.Rand) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64() * 10)
	}
	return s
}

func randI64(n int, r *rand.Rand) []int64 {
	s := make([]int64, n)
	for i := range s {
		s[i] = int64(r.IntN(2001) - 1000)
	}
	return s
}

func clone[T any](s []T) []T { return append([]T(nil), s...) }

// ---------- elementwise ----------

func TestElementwiseMatchesScalarLoop(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for n := range maxLen + 1 {
		a, b := randF64(n, r), randF64(n, r)
		// Keep divisors away from zero so Div compares a finite result.
		for i := range b {
			if b[i] == 0 {
				b[i] = 1
			}
		}

		check := func(name string, got []float64, want func(x, y float64) float64) {
			t.Helper()
			for i := range got {
				if w := want(a[i], b[i]); got[i] != w && !(math.IsNaN(got[i]) && math.IsNaN(w)) {
					t.Fatalf("%s n=%d i=%d: got %v want %v", name, n, i, got[i], w)
				}
			}
		}

		dst := make([]float64, n)
		simd.AddInto(dst, a, b)
		check("Add", dst, func(x, y float64) float64 { return x + y })
		simd.SubInto(dst, a, b)
		check("Sub", dst, func(x, y float64) float64 { return x - y })
		simd.MulInto(dst, a, b)
		check("Mul", dst, func(x, y float64) float64 { return x * y })
		simd.DivInto(dst, a, b)
		check("Div", dst, func(x, y float64) float64 { return x / y })
		// The conversions around the products are load-bearing on any
		// architecture where Go fuses a multiply into a following add — arm64,
		// ppc64, s390x and riscv64 all do, amd64 does not. Without them this
		// expectation is computed with one rounding where the kernel uses two,
		// and the test fails by an ULP on four of the six architectures while
		// passing on the machine it was written on.
		simd.AddScaledInto(dst, a, b, 2.5)
		check("AddScaled", dst, func(x, y float64) float64 { return x + float64(y*2.5) })
		simd.LerpInto(dst, a, b, 0.25)
		check("Lerp", dst, func(x, y float64) float64 { return x + float64((y-x)*0.25) })
	}
}

// In-place must be exactly the destination form aimed at its own input.
// Getting this wrong is silent, and only shows up as corrupted data.
func TestInPlaceMatchesInto(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for n := range maxLen + 1 {
		a, b := randF64(n, r), randF64(n, r)

		want := make([]float64, n)
		simd.AddInto(want, a, b)
		got := clone(a)
		simd.Add(got, b)
		if !equalF64(got, want) {
			t.Fatalf("Add in place n=%d: got %v want %v", n, got, want)
		}

		simd.ScaleInto(want, a, 3)
		got = clone(a)
		simd.Scale(got, 3)
		if !equalF64(got, want) {
			t.Fatalf("Scale in place n=%d", n)
		}

		simd.AbsInto(want, a)
		got = clone(a)
		simd.Abs(got)
		if !equalF64(got, want) {
			t.Fatalf("Abs in place n=%d", n)
		}

		simd.ReverseInto(want, a)
		got = clone(a)
		simd.Reverse(got)
		if !equalF64(got, want) {
			t.Fatalf("Reverse in place n=%d: got %v want %v", n, got, want)
		}
	}
}

// Sizing: every operation processes the minimum of its slice lengths, and must
// not touch a single element past it.
func TestShorterDestinationBoundsTheWork(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float64{1, 1, 1, 1, 1, 1, 1, 1}
	dst := make([]float64, 8)
	const sentinel = -999
	for i := range dst {
		dst[i] = sentinel
	}
	simd.AddInto(dst[:3], a, b)
	for i := 3; i < len(dst); i++ {
		if dst[i] != sentinel {
			t.Fatalf("wrote past the destination at index %d: %v", i, dst)
		}
	}
	for i := range 3 {
		if dst[i] != a[i]+b[i] {
			t.Fatalf("dst[%d] = %v, want %v", i, dst[i], a[i]+b[i])
		}
	}
}

// ---------- IEEE 754 corner cases ----------

// The seam between the vector body, the scalar tail and the fallback is where
// viterin/vek#11 lives: its accelerated and unaccelerated paths disagree on
// NaN, so an answer changes with the length of the input. These cases pin the
// semantics down so a backend cannot drift.
func TestNaNAndSignedZeroSemantics(t *testing.T) {
	nan := math.NaN()
	inf := math.Inf(1)
	negZero := math.Copysign(0, -1)

	t.Run("Max propagates NaN", func(t *testing.T) {
		for _, in := range [][]float64{
			{nan, 1, 2}, {1, nan, 2}, {1, 2, nan}, {nan},
		} {
			if got := simd.Max(clone(in)); !math.IsNaN(got) {
				t.Errorf("Max(%v) = %v, want NaN", in, got)
			}
			if got := simd.Min(clone(in)); !math.IsNaN(got) {
				t.Errorf("Min(%v) = %v, want NaN", in, got)
			}
		}
	})

	t.Run("Max prefers +0 over -0", func(t *testing.T) {
		if got := simd.Max([]float64{negZero, 0}); math.Signbit(got) {
			t.Errorf("Max(-0, +0) = -0, want +0")
		}
		if got := simd.Min([]float64{0, negZero}); !math.Signbit(got) {
			t.Errorf("Min(+0, -0) = +0, want -0")
		}
	})

	t.Run("Abs clears the sign bit", func(t *testing.T) {
		a := []float64{negZero, -1, nan, -inf}
		simd.Abs(a)
		if math.Signbit(a[0]) {
			t.Error("Abs(-0) kept the sign bit; want +0")
		}
		if a[1] != 1 {
			t.Errorf("Abs(-1) = %v", a[1])
		}
		if !math.IsNaN(a[2]) || math.Signbit(a[2]) {
			t.Errorf("Abs(NaN) = %v, want a positive NaN", a[2])
		}
		if !math.IsInf(a[3], 1) {
			t.Errorf("Abs(-Inf) = %v, want +Inf", a[3])
		}
	})

	t.Run("Neg flips the sign bit", func(t *testing.T) {
		a := []float64{0, negZero, nan}
		simd.Neg(a)
		if !math.Signbit(a[0]) {
			t.Error("Neg(+0) = +0, want -0")
		}
		if math.Signbit(a[1]) {
			t.Error("Neg(-0) = -0, want +0")
		}
		if !math.IsNaN(a[2]) {
			t.Error("Neg(NaN) is not NaN")
		}
	})

	t.Run("Div by zero does not panic", func(t *testing.T) {
		a := []float64{1, -1, 0}
		simd.Div(a, []float64{0, 0, 0})
		if !math.IsInf(a[0], 1) || !math.IsInf(a[1], -1) || !math.IsNaN(a[2]) {
			t.Errorf("Div by zero gave %v, want [+Inf -Inf NaN]", a)
		}
	})
}

// ---------- reductions ----------

func TestReductionsMatchScalarLoop(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for n := 1; n <= maxLen; n++ {
		a, b := randF64(n, r), randF64(n, r)

		// Sum uses a 16-accumulator tree, so it will not equal a naive
		// left-to-right sum bit for bit. Compare within a tolerance scaled to
		// the magnitude involved; the bit-identity guarantee is between tiers,
		// not against a different algorithm.
		var naive float64
		for _, v := range a {
			naive += v
		}
		if got := simd.Sum(a); math.Abs(got-naive) > 1e-9*math.Abs(naive)+1e-9 {
			t.Errorf("Sum n=%d: got %v, naive %v", n, got, naive)
		}

		var wantDot float64
		for i := range a {
			wantDot += a[i] * b[i]
		}
		if got := simd.Dot(a, b); math.Abs(got-wantDot) > 1e-9*math.Abs(wantDot)+1e-9 {
			t.Errorf("Dot n=%d: got %v, naive %v", n, got, wantDot)
		}

		wantMin, wantMax := a[0], a[0]
		for _, v := range a {
			wantMin, wantMax = math.Min(wantMin, v), math.Max(wantMax, v)
		}
		if got := simd.Min(a); got != wantMin {
			t.Errorf("Min n=%d: got %v want %v", n, got, wantMin)
		}
		if got := simd.Max(a); got != wantMax {
			t.Errorf("Max n=%d: got %v want %v", n, got, wantMax)
		}
		lo, hi := simd.MinMax(a)
		if lo != wantMin || hi != wantMax {
			t.Errorf("MinMax n=%d: got (%v,%v) want (%v,%v)", n, lo, hi, wantMin, wantMax)
		}
		if got := a[simd.ArgMin(a)]; got != wantMin {
			t.Errorf("ArgMin n=%d points at %v, want %v", n, got, wantMin)
		}
		if got := a[simd.ArgMax(a)]; got != wantMax {
			t.Errorf("ArgMax n=%d points at %v, want %v", n, got, wantMax)
		}
	}
}

// Integer reductions are exact, so they must match a naive loop bit for bit,
// including the wrap on overflow.
func TestIntegerReductionsAreExact(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for n := range maxLen + 1 {
		a := randI64(n, r)
		var want int64
		for _, v := range a {
			want += v
		}
		if got := simd.Sum(a); got != want {
			t.Fatalf("Sum n=%d: got %d want %d", n, got, want)
		}
	}
	// Wrapping is the documented behaviour, not an error.
	if got := simd.Sum([]int64{math.MaxInt64, 1}); got != math.MinInt64 {
		t.Errorf("Sum overflow = %d, want wrap to %d", got, int64(math.MinInt64))
	}
	a := []int32{math.MinInt32}
	simd.Abs(a)
	if a[0] != math.MinInt32 {
		t.Errorf("Abs(MinInt32) = %d, want it to wrap to itself", a[0])
	}
}

func TestSumOfEmptyIsZero(t *testing.T) {
	if got := simd.Sum([]float64{}); got != 0 {
		t.Errorf("Sum(empty) = %v, want 0", got)
	}
	if got := simd.Dot([]float64{}, []float64{}); got != 0 {
		t.Errorf("Dot(empty) = %v, want 0", got)
	}
}

func TestMinMaxOfEmptyPanics(t *testing.T) {
	for name, fn := range map[string]func(){
		"Min":    func() { simd.Min([]float64{}) },
		"Max":    func() { simd.Max([]float64{}) },
		"ArgMin": func() { simd.ArgMin([]float64{}) },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s(empty) did not panic", name)
				}
			}()
			fn()
		})
	}
}

// ---------- norms and distances ----------

func TestNormsAndDistances(t *testing.T) {
	a := []float64{3, 4}
	b := []float64{0, 0}

	if got := simd.Norm(a); math.Abs(got-5) > 1e-12 {
		t.Errorf("Norm = %v, want 5", got)
	}
	if got := simd.L1Norm([]float64{-3, 4}); got != 7 {
		t.Errorf("L1Norm = %v, want 7", got)
	}
	if got := simd.SumSquares(a); got != 25 {
		t.Errorf("SumSquares = %v, want 25", got)
	}
	if got := simd.Distance(a, b); math.Abs(got-5) > 1e-12 {
		t.Errorf("Distance = %v, want 5", got)
	}
	if got := simd.SquaredDistance(a, b); got != 25 {
		t.Errorf("SquaredDistance = %v, want 25", got)
	}
	if got := simd.ManhattanDistance(a, b); got != 7 {
		t.Errorf("ManhattanDistance = %v, want 7", got)
	}
	// Orthogonal, parallel, antiparallel.
	if got := simd.CosineSimilarity([]float64{1, 0}, []float64{0, 1}); got != 0 {
		t.Errorf("cosine of perpendicular = %v, want 0", got)
	}
	if got := simd.CosineSimilarity([]float64{1, 2}, []float64{2, 4}); math.Abs(got-1) > 1e-12 {
		t.Errorf("cosine of parallel = %v, want 1", got)
	}
	if got := simd.CosineSimilarity([]float64{1, 0}, []float64{-1, 0}); math.Abs(got+1) > 1e-12 {
		t.Errorf("cosine of antiparallel = %v, want -1", got)
	}
	// A zero vector has no direction; the answer is defined as 0, not NaN.
	if got := simd.CosineSimilarity([]float64{0, 0}, []float64{1, 1}); got != 0 {
		t.Errorf("cosine with a zero vector = %v, want 0", got)
	}
}

// ---------- statistics ----------

func TestStatisticsAgainstKnownValues(t *testing.T) {
	// A textbook set with an exact mean and standard deviation.
	a := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	if got := simd.Mean(a); got != 5 {
		t.Errorf("Mean = %v, want 5", got)
	}
	if got := simd.Variance(a); got != 4 {
		t.Errorf("Variance = %v, want 4", got)
	}
	if got := simd.StdDev(a); got != 2 {
		t.Errorf("StdDev = %v, want 2", got)
	}
	// Sample variance divides by n-1: 32/7.
	if got := simd.SampleVariance(a); math.Abs(got-32.0/7.0) > 1e-12 {
		t.Errorf("SampleVariance = %v, want %v", got, 32.0/7.0)
	}

	// The two-pass form must survive a large offset, which is exactly where
	// the one-pass sum-of-squares identity collapses.
	shifted := clone(a)
	simd.AddScalar(shifted, 1e8)
	if got := simd.Variance(shifted); math.Abs(got-4) > 1e-6 {
		t.Errorf("Variance with a 1e8 offset = %v, want 4 — cancellation", got)
	}
}

func TestRescaleAndNormalize(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	simd.Rescale(a, 0, 1)
	if a[0] != 0 || a[len(a)-1] != 1 {
		t.Errorf("Rescale did not hit the endpoints: %v", a)
	}

	v := []float64{3, 4}
	simd.Normalize(v)
	if math.Abs(simd.Norm(v)-1) > 1e-15 {
		t.Errorf("Normalize gave length %v, want 1", simd.Norm(v))
	}

	z := []float64{7, 7, 7}
	simd.Normalize(z) // no direction to preserve, must not divide by zero
	simd.Standardize(z)
	for _, x := range z {
		if x != 0 {
			t.Errorf("Standardize of a constant slice = %v, want zeros", z)
			break
		}
	}

	s := []float64{1, 2, 3, 4}
	simd.Standardize(s)
	if m := simd.Mean(s); math.Abs(m) > 1e-12 {
		t.Errorf("Standardize left mean %v, want 0", m)
	}
	if sd := simd.StdDev(s); math.Abs(sd-1) > 1e-12 {
		t.Errorf("Standardize left stddev %v, want 1", sd)
	}
}

// ---------- scans ----------

func TestScans(t *testing.T) {
	a := []float64{1, 2, 3, 4}

	cs := clone(a)
	simd.CumSum(cs)
	if !equalF64(cs, []float64{1, 3, 6, 10}) {
		t.Errorf("CumSum = %v", cs)
	}

	d := make([]float64, 3)
	simd.DiffInto(d, a)
	if !equalF64(d, []float64{1, 1, 1}) {
		t.Errorf("Diff = %v", d)
	}

	cm := make([]float64, 4)
	simd.CumMinInto(cm, []float64{3, 1, 4, 0})
	if !equalF64(cm, []float64{3, 1, 1, 0}) {
		t.Errorf("CumMin = %v", cm)
	}
	simd.CumMaxInto(cm, []float64{3, 1, 4, 0})
	if !equalF64(cm, []float64{3, 3, 4, 4}) {
		t.Errorf("CumMax = %v", cm)
	}

	cp := clone(a)
	simd.CumProd(cp)
	if !equalF64(cp, []float64{1, 2, 6, 24}) {
		t.Errorf("CumProd = %v", cp)
	}
	if got := simd.Prod(a); got != 24 {
		t.Errorf("Prod = %v, want 24", got)
	}
}

// ---------- transcendentals ----------

// Rule 6 of the kernel contract: the transcendentals promise an accuracy
// bound, not bit identity. The bound is checked against the standard library,
// which is the reference for correctly rounded results.
func TestTranscendentalsAgainstStdlib(t *testing.T) {
	xs := []float64{-3, -1, -0.5, 0.1, 0.5, 1, 2, 3, 10}
	unary := map[string]struct {
		fn   func([]float64)
		want func(float64) float64
	}{
		"Exp":         {simd.Exp[float64], math.Exp},
		"Exp2":        {simd.Exp2[float64], math.Exp2},
		"Expm1":       {simd.Expm1[float64], math.Expm1},
		"Sin":         {simd.Sin[float64], math.Sin},
		"Cos":         {simd.Cos[float64], math.Cos},
		"Tan":         {simd.Tan[float64], math.Tan},
		"Atan":        {simd.Atan[float64], math.Atan},
		"Sinh":        {simd.Sinh[float64], math.Sinh},
		"Tanh":        {simd.Tanh[float64], math.Tanh},
		"Cbrt":        {simd.Cbrt[float64], math.Cbrt},
		"Floor":       {simd.Floor[float64], math.Floor},
		"Ceil":        {simd.Ceil[float64], math.Ceil},
		"Trunc":       {simd.Trunc[float64], math.Trunc},
		"Round":       {simd.Round[float64], math.Round},
		"RoundToEven": {simd.RoundToEven[float64], math.RoundToEven},
	}
	for name, c := range unary {
		t.Run(name, func(t *testing.T) {
			got := clone(xs)
			c.fn(got)
			for i, x := range xs {
				want := c.want(x)
				if !closeEnough(got[i], want) {
					t.Errorf("%s(%v) = %v, want %v", name, x, got[i], want)
				}
			}
		})
	}

	// Positive-domain functions get their own inputs.
	pos := []float64{0.1, 0.5, 1, 2, 10, 1000}
	for name, c := range map[string]struct {
		fn   func([]float64)
		want func(float64) float64
	}{
		"Log":   {simd.Log[float64], math.Log},
		"Log2":  {simd.Log2[float64], math.Log2},
		"Log10": {simd.Log10[float64], math.Log10},
		"Log1p": {simd.Log1p[float64], math.Log1p},
		"Sqrt":  {simd.Sqrt[float64], math.Sqrt},
	} {
		t.Run(name, func(t *testing.T) {
			got := clone(pos)
			c.fn(got)
			for i, x := range pos {
				if want := c.want(x); !closeEnough(got[i], want) {
					t.Errorf("%s(%v) = %v, want %v", name, x, got[i], want)
				}
			}
		})
	}
}

// Rounding is exact, so it must be equal, not merely close — including the
// half-way cases where Round and RoundToEven deliberately differ.
func TestRoundingIsExact(t *testing.T) {
	xs := []float64{-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2.4, 2.6}
	got := clone(xs)
	simd.Round(got)
	for i, x := range xs {
		if want := math.Round(x); got[i] != want {
			t.Errorf("Round(%v) = %v, want %v", x, got[i], want)
		}
	}
	got = clone(xs)
	simd.RoundToEven(got)
	for i, x := range xs {
		if want := math.RoundToEven(x); got[i] != want {
			t.Errorf("RoundToEven(%v) = %v, want %v", x, got[i], want)
		}
	}
}

// Sigmoid must not overflow to NaN at the extremes, which the naive
// 1/(1+exp(-x)) form does.
func TestSigmoidIsStableAtTheExtremes(t *testing.T) {
	a := []float64{-1000, -50, 0, 50, 1000}
	simd.Sigmoid(a)
	want := []float64{0, 0, 0.5, 1, 1}
	for i := range a {
		if math.IsNaN(a[i]) {
			t.Fatalf("Sigmoid produced NaN at index %d", i)
		}
		if math.Abs(a[i]-want[i]) > 1e-9 {
			t.Errorf("Sigmoid index %d = %v, want about %v", i, a[i], want[i])
		}
	}
}

// ---------- signal and polynomial ----------

func TestPolyEval(t *testing.T) {
	// 1 + 2x + 3x²
	coeffs := []float64{1, 2, 3}
	x := []float64{0, 1, 2, -1}
	got := make([]float64, len(x))
	simd.PolyEvalInto(got, x, coeffs)
	want := []float64{1, 6, 17, 2}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("PolyEval at x=%v: got %v want %v", x[i], got[i], want[i])
		}
	}
}

func TestConvolveAndCorrelate(t *testing.T) {
	sig := []float64{1, 2, 3, 4, 5}
	ker := []float64{1, 0, -1}

	got := make([]float64, len(sig)-len(ker)+1)
	simd.CorrelateInto(got, sig, ker)
	// Correlation: sum(w[j]*ker[j]) = 1*1 + 2*0 + 3*(-1) = -2, and so on.
	if !equalF64(got, []float64{-2, -2, -2}) {
		t.Errorf("Correlate = %v, want [-2 -2 -2]", got)
	}

	simd.ConvolveInto(got, sig, ker)
	// Convolution reverses the kernel, flipping the sign here.
	if !equalF64(got, []float64{2, 2, 2}) {
		t.Errorf("Convolve = %v, want [2 2 2]", got)
	}
}

func TestMovingAverage(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	got := make([]float64, 3)
	simd.MovingAverageInto(got, a, 3)
	if !equalF64(got, []float64{2, 3, 4}) {
		t.Errorf("MovingAverage = %v, want [2 3 4]", got)
	}
}

// ---------- quadrature and ODEs ----------

func TestQuadratureAgainstClosedForm(t *testing.T) {
	// Integrate x² from 0 to 1, which is exactly 1/3.
	const n = 101
	const h = 1.0 / (n - 1)
	y := make([]float64, n)
	for i := range y {
		x := float64(i) * h
		y[i] = x * x
	}
	// Simpson is exact for polynomials up to degree three.
	if got := simd.Simpson(y, h); math.Abs(got-1.0/3.0) > 1e-14 {
		t.Errorf("Simpson = %v, want 1/3", got)
	}
	// Trapezoid is second order, so it carries a small predictable error.
	if got := simd.Trapezoid(y, h); math.Abs(got-1.0/3.0) > 1e-4 {
		t.Errorf("Trapezoid = %v, want about 1/3", got)
	}
}

func TestRK4AgainstAnalyticSolution(t *testing.T) {
	// dy/dt = y, y(0) = 1, so y(t) = e**t. Fourth order should land within a
	// very tight tolerance over this interval.
	y := []float64{1}
	w := simd.NewRK4Workspace[float64](1)
	deriv := func(_ float64, y, dydt []float64) { copy(dydt, y) }

	const h = 0.01
	for i := range 100 {
		simd.RK4Step(y, float64(i)*h, h, deriv, w)
	}
	// RK4's global error is O(h^4), so h=0.01 over unit time allows about
	// 1e-8. Anything much larger means the step is wrong, not merely rounded.
	if want := math.E; math.Abs(y[0]-want) > 1e-8 {
		t.Errorf("RK4 integrating dy/dt=y to t=1 gave %v, want e=%v (err %.2e)",
			y[0], want, math.Abs(y[0]-want))
	}

	// The workspace makes stepping allocation-free.
	y2 := []float64{1}
	if got := testing.AllocsPerRun(50, func() { simd.RK4Step(y2, 0, h, deriv, w) }); got != 0 {
		t.Errorf("RK4Step allocated %.1f times per step, want 0", got)
	}
}

func TestEulerAndVerlet(t *testing.T) {
	// Euler on dy/dt = y is crude but must move in the right direction.
	y, dydt := []float64{1}, make([]float64, 1)
	simd.EulerStep(y, dydt, 0, 0.1, func(_ float64, y, out []float64) { copy(out, y) })
	if math.Abs(y[0]-1.1) > 1e-15 {
		t.Errorf("EulerStep = %v, want 1.1", y[0])
	}

	// Verlet under constant gravity: after one step of h=1 from rest,
	// position should have moved by a*h²/2 and velocity by a*h.
	pos, vel, acc := []float64{0}, []float64{0}, []float64{-10}
	simd.VerletStep(pos, vel, acc, 1, func(_, out []float64) { out[0] = -10 })
	if math.Abs(pos[0]+5) > 1e-12 {
		t.Errorf("VerletStep position = %v, want -5", pos[0])
	}
	if math.Abs(vel[0]+10) > 1e-12 {
		t.Errorf("VerletStep velocity = %v, want -10", vel[0])
	}
}

// ---------- regression ----------

func TestLinearRegressionRecoversAnExactLine(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4, 5}
	y := make([]float64, len(x))
	for i := range x {
		y[i] = 3*x[i] + 7
	}
	slope, intercept := simd.LinearRegression(x, y)
	if math.Abs(slope-3) > 1e-10 || math.Abs(intercept-7) > 1e-10 {
		t.Errorf("LinearRegression = (%v, %v), want (3, 7)", slope, intercept)
	}
	if got := simd.Correlation(x, y); math.Abs(got-1) > 1e-10 {
		t.Errorf("Correlation of an exact line = %v, want 1", got)
	}
	// A flat x determines no line.
	if s, b := simd.LinearRegression([]float64{1, 1, 1}, []float64{1, 2, 3}); s != 0 || b != 0 {
		t.Errorf("LinearRegression with no spread in x = (%v,%v), want (0,0)", s, b)
	}
}

// ---------- machine learning ----------

func TestSoftmaxAndLogSumExp(t *testing.T) {
	a := []float64{1, 2, 3}
	got := clone(a)
	simd.Softmax(got)

	if s := simd.Sum(got); math.Abs(s-1) > 1e-12 {
		t.Errorf("Softmax sums to %v, want 1", s)
	}
	for i, v := range got {
		if v < 0 || v > 1 {
			t.Errorf("Softmax index %d = %v, outside [0,1]", i, v)
		}
	}
	// Softmax is shift invariant, which is why subtracting the max is safe.
	shifted := clone(a)
	simd.AddScalar(shifted, 1000)
	simd.Softmax(shifted)
	for i := range got {
		if math.Abs(got[i]-shifted[i]) > 1e-12 {
			t.Errorf("Softmax is not shift invariant at %d: %v vs %v", i, got[i], shifted[i])
		}
	}
	// The whole point of subtracting the max: no overflow to NaN.
	big := []float64{1000, 1001, 1002}
	simd.Softmax(big)
	for _, v := range big {
		if math.IsNaN(v) {
			t.Fatalf("Softmax overflowed on large inputs: %v", big)
		}
	}

	if got := simd.LogSumExp([]float64{0, 0}); math.Abs(got-math.Ln2) > 1e-12 {
		t.Errorf("LogSumExp([0,0]) = %v, want ln2", got)
	}
	if got := simd.LogSumExp([]float64{1000, 1000}); math.IsInf(got, 0) || math.IsNaN(got) {
		t.Errorf("LogSumExp overflowed: %v", got)
	}
}

func TestActivations(t *testing.T) {
	a := []float64{-2, -0.5, 0, 0.5, 2}

	relu := clone(a)
	simd.ReLU(relu)
	if !equalF64(relu, []float64{0, 0, 0, 0.5, 2}) {
		t.Errorf("ReLU = %v", relu)
	}

	leaky := clone(a)
	simd.LeakyReLU(leaky, 0.1)
	if math.Abs(leaky[0]+0.2) > 1e-15 || leaky[4] != 2 {
		t.Errorf("LeakyReLU = %v", leaky)
	}

	// Softplus must not overflow, and must approach ReLU at the extremes.
	sp := []float64{-1000, 0, 1000}
	simd.Softplus(sp)
	if math.IsNaN(sp[0]) || math.IsNaN(sp[2]) {
		t.Fatalf("Softplus produced NaN: %v", sp)
	}
	if math.Abs(sp[1]-math.Ln2) > 1e-12 {
		t.Errorf("Softplus(0) = %v, want ln2", sp[1])
	}
	if math.Abs(sp[2]-1000) > 1e-9 {
		t.Errorf("Softplus(1000) = %v, want about 1000", sp[2])
	}

	// GELU and SiLU are odd-ish and must pass through zero.
	g := []float64{0}
	simd.GELU(g)
	if g[0] != 0 {
		t.Errorf("GELU(0) = %v, want 0", g[0])
	}
	s := []float64{0}
	simd.SiLU(s)
	if s[0] != 0 {
		t.Errorf("SiLU(0) = %v, want 0", s[0])
	}
}

func TestNormalizationLayers(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	ln := clone(a)
	simd.LayerNorm(ln, 1e-5)
	if m := simd.Mean(ln); math.Abs(m) > 1e-12 {
		t.Errorf("LayerNorm left mean %v, want 0", m)
	}

	rms := []float64{3, 4}
	simd.RMSNorm(rms, 0)
	// RMS of [3,4] is sqrt(12.5), so the result has RMS 1.
	got := math.Sqrt((rms[0]*rms[0] + rms[1]*rms[1]) / 2)
	if math.Abs(got-1) > 1e-12 {
		t.Errorf("RMSNorm left RMS %v, want 1", got)
	}

	// A constant input must not divide by zero.
	c := []float64{5, 5, 5}
	simd.LayerNorm(c, 1e-5)
	for _, v := range c {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("LayerNorm of a constant slice = %v", c)
			break
		}
	}
}

// ---------- bytes and text ----------

func TestBytesAgainstStdlib(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	for n := range maxLen + 1 {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte(r.IntN(4)) // a small alphabet, so hits are frequent
		}
		other := clone(b)
		if n > 0 {
			other[r.IntN(n)] ^= 0xff
		}

		for _, c := range []byte{0, 1, 9} {
			if got, want := simd.IndexByte(b, c), bytes.IndexByte(b, c); got != want {
				t.Fatalf("IndexByte n=%d c=%d: got %d want %d", n, c, got, want)
			}
			if got, want := simd.CountByte(b, c), bytes.Count(b, []byte{c}); got != want {
				t.Fatalf("Count n=%d c=%d: got %d want %d", n, c, got, want)
			}
		}
		if got, want := simd.Equal(b, other), bytes.Equal(b, other); got != want {
			t.Fatalf("Equal n=%d: got %v want %v", n, got, want)
		}
		if got, want := simd.Compare(b, other), bytes.Compare(b, other); got != want {
			t.Fatalf("Compare n=%d: got %d want %d", n, got, want)
		}
		if n > 2 {
			needle := b[1:3]
			if got, want := simd.Index(b, needle), bytes.Index(b, needle); got != want {
				t.Fatalf("Index n=%d: got %d want %d", n, got, want)
			}
		}
		if got, want := simd.IndexAny(b, []byte{1, 2}), bytes.IndexAny(b, "\x01\x02"); got != want {
			t.Fatalf("IndexAny n=%d: got %d want %d", n, got, want)
		}
	}
}

func TestIndexAllFindsEveryOccurrence(t *testing.T) {
	b := []byte("a,bb,,ccc,")
	offs := make([]int32, 16)
	n := simd.IndexAll(offs, b, ',')
	want := []int32{1, 4, 5, 9}
	if n != len(want) {
		t.Fatalf("IndexAll found %d, want %d", n, len(want))
	}
	for i := range want {
		if offs[i] != want[i] {
			t.Errorf("IndexAll[%d] = %d, want %d", i, offs[i], want[i])
		}
	}
	// A short destination bounds the work rather than erroring.
	short := make([]int32, 2)
	if got := simd.IndexAll(short, b, ','); got != 2 {
		t.Errorf("IndexAll with a short destination returned %d, want 2", got)
	}
}

func TestTextHelpers(t *testing.T) {
	if !simd.IsASCII([]byte("hello")) {
		t.Error("IsASCII said no to ASCII")
	}
	if simd.IsASCII([]byte("héllo")) {
		t.Error("IsASCII said yes to non-ASCII")
	}
	for _, s := range []string{"", "hello", "héllo", "日本語"} {
		if got, want := simd.ValidUTF8([]byte(s)), utf8.ValidString(s); got != want {
			t.Errorf("ValidUTF8(%q) = %v, want %v", s, got, want)
		}
	}
	if simd.ValidUTF8([]byte{0xff, 0xfe}) {
		t.Error("ValidUTF8 accepted invalid input")
	}

	// ASCII case folding must leave multi-byte sequences alone.
	b := []byte("Héllo, World!")
	up := clone(b)
	simd.ToUpperASCII(up)
	if string(up) != "HéLLO, WORLD!" {
		t.Errorf("ToUpperASCII = %q", up)
	}
	lo := clone(b)
	simd.ToLowerASCII(lo)
	if string(lo) != "héllo, world!" {
		t.Errorf("ToLowerASCII = %q", lo)
	}
	if !simd.EqualFoldASCII([]byte("Content-Type"), []byte("content-type")) {
		t.Error("EqualFoldASCII said no to a case difference")
	}
	if simd.EqualFoldASCII([]byte("a"), []byte("ab")) {
		t.Error("EqualFoldASCII ignored a length difference")
	}

	rb := []byte("a-b-c")
	simd.ReplaceByte(rb, '-', '_')
	if string(rb) != "a_b_c" {
		t.Errorf("ReplaceByte = %q", rb)
	}
}

func TestHexAgainstStdlib(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	for n := range 40 {
		src := make([]byte, n)
		for i := range src {
			src[i] = byte(r.IntN(256))
		}
		enc := make([]byte, 2*n)
		if got := simd.HexEncode(enc, src); got != 2*n {
			t.Fatalf("HexEncode wrote %d, want %d", got, 2*n)
		}
		if want := hex.EncodeToString(src); string(enc) != want {
			t.Fatalf("HexEncode = %q, want %q", enc, want)
		}

		dec := make([]byte, n)
		got, ok := simd.HexDecode(dec, enc)
		if !ok || got != n || !bytes.Equal(dec, src) {
			t.Fatalf("HexDecode round trip failed at n=%d", n)
		}
	}
	// Invalid input is reported, not panicked on.
	dec := make([]byte, 4)
	if _, ok := simd.HexDecode(dec, []byte("zz")); ok {
		t.Error("HexDecode accepted invalid digits")
	}
	if _, ok := simd.HexDecode(dec, []byte("abc")); ok {
		t.Error("HexDecode accepted an odd-length input")
	}
}

func TestBitwiseOps(t *testing.T) {
	a := []byte{0b1100, 0b1010}
	b := []byte{0b1010, 0b0110}

	for _, c := range []struct {
		name string
		fn   func(dst, x, y []byte)
		want []byte
	}{
		{"And", simd.AndInto, []byte{0b1000, 0b0010}},
		{"Or", simd.OrInto, []byte{0b1110, 0b1110}},
		{"Xor", simd.XorInto, []byte{0b0110, 0b1100}},
		{"AndNot", simd.AndNotInto, []byte{0b0100, 0b1000}},
	} {
		dst := make([]byte, 2)
		c.fn(dst, a, b)
		if !bytes.Equal(dst, c.want) {
			t.Errorf("%s = %v, want %v", c.name, dst, c.want)
		}
	}

	if got := simd.PopCount([]byte{0xff, 0x0f}); got != 12 {
		t.Errorf("PopCount = %d, want 12", got)
	}
}

// ---------- fuzzing ----------

// The differential harness: whatever the inputs, the accelerated path must
// agree with an obvious scalar loop. This is the shape that will compare tiers
// against the reference once generated backends exist; for now it holds the
// reference to its own contract.
func FuzzDifferential(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4}, []byte{5, 6, 7, 8})
	f.Add([]byte{}, []byte{})
	f.Add(bytes.Repeat([]byte{0xff}, 65), bytes.Repeat([]byte{0x01}, 65))

	f.Fuzz(func(t *testing.T, ab, bb []byte) {
		n := min(len(ab), len(bb)) / 8
		if n == 0 {
			return
		}
		a, b := bytesToF64(ab, n), bytesToF64(bb, n)

		dst := make([]float64, n)
		simd.AddInto(dst, a, b)
		for i := range dst {
			want := a[i] + b[i]
			if dst[i] != want && !(math.IsNaN(dst[i]) && math.IsNaN(want)) {
				t.Fatalf("Add mismatch at %d: got %v want %v", i, dst[i], want)
			}
		}

		// Min and Max must agree with a scalar loop using the same NaN rule.
		wantMin, wantMax := a[0], a[0]
		sawNaN := false
		for _, v := range a {
			if math.IsNaN(v) {
				sawNaN = true
			}
			wantMin, wantMax = math.Min(wantMin, v), math.Max(wantMax, v)
		}
		gotMin, gotMax := simd.MinMax(a)
		if sawNaN {
			if !math.IsNaN(gotMin) || !math.IsNaN(gotMax) {
				t.Fatalf("MinMax did not propagate NaN: got (%v,%v)", gotMin, gotMax)
			}
		} else if gotMin != wantMin || gotMax != wantMax {
			t.Fatalf("MinMax = (%v,%v), want (%v,%v)", gotMin, gotMax, wantMin, wantMax)
		}

		// Bytes: must match the standard library exactly.
		if got, want := simd.IndexByte(ab, 0x7f), bytes.IndexByte(ab, 0x7f); got != want {
			t.Fatalf("IndexByte = %d, want %d", got, want)
		}
		if got, want := simd.Compare(ab, bb), bytes.Compare(ab, bb); got != want {
			t.Fatalf("Compare = %d, want %d", got, want)
		}
		if got, want := simd.ValidUTF8(ab), utf8.Valid(ab); got != want {
			t.Fatalf("ValidUTF8 = %v, want %v", got, want)
		}
	})
}

func bytesToF64(b []byte, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		var u uint64
		for j := range 8 {
			u = u<<8 | uint64(b[i*8+j])
		}
		out[i] = math.Float64frombits(u)
	}
	return out
}

// ---------- helpers ----------

func equalF64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] && !(math.IsNaN(a[i]) && math.IsNaN(b[i])) {
			return false
		}
	}
	return true
}

// closeEnough compares against the standard library with a tolerance that
// accommodates the double rounding the reference does for float32, and the ULP
// bound rule 6 allows for transcendentals.
func closeEnough(got, want float64) bool {
	if math.IsNaN(got) && math.IsNaN(want) {
		return true
	}
	if got == want {
		return true
	}
	return math.Abs(got-want) <= 1e-12*math.Max(1, math.Abs(want))
}
