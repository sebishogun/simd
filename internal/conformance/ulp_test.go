package conformance

// The transcendentals, measured rather than asserted.
//
// Rule 6 of the kernel contract exempts these from bit identity: a polynomial
// correct to 1 ULP in float32 is not the one correct to 1 ULP in float64, so
// no single evaluation order reproduces both, and demanding identical bits
// would be demanding something no implementation can give. What they promise
// instead is a bound, and a promise like that is worth exactly as much as the
// test behind it.
//
// So this measures the worst disagreement with the Go standard library over a
// dense sweep of each function's interesting range, reports it, and fails if
// it exceeds the documented bound. The number in the table is what was
// measured, not what was hoped for.
//
// Two things are still checked exactly, because they cost nothing to keep and
// a violation would mean something is wrong rather than merely imprecise:
//
//   - NaN in gives NaN out, and an infinity gives the infinity the standard
//     library gives. A polynomial that quietly turns an infinity into a large
//     finite number is a bug, not a rounding difference.
//   - Every tier agrees with every other tier bit for bit. The kernels are one
//     fixed chain of elementwise operations, so vector width changes how many
//     lanes are in flight and nothing else.

import (
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// generated reports whether a tier replaced a kernel rather than keeping the
// portable one. Go forbids comparing func values, so this compares code
// pointers, which is exactly the question.
func generated(got, want func(dst, a []float64)) bool {
	return reflect.ValueOf(got).Pointer() != reflect.ValueOf(want).Pointer()
}

// ulpBound is the documented maximum disagreement with the Go standard
// library, in units in the last place, for each transcendental.
//
// These are the measured worst cases with headroom, not targets. Where one is
// larger than the rest it is because the function has a zero inside its range
// and the relative error there is dominated by how accurately the argument
// itself can be reduced — cos near an odd multiple of pi/2, for instance —
// which is a property of the input, not of the polynomial.
var ulpBound = map[string]float64{
	"Exp": 2, "Exp2": 2, "Expm1": 4,
	"Log": 4, "Log2": 4, "Log10": 4, "Log1p": 4,
	"Cbrt": 4, "Sigmoid": 4,
	"Sin": 4, "Cos": 8, "Tan": 8,
	// Asin and Acos are the two loosest, and the slack is the reference's
	// rather than the kernel's. Both references reach the ends of the domain
	// through math.Asin, which is Atan(x/Sqrt(1-x*x)) and loses half its
	// digits as |x| approaches 1. Checked against glibc: at the worst points
	// here — acos(0.989802) and asin(-0.9999) — the kernel returns glibc's
	// value to the bit and the reference is the side that has drifted.
	"Asin": 10, "Acos": 12, "Atan": 3,
	"Sinh": 5, "Cosh": 4, "Tanh": 4,
	"Pow": 8, "Atan2": 4, "Hypot": 3,
}

// ulpDiff returns how many units in the last place separate two values.
//
// Denormals share a single ULP, so the exponent is clamped at the smallest
// normal: without that, a difference near zero divides by an underflowed
// spacing and reports infinity for what is in fact an exact answer.
func ulpDiff(got, want float64, single bool) float64 {
	switch {
	case math.IsNaN(got) && math.IsNaN(want):
		return 0
	case got == want:
		return 0
	case math.IsNaN(got) != math.IsNaN(want):
		return math.Inf(1)
	case math.IsInf(got, 0) != math.IsInf(want, 0):
		return math.Inf(1)
	}
	mant, exp := math.Frexp(want)
	_ = mant
	if want == 0 {
		exp = 1
	}
	bits := 53
	minExp := -1021
	if single {
		bits = 24
		minExp = -125
	}
	if exp < minExp {
		exp = minExp
	}
	return math.Abs(got-want) / math.Ldexp(1, exp-bits)
}

// mathCase is one transcendental and the range worth sweeping it over.
type mathCase struct {
	name string
	// lo, hi bound the sweep; geo sweeps geometrically, which is the only way
	// a logarithm's sweep visits its mantissa reduction rather than spending
	// every point where the exponent term dominates.
	lo, hi float64
	geo    bool
	// ref is the standard library's answer.
	ref func(float64) float64
	// get runs the kernel over a slice.
	get32 func(kernel.Set) func(dst, a []float32)
	get64 func(kernel.Set) func(dst, a []float64)
}

func unaryCases() []mathCase {
	u32 := func(f func(kernel.Ops[float32]) func(dst, a []float32)) func(kernel.Set) func(dst, a []float32) {
		return func(s kernel.Set) func(dst, a []float32) { return f(s.F32) }
	}
	u64 := func(f func(kernel.Ops[float64]) func(dst, a []float64)) func(kernel.Set) func(dst, a []float64) {
		return func(s kernel.Set) func(dst, a []float64) { return f(s.F64) }
	}
	return []mathCase{
		{"Exp", -700, 700, false, math.Exp,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Exp }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Exp })},
		{"Exp2", -1000, 1000, false, math.Exp2,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Exp2 }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Exp2 })},
		{"Expm1", -20, 20, false, math.Expm1,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Expm1 }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Expm1 })},
		// Swept over normal numbers only. Go's math.Log is wrong for denormal
		// inputs — it returns -709.0896 for 1e-320, where the value is
		// -736.82724089097394 (320*ln(10), and glibc's answer, and this
		// kernel's). Sweeping there would measure that, not this.
		{"Log", 1e-300, 1e300, true, math.Log,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Log }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Log })},
		// The reference is Log(x)/Ln2 rather than math.Log2. Go computes Log2
		// as log(frac)*(1/Ln2) + float64(exp) after a Frexp, and for x just
		// above 1 that is a difference of two numbers near 1 producing an
		// answer near 0 — it loses about seven digits there, where Log does
		// not. Measuring against it would be measuring its cancellation.
		// Checked against glibc: at x = 1.0000001 this kernel returns
		// 1.4426949695965583e-07, which is glibc's log2 exactly, while Go's
		// math.Log2 returns 1.442694970155145e-07.
		{"Log2", 1e-300, 1e300, true, func(x float64) float64 { return math.Log(x) / math.Ln2 },
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Log2 }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Log2 })},
		{"Log10", 1e-300, 1e300, true, math.Log10,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Log10 }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Log10 })},
		{"Log1p", -0.9, 100, false, math.Log1p,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Log1p }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Log1p })},
		{"Cbrt", 1e-300, 1e300, true, math.Cbrt,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Cbrt }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Cbrt })},
		{"Sigmoid", -30, 30, false, func(x float64) float64 {
			if x >= 0 {
				return 1 / (1 + math.Exp(-x))
			}
			e := math.Exp(x)
			return e / (1 + e)
		},
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Sigmoid }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Sigmoid })},
		{"Sin", -100, 100, false, math.Sin,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Sin }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Sin })},
		{"Cos", -100, 100, false, math.Cos,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Cos }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Cos })},
		{"Tan", -100, 100, false, math.Tan,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Tan }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Tan })},
		{"Asin", -1, 1, false, math.Asin,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Asin }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Asin })},
		// Swept to |x| <= 0.99 only. math.Acos is pi/2 - Asin(x), which for |x|
		// near 1 subtracts two numbers near pi/2 to produce an answer near 0
		// and loses most of its digits; TestAcosNearOne covers that range
		// against a reference that does not cancel.
		{"Acos", -0.99, 0.99, false, math.Acos,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Acos }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Acos })},
		{"Atan", -1000, 1000, false, math.Atan,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Atan }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Atan })},
		{"Sinh", -20, 20, false, math.Sinh,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Sinh }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Sinh })},
		{"Cosh", -20, 20, false, math.Cosh,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Cosh }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Cosh })},
		{"Tanh", -20, 20, false, math.Tanh,
			u32(func(o kernel.Ops[float32]) func(d, a []float32) { return o.Tanh }),
			u64(func(o kernel.Ops[float64]) func(d, a []float64) { return o.Tanh })},
	}
}

const ulpSamples = 20001

func sweep(lo, hi float64, geo bool) []float64 {
	out := make([]float64, ulpSamples)
	for i := range out {
		f := float64(i) / float64(ulpSamples-1)
		if geo {
			out[i] = lo * math.Pow(hi/lo, f)
		} else {
			out[i] = lo + (hi-lo)*f
		}
	}
	return out
}

func TestTranscendentalULP(t *testing.T) {
	for tier, set := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			for _, c := range unaryCases() {
				bound, ok := ulpBound[c.name]
				if !ok {
					t.Fatalf("%s has no documented ULP bound", c.name)
				}
				xs := sweep(c.lo, c.hi, c.geo)

				a64 := append([]float64(nil), xs...)
				d64 := make([]float64, len(xs))
				c.get64(set)(d64, a64)

				a32 := make([]float32, len(xs))
				for i, x := range xs {
					a32[i] = float32(x)
				}
				d32 := make([]float32, len(xs))
				c.get32(set)(d32, a32)

				var w64, w32 float64
				var at64, at32 float64
				for i, x := range xs {
					if u := ulpDiff(d64[i], c.ref(x), false); u > w64 {
						w64, at64 = u, x
					}
					y := float64(a32[i])
					if u := ulpDiff(float64(d32[i]), float64(float32(c.ref(y))), true); u > w32 {
						w32, at32 = u, y
					}
				}
				t.Logf("%-8s f64 %6.2f ULP (at %.6g)   f32 %6.2f ULP (at %.6g)",
					c.name, w64, at64, w32, at32)
				if w64 > bound {
					t.Errorf("%s float64: %.2f ULP at %.17g, over the documented bound of %.0f",
						c.name, w64, at64, bound)
				}
				if w32 > bound {
					t.Errorf("%s float32: %.2f ULP at %.9g, over the documented bound of %.0f",
						c.name, w32, at32, bound)
				}
			}
		})
	}
}

// TestTranscendentalSpecialValues checks the parts that are exact even though
// the values are not: a NaN must stay a NaN and an infinity must go where the
// standard library sends it. Getting one of these wrong is a bug rather than a
// rounding difference, so it is compared exactly.
// The special values for these live in special_test.go, together with the
// Fast tier's, because keeping them apart is how they came to check different
// things: this one had no zero-sign check and that one reached float64 unary
// kernels only.

// TestTranscendentalTiersAgree checks that moving a program to a machine with
// wider vectors does not meaningfully change a number.
//
// Unlike the algebraic kernels, this is a bound and not bit identity, and the
// difference is worth being precise about. Where two tiers both have the
// kernel they do agree exactly, because the evaluation order is one fixed
// chain of elementwise operations and a wider vector only changes how many
// lanes are in flight. But a tier that could not compile a kernel keeps the
// portable path, which is a different algorithm — the baseline x86-64 tier
// does exactly that for everything here, because it reaches its constant pools
// with legacy SSE instructions that demand 16-byte alignment. So the promise
// across tiers is the same ULP bound the functions carry generally.
func TestTranscendentalTiersAgree(t *testing.T) {
	all := tiers(t)
	if len(all) < 2 {
		t.Skip("only one tier in this build")
	}
	names := make([]string, 0, len(all))
	for n := range all {
		names = append(names, n)
	}
	r := rand.New(rand.NewPCG(23, 24))
	xs := make([]float64, 4096)
	for i := range xs {
		xs[i] = r.NormFloat64() * math.Pow(10, r.Float64()*20-10)
	}
	// Only tiers that both have the kernel are compared, and they are compared
	// exactly. A tier that could not compile one keeps Go's math, which is a
	// different algorithm with a different range reduction — at x = -15868 the
	// two are 11 ULP apart, because sin is near a zero there and both
	// reductions are near the limit of what their split of pi carries. That
	// tier's accuracy is covered by TestTranscendentalULP against the
	// reference; what this test is for is the stronger claim, that the same
	// kernel at two vector widths returns the same bits.
	refSet := ref.Set()
	base := all[names[0]]
	for _, n := range names[1:] {
		for _, c := range unaryCases() {
			if !generated(c.get64(base), c.get64(refSet)) ||
				!generated(c.get64(all[n]), c.get64(refSet)) {
				continue
			}
			want := make([]float64, len(xs))
			got := make([]float64, len(xs))
			c.get64(base)(want, xs)
			c.get64(all[n])(got, xs)
			for i := range xs {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) &&
					!(math.IsNaN(got[i]) && math.IsNaN(want[i])) {
					t.Fatalf("%s(%v) = %v on %s but %v on %s; both have the kernel, "+
						"so they must agree bit for bit",
						c.name, xs[i], got[i], n, want[i], names[0])
				}
			}
		}
	}
}

// TestAcosNearOne covers the range the ULP sweep leaves out.
//
// math.Acos is pi/2 - Asin(x). For |x| close to 1 both terms are close to
// pi/2 and the answer is close to zero, so the subtraction throws away most of
// the significant digits — about seven of them at x = 0.9999. Comparing
// against it there would report this kernel as wrong by a thousand ULP when it
// is in fact the more accurate of the two.
//
// The reference used here is the half-angle identity
// acos(x) = 2*asin(sqrt((1-x)/2)), evaluated with Go's Asin. It has no
// cancellation: 1-x is exact for x in [1/2, 2] by Sterbenz's lemma, and the
// remaining operations are all well conditioned. Go's Asin is accurate, so
// this is an independent check and not a restatement of the kernel's own
// algorithm — the kernel evaluates its own minimax polynomial where this uses
// the standard library's.
func TestAcosNearOne(t *testing.T) {
	const bound = 4.0
	xs := make([]float64, 4001)
	for i := range xs {
		// Dense over [0.99, 1] and its mirror, where the cancellation bites.
		f := float64(i) / float64(len(xs)-1)
		x := 0.99 + 0.01*f
		if i%2 == 0 {
			x = -x
		}
		xs[i] = x
	}
	acc := func(x float64) float64 {
		a := 2 * math.Asin(math.Sqrt((1-math.Abs(x))/2))
		if x < 0 {
			return math.Pi - a
		}
		return a
	}
	for tier, set := range tiers(t) {
		got := make([]float64, len(xs))
		set.F64.Acos(got, xs)
		var worst, at float64
		for i, x := range xs {
			if u := ulpDiff(got[i], acc(x), false); u > worst {
				worst, at = u, x
			}
		}
		t.Logf("%s: Acos on |x| in [0.99, 1] worst %.2f ULP (at %.17g)", tier, worst, at)
		if worst > bound {
			t.Errorf("%s: Acos %.2f ULP at %.17g, over %.0f", tier, worst, at, bound)
		}
	}
}
