package conformance

// Special values, for every transcendental, both tiers, both widths, both
// arities.
//
// This is one test rather than two because it used to be two, and they drifted
// into covering different things. The accurate tier's version checked NaN and
// the infinities but not the sign of a zero; the Fast tier's version checked
// zeros but read its table from unaryCases, so it reached float64 unary
// kernels only. Between them they left three real defects standing in shared
// source that both tiers compile:
//
//	asinh(-0) = +0   the sign was restored with `x < 0 ? -v : v`, and
//	erf(-0)   = +0   -0.0 < 0.0 is false
//	erf(0)    = 1e-9 the A&S coefficients sum to 0.999999999, not 1
//	pow(-0,1) = +0   the x == 0 branch never looked at the sign bit
//	pow(-0,-1)= +Inf and C99 F.10.4.4 says -Inf for an odd integer exponent
//
// and a fourth, simd_fast_hypot_f32 on ppc64le returning 0 where it owed NaN,
// which was a truncated constant pool but sat outside both tests at once —
// float32 and binary. See docs/wrong.md entry 53.
//
// So the coverage is not chosen here, it is enumerated: TestSpecialValueTable
// walks kernel.Ops by reflection and fails if any Fast slot is missing from
// the table below. Adding a transcendental without adding it here is a build
// failure rather than a silent hole.

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
)

// specialValues are the inputs where meaning, rather than accuracy, is at
// stake. Every one is exact in float32, so one list serves both widths.
var specialValues = []float64{
	math.NaN(), math.Inf(1), math.Inf(-1), 0, math.Copysign(0, -1),
}

// specialFinite are added to the binary sweep because most of the interesting
// two-argument rules are mixed rather than all-special: Pow(x, ±0) is 1 for
// every x including NaN, Pow(-0, -3) is -Inf, Atan2(±0, -1) is ±pi, and
// Hypot(±Inf, NaN) is +Inf. All are exact in float32.
var specialFinite = []float64{1, -1, 2, 0.5}

func sigmoidRef(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// specialUnary is every one-argument transcendental and the standard library's
// answer. The key is the accurate slot name in kernel.Ops; the Fast slot is
// that name with a "Fast" prefix.
var specialUnary = map[string]func(float64) float64{
	"Exp": math.Exp, "Exp2": math.Exp2, "Expm1": math.Expm1,
	"Log": math.Log, "Log2": math.Log2, "Log10": math.Log10,
	"Log1p": math.Log1p, "Cbrt": math.Cbrt, "Sigmoid": sigmoidRef,
	"Sin": math.Sin, "Cos": math.Cos, "Tan": math.Tan,
	"Asin": math.Asin, "Acos": math.Acos, "Atan": math.Atan,
	"Sinh": math.Sinh, "Cosh": math.Cosh, "Tanh": math.Tanh,
	"Asinh": math.Asinh, "Acosh": math.Acosh, "Atanh": math.Atanh,
	"Erf": math.Erf, "Erfc": math.Erfc,
}

// specialBinary is the same for the two-argument transcendentals.
var specialBinary = map[string]func(a, b float64) float64{
	"Pow": math.Pow, "Atan2": math.Atan2, "Hypot": math.Hypot,
}

// wantSpecial checks the three properties no tier gets to trade away. A finite
// non-zero answer is skipped: accuracy is what the Fast tier buys with, and
// the ULP sweeps are where that is held to account.
func wantSpecial(t *testing.T, call string, got, want float64) {
	t.Helper()
	switch {
	case math.IsNaN(want):
		if !math.IsNaN(got) {
			t.Errorf("%s = %v, want NaN", call, got)
		}
	case math.IsInf(want, 0):
		if !math.IsInf(got, int(math.Copysign(1, want))) {
			t.Errorf("%s = %v, want %v", call, got, want)
		}
	case want == 0:
		// The sign of a zero is part of the answer, and no amount of accuracy
		// slack licenses losing it.
		if got != 0 || math.Signbit(got) != math.Signbit(want) {
			t.Errorf("%s = %v, want %v", call, got, want)
		}
	}
}

func runSpecialUnary[T float32 | float64](t *testing.T, label string, g func(dst, a []T), ref func(float64) float64) {
	t.Helper()
	in := make([]T, len(specialValues))
	for i, x := range specialValues {
		in[i] = T(x)
	}
	dst := make([]T, len(in))
	g(dst, in)
	for i, x := range specialValues {
		wantSpecial(t, fmt.Sprintf("%s(%v)", label, x), float64(dst[i]), ref(x))
	}
}

func runSpecialBinary[T float32 | float64](t *testing.T, label string, g func(dst, a, b []T), ref func(a, b float64) float64) {
	t.Helper()
	args := append(append([]float64(nil), specialValues...), specialFinite...)
	var xs, ys []float64
	for _, x := range args {
		for _, y := range args {
			xs = append(xs, x)
			ys = append(ys, y)
		}
	}
	a := make([]T, len(xs))
	b := make([]T, len(ys))
	for i := range xs {
		a[i], b[i] = T(xs[i]), T(ys[i])
	}
	dst := make([]T, len(xs))
	g(dst, a, b)
	for i := range xs {
		wantSpecial(t, fmt.Sprintf("%s(%v, %v)", label, xs[i], ys[i]), float64(dst[i]), ref(xs[i], ys[i]))
	}
}

// slot fetches a kernel by field name, reporting nil for a slot this target
// does not fill. A wrong name is a test bug and fails loudly rather than
// silently skipping, which is the failure mode this whole file exists for.
func slot[F any](t *testing.T, ops any, name string) F {
	t.Helper()
	var zero F
	f := reflect.ValueOf(ops).FieldByName(name)
	if !f.IsValid() {
		t.Fatalf("kernel.Ops has no field %q", name)
	}
	if f.IsNil() {
		return zero
	}
	v, ok := f.Interface().(F)
	if !ok {
		t.Fatalf("kernel.Ops.%s is %s, not %T", name, f.Type(), zero)
	}
	return v
}

func TestSpecialValues(t *testing.T) {
	for tier, set := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			for _, name := range sortedKeys(specialUnary) {
				ref := specialUnary[name]
				for _, field := range []string{name, "Fast" + name} {
					if g := slot[func(dst, a []float32)](t, set.F32, field); g != nil {
						runSpecialUnary(t, field+"/f32", g, ref)
					}
					if g := slot[func(dst, a []float64)](t, set.F64, field); g != nil {
						runSpecialUnary(t, field+"/f64", g, ref)
					}
				}
			}
			for _, name := range sortedKeys(specialBinary) {
				ref := specialBinary[name]
				for _, field := range []string{name, "Fast" + name} {
					if g := slot[func(dst, a, b []float32)](t, set.F32, field); g != nil {
						runSpecialBinary(t, field+"/f32", g, ref)
					}
					if g := slot[func(dst, a, b []float64)](t, set.F64, field); g != nil {
						runSpecialBinary(t, field+"/f64", g, ref)
					}
				}
			}
		})
	}
}

// notPointwise are Fast slots this file deliberately does not cover, and the
// reason has to be that they are covered better somewhere else.
//
// The scans are not pointwise functions: the answer at element i depends on
// every element before it, so there is no f(x) to compare a special value
// against. Both run in the conformance differential against ref over the
// adversarial generator, which feeds ±0, NaN and the infinities *in sequence*
// and compares the whole output — a stronger check than this one, and the
// check that caught the SPLAT sign-of-zero bug. See conformance_test.go:421.
var notPointwise = map[string]bool{"CumSum": true, "CumProd": true}

// TestSpecialValueTable is the mechanical half: every Fast slot in kernel.Ops
// must be in one of the tables above, or in notPointwise with a reason. The
// tables name the accurate slot and derive the Fast one, so this covers both,
// and a transcendental added without a special-value entry fails here rather
// than shipping uncovered — which is how erf and asinh shipped wrong.
func TestSpecialValueTable(t *testing.T) {
	typ := reflect.TypeFor[kernel.Ops[float64]]()
	for i := range typ.NumField() {
		f := typ.Field(i)
		base, isFast := strings.CutPrefix(f.Name, "Fast")
		if !isFast || base == "" {
			continue
		}
		if specialUnary[base] != nil || specialBinary[base] != nil || notPointwise[base] {
			continue
		}
		t.Errorf("kernel.Ops.%s has no entry in specialUnary or specialBinary; "+
			"add %q so its special values are checked, or add it to notPointwise "+
			"with the reason it is covered elsewhere", f.Name, base)
	}
}

func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
