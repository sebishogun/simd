package conformance

// The Fast tier's accuracy, measured the same way the accurate tier's is.
//
// The bound this tier promises is looser and the check is otherwise identical:
// sweep the domain, compare against the standard library, report the worst
// point and the input that produced it. What is *not* checked here, and is
// checked for every other kernel in this package, is that two tiers agree bit
// for bit. They cannot: these are compiled with fused multiply-add, so a
// machine with an FMA and one without give different answers, and that is the
// trade the name is warning about.
//
// A target with no Fast kernel runs the accurate one, so on those this
// measures the accurate tier a second time and passes comfortably. That is the
// intended behaviour and not a gap: a bound is an upper bound.

import (
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
)

// fastBoundFor is what a Fast function promises, as a rule rather than a
// table, so the two cannot drift apart.
//
// The rule is max(3.5, the accurate bound), plus two for the arcs. It has that
// shape rather than a flat 3.5 because a function whose *accurate* kernel is
// already 8 ULP from the standard library — Cos and Tan, where the slack is
// the argument reduction near an odd multiple of pi/2 and not the polynomial
// at all — cannot be held to 3.5 by making it cheaper. For those the Fast tier
// is no worse than the accurate one and inherits the bound.
//
// Asin and Acos take two more, and that is the one place the shorter
// polynomial genuinely shows. Even there most of the figure belongs to the
// reference, which reaches the ends of the domain through math.Asin and loses
// half its digits as |x| approaches 1; see the note on ulpBound.
//
// Measured on this host, the Fast tier comes in well inside all of it: Exp 2,
// Exp2 1, Expm1 3, the logarithms 0, Log1p 3, Cbrt 0, Sigmoid 2, Sin 4, Cos 5,
// Tan 6, Asin 8, Acos 10, Atan 2, Sinh 4, Cosh 1, Tanh 3. The test prints its
// measurement every run, so a regression shows up as a number moving rather
// than as a silent pass.
func fastBoundFor(name string) (float64, bool) {
	acc, ok := ulpBound[name]
	if !ok {
		return 0, false
	}
	b := max(3.5, acc)
	switch name {
	case "Asin", "Acos":
		b += 2
	}
	return b, true
}

func fastSlot32(o kernel.Ops[float32], name string) func(dst, a []float32) {
	switch name {
	case "Exp":
		return o.FastExp
	case "Exp2":
		return o.FastExp2
	case "Expm1":
		return o.FastExpm1
	case "Log":
		return o.FastLog
	case "Log2":
		return o.FastLog2
	case "Log10":
		return o.FastLog10
	case "Log1p":
		return o.FastLog1p
	case "Cbrt":
		return o.FastCbrt
	case "Sigmoid":
		return o.FastSigmoid
	case "Sin":
		return o.FastSin
	case "Cos":
		return o.FastCos
	case "Tan":
		return o.FastTan
	case "Asin":
		return o.FastAsin
	case "Acos":
		return o.FastAcos
	case "Atan":
		return o.FastAtan
	case "Sinh":
		return o.FastSinh
	case "Cosh":
		return o.FastCosh
	case "Tanh":
		return o.FastTanh
	}
	return nil
}

func fastSlot64(o kernel.Ops[float64], name string) func(dst, a []float64) {
	switch name {
	case "Exp":
		return o.FastExp
	case "Exp2":
		return o.FastExp2
	case "Expm1":
		return o.FastExpm1
	case "Log":
		return o.FastLog
	case "Log2":
		return o.FastLog2
	case "Log10":
		return o.FastLog10
	case "Log1p":
		return o.FastLog1p
	case "Cbrt":
		return o.FastCbrt
	case "Sigmoid":
		return o.FastSigmoid
	case "Sin":
		return o.FastSin
	case "Cos":
		return o.FastCos
	case "Tan":
		return o.FastTan
	case "Asin":
		return o.FastAsin
	case "Acos":
		return o.FastAcos
	case "Atan":
		return o.FastAtan
	case "Sinh":
		return o.FastSinh
	case "Cosh":
		return o.FastCosh
	case "Tanh":
		return o.FastTanh
	}
	return nil
}

func TestFastTranscendentalULP(t *testing.T) {
	for tier, set := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			for _, c := range unaryCases() {
				bound, ok := fastBoundFor(c.name)
				if !ok {
					t.Fatalf("Fast%s has no documented ULP bound", c.name)
				}
				g64, g32 := fastSlot64(set.F64, c.name), fastSlot32(set.F32, c.name)
				if g64 == nil || g32 == nil {
					continue
				}
				xs := sweep(c.lo, c.hi, c.geo)

				a64 := append([]float64(nil), xs...)
				d64 := make([]float64, len(xs))
				g64(d64, a64)

				a32 := make([]float32, len(xs))
				for i, x := range xs {
					a32[i] = float32(x)
				}
				d32 := make([]float32, len(xs))
				g32(d32, a32)

				var w64, w32, at64, at32 float64
				for i, x := range xs {
					if u := ulpDiff(d64[i], c.ref(x), false); u > w64 {
						w64, at64 = u, x
					}
					y := float64(a32[i])
					if u := ulpDiff(float64(d32[i]), float64(float32(c.ref(y))), true); u > w32 {
						w32, at32 = u, y
					}
				}
				t.Logf("Fast%-8s f64 %6.2f ULP (at %g)   f32 %6.2f ULP (at %g)",
					c.name, w64, at64, w32, at32)
				if w64 > bound {
					t.Errorf("Fast%s float64: %.2f ULP at %g, over the documented bound of %g",
						c.name, w64, at64, bound)
				}
				if w32 > bound {
					t.Errorf("Fast%s float32: %.2f ULP at %g, over the documented bound of %g",
						c.name, w32, at32, bound)
				}
			}
		})
	}
}
