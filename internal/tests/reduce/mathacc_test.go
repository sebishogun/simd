package reduce

// Accuracy bounds for the transcendentals added most recently, pinned as
// tests rather than left as claims in a comment.
//
// The bounds here are the ones the doc comments state. If an implementation
// changes and quietly loses a digit, this fails; and if one improves, the
// failure is a prompt to tighten the documented bound rather than to leave it
// pessimistic.

import (
	"math"
	"testing"

	"github.com/sebishogun/simd"
)

func sweep(lo, hi float64, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = lo + (hi-lo)*float64(i)/float64(n-1)
	}
	return x
}

func TestTranscendentalAccuracy(t *testing.T) {
	// The inverse hyperbolics are built on log1p and sqrt, both accurate, so
	// they hold a few ULP. 1e-15 relative is comfortably above the ~4e-16
	// measured and comfortably below anything that would matter.
	hyperbolic := []struct {
		name string
		f    func(dst, a []float64)
		g    func(float64) float64
		lo   float64
		hi   float64
	}{
		{"Asinh", simd.AsinhInto[float64], math.Asinh, -100, 100},
		{"Acosh", simd.AcoshInto[float64], math.Acosh, 1, 100},
		{"Atanh", simd.AtanhInto[float64], math.Atanh, -0.999, 0.999},
	}
	for _, c := range hyperbolic {
		x := sweep(c.lo, c.hi, 5001)
		d := make([]float64, len(x))
		c.f(d, x)
		for i, v := range x {
			w := c.g(v)
			if math.IsNaN(w) || math.IsInf(w, 0) {
				continue
			}
			err := math.Abs(d[i] - w)
			if w != 0 {
				err /= math.Abs(w)
			}
			if err > 1e-15 {
				t.Errorf("%s(%v) = %v, want %v (relative error %.3e > 1e-15)",
					c.name, v, d[i], w, err)
				break
			}
		}
	}

	// Erf's documented bound is ABSOLUTE, because erf is bounded by 1. Stating
	// it as a relative bound would be a much weaker claim near zero and an
	// unmeetable one far out.
	x := sweep(-6, 6, 20001)
	d := make([]float64, len(x))
	simd.ErfInto(d, x)
	var worst float64
	for i, v := range x {
		if e := math.Abs(d[i] - math.Erf(v)); e > worst {
			worst = e
		}
		_ = v
	}
	if worst > 1.5e-7 {
		t.Errorf("Erf absolute error %.3e exceeds the documented 1.5e-7", worst)
	}

	// Erfc has no kernel and must be exactly Go's, which is the entire reason
	// it has no kernel. Any deviation means one was added without revisiting
	// the tail accuracy that made it a bad idea.
	simd.ErfcInto(d, x)
	for i, v := range x {
		if w := math.Erfc(v); d[i] != w {
			t.Fatalf("Erfc(%v) = %v, want exactly %v — Erfc must stay portable; "+
				"see the note at ErfcInto", v, d[i], w)
		}
	}

	// The edges C99 fixes.
	edges := []float64{math.Inf(1), math.Inf(-1), math.NaN(), 0, 1, -1}
	e := make([]float64, len(edges))
	simd.AtanhInto(e, edges)
	if !math.IsInf(e[4], 1) || !math.IsInf(e[5], -1) {
		t.Errorf("Atanh(±1) = %v, %v, want ±Inf", e[4], e[5])
	}
	if !math.IsNaN(e[0]) || !math.IsNaN(e[2]) {
		t.Errorf("Atanh(+Inf)=%v Atanh(NaN)=%v, want NaN", e[0], e[2])
	}
	simd.AcoshInto(e, []float64{0.5, 1, math.NaN(), 2, 2, 2})
	if !math.IsNaN(e[0]) {
		t.Errorf("Acosh(0.5) = %v, want NaN", e[0])
	}
	if e[1] != 0 {
		t.Errorf("Acosh(1) = %v, want 0", e[1])
	}
}
