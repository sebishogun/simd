package simd_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// naiveConvolve is the definition. Both implementations are checked against
// it, and against each other at lengths that straddle the crossover, because
// the two paths are entirely different code and only one of them runs for any
// given input.
func naiveConvolve(a, b []float64) []float64 {
	out := make([]float64, len(a)+len(b)-1)
	for i, x := range a {
		for j, y := range b {
			out[i+j] += x * y
		}
	}
	return out
}

func TestConvolveFull(t *testing.T) {
	r := rand.New(rand.NewPCG(131, 137))
	for _, dims := range [][2]int{
		{1, 1}, {1, 5}, {5, 1}, {4, 4}, {17, 3}, {64, 64}, {1000, 7}, {333, 129},
	} {
		n, m := dims[0], dims[1]
		a := make([]float64, n)
		b := make([]float64, m)
		for i := range a {
			a[i] = r.NormFloat64()
		}
		for i := range b {
			b[i] = r.NormFloat64()
		}
		want := naiveConvolve(a, b)
		t.Run(fmt.Sprintf("%dx%d", n, m), func(t *testing.T) {
			got := simd.ConvolveFull(a, b)
			if len(got) != len(want) {
				t.Fatalf("len = %d, want %d", len(got), len(want))
			}
			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-10 {
					t.Fatalf("[%d] = %v, want %v", i, got[i], want[i])
				}
			}
			// Commutative.
			rev := simd.ConvolveFull(b, a)
			for i := range want {
				if math.Abs(rev[i]-want[i]) > 1e-10 {
					t.Fatalf("not commutative at %d", i)
				}
			}
		})
	}

	// The two paths must agree. Forced explicitly, because the dispatcher only
	// ever runs one of them and a bug in the other would never be seen.
	t.Run("pathsAgree", func(t *testing.T) {
		for _, m := range []int{1, 2, 63, 64, 65, 512, 1249, 1250, 1251} {
			const n = 4096
			a := make([]float64, n)
			b := make([]float64, m)
			for i := range a {
				a[i] = r.NormFloat64()
			}
			for i := range b {
				b[i] = r.NormFloat64()
			}
			d1 := make([]float64, n+m-1)
			d2 := make([]float64, n+m-1)
			simd.ConvolveForTest(d1, a, b, false)
			simd.ConvolveForTest(d2, a, b, true)
			for i := range d1 {
				// The FFT path accumulates rounding across log(size) stages,
				// so the tolerance scales with the transform size rather than
				// being a flat epsilon.
				if math.Abs(d1[i]-d2[i]) > 1e-9*float64(n) {
					t.Fatalf("m=%d: paths differ at %d: %v vs %v", m, i, d1[i], d2[i])
				}
			}
		}
	})

	// An impulse convolves to a copy, which catches an off-by-one in the
	// alignment that a random comparison can absorb.
	t.Run("impulse", func(t *testing.T) {
		a := []float64{1, 2, 3, 4, 5}
		imp := []float64{0, 0, 1}
		got := simd.ConvolveFull(a, imp)
		want := []float64{0, 0, 1, 2, 3, 4, 5}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})

	// Correlation against the definition, and against the lag convention.
	//
	// The first version of this test asserted the peak position from memory
	// and was wrong; checking the whole vector against a worked example is
	// both stricter and harder to fool.
	t.Run("correlate", func(t *testing.T) {
		got := simd.CorrelateFull([]float64{1, 2, 3}, []float64{10, 20})
		want := []float64{20, 50, 80, 30}
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-12 {
				t.Fatalf("[%d] = %v, want %v (whole: %v)", i, got[i], want[i], got)
			}
		}
		// An impulse at index 2 correlated with an impulse at index 1 peaks
		// where the two align, which is index 2 + (len(b)-1) - 1 = 3.
		g := simd.CorrelateFull([]float64{0, 0, 1, 0, 0}, []float64{0, 1, 0})
		peak, at := 0.0, -1
		for i, v := range g {
			if v > peak {
				peak, at = v, i
			}
		}
		if at != 3 {
			t.Errorf("peak at %d, want 3 (whole: %v)", at, g)
		}
	})

	// Degenerate inputs do nothing rather than panicking.
	t.Run("degenerate", func(t *testing.T) {
		if simd.ConvolveFull([]float64{}, []float64{1}) != nil {
			t.Error("empty input should give nil")
		}
		short := make([]float64, 2)
		simd.ConvolveFullInto(short, []float64{1, 2, 3}, []float64{1, 2})
		for i, v := range short {
			if v != 0 {
				t.Errorf("short dst[%d] = %v, want untouched", i, v)
			}
		}
	})
}
