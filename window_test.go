package simd_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/sebishogun/simd"
)

// Against the definitions, which is the only thing worth checking: a window is
// a formula, and the risk is an off-by-one in the denominator, not an
// arithmetic slip.
func TestWindows(t *testing.T) {
	type wf struct {
		name string
		fill func([]float64)
		want func(i, n int) float64
	}
	windows := []wf{
		{"Hann", simd.Hann[float64], func(i, n int) float64 {
			return 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1))
		}},
		{"HannPeriodic", simd.HannPeriodic[float64], func(i, n int) float64 {
			return 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
		}},
		{"Hamming", simd.Hamming[float64], func(i, n int) float64 {
			return 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1))
		}},
		{"Blackman", simd.Blackman[float64], func(i, n int) float64 {
			x := 2 * math.Pi * float64(i) / float64(n-1)
			return 0.42 - 0.5*math.Cos(x) + 0.08*math.Cos(2*x)
		}},
		{"Bartlett", simd.Bartlett[float64], func(i, n int) float64 {
			h := float64(n-1) / 2
			return 1 - math.Abs(float64(i)-h)/h
		}},
	}
	for _, w := range windows {
		for _, n := range []int{2, 3, 8, 65, 1000} {
			d := make([]float64, n)
			w.fill(d)
			t.Run(fmt.Sprintf("%s/n=%d", w.name, n), func(t *testing.T) {
				for i := range d {
					if e := math.Abs(d[i] - w.want(i, n)); e > 1e-12 {
						t.Fatalf("[%d] = %v, want %v (err %.3e)", i, d[i], w.want(i, n), e)
					}
				}
			})
		}
	}

	// The properties a caller relies on, which the pointwise check does not
	// make obvious.
	t.Run("properties", func(t *testing.T) {
		for _, n := range []int{9, 64, 501} {
			d := make([]float64, n)
			simd.Hann(d)
			// Symmetric, and zero at both ends.
			for i := range d {
				if math.Abs(d[i]-d[n-1-i]) > 1e-12 {
					t.Fatalf("Hann n=%d not symmetric at %d", n, i)
				}
			}
			if d[0] > 1e-12 || d[n-1] > 1e-12 {
				t.Fatalf("Hann n=%d endpoints %v %v, want 0", n, d[0], d[n-1])
			}
			// A periodic Hann is not zero at the far end; that is the whole
			// difference and it is worth pinning.
			p := make([]float64, n)
			simd.HannPeriodic(p)
			if p[n-1] < 1e-6 {
				t.Fatalf("HannPeriodic n=%d ends at %v, want nonzero", n, p[n-1])
			}
			// Bartlett peaks at 1 in the middle of an odd-length window.
			if n%2 == 1 {
				b := make([]float64, n)
				simd.Bartlett(b)
				if math.Abs(b[n/2]-1) > 1e-12 {
					t.Fatalf("Bartlett n=%d peak %v, want 1", n, b[n/2])
				}
			}
		}
	})

	// Degenerate lengths must not divide by zero.
	t.Run("degenerate", func(t *testing.T) {
		for _, w := range windows {
			w.fill(nil)
			one := make([]float64, 1)
			w.fill(one)
			if one[0] != 1 {
				t.Errorf("%s of length 1 = %v, want 1", w.name, one[0])
			}
		}
	})

	// float32 goes through the same generic path.
	t.Run("float32", func(t *testing.T) {
		d := make([]float32, 16)
		simd.Hamming(d)
		for i := range d {
			w := 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/15)
			if math.Abs(float64(d[i])-w) > 1e-6 {
				t.Fatalf("float32 Hamming[%d] = %v, want %v", i, d[i], w)
			}
		}
	})

	// ApplyWindowInto is the pairing the windows exist for.
	t.Run("apply", func(t *testing.T) {
		a := []float64{1, 2, 3, 4}
		w := []float64{0.5, 1, 1, 0.5}
		dst := make([]float64, 4)
		simd.ApplyWindowInto(dst, a, w)
		for i, want := range []float64{0.5, 2, 3, 2} {
			if dst[i] != want {
				t.Fatalf("apply[%d] = %v, want %v", i, dst[i], want)
			}
		}
	})
}
