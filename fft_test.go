package simd_test

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// naiveDFT is the definition, O(n^2), and the only oracle worth checking an
// FFT against: anything faster shares assumptions with the thing being tested.
func naiveDFT(a []complex128, inverse bool) []complex128 {
	n := len(a)
	out := make([]complex128, n)
	sign := -2 * math.Pi
	if inverse {
		sign = 2 * math.Pi
	}
	for k := range n {
		var s complex128
		for j := range n {
			ang := sign * float64(k) * float64(j) / float64(n)
			s += a[j] * cmplx.Exp(complex(0, ang))
		}
		if inverse {
			s /= complex(float64(n), 0)
		}
		out[k] = s
	}
	return out
}

func TestFFT(t *testing.T) {
	r := rand.New(rand.NewPCG(89, 97))
	for _, n := range []int{1, 2, 4, 8, 16, 64, 256, 1024} {
		a := make([]complex128, n)
		for i := range a {
			a[i] = complex(r.NormFloat64(), r.NormFloat64())
		}
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			got := simd.FFT(a)
			want := naiveDFT(a, false)
			// The error grows as sqrt(log n) for a radix-2 FFT, so the bound
			// scales rather than being a flat epsilon.
			tol := 1e-12 * float64(n)
			for i := range want {
				if cmplx.Abs(got[i]-want[i]) > tol {
					t.Fatalf("FFT[%d] = %v, want %v (tol %g)", i, got[i], want[i], tol)
				}
			}

			// The inverse must recover the input, which is the property a
			// caller actually depends on.
			back := simd.IFFT(got)
			for i := range a {
				if cmplx.Abs(back[i]-a[i]) > tol {
					t.Fatalf("IFFT(FFT(x))[%d] = %v, want %v", i, back[i], a[i])
				}
			}

			// And the inverse alone must match the definition.
			gi := simd.IFFT(a)
			wi := naiveDFT(a, true)
			for i := range wi {
				if cmplx.Abs(gi[i]-wi[i]) > tol {
					t.Fatalf("IFFT[%d] = %v, want %v", i, gi[i], wi[i])
				}
			}
		})
	}

	// Known transforms, where an error in the sign convention or the scaling
	// shows up as an obviously wrong answer rather than a small one.
	t.Run("known", func(t *testing.T) {
		// A constant transforms to a spike at zero of magnitude n.
		c := []complex128{1, 1, 1, 1, 1, 1, 1, 1}
		g := simd.FFT(c)
		if cmplx.Abs(g[0]-8) > 1e-12 {
			t.Errorf("FFT(constant)[0] = %v, want 8", g[0])
		}
		for i := 1; i < len(g); i++ {
			if cmplx.Abs(g[i]) > 1e-12 {
				t.Errorf("FFT(constant)[%d] = %v, want 0", i, g[i])
			}
		}
		// A unit impulse transforms to all ones.
		d := make([]complex128, 8)
		d[0] = 1
		g = simd.FFT(d)
		for i := range g {
			if cmplx.Abs(g[i]-1) > 1e-12 {
				t.Errorf("FFT(impulse)[%d] = %v, want 1", i, g[i])
			}
		}
		// The sign convention, pinned both ways round. With the forward
		// kernel exp(-2*pi*i*k*j/n), an input of exp(+2*pi*i*j/n) sums to
		// exp(-2*pi*i*j*(k-1)/n) and spikes at bin 1; the conjugate input
		// spikes at bin n-1. Asserting only one of these would pass under the
		// opposite convention with the bins swapped.
		e := make([]complex128, 8)
		for j := range e {
			e[j] = cmplx.Exp(complex(0, 2*math.Pi*float64(j)/8))
		}
		g = simd.FFT(e)
		if cmplx.Abs(g[1]-8) > 1e-12 {
			t.Errorf("exp(+i): bin 1 = %v, want 8", g[1])
		}
		for j := range e {
			e[j] = cmplx.Exp(complex(0, -2*math.Pi*float64(j)/8))
		}
		g = simd.FFT(e)
		if cmplx.Abs(g[7]-8) > 1e-12 {
			t.Errorf("exp(-i): bin 7 = %v, want 8", g[7])
		}
	})

	// Non-powers of two and degenerate lengths are refused, not mistransformed.
	t.Run("refused", func(t *testing.T) {
		for _, n := range []int{0, 3, 5, 6, 7, 100} {
			if p := simd.NewFFTPlan(n); p != nil {
				t.Errorf("NewFFTPlan(%d) = %v, want nil", n, p)
			}
			if simd.FFT(make([]complex128, n)) != nil {
				t.Errorf("FFT of length %d should be nil", n)
			}
		}
	})

	// A plan is reusable and transforming does not modify it or the input.
	t.Run("planReuse", func(t *testing.T) {
		p := simd.NewFFTPlan(64)
		if p.Len() != 64 {
			t.Fatalf("Len = %d, want 64", p.Len())
		}
		src := make([]complex128, 64)
		for i := range src {
			src[i] = complex(float64(i), float64(-i))
		}
		orig := append([]complex128(nil), src...)
		d1 := make([]complex128, 64)
		d2 := make([]complex128, 64)
		simd.FFTInto(p, d1, src)
		simd.FFTInto(p, d2, src)
		for i := range d1 {
			if d1[i] != d2[i] {
				t.Fatalf("plan reuse differs at %d", i)
			}
			if src[i] != orig[i] {
				t.Fatalf("src modified at %d", i)
			}
		}
	})
}

func TestHilbert(t *testing.T) {
	// The defining property: the analytic signal's real part is the input.
	// Anything that gets the doubling or the bin handling wrong breaks this
	// before it breaks anything subtler.
	r := rand.New(rand.NewPCG(107, 109))
	for _, n := range []int{1, 2, 4, 16, 256, 1024} {
		src := make([]float64, n)
		for i := range src {
			src[i] = r.NormFloat64()
		}
		got := simd.Hilbert(src)
		for i := range src {
			if math.Abs(real(got[i])-src[i]) > 1e-11 {
				t.Fatalf("n=%d: real(analytic)[%d] = %v, want %v", n, i, real(got[i]), src[i])
			}
		}
	}

	// A cosine's analytic signal is exp(i*w*t), so its envelope is flat at the
	// amplitude and its imaginary part is the sine. This is the case that
	// catches doubling the DC or Nyquist bin: that shows up as an offset and
	// an alternating ripple in the envelope, which a flat-envelope check sees
	// immediately.
	const n = 1024
	const cycles = 8
	sig := make([]float64, n)
	for i := range sig {
		sig[i] = 3 * math.Cos(2*math.Pi*cycles*float64(i)/n)
	}
	a := simd.Hilbert(sig)
	env := make([]float64, n)
	simd.AbsComplexInto(env, a)
	for i := range env {
		if math.Abs(env[i]-3) > 1e-9 {
			t.Fatalf("envelope[%d] = %v, want 3 (flat)", i, env[i])
		}
		want := 3 * math.Sin(2*math.Pi*cycles*float64(i)/n)
		if math.Abs(imag(a[i])-want) > 1e-9 {
			t.Fatalf("imag[%d] = %v, want %v", i, imag(a[i]), want)
		}
	}

	// An amplitude-modulated carrier: the envelope must recover the
	// modulation, which is what this is actually used for.
	mod := make([]float64, n)
	for i := range mod {
		m := 1 + 0.5*math.Sin(2*math.Pi*2*float64(i)/n)
		mod[i] = m * math.Cos(2*math.Pi*64*float64(i)/n)
	}
	am := simd.Hilbert(mod)
	simd.AbsComplexInto(env, am)
	for i := range env {
		want := 1 + 0.5*math.Sin(2*math.Pi*2*float64(i)/n)
		if math.Abs(env[i]-want) > 1e-3 {
			t.Fatalf("AM envelope[%d] = %v, want %v", i, env[i], want)
		}
	}

	// Refused for non-powers of two, like the FFT it is built on.
	for _, bad := range []int{0, 3, 100} {
		if simd.Hilbert(make([]float64, bad)) != nil {
			t.Errorf("Hilbert of length %d should be nil", bad)
		}
	}
}
