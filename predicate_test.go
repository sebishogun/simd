package simd_test

import (
	"math"
	"testing"

	"github.com/sebishogun/simd"
)

// The predicates are compositions of accelerated comparisons, so the thing
// worth testing is that each one agrees with the standard library on every
// awkward value — both NaN payloads, both infinities, both zeros, the
// denormals and the finite extremes — and that they keep agreeing above the
// dispatch thresholds, where a different code path runs underneath.
func TestPredicates(t *testing.T) {
	odd := []float64{
		math.NaN(), math.Float64frombits(0x7ff0000000000001), // quiet and signalling
		math.Inf(1), math.Inf(-1),
		0, math.Copysign(0, -1),
		math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
		math.MaxFloat64, -math.MaxFloat64,
		1, -1, 0.5, -3.25,
	}
	for _, n := range []int{1, 7, 16, 64, 65, 256, 1000} {
		a := make([]float64, n)
		for i := range a {
			a[i] = odd[i%len(odd)]
		}
		mask := make([]bool, n)
		scratch := make([]float64, n)

		simd.IsNaNInto(mask, a)
		wantCount := 0
		for i, v := range a {
			if got, want := mask[i], math.IsNaN(v); got != want {
				t.Fatalf("n=%d IsNaN[%d] of %v = %v, want %v", n, i, v, got, want)
			}
			if math.IsNaN(v) {
				wantCount++
			}
		}
		if got := simd.CountNaN(a, mask); got != wantCount {
			t.Fatalf("n=%d CountNaN = %d, want %d", n, got, wantCount)
		}
		if got := simd.AnyNaN(a, mask); got != (wantCount > 0) {
			t.Fatalf("n=%d AnyNaN = %v, want %v", n, got, wantCount > 0)
		}

		simd.IsInfInto(mask, a, scratch)
		for i, v := range a {
			if got, want := mask[i], math.IsInf(v, 0); got != want {
				t.Fatalf("n=%d IsInf[%d] of %v = %v, want %v", n, i, v, got, want)
			}
		}

		simd.IsFiniteInto(mask, a, scratch)
		for i, v := range a {
			want := !math.IsInf(v, 0) && !math.IsNaN(v)
			if mask[i] != want {
				t.Fatalf("n=%d IsFinite[%d] of %v = %v, want %v", n, i, v, mask[i], want)
			}
		}
	}

	// float32 goes through the same generic path and has its own NaN and
	// infinity bit patterns, so it is not covered by the float64 case.
	a32 := []float32{float32(math.NaN()), float32(math.Inf(-1)), 0, 1}
	m32 := make([]bool, len(a32))
	simd.IsNaNInto(m32, a32)
	if !m32[0] || m32[1] || m32[2] || m32[3] {
		t.Errorf("float32 IsNaN = %v, want [true false false false]", m32)
	}

	// NanSum and NanMean against the obvious loop, on data that is part NaN,
	// entirely NaN and entirely finite.
	for _, n := range []int{0, 1, 5, 64, 257, 1000} {
		for _, shape := range []string{"mixed", "allNaN", "noNaN"} {
			a := make([]float64, n)
			for i := range a {
				switch {
				case shape == "allNaN" || (shape == "mixed" && i%3 == 0):
					a[i] = math.NaN()
				default:
					a[i] = float64(i%17) - 8
				}
			}
			var wantSum float64
			wantK := 0
			for _, v := range a {
				if !math.IsNaN(v) {
					wantSum += v
					wantK++
				}
			}
			scratch := make([]float64, n)
			mask := make([]bool, n)
			if got := simd.NanSum(a, scratch, mask); got != wantSum {
				t.Fatalf("NanSum n=%d %s = %v, want %v", n, shape, got, wantSum)
			}
			gotMean, gotK := simd.NanMean(a, scratch, mask)
			if gotK != wantK {
				t.Fatalf("NanMean n=%d %s count = %d, want %d", n, shape, gotK, wantK)
			}
			if wantK == 0 {
				if !math.IsNaN(gotMean) {
					t.Fatalf("NanMean n=%d %s = %v, want NaN", n, shape, gotMean)
				}
			} else if gotMean != wantSum/float64(wantK) {
				t.Fatalf("NanMean n=%d %s = %v, want %v", n, shape, gotMean, wantSum/float64(wantK))
			}
		}
	}

	// Sign, including the two decisions that had to be made rather than
	// inherited: NaN propagates, and both zeros give +0.
	for _, n := range []int{0, 1, 5, 64, 257, 1000} {
		a := make([]float64, n)
		for i := range a {
			a[i] = odd[i%len(odd)]
		}
		dst := make([]float64, n)
		simd.SignInto(dst, a, make([]float64, n), make([]bool, n))
		for i, v := range a {
			switch {
			case math.IsNaN(v):
				if !math.IsNaN(dst[i]) {
					t.Fatalf("Sign n=%d [%d] of NaN = %v, want NaN", n, i, dst[i])
				}
			case v > 0:
				if dst[i] != 1 {
					t.Fatalf("Sign n=%d [%d] of %v = %v, want 1", n, i, v, dst[i])
				}
			case v < 0:
				if dst[i] != -1 {
					t.Fatalf("Sign n=%d [%d] of %v = %v, want -1", n, i, v, dst[i])
				}
			default: // both zeros
				if dst[i] != 0 || math.Signbit(dst[i]) {
					t.Fatalf("Sign n=%d [%d] of %v = %v (signbit %v), want +0",
						n, i, v, dst[i], math.Signbit(dst[i]))
				}
			}
		}
	}

	// A short scratch must cost an allocation, not correctness.
	if got := simd.CountNaN([]float64{math.NaN(), 1}, nil); got != 1 {
		t.Errorf("CountNaN with nil scratch = %d, want 1", got)
	}
}
