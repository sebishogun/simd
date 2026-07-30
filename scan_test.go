package simd_test

// The Fast scans make two promises and break one, and all three need a test.
//
//  1. They are NOT equal to the serial loop. If they were, the Fast prefix
//     would be a lie and CumSum should just be this.
//  2. They ARE equal across every tier, including the portable one. This is
//     the promise that is actually load-bearing, and it is what the
//     conformance suite checks by running the whole thing under each GOSIMD.
//  3. They are close to the serial loop, and on hard input closer to the
//     truth than it is.
//
// The integer CumProd is the opposite case: it now runs the same blocked
// algorithm and must be EXACTLY the serial loop, including where it overflows.

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// serialCumSum and serialCumProd are the naive loops, written here rather than
// taken from the package so the oracle cannot drift with the implementation.
func serialCumSum[T float32 | float64](a []T) []T {
	out := make([]T, len(a))
	var r T
	for i, v := range a {
		r += v
		out[i] = r
	}
	return out
}

func serialCumProd[T float32 | float64](a []T) []T {
	out := make([]T, len(a))
	r := T(1)
	for i, v := range a {
		r *= v
		out[i] = r
	}
	return out
}

func TestFastCumSum(t *testing.T) {
	r := rand.New(rand.NewPCG(271, 277))

	// Lengths straddling the block size in both element widths — 16 for
	// float32, 8 for float64 — and the tail past the last whole block.
	for _, n := range []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 1000, 100000} {
		a64 := make([]float64, n)
		a32 := make([]float32, n)
		for i := range a64 {
			a64[i] = r.NormFloat64()
			a32[i] = float32(a64[i])
		}

		t.Run(fmt.Sprintf("n=%d/close", n), func(t *testing.T) {
			got := make([]float64, n)
			simd.FastCumSumInto(got, a64)
			want := serialCumSum(a64)
			for i := range want {
				// Both are sums of the same i+1 values; they may differ, but
				// only by accumulated rounding. A generous absolute bound
				// scaled by the running magnitude catches a real bug (a
				// dropped term, a wrong lane) without asserting bit equality
				// this deliberately does not have.
				scale := math.Abs(want[i]) + 1
				if diff := math.Abs(got[i] - want[i]); diff > 1e-9*scale {
					t.Fatalf("element %d: got %v, want ~%v (diff %g)",
						i, got[i], want[i], diff)
				}
			}
			g32 := make([]float32, n)
			simd.FastCumSumInto(g32, a32)
			w32 := serialCumSum(a32)
			for i := range w32 {
				scale := float64(absF32(w32[i])) + 1
				if diff := math.Abs(float64(g32[i] - w32[i])); diff > 1e-3*scale {
					t.Fatalf("f32 element %d: got %v, want ~%v", i, g32[i], w32[i])
				}
			}
		})

		t.Run(fmt.Sprintf("n=%d/prod", n), func(t *testing.T) {
			// Values near one, so a running product neither overflows nor
			// underflows over 100,000 terms.
			b := make([]float64, n)
			for i := range b {
				b[i] = 1 + r.NormFloat64()*1e-6
			}
			got := make([]float64, n)
			simd.FastCumProdInto(got, b)
			want := serialCumProd(b)
			for i := range want {
				if diff := math.Abs(got[i] - want[i]); diff > 1e-9*math.Abs(want[i]) {
					t.Fatalf("prod element %d: got %v, want ~%v", i, got[i], want[i])
				}
			}
		})
	}

	// In place, which is the non-Into form.
	t.Run("inPlace", func(t *testing.T) {
		a := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		want := serialCumSum(a)
		b := append([]float64(nil), a...)
		simd.FastCumSum(b)
		for i := range want {
			if math.Abs(b[i]-want[i]) > 1e-12 {
				t.Fatalf("in place element %d: got %v, want %v", i, b[i], want[i])
			}
		}
	})

	// The point of the Fast prefix: on input long enough for the grouping to
	// matter, it must actually differ from the serial loop. A test that only
	// checked closeness would pass if FastCumSum silently became CumSum.
	//
	// float32 only. On float64 FastCumSum IS CumSum, deliberately — the
	// blocked form measured 0.91x there, and a Fast function slower than the
	// plain one is not worth its name. The float64 half of that contract is
	// asserted just below, in the opposite direction.
	t.Run("differsFromSerial", func(t *testing.T) {
		a := make([]float32, 10000)
		for i := range a {
			a[i] = float32(r.NormFloat64())
		}
		got := make([]float32, len(a))
		simd.FastCumSumInto(got, a)
		want := serialCumSum(a)
		differ := 0
		for i := range want {
			if got[i] != want[i] {
				differ++
			}
		}
		if differ == 0 {
			t.Fatal("FastCumSum on float32 matched the serial loop exactly " +
				"on 10,000 random values — either it is not the blocked scan " +
				"any more, or the dispatcher never reached the kernel")
		}
	})

	// And the documented float64 behaviour: identical to CumSum, bit for bit.
	// If a float64 kernel is ever added, this fails and the doc comment has to
	// be revisited rather than quietly becoming untrue.
	t.Run("float64IsSerial", func(t *testing.T) {
		a := make([]float64, 10000)
		for i := range a {
			a[i] = r.NormFloat64()
		}
		fast := make([]float64, len(a))
		plain := make([]float64, len(a))
		simd.FastCumSumInto(fast, a)
		simd.CumSumInto(plain, a)
		for i := range a {
			if fast[i] != plain[i] {
				t.Fatalf("float64 element %d: FastCumSum %v, CumSum %v — the "+
					"doc comment says these are the same function on float64",
					i, fast[i], plain[i])
			}
		}
	})

	// And the accuracy claim in the doc comment, on the case that makes it: a
	// huge first value followed by ones, where a serial accumulator cannot
	// represent the increment at all. float32 again, since that is where the
	// blocked scan runs; the float32 spacing at 1e8 is 8, so the increment is
	// lost just as thoroughly.
	t.Run("moreAccurateThanSerial", func(t *testing.T) {
		const n = 100000
		a := make([]float32, n)
		a[0] = 1e8
		for i := 1; i < n; i++ {
			a[i] = 1
		}
		got := make([]float32, n)
		simd.FastCumSumInto(got, a)
		want := serialCumSum(a)

		// The truth is exact in float64 here: 1e8 + k for k under 100,000.
		var fastErr, serialErr float64
		for i := range a {
			truth := 1e8 + float64(i)
			fastErr += math.Abs(float64(got[i]) - truth)
			serialErr += math.Abs(float64(want[i]) - truth)
		}
		if fastErr >= serialErr {
			t.Fatalf("FastCumSum total error %g is not below CumSum's %g on "+
				"1e8-then-ones, but the doc comment says it is",
				fastErr, serialErr)
		}
		t.Logf("total absolute error over %d elements: fast %g, serial %g (%.0fx better)",
			n, fastErr, serialErr, serialErr/max(fastErr, 1))
	})
}

func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// The integer product scan is EXACT, so it gets equality and not a tolerance —
// including on inputs chosen to overflow, since two's-complement
// multiplication is associative precisely because wrapping is exact.
func TestCumProdIntExact(t *testing.T) {
	r := rand.New(rand.NewPCG(281, 283))
	for _, n := range []int{0, 1, 15, 16, 17, 1000, 100000} {
		a := make([]int32, n)
		for i := range a {
			a[i] = int32(r.IntN(11) + 2) // grows fast, overflows quickly
		}
		got := make([]int32, n)
		simd.CumProdInto(got, a)

		want := make([]int32, n)
		run := int32(1)
		for i, v := range a {
			run *= v
			want[i] = run
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d element %d: got %d, want %d (exactness is the "+
					"whole claim for the integer scan)", n, i, got[i], want[i])
			}
		}
	}

	// Negative values and zeros, which change the sign pattern and sink the
	// running product to zero partway through.
	a := []int32{-3, 7, 0, 5, -2, -11, 1 << 30, 3, -1}
	got := make([]int32, len(a))
	simd.CumProdInto(got, a)
	run := int32(1)
	for i, v := range a {
		run *= v
		if got[i] != run {
			t.Fatalf("element %d: got %d, want %d", i, got[i], run)
		}
	}
}
