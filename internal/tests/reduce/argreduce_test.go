package reduce

// ArgMin and ArgMax, against the three rules that are easy to get wrong.
//
// The value these return is a position, so an implementation can be arithmetically
// right and still wrong: it can find the correct minimum and report the wrong
// index for it. That is invisible to any test that only checks a[ArgMin(a)] is
// the smallest value, which is why none of the cases below do that.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

// TestArgTiesKeepEarliest pins the first rule: when a value appears more than
// once, the earliest index wins. A vectorized implementation naturally finds
// whichever lane happened to hold it, so this is the case that separates a
// correct horizontal reduction from a plausible one.
func TestArgTiesKeepEarliest(t *testing.T) {
	for _, n := range []int{3, 16, 17, 64, 100, 1000} {
		a := make([]float64, n)
		for i := range a {
			a[i] = 5 // every element identical
		}
		if got := simd.ArgMin(a); got != 0 {
			t.Errorf("n=%d: ArgMin of all-equal = %d, want 0", n, got)
		}
		if got := simd.ArgMax(a); got != 0 {
			t.Errorf("n=%d: ArgMax of all-equal = %d, want 0", n, got)
		}

		// A minimum repeated at two known positions, deliberately in
		// different lanes of a 16-wide block.
		b := make([]float64, n)
		for i := range b {
			b[i] = 100
		}
		if n > 20 {
			b[3], b[19] = -1, -1
			if got := simd.ArgMin(b); got != 3 {
				t.Errorf("n=%d: ArgMin with the min at 3 and 19 = %d, want 3", n, got)
			}
		}
	}
}

// TestArgNaNBeatsEverything pins the second rule, which is the surprising one:
// a NaN anywhere makes the answer the index of the *first* NaN, even when a
// smaller value appeared before it. The reference reaches this by letting NaN
// poison its running best and returning immediately.
func TestArgNaNBeatsEverything(t *testing.T) {
	nan := math.NaN()

	// The minimum is at index 5; the first NaN is later, at 100. The NaN wins.
	a := make([]float64, 200)
	for i := range a {
		a[i] = 50
	}
	a[5] = -1000
	a[100] = nan
	a[150] = nan
	if got := simd.ArgMin(a); got != 100 {
		t.Errorf("ArgMin with min at 5 and first NaN at 100 = %d, want 100", got)
	}
	if got := simd.ArgMax(a); got != 100 {
		t.Errorf("ArgMax with first NaN at 100 = %d, want 100", got)
	}

	// A NaN in the very first position poisons everything from the start.
	b := make([]float64, 64)
	for i := range b {
		b[i] = float64(i)
	}
	b[0] = nan
	if got := simd.ArgMin(b); got != 0 {
		t.Errorf("ArgMin with NaN at 0 = %d, want 0", got)
	}

	// Two NaNs, neither first: the earlier one wins.
	c := make([]float64, 300)
	for i := range c {
		c[i] = 1
	}
	c[201], c[17] = nan, nan
	if got := simd.ArgMin(c); got != 17 {
		t.Errorf("ArgMin with NaNs at 17 and 201 = %d, want 17", got)
	}
}

// TestArgSignedZero pins the third rule, and it is a consequence of the
// reference's formulation rather than a choice. The comparison is `<`, not the
// IEEE-754-2019 total order, so -0 and +0 compare equal and neither displaces
// the other — ArgMin of {+0, -0} is 0, not 1.
func TestArgSignedZero(t *testing.T) {
	negZero := math.Copysign(0, -1)
	a := []float64{0, negZero, 0, negZero}
	if got := simd.ArgMin(a); got != 0 {
		t.Errorf("ArgMin{+0,-0,+0,-0} = %d, want 0 (they compare equal, so the first stays)", got)
	}
	b := []float64{negZero, 0}
	if got := simd.ArgMin(b); got != 0 {
		t.Errorf("ArgMin{-0,+0} = %d, want 0", got)
	}
}

// TestArgAgainstScalar is the broad check: random data at lengths that straddle
// the 16-lane block, compared against an obviously-correct scalar walk.
func TestArgAgainstScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{1, 2, 15, 16, 17, 31, 33, 64, 65, 127, 1000, 4096} {
		a := make([]float64, n)
		for i := range a {
			// A small value range so ties happen often, which is the point.
			a[i] = float64(rng.Intn(7))
		}
		wantMin, wantMax := 0, 0
		for i := 1; i < n; i++ {
			if a[i] < a[wantMin] {
				wantMin = i
			}
			if a[i] > a[wantMax] {
				wantMax = i
			}
		}
		if got := simd.ArgMin(a); got != wantMin {
			t.Fatalf("n=%d: ArgMin = %d, want %d", n, got, wantMin)
		}
		if got := simd.ArgMax(a); got != wantMax {
			t.Fatalf("n=%d: ArgMax = %d, want %d", n, got, wantMax)
		}
	}
}

func TestArgIntegers(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for _, n := range []int{1, 16, 17, 65, 1000} {
		a := make([]int32, n)
		for i := range a {
			a[i] = rng.Int31n(11) - 5
		}
		wantMin, wantMax := 0, 0
		for i := 1; i < n; i++ {
			if a[i] < a[wantMin] {
				wantMin = i
			}
			if a[i] > a[wantMax] {
				wantMax = i
			}
		}
		if got := simd.ArgMin(a); got != wantMin {
			t.Fatalf("n=%d int32: ArgMin = %d, want %d", n, got, wantMin)
		}
		if got := simd.ArgMax(a); got != wantMax {
			t.Fatalf("n=%d int32: ArgMax = %d, want %d", n, got, wantMax)
		}
	}
}
