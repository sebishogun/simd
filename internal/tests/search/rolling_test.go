package search

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// naiveRolling is the specification, written out here rather than called from
// the library so the test does not check the implementation against itself.
func naiveRolling[T float64 | int32](a []T, window int, keepMin bool) []T {
	if window <= 0 || window > len(a) {
		return nil
	}
	out := make([]T, len(a)-window+1)
	for i := range out {
		v := a[i]
		for _, x := range a[i+1 : i+window] {
			if keepMin == (x < v) {
				v = x
			}
		}
		out[i] = v
	}
	return out
}

func TestRollingMinMax(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	// Lengths on both sides of the dispatch threshold, of a vector width, and
	// of the kernel's 4096-element tile, so the blocked path, its tail, the
	// tile boundary and the reference path all run.
	for _, n := range []int{1, 2, 7, 15, 16, 17, 63, 64, 65, 1000, 4097, 9000} {
		a := make([]float64, n)
		ai := make([]int32, n)
		for i := range a {
			a[i] = r.NormFloat64()
			ai[i] = int32(r.IntN(2000) - 1000)
		}
		for _, w := range []int{1, 2, 3, 8, 16, 17, 64, 255} {
			if w > n {
				continue
			}
			want := naiveRolling(a, w, true)
			got := make([]float64, len(want))
			simd.RollingMinInto(got, a, w)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("RollingMin float64 n=%d w=%d: at %d got %v want %v",
						n, w, i, got[i], want[i])
				}
			}

			want = naiveRolling(a, w, false)
			simd.RollingMaxInto(got, a, w)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("RollingMax float64 n=%d w=%d: at %d got %v want %v",
						n, w, i, got[i], want[i])
				}
			}

			wantI := naiveRolling(ai, w, true)
			gotI := make([]int32, len(wantI))
			simd.RollingMinInto(gotI, ai, w)
			for i := range wantI {
				if gotI[i] != wantI[i] {
					t.Fatalf("RollingMin int32 n=%d w=%d: at %d got %v want %v",
						n, w, i, gotI[i], wantI[i])
				}
			}
		}
	}
}

// The window extreme is IEEE 754-2019 minimum, the same one Min and Minimum
// use: NaN propagates rather than being skipped, and -0 orders below +0. This
// is what rules out the monotonic-deque implementation, so it is the property
// most worth pinning.
func TestRollingSpecialValues(t *testing.T) {
	nan := math.NaN()
	a := []float64{1, nan, 3, 4, 5, 6, 7, 8}
	got := make([]float64, len(a)-3+1)
	simd.RollingMinInto(got, a, 3)
	for i := range 2 { // windows 0 and 1 contain the NaN at index 1
		if !math.IsNaN(got[i]) {
			t.Errorf("window %d: got %v, want NaN to propagate", i, got[i])
		}
	}
	for i := 2; i < len(got); i++ {
		if math.IsNaN(got[i]) {
			t.Errorf("window %d: NaN leaked past its window", i)
		}
	}

	z := []float64{0, math.Copysign(0, -1), 0, 0}
	gz := make([]float64, 2)
	simd.RollingMinInto(gz, z, 3)
	if !math.Signbit(gz[0]) || !math.Signbit(gz[1]) {
		t.Errorf("RollingMin over {+0,-0,+0,+0} = %v %v; -0 must win", gz[0], gz[1])
	}
	simd.RollingMaxInto(gz, z, 3)
	if math.Signbit(gz[0]) || math.Signbit(gz[1]) {
		t.Errorf("RollingMax over {+0,-0,+0,+0} = %v %v; +0 must win", gz[0], gz[1])
	}
}

// A window that cannot produce output must write nothing rather than panic or
// scribble. The kernel takes three lengths and the guard clamps none of them,
// which is exactly where a mistake would show up.
func TestRollingDegenerateWindows(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	for _, w := range []int{0, -1, 5, 100} {
		dst := []float64{-7, -7, -7, -7}
		simd.RollingMinInto(dst, a, w)
		for i, v := range dst {
			if v != -7 {
				t.Errorf("window %d wrote %v at %d; it should write nothing", w, v, i)
			}
		}
	}
	// A destination shorter than the window count is filled as far as it goes
	// and no further.
	dst := []float64{-7, -7}
	simd.RollingMinInto(dst[:1], a, 2)
	if dst[1] != -7 {
		t.Errorf("wrote past the end of dst: %v", dst)
	}
}

func TestRollingNoAlloc(t *testing.T) {
	a := make([]float64, 4096)
	dst := make([]float64, len(a)-8+1)
	if n := testing.AllocsPerRun(50, func() { simd.RollingMinInto(dst, a, 8) }); n != 0 {
		t.Errorf("RollingMinInto allocated %v times per run", n)
	}
}
