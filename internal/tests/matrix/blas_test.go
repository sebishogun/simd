package matrix

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

// The kernels are checked against the portable reference at every length that
// straddles a vector boundary, including the ones below the dispatch threshold
// where the reference is what runs. Exact equality, not a tolerance: these are
// elementwise operations under the bit-identity contract, and the reference
// hoists the row scale exactly as the kernel does.

func TestRankOneMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 43))
	for _, m := range []int{0, 1, 2, 7, 16, 33, 64} {
		for _, n := range []int{0, 1, 3, 15, 16, 17, 64, 129} {
			a := randF64(r, m*n)
			x := randF64(r, m)
			y := randF64(r, n)
			alpha := r.NormFloat64()

			got := append([]float64(nil), a...)
			want := append([]float64(nil), a...)
			simd.RankOneInto(got, x, y, alpha, m, n)
			ref.RankOneFloat(want, x, y, alpha, m, n)

			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("m=%d n=%d: differs at %d: %v vs %v", m, n, i, got[i], want[i])
				}
			}
		}
	}
}

func TestRankOneFloat32(t *testing.T) {
	r := rand.New(rand.NewPCG(47, 53))
	for _, d := range [][2]int{{1, 1}, {8, 8}, {17, 33}, {64, 129}} {
		m, n := d[0], d[1]
		a := randF32(r, m*n)
		x, y := randF32(r, m), randF32(r, n)
		alpha := float32(r.NormFloat64())
		got := append([]float32(nil), a...)
		want := append([]float32(nil), a...)
		simd.RankOneInto(got, x, y, alpha, m, n)
		ref.RankOneFloat(want, x, y, alpha, m, n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("m=%d n=%d: differs at %d", m, n, i)
			}
		}
	}
}

func TestRotateMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(59, 61))
	for _, n := range []int{0, 1, 3, 15, 16, 17, 64, 70, 1000} {
		x, y := randF64(r, n), randF64(r, n)
		// A real rotation, so c*c+s*s is 1 and the values stay in range.
		theta := r.NormFloat64()
		c, s := math.Cos(theta), math.Sin(theta)

		gx, gy := append([]float64(nil), x...), append([]float64(nil), y...)
		wx, wy := append([]float64(nil), x...), append([]float64(nil), y...)
		simd.Rotate(gx, gy, c, s)
		ref.RotateFloat(wx, wy, c, s)
		for i := range wx {
			if gx[i] != wx[i] || gy[i] != wy[i] {
				t.Fatalf("n=%d: differs at %d: (%v,%v) vs (%v,%v)", n, i, gx[i], gy[i], wx[i], wy[i])
			}
		}
	}
}

// The rotation must use the ORIGINAL x[i] in both assignments. Overwriting x
// first and then reading it back gives a different, wrong answer, and it is the
// obvious way to write this loop.
func TestRotateUsesOriginalX(t *testing.T) {
	x := []float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
	y := []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
	c, s := 0.6, 0.8
	simd.Rotate(x, y, c, s)
	// x0 = c*1 + s*0 = 0.6 ; y0 = c*0 - s*1 = -0.8
	if x[0] != 0.6 || y[0] != -0.8 {
		t.Fatalf("got x[0]=%v y[0]=%v, want 0.6 and -0.8", x[0], y[0])
	}
	// If the loop had written x[0] before computing y[0], y[0] would be
	// -s*0.6 = -0.48 instead.
	if y[0] == -0.48 {
		t.Fatal("y was computed from the already-rotated x")
	}
}

func TestSwapMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(67, 71))
	for _, n := range []int{0, 1, 3, 15, 16, 17, 64, 70, 1000} {
		x, y := randF64(r, n), randF64(r, n)
		gx, gy := append([]float64(nil), x...), append([]float64(nil), y...)
		simd.Swap(gx, gy)
		for i := range x {
			if gx[i] != y[i] || gy[i] != x[i] {
				t.Fatalf("n=%d: not swapped at %d", n, i)
			}
		}
	}
}

func TestSwapIntegers(t *testing.T) {
	x := make([]int32, 100)
	y := make([]int32, 100)
	for i := range x {
		x[i], y[i] = int32(i), int32(-i)
	}
	simd.Swap(x, y)
	for i := range x {
		if x[i] != int32(-i) || y[i] != int32(i) {
			t.Fatalf("not swapped at %d", i)
		}
	}
}

// Mismatched lengths must stop at the shorter one rather than reading past it.
func TestSwapAndRotateRespectShorterSlice(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	y := []float64{9, 9, 9}
	simd.Swap(x, y)
	if x[0] != 9 || x[3] != 4 {
		t.Fatalf("swap ran past the shorter slice: x=%v", x[:5])
	}
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	b := []float64{1, 1}
	simd.Rotate(a, b, 1, 0)
	if a[5] != 6 {
		t.Fatalf("rotate ran past the shorter slice: a=%v", a[:8])
	}
}

// RankOne must not write outside the m*n it was given, and must do nothing at
// all when a slice cannot hold the stated dimensions.
func TestRankOneBounds(t *testing.T) {
	a := make([]float64, 4*4+8)
	for i := range a {
		a[i] = -1
	}
	x, y := []float64{1, 1, 1, 1}, []float64{1, 1, 1, 1}
	simd.RankOneInto(a, x, y, 2, 4, 4)
	for i := 16; i < len(a); i++ {
		if a[i] != -1 {
			t.Fatalf("wrote past m*n at index %d", i)
		}
	}

	short := make([]float64, 3)
	simd.RankOneInto(short, x, y, 2, 4, 4)
	for i := range short {
		if short[i] != 0 {
			t.Fatalf("wrote into a slice too short for the dimensions")
		}
	}
}
