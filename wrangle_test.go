package simd_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/sebishogun/simd"
)

// The sizes straddle simd.SelectMinLenForTest, because below it TopK sorts and
// above it it selects — two different code paths that must agree.
func TestTopK(t *testing.T) {
	r := rand.New(rand.NewPCG(53, 59))
	for _, n := range []int{0, 1, 5, 511, 512, 513, 4096} {
		for _, k := range []int{0, 1, 3, 10, n, n + 5} {
			if k < 0 {
				continue
			}
			src := make([]float64, n)
			for i := range src {
				src[i] = r.NormFloat64()
			}
			// Oracle: sort a copy with NaN last, take from each end.
			sorted := append([]float64(nil), src...)
			slices.Sort(sorted)

			t.Run(fmt.Sprintf("n=%d/k=%d", n, k), func(t *testing.T) {
				want := min(k, n)
				got := simd.TopK(append([]float64(nil), src...), k)
				if len(got) != want {
					t.Fatalf("TopK returned %d, want %d", len(got), want)
				}
				for i := range got {
					if exp := sorted[n-1-i]; got[i] != exp {
						t.Fatalf("TopK[%d] = %v, want %v", i, got[i], exp)
					}
				}
				bot := simd.BottomK(append([]float64(nil), src...), k)
				if len(bot) != want {
					t.Fatalf("BottomK returned %d, want %d", len(bot), want)
				}
				for i := range bot {
					if bot[i] != sorted[i] {
						t.Fatalf("BottomK[%d] = %v, want %v", i, bot[i], sorted[i])
					}
				}
			})
		}
	}

	// NaN goes last, as in Sort, so it only appears in TopK when k reaches it.
	t.Run("nan", func(t *testing.T) {
		a := make([]float64, 1000)
		for i := range a {
			a[i] = float64(i)
		}
		a[7], a[900] = math.NaN(), math.NaN()
		got := simd.TopK(append([]float64(nil), a...), 3)
		if !math.IsNaN(got[0]) || !math.IsNaN(got[1]) {
			t.Fatalf("TopK with NaN = %v, want the two NaNs first", got)
		}
		if got[2] != 999 {
			t.Fatalf("TopK[2] = %v, want 999", got[2])
		}
		bot := simd.BottomK(append([]float64(nil), a...), 3)
		for i, v := range bot {
			if math.IsNaN(v) {
				t.Fatalf("BottomK[%d] is NaN; NaN sorts last", i)
			}
		}
	})

	// A short scratch must fall back, not fail.
	t.Run("shortScratch", func(t *testing.T) {
		a := make([]float64, 2000)
		for i := range a {
			a[i] = float64((i * 7919) % 2000)
		}
		dst := make([]float64, 5)
		n := simd.TopKInto(dst, a, 5, nil)
		if n != 5 {
			t.Fatalf("returned %d, want 5", n)
		}
		for i, want := range []float64{1999, 1998, 1997, 1996, 1995} {
			if dst[i] != want {
				t.Fatalf("dst[%d] = %v, want %v", i, dst[i], want)
			}
		}
	})
}

func TestInterp(t *testing.T) {
	xp := []float64{0, 1, 2, 4, 8}
	fp := []float64{0, 10, 20, 40, 80}

	cases := []struct{ x, want float64 }{
		{-1, 0},   // below the first knot clamps
		{0, 0},    // exactly the first knot
		{0.5, 5},  // inside the first interval
		{1, 10},   // exactly a knot
		{3, 30},   // inside a wider interval
		{8, 80},   // exactly the last knot
		{100, 80}, // above the last knot clamps
		{6, 60},   // midpoint of the widest interval
	}
	x := make([]float64, len(cases))
	want := make([]float64, len(cases))
	for i, c := range cases {
		x[i], want[i] = c.x, c.want
	}
	got := simd.Interp(x, xp, fp)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("Interp(%v) = %v, want %v", x[i], got[i], want[i])
		}
	}

	// NaN propagates rather than clamping to the first knot.
	if g := simd.Interp([]float64{math.NaN()}, xp, fp); !math.IsNaN(g[0]) {
		t.Errorf("Interp(NaN) = %v, want NaN", g[0])
	}
	// Degenerate tables.
	if g := simd.Interp([]float64{5}, []float64{3}, []float64{7}); g[0] != 7 {
		t.Errorf("single-knot Interp = %v, want 7", g[0])
	}
	if g := simd.Interp([]float64{5}, nil, nil); len(g) != 1 || g[0] != 0 {
		t.Errorf("empty-table Interp = %v, want the untouched destination", g)
	}
	// A repeated knot must not divide by zero.
	if g := simd.Interp([]float64{1}, []float64{0, 1, 1, 2}, []float64{0, 5, 9, 9}); math.IsNaN(g[0]) {
		t.Errorf("repeated knot gave NaN")
	}

	// Against a brute-force reference on random data.
	r := rand.New(rand.NewPCG(61, 67))
	kn := make([]float64, 64)
	fv := make([]float64, 64)
	for i := range kn {
		kn[i] = float64(i) * 1.5
		fv[i] = r.NormFloat64()
	}
	xs := make([]float64, 1000)
	for i := range xs {
		xs[i] = r.Float64() * 110
	}
	gs := simd.Interp(xs, kn, fv)
	for i, v := range xs {
		var exp float64
		switch {
		case v <= kn[0]:
			exp = fv[0]
		case v >= kn[len(kn)-1]:
			exp = fv[len(fv)-1]
		default:
			j := 0
			for j+1 < len(kn) && kn[j+1] <= v {
				j++
			}
			exp = fv[j] + (v-kn[j])*(fv[j+1]-fv[j])/(kn[j+1]-kn[j])
		}
		if math.Abs(gs[i]-exp) > 1e-9 {
			t.Fatalf("Interp(%v) = %v, want %v", v, gs[i], exp)
		}
	}
}

func TestTranspose(t *testing.T) {
	r := rand.New(rand.NewPCG(71, 73))
	// Sizes straddle the 1024-element threshold and the 32-element block, so
	// both the blocked path and its ragged edges are exercised — a matrix
	// whose dimensions are exact multiples of 32 would never test the partial
	// blocks.
	for _, dims := range [][2]int{
		{0, 0}, {1, 1}, {1, 7}, {7, 1}, {3, 5}, {32, 32}, {33, 31},
		{31, 33}, {64, 16}, {100, 11}, {128, 128}, {200, 7},
	} {
		m, n := dims[0], dims[1]
		t.Run(fmt.Sprintf("%dx%d", m, n), func(t *testing.T) {
			a := make([]float64, m*n)
			for i := range a {
				a[i] = r.NormFloat64()
			}
			got := simd.Transpose(a, m, n)
			for i := range m {
				for j := range n {
					if got[j*m+i] != a[i*n+j] {
						t.Fatalf("T[%d][%d] = %v, want %v", j, i, got[j*m+i], a[i*n+j])
					}
				}
			}
			// Transposing twice is the identity.
			back := simd.Transpose(got, n, m)
			for i := range a {
				if back[i] != a[i] {
					t.Fatalf("double transpose differs at %d", i)
				}
			}
		})
	}

	// Integer element types go through the same kernels.
	t.Run("int32", func(t *testing.T) {
		m, n := 40, 60
		a := make([]int32, m*n)
		for i := range a {
			a[i] = int32(i)
		}
		got := simd.Transpose(a, m, n)
		for i := range m {
			for j := range n {
				if got[j*m+i] != a[i*n+j] {
					t.Fatalf("int T[%d][%d] wrong", j, i)
				}
			}
		}
	})

	// Short slices do nothing rather than panicking, as MatMulInto does.
	t.Run("short", func(t *testing.T) {
		dst := make([]float64, 4)
		simd.TransposeInto(dst, []float64{1, 2}, 3, 3)
		for i, v := range dst {
			if v != 0 {
				t.Fatalf("dst[%d] = %v, want it untouched", i, v)
			}
		}
		if simd.Transpose([]float64{1, 2}, 3, 3) != nil {
			t.Error("Transpose with a short input should return nil")
		}
	})
}
