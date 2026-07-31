package arrays

// The n-ary forms.
//
// The property that matters is not that the answer is close, it is that the
// answer is *the same bits* as writing the binary calls out by hand. Floating
// point addition is not associative, so an implementation that reassociated —
// or that a compiler was allowed to reassociate — would give a different and
// equally defensible answer, and a caller who mixed the two forms would see
// their results change for no visible reason.
//
// So every test below compares against the explicit binary sequence rather than
// against a mathematical ideal, and does it with values whose partial sums are
// not exactly representable, because exact arithmetic hides exactly this bug.

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

func naryInputs(count, n int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	srcs := make([][]float64, count)
	for i := range srcs {
		srcs[i] = make([]float64, n)
		for j := range srcs[i] {
			// Values that make the partial sums inexact, so the order of
			// accumulation is observable in the result.
			srcs[i][j] = rng.NormFloat64() * math.Pow(10, float64(rng.Intn(6)-3))
		}
	}
	return srcs
}

// binaryAdd is the sequence AddAll promises to reproduce.
func binaryAdd(dst []float64, srcs [][]float64) {
	switch len(srcs) {
	case 0:
		for i := range dst {
			dst[i] = 0
		}
		return
	case 1:
		copy(dst, srcs[0])
		return
	}
	simd.AddInto(dst, srcs[0], srcs[1])
	for _, s := range srcs[2:] {
		simd.AddInto(dst, dst, s)
	}
}

func TestAddAllMatchesBinary(t *testing.T) {
	// Counts straddle the arity-4 kernel and the fold boundary above it.
	for _, count := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16} {
		for _, n := range []int{0, 1, 15, 16, 17, 255, 256, 257, 4096} {
			srcs := naryInputs(count, n, int64(count*1000+n))

			got := make([]float64, n)
			want := make([]float64, n)
			simd.AddAll(got, srcs...)
			binaryAdd(want, srcs)

			for i := range want {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
					t.Fatalf("count=%d n=%d: dst[%d] = %v (%#016x), binary gave %v (%#016x)",
						count, n, i, got[i], math.Float64bits(got[i]),
						want[i], math.Float64bits(want[i]))
				}
			}
		}
	}
}

func TestMulAllMatchesBinary(t *testing.T) {
	for _, count := range []int{0, 1, 2, 3, 4, 5, 9} {
		for _, n := range []int{0, 17, 256, 1000} {
			srcs := naryInputs(count, n, int64(count*77+n))

			got := make([]float64, n)
			simd.MulAll(got, srcs...)

			want := make([]float64, n)
			switch len(srcs) {
			case 0:
				for i := range want {
					want[i] = 1
				}
			case 1:
				copy(want, srcs[0])
			default:
				simd.MulInto(want, srcs[0], srcs[1])
				for _, s := range srcs[2:] {
					simd.MulInto(want, want, s)
				}
			}
			for i := range want {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
					t.Fatalf("count=%d n=%d: dst[%d] = %v, binary gave %v",
						count, n, i, got[i], want[i])
				}
			}
		}
	}
}

// TestAddAllOrderMatters is the bit-exactness promise stated as a test: the
// order of the sources is part of the answer, and the implementation must not
// reorder them. If this ever stops finding a difference, something has started
// reassociating and the guarantee has quietly become false.
//
// Choosing inputs for this is less obvious than it looks. The first attempt
// used [big, r, r, -big], which fails to distinguish anything: the small terms
// are below the ULP of the large one and are lost in *both* orders, so both
// come back exactly zero. The difference has to be arranged so the cancellation
// lands at a different point in each order:
//
//	forward   ((1 + 1e17) + 1) + -1e17  =  (1e17 + 1) + -1e17  =  0
//	reversed  ((-1e17 + 1) + 1) + 1e17  =  (-1e17 + 1) + 1e17  =  1
//
// The 1s vanish into 1e17 on the way up and survive on the way down.
func TestAddAllOrderMatters(t *testing.T) {
	const n = 512
	ones, big := make([]float64, n), make([]float64, n)
	ones2, negBig := make([]float64, n), make([]float64, n)
	for i := range n {
		ones[i], ones2[i] = 1, 1
		big[i], negBig[i] = 1e17, -1e17
	}

	fwd := make([]float64, n)
	rev := make([]float64, n)
	simd.AddAll(fwd, ones, big, ones2, negBig)
	simd.AddAll(rev, negBig, ones2, big, ones)

	for i := range n {
		if fwd[i] != 0 {
			t.Fatalf("forward at %d = %v, want 0", i, fwd[i])
		}
		if rev[i] != 1 {
			t.Fatalf("reversed at %d = %v, want 1", i, rev[i])
		}
	}
}

func TestNaryIntegerTypes(t *testing.T) {
	a := []int32{1, 2, 3, 4}
	b := []int32{10, 20, 30, 40}
	c := []int32{100, 200, 300, 400}
	d := []int32{1000, 2000, 3000, 4000}

	dst := make([]int32, 4)
	simd.AddAll(dst, a, b, c, d)
	for i := range dst {
		if want := a[i] + b[i] + c[i] + d[i]; dst[i] != want {
			t.Fatalf("dst[%d] = %d, want %d", i, dst[i], want)
		}
	}

	simd.MulAll(dst, a, b, c)
	for i := range dst {
		if want := a[i] * b[i] * c[i]; dst[i] != want {
			t.Fatalf("mul dst[%d] = %d, want %d", i, dst[i], want)
		}
	}
}

// TestNaryShortestBounds checks the length rule, which is the same one the rest
// of the library uses: the work is bounded by the shortest slice involved.
func TestNaryShortestBounds(t *testing.T) {
	a := []float64{1, 1, 1, 1, 1}
	b := []float64{2, 2, 2}
	c := []float64{3, 3, 3, 3}
	dst := []float64{-1, -1, -1, -1, -1}

	simd.AddAll(dst, a, b, c)
	for i := range 3 {
		if dst[i] != 6 {
			t.Fatalf("dst[%d] = %v, want 6", i, dst[i])
		}
	}
	for i := 3; i < len(dst); i++ {
		if dst[i] != -1 {
			t.Fatalf("dst[%d] = %v; past the shortest source it should be untouched", i, dst[i])
		}
	}
}

func TestNarySpecialValues(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	a := []float64{inf, 1, 0, nan}
	b := []float64{-inf, 2, 0, 1}
	c := []float64{1, 3, math.Copysign(0, -1), 1}

	dst := make([]float64, 4)
	simd.AddAll(dst, a, b, c)

	if !math.IsNaN(dst[0]) {
		t.Errorf("Inf + -Inf + 1 = %v, want NaN", dst[0])
	}
	if dst[1] != 6 {
		t.Errorf("1+2+3 = %v, want 6", dst[1])
	}
	// (+0 + +0) + -0 is +0: only -0 + -0 gives -0.
	if dst[2] != 0 || math.Signbit(dst[2]) {
		t.Errorf("0 + 0 + -0 = %v (signbit %v), want +0", dst[2], math.Signbit(dst[2]))
	}
	if !math.IsNaN(dst[3]) {
		t.Errorf("NaN + 1 + 1 = %v, want NaN", dst[3])
	}
}

// The benchmark that decides whether any of this earns its place: one pass
// against the repeated binary calls it replaces. The interesting axis is size,
// because what is being saved is memory traffic and there is none to save while
// the data is in cache.
func BenchmarkAddAll(b *testing.B) {
	for _, count := range []int{3, 4, 8} {
		for _, n := range []int{256, 4096, 65536, 1 << 20} {
			srcs := make([][]float32, count)
			for i := range srcs {
				srcs[i] = make([]float32, n)
				for j := range srcs[i] {
					srcs[i][j] = float32(i*j%97) * 0.5
				}
			}
			dst := make([]float32, n)

			b.Run(fmt.Sprintf("srcs=%d/n=%d/impl=nary", count, n), func(b *testing.B) {
				b.SetBytes(int64(n) * 4 * int64(count))
				for b.Loop() {
					simd.AddAll(dst, srcs...)
				}
			})
			b.Run(fmt.Sprintf("srcs=%d/n=%d/impl=binary", count, n), func(b *testing.B) {
				b.SetBytes(int64(n) * 4 * int64(count))
				for b.Loop() {
					simd.AddInto(dst, srcs[0], srcs[1])
					for _, s := range srcs[2:] {
						simd.AddInto(dst, dst, s)
					}
				}
			})
		}
	}
}
