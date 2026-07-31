package matrix

import (
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/sebishogun/simd"
)

// The whole claim of the parallel variants is that they change the schedule and
// not the answer. Splitting by output row leaves each element's summation over
// k alone, so this must hold exactly rather than approximately — and it has to
// be checked, because "the blocking cannot depend on m" is an assumption about
// a generated kernel rather than something the type system enforces.
func TestMatMulParallelIsBitIdentical(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 13))
	// Sizes that straddle the work threshold and the packing cliff, and shapes
	// where m does not divide evenly by the worker count.
	for _, d := range [][3]int{
		{1, 1, 1}, {3, 5, 7}, {17, 33, 9}, {64, 64, 64},
		{100, 64, 100}, {128, 128, 128}, {129, 127, 131}, {256, 256, 256},
	} {
		m, k, n := d[0], d[1], d[2]
		a := randF64(r, m*k)
		b := randF64(r, k*n)

		want := make([]float64, m*n)
		simd.MatMulInto(want, a, b, m, k, n)

		got := make([]float64, m*n)
		simd.MatMulParallelInto(got, a, b, m, k, n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("m=%d k=%d n=%d: MatMulParallelInto differs at %d: %v vs %v",
					m, k, n, i, got[i], want[i])
			}
		}
	}
}

func TestMatMulParallelFloat32(t *testing.T) {
	r := rand.New(rand.NewPCG(17, 19))
	for _, d := range [][3]int{{64, 64, 64}, {129, 65, 127}, {256, 128, 256}} {
		m, k, n := d[0], d[1], d[2]
		a, b := randF32(r, m*k), randF32(r, k*n)
		want, got := make([]float32, m*n), make([]float32, m*n)
		simd.MatMulInto(want, a, b, m, k, n)
		simd.MatMulParallelInto(got, a, b, m, k, n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("m=%d k=%d n=%d: differs at %d: %v vs %v", m, k, n, i, got[i], want[i])
			}
		}
	}
}

// A malformed call must do whatever the serial one does, which is not to
// panic: MatMulInto writes nothing when the destination is too short. The
// parallel variant must not invent a panic that the function it mirrors never
// had, and must not raise one inside a worker goroutine, where a caller's
// recover cannot reach it.
func TestMatMulParallelMatchesSerialOnShortSlices(t *testing.T) {
	a := make([]float64, 128*128)
	b := make([]float64, 128*128)
	for _, c := range []struct {
		name               string
		dstLen, aLen, bLen int
	}{
		{"short dst", 2, 128 * 128, 128 * 128},
		{"short a", 128 * 128, 2, 128 * 128},
		{"short b", 128 * 128, 128 * 128, 2},
		{"all empty", 0, 0, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			want := make([]float64, c.dstLen)
			got := make([]float64, c.dstLen)
			serial := recovered(func() { simd.MatMulInto(want, a[:c.aLen], b[:c.bLen], 128, 128, 128) })
			par := recovered(func() { simd.MatMulParallelInto(got, a[:c.aLen], b[:c.bLen], 128, 128, 128) })
			if (serial == nil) != (par == nil) {
				t.Fatalf("serial panic=%v but parallel panic=%v", serial, par)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("result differs from the serial call at %d", i)
				}
			}
		})
	}
}

func recovered(f func()) (v any) {
	defer func() { v = recover() }()
	f()
	return nil
}

func TestMatMulParallelSingleProc(t *testing.T) {
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	r := rand.New(rand.NewPCG(23, 29))
	m, k, n := 128, 128, 128
	a, b := randF64(r, m*k), randF64(r, k*n)
	want, got := make([]float64, m*n), make([]float64, m*n)
	simd.MatMulInto(want, a, b, m, k, n)
	simd.MatMulParallelInto(got, a, b, m, k, n)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("GOMAXPROCS=1 must fall back to the serial kernel; differs at %d", i)
		}
	}
}

func randF64(r *rand.Rand, n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64()
	}
	return s
}

func randF32(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64())
	}
	return s
}

func TestGemvParallelIsBitIdentical(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 37))
	for _, d := range [][2]int{{1, 1}, {5, 7}, {64, 64}, {1000, 1000}, {1024, 1024}, {1023, 1025}} {
		m, k := d[0], d[1]
		a, x := randF64(r, m*k), randF64(r, k)
		want, got := make([]float64, m), make([]float64, m)
		simd.GemvInto(want, a, x, m, k)
		simd.GemvParallelInto(got, a, x, m, k)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("m=%d k=%d: differs at %d: %v vs %v", m, k, i, got[i], want[i])
			}
		}
	}
}
