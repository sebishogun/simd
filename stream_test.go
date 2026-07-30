package simd_test

import (
	"math"
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

// The whole promise: the answer must not depend on where the chunks fell.
//
// This is not a tolerance check. Floating-point addition is not associative,
// so a streaming sum that used one running total would give a different last
// bit for a different chunking — and would pass any test written with a
// tolerance. Bit equality is the only check that distinguishes the two
// designs.
func TestAccumulatorMatchesWholeSliceExactly(t *testing.T) {
	r := rand.New(rand.NewPCG(601, 607))
	for _, n := range []int{0, 1, 15, 16, 17, 31, 33, 64, 1000, 4096} {
		a := make([]float64, n)
		for i := range a {
			// A wide dynamic range, which is where a different summation
			// order actually changes the answer. Uniform values would hide it.
			a[i] = r.NormFloat64() * math.Pow(10, float64(r.IntN(12)-6))
		}
		want := simd.Sum(a)

		// Chunk sizes chosen to fall on, either side of, and far from the
		// sixteen-element lane boundary.
		for _, chunk := range []int{1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 63, 100, 4096} {
			var acc simd.Accumulator[float64]
			for i := 0; i < n; i += chunk {
				acc.Add(a[i:min(i+chunk, n)])
			}
			if got := acc.Sum(); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("n=%d chunk=%d: streaming %v (%#016x) != whole-slice %v (%#016x)",
					n, chunk, got, math.Float64bits(got), want, math.Float64bits(want))
			}
			if acc.Count() != n {
				t.Fatalf("n=%d chunk=%d: Count = %d", n, chunk, acc.Count())
			}
		}
	}
}

func TestAccumulatorFloat32(t *testing.T) {
	r := rand.New(rand.NewPCG(611, 613))
	for _, n := range []int{17, 100, 1000} {
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(r.NormFloat64()) * 1e3
		}
		want := simd.Sum(a)
		for _, chunk := range []int{1, 7, 16, 17, 64} {
			var acc simd.Accumulator[float32]
			for i := 0; i < n; i += chunk {
				acc.Add(a[i:min(i+chunk, n)])
			}
			if got := acc.Sum(); math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("n=%d chunk=%d: %v != %v", n, chunk, got, want)
			}
		}
	}
}

// Ragged chunks — the case a real reader produces, where no two are the same
// size and none is a multiple of the lane count.
func TestAccumulatorRaggedChunks(t *testing.T) {
	r := rand.New(rand.NewPCG(617, 619))
	const n = 5000
	a := make([]float64, n)
	for i := range a {
		a[i] = r.NormFloat64() * 1e4
	}
	want := simd.Sum(a)

	var acc simd.Accumulator[float64]
	for i := 0; i < n; {
		k := 1 + r.IntN(40)
		if i+k > n {
			k = n - i
		}
		acc.Add(a[i : i+k])
		i += k
	}
	if got := acc.Sum(); math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("ragged: %v (%#016x) != %v (%#016x)",
			got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func TestAccumulatorMeanAndReset(t *testing.T) {
	var acc simd.Accumulator[float64]
	if acc.Mean() != 0 || acc.Count() != 0 {
		t.Errorf("zero value: Mean=%v Count=%d", acc.Mean(), acc.Count())
	}
	acc.Add([]float64{1, 2, 3, 4})
	if got := acc.Mean(); got != 2.5 {
		t.Errorf("Mean = %v, want 2.5", got)
	}
	acc.Reset()
	if acc.Count() != 0 || acc.Sum() != 0 {
		t.Errorf("after Reset: Count=%d Sum=%v", acc.Count(), acc.Sum())
	}
}

func TestAccumulatorNoAlloc(t *testing.T) {
	a := make([]float64, 1000)
	var acc simd.Accumulator[float64]
	if n := testing.AllocsPerRun(50, func() { acc.Add(a) }); n != 0 {
		t.Errorf("Add allocated %v times per run, want 0", n)
	}
}

func TestMinMaxAccumulator(t *testing.T) {
	r := rand.New(rand.NewPCG(623, 629))
	const n = 1000
	a := make([]float64, n)
	for i := range a {
		a[i] = r.NormFloat64()
	}
	wantLo, wantHi := simd.MinMax(a)

	var acc simd.MinMaxAccumulator[float64]
	if _, _, ok := acc.MinMax(); ok {
		t.Error("zero value reports ok")
	}
	for i := 0; i < n; i += 37 {
		acc.Add(a[i:min(i+37, n)])
	}
	lo, hi, ok := acc.MinMax()
	if !ok || lo != wantLo || hi != wantHi {
		t.Errorf("got (%v, %v, %v), want (%v, %v, true)", lo, hi, ok, wantLo, wantHi)
	}
}

// NaN must survive a chunk boundary. A plain comparison would drop it and no
// later chunk could bring it back.
func TestMinMaxAccumulatorNaN(t *testing.T) {
	var acc simd.MinMaxAccumulator[float64]
	acc.Add([]float64{1, 2, 3})
	acc.Add([]float64{math.NaN()})
	acc.Add([]float64{4, 5, 6})
	lo, hi, _ := acc.MinMax()
	if !math.IsNaN(lo) || !math.IsNaN(hi) {
		t.Errorf("after a NaN chunk: min=%v max=%v, want both NaN", lo, hi)
	}
}

func TestIntAccumulator(t *testing.T) {
	r := rand.New(rand.NewPCG(631, 641))
	const n = 1000
	a := make([]int32, n)
	for i := range a {
		a[i] = int32(r.Uint32())
	}
	want := simd.Sum(a)

	for _, chunk := range []int{1, 16, 17, 333} {
		var acc simd.IntAccumulator[int32]
		for i := 0; i < n; i += chunk {
			acc.Add(a[i:min(i+chunk, n)])
		}
		if got := acc.Sum(); got != want {
			t.Errorf("chunk=%d: %d != %d", chunk, got, want)
		}
		if acc.Count() != n {
			t.Errorf("chunk=%d: Count = %d", chunk, acc.Count())
		}
	}
}
