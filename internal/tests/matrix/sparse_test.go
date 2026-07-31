package matrix

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// naiveSparseDot is the definition, accumulated the way the reference does —
// sixteen lanes combined by the fixed tree. A running sum here would disagree
// with every accelerated tier in the last bit or two and the test would be
// measuring the accumulator shape rather than the arithmetic.
//
// The float64 conversion around the multiply is load-bearing and is not a
// no-op. Go fuses a multiply feeding an add into an FMA on arm64, riscv64,
// loong64 and ppc64 — but not on amd64 — while the kernels compile with
// -ffp-contract=off and never fuse. Without the conversion forcing the product
// to round first, this helper agrees with the kernel on the development machine
// and disagrees in the last two bits on four other architectures. It did:
// TestSpMV failed on arm64 with 3.2523576610318923 against 3.2523576610318905,
// and the library was right both times.
func naiveSparseDot(v []float64, idx []int32, x []float64) float64 {
	var acc [16]float64
	n := min(len(v), len(idx))
	for i := range n {
		var xv float64
		if k := int(idx[i]); k >= 0 && k < len(x) {
			xv = x[k]
		}
		acc[i%16] += float64(v[i] * xv)
	}
	for w := 8; w >= 1; w /= 2 {
		for j := range w {
			acc[j] += acc[j+w]
		}
	}
	return acc[0]
}

func TestSparseDot(t *testing.T) {
	r := rand.New(rand.NewPCG(71, 72))
	x := make([]float64, 4096)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	// Lengths on both sides of the sixteen-lane block and its tail.
	for _, n := range []int{0, 1, 15, 16, 17, 31, 32, 33, 100, 1000} {
		v := make([]float64, n)
		idx := make([]int32, n)
		for i := range v {
			v[i] = r.NormFloat64()
			idx[i] = int32(r.IntN(len(x)))
		}
		if got, want := simd.SparseDot(v, idx, x), naiveSparseDot(v, idx, x); got != want {
			t.Fatalf("n=%d: got %v want %v (bits %x vs %x)",
				n, got, want, math.Float64bits(got), math.Float64bits(want))
		}
	}
}

// Out-of-range indices contribute nothing, the same contract GatherInto has.
// Both directions, and a length that spans the vector block, so the masked
// gather and the scalar tail both see one.
func TestSparseDotOutOfRange(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	v := make([]float64, 20)
	idx := make([]int32, 20)
	for i := range v {
		v[i] = 1
		idx[i] = int32(i) - 8 // negative at the start, past the end at the tail
	}
	// Only idx 0..3 are in range, at i = 8..11.
	want := 1.0 + 2 + 3 + 4
	if got := simd.SparseDot(v, idx, x); got != want {
		t.Errorf("got %v want %v", got, want)
	}
	if got := simd.SparseDot(v, idx, nil); got != 0 {
		t.Errorf("every index out of range: got %v want 0", got)
	}
}

func TestSpMV(t *testing.T) {
	r := rand.New(rand.NewPCG(73, 74))
	const rows, cols = 200, 500
	// A row length distribution with short rows, long rows and empty ones,
	// because the row loop's cost balance differs across all three.
	var values []float64
	var colIdx []int32
	rowPtr := []int32{0}
	dense := make([][]float64, rows)
	for i := range dense {
		dense[i] = make([]float64, cols)
	}
	for i := range rows {
		nnz := []int{0, 1, 3, 40, 200}[r.IntN(5)]
		seen := map[int32]bool{}
		for range nnz {
			c := int32(r.IntN(cols))
			if seen[c] {
				continue
			}
			seen[c] = true
			val := r.NormFloat64()
			values = append(values, val)
			colIdx = append(colIdx, c)
			dense[i][c] = val
		}
		rowPtr = append(rowPtr, int32(len(values)))
	}

	x := make([]float64, cols)
	for i := range x {
		x[i] = r.NormFloat64()
	}

	got := make([]float64, rows)
	simd.SpMVInto(got, values, colIdx, rowPtr, x)
	for i := range rows {
		// The dense reference multiplies the same row in the same order the
		// sparse one does, so this is bit equality rather than a tolerance:
		// the nonzeros appear in column order in both.
		lo, hi := rowPtr[i], rowPtr[i+1]
		want := naiveSparseDot(values[lo:hi], colIdx[lo:hi], x)
		if got[i] != want {
			t.Fatalf("row %d (%d nonzeros): got %v want %v", i, hi-lo, got[i], want)
		}
	}
}

// A malformed rowPtr is a data error and must not become an out-of-range slice
// expression inside the loop.
func TestSpMVMalformedRowPtr(t *testing.T) {
	values := []float64{1, 2, 3}
	colIdx := []int32{0, 1, 2}
	x := []float64{10, 20, 30}
	for _, rp := range [][]int32{
		{0, 99},         // past the end of values
		{-5, 2},         // negative start
		{2, 0},          // backwards
		{0},             // no rows
		{0, 1, 2, 3, 4}, // more rows than nonzeros
	} {
		dst := make([]float64, max(len(rp)-1, 1))
		simd.SpMVInto(dst, values, colIdx, rp, x) // must not panic
	}
}

func TestSparseNoAlloc(t *testing.T) {
	r := rand.New(rand.NewPCG(75, 76))
	x := make([]float64, 1024)
	v := make([]float64, 512)
	idx := make([]int32, 512)
	for i := range v {
		v[i] = r.NormFloat64()
		idx[i] = int32(r.IntN(len(x)))
	}
	if n := testing.AllocsPerRun(20, func() { _ = simd.SparseDot(v, idx, x) }); n != 0 {
		t.Errorf("SparseDot allocated %v times per run", n)
	}
}
