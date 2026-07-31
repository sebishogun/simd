package simd_test

import (
	"fmt"
	"math"

	"github.com/sebishogun/simd"
)

// Runnable examples for every operation the README's "which function do I
// want" table names.
//
// That table has always claimed each entry has one. It did not: 100 of its 109
// entries had no example at all, which is the kind of promise that is worth
// less than no promise. TestReadmeTableHasExamples now checks the claim on
// every build, so it cannot drift again.
//
// Each of these is deliberately small enough to read the answer off the page.
// An example whose output you have to trust is documentation of the wrong kind.

// ---------- array math ----------

func ExampleAdd() {
	a := []float64{1, 2, 3}
	simd.Add(a, []float64{10, 20, 30})
	fmt.Println(a)
	// Output: [11 22 33]
}

// AddAll sums any number of slices in one pass over memory, rather than one
// pass per slice.
func ExampleAddAll() {
	dst := make([]float64, 3)
	simd.AddAll(dst,
		[]float64{1, 2, 3},
		[]float64{10, 20, 30},
		[]float64{100, 200, 300})
	fmt.Println(dst)
	// Output: [111 222 333]
}

func ExampleMulAll() {
	dst := make([]float64, 3)
	simd.MulAll(dst, []float64{1, 2, 3}, []float64{2, 2, 2}, []float64{5, 5, 5})
	fmt.Println(dst)
	// Output: [10 20 30]
}

func ExampleAddScalar() {
	a := []float64{1, 2, 3}
	simd.AddScalar(a, 100)
	fmt.Println(a)
	// Output: [101 102 103]
}

// AddScaled is axpy: y += a*x, in one pass rather than a multiply pass and an
// add pass.
func ExampleAddScaled() {
	y := []float64{1, 2, 3}
	simd.AddScaled(y, []float64{10, 10, 10}, 0.5)
	fmt.Println(y)
	// Output: [6 7 8]
}

func ExampleScale() {
	a := []float64{1, 2, 3}
	simd.Scale(a, 10)
	fmt.Println(a)
	// Output: [10 20 30]
}

func ExampleClamp() {
	a := []float64{-5, 0.5, 99}
	simd.Clamp(a, 0, 1)
	fmt.Println(a)
	// Output: [0 0.5 1]
}

func ExampleXor() {
	a := []uint8{0b1100, 0b1010}
	simd.Xor(a, []uint8{0b1010, 0b1010})
	fmt.Println(a)
	// Output: [6 0]
}

func ExampleShl() {
	a := []uint32{1, 2, 3}
	simd.Shl(a, 4)
	fmt.Println(a)
	// Output: [16 32 48]
}

// Rotl rotates rather than shifts, so nothing falls off the end.
func ExampleRotl() {
	a := []uint8{0b10000001}
	simd.Rotl(a, 1)
	fmt.Printf("%08b\n", a[0])
	// Output: 00000011
}

func ExampleOnesCountInto() {
	dst := make([]uint8, 1)
	simd.OnesCountInto(dst, []uint8{0b1011})
	fmt.Println(dst[0])
	// Output: 3
}

func ExampleLeadingZerosInto() {
	dst := make([]uint8, 1)
	simd.LeadingZerosInto(dst, []uint8{0b0001_0000})
	fmt.Println(dst[0])
	// Output: 3
}

func ExampleByteSwapInto() {
	dst := make([]uint32, 1)
	simd.ByteSwapInto(dst, []uint32{0x11223344})
	fmt.Printf("%#08x\n", dst[0])
	// Output: 0x44332211
}

// ---------- reductions ----------

func ExampleSum() {
	fmt.Println(simd.Sum([]float64{1, 2, 3, 4}))
	// Output: 10
}

func ExampleMean() {
	fmt.Println(simd.Mean([]float64{1, 2, 3, 4}))
	// Output: 2.5
}

func ExampleStdDev() {
	fmt.Printf("%.4f\n", simd.StdDev([]float64{2, 4, 4, 4, 5, 5, 7, 9}))
	// Output: 2.0000
}

func ExampleVariance() {
	fmt.Println(simd.Variance([]float64{2, 4, 4, 4, 5, 5, 7, 9}))
	// Output: 4
}

func ExampleNorm() {
	fmt.Println(simd.Norm([]float64{3, 4}))
	// Output: 5
}

func ExampleNormalize() {
	a := []float64{3, 4}
	simd.Normalize(a)
	fmt.Println(a)
	// Output: [0.6 0.8]
}

func ExampleDistance() {
	fmt.Println(simd.Distance([]float64{0, 0}, []float64{3, 4}))
	// Output: 5
}

func ExampleCosineSimilarity() {
	fmt.Println(simd.CosineSimilarity([]float64{1, 0}, []float64{0, 1}))
	// Output: 0
}

// MinMax walks the slice once for both, which matters when the slice does not
// fit in cache.
func ExampleMinMax() {
	lo, hi := simd.MinMax([]float64{3, 1, 4, 1, 5})
	fmt.Println(lo, hi)
	// Output: 1 5
}

func ExampleArgMin() {
	fmt.Println(simd.ArgMin([]float64{3, 1, 4, 1, 5}))
	// Output: 1
}

func ExampleArgMax() {
	fmt.Println(simd.ArgMax([]float64{3, 1, 4, 1, 5}))
	// Output: 4
}

// ---------- NaN-aware ----------

// The mask is supplied rather than allocated, so the whole NaN family runs
// without touching the heap.
func ExampleCountNaN() {
	a := []float64{1, math.NaN(), 3, math.NaN()}
	fmt.Println(simd.CountNaN(a, make([]bool, len(a))))
	// Output: 2
}

func ExampleNanSum() {
	a := []float64{1, math.NaN(), 3}
	fmt.Println(simd.NanSum(a, make([]float64, len(a)), make([]bool, len(a))))
	// Output: 4
}

// NanMean returns the mean and how many values it was taken over.
func ExampleNanMean() {
	a := []float64{1, math.NaN(), 3}
	mean, n := simd.NanMean(a, make([]float64, len(a)), make([]bool, len(a)))
	fmt.Println(mean, "over", n)
	// Output: 2 over 2
}

func ExampleIsNaNInto() {
	mask := make([]bool, 3)
	simd.IsNaNInto(mask, []float64{1, math.NaN(), 3})
	fmt.Println(mask)
	// Output: [false true false]
}

// ---------- scans and ordering ----------

func ExampleCumSum() {
	a := []float64{1, 2, 3, 4}
	simd.CumSum(a)
	fmt.Println(a)
	// Output: [1 3 6 10]
}

// DiffInto writes successive differences, so it produces one fewer element
// than it consumes.
func ExampleDiffInto() {
	dst := make([]float64, 3)
	simd.DiffInto(dst, []float64{1, 3, 6, 10})
	fmt.Println(dst)
	// Output: [2 3 4]
}

func ExampleSort() {
	a := []float64{3, 1, 4, 1, 5}
	simd.Sort(a)
	fmt.Println(a)
	// Output: [1 1 3 4 5]
}

// SortInto is Sort with the workspace handed in, so a loop over many batches
// allocates once rather than once per batch. It sorts a in place; scratch is
// workspace and its contents afterwards are unspecified.
func ExampleSortInto() {
	scratch := make([]float64, 3) // allocate once, reuse
	for _, batch := range [][]float64{{3, 1, 4}, {9, 5, 7}} {
		simd.SortInto(batch, scratch)
		fmt.Println(batch)
	}
	// Output:
	// [1 3 4]
	// [5 7 9]
}

// Argsort returns the permutation rather than reordering, which is what you
// want when several columns share one ordering.
func ExampleArgsort() {
	names := []string{"cherry", "apple", "banana"}
	score := []float64{3, 1, 2}

	order := make([]int32, len(score))
	simd.Argsort(order, score)
	for _, i := range order {
		fmt.Print(names[i], " ")
	}
	fmt.Println()
	// Output: apple banana cherry
}

// PartitionInto splits about a pivot and reports where the split landed. Both
// sides keep their original relative order.
func ExamplePartitionInto() {
	dst := make([]float64, 6)
	n := simd.PartitionInto(dst, []float64{5, 1, 9, 3, 7, 2}, 4)
	fmt.Println(dst[:n], dst[n:])
	// Output: [1 3 2] [5 9 7]
}

func ExampleMedian() {
	fmt.Println(simd.Median([]float64{3, 1, 4, 1, 5}))
	// Output: 3
}

// MedianInto sorts into the scratch slice instead of allocating one.
func ExampleMedianInto() {
	a := []float64{3, 1, 4, 1, 5}
	fmt.Println(simd.MedianInto(a, make([]float64, len(a))))
	// Output: 3
}

func ExampleQuantile() {
	fmt.Println(simd.Quantile([]float64{1, 2, 3, 4, 5}, 0.5))
	// Output: 3
}

func ExampleBottomK() {
	fmt.Println(simd.BottomK([]float64{5, 1, 4, 2, 3}, 3))
	// Output: [1 2 3]
}

func ExampleHistogram() {
	// Five values over [0,10) in 2 bins.
	fmt.Println(simd.Histogram([]float64{1, 2, 3, 8, 9}, 2, 0, 10))
	// Output: [3 2]
}

// ---------- selection and gather ----------

func ExampleSelectInto() {
	dst := make([]float64, 3)
	simd.SelectInto(dst, []bool{true, false, true},
		[]float64{1, 2, 3}, []float64{10, 20, 30})
	fmt.Println(dst)
	// Output: [1 20 3]
}

func ExampleGatherInto() {
	dst := make([]float64, 3)
	simd.GatherInto(dst, []float64{10, 20, 30, 40}, []int32{3, 0, 2})
	fmt.Println(dst)
	// Output: [40 10 30]
}

// FilterInto takes an arbitrary Go predicate, which is convenient and not
// fast: the call per element cannot be vectorized. Use a comparison plus
// CompressInto when it matters.
func ExampleFilterInto() {
	dst := make([]float64, 5)
	n := simd.FilterInto(dst, []float64{1, -2, 3, -4, 5},
		func(v float64) bool { return v > 0 })
	fmt.Println(dst[:n])
	// Output: [1 3 5]
}

// ---------- transcendentals ----------

func ExampleExp() {
	a := []float64{0, 1}
	simd.Exp(a)
	fmt.Printf("%.4f %.4f\n", a[0], a[1])
	// Output: 1.0000 2.7183
}

func ExampleLog() {
	a := []float64{1, math.E}
	simd.Log(a)
	fmt.Printf("%.4f %.4f\n", a[0], a[1])
	// Output: 0.0000 1.0000
}

func ExampleSin() {
	a := []float64{0, math.Pi / 2}
	simd.Sin(a)
	fmt.Printf("%.4f %.4f\n", a[0], a[1])
	// Output: 0.0000 1.0000
}

// PopCount counts set bits across a whole byte slice, as one number.
func ExamplePopCount() {
	fmt.Println(simd.PopCount([]byte{0b1011, 0b1000}))
	// Output: 4
}
