package simd_test

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// Search, sorted sets, bit vectors and sparse matrices — the operations whose
// shape is not "one output per input".

// LowerBoundInto is a binary search done for many queries at once. One search
// is a chain of dependent probes and vectorizes on nothing; a batch turns the
// loop nest inside out and becomes elementwise over the queries.
//
// Each answer is the number of elements strictly less than the query, which is
// the index std::lower_bound and sort.SearchInts return — so a query equal to
// a table entry lands *on* it, not after it. 10 below is at index 1, not 2.
func ExampleLowerBoundInto() {
	table := []float64{0, 10, 20, 30}
	queries := []float64{5, 25, 10, 99}

	pos := make([]int32, len(queries))
	simd.LowerBoundInto(pos, table, queries)
	fmt.Println(pos)
	// Output: [1 3 1 4]
}

// IntersectInto keeps the elements present in both. Both inputs must be sorted
// and free of duplicates — the shape a posting list is already in.
func ExampleIntersectInto() {
	a := []int32{1, 3, 5, 7, 9}
	b := []int32{3, 4, 5, 6, 9}

	dst := make([]int32, min(len(a), len(b)))
	n := simd.IntersectInto(dst, a, b)
	fmt.Println(dst[:n])
	// Output: [3 5 9]
}

// DifferenceInto keeps the elements of a that are not in b.
func ExampleDifferenceInto() {
	a := []int32{1, 3, 5, 7, 9}
	b := []int32{3, 4, 5, 6, 9}

	dst := make([]int32, len(a))
	n := simd.DifferenceInto(dst, a, b)
	fmt.Println(dst[:n])
	// Output: [1 7]
}

// Rank and Select are the pair every succinct structure is built on. The table
// is built once — a population count per word, then a prefix sum — and every
// query afterwards is O(1) or O(log n).
//
// The table is an *exclusive* prefix and has one more entry than the vector,
// which is what makes Rank a single addition with no special case at a word
// boundary and makes Select its exact inverse.
func ExampleRankTableInto() {
	v := []uint64{0b1011, 0b1100} // 3 bits set in word 0, 2 in word 1

	table := make([]uint64, len(v)+1)
	simd.RankTableInto(table, v)

	fmt.Println("set bits below position 3:", simd.Rank(v, table, 3))
	fmt.Println("position of the 3rd set bit:", simd.Select(v, table, 2))
	fmt.Println("total:", table[len(v)])
	// Output:
	// set bits below position 3: 2
	// position of the 3rd set bit: 3
	// total: 5
}

// SparseDot is one row of a sparse matrix-vector product. An index outside x
// contributes nothing rather than panicking, the same contract GatherInto has.
func ExampleSparseDot() {
	// Row with nonzeros at columns 0 and 3.
	values := []float64{2, 5}
	colIdx := []int32{0, 3}
	x := []float64{10, 0, 0, 4}

	fmt.Println(simd.SparseDot(values, colIdx, x)) // 2*10 + 5*4
	// Output: 40
}

// SpMVInto is the row loop written out, for a matrix in compressed sparse row
// form. It allocates nothing and is not parallel: the rows are independent, so
// split rowPtr to use goroutines.
func ExampleSpMVInto() {
	//  [ 1 0 2 ]        [1]        [ 1*1 + 2*3 ]   [7]
	//  [ 0 3 0 ]    *   [2]    =   [ 3*2       ] = [6]
	values := []float64{1, 2, 3}
	colIdx := []int32{0, 2, 1}
	rowPtr := []int32{0, 2, 3}
	x := []float64{1, 2, 3}

	dst := make([]float64, len(rowPtr)-1)
	simd.SpMVInto(dst, values, colIdx, rowPtr, x)
	fmt.Println(dst)
	// Output: [7 6]
}

// RollingMinInto writes the minimum of every window. There are
// len(a)-window+1 outputs.
//
// The extreme is IEEE 754-2019 minimum, so a window containing a NaN yields
// NaN. Below a window of about 48 this beats a hand-written monotonic deque
// several times over; above it, write the deque — the doc comment has the
// measurements.
func ExampleRollingMinInto() {
	a := []float64{5, 1, 4, 2, 8, 3}

	dst := make([]float64, len(a)-3+1)
	simd.RollingMinInto(dst, a, 3)
	fmt.Println(dst)
	// Output: [1 1 2 2]
}

func ExampleRollingMaxInto() {
	a := []float64{5, 1, 4, 2, 8, 3}

	dst := make([]float64, len(a)-3+1)
	simd.RollingMaxInto(dst, a, 3)
	fmt.Println(dst)
	// Output: [5 4 8 8]
}

// Rank counts the set bits below a position. It is exclusive — Rank(v, t, 0)
// is 0 — which is what makes Select its exact inverse.
func ExampleRank() {
	v := []uint64{0b1011}
	table := make([]uint64, len(v)+1)
	simd.RankTableInto(table, v)

	// 0b1011 has bits 0, 1 and 3 set. Below position 2 that is two of them.
	fmt.Println(simd.Rank(v, table, 0), simd.Rank(v, table, 2), simd.Rank(v, table, 64))
	// Output: 0 2 3
}

// Select is Rank's inverse: where is the k-th set bit, counting from zero.
func ExampleSelect() {
	v := []uint64{0b1011}
	table := make([]uint64, len(v)+1)
	simd.RankTableInto(table, v)

	fmt.Println(simd.Select(v, table, 0), simd.Select(v, table, 2), simd.Select(v, table, 9))
	// Output: 0 3 -1
}
