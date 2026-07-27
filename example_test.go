package simd_test

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// The plain name works in place and allocates nothing.
func Example() {
	a := []float32{1, 2, 3, 4}
	b := []float32{10, 20, 30, 40}

	simd.Add(a, b) // a += b
	fmt.Println("a += b:", a)

	simd.Scale(a, 0.5) // a *= 0.5
	fmt.Println("a *= .5:", a)

	fmt.Println("sum:", simd.Sum(a))
	fmt.Println("max:", simd.Max(a))

	// Output:
	// a += b: [11 22 33 44]
	// a *= .5: [5.5 11 16.5 22]
	// sum: 55
	// max: 22
}

// Use the Into form when the result belongs somewhere else.
func Example_into() {
	a := []float64{1, 2, 3, 4}
	b := []float64{10, 20, 30, 40}
	dst := make([]float64, 4)

	simd.AddInto(dst, a, b)
	fmt.Println(dst)
	fmt.Println(a, "unchanged")

	// Output:
	// [11 22 33 44]
	// [1 2 3 4] unchanged
}

// The element type is inferred; there are no per-type function names.
func Example_types() {
	i64 := []int64{1 << 40, 2 << 40}
	simd.Add(i64, i64)
	fmt.Println(i64)

	f64 := []float64{0.5, 0.25, 0.125}
	fmt.Println(simd.Sum(f64))

	i32 := []int32{-3, 7, -1}
	simd.Abs(i32)
	fmt.Println(i32)

	// Output:
	// [2199023255552 4398046511104]
	// 0.875
	// [3 7 1]
}

// Lengths need not match; slicing bounds the work.
func Example_sizing() {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float64{1, 1, 1, 1, 1, 1, 1, 1}

	simd.Add(a[:3], b) // only the first three
	fmt.Println(a)

	// Output:
	// [2 3 4 4 5 6 7 8]
}

// AddScaled is AXPY: one pass over memory instead of two.
func Example_fused() {
	a := []float32{1, 2, 3, 4}
	b := []float32{10, 10, 10, 10}

	simd.AddScaled(a, b, 0.5) // a += b * 0.5
	fmt.Println(a)

	// Output:
	// [6 7 8 9]
}

// Whole tasks are one call and reuse the same accelerated primitives.
func Example_scenarios() {
	a := []float64{2, 4, 4, 4, 5, 5, 7, 9}

	fmt.Printf("mean   %.2f\n", simd.Mean(a))
	fmt.Printf("stddev %.2f\n", simd.StdDev(a))

	x := []float64{1, 0}
	y := []float64{0, 1}
	fmt.Printf("cosine %.2f\n", simd.CosineSimilarity(x, y))
	fmt.Printf("dist   %.4f\n", simd.Distance(x, y))

	v := []float64{3, 4}
	simd.Normalize(v)
	fmt.Println("unit  ", v)

	// Output:
	// mean   5.00
	// stddev 2.00
	// cosine 0.00
	// dist   1.4142
	// unit   [0.6 0.8]
}

func Example_bytes() {
	b := []byte("hello world")
	fmt.Println(simd.IndexByte(b, 'w'))
	fmt.Println(simd.CountByte(b, 'l'))
	fmt.Println(simd.Equal(b, []byte("hello world")))

	data := []byte{0xab, 0xcd, 0xef}
	mask := []byte{0x0f, 0x0f, 0x0f}
	simd.And(data, mask) // in place
	fmt.Printf("%x\n", data)

	// Output:
	// 6
	// 3
	// true
	// 0b0d0f
}
