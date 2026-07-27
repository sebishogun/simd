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

// Filtering is two steps on purpose: a comparison writes the mask, and
// CompressInto packs the elements that passed. Splitting it that way is what
// lets one Compress serve every predicate, including ones this library has
// never heard of, instead of shipping GreaterThanFilter and the rest.
func ExampleCompressInto() {
	a := []float64{-3, 7, -1, 4, 0, 9}

	mask := make([]bool, len(a))
	simd.GreaterScalarInto(mask, a, 0) // which elements pass

	out := make([]float64, len(a))
	n := simd.CompressInto(out, a, mask) // pack the ones that did

	fmt.Println(out[:n])

	// Output:
	// [7 4 9]
}

// IndexAll is the structural-index step of a parser: every delimiter located
// in one pass, then the offsets walked. Like the rest of the text functions it
// takes a string or a []byte and copies neither.
func ExampleIndexAll() {
	line := "id,name,email,created_at"

	commas := make([]int32, 16)
	n := simd.IndexAll(commas, line, ',')

	fields, prev := []string{}, 0
	for _, off := range commas[:n] {
		fields = append(fields, line[prev:off])
		prev = int(off) + 1
	}
	fields = append(fields, line[prev:])

	fmt.Println(n, fields)

	// Output:
	// 3 [id name email created_at]
}

// GemvInto applies a matrix to a vector, which is the operation most callers
// actually want and much cheaper than going through MatMulInto with n=1.
//
// Row i of the result is bit-identical to Dot of that row against x — by
// construction, not by coincidence — so the two can be mixed freely.
func ExampleGemvInto() {
	a := []float64{ // 3x2, row-major
		1, 2,
		3, 4,
		5, 6,
	}
	x := []float64{10, 20}

	y := make([]float64, 3)
	simd.GemvInto(y, a, x, 3, 2)

	fmt.Println(y)
	fmt.Println(y[1] == simd.Dot(a[2:4], x))

	// Output:
	// [50 110 170]
	// true
}

func ExampleMatMulInto() {
	a := []float64{ // 2x3
		1, 2, 3,
		4, 5, 6,
	}
	b := []float64{ // 3x2
		7, 8,
		9, 10,
		11, 12,
	}

	dst := make([]float64, 2*2)
	simd.MatMulInto(dst, a, b, 2, 3, 2)

	fmt.Println(dst[:2])
	fmt.Println(dst[2:])

	// Output:
	// [58 64]
	// [139 154]
}

func ExampleBase64Encode() {
	src := []byte("any carnal pleasure")

	dst := make([]byte, simd.Base64EncodedLen(len(src)))
	n := simd.Base64Encode(dst, src)
	fmt.Println(string(dst[:n]))

	// Decode returns the number of bytes written, or -1 if the input is not
	// valid base64 — there is no error value to ignore by accident.
	back := make([]byte, simd.Base64DecodedLen(n))
	m := simd.Base64Decode(back, dst[:n])
	fmt.Println(string(back[:m]))

	// Output:
	// YW55IGNhcm5hbCBwbGVhc3VyZQ==
	// any carnal pleasure
}

// A whole activation layer in one call. Sigmoid guarantees a documented ULP
// bound; FastSigmoid is a drop-in replacement that trades some of it away.
func ExampleSigmoid() {
	x := []float32{-2, -1, 0, 1, 2}

	simd.Sigmoid(x)
	for _, v := range x {
		fmt.Printf("%.4f ", v)
	}
	fmt.Println()

	// Output:
	// 0.1192 0.2689 0.5000 0.7311 0.8808
}

// Scaling features into [0,1] — the pass in front of most models, and three
// calls with no temporary.
func Example_featureScaling() {
	raw := []float32{-5, 0, 12, 7, 30}

	lo, hi := simd.MinMax(raw)
	simd.AddScalar(raw, -lo)   // shift the minimum to zero
	simd.Scale(raw, 1/(hi-lo)) // and the maximum to one
	simd.Clamp(raw, 0, 1)      // rounding at the ends cannot escape

	for _, v := range raw {
		fmt.Printf("%.2f ", v)
	}
	fmt.Println()

	// Output:
	// 0.00 0.14 0.49 0.34 1.00
}
