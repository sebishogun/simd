package simd_test

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// Encodings: quantization, the narrow float formats, bit packing, run-length
// and the varint widths. These are the operations a column store or an
// inference runtime spends its time in.

// The columnar compression pipeline, in the order the pieces are meant to be
// used: differences first, then zigzag so small negatives stay small, then
// pack to the width that actually fits.
func ExampleBitPackInto() {
	a := []uint32{1, 2, 3, 4, 5, 6, 7, 0}

	// Three bits hold 0..7, so eight values need 24 bits — one 32-bit word.
	// Unpacking needs one word MORE than packing: a value whose bits straddle a
	// word boundary reads the next word, and the last one straddles whenever
	// the total is not a multiple of 32. Size for the reader.
	packed := make([]uint32, (len(a)*3+31)/32+1)
	simd.BitPackInto(packed, a, 3)

	back := make([]uint32, len(a))
	simd.BitUnpackInto(back, packed, 3)
	fmt.Println(back)
	// Output: [1 2 3 4 5 6 7 0]
}

// VarintSize gives the exact encoded length of a whole slice in one
// vectorized pass, so an encoder sizes its buffer once instead of growing it.
//
// Writing the bytes is serial and always will be — where value i lands depends
// on the width of every value before it — but asking how wide each one is
// vectorizes, and that is the part that lets the allocation happen once.
func ExampleVarintSize() {
	a := []uint64{1, 300, 70000}
	fmt.Println(simd.VarintSize(a), "bytes")
	// Output: 6 bytes
}

// VarintLenInto gives the per-value widths. Prefix-summed, they are every
// value's offset in the stream, which is what makes the writes independent.
func ExampleVarintLenInto() {
	lens := make([]int32, 3)
	simd.VarintLenInto(lens, []uint64{1, 300, 70000})
	fmt.Println(lens)
	// Output: [1 2 3]
}

func ExampleAppendVarints() {
	buf := simd.AppendVarints(nil, []uint64{1, 300})
	fmt.Printf("% x\n", buf)
	// Output: 01 ac 02
}

// RunLengthEncodeInt32 turns a run of equal values into a value and a count.
// The run boundaries are found with a vectorized pass; the scratch slice holds
// them so nothing is allocated.
func ExampleRunLengthEncodeInt32() {
	a := []int32{7, 7, 7, 9, 9, 4}

	values := make([]int32, len(a))
	lengths := make([]int32, len(a))
	n := simd.RunLengthEncodeInt32(values, lengths, a, make([]bool, len(a)))

	fmt.Println(values[:n], lengths[:n])
	// Output: [7 9 4] [3 2 1]
}

// RunStartsInto marks every element that begins a run, which is the
// vectorizable half of run-length encoding.
func ExampleRunStartsInto() {
	dst := make([]bool, 6)
	simd.RunStartsInto(dst, []int32{7, 7, 7, 9, 9, 4})
	fmt.Println(dst)
	// Output: [true false false true false true]
}

// Quantization to int8 with a scale and zero point: the operation an inference
// runtime does to every weight and activation.
func ExampleQuantizePerChannelInt8() {
	// Two channels of two values, each with its own scale.
	a := []float32{1, 2, 30, 40}
	scale := []float32{0.5, 10}
	zero := []int32{0, 0}

	dst := make([]int8, len(a))
	simd.QuantizePerChannelInt8(dst, a, scale, zero, 2, 2)
	fmt.Println(dst)
	// Output: [2 4 3 4]
}

// The int8 matrix multiply accumulates into int32, because the products of
// two int8 values overflow int8 immediately. Requantize brings it back down.
func ExampleQMatMulInt8Into() {
	// [1 2] * [1 0] = [1 2]
	// [3 4]   [0 1]   [3 4]
	a := []int8{1, 2, 3, 4}
	b := []int8{1, 0, 0, 1}

	acc := make([]int32, 4)
	simd.QMatMulInt8Into(acc, a, b, 2, 2, 2)
	fmt.Println(acc)

	// q = round(acc*scale) + zeroPoint, rounding half to EVEN — so 1*0.5
	// becomes 0 and 3*0.5 becomes 2, which is what the runtimes this
	// interoperates with do and what round-half-away-from-zero would not.
	out := make([]int8, 4)
	simd.RequantizeInt8Into(out, acc, 0.5, 0)
	fmt.Println(out)
	// Output:
	// [1 2 3 4]
	// [0 1 2 2]
}

// float16 and bfloat16 are storage formats: half the bytes, and the
// conversion is what the vector unit is for.
func ExampleFloat16ToFloat32Into() {
	dst := make([]float32, 2)
	simd.Float16ToFloat32Into(dst, []uint16{0x3c00, 0x4000}) // 1.0, 2.0
	fmt.Println(dst)
	// Output: [1 2]
}

// e4m3 trades exponent range for mantissa, which is the trade inference
// weights want. It has no infinity: the largest value is 448.
func ExampleFloat32ToFloat8E4M3Into() {
	dst := make([]byte, 2)
	simd.Float32ToFloat8E4M3Into(dst, []float32{1, 2})

	back := make([]float32, 2)
	simd.Float8E4M3ToFloat32Into(back, dst)
	fmt.Println(back)
	// Output: [1 2]
}

// Grayscale uses the libjpeg BT.601 weights in Q16, so it agrees with every
// other implementation of the same conversion to the bit.
func ExampleGrayscaleInto() {
	dst := make([]byte, 3)
	simd.GrayscaleInto(dst,
		[]byte{255, 0, 0}, // r
		[]byte{0, 255, 0}, // g
		[]byte{0, 0, 255}) // b
	fmt.Println(dst)
	// Output: [76 150 29]
}

func ExampleRGBToUVInto() {
	u := make([]byte, 1)
	v := make([]byte, 1)
	simd.RGBToUVInto(u, v, []byte{255}, []byte{0}, []byte{0})
	fmt.Println(u[0], v[0])
	// Output: 85 255
}

// A counter-based generator: element i depends on i alone, which is what makes
// it vectorizable, reproducible across architectures, and splittable across
// goroutines without any shared state.
func ExampleRandomInto() {
	a := make([]float64, 4)
	simd.RandomInto(a, 42)

	b := make([]float64, 4)
	simd.RandomInto(b, 42)
	fmt.Println(a[0] == b[0], a[0] >= 0 && a[0] < 1)
	// Output: true true
}

// RequantizeInt8Into brings an int32 accumulator back to int8:
// q = round(acc*scale) + zeroPoint, rounding half to even and saturating
// rather than wrapping.
func ExampleRequantizeInt8Into() {
	dst := make([]int8, 4)
	simd.RequantizeInt8Into(dst, []int32{100, 200, 300, 100000}, 0.5, 0)
	fmt.Println(dst)
	// Output: [50 100 127 127]
}
