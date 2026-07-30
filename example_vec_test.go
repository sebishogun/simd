//go:build goexperiment.simd && amd64

package simd_test

import (
	"fmt"

	"simd/archsimd"

	simd "github.com/sebishogun/simd"
)

// The escape hatch is for expressions the catalogue does not have, and never
// will: a soft clip, x/(1+|x|), is one of an unbounded number of things
// someone might want and is not worth a kernel each.
//
// |v| is written as max(v, -v) because that is what the hardware does; there
// is no Abs method, and reaching for one is the first thing to check rather
// than assume.
func ExampleMapFloat32x8() {
	in := []float32{-4, -2, -1, 0, 1, 2, 4, 8, 16}
	out := make([]float32, len(in))

	one := archsimd.BroadcastFloat32x8(1)
	zero := archsimd.BroadcastFloat32x8(0)

	simd.MapFloat32x8(out, in, func(v simd.F32x8) simd.F32x8 {
		abs := v.Max(zero.Sub(v))
		return v.Div(abs.Add(one))
	})
	fmt.Printf("%.3f %.3f %.3f\n", out[2], out[4], out[8])
	// Output: -0.500 0.500 0.941
}

// Two inputs, and the length is deliberately not a multiple of eight: the
// helper handles the tail with a partial load and store, which is the part
// hand-written SIMD gets wrong and the reason to use the helper at all.
func ExampleZipFloat32x8() {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	dst := make([]float32, len(a))

	simd.ZipFloat32x8(dst, a, b, func(x, y simd.F32x8) simd.F32x8 {
		return x.Mul(y).Sub(x)
	})
	fmt.Println(dst[0], dst[8], dst[10])
	// Output: 1 9 11
}

// Lanes answers "how wide should my blocks be" for the CPU this is running on,
// which has to be answered before writing against a fixed width. Writing
// against F32x16 on a machine without AVX-512 does not fail to compile — it
// runs, slowly, through whatever the compiler emits to fake the width.
func ExampleLanes() {
	if !simd.HasVectorType {
		fmt.Println("no vector type in this build")
		return
	}
	n := simd.Lanes[float32]()
	// 4, 8 or 16 depending on the CPU, so print the shape of the answer
	// rather than a number that would differ between machines.
	fmt.Println(n == 4 || n == 8 || n == 16)
	// Output: true
}
