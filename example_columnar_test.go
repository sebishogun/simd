package simd_test

import (
	"fmt"

	"github.com/sebishogun/simd"
)

func ExampleCompressBitsInto() {
	values := []float64{1.5, 2.5, 3.5, 4.5, 5.5}
	// Arrow layout: one bit per row, LSB-first. 0b10011 keeps rows 0, 1, 4.
	validity := []byte{0b10011}

	dst := make([]float64, len(values))
	n := simd.CompressBitsInto(dst, values, validity)
	fmt.Println(dst[:n])
	// Output:
	// [1.5 2.5 5.5]
}

func ExampleSumValid() {
	values := []float64{10, 20, 30, 40}
	validity := []byte{0b0101} // rows 0 and 2

	fmt.Println(simd.SumValid(values, validity))
	// Output:
	// 40
}

func ExampleCountValid() {
	validity := []byte{0xFF, 0b00000111}
	fmt.Println(simd.CountValid(validity, 16))
	fmt.Println(simd.CountValid(validity, 10)) // only the first 10 bits
	// Output:
	// 11
	// 10
}
