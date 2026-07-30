// Command standardise is the worked example from docs/tutorial.md, kept here
// as a program so that `go build ./...` and `go vet ./...` check it on every
// commit. A snippet in a markdown file can drift out of date silently; this
// one cannot.
//
// It shows the shape the tutorial argues for: struct-of-arrays, so that each
// dimension is a contiguous slice and every operation is one vector pass with
// no allocation.
//
// Run it:
//
//	go run ./docs/examples/standardise
package main

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// Features is struct-of-arrays: one contiguous slice per dimension, which is
// what lets every operation below be a single vector pass. The array-of-
// structs form — []struct{A, B, C, D float64} — would put each dimension's
// values 32 bytes apart, and no vector register can load that without a
// gather.
type Features struct {
	Dims [][]float64 // Dims[d][i] is dimension d of sample i
}

// Standardise rewrites each dimension to zero mean and unit variance, in
// place, allocating nothing.
//
// Four passes over each column and not one temporary: Mean and StdDev are
// reductions that write nothing, and SubScalar and DivScalar are in-place
// elementwise operations.
func Standardise(f Features) {
	for _, col := range f.Dims {
		mean := simd.Mean(col)
		simd.SubScalar(col, mean)
		sd := simd.StdDev(col)
		if sd != 0 {
			simd.DivScalar(col, sd)
		}
	}
}

func main() {
	const (
		dims    = 4
		samples = 1 << 20
	)
	f := Features{Dims: make([][]float64, dims)}
	for d := range f.Dims {
		col := make([]float64, samples)
		for i := range col {
			col[i] = float64((i*7+d)%1000) - 500
		}
		f.Dims[d] = col
	}

	fmt.Println(simd.Describe())
	Standardise(f)
	for d, col := range f.Dims {
		fmt.Printf("dim %d: mean %+.3e  stddev %.6f\n", d, simd.Mean(col), simd.StdDev(col))
	}
}
