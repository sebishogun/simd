package simd_test

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// An FIR filter is a convolution, which is why this package has no dedicated
// FIR function. The taps are the kernel; ConvolveFullInto picks direct or
// frequency-domain by a measured crossover. Here a 4-tap moving average
// smooths a step — the quarters are exact in float64, so the output is too.
func ExampleConvolveFull() {
	signal := []float64{0, 0, 0, 4, 4, 4, 4, 4}
	taps := []float64{0.25, 0.25, 0.25, 0.25}

	full := simd.ConvolveFull(signal, taps)
	// For a causal filter, output[i] depends on samples up to i, which is
	// exactly the first len(signal) entries of the full convolution.
	fmt.Println(full[:len(signal)])
	// Output: [0 0 0 1 2 3 4 4]
}

// The k largest without sorting: TopK selects around the k-th order statistic
// in linear time, so taking the top 3 of a million never sorts the million.
func ExampleTopK() {
	scores := []float64{12, 99, 7, 45, 68, 99, 3, 81}
	fmt.Println(simd.TopK(scores, 3))
	// Output: [99 99 81]
}

// Bincount is the histogram of small integers: counts[v]++ for each v,
// skipping anything out of range rather than panicking.
func ExampleBincount() {
	dice := []int32{3, 6, 3, 1, 6, 6, 2, 3}
	fmt.Println(simd.Bincount(dice, 7)[1:]) // index 0 unused for a die
	// Output: [1 1 3 0 0 3]
}

// Case-insensitive search without allocating lowered copies: the haystack is
// folded once into caller scratch at ToLowerASCII speed, and offsets in the
// fold are offsets in the original.
func ExampleIndexFoldASCII() {
	page := []byte("The QUICK brown fox")
	scratch := make([]byte, len(page))
	fmt.Println(simd.IndexFoldASCII(page, "quick", scratch))
	fmt.Println(simd.ContainsFoldASCII(page, "BROWN FOX", scratch))
	// Output:
	// 4
	// true
}

// JSON string escaping in the append style, so a serializer brings one buffer.
// NeedsEscapeJSON is one accelerated scan, and on clean input the writer can
// skip escaping entirely.
func ExampleAppendEscapeJSON() {
	buf := make([]byte, 0, 64)
	buf = append(buf, '"')
	buf = simd.AppendEscapeJSON(buf, "say \"hi\"\n")
	buf = append(buf, '"')
	fmt.Println(string(buf))
	fmt.Println(simd.NeedsEscapeJSON("clean text"))
	// Output:
	// "say \"hi\"\n"
	// false
}
