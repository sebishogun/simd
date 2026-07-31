package benchmarks

import "math"

// boundaries and naiveDFT are copies of helpers in the root test files rather
// than shared ones: the originals are still used by the unit tests next to the
// code they test, and exporting them from the library to satisfy a benchmark
// would put a test helper in the public API.

// boundaries returns the index of every separator in line, which is the input
// ParseInts and ParseFloats expect.
func boundaries(line []byte, sep byte) []int32 {
	idx := make([]int32, 0, 64)
	for i, c := range line {
		if c == sep {
			idx = append(idx, int32(i))
		}
	}
	idx = append(idx, int32(len(line)))
	return idx
}

// naiveDFT is the O(n^2) transform, as a reference to measure the FFT against.
func naiveDFT(a []complex128, inverse bool) []complex128 {
	n := len(a)
	out := make([]complex128, n)
	sign := -2 * math.Pi
	if inverse {
		sign = 2 * math.Pi
	}
	for k := range out {
		var sum complex128
		for t, v := range a {
			ang := sign * float64(k) * float64(t) / float64(n)
			sum += v * complex(math.Cos(ang), math.Sin(ang))
		}
		out[k] = sum
	}
	return out
}

// Sinks. Assigning a benchmark's result to a package-level variable stops the
// compiler deleting the call it is trying to measure. These are copies of the
// ones in the root test files for the same reason as the helpers above: those
// are still used by the unit tests that stayed behind.
var (
	sinkTextInt  int
	sinkTextBool bool
	sinkTextStr  string
)
