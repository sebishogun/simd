// Command comparison demonstrates the difference between a library that fixes
// its accumulation order and one that does not.
//
// It lives in its own module because it depends on kelindar/simd, and nothing
// a reader imports should drag in a second SIMD library. Run it twice:
//
//	go run .                                  # whatever this CPU supports
//	GOSIMD=scalar go run -tags noasm .        # both libraries' portable paths
//
// kelindar/simd changes answer between the two. This library does not. That is
// the whole point of the comparison and the only claim it makes.
package main

import (
	"fmt"
	"math"

	kel "github.com/kelindar/simd"
	ours "github.com/sebishogun/simd"
)

func main() {
	// One large value followed by many small ones. Adding the small ones to a
	// large running total loses them to rounding; adding them to each other
	// first does not. Which happens depends on how many accumulators the
	// implementation keeps, which is why the answer moves with vector width.
	const n = 1024
	a := make([]float32, n)
	a[0] = 1e8
	for i := 1; i < n; i++ {
		a[i] = 1
	}

	k, o := kel.SumFloat32s(a), ours.Sum(a)
	fmt.Printf("kelindar/simd  %-14v %#x\n", k, math.Float32bits(k))
	fmt.Printf("this library   %-14v %#x\n", o, math.Float32bits(o))
}
