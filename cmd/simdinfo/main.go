// Command simdinfo reports which instruction-set tier the simd package
// selected on this machine, and which ones it could have used.
//
//	$ simdinfo
//	amd64 tier=avx512 available=[scalar sse2 avx2 avx512]
//
//	$ GOSIMD=sse2 simdinfo
//	amd64 tier=sse2 available=[scalar sse2 avx2 avx512] forced
//
//	$ SIMD_DISABLE=avx512 simdinfo
//	amd64 tier=avx2 available=[scalar sse2 avx2] disabled=[avx512]
//
// It is the first thing to run when a benchmark or a numerical result looks
// wrong, and `simdinfo -tiers` drives the per-tier test matrix in the Makefile.
package main

import (
	"flag"
	"fmt"

	"github.com/sebishogun/simd"
)

func main() {
	tiers := flag.Bool("tiers", false, "print every available tier, one per line, instead of the summary")
	flag.Parse()

	if *tiers {
		for _, t := range simd.AvailableTiers() {
			fmt.Println(t)
		}
		return
	}
	fmt.Println(simd.Describe())
}
