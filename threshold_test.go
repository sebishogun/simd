package simd_test

// Tests for the two operations whose dispatch threshold is above the length
// every other test in this repository uses.
//
// simd_test.go and internal/conformance both run at maxLen = 70. Every kernel
// with a threshold at or below 64 is therefore exercised on both sides of its
// crossover by the existing suite, and two are not: the n-ary family switches
// at 256 and CompressInto at 192. Below the threshold the dispatcher calls the
// reference, so those two accelerated paths had no differential coverage at
// all — the same blind spot that hid the signed-zero disagreement in Sort
// until TestSelectAcceleratedMatchesReference went looking above 512.
//
// These use integer element types deliberately. Integer addition and
// multiplication are associative, so any accumulation order gives identical
// results and the test pins the kernel's answer without having to reproduce
// the reference's fold. The float paths are covered by the conformance suite
// below the threshold and by the bit-identity contract above it.

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// naryLens straddle thNary = 256 and CompressInto's 192, and continue far
// enough past both that the kernel's tail handling is exercised at several
// different remainders modulo the vector width.
var thresholdLens = []int{191, 192, 193, 255, 256, 257, 300, 512, 1000, 4099}

func TestNaryAboveThreshold(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 11))
	for _, n := range thresholdLens {
		for _, k := range []int{2, 3, 4, 5} {
			srcs := make([][]int32, k)
			for j := range srcs {
				s := make([]int32, n)
				for i := range s {
					s[i] = int32(r.Uint32())
				}
				srcs[j] = s
			}

			t.Run(fmt.Sprintf("Add/n=%d/k=%d", n, k), func(t *testing.T) {
				got := make([]int32, n)
				simd.AddAll(got, srcs...)
				for i := range n {
					var want int32
					for _, s := range srcs {
						want += s[i]
					}
					if got[i] != want {
						t.Fatalf("AddAll[%d] = %d, want %d", i, got[i], want)
					}
				}
			})

			t.Run(fmt.Sprintf("Mul/n=%d/k=%d", n, k), func(t *testing.T) {
				got := make([]int32, n)
				simd.MulAll(got, srcs...)
				for i := range n {
					want := int32(1)
					for _, s := range srcs {
						want *= s[i]
					}
					if got[i] != want {
						t.Fatalf("MulAll[%d] = %d, want %d", i, got[i], want)
					}
				}
			})
		}
	}
}

func TestCompressAboveThreshold(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 17))
	// The threshold comment records that the crossover moves with match
	// density, so the densities it was measured at are the densities to test.
	for _, n := range thresholdLens {
		for _, pct := range []int{0, 1, 25, 50, 90, 100} {
			src := make([]int32, n)
			keep := make([]bool, n)
			for i := range src {
				src[i] = int32(i)
				keep[i] = int(r.UintN(100)) < pct
			}
			t.Run(fmt.Sprintf("n=%d/density=%d", n, pct), func(t *testing.T) {
				dst := make([]int32, n)
				got := simd.CompressInto(dst, src, keep)

				want := make([]int32, 0, n)
				for i, v := range src {
					if keep[i] {
						want = append(want, v)
					}
				}
				if got != len(want) {
					t.Fatalf("CompressInto = %d, want %d", got, len(want))
				}
				for i := range want {
					if dst[i] != want[i] {
						t.Fatalf("dst[%d] = %d, want %d", i, dst[i], want[i])
					}
				}
			})
		}
	}
}
