package search

// Compression, against the property that defines it.
//
// The interesting failures here are not wrong values but wrong *counts* and
// wrong *positions*, because the kernel advances its destination pointer by a
// population count computed separately from the store. A mask whose popcount
// disagrees with the compress by one lane produces output that is entirely
// plausible — right elements, right order, one slot off — which is why every
// case below checks the count, the contents and the bytes past the count.

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

// refCompress is the definition, written as simply as possible so that it is
// obviously right rather than efficiently right.
func refCompress[T comparable](dst, src []T, mask []bool) (out []T) {
	for i := 0; i < len(src) && i < len(mask); i++ {
		if mask[i] && len(out) < len(dst) {
			out = append(out, src[i])
		}
	}
	return out
}

func TestCompress(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	// Densities matter more than sizes here. An all-true mask and an all-false
	// mask are the two ends the pointer arithmetic can be wrong at, and the
	// alternating masks are where a per-lane popcount and a whole-block one
	// disagree first.
	densities := []struct {
		name string
		gen  func(i int) bool
	}{
		{"none", func(int) bool { return false }},
		{"all", func(int) bool { return true }},
		{"alternating", func(i int) bool { return i%2 == 0 }},
		{"first-only", func(i int) bool { return i == 0 }},
		{"last-of-block", func(i int) bool { return i%16 == 15 }},
		{"sparse", func(i int) bool { return i%97 == 0 }},
		{"random", func(int) bool { return rng.Intn(2) == 0 }},
	}

	// Lengths straddling the 16-lane block and the 64-element threshold, so
	// both the vector body and the scalar tail are exercised, and so is the
	// boundary between them.
	lengths := []int{0, 1, 15, 16, 17, 31, 63, 64, 65, 127, 128, 129, 1000, 4096}

	for _, n := range lengths {
		for _, d := range densities {
			t.Run(fmt.Sprintf("n=%d/%s", n, d.name), func(t *testing.T) {
				src := make([]int32, n)
				mask := make([]bool, n)
				for i := range src {
					src[i] = int32(i)*7 - 3
					mask[i] = d.gen(i)
				}

				const canary = -999
				dst := make([]int32, n)
				for i := range dst {
					dst[i] = canary
				}

				got := simd.CompressInto(dst, src, mask)
				want := refCompress(dst, src, mask)

				if got != len(want) {
					t.Fatalf("count = %d, want %d", got, len(want))
				}
				for i, w := range want {
					if dst[i] != w {
						t.Fatalf("dst[%d] = %d, want %d", i, dst[i], w)
					}
				}
				// The tail past the count must be untouched. A compress that
				// writes a full block and then reports a smaller count is
				// correct; one that writes a full block, reports the right
				// count, and has clobbered a slot the caller was using beyond
				// it is not — and only this check separates them.
				for i := got; i < len(dst); i++ {
					if dst[i] != canary {
						t.Fatalf("dst[%d] = %d past the count of %d; it should still be the canary",
							i, dst[i], got)
					}
				}
			})
		}
	}
}

// TestCompressShortDestination covers the contract that a destination too
// small for every match truncates rather than panics. The kernel cannot
// bound-check per lane, so this is the case that has to reach the portable
// path, and the test is here to notice if the guard that sends it there is
// ever removed.
func TestCompressShortDestination(t *testing.T) {
	const n = 4096
	src := make([]int32, n)
	mask := make([]bool, n)
	for i := range src {
		src[i] = int32(i)
		mask[i] = true
	}
	for _, room := range []int{0, 1, 15, 16, 100, n - 1} {
		dst := make([]int32, room)
		if got := simd.CompressInto(dst, src, mask); got != room {
			t.Errorf("with room for %d, wrote %d", room, got)
		}
		for i := range dst {
			if dst[i] != int32(i) {
				t.Errorf("room=%d: dst[%d] = %d, want %d", room, i, dst[i], i)
				break
			}
		}
	}
}

// TestCompressFloatBits checks that compression moves bit patterns rather than
// values. It is a data movement operation, so a NaN payload must survive it
// intact and a negative zero must stay negative — neither of which a version
// that went through a comparison or an arithmetic lane would guarantee.
func TestCompressFloatBits(t *testing.T) {
	payload := math.Float64frombits(0x7FF8_0000_DEAD_BEEF)
	specials := []float64{
		payload, math.NaN(), math.Inf(1), math.Inf(-1),
		0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, math.MaxFloat64,
	}
	src := make([]float64, 256)
	mask := make([]bool, len(src))
	for i := range src {
		src[i] = specials[i%len(specials)]
		mask[i] = i%3 == 0
	}
	dst := make([]float64, len(src))
	n := simd.CompressInto(dst, src, mask)

	k := 0
	for i := range src {
		if !mask[i] {
			continue
		}
		if got, want := math.Float64bits(dst[k]), math.Float64bits(src[i]); got != want {
			t.Fatalf("dst[%d] bits = %#016x, want %#016x (from src[%d])", k, got, want, i)
		}
		k++
	}
	if k != n {
		t.Fatalf("count = %d, want %d", n, k)
	}
}

func TestExpandRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{0, 1, 17, 64, 65, 1000} {
		src := make([]int32, n)
		mask := make([]bool, n)
		for i := range src {
			src[i] = int32(rng.Int31())
			mask[i] = rng.Intn(3) != 0
		}
		packed := make([]int32, n)
		k := simd.CompressInto(packed, src, mask)

		const filler = -1
		back := make([]int32, n)
		for i := range back {
			back[i] = filler
		}
		if got := simd.ExpandInto(back, packed[:k], mask); got != k {
			t.Fatalf("n=%d: expand consumed %d, want %d", n, got, k)
		}
		for i := range src {
			want := int32(filler)
			if mask[i] {
				want = src[i]
			}
			if back[i] != want {
				t.Fatalf("n=%d: back[%d] = %d, want %d", n, i, back[i], want)
			}
		}
	}
}

func TestFilterInto(t *testing.T) {
	src := make([]float64, 1000)
	for i := range src {
		src[i] = float64(i) - 500
	}
	dst := make([]float64, len(src))
	n := simd.FilterInto(dst, src, func(v float64) bool { return v > 0 })
	if n != 499 {
		t.Fatalf("kept %d, want 499", n)
	}
	for i, v := range dst[:n] {
		if v != float64(i+1) {
			t.Fatalf("dst[%d] = %v, want %v", i, v, float64(i+1))
		}
	}

	// The composed form must agree with it, since that is the pairing the
	// documentation sends people to.
	mask := make([]bool, len(src))
	simd.GreaterScalarInto(mask, src, 0)
	other := make([]float64, len(src))
	if m := simd.CompressInto(other, src, mask); m != n {
		t.Fatalf("GreaterScalar+Compress kept %d, Filter kept %d", m, n)
	}
	for i := range dst[:n] {
		if other[i] != dst[i] {
			t.Fatalf("at %d: GreaterScalar+Compress gave %v, Filter gave %v", i, other[i], dst[i])
		}
	}
}
