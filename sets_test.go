package simd_test

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/sebishogun/simd"
)

// sortedSet builds a sorted duplicate-free slice of n values drawn from
// [0, span). A small span makes the two sets overlap heavily and a large one
// makes them nearly disjoint; both matter, because the block-skipping loop
// advances one side or both depending on how the values interleave.
func sortedSet(n, span int, r *rand.Rand) []int32 {
	seen := make(map[int32]bool, n)
	out := make([]int32, 0, n)
	for len(out) < n {
		v := int32(r.IntN(span))
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	slices.Sort(out)
	return out
}

func naiveIntersect(a, b []int32) []int32 {
	in := map[int32]bool{}
	for _, v := range b {
		in[v] = true
	}
	var out []int32
	for _, v := range a {
		if in[v] {
			out = append(out, v)
		}
	}
	return out
}

func naiveDifference(a, b []int32) []int32 {
	in := map[int32]bool{}
	for _, v := range b {
		in[v] = true
	}
	var out []int32
	for _, v := range a {
		if !in[v] {
			out = append(out, v)
		}
	}
	return out
}

func TestIntersectAndDifference(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	// Lengths on both sides of the eight-element tile and of the dispatch
	// threshold, and spans from near-total overlap to near-disjoint. The
	// tile-straddling case — a's block still pending when b runs out of full
	// blocks — is the one a first draft of the difference kernel got wrong.
	for _, na := range []int{0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1000, 5000} {
		for _, nb := range []int{0, 1, 8, 9, 17, 64, 999, 5000} {
			for _, span := range []int{16, 64, 4096, 1 << 20} {
				if na > span || nb > span {
					continue
				}
				a := sortedSet(na, span, r)
				b := sortedSet(nb, span, r)

				dst := make([]int32, min(na, nb))
				got := dst[:simd.IntersectInto(dst, a, b)]
				want := naiveIntersect(a, b)
				if !slices.Equal(got, want) {
					t.Fatalf("Intersect na=%d nb=%d span=%d:\n got %v\nwant %v",
						na, nb, span, got, want)
				}

				dst2 := make([]int32, na)
				got = dst2[:simd.DifferenceInto(dst2, a, b)]
				want = naiveDifference(a, b)
				if !slices.Equal(got, want) {
					t.Fatalf("Difference na=%d nb=%d span=%d:\n got %v\nwant %v",
						na, nb, span, got, want)
				}
			}
		}
	}
}

// Identical sets, disjoint sets and one-empty are the cases where an
// off-by-one in the block advance shows up as either everything or nothing.
func TestIntersectAndDifferenceExtremes(t *testing.T) {
	same := make([]int32, 1000)
	for i := range same {
		same[i] = int32(i)
	}
	other := make([]int32, 1000)
	for i := range other {
		other[i] = int32(i + 1000)
	}

	dst := make([]int32, 1000)
	if n := simd.IntersectInto(dst, same, same); n != 1000 || !slices.Equal(dst[:n], same) {
		t.Errorf("Intersect(x, x) returned %d elements", n)
	}
	if n := simd.DifferenceInto(dst, same, same); n != 0 {
		t.Errorf("Difference(x, x) returned %d elements, want 0", n)
	}
	if n := simd.IntersectInto(dst, same, other); n != 0 {
		t.Errorf("Intersect of disjoint sets returned %d elements", n)
	}
	if n := simd.DifferenceInto(dst, same, other); n != 1000 || !slices.Equal(dst[:n], same) {
		t.Errorf("Difference by a disjoint set returned %d elements", n)
	}
	if n := simd.IntersectInto(dst, same, nil); n != 0 {
		t.Errorf("Intersect with an empty set returned %d elements", n)
	}
	if n := simd.DifferenceInto(dst, same, nil); n != 1000 {
		t.Errorf("Difference by an empty set returned %d elements, want 1000", n)
	}
}

// The destination bound is a panic rather than a truncation, because the
// kernel is at its ABI's six-argument limit and cannot be told the length.
// A silent short write would corrupt the caller's data instead.
func TestSetsPanicOnShortDestination(t *testing.T) {
	a := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := []int32{2, 4, 6, 8, 10}
	for _, c := range []struct {
		name string
		call func()
	}{
		{"Intersect", func() { simd.IntersectInto(make([]int32, 4), a, b) }},
		{"Difference", func() { simd.DifferenceInto(make([]int32, 9), a, b) }},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("no panic on a short destination")
				}
			}()
			c.call()
		})
	}
}

func TestSetsAllTypes(t *testing.T) {
	a32 := []uint32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
	b32 := []uint32{3, 7, 11, 15, 19, 23}
	d32 := make([]uint32, len(a32))
	if n := simd.IntersectInto(d32, a32, b32); n != 5 {
		t.Errorf("uint32 Intersect = %d, want 5 (%v)", n, d32[:n])
	}
	a64 := []int64{-5, -3, -1, 1, 3, 5, 7, 9, 11, 13}
	b64 := []int64{-3, 1, 5, 9, 13}
	d64 := make([]int64, len(a64))
	if n := simd.DifferenceInto(d64, a64, b64); n != 5 {
		t.Errorf("int64 Difference = %d, want 5 (%v)", n, d64[:n])
	}
	au := []uint64{1 << 40, 1<<40 + 1, 1<<63 - 1, 1 << 63}
	bu := []uint64{1<<40 + 1, 1 << 63}
	du := make([]uint64, len(au))
	if n := simd.IntersectInto(du, au, bu); n != 2 {
		t.Errorf("uint64 Intersect = %d, want 2 (%v)", n, du[:n])
	}
}

func TestSetsNoAlloc(t *testing.T) {
	r := rand.New(rand.NewPCG(43, 44))
	a := sortedSet(4096, 1<<20, r)
	b := sortedSet(4096, 1<<20, r)
	dst := make([]int32, 4096)
	if n := testing.AllocsPerRun(20, func() { simd.IntersectInto(dst, a, b) }); n != 0 {
		t.Errorf("IntersectInto allocated %v times per run", n)
	}
	if n := testing.AllocsPerRun(20, func() { simd.DifferenceInto(dst, a, b) }); n != 0 {
		t.Errorf("DifferenceInto allocated %v times per run", n)
	}
}
