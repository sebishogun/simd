package simd_test

// Sorting and partitioning.
//
// The important check here is not "is the result sorted". A sort with a bug in
// its partition still returns a permutation, and a permutation of a sorted
// multiset is very often still sorted-looking — drop an element and duplicate
// its neighbour and every ordering check still passes. So every test below also
// verifies that the *multiset* is unchanged, which is the property that
// actually fails when a compress store writes one lane too few.

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/sebishogun/simd"
)

// sameMultiset reports whether two slices contain the same values with the same
// multiplicities, comparing by bit pattern so that NaN and -0 are counted as
// themselves.
func sameMultiset(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	count := map[uint64]int{}
	for _, v := range a {
		count[math.Float64bits(v)]++
	}
	for _, v := range b {
		k := math.Float64bits(v)
		count[k]--
		if count[k] == 0 {
			delete(count, k)
		}
	}
	return len(count) == 0
}

func sortInputs(n int, kind string, seed int64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	a := make([]float64, n)
	switch kind {
	case "random":
		for i := range a {
			a[i] = rng.NormFloat64()
		}
	case "sorted":
		for i := range a {
			a[i] = float64(i)
		}
	case "reversed":
		for i := range a {
			a[i] = float64(n - i)
		}
	case "constant":
		for i := range a {
			a[i] = 7
		}
	case "few-distinct":
		for i := range a {
			a[i] = float64(rng.Intn(3))
		}
	case "sawtooth":
		for i := range a {
			a[i] = float64(i % 64)
		}
	}
	return a
}

var sortKinds = []string{"random", "sorted", "reversed", "constant", "few-distinct", "sawtooth"}

func TestSort(t *testing.T) {
	// Sizes straddle sortCutoff (512) so both the accelerated recursion and
	// the standard-library fallback are exercised, and the pathological
	// distributions above are what drive the recursion into its guard paths.
	for _, n := range []int{0, 1, 2, 511, 512, 513, 1000, 4096, 20000} {
		for _, kind := range sortKinds {
			t.Run(fmt.Sprintf("n=%d/%s", n, kind), func(t *testing.T) {
				a := sortInputs(n, kind, int64(n))
				orig := slices.Clone(a)

				simd.Sort(a)

				if !slices.IsSorted(a) {
					for i := 1; i < len(a); i++ {
						if a[i] < a[i-1] {
							t.Fatalf("not sorted at %d: %v > %v", i, a[i-1], a[i])
						}
					}
				}
				if !sameMultiset(orig, a) {
					t.Fatal("the multiset changed; the sort lost or duplicated elements")
				}
			})
		}
	}
}

// TestSortInto is the allocation-free form, and also checks that a scratch
// slice that is too short falls back rather than corrupting anything.
func TestSortInto(t *testing.T) {
	for _, n := range []int{600, 4096} {
		a := sortInputs(n, "random", 5)
		orig := slices.Clone(a)
		scratch := make([]float64, n)
		simd.SortInto(a, scratch)
		if !slices.IsSorted(a) || !sameMultiset(orig, a) {
			t.Fatalf("n=%d: SortInto produced a wrong result", n)
		}
	}

	a := sortInputs(1000, "random", 6)
	orig := slices.Clone(a)
	simd.SortInto(a, make([]float64, 10)) // far too short
	if !slices.IsSorted(a) || !sameMultiset(orig, a) {
		t.Fatal("a short scratch slice should fall back, not corrupt")
	}
}

func TestSortAllocations(t *testing.T) {
	a := sortInputs(4096, "random", 7)
	scratch := make([]float64, len(a))
	got := testing.AllocsPerRun(20, func() {
		simd.SortInto(a, scratch)
	})
	if got != 0 {
		t.Errorf("SortInto allocated %v times per run, want 0", got)
	}
}

// TestSortNaN pins the ordering choice: NaN goes to the end, which matches
// Median and Quantile in this package and differs from slices.Sort.
func TestSortNaN(t *testing.T) {
	a := []float64{3, math.NaN(), 1, math.NaN(), 2}
	simd.Sort(a)
	if a[0] != 1 || a[1] != 2 || a[2] != 3 {
		t.Fatalf("finite values misordered: %v", a[:3])
	}
	if !math.IsNaN(a[3]) || !math.IsNaN(a[4]) {
		t.Fatalf("NaN did not sort to the end: %v", a)
	}

	// And at a size that goes through the accelerated path.
	const n = 4096
	b := make([]float64, n)
	for i := range b {
		if i%97 == 0 {
			b[i] = math.NaN()
		} else {
			b[i] = float64(n - i)
		}
	}
	simd.Sort(b)
	nans := 0
	for _, v := range b {
		if math.IsNaN(v) {
			nans++
		}
	}
	for i := len(b) - nans; i < len(b); i++ {
		if !math.IsNaN(b[i]) {
			t.Fatalf("NaN not contiguous at the end (index %d)", i)
		}
	}
}

func TestPartitionInto(t *testing.T) {
	for _, n := range []int{0, 1, 15, 16, 17, 63, 64, 65, 1000, 4096} {
		a := sortInputs(n, "random", int64(n+11))
		dst := make([]float64, n)
		const pivot = 0.0

		k := simd.PartitionInto(dst, a, pivot)

		for i := range k {
			if !(dst[i] < pivot) {
				t.Fatalf("n=%d: dst[%d] = %v is not below the pivot", n, i, dst[i])
			}
		}
		for i := k; i < n; i++ {
			if dst[i] < pivot {
				t.Fatalf("n=%d: dst[%d] = %v is below the pivot but after the split", n, i, dst[i])
			}
		}
		if !sameMultiset(a, dst) {
			t.Fatalf("n=%d: the multiset changed", n)
		}

		// Stability: within each side the original order survives.
		var lows, highs []float64
		for _, v := range a {
			if v < pivot {
				lows = append(lows, v)
			} else {
				highs = append(highs, v)
			}
		}
		for i := range lows {
			if dst[i] != lows[i] {
				t.Fatalf("n=%d: low side not stable at %d", n, i)
			}
		}
		for i := range highs {
			if dst[k+i] != highs[i] {
				t.Fatalf("n=%d: high side not stable at %d", n, i)
			}
		}
	}
}

func TestArgsort(t *testing.T) {
	for _, n := range []int{0, 1, 17, 1000} {
		a := sortInputs(n, "random", int64(n+3))
		idx := make([]int32, n)

		if got := simd.Argsort(idx, a); got != n {
			t.Fatalf("n=%d: wrote %d indices", n, got)
		}
		// It is a permutation.
		seen := make([]bool, n)
		for _, v := range idx {
			if v < 0 || int(v) >= n || seen[v] {
				t.Fatalf("n=%d: %d is not a fresh valid index", n, v)
			}
			seen[v] = true
		}
		// And it sorts.
		for i := 1; i < n; i++ {
			if a[idx[i]] < a[idx[i-1]] {
				t.Fatalf("n=%d: order wrong at %d", n, i)
			}
		}
		// GatherInto applies it, which is the documented pairing.
		out := make([]float64, n)
		simd.GatherInto(out, a, idx)
		if !slices.IsSorted(out) || !sameMultiset(a, out) {
			t.Fatalf("n=%d: gathering by the permutation did not sort", n)
		}
	}
}

// TestArgsortStable checks the documented tie behaviour: equal values keep
// their original relative order, so the permutation is deterministic.
func TestArgsortStable(t *testing.T) {
	a := []float64{5, 1, 5, 1, 5, 1}
	idx := make([]int32, len(a))
	simd.Argsort(idx, a)
	want := []int32{1, 3, 5, 0, 2, 4}
	for i := range want {
		if idx[i] != want[i] {
			t.Fatalf("idx = %v, want %v", idx, want)
		}
	}
}

func TestSortedIndex(t *testing.T) {
	a := []float64{1, 3, 5, 7}
	for _, c := range []struct {
		v     float64
		i     int
		found bool
	}{{0, 0, false}, {1, 0, true}, {4, 2, false}, {7, 3, true}, {9, 4, false}} {
		i, found := simd.SortedIndex(a, c.v)
		if i != c.i || found != c.found {
			t.Errorf("SortedIndex(%v) = %d,%v want %d,%v", c.v, i, found, c.i, c.found)
		}
	}
}

func TestSortIntegers(t *testing.T) {
	for _, n := range []int{600, 5000} {
		a := make([]int32, n)
		rng := rand.New(rand.NewSource(int64(n)))
		for i := range a {
			a[i] = rng.Int31n(1000) - 500
		}
		orig := slices.Clone(a)
		simd.Sort(a)
		if !slices.IsSorted(a) {
			t.Fatalf("n=%d: not sorted", n)
		}
		slices.Sort(orig)
		for i := range a {
			if a[i] != orig[i] {
				t.Fatalf("n=%d: differs from slices.Sort at %d", n, i)
			}
		}
	}
}

func BenchmarkSort(b *testing.B) {
	for _, n := range []int{1024, 16384, 262144, 1 << 21} {
		for _, kind := range []string{"random", "few-distinct"} {
			src := sortInputs(n, kind, 42)
			work := make([]float64, n)
			scratch := make([]float64, n)

			b.Run(fmt.Sprintf("n=%d/%s/impl=simd", n, kind), func(b *testing.B) {
				for b.Loop() {
					copy(work, src)
					simd.SortInto(work, scratch)
				}
			})
			b.Run(fmt.Sprintf("n=%d/%s/impl=slices", n, kind), func(b *testing.B) {
				for b.Loop() {
					copy(work, src)
					slices.Sort(work)
				}
			})
		}
	}
}

// BenchmarkMedian measures the accelerated quickselect against the portable
// one, and against sorting — which is what a caller without a Median reaches
// for. The reference path is selected by passing a nil scratch, so both sides
// enter through the same function and the only difference is which algorithm
// runs.
//
// The sizes straddle simd.SelectMinLenForTest so the crossover is visible
// rather than assumed.
func BenchmarkMedian(b *testing.B) {
	for _, n := range []int{256, 512, 1024, 4096, 65536, 1 << 20} {
		for _, kind := range []string{"random", "few-distinct"} {
			src := sortInputs(n, kind, 42)
			work := make([]float64, n)
			scratch := make([]float64, n)

			b.Run(fmt.Sprintf("n=%d/%s/impl=simd", n, kind), func(b *testing.B) {
				for b.Loop() {
					copy(work, src)
					sinkSortF = simd.MedianInto(work, scratch)
				}
			})
			b.Run(fmt.Sprintf("n=%d/%s/impl=portable", n, kind), func(b *testing.B) {
				for b.Loop() {
					copy(work, src)
					sinkSortF = simd.MedianInto(work, nil)
				}
			})
			b.Run(fmt.Sprintf("n=%d/%s/impl=sort", n, kind), func(b *testing.B) {
				for b.Loop() {
					copy(work, src)
					slices.Sort(work)
					sinkSortF = work[n/2]
				}
			})
		}
	}
}

var sinkSortF float64
