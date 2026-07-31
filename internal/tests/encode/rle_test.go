package encode

import (
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

func TestRunStarts(t *testing.T) {
	a := []int32{5, 5, 5, 7, 7, 1, 1, 1, 1, 2}
	want := []bool{true, false, false, true, false, true, false, false, false, true}
	got := make([]bool, len(a))
	simd.RunStartsInto(got, a)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("i=%d: got %v want %v", i, got[i], want[i])
		}
	}
	// Empty and single-element are the boundary cases.
	simd.RunStartsInto(nil, nil)
	one := make([]bool, 1)
	simd.RunStartsInto(one, []int32{9})
	if !one[0] {
		t.Error("a single element must start a run")
	}
}

func TestRunLengthRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(801, 809))
	for _, n := range []int{0, 1, 2, 15, 16, 17, 100, 5000} {
		for _, runLen := range []int{1, 2, 7, 50} {
			a := make([]int32, n)
			for i := range a {
				// Runs of about runLen, so both the run-heavy and the
				// all-distinct cases get exercised.
				a[i] = int32(i / runLen)
				if runLen == 1 {
					a[i] = int32(r.Uint32())
				}
			}
			scratch := make([]bool, n)
			vals := make([]int32, n)
			lens := make([]int32, n)
			runs := simd.RunLengthEncodeInt32(vals, lens, a, scratch)

			// The lengths must account for every element.
			total := 0
			for i := 0; i < runs; i++ {
				total += int(lens[i])
			}
			if total != n {
				t.Fatalf("n=%d runLen=%d: lengths sum to %d", n, runLen, total)
			}
			// Adjacent runs must differ, or they should have been one run.
			for i := 1; i < runs; i++ {
				if vals[i] == vals[i-1] {
					t.Fatalf("n=%d runLen=%d: runs %d and %d have the same value %d",
						n, runLen, i-1, i, vals[i])
				}
			}

			back := make([]int32, n)
			if got := simd.RunLengthDecodeInt32(back, vals[:runs], lens[:runs]); got != n {
				t.Fatalf("n=%d runLen=%d: decoded %d elements", n, runLen, got)
			}
			for i := range a {
				if back[i] != a[i] {
					t.Fatalf("n=%d runLen=%d i=%d: %d != %d", n, runLen, i, back[i], a[i])
				}
			}
		}
	}
}

// A constant column is the case RLE exists for: one run however long it is.
func TestRunLengthConstantColumn(t *testing.T) {
	const n = 10000
	a := make([]int32, n)
	for i := range a {
		a[i] = 42
	}
	scratch := make([]bool, n)
	vals := make([]int32, n)
	lens := make([]int32, n)
	if runs := simd.RunLengthEncodeInt32(vals, lens, a, scratch); runs != 1 {
		t.Fatalf("a constant column encoded to %d runs, want 1", runs)
	}
	if vals[0] != 42 || lens[0] != n {
		t.Errorf("got value %d length %d, want 42 and %d", vals[0], lens[0], n)
	}
}

// Short output must stop cleanly rather than overrun.
func TestRunLengthShortOutput(t *testing.T) {
	a := []int32{1, 2, 3, 4, 5}
	scratch := make([]bool, len(a))
	vals := make([]int32, 2)
	lens := make([]int32, 2)
	if runs := simd.RunLengthEncodeInt32(vals, lens, a, scratch); runs != 2 {
		t.Errorf("got %d runs, want 2 (the output was only that long)", runs)
	}

	dst := make([]int32, 3)
	if got := simd.RunLengthDecodeInt32(dst, []int32{7}, []int32{100}); got != 3 {
		t.Errorf("decode into a length-3 buffer wrote %d", got)
	}
	for i := range dst {
		if dst[i] != 7 {
			t.Errorf("dst[%d] = %d, want 7", i, dst[i])
		}
	}
}

func TestRunLengthNoAlloc(t *testing.T) {
	const n = 4096
	a := make([]int32, n)
	for i := range a {
		a[i] = int32(i / 8)
	}
	scratch := make([]bool, n)
	vals := make([]int32, n)
	lens := make([]int32, n)
	if x := testing.AllocsPerRun(20, func() {
		simd.RunLengthEncodeInt32(vals, lens, a, scratch)
	}); x != 0 {
		t.Errorf("RunLengthEncodeInt32 allocated %v times per run", x)
	}
}
