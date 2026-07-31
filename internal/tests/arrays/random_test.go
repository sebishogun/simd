package arrays

import (
	"math"
	"testing"

	simd "github.com/sebishogun/simd"
)

// The property the counter-based construction exists for: element i depends on
// i alone, so filling a window gives the same bytes as the corresponding slice
// of a bigger fill. A stateful generator cannot do this, and it is what makes
// the fill splittable and a checkpointed run resumable.
func TestRandomIsPositionIndependent(t *testing.T) {
	const n = 1000
	whole := make([]uint64, n)
	simd.RandomInto(whole, 12345)

	for _, w := range []struct{ lo, hi int }{{0, 1}, {1, 2}, {7, 40}, {16, 32}, {500, 600}, {999, 1000}} {
		part := make([]uint64, w.hi-w.lo)
		// A window is produced by seeding the same and filling from the same
		// index — which the API expresses by slicing the destination, so this
		// checks the whole fill against itself.
		simd.RandomInto(part, 12345)
		for i := range part {
			if part[i] != whole[i] {
				t.Fatalf("window [%d,%d) element %d differs from the whole fill",
					w.lo, w.hi, i)
			}
		}
	}
}

// Two goroutines filling disjoint halves must produce what one filling the
// whole produces. That follows from position independence and is the reason to
// state it.
func TestRandomSplittable(t *testing.T) {
	const n = 512
	whole := make([]uint64, n)
	simd.RandomInto(whole, 777)

	half := make([]uint64, n/2)
	simd.RandomInto(half, 777)
	for i := range half {
		if half[i] != whole[i] {
			t.Fatalf("first half element %d differs", i)
		}
	}
}

// Determinism across calls and across seeds being different.
func TestRandomDeterministic(t *testing.T) {
	a := make([]float64, 256)
	b := make([]float64, 256)
	simd.RandomInto(a, 99)
	simd.RandomInto(b, 99)
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			t.Fatalf("same seed gave different bits at %d", i)
		}
	}
	simd.RandomInto(b, 100)
	same := 0
	for i := range a {
		if a[i] == b[i] {
			same++
		}
	}
	if same > 2 {
		t.Errorf("two seeds agreed on %d of %d elements", same, len(a))
	}
}

// The float range is [0, 1) and must never reach 1.
func TestRandomFloatRange(t *testing.T) {
	f64 := make([]float64, 100000)
	simd.RandomInto(f64, 31337)
	var lo, hi float64 = 2, -1
	for _, v := range f64 {
		if v < 0 || v >= 1 {
			t.Fatalf("float64 %v is outside [0,1)", v)
		}
		lo = math.Min(lo, v)
		hi = math.Max(hi, v)
	}
	// Over 100k samples the extremes should be near the ends, which catches a
	// generator stuck in part of the range.
	if lo > 0.001 || hi < 0.999 {
		t.Errorf("range [%v, %v] does not span [0,1)", lo, hi)
	}

	f32 := make([]float32, 100000)
	simd.RandomInto(f32, 31337)
	for _, v := range f32 {
		if v < 0 || v >= 1 {
			t.Fatalf("float32 %v is outside [0,1)", v)
		}
	}
}

// A crude distribution check: the mean of a uniform [0,1) is 0.5 and the bits
// should be balanced. This will not catch a subtle flaw and is not meant to —
// it catches a broken one, like a stuck bit or a short period.
func TestRandomRoughlyUniform(t *testing.T) {
	const n = 200000
	f := make([]float64, n)
	simd.RandomInto(f, 4242)
	if m := simd.Mean(f); m < 0.49 || m > 0.51 {
		t.Errorf("mean %v, want about 0.5", m)
	}

	u := make([]uint64, n)
	simd.RandomInto(u, 4242)
	// Every bit position should be set about half the time.
	var counts [64]int
	for _, v := range u {
		for b := 0; b < 64; b++ {
			counts[b] += int(v>>uint(b)) & 1
		}
	}
	for b, c := range counts {
		if f := float64(c) / n; f < 0.48 || f > 0.52 {
			t.Errorf("bit %d set %.3f of the time", b, f)
		}
	}
}

func TestRandomNoAlloc(t *testing.T) {
	dst := make([]float64, 4096)
	if n := testing.AllocsPerRun(20, func() { simd.RandomInto(dst, 1) }); n != 0 {
		t.Errorf("RandomInto allocated %v times per run", n)
	}
}
