package conformance

// The saturating operations, at the exact boundaries.
//
// The differential sweep in conformance_test.go already covers these against
// the reference, with a quarter of its inputs drawn from the ends of the
// range. This file is the other kind of test: a table that states what the
// answer *is*, so that a reference and a kernel which agreed with each other
// and were both wrong would still fail.
//
// The cases are the ones where an implementation can be plausible and wrong:
// the sum that lands exactly on the limit and must not clamp, the one that
// exceeds it by one and must, the unsigned difference that would wrap to a
// huge value, and the signed pair whose true sum overflows in both directions.

import (
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

type satCase[T comparable] struct {
	a, b, add, sub T
}

func runSat[T comparable](t *testing.T, tier, name string,
	got kernel.Ops[T], cases []satCase[T]) {
	t.Helper()

	// The kernels have a length threshold below which the dispatcher runs the
	// portable path, so each case is repeated to a length that reaches the
	// vector body — and to one that does not, since both must be right.
	for _, n := range []int{1, 3, 16, 17, 32, 33, 64, 65, 127} {
		a := make([]T, n)
		b := make([]T, n)
		wantAdd := make([]T, n)
		wantSub := make([]T, n)
		for i := range a {
			c := cases[i%len(cases)]
			a[i], b[i] = c.a, c.b
			wantAdd[i], wantSub[i] = c.add, c.sub
		}
		for _, op := range []struct {
			label string
			fn    func(dst, x, y []T)
			exp   []T
		}{
			{"SatAdd", got.SatAdd, wantAdd},
			{"SatSub", got.SatSub, wantSub},
		} {
			if op.fn == nil {
				continue
			}
			d := make([]T, n)
			op.fn(d, a, b)
			for i := range d {
				if d[i] != op.exp[i] {
					t.Errorf("%s/%s.%s n=%d i=%d: %v,%v gave %v, want %v",
						tier, name, op.label, n, i, a[i], b[i], d[i], op.exp[i])
					break
				}
			}
		}
	}
}

func TestSaturatingBoundaries(t *testing.T) {
	i8 := []satCase[int8]{
		{100, 27, 127, 73},     // lands exactly on the maximum
		{100, 28, 127, 72},     // one past it
		{127, 1, 127, 126},     // already at the maximum
		{-128, -1, -128, -127}, // already at the minimum
		{-100, -29, -128, -71}, // one past the minimum
		{-128, 127, -1, -128},  // the sum is representable; only the
		{127, -128, -1, 127},   // difference overflows, in each direction
		{0, 0, 0, 0},
	}
	u8 := []satCase[uint8]{
		{200, 55, 255, 145}, // exactly the maximum
		{200, 56, 255, 144}, // one past
		{255, 1, 255, 254},
		{0, 1, 1, 0}, // the unsigned difference clamps at zero, not 255
		{1, 200, 201, 0},
		{0, 0, 0, 0},
	}
	i16 := []satCase[int16]{
		{32000, 767, 32767, 31233},
		{32000, 768, 32767, 31232},
		{-32768, 1, -32767, -32768},
		{-32000, -768, -32768, -31232},
	}
	u16 := []satCase[uint16]{
		{65000, 535, 65535, 64465},
		{65000, 536, 65535, 64464},
		{0, 65535, 65535, 0},
	}
	i32 := []satCase[int32]{
		{2147483000, 647, 2147483647, 2147482353},
		{2147483000, 648, 2147483647, 2147482352},
		{-2147483648, -1, -2147483648, -2147483647},
		{-2147483648, 2147483647, -1, -2147483648},
	}
	u32 := []satCase[uint32]{
		{4294967000, 295, 4294967295, 4294966705},
		{4294967000, 296, 4294967295, 4294966704},
		{0, 1, 1, 0},
	}

	// The reference is checked first and by name. If it were wrong, every tier
	// would agree with it and the differential sweep would pass.
	want := ref.Set()
	t.Run("reference", func(t *testing.T) {
		runSat(t, "reference", "I8", want.I8, i8)
		runSat(t, "reference", "I16", want.I16, i16)
		runSat(t, "reference", "I32", want.I32, i32)
		runSat(t, "reference", "U8", want.U8, u8)
		runSat(t, "reference", "U16", want.U16, u16)
		runSat(t, "reference", "U32", want.U32, u32)
	})

	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			runSat(t, tier, "I8", got.I8, i8)
			runSat(t, tier, "I16", got.I16, i16)
			runSat(t, tier, "I32", got.I32, i32)
			runSat(t, tier, "U8", got.U8, u8)
			runSat(t, tier, "U16", got.U16, u16)
			runSat(t, tier, "U32", got.U32, u32)
		})
	}
}

// TestSaturatingIsNotWrapping is the property the whole feature exists for,
// stated on its own: on the inputs where the two differ, they must differ.
//
// Without it a build in which every saturating kernel had quietly been wired
// to the ordinary one would pass every table above that happens to avoid the
// limits, and this is the cheapest way to say that is not allowed.
func TestSaturatingIsNotWrapping(t *testing.T) {
	want := ref.Set()
	for tier, got := range tiers(t) {
		if got.U8.SatAdd == nil {
			continue
		}
		n := 64
		a, b := make([]uint8, n), make([]uint8, n)
		for i := range a {
			a[i], b[i] = 250, 10 // 260, which wraps to 4 and saturates to 255
		}
		sat, wrap := make([]uint8, n), make([]uint8, n)
		got.U8.SatAdd(sat, a, b)
		want.U8.Add(wrap, a, b)
		for i := range sat {
			if sat[i] != 255 {
				t.Fatalf("%s: SatAdd(250,10) = %d, want 255", tier, sat[i])
			}
			if wrap[i] != 4 {
				t.Fatalf("%s: Add(250,10) = %d, want 4 — the wrapping kernel is wrong",
					tier, wrap[i])
			}
		}
	}
}
