package simd

// The public API over the narrow and unsigned integer types.
//
// internal/conformance checks the kernels against the reference; this checks
// the generic layer above them, where the mistakes are of a different kind. A
// type missing from the switch in ops[T] panics rather than computing the
// wrong answer, and a type present but wired to the wrong group computes an
// answer of the right shape and the wrong value — neither is reachable from a
// kernel test, because the kernel is not what would be wrong.
//
// Everything here is checked against a plain Go loop written out in the test
// rather than against internal/ref, so that a reference which was itself wrong
// for the unsigned types could not make the test pass.

import (
	"math/rand/v2"
	"testing"
)

// intTypes is exercised through a generic helper per shape rather than a table
// of any values, because the API is generic: a test that reached it through
// interfaces would not be compiling the same code the caller compiles.
func genN[T Number](n int, seed uint64) []T {
	r := rand.New(rand.NewPCG(seed, seed+1))
	s := make([]T, n)
	for i := range s {
		s[i] = T(r.Uint64())
	}
	return s
}

// checkElementwise runs the whole in-place / Into contract for one element
// type against loops written out here.
func checkElementwise[T Number](t *testing.T, name string) {
	t.Helper()
	for _, n := range []int{0, 1, 7, 15, 16, 17, 63, 64, 65, 257} {
		a, b := genN[T](n, 1), genN[T](n, 2)

		want := make([]T, n)
		for i := range want {
			want[i] = a[i] + b[i]
		}
		got := make([]T, n)
		AddInto(got, a, b)
		eq(t, name+".AddInto", n, got, want)
		inPlace := append([]T(nil), a...)
		Add(inPlace, b)
		eq(t, name+".Add", n, inPlace, want)

		for i := range want {
			want[i] = a[i] - b[i]
		}
		SubInto(got, a, b)
		eq(t, name+".SubInto", n, got, want)

		for i := range want {
			want[i] = a[i] * b[i]
		}
		MulInto(got, a, b)
		eq(t, name+".MulInto", n, got, want)

		for i := range want {
			want[i] = min(a[i], b[i])
		}
		MinimumInto(got, a, b)
		eq(t, name+".MinimumInto", n, got, want)

		for i := range want {
			want[i] = max(a[i], b[i])
		}
		MaximumInto(got, a, b)
		eq(t, name+".MaximumInto", n, got, want)

		// Neg on an unsigned type is the wraparound complement, which is what
		// Go's own unary minus gives, and Abs is the identity.
		for i := range want {
			want[i] = -a[i]
		}
		NegInto(got, a)
		eq(t, name+".NegInto", n, got, want)

		for i := range want {
			if a[i] < 0 {
				want[i] = -a[i]
			} else {
				want[i] = a[i]
			}
		}
		AbsInto(got, a)
		eq(t, name+".AbsInto", n, got, want)

		var s T
		if len(a) > 0 {
			s = a[0]
		}
		for i := range want {
			want[i] = a[i] + s
		}
		AddScalarInto(got, a, s)
		eq(t, name+".AddScalarInto", n, got, want)
	}
}

func checkReductions[T Number](t *testing.T, name string) {
	t.Helper()
	for _, n := range []int{1, 7, 16, 17, 64, 65, 257} {
		a, b := genN[T](n, 3), genN[T](n, 4)

		var wantSum, wantDot T
		for i := range a {
			wantSum += a[i]
			wantDot += a[i] * b[i]
		}
		if g := Sum(a); g != wantSum {
			t.Errorf("%s.Sum n=%d: got %v want %v", name, n, g, wantSum)
		}
		if g := Dot(a, b); g != wantDot {
			t.Errorf("%s.Dot n=%d: got %v want %v", name, n, g, wantDot)
		}

		wantMin, wantMax := a[0], a[0]
		for _, v := range a {
			wantMin, wantMax = min(wantMin, v), max(wantMax, v)
		}
		if g := Min(a); g != wantMin {
			t.Errorf("%s.Min n=%d: got %v want %v", name, n, g, wantMin)
		}
		if g := Max(a); g != wantMax {
			t.Errorf("%s.Max n=%d: got %v want %v", name, n, g, wantMax)
		}
	}
}

// checkOrderIsUnsigned is the check that matters most for the unsigned types
// and is invisible for every other one: the comparison must be unsigned.
//
// A kernel that compares uint8 lanes as signed is right for everything below
// 0x80 and wrong for everything at or above it, which is half the range and
// none of the values a casual test happens to use.
func checkOrderIsUnsigned[T Number](t *testing.T, name string, small, large T) {
	t.Helper()
	n := 64
	a, b := make([]T, n), make([]T, n)
	for i := range a {
		a[i], b[i] = large, small
	}
	got := make([]T, n)
	MinimumInto(got, a, b)
	for i := range got {
		if got[i] != small {
			t.Fatalf("%s.Minimum(%v,%v) = %v, want %v — compared as signed?",
				name, large, small, got[i], small)
		}
	}
	mask := make([]bool, n)
	LessInto(mask, a, b)
	for i := range mask {
		if mask[i] {
			t.Fatalf("%s.Less(%v,%v) = true, want false — compared as signed?",
				name, large, small)
		}
	}
	if g := Max(a); g != large {
		t.Fatalf("%s.Max of all %v gave %v", name, large, g)
	}
}

func eq[T comparable](t *testing.T, op string, n int, got, want []T) {
	t.Helper()
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s n=%d i=%d: got %v want %v", op, n, i, got[i], want[i])
		}
	}
}

func TestIntegerTypesThroughThePublicAPI(t *testing.T) {
	checkElementwise[int8](t, "int8")
	checkElementwise[int16](t, "int16")
	checkElementwise[int32](t, "int32")
	checkElementwise[int64](t, "int64")
	checkElementwise[uint8](t, "uint8")
	checkElementwise[uint16](t, "uint16")
	checkElementwise[uint32](t, "uint32")
	checkElementwise[uint64](t, "uint64")

	checkReductions[int8](t, "int8")
	checkReductions[int16](t, "int16")
	checkReductions[uint8](t, "uint8")
	checkReductions[uint16](t, "uint16")
	checkReductions[uint32](t, "uint32")
	checkReductions[uint64](t, "uint64")

	// large has the top bit set in every case, so an implementation that
	// compared as signed would call it the smaller of the two.
	checkOrderIsUnsigned[uint8](t, "uint8", 1, 0xff)
	checkOrderIsUnsigned[uint16](t, "uint16", 1, 0xffff)
	checkOrderIsUnsigned[uint32](t, "uint32", 1, 0xffffffff)
	checkOrderIsUnsigned[uint64](t, "uint64", 1, 0xffffffffffffffff)
}

// TestSaturatingThroughThePublicAPI checks the wrapper pairing for the two
// operations added with these types, since neither is covered by the tables in
// api_test.go — those are float64 only.
func TestSaturatingThroughThePublicAPI(t *testing.T) {
	check := func(name string, f func(t *testing.T)) { t.Run(name, f) }

	check("uint8", func(t *testing.T) {
		a := []uint8{0, 1, 128, 200, 254, 255}
		b := []uint8{255, 255, 200, 100, 1, 1}
		wantAdd := []uint8{255, 255, 255, 255, 255, 255}
		wantSub := []uint8{0, 0, 0, 100, 253, 254}
		gotAdd := make([]uint8, len(a))
		SatAddInto(gotAdd, a, b)
		eq(t, "uint8.SatAddInto", len(a), gotAdd, wantAdd)
		gotSub := append([]uint8(nil), a...)
		SatSub(gotSub, b)
		eq(t, "uint8.SatSub", len(a), gotSub, wantSub)
	})

	check("int8", func(t *testing.T) {
		a := []int8{127, -128, 100, -100, 0}
		b := []int8{1, -1, 27, -28, 0}
		wantAdd := []int8{127, -128, 127, -128, 0}
		wantSub := []int8{126, -127, 73, -72, 0}
		gotAdd := append([]int8(nil), a...)
		SatAdd(gotAdd, b)
		eq(t, "int8.SatAdd", len(a), gotAdd, wantAdd)
		gotSub := make([]int8, len(a))
		SatSubInto(gotSub, a, b)
		eq(t, "int8.SatSubInto", len(a), gotSub, wantSub)
	})

	check("bounded by the shorter slice", func(t *testing.T) {
		a := make([]uint16, 40)
		b := make([]uint16, 10)
		for i := range a {
			a[i] = 65535
		}
		for i := range b {
			b[i] = 1
		}
		SatAdd(a, b)
		for i, v := range a {
			if i < 10 && v != 65535 {
				t.Fatalf("i=%d: got %d, want 65535", i, v)
			}
			if i >= 10 && v != 65535 {
				t.Fatalf("i=%d: element past the shorter slice was changed to %d", i, v)
			}
		}
	})
}

// TestIntegerWrappersDoNotAllocate extends the promise in api_test.go to the
// types added after it was written.
func TestIntegerWrappersDoNotAllocate(t *testing.T) {
	const n = 1024
	a8, b8, d8 := make([]uint8, n), make([]uint8, n), make([]uint8, n)
	a16, b16, d16 := make([]int16, n), make([]int16, n), make([]int16, n)
	a64, b64, d64 := make([]uint64, n), make([]uint64, n), make([]uint64, n)
	mask := make([]bool, n)

	for _, c := range []struct {
		name string
		fn   func()
	}{
		{"AddInto/u8", func() { AddInto(d8, a8, b8) }},
		{"SatAddInto/u8", func() { SatAddInto(d8, a8, b8) }},
		{"SatSubInto/i16", func() { SatSubInto(d16, a16, b16) }},
		{"MinimumInto/u64", func() { MinimumInto(d64, a64, b64) }},
		{"Sum/u8", func() { sinkU8API = Sum(a8) }},
		{"Max/u64", func() { sinkU64API = Max(a64) }},
		{"LessInto/u8", func() { LessInto(mask, a8, b8) }},
	} {
		if got := testing.AllocsPerRun(50, c.fn); got != 0 {
			t.Errorf("%s allocated %.0f times per call, want 0", c.name, got)
		}
	}
}

var (
	sinkU8API  uint8
	sinkU64API uint64
)
