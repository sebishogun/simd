//go:build amd64 && !purego

package amd64

import (
	"math"
	"math/rand/v2"
	"testing"

	"golang.org/x/sys/cpu"
)

// Correctness tests for the generated kernels.
//
// This is what the whole pipeline exists to make trustworthy: C compiled by
// clang, lifted out of an object file as raw instruction encodings, wrapped in
// a hand-built ABI prologue, and reassembled by Go's assembler. Any mistake in
// that chain — a wrong frame offset, a misread symbol size, a botched
// prologue — shows up here as a wrong number rather than as a crash, which is
// exactly why it is checked against a plain scalar loop rather than eyeballed.
//
// Lengths are swept from 0 to 70. The vectorized body processes a whole
// register at a time and a scalar remainder loop handles the rest, so the
// interesting cases are all at the boundaries: zero, shorter than one vector,
// exactly one vector, one vector plus one, and so on past the 16-lane
// accumulator block the reductions use.
const maxLen = 70

func randF32(n int, r *rand.Rand) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64() * 8)
	}
	return s
}

func randF64(n int, r *rand.Rand) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64() * 8
	}
	return s
}

func hasAVX2() bool { return cpu.X86.HasAVX2 && cpu.X86.HasFMA }

// tiers returns the tiered implementations of one kernel shape that this CPU
// can actually execute. SSE2 is unconditional on amd64; AVX2 is not.
type binF32 struct {
	name string
	fn   func(dst, a, b []float32)
}

func binF32Tiers(sse2, avx2 func(dst, a, b []float32)) []binF32 {
	t := []binF32{{"sse2", sse2}}
	if hasAVX2() {
		t = append(t, binF32{"avx2", avx2})
	}
	return t
}

func TestAddFloat32(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, impl := range binF32Tiers(addFloat32SSE2, addFloat32AVX2) {
		t.Run(impl.name, func(t *testing.T) {
			for n := range maxLen + 1 {
				a, b := randF32(n, r), randF32(n, r)
				dst := make([]float32, n)
				impl.fn(dst, a, b)
				for i := range dst {
					if want := a[i] + b[i]; dst[i] != want {
						t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], want)
					}
				}
			}
		})
	}
}

func TestMulFloat64(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	impls := []struct {
		name string
		fn   func(dst, a, b []float64)
	}{{"sse2", mulFloat64SSE2}}
	if hasAVX2() {
		impls = append(impls, struct {
			name string
			fn   func(dst, a, b []float64)
		}{"avx2", mulFloat64AVX2})
	}
	for _, impl := range impls {
		t.Run(impl.name, func(t *testing.T) {
			for n := range maxLen + 1 {
				a, b := randF64(n, r), randF64(n, r)
				dst := make([]float64, n)
				impl.fn(dst, a, b)
				for i := range dst {
					if want := a[i] * b[i]; dst[i] != want {
						t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], want)
					}
				}
			}
		})
	}
}

func TestAddInt64(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for n := range maxLen + 1 {
		a, b := make([]int64, n), make([]int64, n)
		for i := range a {
			a[i], b[i] = int64(r.Uint64()), int64(r.Uint64())
		}
		dst := make([]int64, n)
		addInt64SSE2(dst, a, b)
		for i := range dst {
			// Wrapping is the documented behaviour, and Go's + wraps too.
			if want := a[i] + b[i]; dst[i] != want {
				t.Fatalf("n=%d i=%d: got %d want %d", n, i, dst[i], want)
			}
		}
	}
}

// A scalar argument travels through a floating-point register rather than the
// integer sequence, so it exercises a different part of the ABI prologue.
func TestScaleFloat32(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for n := range maxLen + 1 {
		a := randF32(n, r)
		dst := make([]float32, n)
		const s = 2.5
		scaleFloat32SSE2(dst, a, s)
		for i := range dst {
			if want := a[i] * s; dst[i] != want {
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], want)
			}
		}
	}
}

// AXPY mixes three pointers and a float scalar, the widest prologue here.
func TestAddScaledFloat64(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	for n := range maxLen + 1 {
		a, b := randF64(n, r), randF64(n, r)
		dst := make([]float64, n)
		const s = -1.25
		addScaledFloat64SSE2(dst, a, b, s)
		for i := range dst {
			if want := a[i] + b[i]*s; dst[i] != want {
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], want)
			}
		}
	}
}

// Reductions return a value, which exercises the epilogue as well as the
// prologue, and they must reproduce the fixed 16-lane accumulator tree that
// the portable reference uses. Comparing against that exact shape rather than
// a naive running sum is the point: it is the numerical contract.
func TestSumFloat64MatchesTheContract(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	for n := range maxLen + 1 {
		a := randF64(n, r)
		if got, want := sumFloat64SSE2(a), refSum(a); got != want {
			t.Fatalf("n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestDotFloat64MatchesTheContract(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	for n := range maxLen + 1 {
		a, b := randF64(n, r), randF64(n, r)
		if got, want := dotFloat64SSE2(a, b), refDot(a, b); got != want {
			t.Fatalf("n=%d: got %v want %v", n, got, want)
		}
	}
}

// refSum and refDot are the contract from package kernel, written out here so
// this test does not depend on the library it is verifying.
const sumLanes = 16

func refSum(a []float64) float64 {
	var acc [sumLanes]float64
	i := 0
	for ; i+sumLanes <= len(a); i += sumLanes {
		for j := range sumLanes {
			acc[j] += a[i+j]
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		acc[j] += a[i]
	}
	for w := sumLanes / 2; w >= 1; w /= 2 {
		for j := range w {
			acc[j] += acc[j+w]
		}
	}
	return acc[0]
}

func refDot(a, b []float64) float64 {
	var acc [sumLanes]float64
	i := 0
	for ; i+sumLanes <= len(a); i += sumLanes {
		for j := range sumLanes {
			acc[j] += a[i+j] * b[i+j]
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		acc[j] += a[i] * b[i]
	}
	for w := sumLanes / 2; w >= 1; w /= 2 {
		for j := range w {
			acc[j] += acc[j+w]
		}
	}
	return acc[0]
}

// The two tiers must agree with each other exactly, not merely each with the
// scalar loop. This is the check that a 128-bit and a 256-bit implementation
// of the same reduction produce the same bits, which is the guarantee the
// library makes and the one viterin/vek breaks.
func TestTiersAgreeExactly(t *testing.T) {
	if !hasAVX2() {
		t.Skip("no AVX2 on this CPU")
	}
	r := rand.New(rand.NewPCG(15, 16))
	for n := range maxLen + 1 {
		a, b := randF64(n, r), randF64(n, r)
		if s2, a2 := sumFloat64SSE2(a), sumFloat64AVX2(a); s2 != a2 {
			t.Fatalf("Sum n=%d: sse2 %v != avx2 %v", n, s2, a2)
		}
		if s2, a2 := dotFloat64SSE2(a, b), dotFloat64AVX2(a, b); s2 != a2 {
			t.Fatalf("Dot n=%d: sse2 %v != avx2 %v", n, s2, a2)
		}
		d1, d2 := make([]float64, n), make([]float64, n)
		addFloat64SSE2(d1, a, b)
		addFloat64AVX2(d2, a, b)
		for i := range d1 {
			if d1[i] != d2[i] {
				t.Fatalf("Add n=%d i=%d: sse2 %v != avx2 %v", n, i, d1[i], d2[i])
			}
		}
	}
}

// Non-finite values must survive the round trip. A kernel that quietly turns
// NaN into something else is the bug that makes a library untrustworthy.
func TestNonFiniteValues(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	a := []float64{nan, inf, -inf, 0, math.Copysign(0, -1), 1e308, 5e-324}
	b := []float64{1, 1, 1, 1, 1, 1e308, 5e-324}
	dst := make([]float64, len(a))
	addFloat64SSE2(dst, a, b)
	for i := range dst {
		want := a[i] + b[i]
		if dst[i] != want && !(math.IsNaN(dst[i]) && math.IsNaN(want)) {
			t.Errorf("i=%d: got %v want %v", i, dst[i], want)
		}
	}
}

// A kernel must write nothing past the length it was given.
func TestNoWriteBeyondLength(t *testing.T) {
	const n, guard = 20, 8
	buf := make([]float64, n+guard)
	for i := range buf {
		buf[i] = -12345
	}
	a := make([]float64, n)
	b := make([]float64, n)
	for i := range a {
		a[i], b[i] = float64(i), 1
	}
	addFloat64SSE2(buf[:n], a, b)
	for i := n; i < len(buf); i++ {
		if buf[i] != -12345 {
			t.Fatalf("wrote past the destination at index %d: %v", i, buf[i])
		}
	}
}

// //go:noescape is an unchecked promise that the assembly does not leak its
// pointer arguments. If it were wrong, or absent, every slice handed to a
// kernel would be heap-allocated.
func TestKernelsDoNotAllocate(t *testing.T) {
	a, b := make([]float64, 256), make([]float64, 256)
	dst := make([]float64, 256)
	cases := map[string]func(){
		"add":   func() { addFloat64SSE2(dst, a, b) },
		"sum":   func() { sinkF = sumFloat64SSE2(a) },
		"dot":   func() { sinkF = dotFloat64SSE2(a, b) },
		"scale": func() { scaleFloat64SSE2(dst, a, 2) },
		"axpy":  func() { addScaledFloat64SSE2(dst, a, b, 2) },
	}
	for name, fn := range cases {
		if got := testing.AllocsPerRun(50, fn); got != 0 {
			t.Errorf("%s allocated %.1f times per call, want 0", name, got)
		}
	}
}

var sinkF float64
