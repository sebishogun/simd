// Package conformance checks every generated kernel against the portable
// reference, for every instruction-set tier the host can execute.
//
// This is the test the whole design exists to make possible. Backends are
// swapped as complete sets rather than per function, so every tier compiled
// into the binary can be exercised in one process — no re-running the suite
// once per GOSIMD setting, and no chance of a tier being skipped because
// nobody thought to set the variable.
//
// What is checked, and why it is checked this way:
//
//   - Elementwise and scalar operations must be bit-identical to the
//     reference. Not close: identical, including NaN payloads, ±Inf, ±0 and
//     denormals. No reassociation is possible in an elementwise operation, so
//     there is no reason for a difference and any difference is a bug.
//   - Reductions must be bit-identical too, which is the contract that stops a
//     sum changing value when a program moves to a machine with wider vectors.
//   - Lengths are swept from 0 to 70 rather than sampled. A vectorized body
//     handles whole blocks and a remainder loop handles the rest, so every
//     interesting case is at a boundary. viterin/vek#11 and its Any bug both
//     live exactly there.
//   - Inputs include NaN, infinities, signed zeros, denormals and the extremes
//     of each integer type, because those are where a compare-and-blend
//     implementation of minimum diverges from a naive one.
package conformance

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
	"unsafe"

	"github.com/sebishogun/simd/internal/backend"
	"github.com/sebishogun/simd/internal/cpu"
	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

const maxLen = 70

// tiers returns every tier with a backend compiled into this binary, paired
// with the reference to compare it against.
// tiers returns every tier that is both compiled into this binary and
// executable by this CPU.
//
// Both halves matter. Exercising every compiled-in tier is the point — it is
// what lets one process compare avx2 against avx512 against the reference
// without re-running the suite per GOSIMD setting. But a tier the host cannot
// execute is not a test failure, it is a SIGILL: the process dies on the first
// instruction with no message, which is exactly what happened on a riscv64
// emulator without the vector extension. The same would happen to anyone
// running this suite on an amd64 machine older than AVX-512.
//
// A skipped tier is logged rather than passed over quietly. A suite that
// silently tested nothing would look identical to one that passed.
func tiers(t *testing.T) map[string]kernel.Set {
	runnable := map[string]bool{}
	for _, tr := range cpu.Detail().Available {
		runnable[tr.String()] = true
	}
	out := map[string]kernel.Set{}
	for _, name := range backend.Tiers() {
		s, ok := backend.Lookup(name)
		if !ok {
			continue
		}
		if !runnable[name] {
			t.Logf("skipping tier %s: compiled in, but this CPU cannot execute it (%s)",
				name, cpu.Describe())
			continue
		}
		out[name] = s
	}
	if len(out) == 0 {
		t.Skip("no generated backend in this build that this CPU can execute")
	}
	return out
}

// ---------- input generation ----------

// awkwardF64 are the values an implementation is most likely to get wrong.
var awkwardF64 = []float64{
	0, math.Copysign(0, -1),
	1, -1, 0.5, -0.5,
	math.Inf(1), math.Inf(-1),
	math.NaN(),
	math.MaxFloat64, -math.MaxFloat64,
	math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
	1e308, -1e308, 1e-308,
}

func genF64(n int, r *rand.Rand) []float64 {
	s := make([]float64, n)
	for i := range s {
		// Mix ordinary values with the awkward ones, so both the common path
		// and the corner cases are covered at every length.
		if r.IntN(4) == 0 {
			s[i] = awkwardF64[r.IntN(len(awkwardF64))]
		} else {
			s[i] = r.NormFloat64() * 100
		}
	}
	return s
}

func genF32(n int, r *rand.Rand) []float32 {
	s := make([]float32, n)
	for i := range s {
		if r.IntN(4) == 0 {
			s[i] = float32(awkwardF64[r.IntN(len(awkwardF64))])
		} else {
			s[i] = float32(r.NormFloat64() * 100)
		}
	}
	return s
}

// intElem is every integer element type, for the one generator that serves
// them all.
type intElem interface {
	~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64
}

// genInt produces integer inputs weighted towards the values that separate a
// right kernel from a plausible one.
//
// A quarter of the elements are drawn from the ends of the range, because that
// is where the interesting disagreements are: wrapping against saturation, Abs
// of the most negative value, and — the reason this matters more for the
// unsigned types than the signed ones — a comparison done as signed when it
// should be unsigned, which is correct for the whole bottom half of the range
// and wrong for the top. Uniform random bytes would reach 0x80..0xff often
// enough to catch that one; uniform random int32s would essentially never
// produce MaxInt32.
func genInt[T intElem](n int, r *rand.Rand) []T {
	lo, hi := intBounds[T]()
	extremes := []T{0, 1, lo, hi, hi - 1, lo + 1, hi / 2}
	s := make([]T, n)
	for i := range s {
		if r.IntN(4) == 0 {
			s[i] = extremes[r.IntN(len(extremes))]
			continue
		}
		s[i] = T(r.Uint64())
	}
	return s
}

// intBounds is the closed range of T, derived from the type rather than
// tabulated: the size gives the width and the wraparound of 0-1 gives the
// signedness, so a type cannot be given the wrong bounds by hand.
func intBounds[T intElem]() (lo, hi T) {
	var zero T
	bits := 8 * uint(unsafe.Sizeof(zero))
	if zero-1 > zero {
		return 0, ^T(0)
	}
	hi = T(uint64(1)<<(bits-1) - 1)
	return -hi - 1, hi
}

func genI32(n int, r *rand.Rand) []int32 { return genInt[int32](n, r) }
func genI64(n int, r *rand.Rand) []int64 { return genInt[int64](n, r) }

// same compares bit patterns, so -0 does not equal +0 and a lost sign bit is
// caught. The one exception is the NaN payload.
//
// Which NaN survives an operation with two NaN operands is not specified by
// IEEE 754, and hardware genuinely differs — x86 returns the first source
// operand, other architectures make other choices. Demanding identical
// payloads would be demanding something no implementation can promise. What
// the library does promise, and what is checked, is that a NaN in produces a
// NaN out.
func same[T comparable](got, want []T) (int, bool) {
	if len(got) != len(want) {
		return -1, false
	}
	for i := range got {
		if !sameScalar(got[i], want[i]) {
			return i, false
		}
	}
	return 0, true
}

func sameScalar[T comparable](a, b T) bool {
	switch x := any(a).(type) {
	case float64:
		y := any(b).(float64)
		if math.IsNaN(x) || math.IsNaN(y) {
			return math.IsNaN(x) && math.IsNaN(y)
		}
		return math.Float64bits(x) == math.Float64bits(y)
	case float32:
		x64, y64 := float64(x), float64(any(b).(float32))
		if math.IsNaN(x64) || math.IsNaN(y64) {
			return math.IsNaN(x64) && math.IsNaN(y64)
		}
		return math.Float32bits(x) == math.Float32bits(any(b).(float32))
	}
	return a == b
}

// ---------- shape checkers ----------
//
// Each takes the kernel under test and the reference, runs both on the same
// inputs at every length, and compares bit patterns.

func checkBinary[T comparable](t *testing.T, tier, op string,
	got, want func(dst, a, b []T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(1, 2))
	for n := range maxLen + 1 {
		a, b := gen(n, r), gen(n, r)
		g, w := make([]T, n), make([]T, n)
		got(g, a, b)
		want(w, a, b)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d: got %v want %v (a=%v b=%v)",
				tier, op, n, i, g[i], w[i], a[i], b[i])
		}
	}
}

func checkUnary[T comparable](t *testing.T, tier, op string,
	got, want func(dst, a []T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(3, 4))
	for n := range maxLen + 1 {
		a := gen(n, r)
		g, w := make([]T, n), make([]T, n)
		got(g, a)
		want(w, a)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d: got %v want %v (a=%v)", tier, op, n, i, g[i], w[i], a[i])
		}
	}
}

// checkLowerBound compares the batch binary search.
//
// The table has to be sorted, which the generic generator does not produce, so
// it is sorted here. The queries are deliberately NOT sorted and are drawn from
// the same distribution as the table, so a good share of them land exactly on a
// table entry — which is the case lower_bound is defined by and the one an
// off-by-one in the descent gets wrong while still looking plausible.
//
// Table lengths run past the dispatch threshold and sit on both sides of a
// power of two, because the kernel steps down through powers of two and the top
// step is the one a length of exactly 2^k would hide.
func checkLowerBound[T comparable](t *testing.T, tier, op string,
	got, want func(dst []int32, a, q []T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(23, 24))
	for _, n := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 200} {
		a := gen(n, r)
		slices.SortFunc(a, compareElem[T])
		for _, m := range []int{0, 1, 17, 64, 100} {
			q := gen(m, r)
			// Half the queries replaced by table entries, so exact hits are
			// common rather than accidental.
			for i := 0; i < len(q) && n > 0; i += 2 {
				q[i] = a[r.IntN(n)]
			}
			g, w := make([]int32, m), make([]int32, m)
			got(g, a, q)
			want(w, a, q)
			if i, ok := same(g, w); !ok {
				t.Fatalf("%s/%s ntab=%d nq=%d i=%d: got %d want %d",
					tier, op, n, m, i, g[i], w[i])
			}
		}
	}
}

// checkSet compares a sorted-set operation.
//
// Two things about the inputs are load-bearing, and getting either wrong makes
// this test pass while checking nothing.
//
// They have to overlap. Two independently generated sets of int32 essentially
// never intersect, so a first version of this compared two empty answers on
// every iteration and would have accepted any implementation at all. Here b is
// built from a — every third element, plus fresh values to give a elements that
// b lacks — so both the match and the no-match path run on every case.
//
// They have to be long enough to reach the kernel. The dispatch threshold for
// these is 64 elements, so lengths stop at maxLen would test the portable
// reference against itself. The lengths below run to several tiles past it.
//
// The destination is allocated to the full length the public wrapper requires,
// because the kernel is at its ABI's six-argument limit and is not told the
// destination's length; a short one would be testing an overrun.
func checkSet[T comparable](t *testing.T, tier, op string,
	got, want func(dst, a, b []T) int, gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(19, 20))
	for _, n := range []int{0, 1, 7, 8, 9, 63, 64, 65, 71, 128, 129, 500} {
		a := sortedDistinct(gen(n, r))
		// Every third element of a, so a keeps elements b does not have, plus
		// as many fresh values so b keeps elements a does not have. Both are
		// needed: intersection differs from difference only on those.
		for _, extra := range []int{0, 1, n / 2, n} {
			b := make([]T, 0, len(a)/3+extra)
			for i := 0; i < len(a); i += 3 {
				b = append(b, a[i])
			}
			b = sortedDistinct(append(b, gen(extra, r)...))

			g, w := make([]T, len(a)), make([]T, len(a))
			gn, wn := got(g, a, b), want(w, a, b)
			if gn != wn {
				t.Fatalf("%s/%s na=%d nb=%d: wrote %d, want %d",
					tier, op, len(a), len(b), gn, wn)
			}
			if i, ok := same(g[:gn], w[:wn]); !ok {
				t.Fatalf("%s/%s na=%d nb=%d i=%d: got %v want %v",
					tier, op, len(a), len(b), i, g[i], w[i])
			}
		}
	}
}

// sortedDistinct sorts in place and drops duplicates, which is the input
// contract the set operations are allowed to assume.
//
// The ordering goes through a type switch because checkOps is generic over
// `comparable`, which has no <. That is the right constraint for the rest of
// the suite — same() compares for equality and nothing else needs an order —
// so the switch lives here rather than being pushed up into every caller.
func sortedDistinct[T comparable](a []T) []T {
	slices.SortFunc(a, compareElem[T])
	return slices.Compact(a)
}

func compareElem[T comparable](x, y T) int {
	switch xv := any(x).(type) {
	case int8:
		return cmp.Compare(xv, any(y).(int8))
	case int16:
		return cmp.Compare(xv, any(y).(int16))
	case int32:
		return cmp.Compare(xv, any(y).(int32))
	case int64:
		return cmp.Compare(xv, any(y).(int64))
	case uint8:
		return cmp.Compare(xv, any(y).(uint8))
	case uint16:
		return cmp.Compare(xv, any(y).(uint16))
	case uint32:
		return cmp.Compare(xv, any(y).(uint32))
	case uint64:
		return cmp.Compare(xv, any(y).(uint64))
	case float32:
		return cmp.Compare(xv, any(y).(float32))
	case float64:
		return cmp.Compare(xv, any(y).(float64))
	}
	return 0
}

// checkRolling compares a sliding-window operation across window sizes.
//
// The output length is len(a)-window+1, which is neither of the two lengths
// the generated guard knows about, so this is the shape where a clamping
// mistake shows up as a silently short answer rather than a crash. The
// destination is deliberately allocated to exactly that length: anything
// written past it corrupts the heap and the race detector or the allocator
// will say so.
func checkRolling[T comparable](t *testing.T, tier, op string,
	got, want func(dst, a []T, window int), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(17, 18))
	for n := range maxLen + 1 {
		a := gen(n, r)
		// Window sizes spanning one, a vector width, and the whole slice,
		// plus the two degenerate cases that must write nothing.
		for _, w := range []int{0, 1, 2, 3, 8, 16, 17, n, n + 1} {
			if w <= 0 || w > n {
				// Still call both: writing nothing is part of the contract,
				// and a kernel that scribbled here would be caught by the
				// comparison below.
				g, wv := make([]T, max(n, 1)), make([]T, max(n, 1))
				got(g, a, w)
				want(wv, a, w)
				if i, ok := same(g, wv); !ok {
					t.Fatalf("%s/%s n=%d window=%d i=%d: got %v want %v",
						tier, op, n, w, i, g[i], wv[i])
				}
				continue
			}
			m := n - w + 1
			g, wv := make([]T, m), make([]T, m)
			got(g, a, w)
			want(wv, a, w)
			if i, ok := same(g, wv); !ok {
				t.Fatalf("%s/%s n=%d window=%d i=%d: got %v want %v",
					tier, op, n, w, i, g[i], wv[i])
			}
		}
	}
}

// checkScalar compares an operation taking one scalar operand.
//
// nonZero excludes a zero scalar, which is only needed for integer division:
// Go's / panics on a zero divisor and the kernels inherit that, so feeding one
// in would be testing the panic rather than the arithmetic.
func checkScalar[T comparable](t *testing.T, tier, op string,
	got, want func(dst, a []T, s T), gen func(int, *rand.Rand) []T, nonZero bool) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	var zero T
	r := rand.New(rand.NewPCG(5, 6))
	for n := range maxLen + 1 {
		a := gen(n, r)
		one := gen(1, r)
		s := one[0]
		for nonZero && s == zero {
			s = gen(1, r)[0]
		}
		g, w := make([]T, n), make([]T, n)
		got(g, a, s)
		want(w, a, s)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d s=%v: got %v want %v", tier, op, n, i, s, g[i], w[i])
		}
	}
}

func checkReduce1[T comparable](t *testing.T, tier, op string,
	got, want func(a []T) T, gen func(int, *rand.Rand) []T, skipEmpty bool) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(7, 8))
	start := 0
	if skipEmpty {
		start = 1
	}
	for n := start; n <= maxLen; n++ {
		a := gen(n, r)
		g, w := got(a), want(a)
		if !sameScalar(g, w) {
			t.Fatalf("%s/%s n=%d: got %v want %v", tier, op, n, g, w)
		}
	}
}

func checkReduce2[T comparable](t *testing.T, tier, op string,
	got, want func(a, b []T) T, gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(9, 10))
	for n := range maxLen + 1 {
		a, b := gen(n, r), gen(n, r)
		g, w := got(a, b), want(a, b)
		if !sameScalar(g, w) {
			t.Fatalf("%s/%s n=%d: got %v want %v", tier, op, n, g, w)
		}
	}
}

func checkClamp[T comparable](t *testing.T, tier, op string,
	got, want func(dst, a []T, lo, hi T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(11, 12))
	for n := range maxLen + 1 {
		a := gen(n, r)
		b := gen(2, r)
		lo, hi := b[0], b[1]
		g, w := make([]T, n), make([]T, n)
		got(g, a, lo, hi)
		want(w, a, lo, hi)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d lo=%v hi=%v: got %v want %v",
				tier, op, n, i, lo, hi, g[i], w[i])
		}
	}
}

func checkFill[T comparable](t *testing.T, tier, op string,
	got, want func(dst []T, v T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(13, 14))
	for n := range maxLen + 1 {
		v := gen(1, r)[0]
		g, w := make([]T, n), make([]T, n)
		got(g, v)
		want(w, v)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d: got %v want %v", tier, op, n, i, g[i], w[i])
		}
	}
}

func checkLerp[T comparable](t *testing.T, tier, op string,
	got, want func(dst, a, b []T, s T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(15, 16))
	for n := range maxLen + 1 {
		a, b := gen(n, r), gen(n, r)
		s := gen(1, r)[0]
		g, w := make([]T, n), make([]T, n)
		got(g, a, b, s)
		want(w, a, b, s)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d s=%v: got %v want %v", tier, op, n, i, s, g[i], w[i])
		}
	}
}

// ---------- the suite ----------

// checkOps runs every shape in one Ops group.
func checkOps[T comparable](t *testing.T, tier, typeName string,
	got, want kernel.Ops[T], gen func(int, *rand.Rand) []T) {

	p := func(op string) string { return typeName + "." + op }

	checkBinary(t, tier, p("Add"), got.Add, want.Add, gen)
	checkBinary(t, tier, p("Sub"), got.Sub, want.Sub, gen)
	checkBinary(t, tier, p("Mul"), got.Mul, want.Mul, gen)
	checkBinary(t, tier, p("Div"), got.Div, want.Div, gen)
	checkBinary(t, tier, p("Minimum"), got.Minimum, want.Minimum, gen)
	checkBinary(t, tier, p("Maximum"), got.Maximum, want.Maximum, gen)

	checkUnary(t, tier, p("Abs"), got.Abs, want.Abs, gen)
	checkUnary(t, tier, p("Neg"), got.Neg, want.Neg, gen)
	checkUnary(t, tier, p("Sqrt"), got.Sqrt, want.Sqrt, gen)
	checkUnary(t, tier, p("Reciprocal"), got.Reciprocal, want.Reciprocal, gen)
	checkUnary(t, tier, p("Floor"), got.Floor, want.Floor, gen)
	checkUnary(t, tier, p("Ceil"), got.Ceil, want.Ceil, gen)
	checkUnary(t, tier, p("Trunc"), got.Trunc, want.Trunc, gen)
	checkUnary(t, tier, p("Round"), got.Round, want.Round, gen)
	checkUnary(t, tier, p("RoundToEven"), got.RoundToEven, want.RoundToEven, gen)
	checkUnary(t, tier, p("Reverse"), got.Reverse, want.Reverse, gen)

	checkScalar(t, tier, p("Scale"), got.Scale, want.Scale, gen, false)
	checkScalar(t, tier, p("AddScalar"), got.AddScalar, want.AddScalar, gen, false)
	checkScalar(t, tier, p("SubScalar"), got.SubScalar, want.SubScalar, gen, false)
	// Integer division panics on a zero divisor, in the kernels exactly as in
	// Go, so the scalar is kept away from zero.
	checkScalar(t, tier, p("DivScalar"), got.DivScalar, want.DivScalar, gen, true)

	checkClamp(t, tier, p("Clamp"), got.Clamp, want.Clamp, gen)
	checkFill(t, tier, p("Fill"), got.Fill, want.Fill, gen)
	checkLerp(t, tier, p("Lerp"), got.Lerp, want.Lerp, gen)
	checkLerp(t, tier, p("AddScaled"), got.AddScaled, want.AddScaled, gen)

	checkUnary(t, tier, p("CumSum"), got.CumSum, want.CumSum, gen)
	checkUnary(t, tier, p("CumProd"), got.CumProd, want.CumProd, gen)
	checkUnary(t, tier, p("CumMin"), got.CumMin, want.CumMin, gen)
	checkUnary(t, tier, p("CumMax"), got.CumMax, want.CumMax, gen)

	checkRolling(t, tier, p("RollingMin"), got.RollingMin, want.RollingMin, gen)
	checkRolling(t, tier, p("RollingMax"), got.RollingMax, want.RollingMax, gen)

	checkLowerBound(t, tier, p("LowerBound"), got.LowerBound, want.LowerBound, gen)

	checkSet(t, tier, p("Intersect"), got.Intersect, want.Intersect, gen)
	checkSet(t, tier, p("Difference"), got.Difference, want.Difference, gen)

	// The Fast scans belong here for exactly the reason their name might
	// suggest they do not. What Fast relaxes is agreement with a naive serial
	// loop; agreement with the REFERENCE — which runs the same blocked
	// grouping — is not relaxed at all, so these are bit-identity checks like
	// every other line above. If they ever needed a tolerance, the tiers would
	// have diverged and the promise would be gone.
	checkUnary(t, tier, p("FastCumSum"), got.FastCumSum, want.FastCumSum, gen)
	checkUnary(t, tier, p("FastCumProd"), got.FastCumProd, want.FastCumProd, gen)

	checkReduce1(t, tier, p("Sum"), got.Sum, want.Sum, gen, false)
	checkReduce1(t, tier, p("Prod"), got.Prod, want.Prod, gen, false)
	checkReduce1(t, tier, p("SumSquares"), got.SumSquares, want.SumSquares, gen, false)
	checkReduce1(t, tier, p("L1Norm"), got.L1Norm, want.L1Norm, gen, false)
	checkReduce1(t, tier, p("Norm"), got.Norm, want.Norm, gen, false)
	checkReduce1(t, tier, p("Min"), got.Min, want.Min, gen, true)
	checkReduce1(t, tier, p("Max"), got.Max, want.Max, gen, true)

	checkReduce2(t, tier, p("Dot"), got.Dot, want.Dot, gen)
	checkReduce2(t, tier, p("SumSqDiff"), got.SumSqDiff, want.SumSqDiff, gen)
	checkReduce2(t, tier, p("L1Diff"), got.L1Diff, want.L1Diff, gen)

	checkBinary(t, tier, p("SatAdd"), got.SatAdd, want.SatAdd, gen)
	checkBinary(t, tier, p("SatSub"), got.SatSub, want.SatSub, gen)
}

// checkIntegerOps runs checkOps over the six element types that were added
// after the original four, so that they are exercised by exactly the same
// battery rather than by a reduced one of their own.
func checkIntegerOps(t *testing.T, tier string, got, want kernel.Set) {
	t.Helper()
	checkOps(t, tier, "I8", got.I8, want.I8, genInt[int8])
	checkOps(t, tier, "I16", got.I16, want.I16, genInt[int16])
	checkOps(t, tier, "U8", got.U8, want.U8, genInt[uint8])
	checkOps(t, tier, "U16", got.U16, want.U16, genInt[uint16])
	checkOps(t, tier, "U32", got.U32, want.U32, genInt[uint32])
	checkOps(t, tier, "U64", got.U64, want.U64, genInt[uint64])
}

// TestEveryTierMatchesTheReference is the headline check: every kernel of
// every compiled-in tier produces bit-identical results to the portable
// implementation.
func TestEveryTierMatchesTheReference(t *testing.T) {
	want := ref.Set()
	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			checkOps(t, tier, "F32", got.F32, want.F32, genF32)
			checkOps(t, tier, "F64", got.F64, want.F64, genF64)
			checkOps(t, tier, "I32", got.I32, want.I32, genI32)
			checkOps(t, tier, "I64", got.I64, want.I64, genI64)
			checkIntegerOps(t, tier, got, want)
		})
	}
}

// TestTiersAgreeWithEachOther is the guarantee users actually rely on: a
// result does not change when the program moves to a machine with a different
// vector width. Comparing each tier to the reference implies it, but checking
// it directly is what the promise says.
func TestTiersAgreeWithEachOther(t *testing.T) {
	all := tiers(t)
	if len(all) < 2 {
		t.Skip("only one tier in this build")
	}
	names := make([]string, 0, len(all))
	for n := range all {
		names = append(names, n)
	}
	first := all[names[0]]
	for _, n := range names[1:] {
		t.Run(names[0]+" vs "+n, func(t *testing.T) {
			checkOps(t, n, "F32", all[n].F32, first.F32, genF32)
			checkOps(t, n, "F64", all[n].F64, first.F64, genF64)
			checkOps(t, n, "I32", all[n].I32, first.I32, genI32)
			checkOps(t, n, "I64", all[n].I64, first.I64, genI64)
			checkIntegerOps(t, n, all[n], first)
		})
	}
}

// TestNoNilKernels guards the property that makes partial backends safe: a
// tier is always a complete set, with the portable implementation standing in
// for anything not generated. A nil here is a crash waiting for whichever
// operation nobody got to.
func TestNoNilKernels(t *testing.T) {
	for tier, s := range tiers(t) {
		if s.F64.Add == nil || s.F64.Sum == nil || s.Bytes.IndexByte == nil ||
			s.Mask.All == nil || s.I32.Add == nil {
			t.Errorf("tier %s has nil kernels; backends must start from the reference", tier)
		}
		if s.Name != tier {
			t.Errorf("tier %s reports Name %q", tier, s.Name)
		}
	}
}

func TestReportSelection(t *testing.T) {
	t.Logf("%s", cpu.Describe())
	t.Logf("tiers with generated kernels: %v", backend.Tiers())
	for _, n := range backend.Tiers() {
		s, _ := backend.Lookup(n)
		t.Logf("  %-8s %s", n, describeCoverage(s))
	}
}

// describeCoverage counts how many kernels of a tier differ from the reference,
// which is how many were actually generated for it.
func describeCoverage(s kernel.Set) string {
	base := ref.Set()
	n := 0
	// Comparing function values is not allowed in Go, so compare the behaviour
	// that matters instead: a generated kernel is one whose pointer differs.
	// fmt.Sprintf on a func prints its address, which is enough here and is
	// only used for a log line.
	cmp := func(a, b any) {
		if fmt.Sprintf("%p", a) != fmt.Sprintf("%p", b) {
			n++
		}
	}
	cmp(s.F32.Add, base.F32.Add)
	cmp(s.F64.Add, base.F64.Add)
	cmp(s.I32.Add, base.I32.Add)
	cmp(s.I64.Add, base.I64.Add)
	cmp(s.F64.Sum, base.F64.Sum)
	cmp(s.F64.Sqrt, base.F64.Sqrt)
	return fmt.Sprintf("%d/6 sampled kernels generated", n)
}
