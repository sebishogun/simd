package conformance

// What the narrow and unsigned integer kernels are worth, measured the same
// way as everything else here: the minimum over many repetitions, against the
// portable implementation of the identical operation.
//
// The question is not whether these are faster — every one of them is a whole
// vector register against one element per iteration — but by how much, and
// where the answer stops improving. A uint8 kernel puts thirty-two elements in
// an AVX2 register against four for float64, so at a length that fits in L1
// the ratio should be about eight times better; past L2 it should collapse to
// the same number for every type, because the bottleneck stops being
// instructions and becomes bandwidth. Both are visible in the output and
// neither is worth guessing at.

import (
	"testing"
	"time"

	"github.com/sebishogun/simd/internal/cpu"
	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/perf"
	"github.com/sebishogun/simd/internal/ref"
)

// intPerfRow is one kernel of one element type at one length.
type intPerfRow struct {
	name       string
	bytes      int64
	got, want  func()
	skipReason string
}

// intPerfCases builds the rows for one element type. The width is the element
// size, so the throughput figures are comparable across types rather than
// counting elements.
func intPerfCases[T comparable](name string, width int, n int,
	got, want kernel.Ops[T], gen func(int) []T) []intPerfRow {

	a, b := gen(n), gen(n)
	dst := make([]T, n)
	mask := make([]bool, n)
	e := int64(n * width)

	pair := func(op string, bytes int64, g, w func()) intPerfRow {
		return intPerfRow{name: name + "." + op, bytes: bytes, got: g, want: w}
	}

	rows := []intPerfRow{
		pair("Add", e*3, func() { got.Add(dst, a, b) }, func() { want.Add(dst, a, b) }),
		pair("Mul", e*3, func() { got.Mul(dst, a, b) }, func() { want.Mul(dst, a, b) }),
		pair("Minimum", e*3, func() { got.Minimum(dst, a, b) }, func() { want.Minimum(dst, a, b) }),
		pair("Sum", e, func() { sinkAny = got.Sum(a) }, func() { sinkAny = want.Sum(a) }),
		pair("Max", e, func() { sinkAny = got.Max(a) }, func() { sinkAny = want.Max(a) }),
		pair("LessMask", e*2+int64(n),
			func() { got.LessMask(mask, a, b) },
			func() { want.LessMask(mask, a, b) }),
	}
	if got.SatAdd != nil && want.SatAdd != nil {
		rows = append(rows,
			pair("SatAdd", e*3, func() { got.SatAdd(dst, a, b) }, func() { want.SatAdd(dst, a, b) }),
			pair("SatSub", e*3, func() { got.SatSub(dst, a, b) }, func() { want.SatSub(dst, a, b) }))
	}
	return rows
}

// sinkAny keeps a reduction's result live without needing one variable per
// element type. Assigning to an interface would allocate, which would be
// measured; assigning to a blank-typed any of a non-pointer value under -gcflags
// still boxes, so this is deliberately written as a generic function whose
// argument escapes nowhere.
var sinkAny any

func seqOf[T comparable](make0 func(int) []T) func(int) []T { return make0 }

func TestPerfIntegerTypes(t *testing.T) {
	if testing.Short() {
		t.Skip("repetition testing needs time")
	}
	opt := perf.DefaultOptions()
	opt.Patience = 40 * time.Millisecond

	want := ref.Set()
	for tier, got := range tiers(t) {
		t.Logf("\n=== %s (host: %s) ===", tier, cpu.Describe())
		for _, n := range perfLens {
			t.Logf("\n  n=%d", n)
			var rows []intPerfRow
			rows = append(rows, intPerfCases("i8", 1, n, got.I8, want.I8, seqOf(seqI8))...)
			rows = append(rows, intPerfCases("u8", 1, n, got.U8, want.U8, seqOf(seqU8))...)
			rows = append(rows, intPerfCases("i16", 2, n, got.I16, want.I16, seqOf(seqI16))...)
			rows = append(rows, intPerfCases("u16", 2, n, got.U16, want.U16, seqOf(seqU16))...)
			rows = append(rows, intPerfCases("i32", 4, n, got.I32, want.I32, seqOf(seqI32))...)
			rows = append(rows, intPerfCases("u32", 4, n, got.U32, want.U32, seqOf(seqU32))...)
			rows = append(rows, intPerfCases("u64", 8, n, got.U64, want.U64, seqOf(seqU64))...)
			for _, row := range rows {
				if row.got == nil || row.want == nil {
					continue
				}
				a := perf.Measure("", row.bytes, row.got, opt)
				b := perf.Measure("", row.bytes, row.want, opt)
				t.Logf("    %-16s asm %9.1f ns (%6.2f GB/s)  portable %9.1f ns  %5.2fx",
					row.name, a.Min, a.GBPerSec(), b.Min, b.Min/a.Min)
			}
		}
	}
}

// The generators are per type rather than generic because a generic one over
// a union constraint cannot be passed where a concrete func is wanted without
// an instantiation per type anyway, and these read better.
func seqI8(n int) []int8 {
	s := make([]int8, n)
	for i := range s {
		s[i] = int8(i%251 - 125)
	}
	return s
}

func seqU8(n int) []uint8 {
	s := make([]uint8, n)
	for i := range s {
		s[i] = uint8(i % 251)
	}
	return s
}

func seqI16(n int) []int16 {
	s := make([]int16, n)
	for i := range s {
		s[i] = int16(i%65521 - 32760)
	}
	return s
}

func seqU16(n int) []uint16 {
	s := make([]uint16, n)
	for i := range s {
		s[i] = uint16(i % 65521)
	}
	return s
}

func seqI32(n int) []int32 {
	s := make([]int32, n)
	for i := range s {
		s[i] = int32(i) - 1000
	}
	return s
}

func seqU32(n int) []uint32 {
	s := make([]uint32, n)
	for i := range s {
		s[i] = uint32(i)
	}
	return s
}

func seqU64(n int) []uint64 {
	s := make([]uint64, n)
	for i := range s {
		s[i] = uint64(i)
	}
	return s
}
