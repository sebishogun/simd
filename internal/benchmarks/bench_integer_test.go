package benchmarks

// Benchmarks for the narrow and unsigned integer types, and for the two
// saturating operations.
//
// These are separate from bench_test.go because the question they answer is
// different. That file asks what the assembly is worth for float64, the type
// everything else is compared against. This one asks whether the narrow types
// pay for themselves at all, and the answer is not obvious in advance: a
// uint8 kernel moves an eighth of the bytes per element that a float64 kernel
// does, so the same slice length is an eighth of the work and the fixed call
// cost is eight times as significant — but a single register holds thirty-two
// of them on AVX2 rather than four, so where the vector body does dominate it
// dominates by more.
//
// Run against the portable build to get the ratio that matters:
//
//	go test -run '^$' -bench 'Int|Sat' -count 10 > asm.txt
//	go test -run '^$' -bench 'Int|Sat' -count 10 -tags purego > pure.txt
//	benchstat pure.txt asm.txt
//
// SetBytes is the bytes of *input* touched per iteration, so the throughput
// figure is comparable across element widths.

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/sebishogun/simd"
)

var intSizes = []int{16, 64, 256, 4096, 65536}

func fillSeq[T simd.Number](n int) []T {
	s := make([]T, n)
	for i := range s {
		s[i] = T(i%251 + 1)
	}
	return s
}

// benchBinary is one elementwise kernel over one element type, swept over the
// sizes. The name carries the element width so a reader of the output can see
// the per-byte cost line up across types.
func benchBinary[T simd.Number](b *testing.B, name string, op func(dst, a, c []T)) {
	var zero T
	width := int(unsafe.Sizeof(zero))
	for _, n := range intSizes {
		x, y, dst := fillSeq[T](n), fillSeq[T](n), make([]T, n)
		b.Run(fmt.Sprintf("%s/n=%d", name, n), func(b *testing.B) {
			b.SetBytes(int64(n * width))
			for b.Loop() {
				op(dst, x, y)
			}
		})
	}
}

func BenchmarkAddInt(b *testing.B) {
	benchBinary(b, "i8", simd.AddInto[int8])
	benchBinary(b, "i16", simd.AddInto[int16])
	benchBinary(b, "i32", simd.AddInto[int32])
	benchBinary(b, "i64", simd.AddInto[int64])
	benchBinary(b, "u8", simd.AddInto[uint8])
	benchBinary(b, "u16", simd.AddInto[uint16])
	benchBinary(b, "u32", simd.AddInto[uint32])
	benchBinary(b, "u64", simd.AddInto[uint64])
}

func BenchmarkMulInt(b *testing.B) {
	benchBinary(b, "i8", simd.MulInto[int8])
	benchBinary(b, "i16", simd.MulInto[int16])
	benchBinary(b, "u8", simd.MulInto[uint8])
	benchBinary(b, "u32", simd.MulInto[uint32])
}

func BenchmarkMinimumInt(b *testing.B) {
	benchBinary(b, "i8", simd.MinimumInto[int8])
	benchBinary(b, "u8", simd.MinimumInto[uint8])
	benchBinary(b, "u16", simd.MinimumInto[uint16])
	benchBinary(b, "u32", simd.MinimumInto[uint32])
}

// BenchmarkSatAdd and BenchmarkSatSub are the ones worth watching. The kernel
// is a single instruction per lane where the portable path is a widening add,
// two compares and a narrow, so the ratio here should be the largest in the
// suite for any type at all.
func BenchmarkSatAdd(b *testing.B) {
	benchBinary(b, "i8", simd.SatAddInto[int8])
	benchBinary(b, "i16", simd.SatAddInto[int16])
	benchBinary(b, "i32", simd.SatAddInto[int32])
	benchBinary(b, "u8", simd.SatAddInto[uint8])
	benchBinary(b, "u16", simd.SatAddInto[uint16])
	benchBinary(b, "u32", simd.SatAddInto[uint32])
}

func BenchmarkSatSub(b *testing.B) {
	benchBinary(b, "i8", simd.SatSubInto[int8])
	benchBinary(b, "u8", simd.SatSubInto[uint8])
	benchBinary(b, "u16", simd.SatSubInto[uint16])
}

func BenchmarkSumInt(b *testing.B) {
	for _, tc := range []struct {
		name string
		run  func(n int) func()
	}{
		{"i8", func(n int) func() { x := fillSeq[int8](n); return func() { sinkI8 = simd.Sum(x) } }},
		{"i16", func(n int) func() { x := fillSeq[int16](n); return func() { sinkI16 = simd.Sum(x) } }},
		{"i32", func(n int) func() { x := fillSeq[int32](n); return func() { sinkI32 = simd.Sum(x) } }},
		{"u8", func(n int) func() { x := fillSeq[uint8](n); return func() { sinkU8 = simd.Sum(x) } }},
		{"u32", func(n int) func() { x := fillSeq[uint32](n); return func() { sinkU32 = simd.Sum(x) } }},
		{"u64", func(n int) func() { x := fillSeq[uint64](n); return func() { sinkU64 = simd.Sum(x) } }},
	} {
		for _, n := range intSizes {
			f := tc.run(n)
			b.Run(fmt.Sprintf("%s/n=%d", tc.name, n), func(b *testing.B) {
				for b.Loop() {
					f()
				}
			})
		}
	}
}

func BenchmarkMinReduceInt(b *testing.B) {
	for _, tc := range []struct {
		name string
		run  func(n int) func()
	}{
		{"i8", func(n int) func() { x := fillSeq[int8](n); return func() { sinkI8 = simd.Min(x) } }},
		{"u8", func(n int) func() { x := fillSeq[uint8](n); return func() { sinkU8 = simd.Min(x) } }},
		{"u32", func(n int) func() { x := fillSeq[uint32](n); return func() { sinkU32 = simd.Min(x) } }},
	} {
		for _, n := range intSizes {
			f := tc.run(n)
			b.Run(fmt.Sprintf("%s/n=%d", tc.name, n), func(b *testing.B) {
				for b.Loop() {
					f()
				}
			})
		}
	}
}

// BenchmarkLessMaskInt exercises the comparison path, which is where an
// unsigned type differs from a signed one in the instruction selected rather
// than merely in the width.
func BenchmarkLessMaskInt(b *testing.B) {
	for _, n := range intSizes {
		x, y := fillSeq[uint16](n), fillSeq[uint16](n)
		m := make([]bool, n)
		b.Run(fmt.Sprintf("u16/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 2))
			for b.Loop() {
				simd.LessInto(m, x, y)
			}
		})
	}
}

var (
	sinkI8  int8
	sinkI16 int16
	sinkI32 int32
	sinkU8  uint8
	sinkU32 uint32
	sinkU64 uint64
)
