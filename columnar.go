package simd

import "math/bits"

// The columnar family: Arrow-style validity bitmaps driving typed value
// buffers. The layout is Arrow's -- one bit per row, LSB-first within a
// byte -- but nothing here imports arrow-go: the operations take raw
// slices and a []byte bitmap, so an arrow user hands over the buffers an
// Array already holds, and everyone else gets the same columnar toolkit
// on plain Go data. Take -- gather rows by index -- is [GatherInto],
// unchanged.

// ColumnarElem are the element types the columnar family covers: the four
// full-width types the compress instruction handles. The narrow integers
// would need AVX512_VBMI2, which is above the tier this library gates
// avx512 on.
type ColumnarElem interface {
	float32 | float64 | int32 | int64
}

// CompressBitsInto packs the elements of src whose validity bit is set
// down into dst and returns how many there were. It is the columnar
// filter: apply a predicate once into a bitmap, then compress each column
// by it.
//
// validity must hold at least (len(src)+7)/8 bytes; rows past 8*len(validity)
// are treated as absent. dst must have room for every element of src --
// the compress store is unconditional, which is the entire reason it is
// fast -- so a shorter dst takes the portable path.
func CompressBitsInto[T ColumnarElem](dst, src []T, validity []byte) int {
	return ops[T]().CompressBits(dst, src, validity)
}

// SumValid adds the elements whose validity bit is set and returns the
// total. The value under a clear bit is never read, so a null slot may
// hold anything, including NaN -- Arrow leaves it undefined and so does
// this.
//
// The accumulation order is [Sum]'s exactly: SumValid over a bitmap of
// ones is bit-identical to Sum, and every tier reproduces the same bits.
func SumValid[T ColumnarElem](a []T, validity []byte) T {
	return ops[T]().SumValid(a, validity)
}

// CountValid returns how many of the first n validity bits are set: the
// columnar non-null count. n may exceed 8*len(validity); the missing bits
// count as clear.
func CountValid(validity []byte, n int) int {
	if n <= 0 {
		return 0
	}
	if max := 8 * len(validity); n > max {
		n = max
	}
	c := PopCount(validity[:n/8])
	if rem := n % 8; rem != 0 {
		c += bits.OnesCount8(validity[n/8] & (1<<rem - 1))
	}
	return c
}
