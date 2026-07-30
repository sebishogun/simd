package simd

// LEB128 varints: the widths, vectorized, and an encoder built on them.
//
// # What vectorizes and what does not
//
// Writing a varint stream is serial. Where value i lands depends on the width
// of every value before it, and that is a genuine loop-carried dependency
// through the output cursor — the same one compress.c describes, and no
// rewriting of the loop removes it.
//
// The question asked *first* does vectorize: how wide is each value. That is
// four unsigned comparisons per element and nothing else, and it is worth
// having on its own for two reasons.
//
// The first is allocation. [VarintSize] gives the exact encoded length of a
// whole slice in one vectorized pass, so an encoder sizes its buffer once
// instead of growing it — and growing a byte slice by append means copying
// everything written so far, repeatedly, which for a large column costs more
// than the encoding.
//
// The second is parallelism. The widths, prefix-summed, give every value its
// own offset, at which point the writes are independent and a caller that wants
// to split the work across goroutines can. [VarintLenInto] and [CumSumInto]
// together produce that table.
//
// # The widths are computed by comparison, not by counting leading zeros
//
// ceil(bits.Len(x)/7) is the obvious formula and it is what the reference
// computes. The kernel adds four unsigned comparisons instead, because
// bits.Len lowers to a leading-zero count that has no vector form on SSE2 or
// AVX2: LLVM scalarizes it lane by lane and gives back a kernel slower than
// the loop it replaced. Four compares and three adds cost the same everywhere.

// VarintValue is the pair of widths LEB128 is defined for here. Signed values
// are encoded by zigzagging first — see [ZigzagEncodeInt32Into] — which is what
// Protocol Buffers calls sint32 and sint64, and what keeps a small negative
// number one byte rather than ten.
type VarintValue interface{ uint32 | uint64 }

// VarintLenInto writes the LEB128 width of each element of a into dst: 1 to 5
// bytes for uint32, 1 to 10 for uint64.
//
// dst and a must be the same length; the shorter bounds the work.
//
// Prefix-summing the result with [CumSumInto] gives every value its offset in
// the encoded stream, which is what makes the writes independent.
func VarintLenInto[T VarintValue](dst []int32, a []T) {
	switch v := any(a).(type) {
	case []uint32:
		active.Convert.VarintLenU32(dst, v)
	case []uint64:
		active.Convert.VarintLenU64(dst, v)
	}
}

// VarintSize returns the total number of bytes the whole slice encodes to.
//
// One vectorized pass, no allocation. This is the number to pass to make when
// building the destination buffer: sized exactly, it is written once, where an
// append-and-grow encoder copies what it has already written every time the
// slice doubles.
func VarintSize[T VarintValue](a []T) int {
	switch v := any(a).(type) {
	case []uint32:
		return active.Convert.VarintSizeU32(v)
	case []uint64:
		return active.Convert.VarintSizeU64(v)
	}
	panic("simd: unreachable")
}

// AppendVarints encodes every element of a and appends the result to dst.
//
// The buffer is grown once, to exactly the size [VarintSize] reports, and the
// bytes are then written without any further bounds growth. The emission
// itself is the serial loop it has to be; what the vectorized pass buys is
// that it happens exactly once into exactly the right amount of memory.
func AppendVarints[T VarintValue](dst []byte, a []T) []byte {
	n := VarintSize(a)
	if cap(dst)-len(dst) < n {
		grown := make([]byte, len(dst), len(dst)+n)
		copy(grown, dst)
		dst = grown
	}
	off := len(dst)
	dst = dst[:off+n]
	buf := dst[off:]
	k := 0
	for _, v := range a {
		x := uint64(v)
		for x >= 0x80 {
			buf[k] = byte(x) | 0x80
			x >>= 7
			k++
		}
		buf[k] = byte(x)
		k++
	}
	return dst
}
