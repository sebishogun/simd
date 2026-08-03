package ref

// The portable forms of the BLAS-shaped kernels. Every generated kernel is
// differential-tested against these, and they run wherever a kernel could not
// be generated or the input is below the dispatch threshold.

// RankOneFloat is a[i*n+j] += alpha*x[i]*y[j].
//
// The row scale is hoisted exactly as the kernel hoists it. Writing
// alpha*x[i]*y[j] in the inner loop would associate the multiplications
// differently and disagree with the kernel in the last place, which the
// differential suite would then report as a kernel bug.
func RankOneFloat[T Float](a, x, y []T, alpha T, m, n int) {
	if m < 0 || n < 0 || len(a) < m*n || len(x) < m || len(y) < n {
		return
	}
	for i := 0; i < m; i++ {
		s := alpha * x[i]
		row := a[i*n : i*n+n]
		for j := range row {
			// The conversion is load-bearing. Go fuses a multiply into an
			// adjacent add on arm64, riscv64, loong64 and ppc64 but not on
			// amd64, so without it this reference computes fma(s, y[j], row[j])
			// on four of the six architectures while the kernel — compiled with
			// -ffp-contract=off — always rounds the product first. They then
			// disagree in the last place, on those targets only. Third time
			// this trap has been paid for in this repository.
			row[j] += T(s * y[j])
		}
	}
}

// RotateFloat applies a Givens rotation to a pair of vectors, using the
// original x[i] in both assignments.
func RotateFloat[T Float](x, y []T, c, s T) {
	n := min(len(x), len(y))
	for i := 0; i < n; i++ {
		xi, yi := x[i], y[i]
		// Each product rounds before the add, for the same reason as RankOne.
		x[i] = T(c*xi) + T(s*yi)
		y[i] = T(c*yi) - T(s*xi)
	}
}

// SwapFloat exchanges two vectors.
func SwapFloat[T Float](x, y []T) {
	n := min(len(x), len(y))
	for i := 0; i < n; i++ {
		x[i], y[i] = y[i], x[i]
	}
}

// SwapInt is SwapFloat for the integer types.
func SwapInt[T Integer](x, y []T) {
	n := min(len(x), len(y))
	for i := 0; i < n; i++ {
		x[i], y[i] = y[i], x[i]
	}
}

// IndexAllAny writes the offset of every byte in b that equals any of the up to
// eight values packed into chars, one per byte, and returns how many it wrote.
//
// It stops when dst fills, matching the kernel: a caller who sizes dst for the
// expected number of matches and gets more must get a truncated answer rather
// than an overrun.
func IndexAllAny(dst []int32, b []byte, chars uint64) int {
	c := [8]byte{
		byte(chars), byte(chars >> 8), byte(chars >> 16), byte(chars >> 24),
		byte(chars >> 32), byte(chars >> 40), byte(chars >> 48), byte(chars >> 56),
	}
	k := 0
	for i := 0; i < len(b); i++ {
		x := b[i]
		if x != c[0] && x != c[1] && x != c[2] && x != c[3] &&
			x != c[4] && x != c[5] && x != c[6] && x != c[7] {
			continue
		}
		if k == len(dst) {
			break
		}
		dst[k] = int32(i)
		k++
	}
	return k
}

// The mask builders write one bit per byte of b into dst, least-significant bit
// first: bit i of dst[i/8] describes b[i]. dst must have room for
// (len(b)+7)/8 bytes, and trailing bits of the last byte are cleared, so the
// mask of a slice whose length is not a multiple of eight has no set bits past
// its end.
//
// These run below the dispatch threshold and on the targets where a kernel
// could not be generated, which for these operations is s390x and ppc64le.
// Below the threshold is not a rare case — a small document goes through them
// entirely — and the byte-at-a-time version they replace, a compare and a
// read-modify-write of dst per byte, was half the cost of parsing a 64-byte
// JSON document.
//
// Eight bytes at a time instead. The comparison is the standard has-a-zero-byte
// trick, which leaves 0x80 in every byte that matched, and the eight flags
// become eight bits by one multiply: for a word holding 0 or 1 in each byte,
// multiplying by 0x0102040810204080 accumulates byte k into bit k of the top
// byte, and no column can carry into the next.

const (
	maskLo     = 0x0101010101010101
	maskHi     = 0x8080808080808080
	maskGather = 0x0102040810204080
)

// le64 reads eight bytes as a little-endian word. The mask is defined
// least-significant bit first, so this is the byte order on every target.
func le64(b []byte) uint64 {
	_ = b[7]
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

// gatherFlags turns a word holding 0x80 in each matching byte into eight bits.
func gatherFlags(z uint64) byte { return byte((z >> 7) * maskGather >> 56) }

// eqZero returns 0x80 in each byte of x that is zero, and 0 elsewhere.
//
// Not the familiar (x-lo) &^ x & hi. That expression answers "is any byte
// zero", which is all it is usually asked, and its set bits are not reliably
// the zero bytes: the subtraction borrows across byte boundaries, so a zero
// byte can light up its neighbour. Here the answer is written into a mask, one
// bit per byte, and a bit in the wrong place is a wrong answer. Caught by
// GOSIMD=scalar in test-tiers, which is what that lane is for.
//
// This form cannot borrow: each byte of (x&lo7)+lo7 is at most 0xfe.
func eqZero(x uint64) uint64 {
	const lo7 = 0x7f7f7f7f7f7f7f7f
	return ^(((x & lo7) + lo7) | x | lo7)
}

// maskTail finishes the bytes past the last whole word, writing the partial
// byte with its unused bits clear.
func maskTail(dst, b []byte, i, w int, match func(byte) bool) {
	if i >= len(b) {
		return
	}
	var m byte
	for j := 0; i+j < len(b); j++ {
		if match(b[i+j]) {
			m |= 1 << uint(j)
		}
	}
	dst[w] = m
}

// MaskBits sets the bit where the byte equals c.
func MaskBits(dst, b []byte, c byte) {
	splat := uint64(maskLo) * uint64(c)
	i, w := 0, 0
	for ; i+8 <= len(b); i, w = i+8, w+1 {
		dst[w] = gatherFlags(eqZero(le64(b[i:]) ^ splat))
	}
	maskTail(dst, b, i, w, func(x byte) bool { return x == c })
}

// MaskBitsAny is [MaskBits] for a set of up to eight bytes packed one per byte
// of chars, the same encoding [IndexAllAny] takes.
func MaskBitsAny(dst, b []byte, chars uint64) {
	c := [8]byte{
		byte(chars), byte(chars >> 8), byte(chars >> 16), byte(chars >> 24),
		byte(chars >> 32), byte(chars >> 40), byte(chars >> 48), byte(chars >> 56),
	}
	var sp [8]uint64
	for k, ch := range c {
		sp[k] = uint64(maskLo) * uint64(ch)
	}
	i, w := 0, 0
	for ; i+8 <= len(b); i, w = i+8, w+1 {
		x := le64(b[i:])
		z := eqZero(x^sp[0]) | eqZero(x^sp[1]) | eqZero(x^sp[2]) | eqZero(x^sp[3]) |
			eqZero(x^sp[4]) | eqZero(x^sp[5]) | eqZero(x^sp[6]) | eqZero(x^sp[7])
		dst[w] = gatherFlags(z)
	}
	maskTail(dst, b, i, w, func(x byte) bool {
		return x == c[0] || x == c[1] || x == c[2] || x == c[3] ||
			x == c[4] || x == c[5] || x == c[6] || x == c[7]
	})
}

// MaskBitsLess is [MaskBits] for an inequality: the bit is set where the byte
// is below c.
func MaskBitsLess(dst, b []byte, c byte) {
	// The word form clears the high bit before subtracting, so a byte above
	// ASCII cannot borrow into its neighbour — which is only sound while c is
	// itself at or below 0x80. Above that the byte loop stands in; being right
	// matters more than being quick at a case no caller has.
	if c > 0x80 {
		for i := 0; i < len(b); i++ {
			if i%8 == 0 {
				dst[i/8] = 0
			}
			if b[i] < c {
				dst[i/8] |= 1 << (i % 8)
			}
		}
		return
	}
	splat := uint64(maskLo) * uint64(c)
	i, w := 0, 0
	for ; i+8 <= len(b); i, w = i+8, w+1 {
		x := le64(b[i:])
		// Setting the high bit of each byte before subtracting is what stops a
		// borrow leaving its byte: 0x80 + low7 - c cannot go negative while c
		// is at most 0x80. The high bit of the result is then clear exactly
		// where the byte is below c, and clearing it again wherever the
		// original byte was above ASCII finishes it -- such a byte is at least
		// 0x80 and so never below c.
		d := (x&^uint64(maskHi) | maskHi) - splat
		dst[w] = gatherFlags(^d &^ x & maskHi)
	}
	maskTail(dst, b, i, w, func(x byte) bool { return x < c })
}

// MaskBitsAny4 is [MaskBitsAny] for a set of at most four bytes, packed one per
// byte of a uint32.
func MaskBitsAny4(dst, b []byte, chars uint32) {
	c := [4]byte{byte(chars), byte(chars >> 8), byte(chars >> 16), byte(chars >> 24)}
	s0 := uint64(maskLo) * uint64(c[0])
	s1 := uint64(maskLo) * uint64(c[1])
	s2 := uint64(maskLo) * uint64(c[2])
	s3 := uint64(maskLo) * uint64(c[3])
	i, w := 0, 0
	for ; i+8 <= len(b); i, w = i+8, w+1 {
		x := le64(b[i:])
		dst[w] = gatherFlags(eqZero(x^s0) | eqZero(x^s1) | eqZero(x^s2) | eqZero(x^s3))
	}
	maskTail(dst, b, i, w, func(x byte) bool {
		return x == c[0] || x == c[1] || x == c[2] || x == c[3]
	})
}
