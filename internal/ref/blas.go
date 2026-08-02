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

// MaskBits writes one bit per byte of b into dst, set where the byte equals c,
// least-significant bit first: bit i of dst[i/8] describes b[i].
//
// dst must have room for (len(b)+7)/8 bytes. Trailing bits of the last byte
// are cleared, so the mask of a slice whose length is not a multiple of eight
// has no set bits past its end.
func MaskBits(dst, b []byte, c byte) {
	for i := 0; i < len(b); i++ {
		if i%8 == 0 {
			dst[i/8] = 0
		}
		if b[i] == c {
			dst[i/8] |= 1 << (i % 8)
		}
	}
}

// MaskBitsAny is [MaskBits] for a set of up to eight bytes packed one per byte
// of chars, the same encoding [IndexAllAny] takes.
func MaskBitsAny(dst, b []byte, chars uint64) {
	c := [8]byte{
		byte(chars), byte(chars >> 8), byte(chars >> 16), byte(chars >> 24),
		byte(chars >> 32), byte(chars >> 40), byte(chars >> 48), byte(chars >> 56),
	}
	for i := 0; i < len(b); i++ {
		if i%8 == 0 {
			dst[i/8] = 0
		}
		x := b[i]
		if x == c[0] || x == c[1] || x == c[2] || x == c[3] ||
			x == c[4] || x == c[5] || x == c[6] || x == c[7] {
			dst[i/8] |= 1 << (i % 8)
		}
	}
}

// MaskBitsLess is [MaskBits] for an inequality: the bit is set where the byte
// is below c.
func MaskBitsLess(dst, b []byte, c byte) {
	for i := 0; i < len(b); i++ {
		if i%8 == 0 {
			dst[i/8] = 0
		}
		if b[i] < c {
			dst[i/8] |= 1 << (i % 8)
		}
	}
}

// MaskBitsAny4 is [MaskBitsAny] for a set of at most four bytes, packed one per
// byte of a uint32.
func MaskBitsAny4(dst, b []byte, chars uint32) {
	c := [4]byte{byte(chars), byte(chars >> 8), byte(chars >> 16), byte(chars >> 24)}
	for i := 0; i < len(b); i++ {
		if i%8 == 0 {
			dst[i/8] = 0
		}
		x := b[i]
		if x == c[0] || x == c[1] || x == c[2] || x == c[3] {
			dst[i/8] |= 1 << (i % 8)
		}
	}
}
