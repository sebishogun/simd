package ref

import "unicode/utf8"

// Text scanning kernels.
//
// These are the primitives a tokenizer is built from. They are the part of
// parsing that genuinely benefits from vectors — classifying a whole register
// of bytes per instruction — as opposed to the branch-heavy state machine that
// consumes their output.

// indexAll writes the offset of every occurrence of c into dst and returns how
// many it found, stopping if dst fills up.
//
// This is the structural-index step of a vectorized parser. A scalar
// implementation is a plain loop; the vector one compares a whole register
// against a broadcast c, turns the result into a bitmask, and walks the set
// bits, which is where the order-of-magnitude comes from.
func indexAll(dst []int32, b []byte, c byte) int {
	n := 0
	for i := range b {
		if b[i] == c {
			if n == len(dst) {
				return n
			}
			dst[n] = int32(i)
			n++
		}
	}
	return n
}

// charSet is a 256-bit membership table. Building it once and testing against
// it keeps indexAny linear in the input rather than in input times set size.
type charSet [4]uint64

func makeCharSet(chars []byte) charSet {
	var s charSet
	for _, c := range chars {
		s[c>>6] |= 1 << (c & 63)
	}
	return s
}

func (s *charSet) has(c byte) bool { return s[c>>6]&(1<<(c&63)) != 0 }

func indexAny(b, chars []byte) int {
	if len(chars) == 0 {
		return -1
	}
	set := makeCharSet(chars)
	for i := range b {
		if set.has(b[i]) {
			return i
		}
	}
	return -1
}

func countAny(b, chars []byte) int {
	if len(chars) == 0 {
		return 0
	}
	set := makeCharSet(chars)
	n := 0
	for i := range b {
		if set.has(b[i]) {
			n++
		}
	}
	return n
}

// index is substring search, matching bytes.Index.
//
// The vectorized form of this compares broadcasts of the needle's first and
// last bytes against the haystack, intersects the two masks, and only then
// verifies candidates — so most positions are rejected without a memcmp. The
// reference just needs to define the answer.
func index(haystack, needle []byte) int {
	m := len(needle)
	switch {
	case m == 0:
		return 0
	case m > len(haystack):
		return -1
	case m == 1:
		return indexByte(haystack, needle[0])
	}
	first, last := needle[0], needle[m-1]
	for i := 0; i+m <= len(haystack); i++ {
		if haystack[i] == first && haystack[i+m-1] == last &&
			equalBytes(haystack[i:i+m], needle) {
			return i
		}
	}
	return -1
}

func isASCII(b []byte) bool {
	for i := range b {
		if b[i] >= 0x80 {
			return false
		}
	}
	return true
}

// validUTF8 defers to the standard library, which already has a tuned
// implementation. A vector backend replaces it with the range-and-continuation
// table check that classifies a register at a time.
func validUTF8(b []byte) bool { return utf8.Valid(b) }

// ASCII case folding is a range compare and a constant flip, with no
// dependence on the byte's own value beyond the range test — which is exactly
// the shape a vector unit handles well. Bytes outside A-Z or a-z pass through,
// so this is safe on UTF-8: continuation bytes are all ≥ 0x80 and untouched.

func toUpperASCII(dst, b []byte) {
	n := min(len(dst), len(b))
	dst, b = dst[:n], b[:n]
	for i := range dst {
		c := b[i]
		if c >= 'a' && c <= 'z' {
			c -= 'a' - 'A'
		}
		dst[i] = c
	}
}

func toLowerASCII(dst, b []byte) {
	n := min(len(dst), len(b))
	dst, b = dst[:n], b[:n]
	for i := range dst {
		c := b[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		dst[i] = c
	}
}

func equalFoldASCII(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	b = b[:len(a)]
	for i := range a {
		x, y := a[i], b[i]
		if x >= 'A' && x <= 'Z' {
			x += 'a' - 'A'
		}
		if y >= 'A' && y <= 'Z' {
			y += 'a' - 'A'
		}
		if x != y {
			return false
		}
	}
	return true
}

func replaceByte(dst, b []byte, old, new byte) {
	n := min(len(dst), len(b))
	dst, b = dst[:n], b[:n]
	for i := range dst {
		if c := b[i]; c == old {
			dst[i] = new
		} else {
			dst[i] = c
		}
	}
}

const hexDigits = "0123456789abcdef"

// hexEncode writes two lowercase hex digits per input byte and returns how
// many bytes it wrote.
func hexEncode(dst, src []byte) int {
	n := min(len(dst)/2, len(src))
	for i := range n {
		v := src[i]
		dst[i*2] = hexDigits[v>>4]
		dst[i*2+1] = hexDigits[v&0x0f]
	}
	return n * 2
}

// hexDecode reads two hex digits per output byte. It returns the number of
// bytes written and whether the input was entirely valid; on invalid input it
// stops at the offending pair and reports false.
func hexDecode(dst, src []byte) (int, bool) {
	n := min(len(dst), len(src)/2)
	for i := range n {
		hi, ok1 := unhex(src[i*2])
		lo, ok2 := unhex(src[i*2+1])
		if !ok1 || !ok2 {
			return i, false
		}
		dst[i] = hi<<4 | lo
	}
	return n, len(src)%2 == 0
}

func unhex(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}
