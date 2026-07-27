package ref

import (
	"bytes"
	"unicode/utf8"
)

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

// index, lastIndex and countSeq are the substring operations the standard
// library defines exactly; see the note in ref.go on why these delegate.
func index(haystack, needle []byte) int     { return bytes.Index(haystack, needle) }
func lastIndex(haystack, needle []byte) int { return bytes.LastIndex(haystack, needle) }

// countSeq is bytes.Count for every separator but the empty one, which
// bytes.Count answers by counting runes. That is a UTF-8 question rather than
// a byte one; the wrapper in package simd answers it and never reaches here.
func countSeq(haystack, needle []byte) int {
	if len(needle) == 0 {
		return 0
	}
	return bytes.Count(haystack, needle)
}

// indexNotAny is the complement of indexAny: the first byte that is not in the
// set, or -1 if every byte is.
//
// An empty set contains nothing, so the first byte of a non-empty slice is
// already not in it. That is the opposite of indexAny's answer for the same
// input and both are right: "find one in the set" fails, "find one outside it"
// succeeds immediately.
func indexNotAny(b, chars []byte) int {
	if len(chars) == 0 {
		if len(b) == 0 {
			return -1
		}
		return 0
	}
	set := makeCharSet(chars)
	for i := range b {
		if !set.has(b[i]) {
			return i
		}
	}
	return -1
}

// lastIndexNotAny is indexNotAny from the other end.
//
// An empty set again contains nothing, so the last byte is already outside it;
// for an empty slice there is no such byte and the answer is -1, which the
// arithmetic gives without a special case.
func lastIndexNotAny(b, chars []byte) int {
	if len(chars) == 0 {
		return len(b) - 1
	}
	set := makeCharSet(chars)
	for i := len(b) - 1; i >= 0; i-- {
		if !set.has(b[i]) {
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

// Exported entry points for generated code.

func IndexByte(b []byte, c byte) int     { return indexByte(b, c) }
func LastIndexByte(b []byte, c byte) int { return lastIndexByte(b, c) }
func CountByte(b []byte, c byte) int     { return countByte(b, c) }
func EqualBytes(a, b []byte) bool        { return equalBytes(a, b) }
func PopCount(b []byte) int              { return popCount(b) }
func IsASCII(b []byte) bool              { return isASCII(b) }

func BitAnd(dst, a, b []byte)      { bitAnd(dst, a, b) }
func BitOr(dst, a, b []byte)       { bitOr(dst, a, b) }
func BitXor(dst, a, b []byte)      { bitXor(dst, a, b) }
func BitAndNot(dst, a, b []byte)   { bitAndNot(dst, a, b) }
func BitNot(dst, a []byte)         { bitNot(dst, a) }
func FillBytes(dst []byte, v byte) { fillBytes(dst, v) }

func ToUpperASCII(dst, b []byte)                { toUpperASCII(dst, b) }
func ToLowerASCII(dst, b []byte)                { toLowerASCII(dst, b) }
func ReplaceByte(dst, b []byte, old, with byte) { replaceByte(dst, b, old, with) }

func Index(haystack, needle []byte) int { return index(haystack, needle) }

func ValidUTF8(b []byte) bool { return validUTF8(b) }

func IndexNotAny(b, chars []byte) int       { return indexNotAny(b, chars) }
func LastIndex(haystack, needle []byte) int { return lastIndex(haystack, needle) }
func CountSeq(haystack, needle []byte) int  { return countSeq(haystack, needle) }

func LastIndexNotAny(b, chars []byte) int { return lastIndexNotAny(b, chars) }
