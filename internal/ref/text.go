package ref

import (
	"bytes"
	"strconv"
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

// The UTF-16 fast path. These three define the semantics the kernels must
// match: an offset that is len(b) when nothing matched — not -1, because the
// caller uses it as a run length — and a widen and narrow that are only ever
// asked to move ASCII, so the narrow's truncation is exact by precondition.

func IndexNonASCII(b []byte) int {
	for i, c := range b {
		if c >= 0x80 {
			return i
		}
	}
	return len(b)
}

func IndexNonASCII16(b []uint16) int {
	for i, c := range b {
		if c >= 0x80 {
			return i
		}
	}
	return len(b)
}

func WidenU8U16(dst []uint16, s []byte) {
	for i := range dst {
		dst[i] = uint16(s[i])
	}
}

func NarrowU16U8(dst []byte, s []uint16) {
	for i := range dst {
		dst[i] = byte(s[i])
	}
}

func IndexNotAny(b, chars []byte) int       { return indexNotAny(b, chars) }
func LastIndex(haystack, needle []byte) int { return lastIndex(haystack, needle) }
func CountSeq(haystack, needle []byte) int  { return countSeq(haystack, needle) }

func LastIndexNotAny(b, chars []byte) int { return lastIndexNotAny(b, chars) }

// ---------- base64 ----------
//
// RFC 4648 with padding, matching encoding/base64.StdEncoding.
//
// Unlike the byte scanners above these do not delegate to the standard
// library, because the contract differs in a way that matters: these write
// into a caller's buffer and report -1 rather than allocating or panicking.
// Building the standard library's answer and copying it would allocate, which
// this package does not do.

const b64Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

func b64Encode(dst, src []byte) int {
	full := len(src) / 3
	rem := len(src) - full*3
	need := (full + boolToInt(rem != 0)) * 4
	if len(dst) < need {
		return -1
	}
	for i := range full {
		x, y, z := src[i*3], src[i*3+1], src[i*3+2]
		dst[i*4] = b64Alphabet[x>>2]
		dst[i*4+1] = b64Alphabet[(x&0x03)<<4|y>>4]
		dst[i*4+2] = b64Alphabet[(y&0x0f)<<2|z>>6]
		dst[i*4+3] = b64Alphabet[z&0x3f]
	}
	if rem != 0 {
		x := src[full*3]
		var y byte
		if rem == 2 {
			y = src[full*3+1]
		}
		dst[full*4] = b64Alphabet[x>>2]
		dst[full*4+1] = b64Alphabet[(x&0x03)<<4|y>>4]
		dst[full*4+2] = '='
		if rem == 2 {
			dst[full*4+2] = b64Alphabet[(y&0x0f)<<2]
		}
		dst[full*4+3] = '='
	}
	return need
}

// b64Value is a character's six-bit value, or 64 for anything outside the
// alphabet. The kernel uses the same encoding of "invalid" so that a whole
// register's worth can be validated with one OR; see csrc/bytes.c.
func b64Value(c byte) byte {
	switch {
	case c >= 'A' && c <= 'Z':
		return c - 'A'
	case c >= 'a' && c <= 'z':
		return c - 'a' + 26
	case c >= '0' && c <= '9':
		return c - '0' + 52
	case c == '+':
		return 62
	case c == '/':
		return 63
	}
	return 64
}

func b64Decode(dst, src []byte) int {
	if len(src)%4 != 0 {
		return -1
	}
	if len(src) == 0 {
		return 0
	}
	pad := 0
	if src[len(src)-1] == '=' {
		pad++
	}
	if len(src) >= 2 && src[len(src)-2] == '=' {
		pad++
	}
	need := len(src)/4*3 - pad
	if len(dst) < need {
		return -1
	}
	groups := len(src)/4 - 1
	var bad byte
	for i := range groups {
		a0, a1 := b64Value(src[i*4]), b64Value(src[i*4+1])
		a2, a3 := b64Value(src[i*4+2]), b64Value(src[i*4+3])
		bad |= a0 | a1 | a2 | a3
		dst[i*3] = a0<<2 | a1>>4
		dst[i*3+1] = a1<<4 | a2>>2
		dst[i*3+2] = a2<<6 | a3
	}
	if bad&0x40 != 0 {
		return -1
	}
	i := groups
	a0, a1 := b64Value(src[i*4]), b64Value(src[i*4+1])
	var a2, a3 byte
	if pad < 2 {
		a2 = b64Value(src[i*4+2])
	}
	if pad < 1 {
		a3 = b64Value(src[i*4+3])
	}
	if (a0|a1|a2|a3)&0x40 != 0 {
		return -1
	}
	dst[i*3] = a0<<2 | a1>>4
	if pad < 2 {
		dst[i*3+1] = a1<<4 | a2>>2
	}
	if pad < 1 {
		dst[i*3+2] = a2<<6 | a3
	}
	return need
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// HexDecode is exported for the generated dispatch tables, which name the
// reference by its exported identifier. The two results are what kept this
// portable before the generator could return a pair.
func HexDecode(dst, src []byte) (int, bool) { return hexDecode(dst, src) }

// ParseInts is the reference for the integer field parser. Fields are
// src[start:idx[k]] with start one past the previous separator, which is the
// shape IndexAll produces.
func ParseInts(dst []int64, src []byte, idx []int32) (int, bool) {
	n := min(len(dst), len(idx))
	start := 0
	for k := range n {
		end := int(idx[k])
		if end > len(src) || end < start {
			return k, false
		}
		f := src[start:end]
		start = end + 1
		neg := false
		if len(f) > 0 && (f[0] == '-' || f[0] == '+') {
			neg = f[0] == '-'
			f = f[1:]
		}
		if len(f) == 0 || len(f) > 19 {
			return k, false
		}
		var acc uint64
		for _, c := range f {
			d := c - '0'
			if d > 9 {
				return k, false
			}
			acc = acc*10 + uint64(d)
		}
		limit := uint64(1<<63 - 1)
		if neg {
			limit = 1 << 63
		}
		if acc > limit {
			return k, false
		}
		if neg {
			dst[k] = int64(-acc)
		} else {
			dst[k] = int64(acc)
		}
	}
	return n, true
}

// ParseUints is ParseInts over the full uint64 range and with no sign.
//
// A leading '+' is rejected rather than skipped, matching strconv.ParseUint
// with bitSize 64, which accepts no sign at all.
func ParseUints(dst []uint64, src []byte, idx []int32) (int, bool) {
	n := min(len(dst), len(idx))
	start := 0
	for k := range n {
		end := int(idx[k])
		if end > len(src) || end < start {
			return k, false
		}
		f := src[start:end]
		start = end + 1
		if len(f) == 0 || len(f) > 20 {
			return k, false
		}
		var acc uint64
		for _, c := range f {
			d := c - '0'
			if d > 9 {
				return k, false
			}
			// Horner here, unlike the kernel, because the reference is written
			// for obviousness and the overflow test is easier to see: acc
			// exceeding maxUint64/10 before the multiply, or the add carrying.
			if acc > (1<<64-1)/10 {
				return k, false
			}
			next := acc*10 + uint64(d)
			if next < acc {
				return k, false
			}
			acc = next
		}
		dst[k] = acc
	}
	return n, true
}

// FormatInts is the reference formatter: exact-fit, so it succeeds wherever
// success is possible, which is what makes it the safe fallback for short
// destinations.
func FormatInts(dst []byte, vals []int64, sep byte) int {
	w := 0
	var buf [20]byte
	for i, v := range vals {
		s := strconv.AppendInt(buf[:0], v, 10)
		if len(dst)-w < len(s)+1 {
			// room for the digits and, except at the end, the separator
			need := len(s)
			if i != len(vals)-1 {
				need++
			}
			if len(dst)-w < need {
				return -1
			}
		}
		w += copy(dst[w:], s)
		if i != len(vals)-1 {
			dst[w] = sep
			w++
		}
	}
	return w
}

func B64Encode(dst, src []byte) int { return b64Encode(dst, src) }
func B64Decode(dst, src []byte) int { return b64Decode(dst, src) }
