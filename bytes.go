package simd

import "bytes"

// This file holds the byte and bit operations. They take []byte directly
// rather than going through the generic Number path, because that is how
// callers already hold this data.
//
// The same convention applies as elsewhere: the plain name works in place on
// its first argument, the Into form takes a destination. Nothing here
// allocates.

// ---------- queries ----------

// IndexByte returns the index of the first occurrence of c in s, or -1.
//
// It matches bytes.IndexByte and strings.IndexByte and is a drop-in
// replacement for either.
func IndexByte[S Text](s S, c byte) int {
	b := textBytes(s)
	if len(b) < shortText {
		return bytes.IndexByte(b, c)
	}
	return active.Bytes.IndexByte(b, c)
}

// LastIndexByte returns the index of the last occurrence of c in s, or -1.
func LastIndexByte[S Text](s S, c byte) int {
	return active.Bytes.LastIndexByte(textBytes(s), c)
}

// Equal reports whether a and b are the same length and hold the same bytes.
//
// It matches bytes.Equal. A nil argument is equivalent to an empty slice.
func Equal[S, T Text](a S, b T) bool { return active.Bytes.Equal(textBytes(a), textBytes(b)) }

// Compare returns -1, 0 or +1 ordering a before, equal to, or after b,
// lexicographically by content and then by length.
//
// It matches bytes.Compare.
func Compare[S, T Text](a S, b T) int { return active.Bytes.Compare(textBytes(a), textBytes(b)) }

// PopCount returns the total number of set bits across every byte of b.
func PopCount[S Text](s S) int { return active.Bytes.PopCount(textBytes(s)) }

// ---------- in place ----------

// And clears in a every bit not set in b: a[i] &= b[i].
//
// It processes min(len(a), len(b)) bytes and allocates nothing.
// Use [AndInto] to write the result elsewhere.
func And(a, b []byte) { AndInto(a, a, b) }

// Or sets in a every bit set in b: a[i] |= b[i].
//
// It processes min(len(a), len(b)) bytes and allocates nothing.
// Use [OrInto] to write the result elsewhere.
func Or(a, b []byte) { OrInto(a, a, b) }

// Xor flips in a every bit set in b: a[i] ^= b[i].
//
// It processes min(len(a), len(b)) bytes and allocates nothing.
// Use [XorInto] to write the result elsewhere.
func Xor(a, b []byte) { XorInto(a, a, b) }

// AndNot clears in a every bit set in b: a[i] &^= b[i].
//
// It processes min(len(a), len(b)) bytes and allocates nothing.
// Use [AndNotInto] to write the result elsewhere.
func AndNot(a, b []byte) { AndNotInto(a, a, b) }

// ---------- with a destination ----------

// AndInto sets dst[i] = a[i] & b[i].
//
// It processes min(len(dst), len(a), len(b)) bytes. dst may alias a or b.
func AndInto(dst, a, b []byte) { active.Bytes.And(dst, a, b) }

// OrInto sets dst[i] = a[i] | b[i].
//
// It processes min(len(dst), len(a), len(b)) bytes. dst may alias a or b.
func OrInto(dst, a, b []byte) { active.Bytes.Or(dst, a, b) }

// XorInto sets dst[i] = a[i] ^ b[i].
//
// It processes min(len(dst), len(a), len(b)) bytes. dst may alias a or b.
func XorInto(dst, a, b []byte) { active.Bytes.Xor(dst, a, b) }

// AndNotInto sets dst[i] = a[i] &^ b[i], clearing in a every bit set in b.
//
// It processes min(len(dst), len(a), len(b)) bytes. dst may alias a or b.
func AndNotInto(dst, a, b []byte) { active.Bytes.AndNot(dst, a, b) }
