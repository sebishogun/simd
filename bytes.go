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

// HammingDistance returns the number of bit positions at which a and b
// differ, over the shorter of the two.
//
// The name is spelled out because [Hamming] is already the Hamming *window*,
// a different thing from the same person: that one shapes a signal before an
// FFT, this one compares two bit vectors.
//
// This is the fused popcount(a^b). Both halves are separately available here
// as [XorInto] and [PopCount], and chaining them is the wrong way to do it:
// that needs a destination buffer the size of the input and three passes over
// memory where this makes one. At the sizes Hamming distance is used at —
// binary embedding search, LSH buckets, SimHash near-duplicate detection —
// the intermediate is most of the cost.
//
// The result is exact and identical on every instruction set. It allocates
// nothing.
func HammingDistance[S, T Text](a S, b T) int {
	return active.Bytes.Hamming(textBytes(a), textBytes(b))
}

// HammingDistanceWords is [HammingDistance] for a bit vector already stored
// as []uint64, which is the layout most binary-embedding indexes use. It
// gives the same answer as [HammingDistance] over the same bytes and saves
// the caller an allocating conversion.
func HammingDistanceWords(a, b []uint64) int { return active.Bytes.HammingWords(a, b) }

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

// ---------- colour ----------

// GrayscaleInto writes the BT.601 luma of three planar colour channels:
//
//	Y = 0.299 R + 0.587 G + 0.114 B
//
// The channels are separate slices — one per component — rather than
// interleaved RGBRGB. That is the layout a vector unit can use, and it is the
// same struct-of-arrays advice the tutorial gives for everything else here.
//
// The arithmetic is Q8 fixed point, so the result is exact and identical on
// every instruction set rather than carrying a floating-point error bound. It
// rounds to nearest; truncating would bias an image dark by half a level.
//
// It writes min of every argument's length and allocates nothing.
func GrayscaleInto(dst, r, g, b []byte) { active.Bytes.Grayscale(dst, r, g, b) }

// RGBToUVInto writes the two full-range (JFIF) chroma planes of three planar
// colour channels. The luma plane is [GrayscaleInto], which computes the same
// Y — so a full Y'CbCr conversion is the two calls, and a caller who wants
// luma alone makes one.
//
// They are separate rather than one call because seven arguments is one more
// than the SysV amd64 ABI passes in registers, and the fused form was declined
// by the generator on every amd64 tier.
//
// U and V are biased by 128 so they fit a byte, which is what every 8-bit
// full-range format does. Each chroma row of the matrix sums to zero, so a
// grey input gives exactly 128 in both planes — which is why a round trip does
// not tint a greyscale image.
//
// Like [GrayscaleInto] it is Q8 fixed point, exact, and allocation-free.
func RGBToUVInto(u, v, r, g, b []byte) { active.Bytes.RGBToUV(u, v, r, g, b) }
