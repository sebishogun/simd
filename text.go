package simd

// Text and byte-scanning helpers.
//
// These are the operations a tokenizer, parser or log processor spends its
// time in, and the part of that work a vector unit actually helps with: a
// whole register of bytes classified per instruction. The branch-heavy state
// machine that consumes their output does not vectorize, and is not here.
//
// Everything takes []byte. For a string, convert without copying at the call
// site if you are on Go 1.20 or later:
//
//	simd.IndexAny([]byte(s), delims)   // copies
//	simd.IndexAny(unsafe.Slice(unsafe.StringData(s), len(s)), delims)  // does not
//
// Nothing here allocates.

// IndexAll writes the offset of every occurrence of c in b into dst, and
// returns how many it found. It stops early if dst fills up, so a short dst
// bounds the work rather than being an error.
//
// This is the structural-index step of a vectorized parser: run it once for
// each delimiter class you care about and you have the shape of the document
// before you have looked at a single byte twice.
//
//	offsets := make([]int32, 1024)
//	n := simd.IndexAll(offsets, line, ',')
//	for _, off := range offsets[:n] { ... }
func IndexAll(dst []int32, b []byte, c byte) int { return active.Bytes.IndexAll(dst, b, c) }

// IndexAny returns the index of the first byte in b that is also in chars, or
// -1. It matches bytes.IndexAny for single-byte needles.
//
// The set is turned into a 256-bit table once, so cost is linear in b and
// independent of how many characters you are looking for.
func IndexAny(b, chars []byte) int { return active.Bytes.IndexAny(b, chars) }

// CountAny returns how many bytes of b are in chars.
func CountAny(b, chars []byte) int { return active.Bytes.CountAny(b, chars) }

// Index returns the index of the first occurrence of needle in haystack, or
// -1. An empty needle returns 0.
//
// It matches bytes.Index and is a drop-in replacement.
func Index(haystack, needle []byte) int { return active.Bytes.Index(haystack, needle) }

// IsASCII reports whether every byte is below 0x80.
//
// It is worth checking before text processing, because the ASCII path of most
// algorithms is dramatically simpler than the general one.
func IsASCII(b []byte) bool { return active.Bytes.IsASCII(b) }

// ValidUTF8 reports whether b is entirely well-formed UTF-8.
//
// It matches utf8.Valid.
func ValidUTF8(b []byte) bool { return active.Bytes.ValidUTF8(b) }

// ToUpperASCII maps a-z to A-Z in place, leaving every other byte alone.
//
// Only ASCII is folded, which makes this safe to run over UTF-8: continuation
// bytes are all 0x80 or above and are untouched. For full Unicode folding use
// the strings package; that is not a vectorizable operation.
func ToUpperASCII(b []byte) { active.Bytes.ToUpperASCII(b, b) }

// ToLowerASCII maps A-Z to a-z in place, leaving every other byte alone.
// See [ToUpperASCII] on why this is UTF-8 safe.
func ToLowerASCII(b []byte) { active.Bytes.ToLowerASCII(b, b) }

// ToUpperASCIIInto writes the ASCII-uppercased b into dst. dst may alias b.
func ToUpperASCIIInto(dst, b []byte) { active.Bytes.ToUpperASCII(dst, b) }

// ToLowerASCIIInto writes the ASCII-lowercased b into dst. dst may alias b.
func ToLowerASCIIInto(dst, b []byte) { active.Bytes.ToLowerASCII(dst, b) }

// EqualFoldASCII reports whether a and b are equal ignoring ASCII case.
//
// Unlike bytes.EqualFold it does not perform Unicode case folding, which makes
// it both faster and wrong for non-ASCII input — use it for protocol tokens
// (HTTP headers, keywords, hex digits), not for user-facing text.
func EqualFoldASCII(a, b []byte) bool { return active.Bytes.EqualFoldASCII(a, b) }

// ReplaceByte replaces every occurrence of old with new, in place.
func ReplaceByte(b []byte, old, new byte) { active.Bytes.ReplaceByte(b, b, old, new) }

// ReplaceByteInto writes b into dst with every old replaced by new.
// dst may alias b.
func ReplaceByteInto(dst, b []byte, old, new byte) { active.Bytes.ReplaceByte(dst, b, old, new) }

// HexEncode writes the lowercase hexadecimal encoding of src into dst and
// returns the number of bytes written. dst needs room for 2*len(src).
//
// It matches encoding/hex.Encode.
func HexEncode(dst, src []byte) int { return active.Bytes.HexEncode(dst, src) }

// HexDecode decodes hexadecimal from src into dst, returning the number of
// bytes written and whether the whole input was valid. Both upper and lower
// case digits are accepted.
//
// On a bad digit it stops there and reports false, with the bytes decoded so
// far already written. An odd-length input also reports false.
func HexDecode(dst, src []byte) (int, bool) { return active.Bytes.HexDecode(dst, src) }
