package simd

// Text and byte-scanning helpers.
//
// These are the operations a tokenizer, parser or log processor spends its
// time in, and the part of that work a vector unit actually helps with: a
// whole register of bytes classified per instruction. The branch-heavy state
// machine that consumes their output does not vectorize, and is not here.
//
// Everything that only reads its input takes [Text], so a string works as
// directly as a []byte:
//
//	simd.Index(line, ",")                 // string
//	simd.Index(buf, []byte(","))          // []byte
//
// The string form does not copy. It also does not allocate: the type switch
// that distinguishes the two folds away at instantiation, and there is a test
// that says so. Functions that *write* still take a []byte destination, for
// the obvious reason.
//
// Nothing here allocates.

import (
	"bytes"
	"unicode/utf8"
	"unsafe"
)

// Text is the constraint for input a scanning function only reads.
//
// The two types are listed exactly rather than as approximations, the same
// choice [Number] makes: a defined type such as `type Token string` is not
// accepted, because supporting it would mean reinterpreting the value and this
// package does not do that behind a caller's back.
type Text interface {
	string | []byte
}

// textBytes views a Text as bytes without copying.
//
// The result aliases the caller's memory and, for the string case, aliases
// memory the language guarantees is immutable. Every caller below only reads
// it. Nothing here may be handed to a kernel that writes through it, which is
// why the writing functions take a []byte destination separately rather than
// being generic in both.
func textBytes[S Text](s S) []byte {
	if b, ok := any(s).([]byte); ok {
		return b
	}
	str := any(s).(string)
	return unsafe.Slice(unsafe.StringData(str), len(str))
}

// textSlice reslices a Text and gives the result back as the same type, which
// the language will not do directly: string and []byte share no underlying
// type, so a slice expression on the type parameter is not valid.
func textSlice[S Text](s S, lo, hi int) S {
	if b, ok := any(s).([]byte); ok {
		return any(b[lo:hi]).(S)
	}
	return any(any(s).(string)[lo:hi]).(S)
}

// shortText is the length below which the scanning functions call the standard
// library directly instead of dispatching.
//
// Below every kernel's threshold the dispatcher already ends up in `bytes` —
// the reference delegates there for the functions the two define identically —
// so this changes no answer. What it removes is the journey: an indirect call
// through the backend's function table, which the compiler cannot devirtualize,
// and then the guard's own length test. Measured at about 1.9ns, which is
// nothing against a kilobyte and 43% against sixty-four bytes.
//
// It is deliberately below the smallest kernel threshold, so it never
// pre-empts a decision the measured thresholds make. And it is applied only
// where the standard library is the faster of the two at this size: not to
// LastIndex or LastIndexByte, whose standard library implementations are Go
// loops that this package beats by two orders of magnitude at every length.
//
// What it does not remove, and nothing can, is the call itself. strings.IndexByte
// is a compiler intrinsic: the assembly is inlined into the caller and there is
// no call at all. A function that chooses an implementation at run time cannot
// be that, so on an eight-byte input this package costs about 1.2ns more than
// the standard library. That is the floor for runtime dispatch and it is
// invisible at any length where a vector unit is relevant.
const shortText = 64

// ---------- searching ----------

// Index returns the index of the first occurrence of needle in haystack, or
// -1. An empty needle returns 0.
//
// It matches bytes.Index and strings.Index and is a drop-in replacement for
// either.
func Index[S, T Text](haystack S, needle T) int {
	h, n := textBytes(haystack), textBytes(needle)
	if len(h) < shortText {
		return bytes.Index(h, n)
	}
	return tblBytesIndex[tierIdx](h, n)
}

// LastIndex returns the index of the last occurrence of needle in haystack, or
// -1. An empty needle returns len(haystack).
//
// It matches bytes.LastIndex and strings.LastIndex.
func LastIndex[S, T Text](haystack S, needle T) int {
	return tblBytesLastIndex[tierIdx](textBytes(haystack), textBytes(needle))
}

// IndexAny returns the index of the first byte of s that is also in chars, or
// -1.
//
// The set is turned into a 256-bit table once, so the cost is linear in s and
// independent of how many characters you are looking for.
//
// Note the difference from strings.IndexAny, which searches for Unicode code
// points: this is a byte scan. For an ASCII set the two agree, because no byte
// of a multi-byte UTF-8 sequence is below 0x80. For a set containing non-ASCII
// characters they do not, and this is the wrong function.
func IndexAny[S, T Text](s S, chars T) int {
	return tblBytesIndexAny[tierIdx](textBytes(s), textBytes(chars))
}

// IndexAnyOrLess returns the index of the first byte of s that is either in
// chars or numerically below lo, or -1 if no byte is either.
//
// This is [IndexAny] with a threshold folded in, and it exists because the
// question callers actually have is usually both at once: find the next byte
// that is not ordinary text — a delimiter, a quote, a backslash, or any
// control character. Asking separately means two scans over the same bytes and
// then taking the smaller answer, which costs twice as much and, on a string
// that answers immediately, twice as much of nothing.
//
//	// The inner loop of a JSON string encoder: copy up to the next byte that
//	// has to be rewritten, and start again after it.
//	n := simd.IndexAnyOrLess(s, `"\`, 0x20)
//
// lo of 0 excludes nothing, since no byte is below it, and the call is then
// exactly [IndexAny].
func IndexAnyOrLess[S, T Text](s S, chars T, lo byte) int {
	return tblBytesIndexAnyOrLess[tierIdx](textBytes(s), textBytes(chars), lo)
}

// JSONCopyRun copies the bytes at the front of s that a JSON encoder may write
// verbatim into dst, and returns how many it copied.
//
// It stops at the first byte that has to be rewritten: a control character, a
// quote or a backslash, and — when html is true — also `<`, `>`, `&` and the
// 0xE2 that leads U+2028 and U+2029, which is the set encoding/json escapes by
// default.
//
// The scan and the copy are one pass. Asking where the run ends and then
// copying it reads every byte twice, which in an encoder is two passes over
// every string in the document.
//
// A byte above ASCII is copied like any other, so text that is not ASCII runs
// at the same rate as text that is. That is only correct for a caller that has
// already established the string is valid UTF-8 — no byte of a multi-byte
// sequence can collide with the set, but an invalid one might.
//
// dst must have room for len(s) bytes. It is written up to the returned length
// and no further.
func JSONCopyRun[S Text](dst []byte, s S, html bool) int {
	var h byte
	if html {
		h = 1
	}
	return tblBytesJSONCopyRun[tierIdx](dst, textBytes(s), h)
}

// JSONCopyValid copies the bytes at the front of s that a JSON encoder may
// write verbatim, proves them valid UTF-8 in the same pass, and returns how
// many — or a negative number if they were not valid.
//
// [JSONCopyRun] answers only the first question, which leaves the caller to
// walk the same bytes again for the second. This answers both in one pass, and
// unlike [JSONQuote] it needs no more room than len(s), because it still stops
// at the first byte needing an escape rather than writing it.
//
// A negative result says only that validation failed. Where it failed is the
// caller's to find, and callers that care are already rescanning to place
// replacement characters.
func JSONCopyValid[S Text](dst []byte, s S, html bool) int {
	var h byte
	if html {
		h = 1
	}
	return tblBytesJSONCopyValid[tierIdx](dst, textBytes(s), h)
}

// JSONQuote copies s into dst with JSON escapes written in place and returns
// how many bytes it wrote.
//
// The difference from [JSONCopyRun] is that this does not stop at a byte
// needing an escape — it writes the escape and carries on. A string with five
// escapes costs one call rather than five, which is what makes it worth the
// larger destination.
//
// dst must have room for 6*len(s) bytes: that is every byte becoming \u00XX,
// the worst case, and reserving it is what removes the per-byte space check. A
// caller who cannot spare that should use [JSONCopyRun] and handle the escapes
// itself.
//
// html selects whether <, > and & are escaped, and whether U+2028 and U+2029
// are — encoding/json escapes all five by default. A byte above ASCII is copied
// through, which is correct only for a string already known to be valid UTF-8.
func JSONQuote[S Text](dst []byte, s S, html bool) int {
	var h byte
	if html {
		h = 1
	}
	return tblBytesJSONQuote[tierIdx](dst, textBytes(s), h)
}

// IndexNotAny returns the index of the first byte of s that is *not* in chars,
// or -1 if every byte is.
//
// This is the primitive under trimming and under skipping a run of whitespace,
// which is where a tokenizer spends the time it is not spending in [IndexAny].
// An empty set contains nothing, so every byte is outside it and the answer
// for a non-empty s is 0.
func IndexNotAny[S, T Text](s S, chars T) int {
	return tblBytesIndexNotAny[tierIdx](textBytes(s), textBytes(chars))
}

// LastIndexNotAny returns the index of the last byte of s that is not in
// chars, or -1 if every byte is.
func LastIndexNotAny[S, T Text](s S, chars T) int {
	return tblBytesLastIndexNotAny[tierIdx](textBytes(s), textBytes(chars))
}

// Contains reports whether needle is within haystack.
func Contains[S, T Text](haystack S, needle T) bool {
	return Index(haystack, needle) >= 0
}

// ContainsByte reports whether c is within s.
func ContainsByte[S Text](s S, c byte) bool { return IndexByte(s, c) >= 0 }

// ContainsAny reports whether any byte of chars is within s.
func ContainsAny[S, T Text](s S, chars T) bool { return IndexAny(s, chars) >= 0 }

// HasPrefix reports whether s begins with prefix.
func HasPrefix[S, T Text](s S, prefix T) bool {
	sb, pb := textBytes(s), textBytes(prefix)
	return len(sb) >= len(pb) && tblBytesEqual[tierIdx](sb[:len(pb)], pb)
}

// HasSuffix reports whether s ends with suffix.
func HasSuffix[S, T Text](s S, suffix T) bool {
	sb, xb := textBytes(s), textBytes(suffix)
	return len(sb) >= len(xb) && tblBytesEqual[tierIdx](sb[len(sb)-len(xb):], xb)
}

// ---------- counting ----------

// Count returns the number of non-overlapping occurrences of needle in
// haystack.
//
// It matches bytes.Count and strings.Count, including for the empty needle,
// which counts the runes of haystack plus one. That case is answered here
// rather than in a kernel, because it is a question about UTF-8 rather than
// about bytes.
func Count[S, T Text](haystack S, needle T) int {
	hb, nb := textBytes(haystack), textBytes(needle)
	if len(hb) < shortText && len(nb) != 0 {
		return bytes.Count(hb, nb)
	}
	if len(nb) == 0 {
		// utf8.RuneCount rather than a byte scan, because the two disagree on
		// malformed input and bytes.Count follows utf8.RuneCount: a lone
		// continuation byte is one rune there and would be none under the
		// obvious "count the bytes that are not continuations" test.
		return utf8.RuneCount(hb) + 1
	}
	return tblBytesCountSeq[tierIdx](hb, nb)
}

// CountByte returns the number of occurrences of c in s.
func CountByte[S Text](s S, c byte) int {
	b := textBytes(s)
	if len(b) < shortText {
		return bytes.Count(b, []byte{c})
	}
	return tblBytesCount[tierIdx](b, c)
}

// CountAny returns how many bytes of s are in chars.
func CountAny[S, T Text](s S, chars T) int {
	return tblBytesCountAny[tierIdx](textBytes(s), textBytes(chars))
}

// ---------- trimming ----------

// TrimLeftAny returns s with every leading byte that is in cutset removed.
//
// As with [IndexAny] this is a byte-set operation, not a rune-set one, so it
// matches strings.TrimLeft only for an ASCII cutset.
func TrimLeftAny[S, T Text](s S, cutset T) S {
	i := IndexNotAny(s, cutset)
	if i < 0 {
		return textSlice(s, 0, 0)
	}
	return textSlice(s, i, len(textBytes(s)))
}

// TrimRightAny returns s with every trailing byte that is in cutset removed.
func TrimRightAny[S, T Text](s S, cutset T) S {
	i := LastIndexNotAny(s, cutset)
	return textSlice(s, 0, i+1)
}

// TrimAny returns s with every leading and trailing byte that is in cutset
// removed.
func TrimAny[S, T Text](s S, cutset T) S {
	lo := IndexNotAny(s, cutset)
	if lo < 0 {
		return textSlice(s, 0, 0)
	}
	hi := LastIndexNotAny(s, cutset)
	return textSlice(s, lo, hi+1)
}

// asciiSpace is the cutset [TrimSpaceASCII] uses.
const asciiSpace = " \t\n\v\f\r"

// TrimSpaceASCII returns s with leading and trailing ASCII whitespace removed.
//
// It is strings.TrimSpace restricted to the six bytes ASCII calls space, which
// is what a protocol parser wants: the Unicode set adds NEL, NBSP and the
// whole of the Zs category, none of which appear in a header line and all of
// which cost a rune decode to recognize.
func TrimSpaceASCII[S Text](s S) S { return TrimAny(s, asciiSpace) }

// ---------- structure ----------

// IndexAll writes the offset of every occurrence of c in s into dst, and
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
//
// Offsets are int32, so an input longer than 2 GiB cannot be indexed past that
// point; every path here truncates identically rather than one disagreeing
// with another. Split such an input and add the base yourself.
func IndexAll[S Text](dst []int32, s S, c byte) int {
	return tblBytesIndexAll[tierIdx](dst, textBytes(s), c)
}

// ---------- classification ----------

// IsASCII reports whether every byte is below 0x80.
//
// It is worth checking before text processing, because the ASCII path of most
// algorithms is dramatically simpler than the general one.
func IsASCII[S Text](s S) bool { return tblBytesIsASCII[tierIdx](textBytes(s)) }

// ValidUTF8 reports whether s is entirely well-formed UTF-8.
//
// It matches utf8.Valid and utf8.ValidString.
func ValidUTF8[S Text](s S) bool { return tblBytesValidUTF8[tierIdx](textBytes(s)) }

// EqualFoldASCII reports whether a and b are equal ignoring ASCII case.
//
// Unlike bytes.EqualFold it does not perform Unicode case folding, which makes
// it both faster and wrong for non-ASCII input — use it for protocol tokens
// (HTTP headers, keywords, hex digits), not for user-facing text.
func EqualFoldASCII[S, T Text](a S, b T) bool {
	return tblBytesEqualFoldASCII[tierIdx](textBytes(a), textBytes(b))
}

// ---------- transformation ----------

// ToUpperASCII maps a-z to A-Z in place, leaving every other byte alone.
//
// Only ASCII is folded, which makes this safe to run over UTF-8: continuation
// bytes are all 0x80 or above and are untouched. For full Unicode folding use
// the strings package; that is not a vectorizable operation.
func ToUpperASCII(b []byte) { tblBytesToUpperASCII[tierIdx](b, b) }

// ToLowerASCII maps A-Z to a-z in place, leaving every other byte alone.
// See [ToUpperASCII] on why this is UTF-8 safe.
func ToLowerASCII(b []byte) { tblBytesToLowerASCII[tierIdx](b, b) }

// ToUpperASCIIInto writes the ASCII-uppercased s into dst. dst may alias s
// when s is a []byte.
func ToUpperASCIIInto[S Text](dst []byte, s S) {
	tblBytesToUpperASCII[tierIdx](dst, textBytes(s))
}

// ToLowerASCIIInto writes the ASCII-lowercased s into dst. dst may alias s
// when s is a []byte.
func ToLowerASCIIInto[S Text](dst []byte, s S) {
	tblBytesToLowerASCII[tierIdx](dst, textBytes(s))
}

// ReplaceByte replaces every occurrence of old with new, in place.
func ReplaceByte(b []byte, old, new byte) { tblBytesReplaceByte[tierIdx](b, b, old, new) }

// ReplaceByteInto writes s into dst with every old replaced by new.
// dst may alias s when s is a []byte.
func ReplaceByteInto[S Text](dst []byte, s S, old, new byte) {
	tblBytesReplaceByte[tierIdx](dst, textBytes(s), old, new)
}

// HexEncode writes the lowercase hexadecimal encoding of src into dst and
// returns the number of bytes written. dst needs room for 2*len(src).
//
// It matches encoding/hex.Encode.
func HexEncode[S Text](dst []byte, src S) int {
	return tblBytesHexEncode[tierIdx](dst, textBytes(src))
}

// HexDecode decodes hexadecimal from src into dst, returning the number of
// bytes written and whether the whole input was valid. Both upper and lower
// case digits are accepted.
//
// On a bad digit it stops there and reports false, with the bytes decoded so
// far already written. An odd-length input also reports false.
func HexDecode[S Text](dst []byte, src S) (int, bool) {
	return tblBytesHexDecode[tierIdx](dst, textBytes(src))
}

// ---------- base64 ----------

// Base64EncodedLen is the length [Base64Encode] needs in dst for n input
// bytes, matching base64.StdEncoding.EncodedLen.
func Base64EncodedLen(n int) int { return (n + 2) / 3 * 4 }

// Base64DecodedLen is the largest length [Base64Decode] can write for n input
// bytes, matching base64.StdEncoding.DecodedLen. The actual count is smaller
// when the input is padded, and is what Base64Decode returns.
func Base64DecodedLen(n int) int { return n / 4 * 3 }

// Base64Encode writes the standard base64 encoding of src into dst and returns
// how many bytes it wrote, or -1 if dst is shorter than [Base64EncodedLen].
//
// It matches base64.StdEncoding: the RFC 4648 alphabet, with padding. Nothing
// is allocated — which is the difference from encoding/base64's EncodeToString
// and the reason the length is the caller's to provide.
func Base64Encode[S Text](dst []byte, src S) int {
	return tblBytesB64Encode[tierIdx](dst, textBytes(src))
}

// Base64Decode writes the decoded bytes of src into dst and returns how many
// it wrote, or -1 if src is not valid standard base64 or dst is too short.
//
// It matches base64.StdEncoding.Decode, except in how it reports a problem:
// one number rather than a count and an error, because returning an error
// would allocate.
//
// Validation is not a separate pass. Every character's value is folded
// together as it is decoded and one bit says whether anything was outside the
// alphabet, so a rejected input costs the same as an accepted one rather than
// a branch per character.
func Base64Decode[S Text](dst []byte, src S) int {
	return tblBytesB64Decode[tierIdx](dst, textBytes(src))
}

// ParseInts converts the fields of src into signed integers, writing them to
// dst, and returns how many it converted and whether every one was valid.
//
// idx holds the offset of each field separator, which is exactly what
// [IndexAll] produces, plus a final entry at len(src) if the last field is not
// separator-terminated:
//
//	n := simd.IndexAll(idx, line, ',')
//	idx[n] = int32(len(line))
//	count, ok := simd.ParseInts(vals, line, idx[:n+1])
//
// It stops at the first field that is not a valid integer and returns that
// field's index, so a caller can report where the input went wrong. A field is
// invalid if it is empty, contains a non-digit after an optional leading + or
// -, or names a value outside the int64 range — an over-long field is
// rejected rather than wrapped.
//
// # Why the separator scan is not part of this
//
// It is already fast and it is not where the time goes. On 200,000 short CSV
// fields [IndexAll] alone runs at 4.06 GB/s, and the same scan followed by
// strconv.Atoi at 0.83 — so the scan is a fifth of the work and the conversion
// is the other four fifths. Splitting them lets a caller reuse a scan, and
// keeps this kernel to the part that was actually slow.
func ParseInts[S Text](dst []int64, src S, idx []int32) (int, bool) {
	return tblBytesParseInts[tierIdx](dst, textBytes(src), idx)
}

// ParseUints is [ParseInts] over the full uint64 range.
//
// It is a separate kernel rather than a wrapper because the signed one's limit
// is 2^63: every value above that — half the domain, and the half a caller
// reaches for uint64 to get — would be rejected by it.
//
// No sign is accepted, not even a leading '+', matching strconv.ParseUint.
func ParseUints[S Text](dst []uint64, src S, idx []int32) (int, bool) {
	return tblBytesParseUints[tierIdx](dst, textBytes(src), idx)
}

// FormatInts writes vals as decimal text separated by sep — the inverse of
// [ParseInts] — and returns how many bytes it wrote, or -1 if dst cannot hold
// the result.
//
// Size dst at 21 bytes per value — a sign, up to nineteen digits and the
// separator — and the fast path never needs to measure first:
//
//	dst := make([]byte, 21*len(vals))
//	n := simd.FormatInts(dst, vals, ',')
//	line := dst[:n]
//
// A tighter dst still works when the rendering actually fits; it just runs
// the exact-fit reference. -1 means not even the exact rendering fits.
//
// Measured over 200,000 values, both sides reusing their buffers: 957µs
// against 1678µs for a strconv.AppendInt loop — 1.75x. (The C kernel alone
// probes at 3.3x; the gap is the call and guard overhead a Go caller actually
// pays, and the honest number is the one that includes it.) The kernel
// renders two digits per table lookup, halving both the divisions and the
// stores. No separator follows the last value.
func FormatInts(dst []byte, vals []int64, sep byte) int {
	return tblBytesFormatInts[tierIdx](dst, vals, sep)
}

// IndexAllAny writes the offset of every byte in s that equals any of the bytes
// in chars into dst, and returns how many it found.
//
// This is [IndexAll] for a set, and it exists because asking for several
// delimiters one at a time means several passes over the input. A JSON parser
// wants six at once; doing that as six calls reads the document six times, and
// this reads it once.
//
// At most eight distinct bytes. A longer set falls back to the portable path,
// which is correct and not faster — the limit is the packed argument the kernel
// receives, and eight is what fits beside the pointers.
//
// Like IndexAll it stops when dst fills, so a short dst bounds the work rather
// than being an error.
func IndexAllAny[S Text](dst []int32, s S, chars string) int {
	if len(chars) == 0 || len(dst) == 0 {
		return 0
	}
	if len(chars) > 8 {
		return indexAllAnyPortable(dst, textBytes(s), chars)
	}
	// Pack the set, repeating the first byte to fill. A duplicate compare costs
	// nothing; a shorter set would need another argument to say so.
	var packed uint64
	for i := 0; i < 8; i++ {
		c := chars[0]
		if i < len(chars) {
			c = chars[i]
		}
		packed |= uint64(c) << (8 * i)
	}
	return tblBytesIndexAllAny[tierIdx](dst, textBytes(s), packed)
}

// MaskBits writes one bit per byte of s into dst, set where the byte equals c.
//
// Bit i of dst[i/8] describes s[i], least-significant bit first, so dst is an
// eighth the size of s however many bytes match. dst must have room for
// [MaskLen](len(s)) bytes; it panics otherwise, because a short mask is a
// silently wrong answer rather than a truncated one.
//
// This is [IndexAll] in a different representation, and which one to want
// depends on what happens next. A list of offsets is smaller when matches are
// rare and is what a loop over the matches needs. A bitmask is smaller when
// they are common, and is the right form when the next question is itself
// bitwise — "which of these are inside a quoted string" is a shift and an
// and-not over a mask, and a search per offset over a list.
func MaskBits[S Text](dst []byte, s S, c byte) {
	b := textBytes(s)
	if len(dst) < MaskLen(len(b)) {
		panic("simd: MaskBits: dst too short")
	}
	tblBytesMaskBits[tierIdx](dst, b, c)
}

// MaskBitsAny is [MaskBits] for a set: the bit is set where the byte equals any
// of the bytes in chars.
//
// At most eight distinct bytes, for the reason given on [IndexAllAny]. A longer
// set falls back to the portable path.
func MaskBitsAny[S Text](dst []byte, s S, chars string) {
	b := textBytes(s)
	if len(dst) < MaskLen(len(b)) {
		panic("simd: MaskBitsAny: dst too short")
	}
	if len(chars) == 0 {
		clear(dst[:MaskLen(len(b))])
		return
	}
	if len(chars) > 8 {
		maskBitsAnyPortable(dst, b, chars)
		return
	}
	// Four or fewer takes the four-wide kernel. A vector unit compares every
	// byte of the set whether or not the caller supplied them, so a set of four
	// padded out to eight does twice the work for the same answer — and four is
	// what real sets tend to be. Measured on 1 MiB: 41 GB/s against 74.
	if len(chars) <= 4 {
		var packed4 uint32
		for i := 0; i < 4; i++ {
			ch := chars[0]
			if i < len(chars) {
				ch = chars[i]
			}
			packed4 |= uint32(ch) << (8 * i)
		}
		tblBytesMaskBitsAny4[tierIdx](dst, b, packed4)
		return
	}
	var packed uint64
	for i := 0; i < 8; i++ {
		ch := chars[0]
		if i < len(chars) {
			ch = chars[i]
		}
		packed |= uint64(ch) << (8 * i)
	}
	tblBytesMaskBitsAny[tierIdx](dst, b, packed)
}

// MaskBitsLess is [MaskBits] for an inequality: the bit is set where the byte
// is below c.
//
// It answers the range questions a set of eight cannot. "Is there a control
// character here" is thirty-two values as a set and one call as a comparison.
func MaskBitsLess[S Text](dst []byte, s S, c byte) {
	b := textBytes(s)
	if len(dst) < MaskLen(len(b)) {
		panic("simd: MaskBitsLess: dst too short")
	}
	tblBytesMaskBitsLess[tierIdx](dst, b, c)
}

// MaskLen is how many bytes a [MaskBits] destination needs for n input bytes.
func MaskLen(n int) int { return (n + 7) / 8 }

func maskBitsAnyPortable(dst, b []byte, chars string) {
	for i := 0; i < len(b); i++ {
		if i%8 == 0 {
			dst[i/8] = 0
		}
		hit := false
		for j := 0; j < len(chars); j++ {
			if chars[j] == b[i] {
				hit = true
				break
			}
		}
		if hit {
			dst[i/8] |= 1 << (i % 8)
		}
	}
}

func indexAllAnyPortable(dst []int32, b []byte, chars string) int {
	k := 0
	for i := 0; i < len(b); i++ {
		if !containsByte(chars, b[i]) {
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

func containsByte(s string, c byte) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return true
		}
	}
	return false
}

// JSONMasks writes the five masks a JSON indexer wants, in one pass over s.
//
// A two-stage JSON parser classifies its input five ways: the quotes, the
// backslashes, the four brackets, the bytes below 0x20 that a string may not
// contain, and the four bytes JSON allows as whitespace. Asking for them one at
// a time is five passes over the document and five dispatches; this is one load
// per block and five predicate stores.
//
// dst holds five regions of ((len(s)+63)/64)*8 bytes — whole 64-bit words,
// because that is how a mask is read — in that order: quote, escape,
// structural, control, whitespace. It must be at least five times that long.
//
// [MaskWords] gives the region size.
//
// want selects which regions are written, one bit each, low bit first —
// JSONMaskQuote through JSONMaskSpace. The regions are laid out as though all
// five were present whatever is asked for, so a caller's offsets do not change
// with the selection.
//
// The character sets are JSON's and are compiled in. A version taking them as
// arguments could not fold the comparisons, which is the whole point.
func JSONMasks[S Text](dst []byte, s S, want uint32) {
	tblBytesJSONMasks[tierIdx](dst, textBytes(s), want)
}

// The bits of JSONMasks's want, and the order of its output regions.
const (
	JSONMaskQuote      = 1 << iota // `"`
	JSONMaskEscape                 // `\`
	JSONMaskStructural             // `{`, `}`, `[`, `]`
	JSONMaskControl                // below 0x20
	JSONMaskSpace                  // space, tab, newline, carriage return

	// JSONMaskAll asks for every region.
	JSONMaskAll = JSONMaskQuote | JSONMaskEscape | JSONMaskStructural |
		JSONMaskControl | JSONMaskSpace
)

// MaskWords is the size of one [JSONMasks] region for an input of n bytes:
// enough whole 64-bit words to hold a bit per byte.
func MaskWords(n int) int { return ((n + 63) / 64) * 8 }

// JSONStage1 runs the JSON indexer's first word pass over the five
// [JSONMasks] regions, fused: escape resolution, quote parity, the
// in-string mask, and the counts and worklists that ride along.
//
// out holds three regions of nw 64-bit words -- the in-string mask, the
// whitespace words, and the escape-verification targets. carr carries the
// three inter-word states (escape run, string state, leader bit) between
// slabs and must be zero for a fresh document. res accumulates the
// structural and whitespace counts and reports the first in-string control
// character's byte position in res[2], which the caller seeds with -1.
func JSONStage1(out []uint64, masks []byte, nw int, carr []uint64, res []int64) {
	tblBytesJSONStage1[tierIdx](out, masks, nw, carr, res)
}

// JSONValidTokens reports whether the significant bytes of b -- outside a
// string and not whitespace, read from the [JSONStage1] masks -- form one
// well-formed JSON value: 1 yes, 0 no, -1 the nesting outran stk (deeper
// than 64*(len(stk)+1) levels) and the caller must walk it itself.
func JSONValidTokens(b []byte, masks []uint64, stk []uint64) int {
	return tblBytesJSONValidTokens[tierIdx](b, masks, stk)
}

// JSONValid reports whether b is one well-formed JSON value, in a single
// fused pass with no mask buffers: 1 yes, 0 no, -1 the nesting outran stk
// (deeper than 64*(len(stk)+1) levels) and the caller must walk it itself.
func JSONValid(b []byte, stk []uint64) int {
	return tblBytesJSONValid[tierIdx](b, stk)
}
