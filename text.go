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
	return active.Bytes.Index(h, n)
}

// LastIndex returns the index of the last occurrence of needle in haystack, or
// -1. An empty needle returns len(haystack).
//
// It matches bytes.LastIndex and strings.LastIndex.
func LastIndex[S, T Text](haystack S, needle T) int {
	return active.Bytes.LastIndex(textBytes(haystack), textBytes(needle))
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
	return active.Bytes.IndexAny(textBytes(s), textBytes(chars))
}

// IndexNotAny returns the index of the first byte of s that is *not* in chars,
// or -1 if every byte is.
//
// This is the primitive under trimming and under skipping a run of whitespace,
// which is where a tokenizer spends the time it is not spending in [IndexAny].
// An empty set contains nothing, so every byte is outside it and the answer
// for a non-empty s is 0.
func IndexNotAny[S, T Text](s S, chars T) int {
	return active.Bytes.IndexNotAny(textBytes(s), textBytes(chars))
}

// LastIndexNotAny returns the index of the last byte of s that is not in
// chars, or -1 if every byte is.
func LastIndexNotAny[S, T Text](s S, chars T) int {
	return active.Bytes.LastIndexNotAny(textBytes(s), textBytes(chars))
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
	return len(sb) >= len(pb) && active.Bytes.Equal(sb[:len(pb)], pb)
}

// HasSuffix reports whether s ends with suffix.
func HasSuffix[S, T Text](s S, suffix T) bool {
	sb, xb := textBytes(s), textBytes(suffix)
	return len(sb) >= len(xb) && active.Bytes.Equal(sb[len(sb)-len(xb):], xb)
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
	return active.Bytes.CountSeq(hb, nb)
}

// CountByte returns the number of occurrences of c in s.
func CountByte[S Text](s S, c byte) int {
	b := textBytes(s)
	if len(b) < shortText {
		return bytes.Count(b, []byte{c})
	}
	return active.Bytes.Count(b, c)
}

// CountAny returns how many bytes of s are in chars.
func CountAny[S, T Text](s S, chars T) int {
	return active.Bytes.CountAny(textBytes(s), textBytes(chars))
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
	return active.Bytes.IndexAll(dst, textBytes(s), c)
}

// ---------- classification ----------

// IsASCII reports whether every byte is below 0x80.
//
// It is worth checking before text processing, because the ASCII path of most
// algorithms is dramatically simpler than the general one.
func IsASCII[S Text](s S) bool { return active.Bytes.IsASCII(textBytes(s)) }

// ValidUTF8 reports whether s is entirely well-formed UTF-8.
//
// It matches utf8.Valid and utf8.ValidString.
func ValidUTF8[S Text](s S) bool { return active.Bytes.ValidUTF8(textBytes(s)) }

// EqualFoldASCII reports whether a and b are equal ignoring ASCII case.
//
// Unlike bytes.EqualFold it does not perform Unicode case folding, which makes
// it both faster and wrong for non-ASCII input — use it for protocol tokens
// (HTTP headers, keywords, hex digits), not for user-facing text.
func EqualFoldASCII[S, T Text](a S, b T) bool {
	return active.Bytes.EqualFoldASCII(textBytes(a), textBytes(b))
}

// ---------- transformation ----------

// ToUpperASCII maps a-z to A-Z in place, leaving every other byte alone.
//
// Only ASCII is folded, which makes this safe to run over UTF-8: continuation
// bytes are all 0x80 or above and are untouched. For full Unicode folding use
// the strings package; that is not a vectorizable operation.
func ToUpperASCII(b []byte) { active.Bytes.ToUpperASCII(b, b) }

// ToLowerASCII maps A-Z to a-z in place, leaving every other byte alone.
// See [ToUpperASCII] on why this is UTF-8 safe.
func ToLowerASCII(b []byte) { active.Bytes.ToLowerASCII(b, b) }

// ToUpperASCIIInto writes the ASCII-uppercased s into dst. dst may alias s
// when s is a []byte.
func ToUpperASCIIInto[S Text](dst []byte, s S) {
	active.Bytes.ToUpperASCII(dst, textBytes(s))
}

// ToLowerASCIIInto writes the ASCII-lowercased s into dst. dst may alias s
// when s is a []byte.
func ToLowerASCIIInto[S Text](dst []byte, s S) {
	active.Bytes.ToLowerASCII(dst, textBytes(s))
}

// ReplaceByte replaces every occurrence of old with new, in place.
func ReplaceByte(b []byte, old, new byte) { active.Bytes.ReplaceByte(b, b, old, new) }

// ReplaceByteInto writes s into dst with every old replaced by new.
// dst may alias s when s is a []byte.
func ReplaceByteInto[S Text](dst []byte, s S, old, new byte) {
	active.Bytes.ReplaceByte(dst, textBytes(s), old, new)
}

// HexEncode writes the lowercase hexadecimal encoding of src into dst and
// returns the number of bytes written. dst needs room for 2*len(src).
//
// It matches encoding/hex.Encode.
func HexEncode[S Text](dst []byte, src S) int {
	return active.Bytes.HexEncode(dst, textBytes(src))
}

// HexDecode decodes hexadecimal from src into dst, returning the number of
// bytes written and whether the whole input was valid. Both upper and lower
// case digits are accepted.
//
// On a bad digit it stops there and reports false, with the bytes decoded so
// far already written. An odd-length input also reports false.
func HexDecode[S Text](dst []byte, src S) (int, bool) {
	return active.Bytes.HexDecode(dst, textBytes(src))
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
	return active.Bytes.B64Encode(dst, textBytes(src))
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
	return active.Bytes.B64Decode(dst, textBytes(src))
}
