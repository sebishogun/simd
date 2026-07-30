package simd

import (
	"slices"
	"unicode/utf16"
	"unicode/utf8"
	"unsafe"
)

// UTF-16 conversion.
//
// # Why this is a run scanner and not one kernel
//
// Converting between UTF-8 and UTF-16 is a dependent scan in general. A rune's
// first byte decides its length, which decides where the next rune starts, so
// nothing about rune n+1 can be computed before rune n is decoded. That is the
// one shape a vector unit cannot help with, and no amount of cleverness in the
// kernel changes it.
//
// What is vectorizable is the ASCII run. Below 0x80 a byte is a whole rune and
// a whole UTF-16 unit, so the conversion collapses to a widen with no
// dependence between lanes. Real text is mostly such runs, even in languages
// that are not written in ASCII, because markup, punctuation, digits and
// whitespace are. So these functions alternate: find the run, widen it in one
// call, decode the handful of runes between runs with encoding/utf8.
//
// The alternative — utf16.Encode([]rune(string(b))) — is what a []byte caller
// writes today, and it builds an entire []rune it does not want, which is why
// it measures 617 MB/s against a 4.8 GB/s validation floor for the same input.
//
// Both directions follow the append convention, so a caller who reuses a
// buffer allocates nothing:
//
//	buf = simd.AppendUTF16(buf[:0], line)

// asciiRunFloor is the shortest ASCII run worth a dispatched call.
//
// Below it the loop widens inline instead. The kernel is faster per byte at
// any length, but a call is not free, and text with non-ASCII every few bytes
// would otherwise pay one call per run and lose to the scalar loop it was
// meant to replace. Set at a cache line: measured on mixed Latin-1 text, where
// runs average ten bytes, going through the dispatcher for every run cost 38%.
const asciiRunFloor = 64

// AppendUTF16 appends the UTF-16 encoding of s to dst and returns the extended
// slice.
//
// Invalid UTF-8 is handled as encoding/utf8 and unicode/utf16 handle it: each
// malformed byte becomes U+FFFD, one unit, which is what utf16.Encode of a
// []rune conversion produces for the same input. Runes above the BMP become
// surrogate pairs, so the result is longer than the rune count.
func AppendUTF16[S Text](dst []uint16, s S) []uint16 {
	b := textBytes(s)
	for len(b) > 0 {
		if b[0] < utf8.RuneSelf {
			// An ASCII run: widen it whole.
			n := asciiRun(b)
			at := len(dst)
			dst = slices.Grow(dst, n)[:at+n]
			if n >= asciiRunFloor {
				active.Bytes.WidenU8U16(dst[at:at+n], b[:n])
			} else {
				for i, c := range b[:n] {
					dst[at+i] = uint16(c)
				}
			}
			b = b[n:]
			continue
		}
		// A non-ASCII run: decode it rune by rune, and stay in this loop
		// rather than rescanning, so that dense non-Latin text never pays for
		// a scan that would return zero every time.
		for len(b) > 0 && b[0] >= utf8.RuneSelf {
			r, size := utf8.DecodeRune(b)
			dst = utf16.AppendRune(dst, r)
			b = b[size:]
		}
	}
	return dst
}

// AppendUTF8 appends the UTF-8 encoding of the UTF-16 units in s to dst and
// returns the extended slice.
//
// Unpaired surrogates become U+FFFD, matching unicode/utf16.Decode.
func AppendUTF8(dst []byte, s []uint16) []byte {
	for len(s) > 0 {
		if s[0] < utf8.RuneSelf {
			n := asciiRun16(s)
			at := len(dst)
			dst = slices.Grow(dst, n)[:at+n]
			if n >= asciiRunFloor {
				active.Bytes.NarrowU16U8(dst[at:at+n], s[:n])
			} else {
				for i, c := range s[:n] {
					dst[at+i] = byte(c)
				}
			}
			s = s[n:]
			continue
		}
		for len(s) > 0 && s[0] >= utf8.RuneSelf {
			r := rune(s[0])
			size := 1
			if utf16.IsSurrogate(r) {
				var r2 rune
				if len(s) > 1 {
					r2 = rune(s[1])
				}
				if dec := utf16.DecodeRune(r, r2); dec != utf8.RuneError {
					r, size = dec, 2
				} else {
					r = utf8.RuneError
				}
			}
			dst = utf8.AppendRune(dst, r)
			s = s[size:]
		}
	}
	return dst
}

// UTF16Len returns the number of UTF-16 units [AppendUTF16] would append for
// s, so that a caller can size a buffer exactly.
//
// It is a full pass over the input. Sizing with len(s) instead is always safe
// and never more than a factor of two too large — the worst case is all-ASCII,
// one unit per byte — so this is for callers who would rather pay the scan
// than the memory.
func UTF16Len[S Text](s S) int {
	b := textBytes(s)
	n := 0
	for len(b) > 0 {
		if b[0] < utf8.RuneSelf {
			k := asciiRun(b)
			n += k
			b = b[k:]
			continue
		}
		for len(b) > 0 && b[0] >= utf8.RuneSelf {
			r, size := utf8.DecodeRune(b)
			n++
			if r > 0xFFFF {
				n++ // surrogate pair
			}
			b = b[size:]
		}
	}
	return n
}

// UTF8Len returns the number of bytes [AppendUTF8] would append for s.
func UTF8Len(s []uint16) int {
	n := 0
	for len(s) > 0 {
		if s[0] < utf8.RuneSelf {
			k := asciiRun16(s)
			n += k
			s = s[k:]
			continue
		}
		for len(s) > 0 && s[0] >= utf8.RuneSelf {
			r := rune(s[0])
			size := 1
			if utf16.IsSurrogate(r) {
				var r2 rune
				if len(s) > 1 {
					r2 = rune(s[1])
				}
				if dec := utf16.DecodeRune(r, r2); dec != utf8.RuneError {
					r, size = dec, 2
				} else {
					r = utf8.RuneError
				}
			}
			n += utf8.RuneLen(r)
			s = s[size:]
		}
	}
	return n
}

// asciiRun and asciiRun16 return the length of the leading run below 0x80.
// Short inputs skip the dispatcher: the answer is usually one or two bytes in
// the mixed case and a call would dominate.
//
// The callers advance by whatever these return, so a zero on input whose first
// element IS ASCII would make no progress and spin forever. That is not a
// hypothetical: a generator bug once shipped a loong64 widen kernel whose
// internal branches all had a displacement of zero, and the emulated lane hung
// for 42 minutes inside exactly this loop (entry 46 of docs/wrong.md). The
// kernel was at fault, but a loop that cannot terminate unless a kernel
// returns a positive number is the wrong shape regardless, so the floor is
// enforced here rather than assumed.
func asciiRun(b []byte) int {
	if len(b) < asciiRunFloor {
		for i, c := range b {
			if c >= utf8.RuneSelf {
				return i
			}
		}
		return len(b)
	}
	n := active.Bytes.IndexNonASCII(b)
	if n <= 0 && b[0] < utf8.RuneSelf {
		return 1 // guaranteed progress; correctness is unaffected
	}
	return n
}

func asciiRun16(s []uint16) int {
	if len(s) < asciiRunFloor {
		for i, c := range s {
			if c >= utf8.RuneSelf {
				return i
			}
		}
		return len(s)
	}
	n := active.Bytes.IndexNonASCII16(s)
	if n <= 0 && s[0] < utf8.RuneSelf {
		return 1
	}
	return n
}

// AppendRunes appends the runes of s to dst and returns the extended slice.
//
// It is [AppendUTF16]'s shape for UTF-32, and it exists for the same reason:
// `[]rune(s)` allocates a fresh slice every call, and a loop over lines or
// records pays for that once per record. This one appends into a buffer you
// own, so the same idiom as everywhere else here applies:
//
//	runes = simd.AppendRunes(runes[:0], line)
//
// The general UTF-8 decode is a dependent scan — a rune's length decides where
// the next one starts — so only the ASCII runs are accelerated. Below 0x80 a
// byte is a whole rune, which makes that case a plain widen with no dependence
// between lanes, and in real text it is nearly all of the input. The runes
// between runs are decoded one at a time.
//
// Invalid UTF-8 becomes utf8.RuneError, one per offending byte, matching
// `[]rune(s)` and utf8.DecodeRune.
func AppendRunes[S Text](dst []rune, s S) []rune {
	b := textBytes(s)
	for len(b) > 0 {
		if b[0] < utf8.RuneSelf {
			n := asciiRun(b)
			at := len(dst)
			dst = slices.Grow(dst, n)[:at+n]
			if n >= asciiRunFloor {
				// A rune is an int32 and the kernel writes uint32; the two
				// have the same layout, and the values are all below 0x80 so
				// the reinterpretation cannot change one.
				active.Bytes.WidenU8U32(runesAsUint32(dst[at:at+n]), b[:n])
			} else {
				for i, c := range b[:n] {
					dst[at+i] = rune(c)
				}
			}
			b = b[n:]
			continue
		}
		// Stay in this loop rather than rescanning, so dense non-Latin text
		// never pays for a scan that would return zero every time.
		for len(b) > 0 && b[0] >= utf8.RuneSelf {
			r, size := utf8.DecodeRune(b)
			dst = append(dst, r)
			b = b[size:]
		}
	}
	return dst
}

// RuneCount returns the number of runes in s, counting each byte of invalid
// UTF-8 as one rune.
//
// It matches utf8.RuneCountInString and utf8.RuneCount. The ASCII run is
// counted without decoding, which is where the time goes in ordinary text.
func RuneCount[S Text](s S) int {
	b := textBytes(s)
	n := 0
	for len(b) > 0 {
		if b[0] < utf8.RuneSelf {
			k := asciiRun(b)
			n += k
			b = b[k:]
			continue
		}
		for len(b) > 0 && b[0] >= utf8.RuneSelf {
			_, size := utf8.DecodeRune(b)
			n++
			b = b[size:]
		}
	}
	return n
}

// runesAsUint32 reinterprets a []rune as []uint32 so the widening kernel can
// write into it directly.
//
// rune is an alias for int32, and int32 and uint32 have identical size,
// alignment and layout, so this is a reinterpretation and not a conversion.
// The alternative is a second pass copying uint32 into rune, which would read
// back everything the kernel just wrote for no gain.
//
// It is only ever called on a run the caller has already proven is ASCII, so
// every value is below 0x80 and the signedness cannot be observed.
func runesAsUint32(r []rune) []uint32 {
	return unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(r))), len(r))
}
