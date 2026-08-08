package simd

// Case-insensitive search, ASCII.
//
// # Why this is a composition and not a kernel
//
// The obvious kernel — a candidate filter that compares both cases of the
// first needle byte at once — exists in other libraries, and nothing stops it
// existing here later. But the composition is hard to beat, because its two
// halves are already the fast paths of this package: [ToLowerASCIIInto] runs
// at 18.9 GB/s, ten times `bytes.ToLower`, and [Index] is the tuned
// candidate-filter search. Folding the haystack once into caller-supplied
// scratch and searching the folded copy costs one extra pass over the data —
// and a caller searching the same haystack for several needles pays that pass
// once, which no per-search kernel could offer.
//
// What a caller writes without this is `strings.Index(strings.ToLower(s),
// strings.ToLower(needle))`, which allocates two copies per call and folds at
// a tenth of the speed.
//
// Only ASCII folds, deliberately, and that is not an approximation. Bytes 0x80
// and above are UTF-8 continuation and lead bytes and are left alone, so UTF-8
// input passes through unharmed; what this does not do is Unicode case rules
// (İ, ß, Σ→σ/ς), which need tables, allocate, and belong to strings.EqualFold.
// The name says ASCII so nobody mistakes it for that.

// IndexFoldASCII returns the index of the first occurrence of needle in
// haystack under ASCII case folding, or -1 if it is not present.
//
// scratch must be at least len(haystack); a shorter one is replaced by an
// allocation. Give it len(haystack)+len(needle) and the call allocates nothing
// at all — the needle's fold is carved from the same scratch. (A stack buffer
// for the needle sounds cheaper and is not: it escapes through the dispatch
// call and heap-allocates every time, which the allocation test caught.)
//
//	scratch := make([]byte, len(page))     // once
//	for _, w := range words {
//	    if simd.IndexFoldASCII(page, w, scratch) >= 0 { ... }
//	}
//
// The returned index is a position in the original haystack — folding does not
// move bytes, so offsets in the folded copy are offsets in the input.
func IndexFoldASCII[S, T Text](haystack S, needle T, scratch []byte) int {
	h := textBytes(haystack)
	n := textBytes(needle)
	if len(n) == 0 {
		return 0
	}
	if len(n) > len(h) {
		return -1
	}
	if len(scratch) < len(h) {
		scratch = make([]byte, len(h))
	}
	folded := scratch[:len(h)]
	tblBytesToLowerASCII[tierIdx](folded, h)

	var nf []byte
	if len(scratch) >= len(h)+len(n) {
		nf = scratch[len(h) : len(h)+len(n)]
	} else {
		nf = make([]byte, len(n))
	}
	tblBytesToLowerASCII[tierIdx](nf, n)

	return Index(folded, nf)
}

// ContainsFoldASCII reports whether needle occurs in haystack under ASCII case
// folding. It is [IndexFoldASCII] >= 0.
func ContainsFoldASCII[S, T Text](haystack S, needle T, scratch []byte) bool {
	return IndexFoldASCII(haystack, needle, scratch) >= 0
}

// CountFoldASCII counts non-overlapping occurrences of needle in haystack
// under ASCII case folding, matching bytes.Count's overlap rule.
//
// An empty needle returns the rune-independent answer len(haystack)+1, as
// bytes.Count does for an empty separator on ASCII input.
func CountFoldASCII[S, T Text](haystack S, needle T, scratch []byte) int {
	h := textBytes(haystack)
	n := textBytes(needle)
	if len(n) == 0 {
		return len(h) + 1
	}
	if len(n) > len(h) {
		return 0
	}
	if len(scratch) < len(h) {
		scratch = make([]byte, len(h))
	}
	folded := scratch[:len(h)]
	tblBytesToLowerASCII[tierIdx](folded, h)

	var nf []byte
	if len(scratch) >= len(h)+len(n) {
		nf = scratch[len(h) : len(h)+len(n)]
	} else {
		nf = make([]byte, len(n))
	}
	tblBytesToLowerASCII[tierIdx](nf, n)

	return Count(folded, nf)
}
