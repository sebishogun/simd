package ref

// LZ4 block decoding, the portable reference and the specification for
// simd_lz4_block_decode.
//
// The block format is sequences of
//
//	token | literal-length extension | literals | offset | match-length ext
//
// with four-bit lengths in the token (15 escapes to 255-run extension
// bytes), a two-byte little-endian match offset into the output already
// written, and a minimum match of four. The final sequence carries only
// literals: input ending exactly after them is the one well-formed way a
// block ends.
//
// LZ4BlockDecode returns the decoded length, or -1 for malformed input:
// truncated anywhere, a zero or too-far offset, or output past cap. The
// kernel must agree on the byte and on the -1, and the differential fuzz
// says so.
func LZ4BlockDecode(dst, src []byte) int {
	d, n := 0, len(src)
	i := 0
	for {
		if i >= n {
			return -1 // a block cannot end before a token
		}
		token := src[i]
		i++
		litLen := int(token >> 4)
		if litLen == 15 {
			for {
				if i >= n {
					return -1
				}
				b := src[i]
				i++
				litLen += int(b)
				if b != 255 {
					break
				}
			}
		}
		if litLen > n-i || litLen > len(dst)-d {
			return -1
		}
		copy(dst[d:], src[i:i+litLen])
		d += litLen
		i += litLen
		if i == n {
			return d // the final sequence: literals only
		}
		if n-i < 2 {
			return -1
		}
		offset := int(src[i]) | int(src[i+1])<<8
		i += 2
		if offset == 0 || offset > d {
			return -1
		}
		matchLen := int(token&0xF) + 4
		if token&0xF == 15 {
			for {
				if i >= n {
					return -1
				}
				b := src[i]
				i++
				matchLen += int(b)
				if b != 255 {
					break
				}
			}
		}
		if matchLen > len(dst)-d {
			return -1
		}
		// Overlap is the point: an offset of one replicates a byte. The
		// per-byte loop is the definition; the kernel is what gets clever.
		for k := 0; k < matchLen; k++ {
			dst[d] = dst[d-offset]
			d++
		}
	}
}
