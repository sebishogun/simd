package simd

// LZ4BlockDecode decodes one raw LZ4 block (the block format, no frame
// header) from src into dst and returns the decoded length, or -1 for
// malformed input: truncation anywhere, a zero or too-far offset, or
// output that would pass len(dst). dst must be sized for the decoded
// output; the block format does not carry that size, the container does.
func LZ4BlockDecode(dst, src []byte) int {
	return tblBytesLZ4BlockDecode[tierIdx](dst, src)
}
