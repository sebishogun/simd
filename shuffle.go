package simd

// Interleave2 weaves two equal-length byte planes: dst[2i]=a[i],
// dst[2i+1]=b[i]. dst must hold 2*min(len(a), len(b)). The primitive
// under AoS<->SoA conversions and bitpack-adjacent layouts.
func Interleave2(dst, a, b []byte) {
	tblBytesInterleave2U8[tierIdx](dst, a, b)
}

// Deinterleave2 is the inverse: a[i]=src[2i], b[i]=src[2i+1], with a and
// b the same length and src at least twice it.
func Deinterleave2(a, b, src []byte) {
	tblBytesDeinterleave2U8[tierIdx](a, b, src)
}

// Transpose8x8Bytes transposes independent 64-byte tiles, each as an 8x8
// byte matrix -- the byte-level shuffle under bitshuffle and small-tile
// columnar layouts. len(dst) must equal len(src) and be a multiple of 64.
func Transpose8x8Bytes(dst, src []byte) {
	tblBytesTranspose8x8U8[tierIdx](dst, src)
}

// Bitshuffle transposes the bits of each 64-byte tile so that output
// plane p holds bit p of every input byte -- the layout that turns
// mostly-small values into runs of zero bytes for whatever compressor
// runs next. Unbitshuffle inverts it. len(dst) must equal len(src) and
// be a multiple of 64.
func Bitshuffle(dst, src []byte) {
	tblBytesBitshuffleU8[tierIdx](dst, src, 0)
}

// Unbitshuffle inverts [Bitshuffle].
func Unbitshuffle(dst, src []byte) {
	tblBytesBitshuffleU8[tierIdx](dst, src, 1)
}
