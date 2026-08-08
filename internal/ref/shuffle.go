package ref

// Interleave2U8: dst[2i]=a[i], dst[2i+1]=b[i]. Specification for
// simd_interleave2_u8; n is min(len(a), len(b)), dst needs 2n.
func Interleave2U8(dst, a, b []byte) {
	n := min(len(a), len(b), len(dst)/2)
	for i := 0; i < n; i++ {
		dst[2*i] = a[i]
		dst[2*i+1] = b[i]
	}
}

// Deinterleave2U8 is the inverse: a[i]=src[2i], b[i]=src[2i+1].
func Deinterleave2U8(a, b, src []byte) {
	n := min(len(a), len(b), len(src)/2)
	for i := 0; i < n; i++ {
		a[i] = src[2*i]
		b[i] = src[2*i+1]
	}
}

// Transpose8x8U8 transposes n independent 64-byte tiles as 8x8 byte
// matrices.
func Transpose8x8U8(dst, src []byte) {
	n := min(len(dst), len(src)) / 64
	for t := 0; t < n; t++ {
		s := src[t*64 : t*64+64]
		d := dst[t*64 : t*64+64]
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				d[c*8+r] = s[r*8+c]
			}
		}
	}
}

// BitshuffleU8 transposes bits over 64-byte tiles: with dir 0, output
// plane p byte g holds bit p of input bytes 8g..8g+7; dir 1 inverts.
// The specification for simd_bitshuffle_u8.
func BitshuffleU8(dst, src []byte, dir byte) {
	n := min(len(dst), len(src))
	tiles := n / 64
	for t := 0; t < tiles; t++ {
		s := src[t*64 : t*64+64]
		d := dst[t*64 : t*64+64]
		if dir == 0 {
			for p := 0; p < 8; p++ {
				for g := 0; g < 8; g++ {
					var b byte
					for k := 0; k < 8; k++ {
						b |= (s[8*g+k] >> uint(p) & 1) << uint(k)
					}
					d[p*8+g] = b
				}
			}
		} else {
			for p := 0; p < 8; p++ {
				for g := 0; g < 8; g++ {
					var b byte
					for k := 0; k < 8; k++ {
						b |= (s[k*8+g] >> uint(p) & 1) << uint(k)
					}
					d[g*8+p] = b
				}
			}
		}
	}
}
