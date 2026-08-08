package ref

// BitUnpackFastU32 is the width-specialized unpack over whole 32-value
// blocks, the specification for simd_bitunpack_fast_u32. Little-endian
// bit order, matching BitPack.
func BitUnpackFastU32(dst, src []uint32, blocks int, bits uint32) {
	if bits == 0 || bits > 32 {
		return
	}
	mask := uint64(1)<<bits - 1
	if bits == 32 {
		mask = 1<<32 - 1
	}
	for b := 0; b < blocks; b++ {
		s := src[uint32(b)*bits:]
		d := dst[32*b:]
		for i := 0; i < 32; i++ {
			bit := i * int(bits)
			w := bit >> 5
			off := uint(bit & 31)
			val := uint64(s[w])
			if off+uint(bits) > 32 {
				val |= uint64(s[w+1]) << 32
			}
			d[i] = uint32((val >> off) & mask)
		}
	}
}
