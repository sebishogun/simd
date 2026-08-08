package ref

// VarintDecodeU64 decodes LEB128 varints from src into dst until either
// runs out, returning values written and bytes consumed. A varint that
// never terminates within ten bytes, or truncation mid-value, stops the
// walk with what was complete. The specification for
// simd_varint_decode_u64.
func VarintDecodeU64(dst []uint64, src []byte) (n, consumed int) {
	d, i := 0, 0
	for d < len(dst) && i < len(src) {
		var v uint64
		shift, l := 0, 0
		for {
			if i+l >= len(src) || l >= 10 {
				return d, i
			}
			b := src[i+l]
			v |= uint64(b&0x7f) << shift
			l++
			if b&0x80 == 0 {
				break
			}
			shift += 7
		}
		dst[d] = v
		d++
		i += l
	}
	return d, i
}
