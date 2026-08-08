package ref

// RLEDecodeInt32 expands (values[k], counts[k]) run-length pairs into dst
// with the public RunLengthDecodeInt32 contract exactly: non-positive
// counts are skipped, the expansion stops when dst is full, and the total
// written comes back. The specification for simd_rle_decode_i32.
func RLEDecodeInt32(dst []int32, values, counts []int32) int {
	n := min(len(values), len(counts))
	out := 0
	for i := 0; i < n && out < len(dst); i++ {
		l := int(counts[i])
		if l <= 0 {
			continue
		}
		if out+l > len(dst) {
			l = len(dst) - out
		}
		v := values[i]
		for j := 0; j < l; j++ {
			dst[out+j] = v
		}
		out += l
	}
	return out
}
