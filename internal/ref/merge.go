package ref

// MergeSortedU32 merges two ascending arrays into dst -- ties taken from
// a first -- returning the total written. dst must hold len(a)+len(b).
// The specification for simd_merge_sorted_u32.
func MergeSortedU32(dst, a, b []uint32) int {
	i, j, d := 0, 0, 0
	for i < len(a) && j < len(b) {
		if a[i] <= b[j] {
			dst[d] = a[i]
			i++
		} else {
			dst[d] = b[j]
			j++
		}
		d++
	}
	d += copy(dst[d:], a[i:])
	d += copy(dst[d:], b[j:])
	return d
}
