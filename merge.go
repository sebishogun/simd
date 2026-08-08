package simd

// MergeSortedUint32 merges two ascending []uint32 into dst -- equal keys
// taken from a first -- and returns the total written, always
// len(a)+len(b), which dst must hold. The merge half of a merge sort and
// the inner loop of compaction and joins: a fixed ladder of min/max
// exchanges per block of eight replaces the two-pointer walk's
// data-dependent branch, 2.6x at a million elements a side.
func MergeSortedUint32(dst, a, b []uint32) int {
	return tblBytesMergeSortedU32[tierIdx](dst, a, b)
}
