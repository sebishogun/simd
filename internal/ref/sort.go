package ref

import "slices"

// Partitioning and sorting.
//
// partitionOut is the portable form of the kernel: out of place, and stable on
// both sides. The name distinguishes it from the in-place Hoare partition in
// compare.go that Median and Quantile use — different operations that happen to
// share a word.
//
// Written this way it is obviously correct, which matters more here than
// elsewhere: a partition with a subtle bug still produces a permutation, so the
// array stays plausible and survives every "is it sorted" check that does not
// also verify multiset equality.

func partitionOut[T number](dst, src []T, pivot T) int {
	if len(dst) < len(src) {
		return 0
	}
	lo := 0
	for _, v := range src {
		if v < pivot {
			dst[lo] = v
			lo++
		}
	}
	hi := lo
	for _, v := range src {
		if !(v < pivot) {
			dst[hi] = v
			hi++
		}
	}
	return lo
}

// Partition is the exported entry point the generated guards call.
func Partition[T Number](dst, src []T, pivot T) int { return partitionOut(dst, src, pivot) }

// SortOrdered sorts in place with the standard library's pdqsort.
//
// It is here as the reference and as the small-case path, and on integers it is
// also the shipped implementation: see the note on the exported Sort.
func SortOrdered[T ordered](a []T) { slices.Sort(a) }

type ordered interface {
	~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}
