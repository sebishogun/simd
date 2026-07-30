package simd

// Sorted-set operations.
//
// Both inputs must be sorted ascending and free of duplicates. That is not
// checked: verifying it costs a pass over both slices, which is what the
// operation itself costs, and a caller with a sorted index already knows.
//
// # Why these are not a merge
//
// The textbook implementation walks both slices with two cursors and advances
// whichever is behind. That is optimal in comparisons and does not vectorize at
// all: each step's decision depends on the last one's, through both cursors.
//
// What vectorizes is a different question. Take a block of eight from each side
// and ask, for every element of a's block at once, whether it appears anywhere
// in b's block — eight broadcast comparisons, no reduction and no branch. Then
// retire whichever block has the smaller maximum, which is sound because both
// sides are sorted, so a retired block can never match anything further along.
// Eight membership questions per instruction, against one per branch.
//
// # Why there is no Union
//
// Intersection and difference emit a subset of one input, so the tile's answer
// is a mask and compacting it touches at most eight elements. A union emits
// everything, in order, which is a merge — and merging two sorted vectors needs
// a bitonic network, the same machinery a vectorized sort needs and which
// csrc/sort.c already says is a different project. Composing it from these two
// would need a merge as well, so there is no way around it.
//
// Until then: `append(IntersectInto…)` is not it, and the honest answer is that
// a two-cursor merge in Go is what a union costs today.

// IntersectInto writes the elements present in both a and b to dst, in
// ascending order, and returns how many there were.
//
// a and b must be sorted ascending with no duplicates. dst must have room for
// min(len(a), len(b)) elements, which is the most an intersection can produce;
// it panics otherwise, because the kernel is at the six-argument limit its ABI
// allows and cannot be told the destination's length.
func IntersectInto[T Integer](dst, a, b []T) int {
	if need := min(len(a), len(b)); len(dst) < need {
		panic("simd: IntersectInto: dst is shorter than min(len(a), len(b))")
	}
	return ops[T]().Intersect(dst, a, b)
}

// DifferenceInto writes the elements of a that are not in b to dst, in
// ascending order, and returns how many there were.
//
// a and b must be sorted ascending with no duplicates. dst must have room for
// len(a) elements; it panics otherwise, for the reason [IntersectInto] gives.
func DifferenceInto[T Integer](dst, a, b []T) int {
	if len(dst) < len(a) {
		panic("simd: DifferenceInto: dst is shorter than a")
	}
	return ops[T]().Difference(dst, a, b)
}
