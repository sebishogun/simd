package simd

import "github.com/sebishogun/simd/internal/ref"

// Compression: keeping the elements that pass a test, and putting them back.
//
// This is the second half of a filter. The first half is a comparison — the
// functions in compare.go already write a []bool mask — and this is what turns
// that mask into a packed result. Splitting it in two is not a compromise: it
// means one Compress serves every predicate, including ones this library has
// never heard of, instead of the library shipping GreaterThanFilter,
// LessThanFilter and the rest of a combinatorial explosion.
//
//	mask := make([]bool, len(a))
//	simd.GreaterScalarInto(mask, a, 0) // the predicate, vectorized
//	n := simd.CompressInto(out, a, mask) // the packing, vectorized
//	out = out[:n]

// CompressInto writes the elements of src whose mask entry is true into dst,
// in order, and returns how many it wrote.
//
// dst is not resized; the return value is the meaningful length, so the usual
// call is followed by a reslice to it. If dst is too short to hold every match
// the result is truncated rather than a panic, which is deliberate — dst is
// normally sized from an estimate of how many will match, and an estimate that
// comes in low should cost a short answer the caller can detect, not a crash.
//
// The mask is read as far as the shorter of src and mask.
//
// This is accelerated only on AVX-512 and SVE2, and that is not a gap waiting
// to be filled. Where dst[k] goes depends on how many earlier elements
// matched, so the iterations are genuinely serial and no compiler on any
// target vectorizes the loop. The two instruction sets that have a compress
// instruction break the dependency in hardware; the other seven run the
// portable loop, which is the same loop their compilers would have produced.
func CompressInto[T Number](dst, src []T, mask []bool) int {
	if f := ops[T]().Compress; f != nil {
		return f(dst, src, mask)
	}
	return ref.Compress(dst, src, mask)
}

// ExpandInto is the inverse of CompressInto: it walks mask, and wherever it is
// true takes the next element of src. It returns how many elements of src were
// consumed.
//
// Positions where the mask is false are left untouched, so filling dst with a
// default first and expanding over it is the way to scatter a packed buffer
// back into a full-width one.
//
// This one is portable on every architecture, and unlike Compress that is
// permanent rather than pending. Compression's serial half is the store, which
// a compress instruction fixes; expansion's is the load, which it does not.
// Both AVX-512 and SVE2 compile this to the same scalar loop everything else
// gets, so there is no kernel to ship.
func ExpandInto[T Number](dst, src []T, mask []bool) int {
	return ref.Expand(dst, src, mask)
}

// FilterInto keeps the elements of src for which pred returns true, writing
// them into dst and returning how many.
//
// The predicate is a Go closure called once per element, so this is the
// convenient form rather than the fast one — a call per element defeats the
// vector unit entirely. It is here because a filter with an arbitrary
// condition is a real thing to want, and writing it out by hand is worse. When
// the predicate is a comparison against a constant, build the mask with the
// vectorized comparison in compare.go and call CompressInto instead; that is
// the path this function exists to be slower than.
func FilterInto[T Number](dst, src []T, pred func(T) bool) int {
	k := 0
	for _, v := range src {
		if !pred(v) {
			continue
		}
		if k == len(dst) {
			return k
		}
		dst[k] = v
		k++
	}
	return k
}
