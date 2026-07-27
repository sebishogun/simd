package ref

// Compression: packing the elements that survive a predicate down into a
// contiguous result.
//
// This is the loop under every "collect the matches" pass, and it is the one
// family in this library that a compiler cannot vectorize. The store address
// depends on how many earlier iterations matched, so the iterations are
// genuinely serial — not accidentally, and not fixably by rewriting the loop.
// Breaking that dependency needs an instruction that packs a masked vector in
// one step, which two of the nine targets have.
//
// The code below is therefore both the reference and the shipped
// implementation almost everywhere, and it is written to be a good scalar
// loop rather than a deliberately naive one.

// compress writes the elements of src whose keep flag is set into dst, in
// order, and returns how many it wrote.
//
// It stops when dst fills. A caller who sizes dst by an estimate and gets more
// matches than they expected gets a short answer, which they can detect from
// the count; the alternative is a panic on a slice that was only ever a guess.
func compress[T any](dst, src []T, keep []bool) int {
	n := min(len(src), len(keep))
	k := 0
	for i := 0; i < n; i++ {
		if !keep[i] {
			continue
		}
		if k == len(dst) {
			return k
		}
		dst[k] = src[i]
		k++
	}
	return k
}

// expand is the inverse: it walks keep, and wherever the flag is set it takes
// the next element of src.
//
// There is no accelerated version of this and there is not going to be a
// useful one. Compression's serial half is the store; expansion's is the load,
// and a compress instruction does not help with that — measured, clang emits
// the same scalar loop for both AVX-512 and SVE2. It stays here, honestly
// portable, because round-tripping a compressed buffer is the common reason to
// want it.
//
// Elements of dst whose flag is clear are left as they were, which makes
// expand usable as a scatter into a pre-filled default.
func expand[T any](dst, src []T, keep []bool) int {
	n := min(len(dst), len(keep))
	k := 0
	for i := 0; i < n; i++ {
		if !keep[i] {
			continue
		}
		if k == len(src) {
			return k
		}
		dst[i] = src[k]
		k++
	}
	return k
}

// The exported entry points. The generated threshold guards call these
// directly rather than through the kernel set, so the short-slice path costs
// no indirect call.

func CompressFloat32(dst, src []float32, keep []bool) int { return compress(dst, src, keep) }
func CompressFloat64(dst, src []float64, keep []bool) int { return compress(dst, src, keep) }
func CompressInt32(dst, src []int32, keep []bool) int     { return compress(dst, src, keep) }
func CompressInt64(dst, src []int64, keep []bool) int     { return compress(dst, src, keep) }

func Compress[T any](dst, src []T, keep []bool) int { return compress(dst, src, keep) }
func Expand[T any](dst, src []T, keep []bool) int   { return expand(dst, src, keep) }
