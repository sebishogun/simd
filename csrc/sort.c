// The partition step of a quicksort, vectorized with compress.
//
// Sorting is a control-flow problem wrapped around a data-movement one, and
// only the inner half vectorizes. The recursion, the pivot choice and the
// small-case handling belong in Go: a kernel here cannot recurse, because Plan
// 9 assembly has no procedure linkage table and these functions are NOSPLIT
// with no stack-growth check. What it can do is the part that dominates the
// profile — walking a block and splitting it about a pivot.
//
// # Two passes, not one
//
// The natural shape is one load, one comparison and two compress stores, the
// low side to the front and the high side to the back. It does not vectorize at
// all: clang emits vucomiss, seta and cmovaq per element and no compress
// appears, because the second store's address depends on the population count
// of the first mask and the two are therefore serially dependent.
//
// Split into one pass per side, each pass has exactly the shape csrc/compress.c
// uses and each vectorizes. src is read twice, which is the price.
//
// # Counting from the mask, not from a second comparison
//
// The count must come from the lane mask itself. Counting with a second scalar
// comparison over src — `lo += (src[i+j] < pivot)` — vectorizes for integers
// and *not* for floats: partition_i32 emitted two vpcompress and partition_f32
// emitted none, falling back to a scalar compare per element. A float
// comparison carries NaN semantics, so the optimizer will not prove that the
// scalar predicate and the vector predicate are the same thing, and the count
// stops being derivable from the mask it must agree with. Summing the mask
// lanes directly leaves one predicate instead of two.
//
// # Out of place
//
// The in-place two-sided partition is what a production AVX-512 sort does and
// is faster, but it holds two blocks in registers while writing both ends of
// the same array, and a read-before-overwrite mistake yields a permutation that
// is still a permutation — the array stays plausible and survives every "is it
// sorted" check that does not also verify multiset equality. This form is
// correct by inspection, and the Go side supplies the scratch, so the library
// still allocates nothing.
//
// Both sides are stable: compress preserves lane order, so elements keep their
// relative positions within their side.

#include "goabi.h"

typedef long isize;

#define PART_LANES 16

typedef _Bool maskxP __attribute__((ext_vector_type(PART_LANES)));

// PART_COUNT sums the set lanes of a mask. Written over the mask rather than
// over the source for the reason above.
#define PART_COUNT(M)                                                     \
  ({                                                                      \
    isize c_ = 0;                                                         \
    for (int j_ = 0; j_ < PART_LANES; j_++) c_ += (M)[j_];                \
    c_;                                                                   \
  })

// The write cursors are plain locals and must stay that way. An earlier version
// factored the two passes through an `isize *w = &lo`, which takes the address
// of the cursor, forces it to memory, and turns every compress store's address
// into a load — a serial dependency the vectorizer will not cross. The compress
// disappeared for float and survived for integer, which made it look like a
// float problem; it was not. A float comparison feeding a compress vectorizes
// perfectly well on its own.
#define PARTITION(T, VT)                                                  \
  isize lo = 0, hi = 0, i = 0;                                            \
  for (; i + PART_LANES <= n; i += PART_LANES) {                          \
    VT v = *(const VT *)(src + i);                                        \
    maskxP m = __builtin_convertvector(v < pivot, maskxP);                \
    __builtin_masked_compress_store(m, v, dst + lo);                      \
    lo += PART_COUNT(m);                                                  \
  }                                                                       \
  for (; i < n; i++)                                                      \
    if (src[i] < pivot) dst[lo++] = src[i];                               \
  hi = lo;                                                                \
  i = 0;                                                                  \
  for (; i + PART_LANES <= n; i += PART_LANES) {                          \
    VT v = *(const VT *)(src + i);                                        \
    maskxP m = __builtin_convertvector(!(v < pivot), maskxP);             \
    __builtin_masked_compress_store(m, v, dst + hi);                      \
    hi += PART_COUNT(m);                                                  \
  }                                                                       \
  for (; i < n; i++)                                                      \
    if (!(src[i] < pivot)) dst[hi++] = src[i];                            \
  *out = lo;

// The vector type is named separately rather than pasted from T, because T is
// sometimes two tokens — "long long" — and pasting produces nonsense.
#define PARTITION_DEFS(T, VNAME, SUF)                                     \
  typedef T VNAME __attribute__((ext_vector_type(PART_LANES), aligned(1))); \
  void simd_partition_##SUF(isize *__restrict out, T *__restrict dst,     \
                            const T *__restrict src, T pivot, isize n) {  \
    PARTITION(T, VNAME)                                                   \
  }

PARTITION_DEFS(float, f32xP, f32)
PARTITION_DEFS(double, f64xP, f64)
PARTITION_DEFS(int, i32xP, i32)
PARTITION_DEFS(long long, i64xP, i64)
