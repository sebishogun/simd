// Compress, expand, and the index scan built on top of them.
//
// This is the one family in the library that autovectorization cannot reach,
// and the reason is worth stating because it is not a compiler deficiency.
// Written the obvious way,
//
//     for (isize i = 0; i < n; i++)
//       if (keep[i]) dst[k++] = src[i];
//
// the address of the store depends on how many *earlier* iterations matched.
// That is a loop-carried dependency through k, and it is real: no rewriting of
// the loop removes it, because the data flow genuinely is serial. LLVM
// vectorizes none of these on any target, at any optimization level, and it is
// right not to.
//
// The instruction that breaks the dependency is compress: given a vector and a
// mask, pack the selected lanes down to the low end in one operation, then
// advance the pointer by the population count. Two of the nine targets have
// one — AVX-512's vpcompressd and SVE2's compact — and on those,
// __builtin_masked_compress_store lowers directly to it. Everywhere else LLVM
// scalarizes the intrinsic back into a branchy per-lane loop, which is slower
// than the plain C above, so those targets are skipped in kernels.go and keep
// the portable implementation. See SkipOn.
//
// The mask has to be built with __builtin_convertvector rather than by
// assigning into a _Bool vector element by element. Per-element assignment
// into a vector of _Bool silently folds to zeroinitializer at -O1 and above —
// the whole compress disappears and the function returns without storing
// anything — which is a miscompile rather than a limitation, and cost an hour
// to find. Build the mask from a comparison and convert it.

#include "goabi.h"

typedef long isize;
typedef unsigned char u8;
typedef int i32;

// COMPRESS_LANES is the widest useful block. AVX-512 holds 16 int32 in a zmm
// and 16 bytes of mask in an xmm, which is the natural pairing for the index
// scan; SVE2 is length-agnostic and takes it as a fixed unroll.
// Per target, for the same reason argreduce.c's lane counts are: on a
// scalable vector architecture LLVM sizes the spill for the largest vector
// length the architecture allows, not the one the machine has. At sixteen
// lanes the RVV compress kernels spilled 640 to 2032 bytes against a 512-byte
// NOSPLIT budget.
#if defined(__riscv_v)
#define COMPRESS_LANES 8
#else
#define COMPRESS_LANES 16
#endif

typedef i32 i32xC __attribute__((ext_vector_type(COMPRESS_LANES), aligned(1)));
typedef u8 u8xC __attribute__((ext_vector_type(COMPRESS_LANES), aligned(1)));
typedef _Bool maskxC __attribute__((ext_vector_type(COMPRESS_LANES)));

// MASK_FROM loads COMPRESS_LANES bytes of KEEP at OFF and turns "nonzero" into
// a lane mask.
#define MASK_FROM(KEEP, OFF)                                             \
  __builtin_convertvector(                                               \
      (*(const u8xC *)((const unsigned char *)(KEEP) + (OFF))) != 0, maskxC)

// POPCOUNT_AT counts the set lanes of KEEP at OFF. It is written as a plain
// sum over bytes rather than as a movemask-and-popcnt because the byte sum is
// what every target vectorizes, and on the two targets that have compress the
// compare result is already in a predicate register the popcount reads
// directly.
#define POPCOUNT_AT(KEEP, OFF)                                           \
  ({                                                                     \
    isize pc_ = 0;                                                       \
    for (int pj_ = 0; pj_ < COMPRESS_LANES; pj_++)                       \
      pc_ += ((KEEP)[(OFF) + pj_] != 0);                                 \
    pc_;                                                                 \
  })

// simd_compress_i32 writes the elements of src whose keep byte is nonzero into
// dst, in order, and returns how many.
//
// dst must have room for the count. The Go side guarantees that by passing
// len(dst) and refusing the call when it is short of len(src); a kernel that
// had to bound-check every lane could not use the instruction at all, since
// the whole point is that the store is unconditional and the *pointer* moves.
void simd_compress_i32(isize *__restrict out, i32 *__restrict dst,
                       const i32 *__restrict src, const u8 *__restrict keep,
                       isize n) {
  isize k = 0, i = 0;
  for (; i + COMPRESS_LANES <= n; i += COMPRESS_LANES) {
    i32xC v = *(const i32xC *)(src + i);
    __builtin_masked_compress_store(MASK_FROM(keep, i), v, dst + k);
    k += POPCOUNT_AT(keep, i);
  }
  for (; i < n; i++)
    if (keep[i]) dst[k++] = src[i];
  *out = k;
}

void simd_compress_i64(isize *__restrict out, long long *__restrict dst,
                       const long long *__restrict src,
                       const u8 *__restrict keep, isize n) {
  typedef long long i64xC
      __attribute__((ext_vector_type(COMPRESS_LANES), aligned(1)));
  isize k = 0, i = 0;
  for (; i + COMPRESS_LANES <= n; i += COMPRESS_LANES) {
    i64xC v = *(const i64xC *)(src + i);
    __builtin_masked_compress_store(MASK_FROM(keep, i), v, dst + k);
    k += POPCOUNT_AT(keep, i);
  }
  for (; i < n; i++)
    if (keep[i]) dst[k++] = src[i];
  *out = k;
}

void simd_compress_f32(isize *__restrict out, float *__restrict dst,
                       const float *__restrict src, const u8 *__restrict keep,
                       isize n) {
  typedef float f32xC
      __attribute__((ext_vector_type(COMPRESS_LANES), aligned(1)));
  isize k = 0, i = 0;
  for (; i + COMPRESS_LANES <= n; i += COMPRESS_LANES) {
    f32xC v = *(const f32xC *)(src + i);
    __builtin_masked_compress_store(MASK_FROM(keep, i), v, dst + k);
    k += POPCOUNT_AT(keep, i);
  }
  for (; i < n; i++)
    if (keep[i]) dst[k++] = src[i];
  *out = k;
}

void simd_compress_f64(isize *__restrict out, double *__restrict dst,
                       const double *__restrict src, const u8 *__restrict keep,
                       isize n) {
  typedef double f64xC
      __attribute__((ext_vector_type(COMPRESS_LANES), aligned(1)));
  isize k = 0, i = 0;
  for (; i + COMPRESS_LANES <= n; i += COMPRESS_LANES) {
    f64xC v = *(const f64xC *)(src + i);
    __builtin_masked_compress_store(MASK_FROM(keep, i), v, dst + k);
    k += POPCOUNT_AT(keep, i);
  }
  for (; i < n; i++)
    if (keep[i]) dst[k++] = src[i];
  *out = k;
}

// Expand — the inverse, taking the next element of src wherever the keep byte
// is set — is deliberately absent from this file. AVX-512 has vpexpandd and
// SVE2 has no equivalent at all, and measured against the plain C loop the
// generated code was 60 instructions on one target and 16 on the other: LLVM
// emits the serial loop either way, because here it is the *load* that carries
// the dependency and the store that does not, which is the half compress does
// not help with. A kernel that is a scalar loop plus a call boundary is slower
// than the scalar loop, so Expand stays portable Go and says so.

// simd_index_all writes the offset of every byte equal to c into dst.
//
// This is the structural-index step of a vectorized parser, and it is what
// compress exists for. The index vector is a constant plus the block base, the
// mask is one compare, and the compress packs the matching offsets down with
// no branch per byte. Where the scalar version costs a mispredicted branch per
// match, this costs one instruction per block regardless of how many matched.
//
// It stops when dst fills, which is why the block loop checks the remaining
// room rather than assuming it: a caller who sizes dst for the expected number
// of matches and gets more must get a truncated answer, not a buffer overrun.
void simd_index_all(isize *__restrict out, i32 *__restrict dst,
                    const u8 *__restrict b, u8 c, isize nd, isize n) {
  isize k = 0, i = 0;
  const i32xC lane = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  // The guard is nd - COMPRESS_LANES rather than nd - popcount, because the
  // store writes at most one full block and the pointer only advances by the
  // count. Trading the last few slots for an unconditional store is what keeps
  // the loop branchless.
  for (; i + COMPRESS_LANES <= n && k + COMPRESS_LANES <= nd;
       i += COMPRESS_LANES) {
    u8xC v = *(const u8xC *)(b + i);
    maskxC m = __builtin_convertvector(v == c, maskxC);
    __builtin_masked_compress_store(m, lane + (i32)i, dst + k);
    for (int j = 0; j < COMPRESS_LANES; j++) k += (b[i + j] == c);
  }
  for (; i < n; i++) {
    if (b[i] != c) continue;
    if (k == nd) break;
    dst[k++] = (i32)i;
  }
  *out = k;
}

// simd_index_all_any writes the offset of every byte that equals any of up to
// eight values into dst.
//
// The structural-index step of a real JSON parser wants six delimiters at once,
// and asking for them one at a time means six passes over the document. This is
// one pass: the mask is an OR of eight compares, and eight compares against
// registers cost far less than reading the input eight times.
//
// The set arrives packed into a u64, one byte per slot, because that is the
// argument the single-byte form already spends and the SysV limit leaves no
// room for another. A caller with fewer than eight repeats one of them — a
// duplicate compare is free, an extra pass is not.
void simd_index_all_any(isize *__restrict out, i32 *__restrict dst,
                        const u8 *__restrict b, unsigned long long chars,
                        isize nd, isize n) {
  const u8 c0 = (u8)chars, c1 = (u8)(chars >> 8), c2 = (u8)(chars >> 16),
           c3 = (u8)(chars >> 24), c4 = (u8)(chars >> 32),
           c5 = (u8)(chars >> 40), c6 = (u8)(chars >> 48),
           c7 = (u8)(chars >> 56);
  isize k = 0, i = 0;
  const i32xC lane = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  for (; i + COMPRESS_LANES <= n && k + COMPRESS_LANES <= nd;
       i += COMPRESS_LANES) {
    u8xC v = *(const u8xC *)(b + i);
    // The comparison yields a signed-char vector rather than u8xC, so the
    // accumulator takes its type from the expression instead of naming one.
    __typeof__(v == c0) hit = (v == c0) | (v == c1) | (v == c2) | (v == c3) |
                              (v == c4) | (v == c5) | (v == c6) | (v == c7);
    maskxC m = __builtin_convertvector(hit, maskxC);
    __builtin_masked_compress_store(m, lane + (i32)i, dst + k);
    for (int j = 0; j < COMPRESS_LANES; j++) {
      u8 x = b[i + j];
      k += (x == c0) | (x == c1) | (x == c2) | (x == c3) | (x == c4) |
           (x == c5) | (x == c6) | (x == c7);
    }
  }
  for (; i < n; i++) {
    u8 x = b[i];
    if (!((x == c0) | (x == c1) | (x == c2) | (x == c3) | (x == c4) |
          (x == c5) | (x == c6) | (x == c7)))
      continue;
    if (k == nd) break;
    dst[k++] = (i32)i;
  }
  *out = k;
}

// ---------- run detection ----------
//
// simd_run_starts marks every element that begins a run of equal values:
// d[0] is always set, and d[i] is set when a[i] differs from a[i-1].
//
// This is the vectorizable half of run-length encoding, and splitting it out
// is what makes RLE fit this library at all. The emit step — writing one
// (value, length) pair per run — has a data-dependent output position and is a
// serial prefix, which no amount of shuffling fixes. The compare is
// elementwise and vectorizes completely.
//
// Together with simd_compress that gives the two-phase shape docs/tutorial.md
// argues for throughout: one vector pass to find the structure, then work over
// the far smaller set of positions it found. A run-heavy column touches each
// element once here and then only once per run.
//
// The output is one byte per element rather than a bitmask because that is
// what simd_compress consumes, and going through a bitmask would mean packing
// and unpacking it for no gain.
#define RUN_STARTS(T, SUF)                                                \
  void simd_run_starts_##SUF(unsigned char *__restrict d,                 \
                             const T *__restrict a, isize n) {            \
    if (n <= 0) return;                                                   \
    d[0] = 1;                                                             \
    for (isize i = 1; i < n; i++) d[i] = a[i] != a[i - 1];                \
  }

RUN_STARTS(int, i32)
RUN_STARTS(long long, i64)
RUN_STARTS(unsigned char, u8)
