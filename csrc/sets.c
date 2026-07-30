// Sorted-set intersection and difference.
//
// Both inputs are sorted and have no duplicates; this is the representation a
// posting list, an inverted index or a column of sorted keys is already in, and
// intersecting two of them is what a conjunctive query spends its time doing.
//
// # Why the merge is not the algorithm
//
// The obvious implementation walks both slices with two cursors and advances
// whichever is behind. It is O(na+nb) and optimal in comparisons, and it does
// not vectorize: each step's decision depends on the previous step's, which is
// a loop-carried dependency through both cursors.
//
// What vectorizes is asking a different question. Take a block of eight from
// each side and ask, for every element of a's block at once, whether it appears
// anywhere in b's block. That is eight broadcast comparisons — no reduction, no
// branch — and it answers eight membership questions per instruction where the
// merge answers one per branch.
//
// # The loop is transposed, and that is the whole trick
//
// Written the way it reads —
//
//	for (p) { int any = 0; for (q) any |= a[i+p] == b[j+q]; hit[p] = any; }
//
// the inner loop is a *reduction* over q, so each of the eight outputs costs a
// horizontal fold. Swapping the loops makes the inner one elementwise over p:
// one splat of b[j+q], one vector comparison against a's whole block, one OR
// into an eight-lane accumulator. Eight vector operations for the tile, and the
// horizontal work disappears.
//
// # Block skipping is what makes it linear
//
// After a tile, the block whose maximum is smaller cannot match anything
// further along the other side, so it advances; on a tie both do. Because both
// sides are sorted, every match a block could ever make has been tested by the
// time that block is retired — which is what lets the tile answer be final.
//
// # What is here and what is not
//
// Union is absent, and not by oversight. Intersection and difference emit a
// subset of one input, so the tile's answer is a mask and the compaction is at
// most eight elements. A union emits everything, in sorted order, which is a
// merge — and merging two sorted vectors needs a bitonic network, the same
// machinery a vectorized sort needs and which csrc/sort.c says outright is a
// different project. See docs/wrong.md.

#include "goabi.h"

typedef long isize;
typedef int i32;
typedef long long i64;
typedef unsigned int u32;
typedef unsigned long long u64;

// SET_BLOCK is the tile edge: eight elements from each side, so the tile is
// sixty-four comparisons done as eight vector operations.
//
// Eight rather than sixteen because the tile is quadratic in the block while
// the elements retired per tile is linear in it. Sixteen would do four times
// the comparisons to retire twice as many elements, and only helps when the two
// sets interleave densely enough that both blocks advance every tile.
#define SET_BLOCK 8

// INTERSECT emits the elements of a that also appear in b.
//
// The tile mask is recomputed from scratch each iteration, which is correct
// even when only b advances: an element of a that matched in an earlier b block
// was emitted then, and cannot match again because b has no duplicates. The
// output stays sorted for the same reason — if a[i+p] first matches in b's
// second block, everything below it in a's block already matched in the first
// or does not match at all, since b's second block starts above b's first
// block's maximum.
#define INTERSECT(T, SUF)                                                  \
  void simd_intersect_##SUF(isize *__restrict out, T *__restrict d,        \
                            const T *__restrict a, isize na,               \
                            const T *__restrict b, isize nb) {             \
    isize i = 0, j = 0, k = 0;                                             \
    while (i + SET_BLOCK <= na && j + SET_BLOCK <= nb) {                   \
      T hit[SET_BLOCK];                                                    \
      for (int p = 0; p < SET_BLOCK; p++) hit[p] = 0;                      \
      for (int q = 0; q < SET_BLOCK; q++) {                                \
        T bv = b[j + q];                                                   \
        for (int p = 0; p < SET_BLOCK; p++) hit[p] |= (a[i + p] == bv);    \
      }                                                                    \
      for (int p = 0; p < SET_BLOCK; p++)                                  \
        if (hit[p]) d[k++] = a[i + p];                                     \
      T amax = a[i + SET_BLOCK - 1], bmax = b[j + SET_BLOCK - 1];          \
      if (amax <= bmax) i += SET_BLOCK;                                    \
      if (bmax <= amax) j += SET_BLOCK;                                    \
    }                                                                      \
    while (i < na && j < nb) {                                             \
      if (a[i] < b[j])                                                     \
        i++;                                                               \
      else if (b[j] < a[i])                                                \
        j++;                                                               \
      else {                                                               \
        d[k++] = a[i];                                                     \
        i++;                                                               \
        j++;                                                               \
      }                                                                    \
    }                                                                      \
    *out = k;                                                              \
  }

// DIFFERENCE emits the elements of a that do not appear in b.
//
// The mask has to persist across tiles here, which intersection's does not. An
// element of a is emitted only once its whole block retires, because until then
// a later block of b may still match it — and a difference emits on the absence
// of a match, which is not knowable early. So `hit` lives outside the loop and
// is cleared when i advances.
//
// The block left pending when the loop ends has to be flushed with its mask
// still applied, not handed to the plain scalar tail. The tile advances j past
// whole blocks of b, so by the time the loop exits, b elements that matched the
// pending a block have been retired and the tail scanning forward from j would
// not find them — it would emit an element that is in both sets. That was the
// bug in the first version of this file, and it only shows up when a's block
// straddles the end of b's last full block.
#define DIFFERENCE(T, SUF)                                                 \
  void simd_difference_##SUF(isize *__restrict out, T *__restrict d,       \
                             const T *__restrict a, isize na,              \
                             const T *__restrict b, isize nb) {            \
    isize i = 0, j = 0, k = 0;                                             \
    T hit[SET_BLOCK];                                                      \
    for (int p = 0; p < SET_BLOCK; p++) hit[p] = 0;                        \
    while (i + SET_BLOCK <= na && j + SET_BLOCK <= nb) {                   \
      for (int q = 0; q < SET_BLOCK; q++) {                                \
        T bv = b[j + q];                                                   \
        for (int p = 0; p < SET_BLOCK; p++) hit[p] |= (a[i + p] == bv);    \
      }                                                                    \
      T amax = a[i + SET_BLOCK - 1], bmax = b[j + SET_BLOCK - 1];          \
      if (amax <= bmax) {                                                  \
        for (int p = 0; p < SET_BLOCK; p++)                                \
          if (!hit[p]) d[k++] = a[i + p];                                  \
        for (int p = 0; p < SET_BLOCK; p++) hit[p] = 0;                    \
        i += SET_BLOCK;                                                    \
      }                                                                    \
      if (bmax <= amax) j += SET_BLOCK;                                    \
    }                                                                      \
    if (i + SET_BLOCK <= na) {                                             \
      for (int p = 0; p < SET_BLOCK; p++) {                                \
        while (j < nb && b[j] < a[i + p]) j++;                             \
        int found = hit[p] != 0;                                           \
        if (j < nb && b[j] == a[i + p]) {                                  \
          found = 1;                                                       \
          j++;                                                             \
        }                                                                  \
        if (!found) d[k++] = a[i + p];                                     \
      }                                                                    \
      i += SET_BLOCK;                                                      \
    }                                                                      \
    while (i < na) {                                                       \
      while (j < nb && b[j] < a[i]) j++;                                   \
      if (j < nb && b[j] == a[i]) {                                        \
        j++;                                                               \
        i++;                                                               \
        continue;                                                          \
      }                                                                    \
      d[k++] = a[i];                                                       \
      i++;                                                                 \
    }                                                                      \
    *out = k;                                                              \
  }

INTERSECT(i32, i32)
INTERSECT(i64, i64)
INTERSECT(u32, u32)
INTERSECT(u64, u64)

DIFFERENCE(i32, i32)
DIFFERENCE(i64, i64)
DIFFERENCE(u32, u32)
DIFFERENCE(u64, u64)
