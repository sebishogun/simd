// Reductions that return a position rather than a value.
//
// ArgMin and ArgMax are a compare-and-blend: keep a running best value per lane
// and the index it was seen at, and update both under the same mask. The
// horizontal step at the end is where the care is, because it has to reproduce
// a rule that was written as a serial scan.
//
// # The rule these must match
//
// From internal/ref, and it is less obvious than it looks:
//
//   - The answer is the index of the *first* minimal element, so a tie keeps the
//     earlier index and a later equal value must not displace it.
//   - A NaN anywhere makes the answer the index of the first NaN, regardless of
//     any smaller value before it. The reference reaches this by letting NaN
//     poison its running best and then returning immediately.
//   - Because the comparison is `<` and not a total order, -0 and +0 compare
//     equal and neither displaces the other. ArgMin of {+0, -0} is 0, not 1,
//     even though the IEEE-754-2019 total order puts -0 first. That is a
//     consequence of the reference's formulation rather than a choice, and the
//     kernel has to inherit it.
//
// # Why this is width-independent
//
// Each lane keeps the first minimum *it* saw, at strictly increasing indices,
// so within a lane the rule holds directly. The horizontal step then takes the
// smallest index among the lanes holding the global best, which is the first
// occurrence in the array whatever the lane count. Nothing here depends on how
// many lanes there are, unlike the summing reductions, so no fixed lane count
// is needed for the contract — ARG_LANES is a code-generation choice only.
//
// The NaN case works the same way: a lane records the first NaN it saw and then
// stops updating, so the smallest index among NaN-holding lanes is the first
// NaN in the array.

#include "goabi.h"

typedef long isize;

// Lane counts differ by element width, and the horizontal loops below are
// unrolled by pragma. Both are load-bearing for the same reason reduce.c gives:
// a vector indexed by a runtime variable is one LLVM must give an address to,
// so it goes to the stack. The first version used sixteen lanes for every type
// and a plain horizontal loop, and the 64-bit kernels spilled 576 bytes — over
// the 512-byte budget a NOSPLIT function has — so they were dropped on every
// target while the 32-bit ones survived.
#define ARG_LANES 16
#define ARG_LANES64 8

// ARG_INIT seeds every lane with the first element at index 0. That is correct
// rather than merely convenient: index 0 is the earliest possible answer, so a
// lane starting there can only ever be displaced by something strictly better,
// and if a[0] is itself a NaN every lane is poisoned immediately and the answer
// is 0 — which is what the reference returns.
#define ARG_INIT(VT, IVT)                                                 \
  VT best = (VT)a[0];                                                     \
  IVT bi = 0;

// ARG_BLEND updates value and index together under one mask, which is what
// keeps them consistent. Writing them under separate conditions is the way this
// goes subtly wrong.
#define ARG_STEP(VT, IVT, TAKE)                                           \
  {                                                                       \
    VT x = *(const VT *)(a + i);                                          \
    IVT ix = base + lane;                                                 \
    IVT take = (IVT)(TAKE);                                               \
    best = take ? x : best;                                               \
    bi = take ? ix : bi;                                                  \
  }

// The float form. `take` is false whenever the current best is already a NaN,
// which is what makes the first NaN stick; and true when the incoming value is
// a NaN, which is what lets the first NaN arrive in the first place.
#define ARG_FLOAT(T, IT, VT, IVT, SUF, OP, CMP, L)                        \
  void simd_arg##OP##_##SUF(isize *__restrict out, const T *__restrict a, \
                            isize n) {                                    \
    if (n <= 0) {                                                         \
      *out = 0;                                                           \
      return;                                                             \
    }                                                                     \
    IVT lane;                                                             \
    _Pragma("clang loop unroll(full)") for (int j = 0; j < (L); j++)      \
        lane[j] = j;                                                      \
    ARG_INIT(VT, IVT)                                                     \
    isize i = 0;                                                          \
    for (; i + (L) <= n; i += (L)) {                                      \
      IVT base = (IVT)(IT)i;                                              \
      ARG_STEP(VT, IVT, (best == best) & ((x != x) | (x CMP best)))       \
    }                                                                     \
    T sbest = best[0];                                                    \
    isize sbi = (isize)bi[0];                                             \
    _Pragma("clang loop unroll(full)") for (int j = 1; j < (L); j++) {    \
      T v = best[j];                                                      \
      isize vi = (isize)bi[j];                                            \
      int better = (sbest == sbest) && ((v != v) || (v CMP sbest));       \
      int tie = (sbest == sbest) && (v == sbest) && vi < sbi;             \
      int nanEarlier = (sbest != sbest) && (v != v) && vi < sbi;          \
      if (better || tie || nanEarlier) {                                  \
        sbest = v;                                                        \
        sbi = vi;                                                         \
      }                                                                   \
    }                                                                     \
    for (; i < n; i++) {                                                  \
      T v = a[i];                                                         \
      if ((sbest == sbest) && ((v != v) || (v CMP sbest))) {              \
        sbest = v;                                                        \
        sbi = i;                                                          \
      }                                                                   \
    }                                                                     \
    *out = sbi;                                                           \
  }

// The integer form, with no NaN to worry about: strictly better displaces, and
// a tie keeps the smaller index.
#define ARG_INT(T, IT, VT, IVT, SUF, OP, CMP, L)                          \
  void simd_arg##OP##_##SUF(isize *__restrict out, const T *__restrict a, \
                            isize n) {                                    \
    if (n <= 0) {                                                         \
      *out = 0;                                                           \
      return;                                                             \
    }                                                                     \
    IVT lane;                                                             \
    _Pragma("clang loop unroll(full)") for (int j = 0; j < (L); j++)      \
        lane[j] = j;                                                      \
    ARG_INIT(VT, IVT)                                                     \
    isize i = 0;                                                          \
    for (; i + (L) <= n; i += (L)) {                                      \
      IVT base = (IVT)(IT)i;                                              \
      ARG_STEP(VT, IVT, (x CMP best))                                     \
    }                                                                     \
    T sbest = best[0];                                                    \
    isize sbi = (isize)bi[0];                                             \
    _Pragma("clang loop unroll(full)") for (int j = 1; j < (L); j++) {    \
      T v = best[j];                                                      \
      isize vi = (isize)bi[j];                                            \
      if ((v CMP sbest) || (v == sbest && vi < sbi)) {                    \
        sbest = v;                                                        \
        sbi = vi;                                                         \
      }                                                                   \
    }                                                                     \
    for (; i < n; i++)                                                    \
      if (a[i] CMP sbest) {                                               \
        sbest = a[i];                                                     \
        sbi = i;                                                          \
      }                                                                   \
    *out = sbi;                                                           \
  }

typedef float f32xA __attribute__((ext_vector_type(ARG_LANES), aligned(1)));
typedef int i32xA __attribute__((ext_vector_type(ARG_LANES), aligned(1)));
typedef double f64xA __attribute__((ext_vector_type(ARG_LANES64), aligned(1)));
typedef long long i64xA __attribute__((ext_vector_type(ARG_LANES64), aligned(1)));

ARG_FLOAT(float, int, f32xA, i32xA, f32, min, <, ARG_LANES)
ARG_FLOAT(float, int, f32xA, i32xA, f32, max, >, ARG_LANES)
ARG_FLOAT(double, long long, f64xA, i64xA, f64, min, <, ARG_LANES64)
ARG_FLOAT(double, long long, f64xA, i64xA, f64, max, >, ARG_LANES64)

ARG_INT(int, int, i32xA, i32xA, i32, min, <, ARG_LANES)
ARG_INT(int, int, i32xA, i32xA, i32, max, >, ARG_LANES)
ARG_INT(long long, long long, i64xA, i64xA, i64, min, <, ARG_LANES64)
ARG_INT(long long, long long, i64xA, i64xA, i64, max, >, ARG_LANES64)
