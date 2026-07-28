// Prefix scans: CumMin and CumMax.
//
// # Why CumSum and CumProd are not here
//
// A scan is the awkward member of this library. A reduction may reassociate as
// long as every tier does it identically, because only the final value is
// observable — that is what the fixed sixteen-lane accumulator in reduce.c
// buys. A scan has no such freedom: every partial result is written to dst, so
// the grouping is part of the answer.
//
// The vectorized scan is the log-shift pattern — shift the vector right by one
// lane and combine, then by two, then four — and after log2(lanes) steps every
// lane holds the scan of everything at or before it. For four elements it
// computes
//
//	d[2] = a0 + (a1 + a2)
//
// where the serial loop computes
//
//	d[2] = (a0 + a1) + a2
//
// Those differ in the last bit, and both are visible in dst. So a vectorized
// CumSum would not agree with a scalar CumSum, and no choice of fixed lane
// count fixes it — the disagreement is with the naive loop a caller expects,
// not between tiers.
//
// CumSum and CumProd therefore stay portable, permanently, for the same reason
// EMA does: the operation is serial through its own output and the contract
// forbids the reassociation that would break the dependency. That is documented
// at their declarations rather than left looking like an oversight.
//
// # And the float min/max scans are built here but not shipped
//
// Associativity makes the scan *valid* for minimum and maximum. It does not
// make it *profitable*, and for floats it is not. A scan costs log2(lanes)
// combines per block where the serial loop costs one per element, so the
// combine has to be cheap for the trade to pay. Integer minimum is one
// instruction and it does pay — 2.4x on int32. IEEE-754-2019 minimum is a
// five-operation select chain, because NaN propagation and -0 ordering are
// precisely what the hardware instruction does not give, so the float scan pays
// five times as much and measured 46% slower than the scalar loop it replaces.
//
// The float kernels below are left compiled and unreferenced rather than
// deleted, because the measurement is worth being able to repeat.
//
// # Why CumMin and CumMax are different
//
// Minimum and maximum are associative, including under IEEE-754-2019 semantics
// where NaN propagates and -0 orders below +0. Regrouping them is therefore not
// observable — min(min(a,b),c) and min(a,min(b,c)) are the same value, bit for
// bit, for every input including NaN. So the log-shift scan is exactly equal to
// the serial one here, and these two get the real thing.

#include "goabi.h"
#include "minmax.h"

typedef long isize;

#define SCAN_LANES 16
#define SCAN_LANES64 8

typedef float f32xS __attribute__((ext_vector_type(SCAN_LANES), aligned(1)));
typedef int i32xS __attribute__((ext_vector_type(SCAN_LANES), aligned(1)));
typedef double f64xS __attribute__((ext_vector_type(SCAN_LANES64), aligned(1)));
typedef long long i64xS __attribute__((ext_vector_type(SCAN_LANES64), aligned(1)));

// SCAN_ASSOC is the log-shift inclusive scan.
//
// The lanes shifted in take the neighbour's own value rather than a sentinel.
// That is what makes each step idempotent for lanes with nothing to their left,
// and it avoids needing an identity element — which for minimum would have to
// be an infinity and would then interact with NaN.
//
// The shift must be a real shuffle. Written as an elementwise select loop —
// sh[j] = (j >= s) ? v[j-s] : v[j], fully unrolled — clang emits sixteen
// inserts per step instead of one permute: sixty instructions for a
// sixteen-element block, and the result was 31% *slower* than the scalar loop
// it was meant to replace. __builtin_shufflevector needs its indices as source
// literals, not merely as constants after unrolling, so the four steps are
// spelled out.
#define SHIFT16(v, s) SHIFT16_##s(v)
#define SHIFT16_1(v)                                                        __builtin_shufflevector(v, v, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,                            12, 13, 14)
#define SHIFT16_2(v)                                                        __builtin_shufflevector(v, v, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,                             11, 12, 13)
#define SHIFT16_4(v)                                                        __builtin_shufflevector(v, v, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,                           10, 11)
#define SHIFT16_8(v)                                                        __builtin_shufflevector(v, v, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,                           6, 7)

#define SHIFT8_1(v) __builtin_shufflevector(v, v, 0, 0, 1, 2, 3, 4, 5, 6)
#define SHIFT8_2(v) __builtin_shufflevector(v, v, 0, 1, 0, 1, 2, 3, 4, 5)
#define SHIFT8_4(v) __builtin_shufflevector(v, v, 0, 1, 2, 3, 0, 1, 2, 3)

#define SCAN_BODY16(VT, VCOMB)                                              v = VCOMB(v, SHIFT16_1(v));                                               v = VCOMB(v, SHIFT16_2(v));                                               v = VCOMB(v, SHIFT16_4(v));                                               v = VCOMB(v, SHIFT16_8(v));

#define SCAN_BODY8(VT, VCOMB)                                               v = VCOMB(v, SHIFT8_1(v));                                                v = VCOMB(v, SHIFT8_2(v));                                                v = VCOMB(v, SHIFT8_4(v));

#define SCAN_ASSOC(T, VT, NAME, L, VCOMB, SCOMB, BODY)                    \
  void simd_cum##NAME(T *__restrict d, const T *__restrict a, isize n) {  \
    if (n <= 0) return;                                                   \
    T run = a[0];                                                         \
    isize i = 0;                                                          \
    for (; i + (L) <= n; i += (L)) {                                      \
      VT v = *(const VT *)(a + i);                                        \
      BODY(VT, VCOMB)                                                     \
      VT carry = (VT)run;                                                 \
      v = VCOMB(v, carry);                                                \
      *(VT *)(d + i) = v;                                                 \
      run = v[(L) - 1];                                                   \
    }                                                                     \
    for (; i < n; i++) {                                                  \
      run = SCOMB(run, a[i]);                                             \
      d[i] = run;                                                         \
    }                                                                     \
  }

// The combines come from minmax.h. They are written out as select chains
// rather than using __builtin_elementwise_min and _max, and that is not
// stylistic: the builtins are minnum and maxnum, which return the *non*-NaN
// operand and treat -0 and +0 as equal. Using them here made CumMin of a slice
// containing a NaN return the neighbouring value, and CumMin of {+0,-0} return
// +0, both of which the differential tests caught at once.
MINMAX_FLOAT(float, f32, __builtin_signbitf)
MINMAX_FLOAT(double, f64, __builtin_signbit)
MINMAX_INT(int, i32)
MINMAX_INT(long long, i64)

#define VMINF32(x, y) VMIN_FLOAT(f32xS, i32xS, x, y)
#define VMAXF32(x, y) VMAX_FLOAT(f32xS, i32xS, x, y)
#define VMINF64(x, y) VMIN_FLOAT(f64xS, i64xS, x, y)
#define VMAXF64(x, y) VMAX_FLOAT(f64xS, i64xS, x, y)

SCAN_ASSOC(float, f32xS, min_f32, SCAN_LANES, VMINF32, min_f32, SCAN_BODY16)
SCAN_ASSOC(float, f32xS, max_f32, SCAN_LANES, VMAXF32, max_f32, SCAN_BODY16)
SCAN_ASSOC(double, f64xS, min_f64, SCAN_LANES64, VMINF64, min_f64, SCAN_BODY8)
SCAN_ASSOC(double, f64xS, max_f64, SCAN_LANES64, VMAXF64, max_f64, SCAN_BODY8)

SCAN_ASSOC(int, i32xS, min_i32, SCAN_LANES, VMIN_INT, min_i32, SCAN_BODY16)
SCAN_ASSOC(int, i32xS, max_i32, SCAN_LANES, VMAX_INT, max_i32, SCAN_BODY16)
SCAN_ASSOC(long long, i64xS, min_i64, SCAN_LANES64, VMIN_INT, min_i64, SCAN_BODY8)
SCAN_ASSOC(long long, i64xS, max_i64, SCAN_LANES64, VMAX_INT, max_i64, SCAN_BODY8)
