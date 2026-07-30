// Prefix scans.
//
// # Why the exact CumSum is not here, and FastCumSum is
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
// The float CumSum and CumProd therefore stay portable, and always will: the
// operation is serial through its own output and the contract forbids the
// reassociation that would break the dependency.
//
// What that argument does NOT justify is leaving the speed on the floor, and
// this file used to say "permanently" without qualification. Two things it
// missed:
//
//   - FastCumSum and FastCumProd. Dropping agreement with the naive loop is
//     exactly what the Fast prefix already means for the transcendentals, and
//     it buys 2.6x on float32 sums and 4.0x on float32 products. Agreement
//     BETWEEN TIERS is kept, because SCAN_LANES is fixed regardless of
//     hardware width.
//
//   - The integer product scan, which needs no prefix at all. Two's-complement
//     multiplication is associative, so the regrouping is not observable and
//     the result is bit-identical to the serial loop.
//
// And "Fast" here does not mean less accurate. Measured against a long-double
// scan of a million values, the log-shift order is CLOSER to the truth than
// the serial loop on every corpus tried: 680k of a million elements closer on
// uniform positive input, and on the classic hard case — 1e16 followed by a
// million ones, where a serial accumulator cannot represent the increment —
// mean absolute error 5.0e+05 serial against 1.0 for the scan. Blocked
// summation has O(log n) error growth where the serial loop has O(n). What
// Fast drops is agreement, not accuracy.
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
      VT carry = SPLAT(VT, run);                                                 \
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

// ---------- the Fast tier: sum and product scans ----------
//
// These are the same log-shift scan with addition and multiplication, and they
// are exactly what the header of this file says a scan may not do: the result
// differs from the serial loop in the last bit, because the grouping differs
// and every partial result is written out. That is why they carry the Fast
// prefix and are never the default.
//
// What they do keep is agreement between tiers. SCAN_LANES is fixed at sixteen
// and eight regardless of hardware width, exactly as the reduction
// accumulators are, so FastCumSum on a Graviton and on an AVX-512 box produce
// identical bits. Only the promise to match a naive scalar loop is dropped.
//
// The shift here cannot be the one above. That one duplicates the neighbour's
// own value into the lanes with nothing to their left, which is right for an
// idempotent combine — min(x, x) is x — and wrong for addition, where lane 0
// would get a[0] + a[0]. So these shift in a real identity element from a
// second vector operand: indices at or above the lane count select from it.
// Zero for a sum, one for a product, neither of which has the NaN interaction
// that ruled an identity out for minimum.

#define SHIFTID16_1(v, z)                                                 \
  __builtin_shufflevector(v, z, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, \
                          12, 13, 14)
#define SHIFTID16_2(v, z)                                                  \
  __builtin_shufflevector(v, z, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  \
                          11, 12, 13)
#define SHIFTID16_4(v, z)                                                  \
  __builtin_shufflevector(v, z, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, \
                          9, 10, 11)
#define SHIFTID16_8(v, z)                                                    \
  __builtin_shufflevector(v, z, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3,  \
                          4, 5, 6, 7)

#define SHIFTID8_1(v, z) __builtin_shufflevector(v, z, 8, 0, 1, 2, 3, 4, 5, 6)
#define SHIFTID8_2(v, z) __builtin_shufflevector(v, z, 8, 9, 0, 1, 2, 3, 4, 5)
#define SHIFTID8_4(v, z) \
  __builtin_shufflevector(v, z, 8, 9, 10, 11, 0, 1, 2, 3)

#define SCAN_BODY16_ID(VT, VCOMB, ID)                                      \
  {                                                                        \
    VT id_ = SPLAT(VT, (ID));                                                     \
    v = VCOMB(v, SHIFTID16_1(v, id_));                                     \
    v = VCOMB(v, SHIFTID16_2(v, id_));                                     \
    v = VCOMB(v, SHIFTID16_4(v, id_));                                     \
    v = VCOMB(v, SHIFTID16_8(v, id_));                                     \
  }

#define SCAN_BODY8_ID(VT, VCOMB, ID)                                       \
  {                                                                        \
    VT id_ = SPLAT(VT, (ID));                                                     \
    v = VCOMB(v, SHIFTID8_1(v, id_));                                      \
    v = VCOMB(v, SHIFTID8_2(v, id_));                                      \
    v = VCOMB(v, SHIFTID8_4(v, id_));                                      \
  }

#define VADD(x, y) ((x) + (y))
#define VMUL(x, y) ((x) * (y))

// SCAN_ID takes the whole symbol name rather than building it from a prefix,
// because these are not all Fast: the integer product scan below is exact and
// must not be called anything that suggests otherwise.
#define SCAN_ID(T, VT, SYM, L, VCOMB, ID, BODY)                           \
  void SYM(T *__restrict d, const T *__restrict a, isize n) {             \
    if (n <= 0) return;                                                   \
    T run = (T)(ID);                                                      \
    isize i = 0;                                                          \
    for (; i + (L) <= n; i += (L)) {                                      \
      VT v = *(const VT *)(a + i);                                        \
      BODY(VT, VCOMB, ID)                                                 \
      VT carry = SPLAT(VT, run);                                                 \
      v = VCOMB(v, carry);                                                \
      *(VT *)(d + i) = v;                                                 \
      run = v[(L) - 1];                                                   \
    }                                                                     \
    for (; i < n; i++) {                                                  \
      run = VCOMB(run, a[i]);                                             \
      d[i] = run;                                                         \
    }                                                                     \
  }

SCAN_ID(float, f32xS, simd_fast_cumsum_f32, SCAN_LANES, VADD, 0.0f, SCAN_BODY16_ID)
SCAN_ID(double, f64xS, simd_fast_cumsum_f64, SCAN_LANES64, VADD, 0.0, SCAN_BODY8_ID)
SCAN_ID(float, f32xS, simd_fast_cumprod_f32, SCAN_LANES, VMUL, 1.0f, SCAN_BODY16_ID)
SCAN_ID(double, f64xS, simd_fast_cumprod_f64, SCAN_LANES64, VMUL, 1.0, SCAN_BODY8_ID)

// ---------- and the one integer scan that needs no Fast prefix ----------
//
// Two's-complement multiplication is associative — wrapping does not change
// that, since the arithmetic is exact in the ring Z/2^32 — so the log-shift
// scan computes bit for bit what the serial loop computes, for every input
// including ones that overflow. Verified over four million deliberately
// overflowing values: zero mismatches. So this is the ordinary CumProd,
// accelerated, with the contract untouched.
//
// # Why only this one of the four integer scans
//
// A scan replaces a chain of n dependent combines with log2(L) per block of L,
// so it wins exactly when the serial chain is latency-bound and loses when the
// serial chain already issues one element per cycle. Measured against the Go
// loop this library ships, at four million elements:
//
//	int32  product    2509 us -> 1230 us   2.04x    multiply latency ~3 cycles
//	int64  product    2596 us -> 3884 us   0.67x    no 64-bit vector multiply
//	                                                below AVX-512DQ
//	int32  sum         980 us -> 1082 us   0.91x    add latency 1 cycle: the
//	int64  sum        1441 us -> 2165 us   0.67x    serial loop is already at
//	                                                the store limit
//
// The two sums lose because a one-cycle dependency is not a dependency worth
// breaking, and the scan's shuffle chain plus the lane extract that carries
// the running value between blocks costs more than it saves. That is the same
// reason the float min and max scans lose, recorded at the top of this file:
// not associativity, which they have, but the price of the combine.
SCAN_ID(int, i32xS, simd_cumprod_i32, SCAN_LANES, VMUL, 1, SCAN_BODY16_ID)

// ---------- the sliding window ----------
//
// RollingMin over a window of w is not a scan, but it belongs beside them: it
// is the other operation whose output depends on a *range* of the input rather
// than one element of it, and the two get confused.
//
// # Why the O(n·w) form rather than the O(n) one
//
// The textbook answer is a monotonic deque: push each element, pop everything
// behind it that it beats, and the front is the window minimum. Two amortized
// comparisons per element, independent of w — asymptotically unbeatable.
//
// It is also entirely serial, entirely branchy, and does not vectorize on any
// target. What is written here does w-1 elementwise passes, which is more
// *work*, but each pass is a plain elementwise minimum over a contiguous run:
// the shape this library is fastest at. At sixteen lanes it is doing sixteen
// windows at once, so the crossover against the deque is around w = lanes,
// not around w = 2.
//
// The passes are tiled so the output block being accumulated stays in L1
// across all w of them. Without the tile the kernel reads and writes the whole
// output array w times over and becomes bandwidth-bound at large n, which is
// the difference between winning and losing at w = 8.
//
// # Why the reference is the same algorithm
//
// Not laziness — NaN. IEEE minimum propagates NaN, and a deque's "pop while
// the back is worse" has no defined behaviour when neither operand orders. The
// reference does the same w-1 combines in the same order with the same
// minimum, so the two agree bit for bit on every input including NaN and
// signed zero, which is the contract. A faster reference that disagreed about
// NaN would be a bug in the library, not an optimization.
#define ROLLING_TILE 4096

#define ROLLING(T, NAME, SCOMB)                                             \
  void simd_rolling_##NAME(T *__restrict d, const T *__restrict a, isize w, \
                           isize nd, isize n) {                            \
    if (w <= 0) return;                                                    \
    isize m = n - w + 1;                                                   \
    if (m > nd) m = nd;                                                    \
    for (isize t = 0; t < m; t += ROLLING_TILE) {                          \
      isize hi = t + ROLLING_TILE < m ? t + ROLLING_TILE : m;              \
      for (isize i = t; i < hi; i++) d[i] = a[i];                          \
      for (isize j = 1; j < w; j++)                                        \
        for (isize i = t; i < hi; i++) d[i] = SCOMB(d[i], a[i + j]);       \
    }                                                                      \
  }

ROLLING(float, min_f32, min_f32)
ROLLING(float, max_f32, max_f32)
ROLLING(double, min_f64, min_f64)
ROLLING(double, max_f64, max_f64)
ROLLING(int, min_i32, min_i32)
ROLLING(int, max_i32, max_i32)
ROLLING(long long, min_i64, min_i64)
ROLLING(long long, max_i64, max_i64)
