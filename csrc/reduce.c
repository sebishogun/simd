// Reduction kernels.
//
// These are separate from arith.c because they are the ones with a numerical
// contract to keep: a reduction must produce the same bits on a 128-bit
// machine and a 512-bit one, which constrains how it may be written. See the
// comment on SUM_LANES below and the documentation on package kernel.
//
// The rules from arith.c apply here too: no function calls, __restrict on
// every pointer, signed loop counters.

#include "goabi.h"

typedef long isize;

// The reductions below accumulate into exactly SUM_LANES independent lanes and
// then fold them with a fixed binary tree, rather than letting the compiler
// choose. That is not an optimization, it is the numerical contract: the same
// shape has to hold on a 128-bit machine and a 512-bit one so that a sum does
// not change value when the program moves to a different server. See the
// documentation on package kernel.
//
// SUM_LANES must equal kernel.SumLanes.
#define SUM_LANES 16

// The accumulators are an explicit vector type, and the remainder is blended
// into a second vector before a single whole-block add. Both details are
// load-bearing and were arrived at by reading the generated code.
//
// Written the obvious way — a T acc[16] array with a runtime-indexed
// remainder loop — LLVM has to give the array an address, so the sixteen
// accumulators live on the stack instead of in registers. Measured spills in
// the float64 sum, and the cost on this machine:
//
//	                    neon   sve2   avx512   sse2
//	array + runtime tail  16     16       21     33   stack references
//	vector + blended tail  0      0        0      0
//
//	n=16   2.40 ns      n=20  10.92 ns
//	n=32   2.44 ns      n=24  11.23 ns
//
// Every length that was a multiple of 16 ran fast and every other length was
// four to five times slower, for less work, because the tail forced the spill.
//
// Guarding the remainder without the vector type does fix amd64 — spills fall
// to zero — but makes LLVM abandon vectorization entirely on arm64, emitting
// 220 scalar instructions and not one vector instruction. The generator's
// vectorization check catches that, which is how it was found. Declaring the
// accumulator as a vector rather than hoping the array is discovered to be one
// removes the guesswork: all four targets now keep it in registers.
//
// Adding the blended tail is exact. Lanes with no element contribute +0, acc
// starts at +0, and x+0 is x for every finite value, infinity and NaN. Each
// element still lands in lane i%SUM_LANES, so the numerical contract in
// package kernel holds bit for bit.
typedef float f32xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));
typedef double f64xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));

#define REDUCE_SUM(T, V)                                  \
  V acc = 0;                                              \
  isize i = 0;                                            \
  for (; i + SUM_LANES <= n; i += SUM_LANES)              \
    acc += *(const V *)(a + i);                           \
  V t = 0;                                                \
  for (int j = 0; j < SUM_LANES; j++)                     \
    if (i + j < n) t[j] = a[i + j];                       \
  acc += t;                                               \
  T r[SUM_LANES];                                         \
  *(V *)r = acc;                                          \
  for (int w = SUM_LANES / 2; w >= 1; w /= 2)             \
    for (int j = 0; j < w; j++) r[j] += r[j + w];         \
  *out = r[0];

void simd_sum_f32(float *__restrict out, const float *__restrict a, isize n) {
  REDUCE_SUM(float, f32xL)
}

void simd_sum_f64(double *__restrict out, const double *__restrict a, isize n) {
  REDUCE_SUM(double, f64xL)
}

#define REDUCE_DOT(T, V)                                  \
  V acc = 0;                                              \
  isize i = 0;                                            \
  for (; i + SUM_LANES <= n; i += SUM_LANES)              \
    acc += *(const V *)(a + i) * *(const V *)(b + i);     \
  V t = 0;                                                \
  for (int j = 0; j < SUM_LANES; j++)                     \
    if (i + j < n) t[j] = a[i + j] * b[i + j];            \
  acc += t;                                               \
  T r[SUM_LANES];                                         \
  *(V *)r = acc;                                          \
  for (int w = SUM_LANES / 2; w >= 1; w /= 2)             \
    for (int j = 0; j < w; j++) r[j] += r[j + w];         \
  *out = r[0];

void simd_dot_f32(float *__restrict out, const float *__restrict a,
                  const float *__restrict b, isize n) {
  REDUCE_DOT(float, f32xL)
}

void simd_dot_f64(double *__restrict out, const double *__restrict a,
                  const double *__restrict b, isize n) {
  REDUCE_DOT(double, f64xL)
}

// ---------- horizontal minimum and maximum ----------
//
// Unlike summation these need no fixed lane count. Minimum and maximum are
// associative and commutative even with NaN propagation — a NaN anywhere wins
// regardless of the order it is met — and -0 beating +0 is associative too.
// So any reduction order gives the same answer and the compiler is free.
//
// The IEEE 754-2019 semantics still have to be written out rather than left to
// a min instruction, for the reason given in arith.c: the hardware
// instructions neither propagate NaN nor order the zeros.
// Horizontal minimum and maximum.
//
// The obvious spelling — a scalar accumulator carried through a five-deep
// chain of ternaries — is a loop-carried dependency that no vectorizer will
// take, and it did not vectorize on a single target here. Written against a
// fixed-lane accumulator it does, on the targets whose backends can express
// the select; the rest keep the portable path, which is what the generator's
// vectorization check is for.
//
// Unlike a sum, the lane layout needs no defending. IEEE minimum with NaN
// propagation is associative and commutative — a NaN anywhere gives a NaN,
// and -0 sorts below +0 whichever side it arrives on — so folding sixteen
// lanes in any order gives the answer the scalar loop gives. That is why this
// can be a plain reduction where Sum has to promise a shape.
//
// The sign test is the whole reason this is not just __builtin_elementwise_min:
// hardware minimum neither propagates NaN nor orders the zeros, so both have
// to be written out.

// A narrower accumulator than SUM_LANES, and it is allowed to be narrower for
// the reason given above: the fold is order-independent, so the lane count is
// a code-generation choice rather than part of the contract. Sixteen lanes of
// float64 is eight XMM registers, and the select chain needs several more
// live at once — on the baseline x86-64 tier, which has eight in total, that
// spilled 528 bytes. Eight lanes fits one AVX-512 register and stays in
// registers everywhere else.
#define MM_LANES 8

typedef float f32xM __attribute__((ext_vector_type(MM_LANES), aligned(1)));
typedef double f64xM __attribute__((ext_vector_type(MM_LANES), aligned(1)));
typedef int i32xM __attribute__((ext_vector_type(MM_LANES), aligned(1)));
typedef long long i64xM __attribute__((ext_vector_type(MM_LANES), aligned(1)));

// VMINMAX applies IEEE minimum or maximum lane-wise. CMP selects which, and
// ZERO says which operand wins when the two compare equal, which is the case
// that distinguishes -0 from +0.
#define VMINMAX(V, IV, M, X, CMP, ZA, ZB)                                 \
  ({                                                                      \
    V m_ = (M), x_ = (X);                                                 \
    (V)((m_ != m_)   ? m_                                                 \
        : (x_ != x_) ? x_                                                 \
        : (x_ CMP m_) ? x_                                                \
        : (m_ CMP x_) ? m_                                                \
                      : ((((IV)x_) < 0) ? ZA : ZB));                      \
  })

#define VMIN(V, IV, M, X) VMINMAX(V, IV, M, X, <, x_, m_)
#define VMAX(V, IV, M, X) VMINMAX(V, IV, M, X, >, m_, x_)

#define REDUCE_MINMAX_LANES(T, V, IV, VOP, SB, CMP, ZA, ZB)               \
  V acc = a[0];                                                           \
  isize i = 0;                                                            \
  for (; i + MM_LANES <= n; i += MM_LANES)                                \
    acc = VOP(V, IV, acc, *(const V *)(a + i));                           \
  /* Lanes with no element keep a[0], which is already in the             \
     accumulator, so they cannot change the answer. */                    \
  V t = a[0];                                                             \
  for (int j = 0; j < MM_LANES; j++)                                      \
    if (i + j < n) t[j] = a[i + j];                                       \
  acc = VOP(V, IV, acc, t);                                               \
  T r[MM_LANES];                                                          \
  *(V *)r = acc;                                                          \
  T m = r[0];                                                             \
  for (int j = 1; j < MM_LANES; j++) {                                    \
    T x = r[j];                                                           \
    m = (m != m)    ? m                                                   \
        : (x != x)  ? x                                                   \
        : (x CMP m) ? x                                                   \
        : (m CMP x) ? m                                                   \
                    : (SB(x) ? ZA : ZB);                                  \
  }                                                                       \
  *out = m;

#define SIGNBIT_F32(x) (__builtin_signbitf(x))
#define SIGNBIT_F64(x) (__builtin_signbit(x))

#define MINMAX_REDUCE(T, V, IV, SUF, SB)                                  \
  void simd_minr_##SUF(T *__restrict out, const T *__restrict a,          \
                       isize n) {                                         \
    REDUCE_MINMAX_LANES(T, V, IV, VMIN, SB, <, x, m)                      \
  }                                                                       \
  void simd_maxr_##SUF(T *__restrict out, const T *__restrict a,          \
                       isize n) {                                         \
    REDUCE_MINMAX_LANES(T, V, IV, VMAX, SB, >, m, x)                      \
  }

MINMAX_REDUCE(float, f32xM, i32xM, f32, SIGNBIT_F32)
MINMAX_REDUCE(double, f64xM, i64xM, f64, SIGNBIT_F64)

#define MINMAX_REDUCE_INT(T, SUF)                                       \
  void simd_minr_##SUF(T *__restrict out, const T *__restrict a,        \
                       isize n) {                                       \
    T m = a[0];                                                         \
    for (isize i = 1; i < n; i++)                                       \
      if (a[i] < m) m = a[i];                                           \
    *out = m;                                                           \
  }                                                                     \
  void simd_maxr_##SUF(T *__restrict out, const T *__restrict a,        \
                       isize n) {                                       \
    T m = a[0];                                                         \
    for (isize i = 1; i < n; i++)                                       \
      if (a[i] > m) m = a[i];                                           \
    *out = m;                                                           \
  }

MINMAX_REDUCE_INT(int, i32)
MINMAX_REDUCE_INT(long long, i64)

// ---------- the remaining summation-shaped reductions ----------
//
// These all follow the SUM_LANES contract, because they are sums: the lane an
// element lands in and the fold that combines the lanes have to match the
// portable implementation exactly, or a result would change with vector width.
//
// EXPR is evaluated per element and its value accumulated.
#define REDUCE_LANES(T, V, EXPR)                          \
  V acc = 0;                                              \
  isize i = 0;                                            \
  for (; i + SUM_LANES <= n; i += SUM_LANES) {            \
    V t;                                                  \
    for (int j = 0; j < SUM_LANES; j++) {                 \
      isize k = i + j;                                    \
      t[j] = (EXPR);                                      \
    }                                                     \
    acc += t;                                             \
  }                                                       \
  {                                                       \
    V t = 0;                                              \
    for (int j = 0; j < SUM_LANES; j++) {                 \
      isize k = i + j;                                    \
      if (k < n) t[j] = (EXPR);                           \
    }                                                     \
    acc += t;                                             \
  }                                                       \
  T r[SUM_LANES];                                         \
  *(V *)r = acc;                                          \
  for (int w = SUM_LANES / 2; w >= 1; w /= 2)             \
    for (int j = 0; j < w; j++) r[j] += r[j + w];         \
  *out = r[0];

#define FLOAT_REDUCTIONS(T, V, SUF, ABS)                                    \
  void simd_sumsq_##SUF(T *__restrict out, const T *__restrict a, isize n) { \
    REDUCE_LANES(T, V, a[k] * a[k])                                         \
  }                                                                          \
  void simd_l1norm_##SUF(T *__restrict out, const T *__restrict a,          \
                         isize n) {                                          \
    REDUCE_LANES(T, V, ABS(a[k]))                                            \
  }                                                                          \
  void simd_sumsqdev_##SUF(T *__restrict out, const T *__restrict a, T c,   \
                           isize n) {                                        \
    REDUCE_LANES(T, V, (a[k] - c) * (a[k] - c))                              \
  }                                                                          \
  void simd_sumsqdiff_##SUF(T *__restrict out, const T *__restrict a,       \
                            const T *__restrict b, isize n) {                \
    REDUCE_LANES(T, V, (a[k] - b[k]) * (a[k] - b[k]))                        \
  }                                                                          \
  void simd_l1diff_##SUF(T *__restrict out, const T *__restrict a,          \
                         const T *__restrict b, isize n) {                   \
    REDUCE_LANES(T, V, ABS(a[k] - b[k]))                                     \
  }

FLOAT_REDUCTIONS(float, f32xL, f32, __builtin_elementwise_abs)
FLOAT_REDUCTIONS(double, f64xL, f64, __builtin_elementwise_abs)

// Integer reductions need no lane discipline: integer addition is associative,
// so no accumulation order is observable and the compiler may choose freely.
#define INT_REDUCTIONS(T, SUF)                                              \
  void simd_sum_##SUF(T *__restrict out, const T *__restrict a, isize n) {  \
    T s = 0;                                                                \
    for (isize i = 0; i < n; i++) s += a[i];                                \
    *out = s;                                                               \
  }                                                                          \
  void simd_prod_##SUF(T *__restrict out, const T *__restrict a, isize n) { \
    T p = 1;                                                                \
    for (isize i = 0; i < n; i++) p *= a[i];                                \
    *out = p;                                                               \
  }                                                                          \
  void simd_dot_##SUF(T *__restrict out, const T *__restrict a,             \
                      const T *__restrict b, isize n) {                     \
    T s = 0;                                                                \
    for (isize i = 0; i < n; i++) s += a[i] * b[i];                         \
    *out = s;                                                               \
  }                                                                          \
  void simd_sumsq_##SUF(T *__restrict out, const T *__restrict a, isize n) { \
    T s = 0;                                                                \
    for (isize i = 0; i < n; i++) s += a[i] * a[i];                         \
    *out = s;                                                               \
  }                                                                          \
  void simd_sumsqdev_##SUF(T *__restrict out, const T *__restrict a, T c,   \
                           isize n) {                                        \
    T s = 0;                                                                \
    for (isize i = 0; i < n; i++) s += (a[i] - c) * (a[i] - c);             \
    *out = s;                                                               \
  }                                                                          \
  void simd_sumsqdiff_##SUF(T *__restrict out, const T *__restrict a,       \
                            const T *__restrict b, isize n) {                \
    T s = 0;                                                                \
    for (isize i = 0; i < n; i++) s += (a[i] - b[i]) * (a[i] - b[i]);       \
    *out = s;                                                               \
  }                                                                          \
  /* Integer absolute value is written through unsigned arithmetic because  */ \
  /* negating the most negative value is undefined in C, and the wrap it    */ \
  /* produces is the answer the Go reference gives.                         */ \
  void simd_l1norm_##SUF(T *__restrict out, const T *__restrict a,          \
                         isize n) {                                          \
    T s = 0;                                                                \
    for (isize i = 0; i < n; i++) {                                         \
      T v = a[i];                                                           \
      s += v < 0 ? (T)(0u - (unsigned long long)v) : v;                     \
    }                                                                        \
    *out = s;                                                               \
  }                                                                          \
  void simd_l1diff_##SUF(T *__restrict out, const T *__restrict a,          \
                         const T *__restrict b, isize n) {                   \
    T s = 0;                                                                \
    for (isize i = 0; i < n; i++) {                                         \
      T v = (T)((unsigned long long)a[i] - (unsigned long long)b[i]);       \
      s += v < 0 ? (T)(0u - (unsigned long long)v) : v;                     \
    }                                                                        \
    *out = s;                                                               \
  }

INT_REDUCTIONS(int, i32)
INT_REDUCTIONS(long long, i64)

// Successive differences. Unlike the running totals it is not a scan — each
// output depends only on two neighbouring inputs — so it vectorizes.
//
// It takes both lengths because the output is one shorter than the input, so
// neither length alone bounds the loop: nd elements are written but a[i+1]
// must stay inside na.
#define DIFF(T, SUF)                                                      \
  void simd_diff_##SUF(T *__restrict d, const T *__restrict a, isize nd,  \
                       isize na) {                                        \
    for (isize i = 0; i < nd && i + 1 < na; i++) d[i] = a[i + 1] - a[i];  \
  }

DIFF(float, f32)
DIFF(double, f64)
DIFF(int, i32)
DIFF(long long, i64)
