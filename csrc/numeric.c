// Numeric kernels that are neither plain elementwise arithmetic nor plain
// reductions: the ones whose shape needed thinking about before they would
// vectorize.
//
// What is here and what is not is a deliberate line. A kernel earns its place
// by having a loop body that is independent across iterations, or a reduction
// whose combining operation is associative. Everything on the other side of
// that line stays in Go:
//
//   - EMA is a first-order recurrence, prev = a*x + (1-a)*prev. Each output
//     needs the one before it. There is a parallel-prefix formulation, but it
//     changes the arithmetic and so the answer, which rule 1 forbids.
//   - Median and Quantile sort. A vectorized sort is a different project.
//   - Ramp needs the constant vector [0,1,2,...], which on every architecture
//     but amd64 is reached through a high/low instruction pair.
//   - Reverse is a permutation that LLVM will not derive from a plain loop.
//
// The float product is absent for a numerical reason rather than a mechanical
// one: products overflow and underflow far more readily than sums, so
// reassociating one changes which intermediate blows up rather than merely
// the rounding.

#include "goabi.h"

typedef long isize;

// ---------- norms ----------
//
// Norm is the Euclidean length. It reuses the sum-of-squares shape rather than
// calling into it, because the generator copies a function's bytes and a call
// would be a call. The accumulator discipline is the same as reduce.c's and
// for the same reason: the answer must not depend on the vector width.

#define SUM_LANES 16

typedef float f32xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));
typedef double f64xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));

#define NORM(T, V)                                                       \
  V acc = 0;                                                             \
  isize i = 0;                                                           \
  for (; i + SUM_LANES <= n; i += SUM_LANES) {                           \
    V v = *(const V *)(a + i);                                           \
    acc += v * v;                                                        \
  }                                                                      \
  V t = 0;                                                               \
  for (int j = 0; j < SUM_LANES; j++)                                    \
    if (i + j < n) t[j] = a[i + j] * a[i + j];                           \
  acc += t;                                                              \
  T r[SUM_LANES];                                                        \
  *(V *)r = acc;                                                         \
  for (int w = SUM_LANES / 2; w >= 1; w /= 2)                            \
    for (int j = 0; j < w; j++) r[j] += r[j + w];                        \
  *out = __builtin_elementwise_sqrt(r[0]);

void simd_norm_f32(float *__restrict out, const float *__restrict a, isize n) {
  NORM(float, f32xL)
}

void simd_norm_f64(double *__restrict out, const double *__restrict a,
                   isize n) {
  NORM(double, f64xL)
}

// ---------- polynomial evaluation ----------
//
// dst[i] = coeffs[0] + x[i]*(coeffs[1] + x[i]*(...)), one polynomial applied
// to every element. The outer loop over elements is independent, which is
// what vectorizes; the inner loop over coefficients is a fixed dependency
// chain per element, exactly like the transcendentals in math.c.
//
// The accumulator is per element rather than per coefficient, so the whole
// polynomial is evaluated for a register's worth of x at a time.
#define POLY_EVAL(T)                                                     \
  isize n = nd < nx ? nd : nx;                                           \
  if (nc == 0) {                                                         \
    for (isize i = 0; i < n; i++) d[i] = 0;                              \
    return;                                                              \
  }                                                                      \
  for (isize i = 0; i < n; i++) {                                        \
    T xv = x[i];                                                         \
    T acc = c[nc - 1];                                                   \
    for (isize k = nc - 2; k >= 0; k--) acc = acc * xv + c[k];           \
    d[i] = acc;                                                          \
  }

void simd_polyeval_f32(float *__restrict d, const float *__restrict x,
                       const float *__restrict c, isize nd, isize nx,
                       isize nc) {
  POLY_EVAL(float)
}

void simd_polyeval_f64(double *__restrict d, const double *__restrict x,
                       const double *__restrict c, isize nd, isize nx,
                       isize nc) {
  POLY_EVAL(double)
}

// ---------- convolution and correlation ----------
//
// Direct form. dst[i] is a dot product of a window of the signal with the
// kernel, and windows do not overlap in their outputs, so the outer loop is
// independent and the inner one is a short reduction the compiler unrolls.
//
// An FFT would be asymptotically better and is a different kernel; this is
// the form that wins for the short kernels callers actually convolve with.
#define CONVOLVE(T, IDX)                                                 \
  isize m = nk;                                                          \
  if (m == 0 || n < m) return;                                           \
  isize outn = n - m + 1;                                                \
  if (nd < outn) outn = nd;                                              \
  for (isize i = 0; i < outn; i++) {                                     \
    T acc = 0;                                                           \
    for (isize j = 0; j < m; j++) acc += s[i + j] * k[IDX];              \
    d[i] = acc;                                                          \
  }

void simd_convolve_f32(float *__restrict d, const float *__restrict s,
                       const float *__restrict k, isize nd, isize n,
                       isize nk) {
  CONVOLVE(float, m - 1 - j)
}

void simd_convolve_f64(double *__restrict d, const double *__restrict s,
                       const double *__restrict k, isize nd, isize n,
                       isize nk) {
  CONVOLVE(double, m - 1 - j)
}

void simd_correlate_f32(float *__restrict d, const float *__restrict s,
                        const float *__restrict k, isize nd, isize n,
                        isize nk) {
  CONVOLVE(float, j)
}

void simd_correlate_f64(double *__restrict d, const double *__restrict s,
                        const double *__restrict k, isize nd, isize n,
                        isize nk) {
  CONVOLVE(double, j)
}

// ---------- tile ----------
//
// Repeat a pattern across the destination. The copy is independent per
// element; the modulo is what would stop it vectorizing, so the pattern index
// is carried as a counter that wraps rather than recomputed.
#define TILE(T)                                                          \
  if (np == 0) return;                                                   \
  isize p = 0;                                                           \
  for (isize i = 0; i < nd; i++) {                                       \
    d[i] = s[p];                                                         \
    p++;                                                                 \
    if (p == np) p = 0;                                                  \
  }

void simd_tile_f32(float *__restrict d, const float *__restrict s, isize nd,
                   isize np) {
  TILE(float)
}

void simd_tile_f64(double *__restrict d, const double *__restrict s, isize nd,
                   isize np) {
  TILE(double)
}

void simd_tile_i32(int *__restrict d, const int *__restrict s, isize nd,
                   isize np) {
  TILE(int)
}

void simd_tile_i64(long long *__restrict d, const long long *__restrict s,
                   isize nd, isize np) {
  TILE(long long)
}

// ---------- gather and scatter ----------
//
// An indexed load is a single instruction on AVX2, AVX-512 and SVE2 and a
// loop everywhere else, which is exactly what the compiler will decide for
// itself given a plain indexed loop. Out-of-range indices are skipped rather
// than clamped, matching the reference: a gather with a bad index is a
// caller's bug and silently reading the wrong element would hide it.
// An out-of-range index leaves the destination element alone rather than
// zeroing it, which is what the reference does. Writing a zero would be a
// different function, and a caller checking for untouched output would never
// see the bad index.
#define GATHER(T)                                                        \
  isize n = nd < ni ? nd : ni;                                           \
  for (isize i = 0; i < n; i++) {                                        \
    int ix = idx[i];                                                     \
    if (ix >= 0 && (isize)ix < ns) d[i] = s[ix];                         \
  }

void simd_gather_f32(float *__restrict d, const float *__restrict s,
                     const int *__restrict idx, isize nd, isize ni, isize ns) {
  GATHER(float)
}

void simd_gather_f64(double *__restrict d, const double *__restrict s,
                     const int *__restrict idx, isize nd, isize ni, isize ns) {
  GATHER(double)
}

void simd_gather_i32(int *__restrict d, const int *__restrict s,
                     const int *__restrict idx, isize nd, isize ni, isize ns) {
  GATHER(int)
}

void simd_gather_i64(long long *__restrict d, const long long *__restrict s,
                     const int *__restrict idx, isize nd, isize ni, isize ns) {
  GATHER(long long)
}

// ---------- scatter ----------
//
// The mirror of gather: an indexed store. Out-of-range indices are skipped,
// and where two indices collide the later one wins, which is what a
// sequential loop gives and what the reference does. That ordering is why
// this cannot become a true vector scatter without care — hardware scatter
// with duplicate indices has an implementation-defined winner — so the plain
// loop is deliberate and the compiler may vectorize it only when it can prove
// the indices distinct.
#define SCATTER(T)                                                       \
  isize n = ni < ns ? ni : ns;                                           \
  for (isize i = 0; i < n; i++) {                                        \
    int ix = idx[i];                                                     \
    if (ix >= 0 && (isize)ix < nd) d[ix] = s[i];                         \
  }

void simd_scatter_f32(float *__restrict d, const int *__restrict idx,
                      const float *__restrict s, isize nd, isize ni,
                      isize ns) {
  SCATTER(float)
}

void simd_scatter_f64(double *__restrict d, const int *__restrict idx,
                      const double *__restrict s, isize nd, isize ni,
                      isize ns) {
  SCATTER(double)
}

void simd_scatter_i32(int *__restrict d, const int *__restrict idx,
                      const int *__restrict s, isize nd, isize ni, isize ns) {
  SCATTER(int)
}

void simd_scatter_i64(long long *__restrict d, const int *__restrict idx,
                      const long long *__restrict s, isize nd, isize ni,
                      isize ns) {
  SCATTER(long long)
}

// ---------- moving average ----------
//
// Each output is the mean of a window, and the windows overlap. A running sum
// would be O(n) instead of O(n*width), but it is a recurrence and it changes
// the arithmetic: subtracting the departing element accumulates a different
// rounding error than summing the window afresh. The reference sums each
// window, so this does too, and the outer loop over windows is what
// vectorizes.
#define MOVING_AVERAGE(T)                                                \
  if (w <= 0 || na < w) return;                                          \
  isize outn = na - w + 1;                                               \
  if (nd < outn) outn = nd;                                              \
  for (isize i = 0; i < outn; i++) {                                     \
    T acc = 0;                                                           \
    for (isize j = 0; j < w; j++) acc += a[i + j];                       \
    d[i] = acc / (T)w;                                                   \
  }

void simd_movavg_f32(float *__restrict d, const float *__restrict a, isize nd,
                     isize na, isize w) {
  MOVING_AVERAGE(float)
}

void simd_movavg_f64(double *__restrict d, const double *__restrict a,
                     isize nd, isize na, isize w) {
  MOVING_AVERAGE(double)
}

// ---------- matrix multiply ----------
//
// Row-major, dst[m*n] = a[m*k] * b[k*n]. The loop order is i, p, j rather
// than the textbook i, j, p: with j innermost, both b and dst are walked
// contiguously and the element of a is a scalar broadcast, which is the shape
// that vectorizes. The textbook order makes the inner loop a dot product with
// a strided access to b, which does not.
//
// The zero-scalar skip matters more than it looks: it is the reference's
// behaviour, and on a sparse a it is most of the work.
// The size checks are the caller's: the generated guard sends a badly sized
// call to the portable path, which is what keeps this to six arguments where
// System V has six integer argument registers. See matMulK in the manifest.
#define MATMUL(T)                                                        \
  for (isize i = 0; i < m * n; i++) d[i] = 0;                            \
  for (isize i = 0; i < m; i++) {                                        \
    for (isize p = 0; p < k; p++) {                                      \
      T s = a[i * k + p];                                                \
      if (s == 0) continue;                                              \
      const T *br = b + p * n;                                           \
      T *row = d + i * n;                                                \
      for (isize j = 0; j < n; j++) row[j] += s * br[j];                 \
    }                                                                    \
  }

void simd_matmul_f32(float *__restrict d, const float *__restrict a,
                     const float *__restrict b, isize m, isize k, isize n) {
  MATMUL(float)
}

void simd_matmul_f64(double *__restrict d, const double *__restrict a,
                     const double *__restrict b, isize m, isize k, isize n) {
  MATMUL(double)
}
