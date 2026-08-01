// BLAS-shaped kernels: the operations a linear algebra library reaches for that
// are not covered by the elementwise, reduction or matrix files.
//
// These exist because gonum's LU, QR, Cholesky and SVD spend their time in
// them. Matrix multiply and matrix-vector were already fast; the decompositions
// were not, because they are built out of rank-1 updates and Givens rotations
// rather than out of gemm.
//
// The rules from elementwise.c apply: no calls, __restrict everywhere, signed
// loop counters, and nothing that would make the result depend on the vector
// width.

#include "goabi.h"
#include "wrap.h"

typedef long isize;

// ---------- rank-1 update ----------
//
// a[i*n + j] += alpha * x[i] * y[j], which BLAS calls ger and which is the
// inner loop of an LU decomposition: eliminate a column, then update the
// trailing submatrix by the outer product of what was eliminated.
//
// The row scale is hoisted out of the inner loop deliberately. Writing
// alpha*x[i]*y[j] inside would be two multiplies per element and, worse, would
// leave the association up to the compiler — (alpha*x[i])*y[j] and
// alpha*(x[i]*y[j]) differ in the last place. Hoisting fixes both.
//
// The inner loop is then a plain axpy over a contiguous row, which is the shape
// every target vectorizes. lda is not a parameter: the caller guarantees rows
// are n apart, and anything else goes to the portable path rather than being
// handled with a stride nobody can prove is safe.
#define GER(T, ST, SUF)                                                 \
  void simd_ger_##SUF(T *__restrict a, const T *__restrict x,           \
                      const T *__restrict y, ST alpha_, isize m,        \
                      isize n) {                                        \
    T alpha = (T)alpha_;                                                \
    for (isize i = 0; i < m; i++) {                                     \
      T s = alpha * x[i];                                               \
      T *__restrict row = a + i * n;                                    \
      for (isize j = 0; j < n; j++) row[j] += s * y[j];                 \
    }                                                                   \
  }

// ---------- Givens rotation ----------
//
// Applied to a pair of vectors:
//
//   x[i] =  c*x[i] + s*y[i]
//   y[i] =  c*y[i] - s*x[i]
//
// using the ORIGINAL x[i] in both, which is why the temporary is not optional.
// This is BLAS drot, and QR and the SVD's bidiagonal phase are made of it.
//
// Both slices are read and written, so they cannot be __restrict against each
// other in the usual way — but BLAS says the vectors do not overlap, and the
// portable path is what runs if a caller violates that.
#define ROT(T, ST, SUF)                                                 \
  void simd_rot_##SUF(T *__restrict x, T *__restrict y, ST c_, ST s_,   \
                      isize n) {                                        \
    T c = (T)c_, s = (T)s_;                                             \
    for (isize i = 0; i < n; i++) {                                     \
      T xi = x[i], yi = y[i];                                           \
      x[i] = c * xi + s * yi;                                           \
      y[i] = c * yi - s * xi;                                           \
    }                                                                   \
  }

// ---------- swap ----------
//
// Exchange two vectors. Memory-bound and trivial, but it is the third most
// called routine in gonum's decompositions — pivoting swaps rows — and a
// scalar loop moves eight bytes at a time where a vector register moves
// sixty-four.
#define SWAP(T, SUF)                                                    \
  void simd_swap_##SUF(T *__restrict x, T *__restrict y, isize n) {     \
    for (isize i = 0; i < n; i++) {                                     \
      T t = x[i];                                                       \
      x[i] = y[i];                                                      \
      y[i] = t;                                                         \
    }                                                                   \
  }

GER(float, float, f32)
GER(double, double, f64)
ROT(float, float, f32)
ROT(double, double, f64)
SWAP(float, f32)
SWAP(double, f64)
SWAP(int, i32)
SWAP(long long, i64)
