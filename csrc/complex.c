// Complex arithmetic.
//
// Go stores a complex as its two components adjacent in memory, so a slice of
// them arrives here as an interleaved array of twice as many reals. These
// kernels read it that way — the pointer is cast to the component type and
// the element at index i occupies positions 2i and 2i+1.
//
// Interleaved is not the layout that multiplies best. A complex product wants
// the real and imaginary parts in separate registers, and getting them there
// from interleaved data costs a deinterleave. LLVM knows this shape and
// generates it: the loop below becomes a strided load pair, the arithmetic,
// and an interleaved store. The alternative — asking callers to keep two
// separate arrays — would vectorize better and would not be Go's complex
// type, so the shuffle is the price of taking what the caller already has.
//
// Add, Sub and Neg do not care about any of that. They are elementwise on the
// components, so they are simply real kernels over 2n elements and vectorize
// exactly as well as the real ones do.

#include "goabi.h"

typedef long isize;

// ---------- component-wise: the ones that need no deinterleave ----------

#define CBINARY(NAME, T, SUF, OP)                                        \
  void simd_c##NAME##_##SUF(T *__restrict d, const T *__restrict a,      \
                            const T *__restrict b, isize n) {            \
    isize m = n * 2;                                                     \
    for (isize i = 0; i < m; i++) d[i] = a[i] OP b[i];                   \
  }

#define CUNARY(NAME, T, SUF, EXPR)                                       \
  void simd_c##NAME##_##SUF(T *__restrict d, const T *__restrict a,      \
                            isize n) {                                   \
    isize m = n * 2;                                                     \
    for (isize i = 0; i < m; i++) {                                      \
      T x = a[i];                                                        \
      d[i] = (EXPR);                                                     \
    }                                                                    \
  }

// ---------- the ones that do ----------

#define CMUL(T, SUF)                                                     \
  void simd_cmul_##SUF(T *__restrict d, const T *__restrict a,           \
                       const T *__restrict b, isize n) {                 \
    for (isize i = 0; i < n; i++) {                                      \
      T ar = a[2 * i], ai = a[2 * i + 1];                                \
      T br = b[2 * i], bi = b[2 * i + 1];                                \
      d[2 * i] = ar * br - ai * bi;                                      \
      d[2 * i + 1] = ar * bi + ai * br;                                  \
    }                                                                    \
  }

// Smith's method. The textbook formula divides by br*br + bi*bi, which
// overflows for operands whose squares do not fit even when the quotient is
// perfectly representable; dividing through by the larger component first
// keeps every intermediate in range.
#define CDIV(T, SUF, ABS)                                                \
  void simd_cdiv_##SUF(T *__restrict d, const T *__restrict a,           \
                       const T *__restrict b, isize n) {                 \
    for (isize i = 0; i < n; i++) {                                      \
      T ar = a[2 * i], ai = a[2 * i + 1];                                \
      T br = b[2 * i], bi = b[2 * i + 1];                                \
      T re, im;                                                          \
      if (ABS(br) >= ABS(bi)) {                                          \
        T r = bi / br;                                                   \
        T den = br + r * bi;                                             \
        re = (ar + ai * r) / den;                                        \
        im = (ai - ar * r) / den;                                        \
      } else {                                                           \
        T r = br / bi;                                                   \
        T den = bi + r * br;                                             \
        re = (ar * r + ai) / den;                                        \
        im = (ai * r - ar) / den;                                        \
      }                                                                  \
      d[2 * i] = re;                                                     \
      d[2 * i + 1] = im;                                                 \
    }                                                                    \
  }

#define CCONJ(T, SUF)                                                    \
  void simd_cconj_##SUF(T *__restrict d, const T *__restrict a,          \
                        isize n) {                                       \
    for (isize i = 0; i < n; i++) {                                      \
      d[2 * i] = a[2 * i];                                               \
      d[2 * i + 1] = -a[2 * i + 1];                                      \
    }                                                                    \
  }

#define CSCALE(T, SUF)                                                   \
  void simd_cscale_##SUF(T *__restrict d, const T *__restrict a, T s,    \
                         isize n) {                                      \
    isize m = n * 2;                                                     \
    for (isize i = 0; i < m; i++) d[i] = a[i] * s;                       \
  }

// The magnitude, through the larger component so that an intermediate cannot
// overflow for a representable answer — the same reasoning as Hypot, and it
// matters more here because both components come straight from the caller.
// The predicate is "im > re", not "re > im", and the difference is NaN.
// Both are false when either operand is NaN, so the first form falls through
// to the *imaginary* part and a NaN real component with a zero imaginary one
// gives a magnitude of zero. The second falls through to the real part and
// keeps the NaN, which is what the reference does and what the answer is.
//
// The infinity check is separate and last because it wins outright: the
// magnitude of a complex number with an infinite component is infinite even
// when the other component is NaN, and the general formula would give NaN.
#define CABS(T, SUF, ABS, INF)                                           \
  void simd_cabs_##SUF(T *__restrict d, const T *__restrict a, isize n) { \
    for (isize i = 0; i < n; i++) {                                      \
      T re = ABS(a[2 * i]), im = ABS(a[2 * i + 1]);                      \
      T hi = (im > re) ? im : re;                                        \
      T lo = (im > re) ? re : im;                                        \
      T t = lo / hi;                                                     \
      T v = hi * __builtin_elementwise_sqrt(1 + t * t);                  \
      v = (hi == 0) ? 0 : v;                                             \
      v = (re == INF || im == INF) ? INF : v;                            \
      d[i] = v;                                                          \
    }                                                                    \
  }

#define CPART(NAME, T, SUF, OFF)                                         \
  void simd_c##NAME##_##SUF(T *__restrict d, const T *__restrict a,      \
                            isize n) {                                   \
    for (isize i = 0; i < n; i++) d[i] = a[2 * i + OFF];                 \
  }

#define CFROMPARTS(T, SUF)                                               \
  void simd_cfromparts_##SUF(T *__restrict d, const T *__restrict re,    \
                             const T *__restrict im, isize n) {          \
    for (isize i = 0; i < n; i++) {                                      \
      d[2 * i] = re[i];                                                  \
      d[2 * i + 1] = im[i];                                              \
    }                                                                    \
  }

#define COMPLEX_SET(T, SUF, ABS, INF)                                         \
  CBINARY(add, T, SUF, +)                                                \
  CBINARY(sub, T, SUF, -)                                                \
  CUNARY(neg, T, SUF, -x)                                                \
  CMUL(T, SUF)                                                           \
  CDIV(T, SUF, ABS)                                                      \
  CCONJ(T, SUF)                                                          \
  CSCALE(T, SUF)                                                         \
  CABS(T, SUF, ABS, INF)                                                      \
  CPART(real, T, SUF, 0)                                                 \
  CPART(imag, T, SUF, 1)                                                 \
  CFROMPARTS(T, SUF)

COMPLEX_SET(float, c64, __builtin_elementwise_abs, __builtin_inff())
COMPLEX_SET(double, c128, __builtin_elementwise_abs, __builtin_inf())

// ---------- complex reductions ----------
//
// Sum, the bilinear dot and the Hermitian dot. These are the three complex
// operations with a numerical contract, for the same reason the real ones in
// reduce.c have it: a reduction must give the same bits on a 128-bit machine
// and a 512-bit one, so the lane an element lands in and the tree that folds
// the lanes are both fixed rather than left to the compiler.
//
// The contract is applied component by component. Sixteen independent real
// accumulators and sixteen imaginary ones, element k contributing to lane
// k%16, folded by the binary tree kernel.CombineTree defines. That is exactly
// what internal/ref/complex.go does.
//
// # Why the sixteen accumulators are two vectors of eight and not one of
// sixteen
//
// The obvious spelling is a single sixteen-lane vector per component. For
// complex64 that is 64 bytes and fine. For complex128 it is 128 bytes, and the
// inner loop needs six of them live at once — two accumulators and the four
// deinterleaved operands — which is 768 bytes of vector state. On AVX-512 that
// is twelve of the thirty-two zmm registers before any product is formed, and
// measured 1120 bytes of spill against a 512-byte NOSPLIT budget: the kernel
// was dropped on every amd64 tier.
//
// Splitting each accumulator into a low half and a high half halves the widest
// live value without changing the arithmetic at all. Lane k%16 still receives
// element k — the first eight of a block go to the low vector and the next
// eight to the high one — and the first step of the fold, r[j] += r[j+8], is
// then just adding the two halves, which the tree wanted anyway.
//
// # The deinterleave
//
// A Go []complex64 is real and imaginary components alternating in one
// allocation, so a run of eight complex values is two vector loads and the
// components have to be separated before they can be multiplied. That
// separation must be a shuffle. Written as an indexed copy it becomes sixteen
// inserts per load, which is the trap scan.c documents at length.
//
// # No FMA
//
// Every target here is compiled with -ffp-contract=off, and it has to be: the
// reference forms each product and rounds it before the addition, so a fused
// multiply-add would carry extra precision through and disagree in the last
// bit.
#define CSUM_HALF 8

typedef float f32xH __attribute__((ext_vector_type(CSUM_HALF), aligned(1)));
typedef float f32xQ __attribute__((ext_vector_type(4), aligned(1)));
typedef float f32xP __attribute__((ext_vector_type(2), aligned(1)));
typedef double f64xH __attribute__((ext_vector_type(CSUM_HALF), aligned(1)));
typedef double f64xQ __attribute__((ext_vector_type(4), aligned(1)));
typedef double f64xP __attribute__((ext_vector_type(2), aligned(1)));

// The even lanes of two adjacent vectors are the real components of eight
// complex values and the odd lanes are the imaginary ones.
#define CDE_RE(v0, v1)                                                   \
  __builtin_shufflevector(v0, v1, 0, 2, 4, 6, 8, 10, 12, 14)
#define CDE_IM(v0, v1)                                                   \
  __builtin_shufflevector(v0, v1, 1, 3, 5, 7, 9, 11, 13, 15)

// CLOAD deinterleaves the eight complex values at p+off into RE and IM.
#define CLOAD(V, p, off, RE, IM)                                         \
  V RE, IM;                                                              \
  {                                                                      \
    V v0 = *(const V *)((p) + 2 * (off));                                \
    V v1 = *(const V *)((p) + 2 * (off) + CSUM_HALF);                    \
    RE = CDE_RE(v0, v1);                                                 \
    IM = CDE_IM(v0, v1);                                                 \
  }

// The fold. Adding the high half to the low half is the w=8 step of
// CombineTree, and the remaining steps halve the vector each time.
#define CFOLD(V, VQ, VP, relo, rehi, imlo, imhi)                         \
  {                                                                      \
    V r8 = (relo) + (rehi), i8 = (imlo) + (imhi);                        \
    VQ r4 = __builtin_shufflevector(r8, r8, 0, 1, 2, 3) +                \
            __builtin_shufflevector(r8, r8, 4, 5, 6, 7);                 \
    VQ i4 = __builtin_shufflevector(i8, i8, 0, 1, 2, 3) +                \
            __builtin_shufflevector(i8, i8, 4, 5, 6, 7);                 \
    VP r2 = __builtin_shufflevector(r4, r4, 0, 1) +                      \
            __builtin_shufflevector(r4, r4, 2, 3);                       \
    VP i2 = __builtin_shufflevector(i4, i4, 0, 1) +                      \
            __builtin_shufflevector(i4, i4, 2, 3);                       \
    out[0] = r2[0] + r2[1];                                              \
    out[1] = i2[0] + i2[1];                                              \
  }

#define CSUM(T, V, VQ, VP, SUF)                                          \
  void simd_csum_##SUF(T *__restrict out, const T *__restrict a,         \
                       isize n) {                                        \
    V relo = 0, rehi = 0, imlo = 0, imhi = 0;                            \
    isize i = 0;                                                         \
    for (; i + 2 * CSUM_HALF <= n; i += 2 * CSUM_HALF) {                 \
      CLOAD(V, a, i, ar0, ai0)                                           \
      CLOAD(V, a, i + CSUM_HALF, ar1, ai1)                               \
      relo += ar0;                                                       \
      imlo += ai0;                                                       \
      rehi += ar1;                                                       \
      imhi += ai1;                                                       \
    }                                                                    \
    V tre = 0, tim = 0, ure = 0, uim = 0;                                \
    for (int j = 0; j < CSUM_HALF; j++) {                                \
      if (i + j < n) {                                                   \
        tre[j] = a[2 * (i + j)];                                         \
        tim[j] = a[2 * (i + j) + 1];                                     \
      }                                                                  \
      isize k = i + CSUM_HALF + j;                                       \
      if (k < n) {                                                       \
        ure[j] = a[2 * k];                                               \
        uim[j] = a[2 * k + 1];                                           \
      }                                                                  \
    }                                                                    \
    relo += tre;                                                         \
    imlo += tim;                                                         \
    rehi += ure;                                                         \
    imhi += uim;                                                         \
    CFOLD(V, VQ, VP, relo, rehi, imlo, imhi)                             \
  }

// OPRE and OPIM are the two signs that separate the bilinear product from the
// Hermitian one. Conjugating a negates its imaginary component, which flips
// the sign of the ai*bi term in the real part and of the ai*br term in the
// imaginary part — so the two kernels are one body with two operators rather
// than a runtime flag, which would cost a select in the inner loop.
#define CDOT_HALF(V, OFF, OPRE, OPIM, ACCRE, ACCIM)                      \
  {                                                                      \
    CLOAD(V, a, OFF, ar, ai)                                             \
    CLOAD(V, b, OFF, br, bi)                                             \
    ACCRE += ar * br OPRE ai * bi;                                       \
    ACCIM += ar * bi OPIM ai * br;                                       \
  }

#define CDOT_TAIL(T, V, LIM, BASE, OPRE, OPIM, ACCRE, ACCIM)             \
  {                                                                      \
    V tre = 0, tim = 0;                                                  \
    for (int j = 0; j < CSUM_HALF; j++) {                                \
      isize k = (BASE) + j;                                              \
      if (k < (LIM)) {                                                   \
        T ar = a[2 * k], ai = a[2 * k + 1];                              \
        T br = b[2 * k], bi = b[2 * k + 1];                              \
        tre[j] = ar * br OPRE ai * bi;                                   \
        tim[j] = ar * bi OPIM ai * br;                                   \
      }                                                                  \
    }                                                                    \
    ACCRE += tre;                                                        \
    ACCIM += tim;                                                        \
  }

#define CDOT(T, V, VQ, VP, SUF, NAME, OPRE, OPIM)                        \
  void simd_c##NAME##_##SUF(T *__restrict out, const T *__restrict a,    \
                            const T *__restrict b, isize n) {            \
    V relo = 0, rehi = 0, imlo = 0, imhi = 0;                            \
    isize i = 0;                                                         \
    for (; i + 2 * CSUM_HALF <= n; i += 2 * CSUM_HALF) {                 \
      CDOT_HALF(V, i, OPRE, OPIM, relo, imlo)                            \
      CDOT_HALF(V, i + CSUM_HALF, OPRE, OPIM, rehi, imhi)                \
    }                                                                    \
    CDOT_TAIL(T, V, n, i, OPRE, OPIM, relo, imlo)                        \
    CDOT_TAIL(T, V, n, i + CSUM_HALF, OPRE, OPIM, rehi, imhi)            \
    CFOLD(V, VQ, VP, relo, rehi, imlo, imhi)                             \
  }

CSUM(float, f32xH, f32xQ, f32xP, c64)
CSUM(double, f64xH, f64xQ, f64xP, c128)
CDOT(float, f32xH, f32xQ, f32xP, c64, dot, -, +)
CDOT(double, f64xH, f64xQ, f64xP, c128, dot, -, +)
CDOT(float, f32xH, f32xQ, f32xP, c64, dotconj, +, -)
CDOT(double, f64xH, f64xQ, f64xP, c128, dotconj, +, -)
