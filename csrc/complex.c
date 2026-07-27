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
