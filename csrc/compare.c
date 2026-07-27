// Comparison, selection and boolean-vector kernels.
//
// A comparison on a vector unit produces a mask register, one lane per
// element. Go's []bool is the portable spelling of that — one byte per
// element, which is what storing such a mask gives you — so these write
// _Bool, whose size C guarantees to be 1 and which matches Go's bool exactly.
//
// Float comparisons follow IEEE 754 without exception, which has one
// consequence worth restating: every comparison involving NaN is false, so
// NotEqual is not the negation of Equal. Writing one as `!other` would be
// wrong, so each is generated from its own operator.

#include "goabi.h"
#include "fold.h"

// ST is the scalar transport type, the same convention as csrc/arith.c: the
// element type for anything as wide as an int, a 64-bit integer for anything
// narrower, so that no ABI's rule for extending a short argument can apply.
#define CMP(NAME, T, ST, SUF, OP)                                        \
  void simd_##NAME##_##SUF(_Bool *__restrict d, const T *__restrict a,   \
                           const T *__restrict b, isize n) {             \
    for (isize i = 0; i < n; i++) d[i] = a[i] OP b[i];                   \
  }                                                                      \
  void simd_##NAME##s_##SUF(_Bool *__restrict d, const T *__restrict a,  \
                            ST v_, isize n) {                            \
    T v = (T)v_;                                                         \
    for (isize i = 0; i < n; i++) d[i] = a[i] OP v;                      \
  }

#define COMPARISONS(T, ST, SUF)                                          \
  CMP(eq, T, ST, SUF, ==)                                                \
  CMP(ne, T, ST, SUF, !=)                                                \
  CMP(lt, T, ST, SUF, <)                                                 \
  CMP(le, T, ST, SUF, <=)                                                \
  CMP(gt, T, ST, SUF, >)                                                 \
  CMP(ge, T, ST, SUF, >=)                                                \
  void simd_select_##SUF(T *__restrict d, const _Bool *__restrict m,     \
                         const T *__restrict yes, const T *__restrict no, \
                         isize n) {                                      \
    /* Both operands are loaded into locals before the select. Written  */ \
    /* as d[i] = m[i] ? yes[i] : no[i], LLVM selects between the two    */ \
    /* base *pointers* and then loads once — csel x11, x2, x3 followed  */ \
    /* by an indexed load — which is a good scalar strength reduction   */ \
    /* and completely unvectorizable. Making both loads unconditionally */ \
    /* live leaves a lane-wise blend, which is one instruction on every */ \
    /* vector unit here.                                                */ \
    for (isize i = 0; i < n; i++) {                                      \
      T y = yes[i], z = no[i];                                           \
      d[i] = m[i] ? y : z;                                               \
    }                                                                    \
  }

COMPARISONS(float, float, f32)
COMPARISONS(double, double, f64)
COMPARISONS(int, int, i32)
COMPARISONS(long long, long long, i64)

// The narrow and unsigned types. Unsigned comparison falls out of the operand
// type, so the same macro is right for both signedness classes — and getting
// it from the type rather than from a choice of operator is the point: a
// hand-written unsigned kernel that compares as signed is wrong only for the
// top half of the range, which no ordinary test input reaches.
COMPARISONS(signed char, long long, i8)
COMPARISONS(short, long long, i16)
COMPARISONS(unsigned char, unsigned long long, u8)
COMPARISONS(unsigned short, unsigned long long, u16)
COMPARISONS(unsigned int, unsigned long long, u32)
COMPARISONS(unsigned long long, unsigned long long, u64)

// ---------- boolean vectors ----------
//
// All and Any are searches, and the reductions in fold.h are what makes them
// vector code without losing the early exit that makes a search worth doing.
// See the comment there for why neither the pure accumulate nor the pure
// early exit is the right shape.

// simd_mask_all asks whether any element is false, which is the same question
// OR_ESCAPE answers, applied to the negation. Phrasing it that way gets the
// early exit: an all-true mask is the only case that has to read everything.
void simd_mask_all(_Bool *__restrict out, const _Bool *__restrict m, isize n) {
  OR_ESCAPE(VLOAD(m, q) ^ (unsigned char)1, 1u - (unsigned char)m[p])
  *out = !hit;
}

void simd_mask_any(_Bool *__restrict out, const _Bool *__restrict m, isize n) {
  OR_ESCAPE(VLOAD(m, q), (unsigned char)m[p])
  *out = hit;
}

// A count has to see every element, so there is no escape here.
void simd_mask_count(isize *__restrict out, const _Bool *__restrict m,
                     isize n) {
  COUNT_BYTES((unsigned char)m[p])
}

#define MASK_BIN(NAME, EXPR)                                             \
  void simd_mask_##NAME(_Bool *__restrict d, const _Bool *__restrict a,  \
                        const _Bool *__restrict b, isize n) {            \
    for (isize i = 0; i < n; i++) {                                      \
      unsigned char x = (unsigned char)a[i], y = (unsigned char)b[i];    \
      d[i] = (_Bool)(EXPR);                                              \
    }                                                                    \
  }

MASK_BIN(and, x &y)
MASK_BIN(or, x | y)
MASK_BIN(xor, x ^ y)

void simd_mask_not(_Bool *__restrict d, const _Bool *__restrict a, isize n) {
  for (isize i = 0; i < n; i++) d[i] = (_Bool)(1u - (unsigned char)a[i]);
}
