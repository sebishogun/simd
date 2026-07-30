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

// ---------- batch lower bound ----------
//
// The first index at or after which a sorted slice is >= the query, for many
// queries at once. std::lower_bound, sort.SearchInts, the bucket lookup in a
// histogram or a quantile table.
//
// # One query does not vectorize; many do
//
// A single binary search is log2(n) dependent steps: each probe's address comes
// from the previous comparison. Nothing in a vector unit helps with that, and a
// branchless scalar search is already close to optimal for one query.
//
// A *batch* is a different problem. Every query walks the same number of steps
// over the same array, so the loop nest can be turned inside out: step on the
// outside, query on the inside. The inner loop is then elementwise over the
// batch — one probe index per lane, one comparison, one masked update — and the
// only thing it needs that plain arithmetic does not is a gather, because each
// lane reads a different element of the table.
//
// That is the whole cost and the whole limit. Where a gather instruction exists
// this is a real vectorization; where it does not, LLVM declines and the kernel
// is dropped for that target, which is the same wall the scatter family hit and
// is recorded in docs/wrong.md entry 59.
//
// # Shar's algorithm, so every query takes the same steps
//
// The textbook lo/hi bisection has a trip count that depends on where the
// answer is, which would make the lanes disagree about when to stop. Stepping
// down through the powers of two instead takes exactly floor(log2(n))+1 steps
// for every query regardless of the answer, so the outer loop is uniform and
// the inner one has nothing to diverge on.
//
// The running position is kept in dst as "one less than the answer", starting
// at -1, which is what makes the update a single conditional add rather than a
// pair of bounds. It is corrected in a final elementwise pass.
#define LOWER_BOUND(T, SUF)                                                \
  void simd_lower_bound_##SUF(int *__restrict d, const T *__restrict s,    \
                              isize ns, const T *__restrict q, isize nq,   \
                              isize ndst) {                                \
    if (nq > ndst) nq = ndst;                                              \
    for (isize i = 0; i < nq; i++) d[i] = -1;                              \
    isize step = 1;                                                        \
    while (step * 2 <= ns) step *= 2;                                      \
    for (; step > 0; step >>= 1) {                                         \
      for (isize i = 0; i < nq; i++) {                                     \
        isize probe = (isize)d[i] + step;                                  \
        int take = probe < ns && s[probe] < q[i];                          \
        d[i] = take ? (int)probe : d[i];                                   \
      }                                                                    \
    }                                                                      \
    for (isize i = 0; i < nq; i++) d[i] += 1;                              \
  }

LOWER_BOUND(float, f32)
LOWER_BOUND(double, f64)
LOWER_BOUND(signed char, i8)
LOWER_BOUND(short, i16)
LOWER_BOUND(int, i32)
LOWER_BOUND(long long, i64)
LOWER_BOUND(unsigned char, u8)
LOWER_BOUND(unsigned short, u16)
LOWER_BOUND(unsigned int, u32)
LOWER_BOUND(unsigned long long, u64)
