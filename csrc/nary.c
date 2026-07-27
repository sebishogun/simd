// Combining three or four slices in a single pass.
//
// The binary form of every operation in this library already exists, so an
// n-ary one looks redundant until you count memory traffic. Written as repeated
// binary calls, dst = a+b+c+d is three passes: read two, write one, then read
// that back and read a third, write again, and once more. Above the last level
// of cache that is three round trips to memory for arithmetic that is one
// instruction per element, and the arithmetic is not what costs.
//
// Done here it is one pass: every source read once, dst written once. The
// measured effect is roughly what the traffic count predicts and is largest
// exactly where it matters, on inputs too big to stay in cache.
//
// Arity stops at four for an ABI reason rather than a taste one. A five-source
// kernel needs seven pointer arguments plus a length, and System V passes six
// integers in registers; the seventh goes on the stack, which this generator
// does not emit prologue code for. Four sources covers essentially every real
// call, and the Go wrapper folds anything longer in groups of four, so a
// sixteen-way sum is four passes rather than fifteen.
//
// # Accumulation order
//
// Left to right, always: ((a+b)+c)+d. That is fixed rather than incidental,
// because floating-point addition is not associative and a caller must get the
// same bits from AddAll that they would get from writing the binary calls out
// by hand. The tests check exactly that, and it is also why the compiler is not
// permitted to reassociate — these are compiled with contraction and
// reassociation off like the rest of the library.

#include "goabi.h"

typedef long isize;

// NARY3 and NARY4 are the loop bodies. They are macros over the operator so
// that add and multiply cannot drift apart in the parenthesisation, which is
// the one thing here that is easy to get silently wrong.
#define NARY3(T, OP)                                                     \
  for (isize i = 0; i < n; i++) d[i] = (T)((T)(a[i] OP b[i]) OP c[i]);

#define NARY4(T, OP)                                                     \
  for (isize i = 0; i < n; i++)                                          \
    d[i] = (T)((T)((T)(a[i] OP b[i]) OP c[i]) OP e[i]);

#define NARY_DEFS(T, SUF)                                                \
  void simd_add3_##SUF(T *__restrict d, const T *__restrict a,           \
                       const T *__restrict b, const T *__restrict c,     \
                       isize n) {                                        \
    NARY3(T, +)                                                          \
  }                                                                      \
  void simd_add4_##SUF(T *__restrict d, const T *__restrict a,           \
                       const T *__restrict b, const T *__restrict c,     \
                       const T *__restrict e, isize n) {                 \
    NARY4(T, +)                                                          \
  }                                                                      \
  void simd_mul3_##SUF(T *__restrict d, const T *__restrict a,           \
                       const T *__restrict b, const T *__restrict c,     \
                       isize n) {                                        \
    NARY3(T, *)                                                          \
  }                                                                      \
  void simd_mul4_##SUF(T *__restrict d, const T *__restrict a,           \
                       const T *__restrict b, const T *__restrict c,     \
                       const T *__restrict e, isize n) {                 \
    NARY4(T, *)                                                          \
  }

NARY_DEFS(float, f32)
NARY_DEFS(double, f64)
NARY_DEFS(int, i32)
NARY_DEFS(long long, i64)
NARY_DEFS(signed char, i8)
NARY_DEFS(short, i16)
NARY_DEFS(unsigned char, u8)
NARY_DEFS(unsigned short, u16)
NARY_DEFS(unsigned int, u32)
NARY_DEFS(unsigned long long, u64)
