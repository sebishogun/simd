// IEEE 754-2019 minimum and maximum, elementwise, for scalars and vectors.
//
// These live in a header rather than in arith.c because more than one kernel
// family needs them and two copies would drift. They did: the prefix scans in
// scan.c were first written with __builtin_elementwise_min and _max, which are
// *not* this operation, and the differential tests caught it immediately —
// CumMin of a slice containing a NaN returned the neighbouring value instead of
// the NaN, and CumMin of {+0, -0} returned +0 instead of -0.
//
// The library promises IEEE 754-2019 minimum and maximum: NaN propagates, and
// -0 compares below +0. Almost nothing in hardware or in the compiler gives
// that for free:
//
//   - x86's MINPS returns its *second* operand when either is NaN.
//   - The minnum/maxnum family, which __builtin_elementwise_min lowers to,
//     deliberately returns the non-NaN operand — that is what "num" means.
//   - Neither family orders -0 below +0; they treat the two as equal and
//     return whichever the instruction happens to pick.
//
// So it is written out as compares and selects. LLVM turns this into exactly
// the compare-and-blend sequence a hand-written kernel would use, and it
// matches the portable reference bit for bit.
//
// One shape note carried over from arith.c: this is a single branchless select
// chain rather than a sequence of early returns. The two compute the same
// thing, but the early-return form leaves LLVM to if-convert five branches
// before it can vectorize, and on three targets it declined to for float64
// while managing it for float32.

#ifndef SIMD_MINMAX_H
#define SIMD_MINMAX_H

// The scalar forms. SIGNBIT is the builtin for the element width.
#define MINMAX_FLOAT(T, SUF, SIGNBIT)                              \
  static inline T min_##SUF(T x, T y) {                            \
    return (x != x)   ? x                                          \
           : (y != y) ? y                                          \
           : (x < y)  ? x                                          \
           : (y < x)  ? y                                          \
                      : (SIGNBIT(x) ? x : y); /* -0 wins */        \
  }                                                                \
  static inline T max_##SUF(T x, T y) {                            \
    return (x != x)   ? x                                          \
           : (y != y) ? y                                          \
           : (x > y)  ? x                                          \
           : (y > x)  ? y                                          \
                      : (SIGNBIT(x) ? y : x); /* +0 wins */        \
  }

#define MINMAX_INT(T, SUF)                                        \
  static inline T min_##SUF(T x, T y) { return x < y ? x : y; }   \
  static inline T max_##SUF(T x, T y) { return x > y ? x : y; }

// The vector forms.
//
// The select chain is identical; only the signbit test differs, because there
// is no elementwise __builtin_signbit. A cast between two ext_vector types of
// the same size is a bitcast rather than a conversion, so reinterpreting the
// float vector as a signed integer vector and testing for negative is exactly
// the sign bit — and it is the same trick fold.h uses to reinterpret byte
// lanes as 64-bit ones.
//
// IV must be a signed integer vector type with the same element width and lane
// count as V.
#define VMIN_FLOAT(V, IV, x, y)                                    \
  (((x) != (x))   ? (x)                                            \
   : ((y) != (y)) ? (y)                                            \
   : ((x) < (y))  ? (x)                                            \
   : ((y) < (x))  ? (y)                                            \
                  : ((((IV)(x)) < 0) ? (x) : (y)))

#define VMAX_FLOAT(V, IV, x, y)                                    \
  (((x) != (x))   ? (x)                                            \
   : ((y) != (y)) ? (y)                                            \
   : ((x) > (y))  ? (x)                                            \
   : ((y) > (x))  ? (y)                                            \
                  : ((((IV)(x)) < 0) ? (y) : (x)))

// Integer vectors need no special handling: the comparison already follows the
// operand type's signedness, and there is no NaN and no signed zero.
#define VMIN_INT(x, y) (((x) < (y)) ? (x) : (y))
#define VMAX_INT(x, y) (((x) > (y)) ? (x) : (y))

#endif // SIMD_MINMAX_H
