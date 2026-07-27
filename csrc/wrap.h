// Arithmetic that wraps, because Go's does and C's does not.
//
// Go defines signed integer overflow: the result wraps, and `math.MinInt32 - 1`
// is `math.MaxInt32` with no discussion. C leaves it undefined, and LLVM is
// entitled to assume it never happens — which is not a theoretical concern for
// a library whose whole contract is that the accelerated path and the portable
// path produce identical bits. An optimizer that decided `s + a[i]*b[i]` could
// not go negative would be within its rights and would silently break the one
// promise this package makes.
//
// The alternatives were measured. `-fwrapv` defines the behaviour globally and
// costs 11% of the vector instructions in reduce.c and 9% in numeric.c,
// because it also takes away the loop-induction reasoning the vectorizer needs.
// Doing the arithmetic in the unsigned domain instead — where C *defines* the
// wrap — costs nothing at all: byte-for-byte the same code, 16 vector
// instructions for an int32 add either way, and one instruction *fewer* for a
// dot product.
//
// So every operation that can overflow goes through these. They are written
// per type rather than as one generic macro because C has no generics, and
// they are in a header rather than in arith.c because reduce.c accumulates
// with the same rules and the two must not drift.
//
// Comparison, selection and clamping are absent deliberately: none of them can
// overflow, and routing a signed comparison through unsigned would invert it.

#ifndef SIMD_WRAP_H
#define SIMD_WRAP_H

// WRAPOPS_INT defines the three overflowing operations for an integer type,
// performed in UT. For an unsigned T, UT is T and the casts vanish; writing
// both signednesses the same way is what keeps the instantiation list below
// readable, and costs nothing.
#define WRAPOPS_INT(T, UT, SUF)                                          \
  static inline T add_##SUF(T x, T y) { return (T)((UT)x + (UT)y); }     \
  static inline T sub_##SUF(T x, T y) { return (T)((UT)x - (UT)y); }     \
  static inline T mul_##SUF(T x, T y) { return (T)((UT)x * (UT)y); }

// WRAPOPS_FLOAT is the same three names for a floating-point type, where the
// operators already mean what they say. Having them lets the kernel macros be
// written once for every element type instead of once per signedness class.
#define WRAPOPS_FLOAT(T, SUF)                                            \
  static inline T add_##SUF(T x, T y) { return x + y; }                  \
  static inline T sub_##SUF(T x, T y) { return x - y; }                  \
  static inline T mul_##SUF(T x, T y) { return x * y; }

WRAPOPS_FLOAT(float, f32)
WRAPOPS_FLOAT(double, f64)

WRAPOPS_INT(int, unsigned int, i32)
WRAPOPS_INT(long long, unsigned long long, i64)
WRAPOPS_INT(signed char, unsigned char, i8)
WRAPOPS_INT(short, unsigned short, i16)
WRAPOPS_INT(unsigned char, unsigned char, u8)
WRAPOPS_INT(unsigned short, unsigned short, u16)
WRAPOPS_INT(unsigned int, unsigned int, u32)
WRAPOPS_INT(unsigned long long, unsigned long long, u64)

#endif // SIMD_WRAP_H
