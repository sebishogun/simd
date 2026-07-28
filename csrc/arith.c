// Elementwise and scalar-operand kernels, for every element type.
//
// The rules from elementwise.c apply here too: no function calls, __restrict
// on every pointer, signed loop counters. Two more that this file depends on:
//
//  1. Use __builtin_elementwise_* rather than the plain math builtins. The
//     plain ones set errno, so clang lowers __builtin_sqrtf to a *call* to
//     sqrtf, and a call to anything the object does not define cannot be
//     resolved in Plan 9 assembly. The elementwise family has no errno
//     semantics and lowers to the instruction.
//
//  2. Never negate a signed integer directly. -INT_MIN is undefined behaviour
//     in C, and the contract here is that Abs and Neg wrap, matching what the
//     hardware instruction does. Doing the arithmetic in the unsigned type and
//     converting back is well defined and produces exactly that.

#include "goabi.h"
#include "wrap.h"

typedef long isize;

// ---------- IEEE 754 minimum and maximum ----------
//
// The definitions live in minmax.h so that scan.c and anything else needing
// them shares one copy. See that file for why the builtins are not usable.
#include "minmax.h"

MINMAX_FLOAT(float, f32, __builtin_signbitf)
MINMAX_FLOAT(double, f64, __builtin_signbit)

MINMAX_INT(int, i32)
MINMAX_INT(long long, i64)
// Unsigned comparison falls out of the operand type, so the same macro is
// correct for both signedness classes.
MINMAX_INT(signed char, i8)
MINMAX_INT(short, i16)
MINMAX_INT(unsigned char, u8)
MINMAX_INT(unsigned short, u16)
MINMAX_INT(unsigned int, u32)
MINMAX_INT(unsigned long long, u64)

// ---------- absolute value and negation ----------

static inline float abs_f32(float x) { return __builtin_elementwise_abs(x); }
static inline double abs_f64(double x) { return __builtin_elementwise_abs(x); }

// Wrapping, so abs(INT_MIN) is INT_MIN, matching PABSD and the reference.
// The subtraction is unsigned because -INT_MIN is undefined behaviour.
static inline int abs_i32(int x) { return x < 0 ? (int)(0u - (unsigned)x) : x; }
static inline long long abs_i64(long long x) {
  return x < 0 ? (long long)(0ull - (unsigned long long)x) : x;
}

static inline int neg_i32(int x) { return (int)(0u - (unsigned)x); }
static inline long long neg_i64(long long x) {
  return (long long)(0ull - (unsigned long long)x);
}

// The narrow and unsigned types.
//
// For an unsigned type the absolute value is the identity — there is nothing
// to take the sign of — and negation is the wraparound complement, which is
// what Go's `-x` on an unsigned gives. For the narrow signed types the same
// wrapping rule as the wide ones applies, so abs(INT8_MIN) is INT8_MIN.
//
// The casts back to T are not decoration: C promotes anything narrower than
// int to int before arithmetic, so without them a signed char subtraction
// would be done in 32 bits and the wraparound would happen at the wrong
// width.
#define ABSNEG_SIGNED(T, UT, SUF)                                        \
  static inline T abs_##SUF(T x) {                                       \
    return x < 0 ? (T)(UT)(0 - (UT)x) : x;                               \
  }                                                                      \
  static inline T neg_##SUF(T x) { return (T)(UT)(0 - (UT)x); }

#define ABSNEG_UNSIGNED(T, SUF)                                          \
  static inline T abs_##SUF(T x) { return x; }                           \
  static inline T neg_##SUF(T x) { return (T)(0 - x); }

ABSNEG_SIGNED(signed char, unsigned char, i8)
ABSNEG_SIGNED(short, unsigned short, i16)
ABSNEG_UNSIGNED(unsigned char, u8)
ABSNEG_UNSIGNED(unsigned short, u16)
ABSNEG_UNSIGNED(unsigned int, u32)
ABSNEG_UNSIGNED(unsigned long long, u64)

// ---------- generators ----------

#define BINARY(NAME, T, SUF, EXPR)                                      \
  void simd_##NAME##_##SUF(T *__restrict d, const T *__restrict a,      \
                           const T *__restrict b, isize n) {            \
    for (isize i = 0; i < n; i++) {                                     \
      T x = a[i], y = b[i];                                             \
      d[i] = (EXPR);                                                    \
    }                                                                   \
  }

// BINARY_FORCED is the same, with the vectorizer's cost model overridden.
//
// Used only for IEEE minimum and maximum, where the ten-operation select
// chain against two doubles per register makes LLVM decline on arm64, riscv64
// and ppc64le while taking the identical float32 loop. The comparison that
// matters is not vector against scalar C, though: a kernel that does not
// vectorize is not generated at all, so the real alternative is the portable
// Go loop, with a bounds check per element and one element per iteration.
// Two lanes of select chain beats that comfortably.
//
// It is deliberately not applied anywhere else. Where LLVM declines for a
// reason that is actually about the target, it should be believed.
#define BINARY_FORCED(NAME, T, SUF, EXPR)                               \
  void simd_##NAME##_##SUF(T *__restrict d, const T *__restrict a,      \
                           const T *__restrict b, isize n) {            \
    _Pragma("clang loop vectorize(enable)")                             \
    for (isize i = 0; i < n; i++) {                                     \
      T x = a[i], y = b[i];                                             \
      d[i] = (EXPR);                                                    \
    }                                                                   \
  }

#define UNARY(NAME, T, SUF, EXPR)                                       \
  void simd_##NAME##_##SUF(T *__restrict d, const T *__restrict a,      \
                           isize n) {                                   \
    for (isize i = 0; i < n; i++) {                                     \
      T x = a[i];                                                       \
      d[i] = (EXPR);                                                    \
    }                                                                   \
  }

// SCALAR takes its scalar operand as ST and immediately narrows it to T.
//
// ST is the element type itself for everything as wide as an int, and a
// 64-bit integer for everything narrower. That is an ABI requirement rather
// than a style: how an argument shorter than a register is extended into one
// is decided by each ABI, and they disagree — RISC-V and LoongArch want an
// *unsigned* 32-bit argument sign-extended to 64 bits where the others
// zero-extend it. Declaring the parameter 64 bits wide removes the question,
// and the Go side widens with the move that matches the Go type. See
// target.Target.MovInt8.
#define SCALAR(NAME, T, ST, SUF, EXPR)                                  \
  void simd_##NAME##_##SUF(T *__restrict d, const T *__restrict a,      \
                           ST s_, isize n) {                            \
    T s = (T)s_;                                                        \
    for (isize i = 0; i < n; i++) {                                     \
      T x = a[i];                                                       \
      d[i] = (EXPR);                                                    \
    }                                                                   \
  }

// The whole elementwise family, per type. ST is the scalar transport type;
// see SCALAR.
#define ARITH_COMMON(T, ST, SUF)                                        \
  BINARY(add, T, SUF, add_##SUF(x, y))                                  \
  BINARY(sub, T, SUF, sub_##SUF(x, y))                                  \
  BINARY(mul, T, SUF, mul_##SUF(x, y))                                  \
  BINARY_FORCED(minimum, T, SUF, min_##SUF(x, y))                       \
  BINARY_FORCED(maximum, T, SUF, max_##SUF(x, y))                       \
  UNARY(abs, T, SUF, abs_##SUF(x))                                      \
  SCALAR(scale, T, ST, SUF, mul_##SUF(x, s))                            \
  SCALAR(addscalar, T, ST, SUF, add_##SUF(x, s))                        \
  SCALAR(subscalar, T, ST, SUF, sub_##SUF(x, s))                        \
  void simd_clamp_##SUF(T *__restrict d, const T *__restrict a, ST lo_, \
                        ST hi_, isize n) {                              \
    T lo = (T)lo_, hi = (T)hi_;                                         \
    for (isize i = 0; i < n; i++)                                       \
      d[i] = min_##SUF(max_##SUF(a[i], lo), hi);                        \
  }                                                                     \
  void simd_fill_##SUF(T *__restrict d, ST v_, isize n) {               \
    T v = (T)v_;                                                        \
    for (isize i = 0; i < n; i++) d[i] = v;                             \
  }                                                                     \
  void simd_ramp_##SUF(T *__restrict d, ST start_, ST step_, isize n) { \
    T start = (T)start_, step = (T)step_;                               \
    for (isize i = 0; i < n; i++)                                       \
      d[i] = add_##SUF(start, mul_##SUF((T)i, step));                   \
  }                                                                     \
  void simd_lerp_##SUF(T *__restrict d, const T *__restrict a,          \
                       const T *__restrict b, ST t_, isize n) {         \
    T t = (T)t_;                                                        \
    for (isize i = 0; i < n; i++)                                       \
      d[i] = add_##SUF(a[i], mul_##SUF(sub_##SUF(b[i], a[i]), t));      \
  }                                                                     \
  void simd_addscaled_##SUF(T *__restrict d, const T *__restrict a,     \
                            const T *__restrict b, ST s_, isize n) {    \
    T s = (T)s_;                                                        \
    for (isize i = 0; i < n; i++)                                       \
      d[i] = add_##SUF(a[i], mul_##SUF(b[i], s));                       \
  }

// Float-only: division, and everything that needs a rounding instruction.
//
// Integer division is deliberately absent. Go panics on a zero divisor while C
// leaves it undefined, so an accelerated integer divide would turn a defined
// panic into whatever the hardware felt like. That one stays portable.
#define ARITH_FLOAT(T, SUF)                                             \
  BINARY(div, T, SUF, x / y)                                            \
  SCALAR(divscalar, T, T, SUF, x / s)                                   \
  UNARY(neg, T, SUF, -x)                                                \
  UNARY(sqrt, T, SUF, __builtin_elementwise_sqrt(x))                    \
  UNARY(recip, T, SUF, (T)1 / x)                                        \
  UNARY(floor, T, SUF, __builtin_elementwise_floor(x))                  \
  UNARY(ceil, T, SUF, __builtin_elementwise_ceil(x))                    \
  UNARY(trunc, T, SUF, __builtin_elementwise_trunc(x))

#define ARITH_INT(T, SUF) UNARY(neg, T, SUF, neg_##SUF(x))

ARITH_COMMON(float, float, f32)
ARITH_COMMON(double, double, f64)
ARITH_COMMON(int, int, i32)
ARITH_COMMON(long long, long long, i64)

ARITH_FLOAT(float, f32)
ARITH_FLOAT(double, f64)

ARITH_INT(int, i32)
ARITH_INT(long long, i64)

// The narrow types transport their scalars in a 64-bit integer of matching
// signedness, which is what makes the widening on the Go side unambiguous.
ARITH_COMMON(signed char, long long, i8)
ARITH_COMMON(short, long long, i16)
ARITH_COMMON(unsigned char, unsigned long long, u8)
ARITH_COMMON(unsigned short, unsigned long long, u16)
ARITH_COMMON(unsigned int, unsigned long long, u32)
ARITH_COMMON(unsigned long long, unsigned long long, u64)

ARITH_INT(signed char, i8)
ARITH_INT(short, i16)
ARITH_INT(unsigned char, u8)
ARITH_INT(unsigned short, u16)
ARITH_INT(unsigned int, u32)
ARITH_INT(unsigned long long, u64)

// ---------- saturating arithmetic ----------
//
// The reason the narrow types are worth having at all. A saturating add is a
// single instruction on every vector unit here — vpaddusb on x86, uqadd on
// NEON — and it is the operation image, audio and fixed-point code actually
// wants, where wrapping turns a bright pixel dark.
//
// Written with the elementwise builtins rather than as a widening add and a
// clamp. The clamp form is what the arithmetic means and it is what the Go
// reference does, but it is not what LLVM recognizes: from it clang produces
// a widen, a pair of compares and a narrow — measured, five vpaddd for a
// uint8 kernel where the builtin gives five vpaddusb. The builtins lower to
// llvm.uadd.sat and llvm.ssub.sat, which every backend here pattern-matches
// to its own instruction, and they saturate at the *operand* type's limits,
// so the semantics are the ones the reference states.
#define SATURATE(T, SUF)                                                 \
  void simd_addsat_##SUF(T *__restrict d, const T *__restrict a,         \
                         const T *__restrict b, isize n) {               \
    for (isize i = 0; i < n; i++)                                        \
      d[i] = __builtin_elementwise_add_sat(a[i], b[i]);                  \
  }                                                                      \
  void simd_subsat_##SUF(T *__restrict d, const T *__restrict a,         \
                         const T *__restrict b, isize n) {               \
    for (isize i = 0; i < n; i++)                                        \
      d[i] = __builtin_elementwise_sub_sat(a[i], b[i]);                  \
  }

SATURATE(signed char, i8)
SATURATE(short, i16)
SATURATE(unsigned char, u8)
SATURATE(unsigned short, u16)
SATURATE(int, i32)
SATURATE(unsigned int, u32)

// Round half away from zero, matching math.Round, and round half to even,
// matching math.RoundToEven and the default IEEE 754 mode.
//
// These are separated out because not every target can do them without a
// call: ppc64le has no roundeven instruction and loong64 has no round, so
// clang emits a libcall there. The manifest excludes them on those
// architectures and the backend keeps the portable implementation, which is
// exactly what a partial backend is for.
UNARY(round, float, f32, __builtin_elementwise_round(x))
UNARY(round, double, f64, __builtin_elementwise_round(x))
UNARY(roundeven, float, f32, __builtin_elementwise_roundeven(x))
UNARY(roundeven, double, f64, __builtin_elementwise_roundeven(x))

// Reverse works in place when d and a are the same slice, and between disjoint
// slices; partial overlap is not supported. Walking from both ends at once is
// what makes the in-place case correct.
#define REVERSE(T, SUF)                                                 \
  void simd_reverse_##SUF(T *d, const T *a, isize n) {                  \
    for (isize i = 0, j = n - 1; i < j; i++, j--) {                     \
      T x = a[j], y = a[i];                                             \
      d[i] = x;                                                         \
      d[j] = y;                                                         \
    }                                                                   \
    if (n & 1) d[n / 2] = a[n / 2];                                     \
  }

REVERSE(float, f32)
REVERSE(double, f64)
REVERSE(int, i32)
REVERSE(long long, i64)
