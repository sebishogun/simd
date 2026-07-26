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

typedef long isize;

// ---------- IEEE 754 minimum and maximum ----------
//
// These are written out rather than using __builtin_elementwise_min/max
// because the contract is specific and the builtins are not: the library
// promises IEEE 754-2019 minimum and maximum, where NaN propagates and -0
// compares below +0. The min/max *instructions* on most architectures do
// neither — x86's MINPS returns its second operand when either is NaN — and
// the maxnum family returns the non-NaN operand rather than propagating.
//
// Written as compares and selects, LLVM vectorizes this into exactly the
// compare-and-blend sequence a hand-written kernel would use, and it matches
// the portable reference bit for bit, which the differential tests check.
#define MINMAX_FLOAT(T, SUF, SIGNBIT)                    \
  static inline T min_##SUF(T x, T y) {                  \
    if (x != x) return x;                                \
    if (y != y) return y;                                \
    if (x < y) return x;                                 \
    if (y < x) return y;                                 \
    return SIGNBIT(x) ? x : y; /* -0 wins */             \
  }                                                      \
  static inline T max_##SUF(T x, T y) {                  \
    if (x != x) return x;                                \
    if (y != y) return y;                                \
    if (x > y) return x;                                 \
    if (y > x) return y;                                 \
    return SIGNBIT(x) ? y : x; /* +0 wins */             \
  }

MINMAX_FLOAT(float, f32, __builtin_signbitf)
MINMAX_FLOAT(double, f64, __builtin_signbit)

#define MINMAX_INT(T, SUF)                                        \
  static inline T min_##SUF(T x, T y) { return x < y ? x : y; }   \
  static inline T max_##SUF(T x, T y) { return x > y ? x : y; }

MINMAX_INT(int, i32)
MINMAX_INT(long long, i64)

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

// ---------- generators ----------

#define BINARY(NAME, T, SUF, EXPR)                                      \
  void simd_##NAME##_##SUF(T *__restrict d, const T *__restrict a,      \
                           const T *__restrict b, isize n) {            \
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

#define SCALAR(NAME, T, SUF, EXPR)                                      \
  void simd_##NAME##_##SUF(T *__restrict d, const T *__restrict a, T s, \
                           isize n) {                                   \
    for (isize i = 0; i < n; i++) {                                     \
      T x = a[i];                                                       \
      d[i] = (EXPR);                                                    \
    }                                                                   \
  }

// The whole elementwise family, per type.
#define ARITH_COMMON(T, SUF)                                            \
  BINARY(add, T, SUF, x + y)                                            \
  BINARY(sub, T, SUF, x - y)                                            \
  BINARY(mul, T, SUF, x *y)                                             \
  BINARY(minimum, T, SUF, min_##SUF(x, y))                              \
  BINARY(maximum, T, SUF, max_##SUF(x, y))                              \
  UNARY(abs, T, SUF, abs_##SUF(x))                                      \
  SCALAR(scale, T, SUF, x *s)                                           \
  SCALAR(addscalar, T, SUF, x + s)                                      \
  SCALAR(subscalar, T, SUF, x - s)                                      \
  void simd_clamp_##SUF(T *__restrict d, const T *__restrict a, T lo,   \
                        T hi, isize n) {                                \
    for (isize i = 0; i < n; i++)                                       \
      d[i] = min_##SUF(max_##SUF(a[i], lo), hi);                        \
  }                                                                     \
  void simd_fill_##SUF(T *__restrict d, T v, isize n) {                 \
    for (isize i = 0; i < n; i++) d[i] = v;                             \
  }                                                                     \
  void simd_ramp_##SUF(T *__restrict d, T start, T step, isize n) {     \
    for (isize i = 0; i < n; i++) d[i] = start + (T)i * step;           \
  }                                                                     \
  void simd_lerp_##SUF(T *__restrict d, const T *__restrict a,          \
                       const T *__restrict b, T t, isize n) {           \
    for (isize i = 0; i < n; i++) d[i] = a[i] + (b[i] - a[i]) * t;      \
  }                                                                     \
  void simd_addscaled_##SUF(T *__restrict d, const T *__restrict a,     \
                            const T *__restrict b, T s, isize n) {      \
    for (isize i = 0; i < n; i++) d[i] = a[i] + b[i] * s;               \
  }

// Float-only: division, and everything that needs a rounding instruction.
//
// Integer division is deliberately absent. Go panics on a zero divisor while C
// leaves it undefined, so an accelerated integer divide would turn a defined
// panic into whatever the hardware felt like. That one stays portable.
#define ARITH_FLOAT(T, SUF)                                             \
  BINARY(div, T, SUF, x / y)                                            \
  SCALAR(divscalar, T, SUF, x / s)                                      \
  UNARY(neg, T, SUF, -x)                                                \
  UNARY(sqrt, T, SUF, __builtin_elementwise_sqrt(x))                    \
  UNARY(recip, T, SUF, (T)1 / x)                                        \
  UNARY(floor, T, SUF, __builtin_elementwise_floor(x))                  \
  UNARY(ceil, T, SUF, __builtin_elementwise_ceil(x))                    \
  UNARY(trunc, T, SUF, __builtin_elementwise_trunc(x))

#define ARITH_INT(T, SUF) UNARY(neg, T, SUF, neg_##SUF(x))

ARITH_COMMON(float, f32)
ARITH_COMMON(double, f64)
ARITH_COMMON(int, i32)
ARITH_COMMON(long long, i64)

ARITH_FLOAT(float, f32)
ARITH_FLOAT(double, f64)

ARITH_INT(int, i32)
ARITH_INT(long long, i64)

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
