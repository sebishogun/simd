// Elementwise kernels.
//
// These are written as plain scalar loops and left to LLVM's vectorizer.
// That is the whole reason this project compiles C rather than hand-writing
// assembly: the same source below produces 512-bit AVX-512 on x86, NEON on
// ARMv8, *scalable* SVE2 on ARMv9, and RVV on RISC-V, because LLVM knows how
// to vectorize a loop for each of them and Go's compiler does not.
//
// Rules every kernel here must follow, all of them enforced by the generator:
//
//   1. No function calls. Plan 9 assembly has no procedure linkage table, so
//      a call to anything the object does not define cannot be resolved. This
//      also rules out __builtin_expf and friends, which lower to a call to
//      libm — the reason transcendentals need explicit polynomials.
//   2. __restrict on every pointer. Without it the vectorizer must assume the
//      output may alias an input, and silently emits a scalar loop. The
//      generator fails the build if a kernel produces no vector instructions,
//      so a missing __restrict is caught rather than shipped.
//   3. Signed loop counters. A long is what the ABI bridge passes, and signed
//      overflow being undefined is what lets LLVM prove the trip count.
//   4. Nothing that needs a constant pool, where it can be avoided. A float
//      constant becomes a PC-relative load from .rodata, which is a reference
//      out of the function and has to be lifted separately.

typedef long isize;

void simd_add_f32(float *__restrict d, const float *__restrict a,
                  const float *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] + b[i];
}

void simd_add_f64(double *__restrict d, const double *__restrict a,
                  const double *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] + b[i];
}

void simd_sub_f32(float *__restrict d, const float *__restrict a,
                  const float *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] - b[i];
}

void simd_sub_f64(double *__restrict d, const double *__restrict a,
                  const double *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] - b[i];
}

void simd_mul_f32(float *__restrict d, const float *__restrict a,
                  const float *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] * b[i];
}

void simd_mul_f64(double *__restrict d, const double *__restrict a,
                  const double *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] * b[i];
}

void simd_add_i32(int *__restrict d, const int *__restrict a,
                  const int *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] + b[i];
}

void simd_add_i64(long long *__restrict d, const long long *__restrict a,
                  const long long *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] + b[i];
}

// simd_scale_f32 multiplies by a scalar broadcast across the vector.
void simd_scale_f32(float *__restrict d, const float *__restrict a, float s,
                    isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] * s;
}

void simd_scale_f64(double *__restrict d, const double *__restrict a, double s,
                    isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] * s;
}

// simd_addscaled_f32 is AXPY: one pass over memory instead of the two that a
// separate multiply and add would cost.
void simd_addscaled_f32(float *__restrict d, const float *__restrict a,
                        const float *__restrict b, float s, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] + b[i] * s;
}

void simd_addscaled_f64(double *__restrict d, const double *__restrict a,
                        const double *__restrict b, double s, isize n) {
  for (isize i = 0; i < n; i++) d[i] = a[i] + b[i] * s;
}

// The reductions below accumulate into exactly SUM_LANES independent lanes and
// then fold them with a fixed binary tree, rather than letting the compiler
// choose. That is not an optimization, it is the numerical contract: the same
// shape has to hold on a 128-bit machine and a 512-bit one so that a sum does
// not change value when the program moves to a different server. See the
// documentation on package kernel.
//
// SUM_LANES must equal kernel.SumLanes.
#define SUM_LANES 16

// The accumulators are an explicit vector type, and the remainder is blended
// into a second vector before a single whole-block add. Both details are
// load-bearing and were arrived at by reading the generated code.
//
// Written the obvious way — a T acc[16] array with a runtime-indexed
// remainder loop — LLVM has to give the array an address, so the sixteen
// accumulators live on the stack instead of in registers. Measured spills in
// the float64 sum, and the cost on this machine:
//
//	                    neon   sve2   avx512   sse2
//	array + runtime tail  16     16       21     33   stack references
//	vector + blended tail  0      0        0      0
//
//	n=16   2.40 ns      n=20  10.92 ns
//	n=32   2.44 ns      n=24  11.23 ns
//
// Every length that was a multiple of 16 ran fast and every other length was
// four to five times slower, for less work, because the tail forced the spill.
//
// Guarding the remainder without the vector type does fix amd64 — spills fall
// to zero — but makes LLVM abandon vectorization entirely on arm64, emitting
// 220 scalar instructions and not one vector instruction. The generator's
// vectorization check catches that, which is how it was found. Declaring the
// accumulator as a vector rather than hoping the array is discovered to be one
// removes the guesswork: all four targets now keep it in registers.
//
// Adding the blended tail is exact. Lanes with no element contribute +0, acc
// starts at +0, and x+0 is x for every finite value, infinity and NaN. Each
// element still lands in lane i%SUM_LANES, so the numerical contract in
// package kernel holds bit for bit.
typedef float f32xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));
typedef double f64xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));

#define REDUCE_SUM(T, V)                                  \
  V acc = 0;                                              \
  isize i = 0;                                            \
  for (; i + SUM_LANES <= n; i += SUM_LANES)              \
    acc += *(const V *)(a + i);                           \
  V t = 0;                                                \
  for (int j = 0; j < SUM_LANES; j++)                     \
    if (i + j < n) t[j] = a[i + j];                       \
  acc += t;                                               \
  T r[SUM_LANES];                                         \
  *(V *)r = acc;                                          \
  for (int w = SUM_LANES / 2; w >= 1; w /= 2)             \
    for (int j = 0; j < w; j++) r[j] += r[j + w];         \
  *out = r[0];

void simd_sum_f32(float *__restrict out, const float *__restrict a, isize n) {
  REDUCE_SUM(float, f32xL)
}

void simd_sum_f64(double *__restrict out, const double *__restrict a, isize n) {
  REDUCE_SUM(double, f64xL)
}

#define REDUCE_DOT(T, V)                                  \
  V acc = 0;                                              \
  isize i = 0;                                            \
  for (; i + SUM_LANES <= n; i += SUM_LANES)              \
    acc += *(const V *)(a + i) * *(const V *)(b + i);     \
  V t = 0;                                                \
  for (int j = 0; j < SUM_LANES; j++)                     \
    if (i + j < n) t[j] = a[i + j] * b[i + j];            \
  acc += t;                                               \
  T r[SUM_LANES];                                         \
  *(V *)r = acc;                                          \
  for (int w = SUM_LANES / 2; w >= 1; w /= 2)             \
    for (int j = 0; j < w; j++) r[j] += r[j + w];         \
  *out = r[0];

void simd_dot_f32(float *__restrict out, const float *__restrict a,
                  const float *__restrict b, isize n) {
  REDUCE_DOT(float, f32xL)
}

void simd_dot_f64(double *__restrict out, const double *__restrict a,
                  const double *__restrict b, isize n) {
  REDUCE_DOT(double, f64xL)
}
