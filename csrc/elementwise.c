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

// Reductions write their answer through an out-pointer rather than returning
// it, and that is not a stylistic choice.
//
// The generator copies a compiled function's bytes verbatim. If the kernel
// returned a value in a register, the generator would have to append an
// instruction storing that register into Go's result slot — but the compiled
// body ends with its own return, and LLVM is free to lay basic blocks out
// after it. Measured: the AVX2 build of the dot product has a jmp *past* its
// return instruction. There is no safe place to append anything.
//
// Writing through a pointer removes the problem entirely. The Go side passes
// the address of its own result slot, the kernel stores into it, and the
// compiled return is the only return there needs to be.
#define REDUCE_SUM(T)                                       \
  T acc[SUM_LANES];                                         \
  for (int j = 0; j < SUM_LANES; j++) acc[j] = 0;           \
  isize i = 0;                                              \
  for (; i + SUM_LANES <= n; i += SUM_LANES)                \
    for (int j = 0; j < SUM_LANES; j++) acc[j] += a[i + j]; \
  for (int j = 0; i < n; i++, j++) acc[j] += a[i];          \
  for (int w = SUM_LANES / 2; w >= 1; w /= 2)               \
    for (int j = 0; j < w; j++) acc[j] += acc[j + w];       \
  *out = acc[0];

void simd_sum_f32(float *__restrict out, const float *__restrict a, isize n) {
  REDUCE_SUM(float)
}

void simd_sum_f64(double *__restrict out, const double *__restrict a, isize n) {
  REDUCE_SUM(double)
}

#define REDUCE_DOT(T)                                                \
  T acc[SUM_LANES];                                                  \
  for (int j = 0; j < SUM_LANES; j++) acc[j] = 0;                    \
  isize i = 0;                                                       \
  for (; i + SUM_LANES <= n; i += SUM_LANES)                         \
    for (int j = 0; j < SUM_LANES; j++) acc[j] += a[i + j] * b[i + j]; \
  for (int j = 0; i < n; i++, j++) acc[j] += a[i] * b[i];            \
  for (int w = SUM_LANES / 2; w >= 1; w /= 2)                        \
    for (int j = 0; j < w; j++) acc[j] += acc[j + w];                \
  *out = acc[0];

void simd_dot_f32(float *__restrict out, const float *__restrict a,
                  const float *__restrict b, isize n) {
  REDUCE_DOT(float)
}

void simd_dot_f64(double *__restrict out, const double *__restrict a,
                  const double *__restrict b, isize n) {
  REDUCE_DOT(double)
}
