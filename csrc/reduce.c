// Reduction kernels.
//
// These are separate from arith.c because they are the ones with a numerical
// contract to keep: a reduction must produce the same bits on a 128-bit
// machine and a 512-bit one, which constrains how it may be written. See the
// comment on SUM_LANES below and the documentation on package kernel.
//
// The rules from arith.c apply here too: no function calls, __restrict on
// every pointer, signed loop counters.

typedef long isize;

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
