// Columnar compute: Arrow-style validity bitmaps driving the value buffers.
//
// The memory model is Arrow's -- a typed value buffer beside a validity
// bitmap, one bit per row, LSB-first within a byte -- but nothing here
// depends on arrow-go: the operations take raw slices and a []byte bitmap,
// so an arrow user hands over the buffers an Array already holds and
// everyone else gets the same columnar toolkit on plain Go data.
//
// Two families:
//
//   compress_bits_*  --  the columnar filter: pack the rows whose bit is
//     set. The same loop-carried dependency compress.c documents (the
//     store address depends on every earlier match) with the same escape
//     (a compress instruction where the target has one), except the mask
//     arrives as packed bits rather than bytes. That is cheaper, not
//     harder: sixteen mask bits are a sixteen-bit load, and on the targets
//     with real compress instructions the predicate register wants exactly
//     that shape.
//
//   sum_valid_*  --  the null-aware sum: add the rows whose bit is set.
//     Accumulation mirrors REDUCE_SUM in reduce.c lane for lane -- same
//     SUM_LANES, same tail handling, same halving tree -- with an invalid
//     lane contributing zero exactly where REDUCE_SUM would have added the
//     value. Zero-substitution does not change the tree's shape, which is
//     what keeps the result bit-identical to the reference on every tier,
//     the same contract every reduction here carries.
//
// Bits become lane masks by testing a broadcast word against a constant
// vector of lane bits -- the movemask inverse, and the one form every
// target vectorizes. Building a _Bool vector element by element folds to
// zeroinitializer at -O1; see the compress.c comment that cost an hour.

#include "goabi.h"

typedef long isize;
typedef unsigned char u8;
typedef unsigned short u16;
typedef int i32;
typedef long long i64;
typedef unsigned int u32;
typedef unsigned long long u64;

#if defined(__riscv_v)
#define CB_LANES 8
#else
#define CB_LANES 16
#endif

typedef _Bool maskxB __attribute__((ext_vector_type(CB_LANES)));
typedef u16 u16xB __attribute__((ext_vector_type(CB_LANES), aligned(1)));

// CB_BITS extracts CB_LANES bits of the bitmap starting at row I (I is a
// multiple of CB_LANES, so the window is byte-aligned for both lane counts).
static inline unsigned CB_BITS(const u8 *__restrict bm, isize i) {
#if CB_LANES == 16
  return (unsigned)bm[i >> 3] | ((unsigned)bm[(i >> 3) + 1] << 8);
#else
  return (unsigned)bm[i >> 3];
#endif
}

// CB_MASK turns those bits into a lane mask.
static inline maskxB CB_MASK(unsigned bits) {
  const u16xB lanebit = {1, 2, 4, 8, 16, 32, 64, 128,
#if CB_LANES == 16
                         256, 512, 1024, 2048, 4096, 8192, 16384, 32768
#endif
  };
  u16xB w = (u16xB)(u16)bits;
  return __builtin_convertvector((w & lanebit) != 0, maskxB);
}

#define COMPRESS_BITS(T, V)                                              \
  isize k = 0, i = 0;                                                    \
  for (; i + CB_LANES <= n; i += CB_LANES) {                             \
    unsigned bits = CB_BITS(bm, i);                                      \
    V v = *(const V *)(src + i);                                         \
    __builtin_masked_compress_store(CB_MASK(bits), v, dst + k);          \
    k += __builtin_popcount(bits);                                       \
  }                                                                      \
  for (; i < n; i++)                                                     \
    if (bm[i >> 3] >> (i & 7) & 1) dst[k++] = src[i];                    \
  *out = k;

typedef i32 cbi32xB __attribute__((ext_vector_type(CB_LANES), aligned(1)));
typedef i64 cbi64xB __attribute__((ext_vector_type(CB_LANES), aligned(1)));
typedef float cbf32xB __attribute__((ext_vector_type(CB_LANES), aligned(1)));
typedef double cbf64xB __attribute__((ext_vector_type(CB_LANES), aligned(1)));

void simd_compress_bits_i32(isize *__restrict out, i32 *__restrict dst,
                            const i32 *__restrict src,
                            const u8 *__restrict bm, isize n) {
  COMPRESS_BITS(i32, cbi32xB)
}

void simd_compress_bits_i64(isize *__restrict out, i64 *__restrict dst,
                            const i64 *__restrict src,
                            const u8 *__restrict bm, isize n) {
  COMPRESS_BITS(i64, cbi64xB)
}

void simd_compress_bits_f32(isize *__restrict out, float *__restrict dst,
                            const float *__restrict src,
                            const u8 *__restrict bm, isize n) {
  COMPRESS_BITS(float, cbf32xB)
}

void simd_compress_bits_f64(isize *__restrict out, double *__restrict dst,
                            const double *__restrict src,
                            const u8 *__restrict bm, isize n) {
  COMPRESS_BITS(double, cbf64xB)
}

// ---------- null-aware sum ----------

// SUM_LANES must equal reduce.c's (and kernel.SumLanes): the null-aware sum
// promises the same bits as Sum over a masked copy, and that only holds if
// the accumulation tree is the same shape.
#define SUM_LANES 16

typedef float svf32xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));
typedef double svf64xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));
typedef i32 svi32xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));
typedef i64 svi64xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));
typedef u16 svu16xL __attribute__((ext_vector_type(SUM_LANES), aligned(1)));
typedef _Bool svmaskxL __attribute__((ext_vector_type(SUM_LANES)));

// A select, not a multiply: Arrow leaves the value under a null bit
// undefined, and garbage * 0 is 0 only until the garbage is NaN or Inf.
#define SUM_VALID(T, V)                                       \
  const svu16xL lanebit = {1, 2, 4, 8, 16, 32, 64, 128, 256,  \
                           512, 1024, 2048, 4096, 8192, 16384, 32768}; \
  V acc = 0;                                                  \
  const V zero = 0;                                           \
  isize i = 0;                                                \
  for (; i + SUM_LANES <= n; i += SUM_LANES) {                \
    unsigned bits = (unsigned)bm[i >> 3] | ((unsigned)bm[(i >> 3) + 1] << 8); \
    svu16xL w = (svu16xL)(u16)bits;                           \
    svmaskxL m = __builtin_convertvector((w & lanebit) != 0, svmaskxL); \
    V v = *(const V *)(a + i);                                \
    acc += m ? v : zero;                                      \
  }                                                           \
  V t = 0;                                                    \
  for (int j = 0; j < SUM_LANES; j++)                         \
    if (i + j < n && (bm[(i + j) >> 3] >> ((i + j) & 7) & 1)) \
      t[j] = a[i + j];                                        \
  acc += t;                                                   \
  T r[SUM_LANES];                                             \
  *(V *)r = acc;                                              \
  for (int w2 = SUM_LANES / 2; w2 >= 1; w2 /= 2)              \
    for (int j = 0; j < w2; j++) r[j] += r[j + w2];           \
  *out = r[0];

void simd_sum_valid_f32(float *__restrict out, const float *__restrict a,
                        const u8 *__restrict bm, isize n) {
  SUM_VALID(float, svf32xL)
}

void simd_sum_valid_f64(double *__restrict out, const double *__restrict a,
                        const u8 *__restrict bm, isize n) {
  SUM_VALID(double, svf64xL)
}

void simd_sum_valid_i32(i32 *__restrict out, const i32 *__restrict a,
                        const u8 *__restrict bm, isize n) {
  SUM_VALID(i32, svi32xL)
}

void simd_sum_valid_i64(i64 *__restrict out, const i64 *__restrict a,
                        const u8 *__restrict bm, isize n) {
  SUM_VALID(i64, svi64xL)
}
