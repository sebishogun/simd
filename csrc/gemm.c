// Matrix multiply and matrix-vector multiply.
//
// # Why this is not the loop it looks like it should be
//
// The obvious row-major matmul walks i, then p, then j, adding a scalar
// multiple of a row of B into a row of D. It vectorizes cleanly — that is what
// this file replaced — and it is memory-bound at any size that matters. Every
// (i, p) pair reads a whole row of B and reads *and writes* a whole row of D,
// so the traffic is 2·m·k·n element accesses for m·k·n multiplies. The
// arithmetic units idle waiting for cache.
//
// The fix is the standard one and the reason BLAS libraries look the way they
// do: compute a small tile of D entirely in registers, accumulating across all
// of k before storing anything. A tile of MR rows by NR columns reads
// MR·k of A and k·NR of B and writes MR·NR of D, so the traffic falls to
// m·n·k·(1/MR + 1/NR) + m·n — a factor of about eight here, and the store
// traffic to D drops from 2·m·k·n to m·n.
//
// The tile has to live in registers or none of that is true. That is why the
// accumulator is an explicit vector type and the row loop is unrolled by
// pragma rather than left to chance: reduce.c documents the same trap, where
// an accumulator written as an array LLVM has to give an address to ends up on
// the stack and the whole point is lost. The generated code is checked for
// this — see the note on GEMM_MR.
//
// # Accumulation order
//
// Unchanged, and this is not an accident of the rewrite. Each output element
// is a single accumulator summed over p ascending, in both the tiled path and
// both tail paths, exactly as the portable reference does it. The
// vectorization is across j — independent output elements — so no two machines
// disagree and the width of the vector unit does not enter into the answer.
//
// Gemv is the opposite case and is handled the opposite way: there the sum
// *is* the reduction, so it uses the library's fixed sixteen-lane accumulator
// from reduce.c, which makes each row of Gemv bit-identical to Dot of that row
// against x. The tests assert exactly that.
//
// # One deliberate semantic change
//
// The previous implementation skipped a zero scalar — `if (s == 0) continue`.
// That is not a no-op under IEEE 754: it is what decides whether a zero in A
// multiplied by an infinity in B produces a NaN. Skipping suppresses it,
// which disagrees with BLAS, with numpy, and with the standard itself.
//
// It also cannot survive register blocking. In the old shape one scalar test
// guarded a whole row of B — n/VL vector operations — so it was nearly free.
// In a tile it would guard a single fused multiply-add, turning a
// two-instruction inner loop into a four-instruction one and costing more than
// the blocking gains on any matrix that is not mostly zeros.
//
// So it is gone, and MatMul now propagates NaN from 0·∞ the way every other
// implementation of this operation does. The reference was changed to match in
// the same commit; the two paths agree bit for bit, which is the property that
// matters.

#include "goabi.h"

typedef long isize;

// GEMM_MR is how many rows of the output tile are accumulated at once, and
// GEMM_VL_* is how many columns. Together they set the register pressure:
// MR accumulator vectors plus one vector of B, so MR+1 must fit with room to
// spare or the tile spills and the rewrite is worse than what it replaced.
//
// The numbers differ per target because the register file does. AVX-512 has
// thirty-two 512-bit registers and a fused multiply-add with four cycles of
// latency and two per cycle of throughput, so it wants eight independent
// accumulator chains to stay busy; a 128-bit target has sixteen registers and
// cannot afford that. These are the values that measured best on this machine
// and are safe elsewhere, not values derived from a model.
//
// The generated assembly is checked for stack traffic inside the tile loop. A
// spill here does not fail any test — it produces correct answers slowly — so
// it has to be caught by looking.
#if defined(__AVX512F__)
#define GEMM_MR 8
#define GEMM_VL_F32 16
#define GEMM_VL_F64 8
#elif defined(__AVX2__)
#define GEMM_MR 6
#define GEMM_VL_F32 8
#define GEMM_VL_F64 4
#else
#define GEMM_MR 4
#define GEMM_VL_F32 4
#define GEMM_VL_F64 2
#endif

typedef float f32xG __attribute__((ext_vector_type(GEMM_VL_F32), aligned(1)));
typedef double f64xG __attribute__((ext_vector_type(GEMM_VL_F64), aligned(1)));

// GEMM_TILE is the microkernel: MR rows by VL columns, held in registers for
// the whole of k.
//
// The load of B is one unaligned vector; the value from A is a scalar that
// broadcasts across the multiply, which is what makes this an outer-product
// update rather than a dot product. A dot product here would need a strided
// read of B and a horizontal sum, and neither vectorizes.
#define GEMM_TILE(T, VT, VL)                                              \
  {                                                                       \
    VT acc[GEMM_MR];                                                      \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR; r++)   \
        acc[r] = (VT)(T)0;                                                \
    for (isize p = 0; p < k; p++) {                                       \
      VT bv = *(const VT *)(b + p * n + j0);                              \
      _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR; r++) \
          acc[r] += a[(i0 + r) * k + p] * bv;                             \
    }                                                                     \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR; r++)   \
        *(VT *)(d + (i0 + r) * n + j0) = acc[r];                          \
  }

// GEMM_EDGE handles one row against a column range, in the same p order as the
// tile so the answers agree. It reads and writes d rather than accumulating in
// a register, which is why it is only used where the tile does not fit.
#define GEMM_EDGE(T, ROW, JLO, JHI)                                       \
  for (isize p = 0; p < k; p++) {                                         \
    T s = a[(ROW) * k + p];                                               \
    const T *br = b + p * n;                                              \
    T *dr = d + (ROW) * n;                                                \
    for (isize j = (JLO); j < (JHI); j++) dr[j] += s * br[j];             \
  }

#define GEMM(T, VT, VL)                                                   \
  for (isize i = 0; i < m * n; i++) d[i] = 0;                             \
  isize i0 = 0;                                                           \
  for (; i0 + GEMM_MR <= m; i0 += GEMM_MR) {                              \
    isize j0 = 0;                                                         \
    for (; j0 + (VL) <= n; j0 += (VL)) GEMM_TILE(T, VT, VL)               \
    for (isize r = 0; r < GEMM_MR; r++) GEMM_EDGE(T, i0 + r, j0, n)       \
  }                                                                       \
  for (; i0 < m; i0++) GEMM_EDGE(T, i0, 0, n)

void simd_matmul_f32(float *__restrict d, const float *__restrict a,
                     const float *__restrict b, isize m, isize k, isize n) {
  GEMM(float, f32xG, GEMM_VL_F32)
}

void simd_matmul_f64(double *__restrict d, const double *__restrict a,
                     const double *__restrict b, isize m, isize k, isize n) {
  GEMM(double, f64xG, GEMM_VL_F64)
}

// ---------- matrix-vector ----------
//
// d[i] = sum over p of a[i*k + p] * x[p], for a row-major m by k matrix.
//
// This is the operation gonum users actually call and the one this library did
// not have. It is a reduction per row rather than a rank-1 update, so unlike
// the matmul above it has a summation order to pin down, and it uses the
// library's standard one: GEMV_LANES independent accumulators, element p
// landing in lane p % GEMV_LANES, folded at the end by a fixed binary tree.
//
// That is the same shape as Dot, deliberately — Gemv row i is bit-identical to
// Dot(a[i*k:(i+1)*k], x), which is a property worth being able to state and
// which the tests check directly. It is also what keeps the answer independent
// of the vector width, so a 128-bit machine and a 512-bit one agree.
//
// GEMV_LANES must equal SUM_LANES in reduce.c and kernel.SumLanes.
#define GEMV_LANES 16

typedef float f32xV __attribute__((ext_vector_type(GEMV_LANES), aligned(1)));
typedef double f64xV __attribute__((ext_vector_type(GEMV_LANES), aligned(1)));

// The remainder is blended into a second vector rather than accumulated by a
// scalar loop, for the reason reduce.c gives at length: a runtime-indexed tail
// forces the accumulator to have an address, which puts it on the stack. Lanes
// with no element contribute +0, and x + 0 is x for every finite value, every
// infinity and every NaN, so the blend is exact.
#define GEMV(T, V)                                                        \
  for (isize i = 0; i < m; i++) {                                         \
    const T *row = a + i * k;                                             \
    V acc = 0;                                                            \
    isize p = 0;                                                          \
    for (; p + GEMV_LANES <= k; p += GEMV_LANES)                          \
      acc += (*(const V *)(row + p)) * (*(const V *)(x + p));             \
    V tail = 0;                                                           \
    for (int l = 0; l < GEMV_LANES; l++)                                  \
      if (p + l < k) tail[l] = row[p + l] * x[p + l];                     \
    acc += tail;                                                          \
    for (int half = GEMV_LANES / 2; half >= 1; half /= 2)                 \
      for (int l = 0; l < half; l++) acc[l] += acc[l + half];             \
    d[i] = acc[0];                                                        \
  }

void simd_gemv_f32(float *__restrict d, const float *__restrict a,
                   const float *__restrict x, isize m, isize k) {
  GEMV(float, f32xV)
}

void simd_gemv_f64(double *__restrict d, const double *__restrict a,
                   const double *__restrict x, isize m, isize k) {
  GEMV(double, f64xV)
}

// ---------- transpose ----------
//
// Blocked, because the naive loop is a cache disaster and the block size is
// the whole optimisation.
//
// Written as `d[j*m + i] = a[i*n + j]` the loop reads one row of a
// contiguously and writes one column of d with a stride of m elements. Every
// write lands in a different cache line, so a matrix wider than the cache
// evicts each line before its next element arrives and the transpose runs at
// one useful byte per line fetched.
//
// Walking a square block at a time fixes it: the block is small enough that
// its rows of a and its columns of d are all resident at once, so each cache
// line is filled completely before it leaves. 32 is chosen so a block of
// float64 is 32*32*8 = 8 KiB, which fits L1 alongside the destination block on
// every target here; for float32 it is half that and there is room to spare.
//
// The inner loops are left as scalar element moves rather than shuffles. LLVM
// vectorises the contiguous read and, where the target has it, uses a gather
// or an unrolled interleave for the write; hand-written shuffle networks are
// per-architecture and this is not where the time goes once the blocking is
// right.
#define TBLOCK 32

#define TRANSPOSE(T, SUF)                                                \
  void simd_transpose_##SUF(T *__restrict d, const T *__restrict a,      \
                            isize m, isize n) {                          \
    for (isize i0 = 0; i0 < m; i0 += TBLOCK) {                           \
      isize imax = i0 + TBLOCK < m ? i0 + TBLOCK : m;                    \
      for (isize j0 = 0; j0 < n; j0 += TBLOCK) {                         \
        isize jmax = j0 + TBLOCK < n ? j0 + TBLOCK : n;                  \
        for (isize i = i0; i < imax; i++)                                \
          for (isize j = j0; j < jmax; j++) d[j * m + i] = a[i * n + j]; \
      }                                                                  \
    }                                                                    \
  }

TRANSPOSE(float, f32)
TRANSPOSE(double, f64)
TRANSPOSE(int, i32)
TRANSPOSE(long long, i64)
