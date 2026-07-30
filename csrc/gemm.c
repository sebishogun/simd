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

// ---------- packed-B matmul ----------
//
// The cliff this exists for: the unpacked microkernel reads b[p*n + j0] for
// every p, walking a column strip of B with stride n. At 512x512 doubles the
// operands stop fitting L2 and throughput fell from 124 to 34 GFLOP/s.
// Packing the strip contiguous fixes the locality, and packing is data
// MOVEMENT, not reassociation: the multiply consumes the same values in the
// same p-ascending order, so the result is bit-identical to simd_matmul and
// the contract at MatMulInto holds.
//
// The packed tile width is fixed PER ELEMENT TYPE, not per tier — sixteen
// floats, eight doubles, everywhere, including the portable reference. That
// is not an optimisation choice but a correctness one: the dispatch table
// pairs the pack and multiply slots independently, so a tier could pack with
// one function and multiply with another, and the two must agree on layout
// no matter how they are mixed. A tier whose vector is narrower than the
// tile reads two sub-vectors per tile; the load is still bp + p*W + h*VL,
// contiguous in p, which is the property the packing exists to create.
//
// Two kernels rather than one taking scratch: dst, a, b, scratch plus m, k, n
// is seven arguments, and the seventh leaves the integer registers on x86-64.
//
// Layout: bp[t*(k*W) + p*W + v] = b[p*n + t*W + v], zero-padded past n.
#define GEMM_PKW_F32 16
#define GEMM_PKW_F64 8

#define GEMM_PACKB(SUF, T, W)                                             \
  void simd_gemm_pack_b_##SUF(T *__restrict bp, const T *__restrict bsrc, \
                              isize k, isize n) {                         \
    isize tiles = (n + (W)-1) / (W);                                      \
    for (isize t = 0; t < tiles; t++) {                                   \
      isize j0 = t * (W);                                                 \
      isize w = n - j0 < (W) ? n - j0 : (W);                              \
      T *dstp = bp + t * k * (W);                                         \
      for (isize p = 0; p < k; p++) {                                     \
        for (isize v = 0; v < w; v++)                                     \
          dstp[p * (W) + v] = bsrc[p * n + j0 + v];                       \
        for (isize v = w; v < (W); v++) dstp[p * (W) + v] = (T)0;         \
      }                                                                   \
    }                                                                     \
  }

// H = W/VL sub-vectors per tile: 1 on the widest tier, 2 on the narrower.
#define GEMM_TILE_PK(T, VT, VL, W)                                        \
  {                                                                       \
    VT acc[GEMM_MR][(W) / (VL)];                                          \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR; r++)  \
        _Pragma("clang loop unroll(full)") for (int h = 0;                \
                                                h < (W) / (VL); h++)      \
            acc[r][h] = (VT)(T)0;                                         \
    const T *bt = bp + (j0 / (W)) * (k * (W));                            \
    for (isize p = 0; p < k; p++) {                                       \
      _Pragma("clang loop unroll(full)") for (int h = 0; h < (W) / (VL);  \
                                              h++) {                      \
        VT bv = *(const VT *)(bt + p * (W) + h * (VL));                   \
        _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR;   \
                                                r++) acc[r][h] +=         \
            a[(i0 + r) * k + p] * bv;                                     \
      }                                                                   \
    }                                                                     \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR; r++)  \
        _Pragma("clang loop unroll(full)") for (int h = 0;                \
                                                h < (W) / (VL); h++)      \
            *(VT *)(d + (i0 + r) * n + j0 + h * (VL)) = acc[r][h];        \
  }

#define GEMM_EDGE_PK(T, ROW, JLO, W)                                      \
  for (isize p = 0; p < k; p++) {                                         \
    T s_ = a[(ROW) * k + p];                                              \
    for (isize j = (JLO); j < n; j++) {                                   \
      isize t_ = j / (W);                                                 \
      d[(ROW) * n + j] +=                                                 \
          s_ * bp[t_ * (k * (W)) + p * (W) + (j - t_ * (W))];             \
    }                                                                     \
  }

#define GEMM_PK(T, VT, VL, W)                                             \
  for (isize z = 0; z < m * n; z++) d[z] = 0;                             \
  isize i0 = 0;                                                           \
  for (; i0 + GEMM_MR <= m; i0 += GEMM_MR) {                              \
    isize j0 = 0;                                                         \
    for (; j0 + (W) <= n; j0 += (W)) GEMM_TILE_PK(T, VT, VL, W)           \
    for (isize r = 0; r < GEMM_MR; r++) GEMM_EDGE_PK(T, i0 + r, j0, W)    \
  }                                                                       \
  for (; i0 < m; i0++) GEMM_EDGE_PK(T, i0, 0, W)

void simd_matmul_pk_f32(float *__restrict d, const float *__restrict a,
                        const float *__restrict bp, isize m, isize k,
                        isize n) {
  GEMM_PK(float, f32xG, GEMM_VL_F32, GEMM_PKW_F32)
}

void simd_matmul_pk_f64(double *__restrict d, const double *__restrict a,
                        const double *__restrict bp, isize m, isize k,
                        isize n) {
  GEMM_PK(double, f64xG, GEMM_VL_F64, GEMM_PKW_F64)
}

GEMM_PACKB(f32, float, GEMM_PKW_F32)
GEMM_PACKB(f64, double, GEMM_PKW_F64)

// ---------- quantized matrix multiply ----------
//
// int8 inputs, int32 accumulator. This is the operation quantization exists
// for: QuantizeInt8 produces int8 tensors and, without this, there is nothing
// to multiply them with.
//
// The accumulator has to be wider than the inputs, and that is not a detail.
// simd_matmul is generic over one type, so instantiating it at int8 would
// accumulate in int8 and overflow after the second or third product — two
// full-scale int8 values already multiply to 16129. The sum over a k of any
// realistic size needs 32 bits: the worst case is k * 127 * 128, which stays
// inside int32 up to k = 132097, far past any layer anyone runs.
//
// The tile is the same shape as GEMM_TILE and for the same reasons — MR rows
// held in registers across the whole of k, B loaded as one vector, A
// broadcast — with one addition: B is loaded as int8 and widened to int32
// before the multiply. __builtin_convertvector lowers that to one instruction
// where the target has it (pmovsxbd on x86, sshll+sxtl on AArch64) rather than
// to a lane-by-lane extract.
//
// The multiply is int32 rather than a widening int8 multiply-add such as
// VPMADDUBSW. That instruction pairs adjacent products and would give a
// different summation order, and it saturates its intermediate, which changes
// the answer for inputs a caller is entitled to pass. Rule 2 says integer
// reductions are bit-identical across tiers; getting there by doing the
// arithmetic in the width the result is defined in costs some throughput and
// keeps the promise.
#define GEMM_VL_I32 GEMM_VL_F32

typedef int i32xG __attribute__((ext_vector_type(GEMM_VL_I32), aligned(1)));
typedef signed char i8xG __attribute__((ext_vector_type(GEMM_VL_I32), aligned(1)));

#define QGEMM_TILE                                                        \
  {                                                                       \
    i32xG acc[GEMM_MR];                                                   \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR; r++)   \
        acc[r] = (i32xG)0;                                                \
    for (isize p = 0; p < k; p++) {                                       \
      i8xG b8 = *(const i8xG *)(b + p * n + j0);                          \
      i32xG bv = __builtin_convertvector(b8, i32xG);                      \
      _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR; r++) \
          acc[r] += (int)a[(i0 + r) * k + p] * bv;                        \
    }                                                                     \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < GEMM_MR; r++)   \
        *(i32xG *)(d + (i0 + r) * n + j0) = acc[r];                       \
  }

// The edge case, in the same p order as the tile so the two agree exactly.
#define QGEMM_EDGE(ROW, JLO, JHI)                                         \
  for (isize p = 0; p < k; p++) {                                         \
    int s = a[(ROW) * k + p];                                             \
    const signed char *br = b + p * n;                                    \
    int *dr = d + (ROW) * n;                                              \
    for (isize j = (JLO); j < (JHI); j++) dr[j] += s * (int)br[j];        \
  }

void simd_qmatmul_i8(int *__restrict d, const signed char *__restrict a,
                     const signed char *__restrict b, isize m, isize k,
                     isize n) {
  for (isize i = 0; i < m * n; i++) d[i] = 0;
  isize i0 = 0;
  for (; i0 + GEMM_MR <= m; i0 += GEMM_MR) {
    isize j0 = 0;
    for (; j0 + GEMM_VL_I32 <= n; j0 += GEMM_VL_I32) QGEMM_TILE
    for (isize r = 0; r < GEMM_MR; r++) QGEMM_EDGE(i0 + r, j0, n)
  }
  for (; i0 < m; i0++) QGEMM_EDGE(i0, 0, n)
}

// Requantize is the other half of an inference layer: take the int32
// accumulator back down to int8 with a scale and zero point.
//
// It is separate from the multiply rather than fused into it because the scale
// is per output channel in real use — one per column of the result — and
// folding a gather into the tile's inner loop would cost more than the extra
// pass. Keeping them apart also means a caller who wants the int32 result, for
// bias addition or a residual connection, does not pay for a conversion it
// then has to undo.
//
// Rounding is half to even, matching QuantizeInt8 and every runtime this
// interoperates with, and it is done in float because the int32 range does not
// allow the (|x| + 2^23) trick the float path uses.
void simd_requantize_i8(signed char *__restrict d, const int *__restrict a,
                        float scale, int zero_point, isize n) {
  for (isize i = 0; i < n; i++) {
    float x = (float)a[i] * scale;
    float t = x < 0.0f ? -x : x;
    const float m = 8388608.0f; /* 2^23 */
    float r = (t + m) - m;
    r = __builtin_copysignf(r, x);
    r = t >= m ? x : r;
    float z = r + (float)zero_point;
    z = z < -128.0f ? -128.0f : z;
    z = z > 127.0f ? 127.0f : z;
    d[i] = (signed char)z;
  }
}
