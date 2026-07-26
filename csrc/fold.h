// Shared reduction machinery for the byte-domain kernels.
//
// Two problems are solved here, and both were found by reading generated code
// rather than by reasoning about it.
//
// # Why the folds are written by hand
//
// Left to itself, LLVM ends a byte reduction with a horizontal shuffle whose
// index vector it loads from .rodata — a tbl on arm64, a vshuf.b on loong64.
// A constant pool is fine on amd64, where the reference is a single
// RIP-relative displacement this generator can patch, but on every RISC target
// here the address is built from a high/low instruction pair whose page
// displacement is not known until link time. Folding by hand keeps each kernel
// a self-contained blob of bytes that can be copied into a TEXT symbol.
//
// # Why there is an escape hatch
//
// "Is any byte set" is a search with an early exit. Written as one, it is
// faster than any vector code on data that answers early, and not vectorizable
// at all. Written as an unconditional accumulate it vectorizes but reads the
// whole input, which measured 3166ns against the portable version's 1.8ns on a
// 256 KiB slice whose first byte already settled the answer — a 1700x loss.
//
// So neither extreme is right. These fold whole blocks with vector code and
// test the accumulator once per block. The wasted work is bounded by one
// block, the per-block cost is a few instructions against a block's worth of
// vector arithmetic, and the answer still comes early when the data allows it.

typedef long isize;

// BYTE_LANES is the accumulator width. It is a fixed number, not the machine's
// vector width, so that a fold produces the same answer on a 128-bit unit and
// a 512-bit one — the same reasoning as SUM_LANES in reduce.c, though here the
// operations are associative and only the code generation is at stake.
#define BYTE_LANES 32

// ESC_BLOCK is how much is folded between escape tests. Four accumulator
// widths: enough that the test costs a few percent, small enough that the work
// thrown away on an early answer stays under a cache line or two.
#define ESC_BLOCK 128

// COUNT_LANES is the width of the counting accumulators, which are 32 bits per
// lane. Eight bits would wrap after 255 matches and sixteen after 65535, both
// reachable on a slice a caller might plausibly pass; 32 cannot overflow
// before the slice itself exceeds addressable memory.
#define COUNT_LANES 16

typedef unsigned char u8xB __attribute__((ext_vector_type(BYTE_LANES), aligned(1)));
typedef unsigned long long u64xB __attribute__((ext_vector_type(BYTE_LANES / 8), aligned(1)));
typedef unsigned int u32xC __attribute__((ext_vector_type(COUNT_LANES), aligned(1)));

// VLOAD reads one accumulator's worth of bytes at a byte offset. The vector
// type is declared aligned(1), so this is an unaligned load and needs no
// promise about where the caller's slice begins.
#define VLOAD(P, OFF) (*(const u8xB *)((const unsigned char *)(P) + (OFF)))

// OR_ANY reduces an accumulator to "was any bit set".
//
// The vector is reinterpreted as 64-bit lanes — a cast between two ext_vector
// types of the same size is a bitcast, not a conversion — and the lanes are
// combined by constant-index extracts. Both details keep it in registers.
// Storing to a local array instead, which is the obvious way to write this,
// gives the array an address and spills the accumulator: measured at a
// 544-byte frame for the equality kernel, over the 256-byte limit a NOSPLIT
// function has.
#define OR_ANY(ACC)                                                      \
  ({                                                                     \
    u64xB w_ = (u64xB)(ACC);                                             \
    unsigned long long any_ = 0;                                         \
    for (int q_ = 0; q_ < BYTE_LANES / 8; q_++) any_ |= w_[q_];          \
    any_;                                                                \
  })

// OR_ESCAPE reduces with a bitwise or, stopping at the first block that
// contains a set bit. It leaves the answer in `hit`, which the caller inverts
// or not depending on whether it is asking "any" or "all".
//
// It takes the same computation twice: VEXPR is the whole-vector form, reading
// through VLOAD at byte offset q, and SEXPR is the single-element form at
// index p. Only the last partial vector needs the element form, but it does
// need it — and writing the block loops element-wise instead is what made the
// frame too large, because thirty-two scalar inserts have thirty-two live
// values where one vector load has one.
//
// The remainder after the last whole block is folded unconditionally, with the
// final partial vector blended against zero. Or with zero is the identity, so
// lanes with no element contribute nothing.
#define OR_ESCAPE(VEXPR, SEXPR)                                          \
  isize i = 0;                                                           \
  unsigned char hit = 0;                                                 \
  for (; i + ESC_BLOCK <= n; i += ESC_BLOCK) {                           \
    u8xB acc = 0;                                                        \
    for (isize s = 0; s < ESC_BLOCK; s += BYTE_LANES) {                  \
      isize q = i + s;                                                   \
      acc |= (VEXPR);                                                    \
    }                                                                    \
    if (OR_ANY(acc)) {                                                   \
      hit = 1;                                                           \
      break;                                                             \
    }                                                                    \
  }                                                                      \
  if (!hit) {                                                            \
    u8xB acc = 0;                                                        \
    for (; i + BYTE_LANES <= n; i += BYTE_LANES) {                       \
      isize q = i;                                                       \
      acc |= (VEXPR);                                                    \
    }                                                                    \
    u8xB t = 0;                                                          \
    for (int j = 0; j < BYTE_LANES; j++)                                 \
      if (i + j < n) {                                                   \
        isize p = i + j;                                                 \
        t[j] = (unsigned char)(SEXPR);                                   \
      }                                                                  \
    acc |= t;                                                            \
    hit = OR_ANY(acc) != 0;                                              \
  }

// COUNT_FOLD sums EXPR, a small non-negative expression of the index p, over
// the whole slice. There is no escape here: a count has to see everything.
#define COUNT_FOLD(EXPR)                                                 \
  u32xC acc = 0;                                                         \
  isize i = 0;                                                           \
  for (; i + COUNT_LANES <= n; i += COUNT_LANES) {                       \
    u32xC v;                                                             \
    for (int j = 0; j < COUNT_LANES; j++) {                              \
      isize p = i + j;                                                   \
      v[j] = (EXPR);                                                     \
    }                                                                    \
    acc += v;                                                            \
  }                                                                      \
  u32xC t = 0;                                                           \
  for (int j = 0; j < COUNT_LANES; j++)                                  \
    if (i + j < n) {                                                     \
      isize p = i + j;                                                   \
      t[j] = (EXPR);                                                     \
    }                                                                    \
  acc += t;                                                              \
  unsigned r[COUNT_LANES];                                               \
  *(u32xC *)r = acc;                                                     \
  for (int w = COUNT_LANES / 2; w >= 1; w /= 2)                          \
    for (int j = 0; j < w; j++) r[j] += r[j + w];                        \
  *out = (isize)r[0];
