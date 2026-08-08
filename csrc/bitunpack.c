// Parquet-style bitpacked decode, width-specialized.
//
// The general kernel takes the bit width as a runtime argument, and no
// vectorizer can hoist a variable shift count into lane constants -- the
// dry run shows it rejected on every tier. Here the width is switched
// ONCE, outside the block loop, and each case is the same loop with the
// shifts as literals, which every tier vectorizes. The macro writes the
// thirty-one cases; the switch is the price of one indirect-free branch
// per call.
//
// Layout matches BitPackInto: little-endian bit order, 32 values per
// ceil(32*W/32) packed words, n a multiple of 32 handled by the guard.

#include "goabi.h"

typedef long isize;
typedef unsigned int u32;
typedef unsigned long long u64;

#define UNPACK_BLOCK(W)                                                   \
  static void unpack_w##W(u32 *__restrict dst, const u32 *__restrict src, \
                          isize blocks) {                                 \
    for (isize b = 0; b < blocks; b++) {                                  \
      const u32 *s = src + (isize)W * b;                                  \
      u32 *d = dst + 32 * b;                                              \
      for (int i = 0; i < 32; i++) {                                      \
        isize bit = (isize)i * W;                                         \
        isize w = bit >> 5;                                               \
        int off = (int)(bit & 31);                                        \
        u64 val = (u64)s[w];                                              \
        if (off + W > 32) val |= (u64)s[w + 1] << 32;                     \
        d[i] = (u32)((val >> off) & (((u64)1 << W) - 1));                 \
      }                                                                   \
    }                                                                     \
  }

UNPACK_BLOCK(1) UNPACK_BLOCK(2) UNPACK_BLOCK(3) UNPACK_BLOCK(4)
UNPACK_BLOCK(5) UNPACK_BLOCK(6) UNPACK_BLOCK(7) UNPACK_BLOCK(8)
UNPACK_BLOCK(9) UNPACK_BLOCK(10) UNPACK_BLOCK(11) UNPACK_BLOCK(12)
UNPACK_BLOCK(13) UNPACK_BLOCK(14) UNPACK_BLOCK(15) UNPACK_BLOCK(16)
UNPACK_BLOCK(17) UNPACK_BLOCK(18) UNPACK_BLOCK(19) UNPACK_BLOCK(20)
UNPACK_BLOCK(21) UNPACK_BLOCK(22) UNPACK_BLOCK(23) UNPACK_BLOCK(24)
UNPACK_BLOCK(25) UNPACK_BLOCK(26) UNPACK_BLOCK(27) UNPACK_BLOCK(28)
UNPACK_BLOCK(29) UNPACK_BLOCK(30) UNPACK_BLOCK(31)

void simd_bitunpack_fast_u32(u32 *__restrict dst, const u32 *__restrict src,
                             isize blocks, u32 bits) {
  switch (bits) {
#define C(W) case W: unpack_w##W(dst, src, blocks); return;
    C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8) C(9) C(10) C(11) C(12) C(13)
    C(14) C(15) C(16) C(17) C(18) C(19) C(20) C(21) C(22) C(23) C(24)
    C(25) C(26) C(27) C(28) C(29) C(30) C(31)
#undef C
  case 32:
    for (isize i = 0; i < 32 * blocks; i++) dst[i] = src[i];
    return;
  default:
    return;
  }
}
