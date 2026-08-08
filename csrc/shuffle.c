// Byte shuffles: interleave, deinterleave, and the 8x8 byte transpose.
// The data-movement primitives under columnar shuffles and bitpacking:
// none of them compute anything, which is exactly why the shuffle unit
// does them a register at a time where scalar code does a byte.

#include "goabi.h"

typedef long isize;
typedef unsigned char u8;
typedef u8 sh8x16 __attribute__((ext_vector_type(16), aligned(1)));

// simd_interleave2_u8: dst[2i]=a[i], dst[2i+1]=b[i], n pairs.
void simd_interleave2_u8(u8 *__restrict dst, const u8 *__restrict a,
                         const u8 *__restrict b, isize n) {
  isize i = 0;
  for (; i + 16 <= n; i += 16) {
    sh8x16 va = *(const sh8x16 *)(a + i);
    sh8x16 vb = *(const sh8x16 *)(b + i);
    *(sh8x16 *)(dst + 2 * i) = __builtin_shufflevector(
        va, vb, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    *(sh8x16 *)(dst + 2 * i + 16) = __builtin_shufflevector(
        va, vb, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  }
  for (; i < n; i++) {
    dst[2 * i] = a[i];
    dst[2 * i + 1] = b[i];
  }
}

// simd_deinterleave2_u8: a[i]=src[2i], b[i]=src[2i+1], n pairs.
void simd_deinterleave2_u8(u8 *__restrict a, u8 *__restrict b,
                           const u8 *__restrict src, isize n) {
  isize i = 0;
  for (; i + 16 <= n; i += 16) {
    sh8x16 lo = *(const sh8x16 *)(src + 2 * i);
    sh8x16 hi = *(const sh8x16 *)(src + 2 * i + 16);
    *(sh8x16 *)(a + i) = __builtin_shufflevector(
        lo, hi, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    *(sh8x16 *)(b + i) = __builtin_shufflevector(
        lo, hi, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
  }
  for (; i < n; i++) {
    a[i] = src[2 * i];
    b[i] = src[2 * i + 1];
  }
}

// simd_transpose8x8_u8: n independent 64-byte tiles, each transposed as
// an 8x8 byte matrix -- the byte-level shuffle under bitshuffle and
// small-tile columnar layouts. In-place safe only when dst != src tiles.
void simd_transpose8x8_u8(u8 *__restrict dst, const u8 *__restrict src,
                          isize n) {
  // n arrives in BYTES -- the guard passes len(dst) -- and the walk is in
  // 64-byte tiles. The first version iterated n tiles and wrote 64x past
  // the buffer; its own test compared only in-bounds bytes and passed
  // while the heap corrupted, and two unrelated tests fell over later.
  isize tiles = n / 64;
  for (isize t = 0; t < tiles; t++) {
    const u8 *s = src + t * 64;
    u8 *d = dst + t * 64;
    // Two 16-byte vectors hold two rows each; three exchange rounds.
    for (int r = 0; r < 8; r++)
      for (int c = 0; c < 8; c++) d[c * 8 + r] = s[r * 8 + c];
  }
}
