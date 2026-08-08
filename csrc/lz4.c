// LZ4 block decode: the reference's byte-defined walk with the copies
// widened. Literals move sixteen bytes per store; matches with an offset
// of at least sixteen do too; the short offsets that make LZ4 LZ4 -- an
// offset of one is a run -- replicate per byte, exactly as the reference
// defines them. Every widened store checks its slop against the true
// bounds first and falls back to the byte loop near either end, so the
// kernel writes nothing the reference would not.
//
// Returns the decoded length, or -1 for malformed input: truncated
// anywhere, a zero or too-far offset, output past cap. Agreement with
// the reference -- byte for byte, -1 for -1 -- is the contract the
// differential fuzz enforces.

#include "goabi.h"

typedef long isize;
typedef unsigned char u8;
typedef u8 lzu8x16 __attribute__((ext_vector_type(16), aligned(1)));

void simd_lz4_block_decode(isize *__restrict out, u8 *__restrict dst,
                           isize dcap, const u8 *__restrict src, isize n) {
  isize d = 0, i = 0;
  for (;;) {
    if (i >= n) goto bad;
    u8 token = src[i++];
    isize litLen = token >> 4;
    if (litLen == 15) {
      for (;;) {
        if (i >= n) goto bad;
        u8 b = src[i++];
        litLen += b;
        if (b != 255) break;
      }
    }
    if (litLen > n - i || litLen > dcap - d) goto bad;
    // Wide literal copy when sixteen bytes of slop exist on both sides.
    {
      isize k = 0;
      if (litLen >= 16 && n - (i + litLen) >= 0 && dcap - d >= litLen) {
        for (; k + 16 <= litLen && i + k + 16 <= n && d + k + 16 <= dcap;
             k += 16)
          *(lzu8x16 *)(dst + d + k) = *(const lzu8x16 *)(src + i + k);
      }
      for (; k < litLen; k++) dst[d + k] = src[i + k];
    }
    d += litLen;
    i += litLen;
    if (i == n) {
      *out = d;
      return;
    }
    if (n - i < 2) goto bad;
    isize offset = (isize)src[i] | ((isize)src[i + 1] << 8);
    i += 2;
    if (offset == 0 || offset > d) goto bad;
    isize matchLen = (isize)(token & 0xF) + 4;
    if ((token & 0xF) == 15) {
      for (;;) {
        if (i >= n) goto bad;
        u8 b = src[i++];
        matchLen += b;
        if (b != 255) break;
      }
    }
    if (matchLen > dcap - d) goto bad;
    if (offset >= 16) {
      isize k = 0;
      for (; k + 16 <= matchLen && d + k + 16 <= dcap; k += 16)
        *(lzu8x16 *)(dst + d + k) = *(const lzu8x16 *)(dst + d + k - offset);
      for (; k < matchLen; k++) dst[d + k] = dst[d + k - offset];
    } else {
      for (isize k = 0; k < matchLen; k++) dst[d + k] = dst[d + k - offset];
    }
    d += matchLen;
  }
bad:
  *out = -1;
}
