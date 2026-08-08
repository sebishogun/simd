// Checksums: Adler-32 and CRC32C.
//
// Adler-32 is the certain win: hash/adler32 in the standard library is a
// scalar Go loop on every architecture, and the sum vectorizes cleanly --
// widen bytes to sixteen bits, multiply by a constant weight ramp for the
// prefix-sum term, defer the modulus to once per NMAX window. CRC32C is
// carry-less-multiply folding where the tier has PCLMUL; the standard
// library already carries strong assembly for it on amd64 and arm64, so
// that kernel earns its place only where measurement says it does.

#include "goabi.h"

typedef long isize;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// ---------- Adler-32 ----------

// MOD is 65521, the largest prime below 2^16. NMAX is the classic 5552:
// the largest n such that a full window of 0xFF bytes cannot overflow a
// u32 accumulator pair before the deferred modulus.
#define AMOD 65521u
#define ANMAX 5552

// Explicit sixteen-lane vectors rather than autovectorization: the nested
// window-then-chunk loop made LLVM decline on every tier but one, and the
// house style is to write the lanes down.
//
// Two u32 accumulator vectors and two vector adds per chunk:
//
//   acc2 += acc1            (before the chunk joins acc1)
//   acc1 += widen(x)
//
// After m chunks, lane j of acc1 holds the byte total of that lane and
// lane j of acc2 holds sum over chunks of the lane's prefix totals, so
//
//   a' = a + hsum(acc1)
//   b' = b + 16*m*a + 16*hsum(acc2) + sum_j (16-j)*acc1[j]
//
// -- the whole weighted structure falls out at the window edge, where the
// deferred modulus already lives. Bounds inside one ANMAX window: acc1
// lanes at most 347*255 < 2^17, acc2 lanes at most 347*346/2*255 < 2^24,
// both comfortably u32.
typedef unsigned char au8x16 __attribute__((ext_vector_type(16), aligned(1)));
typedef u32 au32x16 __attribute__((ext_vector_type(16)));

void simd_adler32(u32 *__restrict out, const u8 *__restrict p, isize n,
                  u32 seed) {
  u32 a = seed & 0xffffu, b = seed >> 16;
  isize i = 0;
  while (i < n) {
    isize win = n - i;
    if (win > ANMAX) win = ANMAX;
    isize end = i + win;
    au32x16 acc1 = 0, acc2 = 0;
    u32 m = 0;
    for (; i + 16 <= end; i += 16, m++) {
      acc2 += acc1;
      acc1 += __builtin_convertvector(*(const au8x16 *)(p + i), au32x16);
    }
    if (m != 0) {
      u32 s1 = 0, s2 = 0, sw = 0;
      for (int j2 = 0; j2 < 16; j2++) {
        s1 += acc1[j2];
        s2 += acc2[j2];
        sw += (u32)(16 - j2) * acc1[j2];
      }
      b += 16u * m * a + 16u * s2 + sw;
      a += s1;
    }
    for (; i < end; i++) {
      a += p[i];
      b += a;
    }
    a %= AMOD;
    b %= AMOD;
  }
  *out = (b << 16) | a;
}

// ---------- CRC32C ----------

#if defined(__PCLMUL__)
#include <wmmintrin.h>
#include <smmintrin.h>

// Reflected CRC32C (Castagnoli). Fold-by-four with PCLMULQDQ while at
// least 64 bytes remain, then drain the four lanes and the tail through
// the SSE4.2 crc32 instruction. Draining through the instruction removes
// every fold-to-one and Barrett constant: the running CRC lives entirely
// inside the lanes (the seed was XORed into lane zero before folding), so
// the drain starts from zero and linearity does the rest. One constant
// pair survives -- the 512-bit-distance fold multipliers for the
// reflected Castagnoli polynomial.
void simd_crc32c(u32 *__restrict out, const u8 *__restrict p, isize n,
                 u32 seed) {
  u32 crc = ~seed;
  isize i = 0;
  if (n >= 64) {
    __m128i x0 = _mm_loadu_si128((const __m128i *)(p + 0));
    __m128i x1 = _mm_loadu_si128((const __m128i *)(p + 16));
    __m128i x2 = _mm_loadu_si128((const __m128i *)(p + 32));
    __m128i x3 = _mm_loadu_si128((const __m128i *)(p + 48));
    x0 = _mm_xor_si128(x0, _mm_cvtsi32_si128((int)crc));
    const __m128i kfold4 = _mm_set_epi64x(0x9e4addf8ll, 0x740eef02ll);
    for (i = 64; i + 64 <= n; i += 64) {
      __m128i y0 = _mm_loadu_si128((const __m128i *)(p + i + 0));
      __m128i y1 = _mm_loadu_si128((const __m128i *)(p + i + 16));
      __m128i y2 = _mm_loadu_si128((const __m128i *)(p + i + 32));
      __m128i y3 = _mm_loadu_si128((const __m128i *)(p + i + 48));
      x0 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x0, kfold4, 0x00),
                                       _mm_clmulepi64_si128(x0, kfold4, 0x11)),
                         y0);
      x1 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x1, kfold4, 0x00),
                                       _mm_clmulepi64_si128(x1, kfold4, 0x11)),
                         y1);
      x2 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x2, kfold4, 0x00),
                                       _mm_clmulepi64_si128(x2, kfold4, 0x11)),
                         y2);
      x3 = _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(x3, kfold4, 0x00),
                                       _mm_clmulepi64_si128(x3, kfold4, 0x11)),
                         y3);
    }
    u64 lane[8];
    _mm_storeu_si128((__m128i *)(lane + 0), x0);
    _mm_storeu_si128((__m128i *)(lane + 2), x1);
    _mm_storeu_si128((__m128i *)(lane + 4), x2);
    _mm_storeu_si128((__m128i *)(lane + 6), x3);
    crc = 0;
    for (int j = 0; j < 8; j++) crc = (u32)__builtin_ia32_crc32di(crc, lane[j]);
  }
  for (; i + 8 <= n; i += 8) {
    u64 v;
    __builtin_memcpy(&v, p + i, 8);
    crc = (u32)__builtin_ia32_crc32di(crc, v);
  }
  for (; i < n; i++) crc = __builtin_ia32_crc32qi(crc, p[i]);
  *out = ~crc;
}
#endif

#if !defined(__PCLMUL__)
// The portable body exists so every target compiles the symbol -- the
// AST check reads the source without target flags, and the verifier then
// skips scalar-only tiers with a stated reason. Bitwise, table-free; the
// reference is the specification either way.
void simd_crc32c(u32 *__restrict out, const u8 *__restrict p, isize n,
                 u32 seed) {
  u32 crc = ~seed;
  for (isize i = 0; i < n; i++) {
    crc ^= p[i];
    for (int k = 0; k < 8; k++)
      crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1)));
  }
  *out = ~crc;
}
#endif
