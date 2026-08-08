// Bulk random fill: eight xoshiro256++ streams in lockstep lanes.
//
// The stream is defined by the reference: eight generators seeded by
// splitmix64 from the caller's seed, emitting round-robin into dst. The
// kernel runs the same eight in vector lanes -- rotl and xor-shift are
// lane ops -- so kernel and reference produce the identical sequence,
// which is what the differential asserts. Not cryptographic; the doc
// says so.

#include "goabi.h"

typedef long isize;
typedef unsigned long long u64;
typedef u64 rnu64x8 __attribute__((ext_vector_type(8), aligned(1)));

static inline rnu64x8 rn_rotl(rnu64x8 x, int k) {
  return (x << k) | (x >> (64 - k));
}

void simd_rand_fill_u64(u64 *__restrict dst, isize n, u64 seed) {
  // splitmix64 per lane seed, matching the reference exactly.
  u64 s[4][8];
  u64 sm = seed;
  for (int lane = 0; lane < 8; lane++) {
    for (int w = 0; w < 4; w++) {
      sm += 0x9E3779B97f4A7C15ull;
      u64 z = sm;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      s[w][lane] = z ^ (z >> 31);
    }
  }
  rnu64x8 s0 = *(rnu64x8 *)s[0], s1 = *(rnu64x8 *)s[1];
  rnu64x8 s2 = *(rnu64x8 *)s[2], s3 = *(rnu64x8 *)s[3];
  isize i = 0;
  for (; i + 8 <= n; i += 8) {
    rnu64x8 result = rn_rotl(s0 + s3, 23) + s0;
    *(rnu64x8 *)(dst + i) = result;
    rnu64x8 t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rn_rotl(s3, 45);
  }
  if (i < n) {
    u64 lane0[8], lane3[8];
    *(rnu64x8 *)lane0 = s0;
    *(rnu64x8 *)lane3 = s3;
    for (int j = 0; i < n; i++, j++) {
      u64 r = lane0[j] + lane3[j];
      r = ((r << 23) | (r >> 41)) + lane0[j];
      dst[i] = r;
    }
  }
}
