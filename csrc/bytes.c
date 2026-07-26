// Byte, bit and text kernels.
//
// These are the operations a tokenizer, parser or log processor spends its
// time in, and the part of that work a vector unit actually helps with: a
// whole register of bytes classified per instruction.
//
// A recurring shape here is the accumulate-rather-than-branch loop. Written
// the obvious way, "is this byte present" is a search with an early exit —
// faster on a short slice, and not vectorizable at all. Accumulating over the
// whole input instead does more work but does it sixteen to sixty-four bytes
// at a time, which wins on anything long enough to be worth calling into
// assembly for. The dispatcher's threshold is what keeps the short case on the
// portable path where the early exit is still available.

#include "fold.h"

typedef unsigned char u8;

void simd_count_byte(isize *__restrict out, const u8 *__restrict b, u8 c,
                     isize n) {
  COUNT_FOLD(b[p] == c)
}

// simd_index_byte returns the offset of the first occurrence, or -1.
//
// The scan is done in blocks: each block is tested for any match without
// branching, and only a block that contains one is examined byte by byte. That
// keeps the common case — no match in this block — fully vectorized, while
// still reporting the *first* match rather than any match.
void simd_index_byte(isize *__restrict out, const u8 *__restrict b, u8 c,
                     isize n) {
  const isize block = 64;
  isize i = 0;
  for (; i + block <= n; i += block) {
    unsigned char hit = 0;
    for (isize j = 0; j < block; j++) hit |= (b[i + j] == c);
    if (hit) {
      for (isize j = 0; j < block; j++)
        if (b[i + j] == c) {
          *out = i + j;
          return;
        }
    }
  }
  for (; i < n; i++)
    if (b[i] == c) {
      *out = i;
      return;
    }
  *out = -1;
}

// simd_last_index_byte scans backwards in blocks, mirroring simd_index_byte:
// each block is tested without branching and only a block holding a match is
// walked, from its end, so the answer is the *last* match rather than any.
//
// The plain reverse loop with an early exit does not vectorize on any target
// here — LLVM will not turn a backwards search into a masked scan.
void simd_last_index_byte(isize *__restrict out, const u8 *__restrict b, u8 c,
                          isize n) {
  const isize block = 64;
  isize i = n;
  for (; i - block >= 0; i -= block) {
    unsigned char hit = 0;
    for (isize j = 1; j <= block; j++) hit |= (b[i - j] == c);
    if (hit) {
      for (isize j = 1; j <= block; j++)
        if (b[i - j] == c) {
          *out = i - j;
          return;
        }
    }
  }
  for (; i > 0; i--)
    if (b[i - 1] == c) {
      *out = i - 1;
      return;
    }
  *out = -1;
}

// simd_equal_bytes compares without an early exit, for the same reason.
void simd_equal_bytes(_Bool *__restrict out, const u8 *__restrict a,
                      const u8 *__restrict b, isize n) {
  OR_ESCAPE(VLOAD(a, q) ^ VLOAD(b, q), a[p] ^ b[p])
  *out = !hit;
}

// simd_popcount counts set bits across the whole slice.
//
// __builtin_popcount lowers to the instruction where one exists and to a
// bit-twiddling sequence where it does not; neither is a call.
void simd_popcount(isize *__restrict out, const u8 *__restrict b, isize n) {
  COUNT_FOLD(__builtin_popcount((unsigned)b[p]))
}

void simd_is_ascii(_Bool *__restrict out, const u8 *__restrict b, isize n) {
  OR_ESCAPE(VLOAD(b, q) & (unsigned char)0x80, b[p] & 0x80)
  *out = !hit;
}

// ---------- bitwise ----------

#define BITWISE(NAME, EXPR)                                              \
  void simd_bit_##NAME(u8 *__restrict d, const u8 *__restrict a,         \
                       const u8 *__restrict b, isize n) {                \
    for (isize i = 0; i < n; i++) {                                      \
      u8 x = a[i], y = b[i];                                             \
      d[i] = (EXPR);                                                     \
    }                                                                    \
  }

BITWISE(and, x &y)
BITWISE(or, x | y)
BITWISE(xor, x ^ y)
BITWISE(andnot, x & (u8)~y)

void simd_bit_not(u8 *__restrict d, const u8 *__restrict a, isize n) {
  for (isize i = 0; i < n; i++) d[i] = (u8)~a[i];
}

void simd_fill_bytes(u8 *__restrict d, u8 v, isize n) {
  for (isize i = 0; i < n; i++) d[i] = v;
}

// ---------- ASCII text ----------
//
// Case folding is a range test and a constant flip, with no dependence on the
// byte beyond that range — exactly the shape a vector unit handles well. Only
// ASCII is folded, which is what makes it safe to run over UTF-8: every
// continuation byte is 0x80 or above and falls outside both ranges.

void simd_to_upper_ascii(u8 *__restrict d, const u8 *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) {
    u8 c = b[i];
    d[i] = (c >= 'a' && c <= 'z') ? (u8)(c - 32) : c;
  }
}

void simd_to_lower_ascii(u8 *__restrict d, const u8 *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) {
    u8 c = b[i];
    d[i] = (c >= 'A' && c <= 'Z') ? (u8)(c + 32) : c;
  }
}

void simd_replace_byte(u8 *__restrict d, const u8 *__restrict b, u8 old,
                       u8 with, isize n) {
  for (isize i = 0; i < n; i++) {
    u8 c = b[i];
    d[i] = (c == old) ? with : c;
  }
}
