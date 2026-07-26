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

#include "goabi.h"
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

// ---------- comparison and search over a set ----------

// simd_compare_bytes orders a before, equal to, or after b, matching
// bytes.Compare: by content at the first differing byte, and by length if one
// is a prefix of the other.
//
// The first difference is found by the same blocked scan simd_index_byte
// uses — each block is tested without branching and only a block containing
// one is walked — so the common case of a long common prefix stays vector
// code, while the answer is still the *first* difference rather than any.
void simd_compare_bytes(isize *__restrict out, const u8 *__restrict a,
                        const u8 *__restrict b, isize na, isize nb) {
  isize n = na < nb ? na : nb;
  const isize block = 64;
  isize i = 0;
  for (; i + block <= n; i += block) {
    unsigned char diff = 0;
    for (isize j = 0; j < block; j++) diff |= (unsigned char)(a[i + j] ^ b[i + j]);
    if (diff) break;
  }
  for (; i < n; i++)
    if (a[i] != b[i]) {
      *out = a[i] < b[i] ? -1 : 1;
      return;
    }
  *out = na < nb ? -1 : na > nb ? 1 : 0;
}

// simd_equal_fold_ascii compares with the ASCII letters folded to one case.
//
// Folding both sides and reducing the difference keeps this branchless, which
// is what lets it vectorize; the early exit comes from the block structure in
// OR_ESCAPE rather than from a test per byte.
#define FOLD(c) ((u8)(((c) >= 'A' && (c) <= 'Z') ? (c) + 32 : (c)))

// VFOLD is the same fold a whole register at a time. A vector comparison
// yields all-ones in the lanes that match, so masking 32 with it adds the
// case bit exactly where the byte is an uppercase letter and nowhere else.
#define VFOLD(P, Q)                                                      \
  ({                                                                     \
    u8xB c_ = VLOAD(P, Q);                                               \
    c_ + ((c_ >= 'A') & (c_ <= 'Z') & (u8xB)32);                         \
  })

void simd_equal_fold_ascii(_Bool *__restrict out, const u8 *__restrict a,
                           const u8 *__restrict b, isize n) {
  OR_ESCAPE(VFOLD(a, q) ^ VFOLD(b, q), FOLD(a[p]) ^ FOLD(b[p]))
  *out = !hit;
}

// simd_index_any returns the offset of the first byte of b that appears in
// set, or -1; simd_count_any counts how many do.
//
// Both compare against each member of the set in turn rather than testing a
// 256-bit membership table. A table lookup is a gather, which does not
// vectorize from plain C on any target here; comparing k times does k times
// the work but does it a register at a time, and k is small for the sets
// callers actually pass — whitespace, delimiters, a handful of punctuation.
void simd_index_any(isize *__restrict out, const u8 *__restrict b,
                    const u8 *__restrict set, isize n, isize k) {
  const isize block = 64;
  isize i = 0;
  for (; i + block <= n; i += block) {
    unsigned char hit = 0;
    for (isize s = 0; s < k; s++)
      for (isize j = 0; j < block; j++) hit |= (b[i + j] == set[s]);
    if (hit) break;
  }
  for (; i < n; i++)
    for (isize s = 0; s < k; s++)
      if (b[i] == set[s]) {
        *out = i;
        return;
      }
  *out = -1;
}

void simd_count_any(isize *__restrict out, const u8 *__restrict b,
                    const u8 *__restrict set, isize n, isize k) {
  isize total = 0;
  for (isize s = 0; s < k; s++) {
    u32xC acc = 0;
    isize i = 0;
    u8 c = set[s];
    for (; i + COUNT_LANES <= n; i += COUNT_LANES) {
      u32xC v;
      for (int j = 0; j < COUNT_LANES; j++) v[j] = (b[i + j] == c);
      acc += v;
    }
    u32xC t = 0;
    for (int j = 0; j < COUNT_LANES; j++)
      if (i + j < n) t[j] = (b[i + j] == c);
    acc += t;
    unsigned r[COUNT_LANES];
    *(u32xC *)r = acc;
    for (int w = COUNT_LANES / 2; w >= 1; w /= 2)
      for (int j = 0; j < w; j++) r[j] += r[j + w];
    total += r[0];
  }
  *out = total;
}

// simd_hex_encode writes two lowercase hex digits per input byte and reports
// how many bytes it wrote.
//
// The digit is computed rather than looked up: a table would be a constant
// pool, and on a RISC target that is an address built from an instruction
// pair.
//
// It takes both lengths and works out the count itself, because the output is
// twice the size of the input and the caller's guard clamps to the shortest
// slice — which would be the wrong number for either one.
void simd_hex_encode(isize *__restrict out, u8 *__restrict d,
                     const u8 *__restrict b, isize nd, isize nb) {
  isize n = nd / 2 < nb ? nd / 2 : nb;
  *out = n * 2;
  for (isize i = 0; i < n; i++) {
    u8 v = b[i];
    u8 hi = (u8)(v >> 4), lo = (u8)(v & 0x0f);
    d[i * 2] = (u8)(hi < 10 ? '0' + hi : 'a' + hi - 10);
    d[i * 2 + 1] = (u8)(lo < 10 ? '0' + lo : 'a' + lo - 10);
  }
}

// simd_index is substring search, matching bytes.Index.
//
// The candidate filter is the standard one: compare a whole register against
// broadcasts of the needle's first and last bytes, and only positions where
// both match are worth a full comparison. On random text that rejects all but
// about one position in 65536 without ever touching the middle of the needle.
//
// Written as two independent compares over a block rather than as a search,
// so the filter itself vectorizes; the verification of a surviving candidate
// is a short scalar loop, which is fine because it almost never runs.
void simd_index(isize *__restrict out, const u8 *__restrict h,
                const u8 *__restrict n_, isize nh, isize nn) {
  if (nn == 0) {
    *out = 0;
    return;
  }
  if (nn > nh) {
    *out = -1;
    return;
  }
  u8 first = n_[0], last = n_[nn - 1];
  isize last_start = nh - nn; // last position a match could begin at
  const isize block = 64;
  isize i = 0;
  for (; i + block <= last_start + 1; i += block) {
    unsigned char any = 0;
    for (isize j = 0; j < block; j++)
      any |= (unsigned char)(h[i + j] == first) & (unsigned char)(h[i + j + nn - 1] == last);
    if (!any) continue;
    for (isize j = 0; j < block; j++) {
      if (h[i + j] != first || h[i + j + nn - 1] != last) continue;
      isize k = 1;
      while (k < nn - 1 && h[i + j + k] == n_[k]) k++;
      if (k >= nn - 1) {
        *out = i + j;
        return;
      }
    }
  }
  for (; i <= last_start; i++) {
    if (h[i] != first || h[i + nn - 1] != last) continue;
    isize k = 1;
    while (k < nn - 1 && h[i + k] == n_[k]) k++;
    if (k >= nn - 1) {
      *out = i;
      return;
    }
  }
  *out = -1;
}
