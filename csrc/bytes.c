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
  COUNT_BYTES(b[p] == c)
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
  isize i = 0;
  for (; i + BYTE_LANES <= n; i += BYTE_LANES) {
    // The mirror of NOTANY_MASK: OR the per-member equality masks instead of
    // ANDing the inequalities. Both are one whole-vector operation per set
    // member per register, and both depend on the accumulator never being
    // indexed by lane — see the note there.
    u8xB v = VLOAD(b, i);
    u8xB hit = 0;
    for (isize s = 0; s < k; s++) hit |= (u8xB)(v == (u8xB)set[s]);
    if (OR_ANY(hit)) break;
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
    // A byte that appears twice in the set is still one member of it, and
    // counting it twice is the difference between this and index_any, where
    // the fold is an OR and a duplicate changes nothing. Found by the
    // differential fuzzer with a 256-byte set of zeros: 65536 against 256.
    //
    // The scan is quadratic in the size of the set and linear in the input,
    // and the set is a handful of delimiters, so this costs nothing that can
    // be measured. It is inside the outer loop rather than a separate pass so
    // that no scratch buffer is needed — the kernels have no stack to spare.
    unsigned char dup = 0;
    for (isize p = 0; p < s; p++) dup |= (set[p] == c);
    if (dup) continue;
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

// simd_last_index is simd_index run backwards, matching bytes.LastIndex.
//
// The filter is the same one and for the same reason — compare a block against
// the needle's first and last bytes, and only where both match is a full
// comparison worth doing — but the blocks are walked from the end, so that the
// first match found is the last one in the haystack and the scan can stop
// there rather than running to the end and keeping the highest.
void simd_last_index(isize *__restrict out, const u8 *__restrict h,
                     const u8 *__restrict n_, isize nh, isize nn) {
  if (nn == 0) {
    *out = nh;
    return;
  }
  if (nn > nh) {
    *out = -1;
    return;
  }
  u8 first = n_[0], last = n_[nn - 1];
  const isize block = 64;
  // The highest position a match can begin at. The scan walks down from here.
  isize i = nh - nn;
  for (; i >= block - 1; i -= block) {
    isize lo = i - block + 1;
    unsigned char any = 0;
    for (isize j = 0; j < block; j++)
      any |= (unsigned char)(h[lo + j] == first) &
             (unsigned char)(h[lo + j + nn - 1] == last);
    if (!any) continue;
    for (isize j = block - 1; j >= 0; j--) {
      isize p = lo + j;
      if (h[p] != first || h[p + nn - 1] != last) continue;
      isize k = 1;
      while (k < nn - 1 && h[p + k] == n_[k]) k++;
      if (k >= nn - 1) {
        *out = p;
        return;
      }
    }
  }
  for (; i >= 0; i--) {
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

// simd_count_seq counts non-overlapping occurrences of a needle, matching
// bytes.Count.
//
// Non-overlapping is what makes it a scan rather than a reduction: after a
// match the next search starts past the whole needle, so the positions are not
// independent and the outer loop cannot be vectorized. What can be, and is, is
// the candidate filter — the same first-and-last-byte test as simd_index — so
// the common case of a needle that does not occur costs one pass at vector
// speed rather than a comparison per position.
void simd_count_seq(isize *__restrict out, const u8 *__restrict h,
                    const u8 *__restrict n_, isize nh, isize nn) {
  if (nn == 0) {
    // bytes.Count with an empty separator returns the number of runes plus
    // one; that is a UTF-8 question rather than a byte one, so the wrapper in
    // package simd answers it and never calls this.
    *out = 0;
    return;
  }
  if (nn > nh) {
    *out = 0;
    return;
  }
  u8 first = n_[0], last = n_[nn - 1];
  isize last_start = nh - nn;
  const isize block = 64;
  isize total = 0;
  isize i = 0;
  while (i <= last_start) {
    if (i + block <= last_start + 1) {
      unsigned char any = 0;
      for (isize j = 0; j < block; j++)
        any |= (unsigned char)(h[i + j] == first) &
               (unsigned char)(h[i + j + nn - 1] == last);
      if (!any) {
        i += block;
        continue;
      }
      // A candidate somewhere in this block. Verify the whole block here and
      // then move past it, rather than dropping out to the scalar tail and
      // re-running the filter one position later — that is a 64-wide compare
      // per byte of input for a needle that occurs often, and it made this
      // twice as slow as strings.Count on prose before the shape was measured.
      //
      // A match beginning near the end of the block may run past it. That is
      // fine: i jumps to the end of the match, the inner loop exits, and the
      // outer one resumes wherever that landed.
      isize end = i + block;
      while (i < end && i <= last_start) {
        if (h[i] != first || h[i + nn - 1] != last) {
          i++;
          continue;
        }
        isize k = 1;
        while (k < nn - 1 && h[i + k] == n_[k]) k++;
        if (k >= nn - 1) {
          total++;
          i += nn;
          continue;
        }
        i++;
      }
      continue;
    }
    if (h[i] != first || h[i + nn - 1] != last) {
      i++;
      continue;
    }
    isize k = 1;
    while (k < nn - 1 && h[i + k] == n_[k]) k++;
    if (k >= nn - 1) {
      total++;
      i += nn;
      continue;
    }
    i++;
  }
  *out = total;
}

// simd_index_not_any returns the offset of the first byte of b that is *not*
// in set, or -1 if every byte is; simd_last_index_not_any is the same question
// from the other end. The pair of them is trimming — TrimLeft is the forward
// one, TrimRight this one, Trim both — and it is the primitive under skipping a
// run of whitespace, which is where a tokenizer spends the time it is not
// spending in index_any.
//
// The loop nest is the whole performance story and it is not the obvious one.
// Written the way the question reads — for each byte, is it in the set? — the
// inner loop is over the set, whose length is a runtime value, and LLVM cannot
// vectorize across the outer loop because of it. That version was measured at
// 220ns against 17ns for index_any on the same input and the same set: twelve
// times slower for a strictly simpler question, entirely scalar.
//
// Turning it inside out fixes it. For each *member* of the set, compare a whole
// register against a broadcast of it and AND the "not equal" masks together;
// after k passes a lane is all-ones exactly where that byte matched no member.
// The inner loop is then over contiguous bytes with a loop-invariant operand,
// which is the shape every vector unit here has an instruction for, and the set
// length only decides how many passes.
// The mask is built with whole-vector operations and no element indexing.
// Writing it as a loop over lanes — notin[j] &= ... — is the same arithmetic
// and six times slower: the accumulator is live across the loop over the set,
// so indexing it gives it an address and it spills, once per set member per
// block. Measured at 1379ns against 220ns for the scalar version it was meant
// to replace. The comment on OR_ANY in fold.h is about the same trap.
#define NOTANY_MASK(OFF)                                                 \
  u8xB notin = (u8xB)(u8)0xFF;                                           \
  u8xB v_ = VLOAD(b, (OFF));                                             \
  for (isize s_ = 0; s_ < k; s_++) {                                     \
    u8xB c_ = (u8xB)set[s_];                                             \
    notin &= (u8xB)(v_ != c_);                                           \
  }

// NOTANY_SCALAR answers the same question for one byte.
#define NOTANY_SCALAR(IDX)                                               \
  ({                                                                     \
    unsigned char in_ = 0;                                               \
    for (isize s_ = 0; s_ < k; s_++) in_ |= (b[(IDX)] == set[s_]);       \
    !in_;                                                                \
  })

void simd_index_not_any(isize *__restrict out, const u8 *__restrict b,
                        const u8 *__restrict set, isize n, isize k) {
  if (k == 0) {
    // An empty set contains nothing, so the first byte is already outside it.
    *out = n == 0 ? -1 : 0;
    return;
  }
  isize i = 0;
  for (; i + BYTE_LANES <= n; i += BYTE_LANES) {
    NOTANY_MASK(i)
    if (OR_ANY(notin)) break;
  }
  for (; i < n; i++)
    if (NOTANY_SCALAR(i)) {
      *out = i;
      return;
    }
  *out = -1;
}

void simd_last_index_not_any(isize *__restrict out, const u8 *__restrict b,
                             const u8 *__restrict set, isize n, isize k) {
  if (k == 0) {
    *out = n - 1;
    return;
  }
  isize i = n - BYTE_LANES;
  for (; i >= 0; i -= BYTE_LANES) {
    NOTANY_MASK(i)
    if (OR_ANY(notin)) break;
  }
  // The block that stopped the scan, and everything below it, byte by byte
  // from the top. i may be negative here, which is the short-input case.
  isize from = i + BYTE_LANES - 1;
  if (from > n - 1) from = n - 1;
  for (isize p = from; p >= 0; p--)
    if (NOTANY_SCALAR(p)) {
      *out = p;
      return;
    }
  *out = -1;
}

// ---------- UTF-8 validation ----------
//
// The classifying part of UTF-8 validation is branchless and so it
// vectorizes; the part that decides where a sequence starts is not, and that
// is what shapes this.
//
// The observation the whole thing rests on: a byte's *length* is determined
// by the byte itself, and every constraint on a well-formed sequence is a
// constraint between a byte and one of the three before it. So the scan runs
// forward once, computes for each byte how many continuations it demands, and
// checks the arithmetic — no backtracking and no state machine.
//
// The overlong, surrogate and out-of-range rules are the ones that make a
// naive implementation wrong rather than merely slow, so they are written out
// rather than folded into a range test:
//
//	C0 C1        overlong two-byte forms, never valid
//	E0 A0..BF    a three-byte form starting E0 needs the second byte >= A0
//	ED 80..9F    ED must not reach the surrogate range D800..DFFF
//	F0 90..BF    a four-byte form starting F0 needs the second byte >= 90
//	F4 80..8F    F4 must not exceed U+10FFFF
//	F5..FF       no valid sequence starts here
void simd_valid_utf8(_Bool *__restrict out, const u8 *__restrict b, isize n) {
  isize i = 0;
  // ASCII runs are the common case and the only part worth vectorizing: the
  // fold reads a whole block and only drops to the byte-wise scan when the
  // block contains something above 0x7f.
  while (i < n) {
    if (i + BYTE_LANES <= n) {
      u8xB v = VLOAD(b, i);
      if (!OR_ANY(v & (u8xB)0x80)) {
        i += BYTE_LANES;
        continue;
      }
    }
    u8 c = b[i];
    if (c < 0x80) {
      i++;
      continue;
    }
    isize need;
    if (c >= 0xc2 && c <= 0xdf) {
      need = 1;
    } else if (c >= 0xe0 && c <= 0xef) {
      need = 2;
    } else if (c >= 0xf0 && c <= 0xf4) {
      need = 3;
    } else {
      *out = 0; // 0x80..0xc1 and 0xf5..0xff never start a sequence
      return;
    }
    if (i + need >= n) {
      *out = 0; // truncated at the end of the input
      return;
    }
    u8 c1 = b[i + 1];
    if (c1 < 0x80 || c1 > 0xbf) {
      *out = 0;
      return;
    }
    if (c == 0xe0 && c1 < 0xa0) {
      *out = 0;
      return;
    }
    if (c == 0xed && c1 > 0x9f) {
      *out = 0;
      return;
    }
    if (c == 0xf0 && c1 < 0x90) {
      *out = 0;
      return;
    }
    if (c == 0xf4 && c1 > 0x8f) {
      *out = 0;
      return;
    }
    for (isize j = 2; j <= need; j++) {
      u8 cj = b[i + j];
      if (cj < 0x80 || cj > 0xbf) {
        *out = 0;
        return;
      }
    }
    i += need + 1;
  }
  *out = 1;
}

// ---------- base64 ----------
//
// Standard alphabet, RFC 4648, with padding. Three input bytes become four
// output characters, and the whole difficulty is that neither the gather of
// the six-bit fields nor the mapping to characters is a shape a vectorizer
// finds on its own.
//
// The field extraction is written per output position rather than as a shift
// of a 24-bit accumulator. Written the accumulator way — load three bytes,
// combine, shift out four times — the loads are a strided gather and the
// combine is a loop-carried expression, and LLVM declines. Written as four
// independent expressions of b[3i], b[3i+1] and b[3i+2], each output lane
// depends only on input lanes at a fixed offset, which is a shuffle, and every
// target here has one.
//
// The character mapping is arithmetic, not a table. A 64-entry lookup is a
// gather on every target and a constant pool on most, and this generator
// cannot relocate a pool on four of the six. The alphabet is five contiguous
// runs, so five compares and four adds give the answer branch-free:
//
//	 0..25  ->  'A' + v          26..51 ->  'a' + v - 26
//	52..61  ->  '0' + v - 52     62     ->  '+'        63 -> '/'
//
// Each step is a select against a constant, which is one instruction per lane.
#define B64_CHAR(V)                                                      \
  ({                                                                     \
    u8 v_ = (V);                                                         \
    int c_ = 'A' + v_;                                                   \
    c_ = v_ > 25 ? 'a' + v_ - 26 : c_;                                   \
    c_ = v_ > 51 ? '0' + v_ - 52 : c_;                                   \
    c_ = v_ == 62 ? '+' : c_;                                            \
    c_ = v_ == 63 ? '/' : c_;                                            \
    (u8) c_;                                                             \
  })

// simd_b64_encode writes the base64 of b into d and reports how many bytes it
// wrote, or -1 if d is too short. Both lengths are taken because the output is
// four thirds of the input rounded up to a multiple of four, which is not a
// number the caller's guard can clamp to.
void simd_b64_encode(isize *__restrict out, u8 *__restrict d,
                     const u8 *__restrict b, isize nd, isize nb) {
  isize full = nb / 3;             // whole three-byte groups
  isize rem = nb - full * 3;       // 0, 1 or 2 trailing bytes
  isize need = (full + (rem ? 1 : 0)) * 4;
  if (nd < need) {
    *out = -1;
    return;
  }
  for (isize i = 0; i < full; i++) {
    u8 x = b[i * 3], y = b[i * 3 + 1], z = b[i * 3 + 2];
    d[i * 4] = B64_CHAR((u8)(x >> 2));
    d[i * 4 + 1] = B64_CHAR((u8)(((x & 0x03) << 4) | (y >> 4)));
    d[i * 4 + 2] = B64_CHAR((u8)(((y & 0x0f) << 2) | (z >> 6)));
    d[i * 4 + 3] = B64_CHAR((u8)(z & 0x3f));
  }
  // The tail, which is at most one group and so is not worth vectorizing.
  if (rem) {
    u8 x = b[full * 3];
    u8 y = rem == 2 ? b[full * 3 + 1] : 0;
    d[full * 4] = B64_CHAR((u8)(x >> 2));
    d[full * 4 + 1] = B64_CHAR((u8)(((x & 0x03) << 4) | (y >> 4)));
    d[full * 4 + 2] = rem == 2 ? B64_CHAR((u8)((y & 0x0f) << 2)) : (u8)'=';
    d[full * 4 + 3] = '=';
  }
  *out = need;
}

// B64_VALUE is the inverse mapping: a character to its six-bit value, or 64
// for anything outside the alphabet, which the caller turns into a rejection.
//
// Written as five range tests for the same reason the forward direction is:
// the alternative is a 256-entry table, which is a gather.
#define B64_VALUE(C)                                                     \
  ({                                                                     \
    u8 c_ = (C);                                                         \
    int v_ = 64;                                                         \
    v_ = (c_ >= 'A' && c_ <= 'Z') ? c_ - 'A' : v_;                       \
    v_ = (c_ >= 'a' && c_ <= 'z') ? c_ - 'a' + 26 : v_;                  \
    v_ = (c_ >= '0' && c_ <= '9') ? c_ - '0' + 52 : v_;                  \
    v_ = c_ == '+' ? 62 : v_;                                            \
    v_ = c_ == '/' ? 63 : v_;                                            \
    (u8) v_;                                                             \
  })

// B64_VW pins the vectorization factor of the decode loop, and it has to be
// pinned per target rather than left to LLVM.
//
// The loop reads four bytes per group and writes three, so LLVM models it as
// interleaved access groups and needs a shuffle tree to deinterleave the loads
// and reinterleave the stores. The cost of that tree grows faster than the
// width, and left alone LLVM chose a factor that spilled 576 bytes on AVX2 and
// 704 on AVX-512, over the 512-byte budget a NOSPLIT function has. Both tiers
// were dropped, which is why base64 decoding ran the portable Go loop on every
// x86 machine while arm64 and riscv64 got a kernel.
//
// Measured on AVX-512 at 1 MiB, decoding into 768 KiB: width 16 gives 116us,
// width 32 gives 70us, width 64 gives 44us, against 912us for the portable
// loop. So wider is better here, up to the point where it spills again — and
// that point is target-dependent. At 64, AVX-512 fits but AVX2 spills 608
// bytes and ppc64le needs a save area, so both are lost. At 32 every target
// fits. Taking 64 only where the registers exist to pay for it keeps all six
// tiers and still gets AVX-512 its best number.
#if defined(__AVX512F__)
#define B64_VW 64
#else
#define B64_VW 32
#endif

// simd_b64_decode writes the decoded bytes of b into d and reports how many it
// wrote, or -1 if the input is not valid base64 or d is too short.
//
// Validation and decoding are one pass, not two. The value of an invalid
// character is 64, which has bit 6 set, so ORing every value together and
// testing that one bit at the end says whether anything was rejected — one
// extra OR per lane rather than a branch per character.
void simd_b64_decode(isize *__restrict out, u8 *__restrict d,
                     const u8 *__restrict b, isize nd, isize nb) {
  if (nb % 4 != 0) {
    *out = -1;
    return;
  }
  if (nb == 0) {
    *out = 0;
    return;
  }
  // Padding is only legal in the final group, and there is at most two of it.
  isize pad = 0;
  if (b[nb - 1] == '=') pad++;
  if (nb >= 2 && b[nb - 2] == '=') pad++;
  isize need = nb / 4 * 3 - pad;
  if (nd < need) {
    *out = -1;
    return;
  }
  isize groups = nb / 4 - 1; // every group but the last, which may be padded
  u8 bad = 0;
  _Pragma("clang loop vectorize_width(B64_VW) interleave_count(1)")
  for (isize i = 0; i < groups; i++) {
    u8 a0 = B64_VALUE(b[i * 4]), a1 = B64_VALUE(b[i * 4 + 1]);
    u8 a2 = B64_VALUE(b[i * 4 + 2]), a3 = B64_VALUE(b[i * 4 + 3]);
    bad |= (u8)(a0 | a1 | a2 | a3);
    d[i * 3] = (u8)((a0 << 2) | (a1 >> 4));
    d[i * 3 + 1] = (u8)((a1 << 4) | (a2 >> 2));
    d[i * 3 + 2] = (u8)((a2 << 6) | a3);
  }
  if (bad & 0x40) {
    *out = -1;
    return;
  }
  // The final group, where the padding lives.
  isize i = groups;
  u8 a0 = B64_VALUE(b[i * 4]), a1 = B64_VALUE(b[i * 4 + 1]);
  u8 a2 = pad < 2 ? B64_VALUE(b[i * 4 + 2]) : 0;
  u8 a3 = pad < 1 ? B64_VALUE(b[i * 4 + 3]) : 0;
  if ((a0 | a1 | a2 | a3) & 0x40) {
    *out = -1;
    return;
  }
  d[i * 3] = (u8)((a0 << 2) | (a1 >> 4));
  if (pad < 2) d[i * 3 + 1] = (u8)((a1 << 4) | (a2 >> 2));
  if (pad < 1) d[i * 3 + 2] = (u8)((a2 << 6) | a3);
  *out = need;
}

// simd_hex_decode decodes hex pairs until one is invalid.
//
// Two results — how many bytes were decoded, and whether the whole input was
// valid — which is why this was portable until the generator learned to return
// a pair. Nothing about it needs a scalar loop.
//
// The shape is validate-then-commit, the same escape-hatch pattern fold.h uses
// for the byte searches: a whole block is checked for any invalid nibble
// without branching, and only a block that is entirely valid is decoded and
// written. A block containing an invalid character drops out to the scalar tail,
// which finds the exact offset. The wasted work is bounded by one block, and
// the common case — all valid — never leaves the vector path.
//
// The nibble value and its validity are computed branchlessly. Folding case
// with |0x20 maps 'A'-'F' onto 'a'-'f' so one comparison covers both, and the
// unsigned subtractions wrap on anything out of range, which turns "is it in
// range" into a single unsigned compare rather than a pair of them.
#define HEX_BLOCK 32

void simd_hex_decode(isize *__restrict n_out, _Bool *__restrict ok_out,
                     u8 *__restrict d, const u8 *__restrict s, isize nd,
                     isize ns) {
  isize n = ns / 2;
  if (nd < n) n = nd;
  isize i = 0;

  for (; i + HEX_BLOCK <= n; i += HEX_BLOCK) {
    unsigned char bad = 0;
    for (isize j = 0; j < HEX_BLOCK * 2; j++) {
      u8 c = s[i * 2 + j];
      u8 dig = (u8)(c - (u8)'0');
      u8 alpha = (u8)((c | 0x20) - (u8)'a');
      bad |= (unsigned char)!((dig <= 9) | (alpha <= 5));
    }
    if (bad) break;
    for (isize j = 0; j < HEX_BLOCK; j++) {
      u8 hc = s[i * 2 + j * 2], lc = s[i * 2 + j * 2 + 1];
      u8 hd = (u8)(hc - (u8)'0'), ha = (u8)((hc | 0x20) - (u8)'a');
      u8 ld = (u8)(lc - (u8)'0'), la = (u8)((lc | 0x20) - (u8)'a');
      u8 hv = (u8)(hd <= 9 ? hd : (u8)(ha + 10));
      u8 lv = (u8)(ld <= 9 ? ld : (u8)(la + 10));
      d[i + j] = (u8)((hv << 4) | lv);
    }
  }

  for (; i < n; i++) {
    u8 hc = s[i * 2], lc = s[i * 2 + 1];
    u8 hd = (u8)(hc - (u8)'0'), ha = (u8)((hc | 0x20) - (u8)'a');
    u8 ld = (u8)(lc - (u8)'0'), la = (u8)((lc | 0x20) - (u8)'a');
    if (!((hd <= 9) | (ha <= 5)) || !((ld <= 9) | (la <= 5))) {
      *n_out = i;
      *ok_out = 0;
      return;
    }
    u8 hv = (u8)(hd <= 9 ? hd : (u8)(ha + 10));
    u8 lv = (u8)(ld <= 9 ? ld : (u8)(la + 10));
    d[i] = (u8)((hv << 4) | lv);
  }
  *n_out = n;
  *ok_out = 1;
}
