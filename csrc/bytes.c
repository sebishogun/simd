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
typedef unsigned short u16;

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

// simd_common_prefix returns how many leading bytes a and b share.
//
// The same blocked scan simd_compare_bytes uses, without the ordering: a whole
// block is reduced to one "did anything differ" before any byte is examined,
// so a long shared prefix — which is the case this is called for, since that
// is what makes it worth vectorizing — stays entirely in vector code.
//
// This is the inner loop of suffix-array construction, trie descent and the
// LCP array, where the strings compared are usually near-identical and the
// answer is usually large. A byte-at-a-time loop pays a compare and a branch
// per shared byte; this pays one vector compare per sixty-four.
void simd_common_prefix(isize *__restrict out, const u8 *__restrict a,
                        const u8 *__restrict b, isize na, isize nb) {
  isize n = na < nb ? na : nb;
  const isize block = 64;
  isize i = 0;
  for (; i + block <= n; i += block) {
    unsigned char diff = 0;
    for (isize j = 0; j < block; j++)
      diff |= (unsigned char)(a[i + j] ^ b[i + j]);
    if (diff) break;
  }
  for (; i < n; i++)
    if (a[i] != b[i]) break;
  *out = i;
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
    for (isize s = 0; s < k; s++) hit |= (u8xB)(v == SPLAT(u8xB, set[s]));
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

// simd_index_any_or_less returns the offset of the first byte of b that is
// either in set or below lo, or -1.
//
// The two questions separately are simd_index_any and a threshold compare, and
// asking them separately means two passes over the string and two answers to
// reconcile. Asked together it is one pass with one more comparison per
// register, and — the part that matters — one early exit, so a string whose
// first byte already answers it costs one block rather than two full scans.
//
// The shape callers want it for is "find the next byte that is not ordinary
// text": a delimiter, a quote, a backslash, or any control character. That is
// the inner loop of JSON and CSV encoding, of header validation, and of every
// escape routine that copies runs verbatim between the bytes it has to rewrite.
void simd_index_any_or_less(isize *__restrict out, const u8 *__restrict b,
                            const u8 *__restrict set, isize n, isize k,
                            u8 lo) {
  isize i = 0;
  for (; i + BYTE_LANES <= n; i += BYTE_LANES) {
    u8xB v = VLOAD(b, i);
    u8xB hit = (u8xB)(v < SPLAT(u8xB, lo));
    for (isize s = 0; s < k; s++) hit |= (u8xB)(v == SPLAT(u8xB, set[s]));
    if (OR_ANY(hit)) break;
  }
  for (; i < n; i++) {
    if (b[i] < lo) {
      *out = i;
      return;
    }
    for (isize s = 0; s < k; s++)
      if (b[i] == set[s]) {
        *out = i;
        return;
      }
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
  u8xB notin = SPLAT(u8xB, (u8)0xFF);                                           \
  u8xB v_ = VLOAD(b, (OFF));                                             \
  for (isize s_ = 0; s_ < k; s_++) {                                     \
    u8xB c_ = SPLAT(u8xB, set[s_]);                                             \
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
// Every constraint on well-formed UTF-8 is a constraint between a byte and one
// of the three bytes before it. Nothing depends on where a sequence started, so
// nothing has to be found before the checking begins: line each byte up with
// its three predecessors, and the whole grammar becomes elementwise
// comparisons over a block.
//
// The rules that make a naive implementation wrong rather than merely slow are
// the overlong, surrogate and out-of-range forms, so they are written out one
// by one rather than folded into a range test:
//
//	C0 C1        overlong two-byte forms, never valid
//	E0 A0..BF    a three-byte form starting E0 needs the second byte >= A0
//	ED 80..9F    ED must not reach the surrogate range D800..DFFF
//	F0 90..BF    a four-byte form starting F0 needs the second byte >= 90
//	F4 80..8F    F4 must not exceed U+10FFFF
//	F5..FF       no valid sequence starts here

// PREV1/2/3 line each byte up with the one, two or three before it, taking the
// carry-in from the previous block. The indices are constants, so this is a
// lane-crossing shuffle of a pair of registers — palignr, valignd, ext, vshuf
// — and not a load from a constant pool, which is what an index vector would
// have cost on the RISC targets. See the note at the top of fold.h.
#define UTF8_PREV1(PRV, CUR) __builtin_shufflevector(PRV, CUR, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62)
#define UTF8_PREV2(PRV, CUR) __builtin_shufflevector(PRV, CUR, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61)
#define UTF8_PREV3(PRV, CUR) __builtin_shufflevector(PRV, CUR, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)

void simd_valid_utf8(_Bool *__restrict out, const u8 *__restrict b, isize n) {
  isize i = 0;
  u8xB prev = 0, err = 0;
  for (; i + BYTE_LANES <= n; i += BYTE_LANES) {
    u8xB cur = VLOAD(b, i);
    // Text is mostly ASCII even when it is not entirely ASCII, and a block of
    // it needs no checking at all. The previous block is included in the test
    // because a leader in its last three bytes has demands that land here.
    if (!OR_ANY((cur | prev) & SPLAT(u8xB, 0x80))) {
      prev = cur;
      continue;
    }
    u8xB p1 = UTF8_PREV1(prev, cur);
    u8xB p2 = UTF8_PREV2(prev, cur);
    u8xB p3 = UTF8_PREV3(prev, cur);

    // Structure, in one rule: a byte must be a continuation exactly when one
    // of the three before it demands one. Equality rather than implication is
    // what makes it sufficient — it rejects a continuation nobody asked for
    // and a leader whose continuations never arrived, so no separate check of
    // sequence length is needed.
    u8xB want = (u8xB)(p1 >= SPLAT(u8xB, 0xC0)) |
                (u8xB)(p2 >= SPLAT(u8xB, 0xE0)) |
                (u8xB)(p3 >= SPLAT(u8xB, 0xF0));
    u8xB is = (u8xB)((cur & SPLAT(u8xB, 0xC0)) == SPLAT(u8xB, 0x80));
    err |= want ^ is;

    // What is left are the forms that are structurally fine and still not
    // valid UTF-8, every one of them decided by the leader and the byte after
    // it alone:
    //
    //	C0 C1        overlong two-byte forms, never valid
    //	E0 80..9F    overlong three-byte form
    //	ED A0..BF    the surrogate range D800..DFFF
    //	F0 80..8F    overlong four-byte form
    //	F4 90..BF    past U+10FFFF
    //	F5..FF       no valid sequence starts here
    err |= (u8xB)(p1 <= SPLAT(u8xB, 0xC1)) & (u8xB)(p1 >= SPLAT(u8xB, 0xC0));
    err |= (u8xB)(p1 == SPLAT(u8xB, 0xE0)) & (u8xB)(cur < SPLAT(u8xB, 0xA0));
    err |= (u8xB)(p1 == SPLAT(u8xB, 0xED)) & (u8xB)(cur > SPLAT(u8xB, 0x9F));
    err |= (u8xB)(p1 == SPLAT(u8xB, 0xF0)) & (u8xB)(cur < SPLAT(u8xB, 0x90));
    err |= (u8xB)(p1 == SPLAT(u8xB, 0xF4)) & (u8xB)(cur > SPLAT(u8xB, 0x8F));
    err |= (u8xB)(p1 >= SPLAT(u8xB, 0xF5));
    prev = cur;
  }
  if (OR_ANY(err)) {
    *out = 0;
    return;
  }

  // The blocks checked every rule about a byte at a position they covered, but
  // a leader in the last three bytes makes demands about positions they did
  // not. So back up to the start of the last character beginning in the
  // covered region — at most three bytes, since anything longer would already
  // have been an error — and finish byte-wise from there.
  if (i > 0) {
    i--;
    for (int k = 0; k < 3 && i > 0 && (b[i] & 0xC0) == 0x80; k++) i--;
  }
  while (i < n) {
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


// simd_json_copy_run copies the bytes of b that a JSON encoder can write
// verbatim, and returns how many that was.
//
// The scan and the copy are the same pass. Done separately -- find the run,
// then memmove it -- the bytes are read twice, and in an encoder that is two
// passes over every string in the document. Fusing them is the whole point of
// this kernel and the only reason it earns its call: a scan on its own was
// measured against the word-at-a-time version it would replace nine times, in
// nine arrangements, and lost every one.
//
// The escape set is compiled in rather than passed, because that is what makes
// it five arguments instead of seven and six is the ABI's limit. html selects
// whether the three HTML characters and the 0xE2 that leads U+2028 are in it;
// encoding/json escapes them by default and its Fast twin does not.
//
// A byte above ASCII is not a stopping point. It is copied like any other, so a
// document of Japanese runs at the vector rate rather than dropping to a byte
// loop -- the caller has already proved the string is valid UTF-8, and no byte
// of a multi-byte sequence can collide with the set.
void simd_json_copy_run(isize *__restrict out, u8 *__restrict dst,
                        const u8 *__restrict b, isize n, u8 html) {
  isize i = 0;
  // Whether the block loop stopped because something needs escaping, as
  // opposed to running out of whole blocks. The overlapping tail below is only
  // sound in the second case: after a hit, everything from i onwards is the
  // caller's answer and skipping ahead to the end would step over it.
  _Bool hitBlock = 0;
  for (; i + BYTE_LANES <= n; i += BYTE_LANES) {
    u8xB v = VLOAD(b, i);
    u8xB hit = (u8xB)(v < SPLAT(u8xB, 0x20)) | (u8xB)(v == SPLAT(u8xB, '"')) |
               (u8xB)(v == SPLAT(u8xB, '\\'));
    if (html) {
      hit |= (u8xB)(v == SPLAT(u8xB, '<')) | (u8xB)(v == SPLAT(u8xB, '>')) |
             (u8xB)(v == SPLAT(u8xB, '&')) | (u8xB)(v == SPLAT(u8xB, 0xE2));
    }
    if (OR_ANY(hit)) {
      hitBlock = 1;
      break;
    }
    // The block is clean, so it goes out whole. Storing before the test would
    // write past the run; storing after it means a clean block is one load and
    // one store and nothing else.
    *(u8xB *)(dst + i) = v;
  }
  // One more block for the tail, ending at n and overlapping what the loop
  // already covered.
  //
  // The tail was a byte loop, and it showed: 32 bytes took 4.1 ns and 48 took
  // 9.3, because the second is one block plus sixteen single-byte steps. A
  // whole extra block costs one load and one compare and re-reads bytes already
  // known clean, which is cheaper than stepping over half of them. The store
  // rewrites bytes with the values they already hold.
  //
  // Only when the tail is clean, which is the common case. If anything in the
  // overlapping block needs escaping the byte loop below still has to find
  // where, and it starts from i rather than from the block, because the hit may
  // be in the part already passed.
  if (!hitBlock && i < n && n >= BYTE_LANES) {
    u8xB v = VLOAD(b, n - BYTE_LANES);
    u8xB hit = (u8xB)(v < SPLAT(u8xB, 0x20)) | (u8xB)(v == SPLAT(u8xB, '"')) |
               (u8xB)(v == SPLAT(u8xB, '\\'));
    if (html) {
      hit |= (u8xB)(v == SPLAT(u8xB, '<')) | (u8xB)(v == SPLAT(u8xB, '>')) |
             (u8xB)(v == SPLAT(u8xB, '&')) | (u8xB)(v == SPLAT(u8xB, 0xE2));
    }
    if (!OR_ANY(hit)) {
      *(u8xB *)(dst + n - BYTE_LANES) = v;
      *out = n;
      return;
    }
  }
  for (; i < n; i++) {
    u8 c = b[i];
    if (c < 0x20 || c == '"' || c == '\\') break;
    if (html && (c == '<' || c == '>' || c == '&' || c == 0xE2)) break;
    dst[i] = c;
  }
  *out = i;
}

// simd_json_copy_valid is simd_json_copy_run with UTF-8 validation folded into
// the same pass, returning the count it copied or a negative value if what it
// covered was not valid UTF-8.
//
// The encoder walked every string three times: once to find the clean ASCII
// prefix, once to prove the rest valid UTF-8, and once more to copy it and find
// the escapes. Three passes over the same bytes, 44% of a struct encode. The
// classifier here is the one from simd_valid_utf8 and the copy is the one from
// simd_json_copy_run; running them together costs one block's worth of vector
// work instead of two passes' worth of memory traffic.
//
// Unlike simd_json_quote this needs no extra output space: it still stops at
// the first byte an encoder cannot write verbatim, so it writes at most n
// bytes. That reservation is what made the quote kernel unusable here.
//
// The negative return says only that validation failed, not where. The caller's
// invalid path rescans to place the replacement characters, and one return
// value keeps this inside the six argument registers the ABI allows.
//
// Validation covers whole blocks, which can be more than was copied when the
// stop is mid-block. That is safe in the direction that matters: the extra
// bytes are still part of the caller's string, so declaring the string invalid
// on their account is right, and the caller sanitises the whole of it anyway.
void simd_json_copy_valid(isize *__restrict out, u8 *__restrict dst,
                          const u8 *__restrict b, isize n, u8 html) {
  isize i = 0;
  u8xB prev = 0, err = 0;
  for (; i + BYTE_LANES <= n; i += BYTE_LANES) {
    u8xB v = VLOAD(b, i);

    // The escape question first, because it ends the copy and the validity
    // question does not.
    u8xB hit = (u8xB)(v < SPLAT(u8xB, 0x20)) | (u8xB)(v == SPLAT(u8xB, '"')) |
               (u8xB)(v == SPLAT(u8xB, '\\'));
    if (html) {
      hit |= (u8xB)(v == SPLAT(u8xB, '<')) | (u8xB)(v == SPLAT(u8xB, '>')) |
             (u8xB)(v == SPLAT(u8xB, '&')) | (u8xB)(v == SPLAT(u8xB, 0xE2));
    }
    if (OR_ANY(hit)) break;

    // Same shape as simd_valid_utf8: a block with nothing above ASCII in it,
    // and nothing pending from the block before, needs no checking.
    if (OR_ANY((v | prev) & SPLAT(u8xB, 0x80))) {
      u8xB p1 = UTF8_PREV1(prev, v);
      u8xB p2 = UTF8_PREV2(prev, v);
      u8xB p3 = UTF8_PREV3(prev, v);
      u8xB want = (u8xB)(p1 >= SPLAT(u8xB, 0xC0)) |
                  (u8xB)(p2 >= SPLAT(u8xB, 0xE0)) |
                  (u8xB)(p3 >= SPLAT(u8xB, 0xF0));
      u8xB is = (u8xB)((v & SPLAT(u8xB, 0xC0)) == SPLAT(u8xB, 0x80));
      err |= want ^ is;
      err |= (u8xB)(p1 <= SPLAT(u8xB, 0xC1)) & (u8xB)(p1 >= SPLAT(u8xB, 0xC0));
      err |= (u8xB)(p1 == SPLAT(u8xB, 0xE0)) & (u8xB)(v < SPLAT(u8xB, 0xA0));
      err |= (u8xB)(p1 == SPLAT(u8xB, 0xED)) & (u8xB)(v > SPLAT(u8xB, 0x9F));
      err |= (u8xB)(p1 == SPLAT(u8xB, 0xF0)) & (u8xB)(v < SPLAT(u8xB, 0x90));
      err |= (u8xB)(p1 == SPLAT(u8xB, 0xF4)) & (u8xB)(v > SPLAT(u8xB, 0x8F));
      err |= (u8xB)(p1 >= SPLAT(u8xB, 0xF5));
    }
    prev = v;
    *(u8xB *)(dst + i) = v;
  }
  if (OR_ANY(err)) {
    *out = -1;
    return;
  }

  // From here it is byte-wise, and it has to finish what the blocks left: a
  // leader in the last three covered bytes makes demands about bytes they did
  // not reach. Back up to the start of that character, validate byte-wise to
  // the stopping point, and copy forward from where the blocks stopped.
  isize v0 = i;
  if (v0 > 0) {
    v0--;
    for (int k = 0; k < 3 && v0 > 0 && (b[v0] & 0xC0) == 0x80; k++) v0--;
  }
  isize stop = i;
  for (; stop < n; stop++) {
    u8 c = b[stop];
    if (c < 0x20 || c == '"' || c == '\\') break;
    if (html && (c == '<' || c == '>' || c == '&' || c == 0xE2)) break;
    dst[stop] = c;
  }
  while (v0 < stop) {
    u8 c = b[v0];
    if (c < 0x80) {
      v0++;
      continue;
    }
    isize need;
    if (c >= 0xC2 && c <= 0xDF) {
      need = 1;
    } else if (c >= 0xE0 && c <= 0xEF) {
      need = 2;
    } else if (c >= 0xF0 && c <= 0xF4) {
      need = 3;
    } else {
      *out = -1;
      return;
    }
    // A sequence that starts inside the copied region and does not finish
    // inside it is invalid, and it is not the next call's problem. What ended
    // the copy is either the end of the input or a byte an encoder must escape
    // -- below 0x20, a quote, a backslash -- and none of those can be a
    // continuation byte. So the sequence is truncated wherever it stopped.
    if (v0 + need >= stop) {
      *out = -1;
      return;
    }
    for (isize k = 1; k <= need; k++) {
      if ((b[v0 + k] & 0xC0) != 0x80) {
        *out = -1;
        return;
      }
    }
    u8 c1 = b[v0 + 1];
    if ((c == 0xE0 && c1 < 0xA0) || (c == 0xED && c1 > 0x9F) ||
        (c == 0xF0 && c1 < 0x90) || (c == 0xF4 && c1 > 0x8F)) {
      *out = -1;
      return;
    }
    v0 += need + 1;
  }
  *out = stop;
}

// simd_json_quote copies b into dst with JSON escapes written in place, and
// returns how many bytes it wrote.
//
// The difference from simd_json_copy_run is the whole point: that one stops at
// the first byte needing an escape and hands control back, so a string with
// five escapes costs five calls and five returns. This writes the escape and
// keeps going, which is what sonic's quote.c does and is the last measured
// difference between the two encoders.
//
// dst must have room for 6*n bytes. That is the worst case -- every byte
// becoming \u00XX -- and reserving it up front is what removes the per-byte
// space check. A caller that cannot afford that should use simd_json_copy_run.
//
// html selects whether the three HTML characters are escaped and whether the
// 0xE2 lead of U+2028 and U+2029 is examined; encoding/json escapes all five by
// default and its Fast twin escapes none.
//
// A byte above ASCII is copied like any other. The caller has already proved
// the string is valid UTF-8, and no byte of a multi-byte sequence collides with
// the set -- except 0xE2, which is checked against its two significant
// followers rather than assumed.
void simd_json_quote(isize *__restrict out, u8 *__restrict dst,
                     const u8 *__restrict b, isize n, u8 html) {
  static const char hex[16] = "0123456789abcdef";
  isize i = 0, o = 0;
  while (i < n) {
    // A whole clean block goes out in one load and one store. After an escape
    // the loop comes back here rather than walking the rest a byte at a time,
    // so a string with escapes still runs at the vector rate between them.
    if (i + BYTE_LANES <= n) {
      u8xB v = VLOAD(b, i);
      u8xB hit = (u8xB)(v < SPLAT(u8xB, 0x20)) | (u8xB)(v == SPLAT(u8xB, '"')) |
                 (u8xB)(v == SPLAT(u8xB, '\\'));
      if (html) {
        hit |= (u8xB)(v == SPLAT(u8xB, '<')) | (u8xB)(v == SPLAT(u8xB, '>')) |
               (u8xB)(v == SPLAT(u8xB, '&')) | (u8xB)(v == SPLAT(u8xB, 0xE2));
      }
      if (!OR_ANY(hit)) {
        *(u8xB *)(dst + o) = v;
        i += BYTE_LANES;
        o += BYTE_LANES;
        continue;
      }
    }
    u8 c = b[i];
    if (c >= 0x20 && c != '"' && c != '\\' &&
        (!html || (c != '<' && c != '>' && c != '&' && c != 0xE2))) {
      dst[o++] = c;
      i++;
      continue;
    }
    switch (c) {
    case '"':
      dst[o++] = '\\';
      dst[o++] = '"';
      i++;
      break;
    case '\\':
      dst[o++] = '\\';
      dst[o++] = '\\';
      i++;
      break;
    case '\b':
      dst[o++] = '\\';
      dst[o++] = 'b';
      i++;
      break;
    case '\f':
      dst[o++] = '\\';
      dst[o++] = 'f';
      i++;
      break;
    case '\n':
      dst[o++] = '\\';
      dst[o++] = 'n';
      i++;
      break;
    case '\r':
      dst[o++] = '\\';
      dst[o++] = 'r';
      i++;
      break;
    case '\t':
      dst[o++] = '\\';
      dst[o++] = 't';
      i++;
      break;
    default:
      // 0xE2 only reaches here when html is set, and only two sequences under
      // it are escaped: U+2028 and U+2029, which a browser reads as line
      // terminators inside a script. Every other 0xE2 is ordinary text.
      if (c == 0xE2) {
        if (i + 2 < n && b[i + 1] == 0x80 &&
            (b[i + 2] == 0xA8 || b[i + 2] == 0xA9)) {
          dst[o++] = '\\';
          dst[o++] = 'u';
          dst[o++] = '2';
          dst[o++] = '0';
          dst[o++] = '2';
          dst[o++] = b[i + 2] == 0xA8 ? '8' : '9';
          i += 3;
        } else {
          dst[o++] = c;
          i++;
        }
        break;
      }
      // A control character with no shorthand: \u00XX.
      dst[o++] = '\\';
      dst[o++] = 'u';
      dst[o++] = '0';
      dst[o++] = '0';
      dst[o++] = hex[c >> 4];
      dst[o++] = hex[c & 0xF];
      i++;
    }
  }
  *out = o;
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

// ---------- integer parsing ----------
//
// The delimiter scan is not the bottleneck and is not done here. On 200,000
// short CSV fields, simd.IndexAll alone runs at 4.06 GB/s and IndexAll plus
// strconv.Atoi at 0.83 -- so the scan is a fifth of the work and the
// conversion is the rest. This kernel takes the boundaries IndexAll already
// produced and does only the conversion.

// simd_parse_ints converts the fields of src delimited by the offsets in idx
// into signed integers.
//
// idx holds the position of each separator, which is what IndexAll produces,
// so field k is src[start..idx[k]) where start is one past the previous
// separator. The last field runs to ns.
//
// It stops at the first field that is not a valid integer and reports how many
// it converted and whether it consumed everything.
void simd_parse_ints(isize *__restrict n_out, _Bool *__restrict ok_out,
                     long long *__restrict dst, const u8 *__restrict src,
                     const int *__restrict idx, isize nidx) {
  // Powers of ten, most significant first, so a field of length L uses the
  // last L weights. A literal so it is a constant pool load.
  static const long long pow10[19] = {1LL,
                                10LL,
                                100LL,
                                1000LL,
                                10000LL,
                                100000LL,
                                1000000LL,
                                10000000LL,
                                100000000LL,
                                1000000000LL,
                                10000000000LL,
                                100000000000LL,
                                1000000000000LL,
                                10000000000000LL,
                                100000000000000LL,
                                1000000000000000LL,
                                10000000000000000LL,
                                100000000000000000LL,
                                1000000000000000000LL};

  isize start = 0;
  for (isize k = 0; k < nidx; k++) {
    isize end = idx[k];
    isize p = start;
    start = end + 1;

    int neg = 0;
    if (p < end && (src[p] == '-' || src[p] == '+')) {
      neg = src[p] == '-';
      p++;
    }
    isize len = end - p;
    // Empty, or more digits than an int64 can hold. 19 digits can still
    // overflow, so that case is caught by the accumulate below.
    if (len <= 0 || len > 19) {
      *n_out = k;
      *ok_out = 0;
      return;
    }
    // Validate and accumulate as a WEIGHTED SUM, not a Horner chain.
    //
    // Written acc = acc*10 + d, each iteration depends on the previous one and
    // the loop cannot be vectorised at all -- measured at zero vector
    // instructions. Written as a sum of d[j] * 10^(len-1-j) the iterations are
    // independent, integer addition is associative so LLVM may reassociate
    // freely, and the whole field becomes a reduction.
    //
    // The accumulator is unsigned 64-bit and len is at most 19, so the largest
    // representable sum is under 10^19, which fits. That is what makes the
    // overflow check below sound: acc cannot itself have wrapped before it is
    // compared.
    unsigned long long acc = 0;
    unsigned char bad = 0;
    for (isize j = 0; j < len; j++) {
      u8 d = (u8)(src[p + j] - (u8)'0');
      bad |= (unsigned char)(d > 9);
      acc += (unsigned long long)d * (unsigned long long)pow10[len - 1 - j];
    }
    if (bad) {
      *n_out = k;
      *ok_out = 0;
      return;
    }
    // Overflow: the magnitude a signed 64-bit value may reach is 2^63-1, or
    // 2^63 for a negative one.
    unsigned long long limit = neg ? 9223372036854775808ULL
                                   : 9223372036854775807ULL;
    if (acc > limit) {
      *n_out = k;
      *ok_out = 0;
      return;
    }
    dst[k] = neg ? (long long)(0ULL - acc) : (long long)acc;
  }
  *n_out = nidx;
  *ok_out = 1;
}

// ---------- integer formatting ----------
//
// simd_parse_uints is simd_parse_ints without the sign, over the full uint64
// range.
//
// Not a wrapper around it: the signed version's limit is 2^63, so every value
// above that would be rejected, and a sign character must be rejected here
// rather than accepted. The weighted-sum accumulate is the same and for the
// same reason — a Horner chain compiles to zero vector instructions.
//
// Twenty digits fit a uint64 where nineteen fit an int64, and 20 digits can
// overflow, so the accumulator has to detect that itself. It cannot do it by
// comparing against a limit the way the signed version does, because a 20-digit
// weighted sum can wrap before the comparison happens. It checks the leading
// digit and the remainder against the 19-digit maximum instead, which is exact
// and needs no wider arithmetic.
void simd_parse_uints(isize *__restrict n_out, _Bool *__restrict ok_out,
                      unsigned long long *__restrict dst,
                      const u8 *__restrict src, const int *__restrict idx,
                      isize nidx) {
  static const unsigned long long pow10u[20] = {1ULL,
                                                10ULL,
                                                100ULL,
                                                1000ULL,
                                                10000ULL,
                                                100000ULL,
                                                1000000ULL,
                                                10000000ULL,
                                                100000000ULL,
                                                1000000000ULL,
                                                10000000000ULL,
                                                100000000000ULL,
                                                1000000000000ULL,
                                                10000000000000ULL,
                                                100000000000000ULL,
                                                1000000000000000ULL,
                                                10000000000000000ULL,
                                                100000000000000000ULL,
                                                1000000000000000000ULL,
                                                10000000000000000000ULL};

  isize start = 0;
  for (isize k = 0; k < nidx; k++) {
    isize end = idx[k];
    isize p = start;
    start = end + 1;

    isize len = end - p;
    if (len <= 0 || len > 20) {
      *n_out = k;
      *ok_out = 0;
      return;
    }
    // A 20-digit field is split: the leading digit is held out of the sum, so
    // the remaining 19 digits cannot overflow (their maximum is under 10^19)
    // and the combination is checked exactly.
    isize lead = 0;
    unsigned long long leadd = 0;
    if (len == 20) {
      leadd = (unsigned long long)(u8)(src[p] - (u8)'0');
      if (leadd > 9) {
        *n_out = k;
        *ok_out = 0;
        return;
      }
      lead = 1;
    }
    unsigned long long acc = 0;
    unsigned char bad = 0;
    for (isize j = lead; j < len; j++) {
      u8 d = (u8)(src[p + j] - (u8)'0');
      bad |= (unsigned char)(d > 9);
      acc += (unsigned long long)d * pow10u[len - 1 - j];
    }
    if (bad) {
      *n_out = k;
      *ok_out = 0;
      return;
    }
    if (lead) {
      // The value is leadd * 10^19 + acc, and the maximum uint64 is
      // 18446744073709551615 = 1 * 10^19 + 8446744073709551615.
      if (leadd > 1 || (leadd == 1 && acc > 8446744073709551615ULL)) {
        *n_out = k;
        *ok_out = 0;
        return;
      }
      acc += leadd * 10000000000000000000ULL;
    }
    dst[k] = acc;
  }
  *n_out = nidx;
  *ok_out = 1;
}

// The inverse of simd_parse_ints, and the measurement that shaped it: the
// two-digits-per-lookup table is what makes fast formatters fast, and the C
// probe ran at 2.07 GB/s of output against strconv.AppendInt's 0.62 — 3.3x —
// before any vector cleverness. The scatter-based design task 56 sketched is
// not needed; this is the LUT trick, tight.
//
// Digits emerge least-significant pair first into a 20-byte local, then copy
// reversed. The negation is MinInt64-safe: -(x+1)+1 in unsigned avoids the
// overflow that plain -x has at the minimum.
//
// Returns the bytes written, or -1 if fewer than 21 bytes per value remain —
// the worst case is a sign, nineteen digits and a separator. The Go guard
// routes short destinations to the reference, which fits exactly, so -1
// escapes only when even the exact answer cannot fit.
// 208 rather than 201: the literal is 200 digits plus a NUL, and a constant
// pool whose length is not a multiple of four cannot be emitted on
// architectures whose narrowest data directive is a word — the generator
// reports "1 trailing byte(s)". The tail is zero-filled and never read.
static const char simd_fmt_pairs[208] =
    "00010203040506070809101112131415161718192021222324"
    "25262728293031323334353637383940414243444546474849"
    "50515253545556575859606162636465666768697071727374"
    "75767778798081828384858687888990919293949596979899";

void simd_format_ints(isize *__restrict out, u8 *__restrict d,
                      const long long *__restrict v, isize nv, isize nd,
                      long long sep) {
  isize w = 0;
  for (isize i = 0; i < nv; i++) {
    if (nd - w < 21) {
      *out = -1;
      return;
    }
    unsigned long long u;
    long long x = v[i];
    if (x < 0) {
      d[w++] = '-';
      u = (unsigned long long)(-(x + 1)) + 1;
    } else {
      u = (unsigned long long)x;
    }
    u8 tmp[20];
    int t = 0;
    while (u >= 100) {
      unsigned r = (unsigned)(u % 100);
      u /= 100;
      tmp[t++] = (u8)simd_fmt_pairs[r * 2 + 1];
      tmp[t++] = (u8)simd_fmt_pairs[r * 2];
    }
    if (u >= 10) {
      tmp[t++] = (u8)simd_fmt_pairs[u * 2 + 1];
      tmp[t++] = (u8)simd_fmt_pairs[u * 2];
    } else {
      tmp[t++] = (u8)('0' + u);
    }
    while (t > 0) d[w++] = tmp[--t];
    if (i != nv - 1) d[w++] = (u8)sep;
  }
  *out = w;
}

// ---------- UTF-16 ----------
//
// The conversion between UTF-8 and UTF-16 is a dependent scan in general: a
// rune's length decides where the next one starts, so nothing about the second
// rune can be computed before the first is decoded. That is the worst possible
// shape for a vector unit, and it is why unicode/utf16 measures 617 MB/s on a
// []byte caller against a 4.8 GB/s validation floor.
//
// What IS vectorizable is the ASCII run, which in real text is nearly all of
// it: below 0x80 a byte is a whole rune and a whole UTF-16 unit, so the
// conversion collapses to a widen (or a narrow), with no dependence between
// lanes at all. These three kernels are that fast path; the Go side finds the
// runs and decodes the handful of non-ASCII runes between them scalar.
//
// Splitting it this way rather than writing one clever kernel also keeps the
// output length independent of the data within a call, which is what lets
// these stay plain one-result kernels.

// simd_index_nonascii returns the offset of the first byte >= 0x80, or n if
// there is none. Blocked exactly like simd_index_byte: a block is tested
// branchlessly and only a block holding a hit is walked.
void simd_index_nonascii(isize *__restrict out, const u8 *__restrict b,
                         isize n) {
  const isize block = 64;
  isize i = 0;
  for (; i + block <= n; i += block) {
    unsigned char hit = 0;
    for (isize j = 0; j < block; j++) hit |= (unsigned char)(b[i + j] >> 7);
    if (hit) {
      for (isize j = 0; j < block; j++)
        if (b[i + j] & 0x80) {
          *out = i + j;
          return;
        }
    }
  }
  for (; i < n; i++)
    if (b[i] & 0x80) {
      *out = i;
      return;
    }
  *out = n;
}

// simd_index_nonascii16 is the same scan over UTF-16 units, for the decode
// direction. A unit below 0x80 encodes as one UTF-8 byte; anything else,
// including every surrogate, is handled scalar.
void simd_index_nonascii16(isize *__restrict out, const u16 *__restrict b,
                           isize n) {
  const isize block = 32;
  isize i = 0;
  for (; i + block <= n; i += block) {
    unsigned char hit = 0;
    for (isize j = 0; j < block; j++) hit |= (b[i + j] >= 0x80);
    if (hit) {
      for (isize j = 0; j < block; j++)
        if (b[i + j] >= 0x80) {
          *out = i + j;
          return;
        }
    }
  }
  for (; i < n; i++)
    if (b[i] >= 0x80) {
      *out = i;
      return;
    }
  *out = n;
}

// simd_widen_u8_u16 zero-extends n bytes into n units, and
// simd_narrow_u16_u8 truncates back. Both are only ever called on a run the
// scan above has already proven is ASCII, so the narrow's truncation cannot
// lose information.
void simd_widen_u8_u16(u16 *__restrict d, const u8 *__restrict s, isize n) {
  for (isize i = 0; i < n; i++) d[i] = (u16)s[i];
}

void simd_narrow_u16_u8(u8 *__restrict d, const u16 *__restrict s, isize n) {
  for (isize i = 0; i < n; i++) d[i] = (u8)s[i];
}

// The same pair for UTF-32, which is the rune slice Go already speaks. Widening
// an ASCII byte to a rune is the whole conversion for that byte — below 0x80 a
// byte is one rune — so the fast path is identical in shape to the UTF-16 one
// and the Go side decodes the handful of multi-byte runes between the runs.
//
// A separate kernel rather than reusing the u16 one and widening again: two
// passes over the data to save one line of C is the wrong trade at the sizes
// this runs on, and the second pass would read what the first just wrote.
void simd_widen_u8_u32(unsigned int *__restrict d, const u8 *__restrict s,
                       isize n) {
  for (isize i = 0; i < n; i++) d[i] = (unsigned int)s[i];
}

void simd_narrow_u32_u8(u8 *__restrict d, const unsigned int *__restrict s,
                        isize n) {
  for (isize i = 0; i < n; i++) d[i] = (u8)s[i];
}

// ---------- Hamming distance ----------
//
// The number of differing bits between two buffers: sum of popcount(a^b).
//
// Both halves already exist here as kernels — Xor and Popcount — and this is
// the fused single-pass form, which is the entire reason to have it. Chaining
// them needs a full intermediate buffer and three passes over memory where
// this makes one, and at the sizes Hamming distance is used at — binary
// embedding search, LSH buckets, SimHash near-duplicate detection — the
// intermediate is most of the cost.
//
// It goes through COUNT_FOLD rather than a hand-written loop, and that is not
// a stylistic choice. The obvious `s += popcount(a[i]^b[i])` with an isize
// accumulator was written first and measured 1.76x SLOWER than chaining Xor
// and PopCount — the thing this kernel exists to beat — because a 64-bit
// running total forces a widening and a horizontal add per element. COUNT_FOLD
// keeps sixteen 32-bit lane accumulators and folds once at the end, which is
// exactly what simd_popcount already does.
//
// COUNT_FOLD and not COUNT_BYTES: popcount of a byte is 0..8, not a predicate,
// so a byte lane would wrap after 32 blocks rather than the 255 that macro
// assumes.
//
// The count is exact and identical on every tier without the fixed
// sixteen-accumulator tree the float reductions need, because integer addition
// is associative and the lane grouping is therefore not observable.
void simd_hamming_u8(isize *__restrict out, const u8 *__restrict a,
                     const u8 *__restrict b, isize n) {
  COUNT_FOLD(__builtin_popcount((unsigned)(a[p] ^ b[p])))
}

// The 64-bit form, for callers whose bit vectors are already words. Over the
// same bytes it gives the same answer as the byte form on every target here,
// all of which are little-endian, but it is a separate kernel because a
// popcount of a 64-bit lane is one instruction where the byte form needs the
// compiler to widen first.
void simd_hamming_u64(isize *__restrict out,
                      const unsigned long long *__restrict a,
                      const unsigned long long *__restrict b, isize n) {
  COUNT_FOLD(__builtin_popcountll(a[p] ^ b[p]))
}

// ---------- colour ----------
//
// Grayscale and RGB-to-chroma over planar (struct-of-arrays) input: one slice
// per channel rather than interleaved RGBRGB. That is the layout a vector unit
// can use, and converting to it is the advice docs/tutorial.md already gives
// for every other operation here. Interleaved input would need a three-way
// deinterleave per block, which is a different kernel and a different argument.
//
// The weights are ITU-R BT.601 in Q16 fixed point, and they are libjpeg's
// exact constants:
//
//   Y = 0.299 R + 0.587 G + 0.114 B
//     = (19595 R + 38470 G + 7471 B + 32768) >> 16
//
// Fixed point rather than float because that is what makes this reproducible:
// integer multiply-add is exact, so every tier and every architecture gives
// the same byte, and the operation stays under rule 1 instead of needing a ULP
// bound for a result that is going to be eight bits anyway.
//
// Q16 rather than the more obvious Q8. Q8 was written first, with weights
// 77/150/29 summing to 256, and it preserves grey exactly — but it puts full
// green at 149 where the true value is 0.587*255 = 149.7, which rounds to 150.
// One level off at a primary is the kind of thing that shows up as a diff
// against every reference implementation. The Q16 weights sum to exactly 65536,
// so grey is still preserved, and they are accurate to well under half a level
// everywhere else. The intermediate reaches 38470*255, about 9.8 million, so a
// 32-bit lane holds it with three bits to spare.
//
// The rounding term is half the scale, which rounds to nearest; truncating
// would bias every image dark by half a level.
#define Y601_R 19595
#define Y601_G 38470
#define Y601_B 7471

void simd_grayscale_u8(u8 *__restrict d, const u8 *__restrict r,
                       const u8 *__restrict g, const u8 *__restrict b,
                       isize n) {
  for (isize i = 0; i < n; i++) {
    unsigned y = (unsigned)Y601_R * r[i] + (unsigned)Y601_G * g[i] +
                 (unsigned)Y601_B * b[i];
    d[i] = (u8)((y + 32768u) >> 16);
  }
}

// RGB to chroma, full-range (JFIF). Both chroma planes in one pass.
//
// The luma plane is simd_grayscale_u8 above — the same Y, unchanged — so a
// full YUV conversion is these two kernels and a caller that wants luma alone
// pays for one. Splitting them is not only tidiness: as a single kernel this
// took three output pointers, three input pointers and a length, and seven
// arguments is one more than the SysV amd64 ABI passes in registers, so the
// generator declined it on every amd64 tier. Six fits everywhere.
//
// libjpeg's Q16 constants again, and each row sums to exactly zero:
//
//   U = -0.16874 R - 0.33126 G + 0.5 B + 128
//     = (-11059 R - 21709 G + 32768 B) >> 16 + 128
//   V =  0.5 R - 0.41869 G - 0.08131 B + 128
//     = (32768 R - 27439 G - 5329 B) >> 16 + 128
//
// Summing to zero is what makes a grey input give exactly 128 in both planes
// rather than drifting, which is the property that stops a round trip through
// this from tinting a greyscale image.
//
// The +128 bias is folded into the shifted term as (128 << 16) rather than
// added afterwards, which is not a micro-optimisation: it makes the operand
// non-negative before the shift. A right shift of a negative value is
// implementation-defined in C, and the chroma sums genuinely go negative. With
// the bias folded in the operand ranges over [65536, 16777216], so the shift is
// unsigned in effect and the source no longer leans on a guarantee the
// standard does not give.
//
// The top of that range is 256, one past a byte, so the clamp stays.
void simd_rgb_to_uv_u8(u8 *__restrict u, u8 *__restrict v,
                       const u8 *__restrict r, const u8 *__restrict g,
                       const u8 *__restrict b, isize n) {
  for (isize i = 0; i < n; i++) {
    int rr = r[i], gg = g[i], bb = b[i];
    int uu = (-11059 * rr - 21709 * gg + 32768 * bb + 32768 + (128 << 16)) >> 16;
    int vv = (32768 * rr - 27439 * gg - 5329 * bb + 32768 + (128 << 16)) >> 16;
    u[i] = (u8)(uu < 0 ? 0 : uu > 255 ? 255 : uu);
    v[i] = (u8)(vv < 0 ? 0 : vv > 255 ? 255 : vv);
  }
}

// ---------------------------------------------------------------- bit masks

// One bit per input byte, which is the representation a bitwise parser wants.
//
// IndexAll answers the same question as a list of offsets, and for a sparse
// match that is the right answer. For a dense one it is not: a JSON document
// is around 40% structural characters, so the offset list is four times the
// size of the document it describes, and every consumer of it pays a scalar
// step per match. A bitmask is an eighth of the document however dense the
// match is, and — the reason it exists — the questions a parser asks next are
// bitwise. "Which quotes are escaped", "which bytes are inside a string" and
// "which structural characters survive" are all shifts, xors and and-nots over
// the mask, sixty-four bytes of input at a time, with no per-match work at all.
//
// The whole loop is two instructions per sixty-four bytes on a target with
// mask registers: compare into a predicate, store the predicate. There is no
// compress, so unlike simd_index_all the lane count is not held down to the
// index vector's width, and no count, so there is no reduction.
//
// out holds one bit per byte of b, least-significant bit first, so bit i of
// out[i/8] describes b[i]. It must have room for (n+7)/8 bytes. Trailing bits
// of the last byte are cleared.

#define MASK_BITS_LANES 64
typedef u8 u8xM __attribute__((ext_vector_type(MASK_BITS_LANES), aligned(1)));
typedef _Bool maskxM __attribute__((ext_vector_type(MASK_BITS_LANES)));

// STORE_MASK_BITS converts a lane-wise comparison to a bit per lane and writes
// it. __builtin_bit_cast is what makes this two instructions: the obvious
// portable spelling — a loop OR-ing `(hit[j] != 0) << j` — is not recognised as
// a movemask by LLVM, and compiles to a spill of the vector to the stack and
// sixty-four scalar loads. Measured, not assumed; see docs/wrong.md.
#define STORE_MASK_BITS(HIT, OUT, I)                                     \
  do {                                                                   \
    maskxM m_ = __builtin_convertvector((HIT), maskxM);                  \
    unsigned long long bits_ = __builtin_bit_cast(unsigned long long, m_); \
    __builtin_memcpy((OUT) + (I) / 8, &bits_, sizeof bits_);             \
  } while (0)

// simd_json_masks writes all five masks a JSON indexer wants from one pass.
//
// A two-stage JSON parser classifies the document five ways -- quotes,
// backslashes, brackets, control characters and whitespace -- and calling
// simd_mask_bits and friends once each means five passes over the input and
// five dispatches. This is one load per block and five predicate stores.
//
// The five outputs are five regions of one buffer, each ((n+63)/64)*8 bytes --
// whole 64-bit words, because that is how they are read -- in the order quote,
// escape, structural, control, whitespace. One pointer rather than
// five because the SysV argument registers run out at six and the caller wants
// n as well; laying them end to end also means the word loop that consumes them
// reads five words that are a fixed stride apart rather than five slices.
//
// want selects which are written, one bit each, low bit first, so a caller
// that only needs quotes and escapes does not pay for the other three stores.
// The regions are still laid out as though all five were present, so the
// offsets do not depend on what was asked for.
//
// The character sets are JSON's and are baked in: `"` and `\\`, the four
// brackets, below 0x20, and the four whitespace bytes JSON allows. A general
// version would take them as arguments and could not fold the comparisons.
void simd_json_masks(u8 *__restrict out, const u8 *__restrict b, isize n,
                     unsigned want) {
  // Word-aligned, not (n+7)/8. Every consumer of these masks reads them as
  // 64-bit words, and a region that stopped at the last byte would let the
  // final word of one run into the next. Seven bytes per region to avoid that.
  const isize stride = ((n + 63) / 64) * 8;
  u8 *__restrict q = out;
  u8 *__restrict e = out + stride;
  u8 *__restrict st = out + 2 * stride;
  u8 *__restrict ct = out + 3 * stride;
  u8 *__restrict ws = out + 4 * stride;

  isize i = 0;
  for (; i + MASK_BITS_LANES <= n; i += MASK_BITS_LANES) {
    u8xM v = *(const u8xM *)(b + i);
    if (want & 1) STORE_MASK_BITS(v == '"', q, i);
    if (want & 2) STORE_MASK_BITS(v == '\\', e, i);
    if (want & 4)
      STORE_MASK_BITS((v == '{') | (v == '}') | (v == '[') | (v == ']'), st, i);
    if (want & 8) STORE_MASK_BITS(v < 0x20, ct, i);
    if (want & 16)
      STORE_MASK_BITS((v == ' ') | (v == '\t') | (v == '\n') | (v == '\r'), ws, i);
  }
  // Everything from the last full block to the end of the last word, so a
  // consumer reading whole words sees zeros past the document.
  for (isize z = (n + 7) / 8; z < stride; z++) {
    if (want & 1) q[z] = 0;
    if (want & 2) e[z] = 0;
    if (want & 4) st[z] = 0;
    if (want & 8) ct[z] = 0;
    if (want & 16) ws[z] = 0;
  }
  for (; i < n; i++) {
    u8 c = b[i];
    isize o = i / 8;
    u8 bit = (u8)(1u << (i % 8));
    if (i % 8 == 0) {
      if (want & 1) q[o] = 0;
      if (want & 2) e[o] = 0;
      if (want & 4) st[o] = 0;
      if (want & 8) ct[o] = 0;
      if (want & 16) ws[o] = 0;
    }
    if ((want & 1) && c == '"') q[o] |= bit;
    if ((want & 2) && c == '\\') e[o] |= bit;
    if ((want & 4) && (c == '{' || c == '}' || c == '[' || c == ']')) st[o] |= bit;
    if ((want & 8) && c < 0x20) ct[o] |= bit;
    if ((want & 16) && (c == ' ' || c == '\t' || c == '\n' || c == '\r'))
      ws[o] |= bit;
  }
}

void simd_mask_bits(u8 *__restrict out, const u8 *__restrict b, u8 c,
                    isize n) {
  isize i = 0;
  for (; i + MASK_BITS_LANES <= n; i += MASK_BITS_LANES) {
    u8xM v = *(const u8xM *)(b + i);
    STORE_MASK_BITS(v == c, out, i);
  }
  for (; i < n; i++) {
    if (i % 8 == 0) out[i / 8] = 0;
    out[i / 8] |= (u8)((b[i] == c) << (i % 8));
  }
}

// simd_mask_bits_lt is simd_mask_bits for an inequality: the bit is set where
// the byte is below c.
//
// It exists for the range tests a set of eight cannot express. Finding the
// control characters a JSON string may not contain means asking about
// thirty-two values, which is four calls to the set form and one to this.
void simd_mask_bits_lt(u8 *__restrict out, const u8 *__restrict b, u8 c,
                       isize n) {
  isize i = 0;
  for (; i + MASK_BITS_LANES <= n; i += MASK_BITS_LANES) {
    u8xM v = *(const u8xM *)(b + i);
    STORE_MASK_BITS(v < c, out, i);
  }
  for (; i < n; i++) {
    if (i % 8 == 0) out[i / 8] = 0;
    out[i / 8] |= (u8)((b[i] < c) << (i % 8));
  }
}

// simd_mask_bits_any4 is simd_mask_bits_any for a set of at most four.
//
// The eight-byte form does eight compares and seven ORs whichever way the
// caller fills the set, because a vector unit has no way to do fewer. Four is
// the size real sets tend to be — a JSON parser wants `{}[]` and ` \t\n\r` —
// and halving the compares halves the kernel. Measured on 1 MiB, 41 GB/s for
// the eight-byte form against 74 GB/s for this one.
void simd_mask_bits_any4(u8 *__restrict out, const u8 *__restrict b,
                         unsigned chars, isize n) {
  const u8 c0 = (u8)chars, c1 = (u8)(chars >> 8), c2 = (u8)(chars >> 16),
           c3 = (u8)(chars >> 24);
  isize i = 0;
  for (; i + MASK_BITS_LANES <= n; i += MASK_BITS_LANES) {
    u8xM v = *(const u8xM *)(b + i);
    __typeof__(v == c0) hit = (v == c0) | (v == c1) | (v == c2) | (v == c3);
    STORE_MASK_BITS(hit, out, i);
  }
  for (; i < n; i++) {
    u8 x = b[i];
    unsigned h = (x == c0) | (x == c1) | (x == c2) | (x == c3);
    if (i % 8 == 0) out[i / 8] = 0;
    out[i / 8] |= (u8)(h << (i % 8));
  }
}

// simd_mask_bits_any is simd_mask_bits for a set of up to eight bytes, packed
// one per byte of chars the way simd_index_all_any takes it. A caller with
// fewer than eight repeats one: a duplicate compare is free.
void simd_mask_bits_any(u8 *__restrict out, const u8 *__restrict b,
                        unsigned long long chars, isize n) {
  const u8 c0 = (u8)chars, c1 = (u8)(chars >> 8), c2 = (u8)(chars >> 16),
           c3 = (u8)(chars >> 24), c4 = (u8)(chars >> 32),
           c5 = (u8)(chars >> 40), c6 = (u8)(chars >> 48),
           c7 = (u8)(chars >> 56);
  isize i = 0;
  for (; i + MASK_BITS_LANES <= n; i += MASK_BITS_LANES) {
    u8xM v = *(const u8xM *)(b + i);
    __typeof__(v == c0) hit = (v == c0) | (v == c1) | (v == c2) | (v == c3) |
                              (v == c4) | (v == c5) | (v == c6) | (v == c7);
    STORE_MASK_BITS(hit, out, i);
  }
  for (; i < n; i++) {
    u8 x = b[i];
    unsigned h = (x == c0) | (x == c1) | (x == c2) | (x == c3) | (x == c4) |
                 (x == c5) | (x == c6) | (x == c7);
    if (i % 8 == 0) out[i / 8] = 0;
    out[i / 8] |= (u8)(h << (i % 8));
  }
}
