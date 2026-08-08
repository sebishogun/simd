// Narrow floating-point conversions: float16 and bfloat16.
//
// Both are storage formats. Nothing here computes *in* them — a vector unit
// that has half-precision arithmetic has it for a different reason than this
// library exists for — and everything here is a widen or a narrow over a whole
// slice, which is what a caller doing inference or graphics actually needs.
//
// The conversions are written in integer arithmetic rather than with __fp16 or
// _Float16. Those types look like the obvious answer and are a trap: on a
// target without hardware support clang lowers them to __truncsfhf2 and
// __extendhfsf2, which are calls into compiler-rt, and a call is fatal here —
// Plan 9 assembly has no procedure linkage table. Bit manipulation compiles to
// shifts and selects on every target and needs nothing linked.
//
// bfloat16 is the top sixteen bits of a float32 and nothing else, so widening
// is a shift and narrowing is a rounded truncation. float16 has its own
// exponent width and bias, so both directions have to rebias and both have to
// handle the denormal and infinity ranges — branch-free, as selects, because a
// branch per element does not vectorize.

#include "goabi.h"

typedef long isize;
typedef unsigned short u16;
typedef unsigned int u32;

union f32bits {
  float f;
  u32 u;
};

static inline float bits_to_float(u32 u) {
  union f32bits v;
  v.u = u;
  return v.f;
}

static inline u32 float_to_bits(float f) {
  union f32bits v;
  v.f = f;
  return v.u;
}

// ---------- bfloat16 ----------

// simd_bf16_to_f32 widens. A bfloat16 is the high half of a float32, so this
// is exactly a shift — every value, including every NaN payload and both
// infinities, is preserved.
void simd_bf16_to_f32(float *__restrict d, const u16 *__restrict a, isize n) {
  for (isize i = 0; i < n; i++) d[i] = bits_to_float((u32)a[i] << 16);
}

// simd_f32_to_bf16 narrows, rounding to nearest even.
//
// The rounding is the whole content: truncating is one instruction and is
// biased, and a bias applied to every weight in a network is not a rounding
// error, it is a drift. Adding 0x7fff plus the low bit of the result before
// truncating is round-to-nearest-even in one add, which is the standard trick.
//
// A NaN must stay a NaN. Rounding can carry into the exponent and turn a NaN
// with a low mantissa into an infinity, so NaN is passed through with its high
// mantissa bit forced on — quieting it, which is what every hardware
// implementation of this conversion does.
void simd_f32_to_bf16(u16 *__restrict d, const float *__restrict a, isize n) {
  for (isize i = 0; i < n; i++) {
    u32 u = float_to_bits(a[i]);
    u32 rounded = u + 0x7fff + ((u >> 16) & 1);
    u32 isnan = ((u & 0x7f800000u) == 0x7f800000u) && (u & 0x007fffffu);
    u32 quiet = (u >> 16) | 0x0040u;
    d[i] = (u16)(isnan ? quiet : (rounded >> 16));
  }
}

// ---------- float16 ----------

// simd_f16_to_f32 widens.
//
// Three ranges and no branches. A normal number needs its exponent rebiased
// from 15 to 127, which is one add once the bits are shifted into position.
// An infinity or NaN has an exponent of all ones in both formats and needs a
// second, larger adjustment. A zero or denormal has no exponent to rebias at
// all, and is renormalized by adding a magic constant and subtracting it back
// as a float — the classic trick, which lets the hardware's own normalization
// do the work that a loop of shifts would otherwise do.
void simd_f16_to_f32(float *__restrict d, const u16 *__restrict a, isize n) {
  const u32 shifted_exp = 0x7c00u << 13; // exponent mask, in f32 position
  const u32 magic = 113u << 23;
  for (isize i = 0; i < n; i++) {
    u16 h = a[i];
    u32 o = (u32)(h & 0x7fffu) << 13;
    u32 exp = shifted_exp & o;
    u32 norm = o + ((127u - 15u) << 23);
    u32 infnan = norm + ((128u - 16u) << 23);
    float sub = bits_to_float(o + ((127u - 15u) << 23) + (1u << 23)) -
                bits_to_float(magic);
    u32 den = float_to_bits(sub);
    u32 out = exp == shifted_exp ? infnan : (exp == 0 ? den : norm);
    d[i] = bits_to_float(out | ((u32)(h & 0x8000u) << 16));
  }
}

// simd_f32_to_f16 narrows, rounding to nearest even.
//
// Harder than the widening in the way narrowing always is: the result may
// overflow to infinity, may fall into the denormal range, and the rounding may
// carry all the way into the exponent. All three are handled by arithmetic
// rather than by cases.
//
// The denormal path multiplies by a scale that places the value's bits where
// the rounding hardware will round them correctly, then subtracts the implied
// leading one. The normal path rebiases and adds the round-to-nearest-even
// term. Which one applies is a select on the exponent.
void simd_f32_to_f16(u16 *__restrict d, const float *__restrict a, isize n) {
  for (isize i = 0; i < n; i++) {
    u32 u = float_to_bits(a[i]);
    u32 sign = (u >> 16) & 0x8000u;
    u32 mag = u & 0x7fffffffu;

    // NaN stays NaN, quieted, and never rounds up into an infinity.
    u32 isnan = mag > 0x7f800000u;

    // Anything at or above 65520 rounds to infinity in float16.
    u32 isinf = mag >= 0x47800000u;

    // The normal path: rebias by (127-15) << 23 and round to nearest even.
    u32 shifted = mag - (112u << 23);
    u32 round = shifted + 0x0fffu + ((shifted >> 13) & 1u);
    u32 normal = round >> 13;

    // The denormal path. Adding 0.5 as a *float* forces the hardware to align
    // the value against an exponent of -1, whose mantissa step is 2^-24 —
    // exactly the step of a float16 denormal — and to round it there, which is
    // the rounding float16 wants. Subtracting the same 0.5 back as an integer
    // then leaves the half's bits and nothing else.
    //
    // The constant is 126<<23 and the two nearby wrong answers both look
    // right. Scaling by 2^-112 first rounds twice and loses the smallest
    // denormal; using 113<<23, the *threshold* constant, aligns against 2^-14
    // instead and leaves the result thirteen bits too high — 0x2000 where
    // 0x0001 belongs. Only exhaustive enumeration distinguishes them, which is
    // why the test walks all 65536 values rather than sampling.
    u32 den = float_to_bits(bits_to_float(mag) + 0.5f) - 0x3f000000u;

    u32 out = mag < 0x38800000u ? den : normal;
    out = isinf ? 0x7c00u : out;
    out = isnan ? 0x7e00u : out;
    d[i] = (u16)(sign | out);
  }
}

// ---------- affine quantization ----------
//
// The int8 quantization every inference runtime uses:
//
//	q = clamp(round(x / scale) + zero_point, -128, 127)
//	x = (q - zero_point) * scale
//
// It is here because it is the most commonly reached-for SIMD operation this
// library did not have, and its absence was a scope decision rather than a
// technical one.
//
// # The two things implementations disagree about
//
// **Rounding.** ONNX, PyTorch and TFLite all specify round-half-to-EVEN, and
// C's rintf() does that under the default rounding mode. lrintf() would too
// but returns a long and does not vectorize; nearbyintf() is rintf() without
// the inexact flag and is equivalent here. Writing (int)(x + 0.5f) instead —
// which is what a naive implementation does — rounds half away from zero, and
// disagrees on exactly the values quantization hits most often, the .5 cases
// that a symmetric scale produces in quantity.
//
// **Clamping order.** The clamp must come after the zero-point is added, not
// before. Adding first can push a value that was in range out of it, and a
// wrapped int8 in an inference pipeline is not a small error — it is a sign
// flip.
//
// The reciprocal is NOT precomputed. Multiplying by 1/scale is one instruction
// cheaper per element and gives different bits: 3.0f/5.0f is 0.6f but
// 3.0f*(1/5.0f) is 0.6000000238f, which rounds to a different integer for
// values sitting near a .5 boundary. The division is what the reference
// implementations specify, so the division is what this does.
// round_half_even_f32 is rintf without the call.
//
// __builtin_rintf lowers to a hardware instruction on most targets and to a
// libm CALL on two of them -- amd64/sse2, which has no roundps before SSE4.1,
// and ppc64le/vsx. A call is fatal here: Plan 9 assembly has no procedure
// linkage table, and the generator rejects the kernel rather than emit one.
//
// The magic-number identity gives the same answer with add, subtract and
// select only. Adding 2^23 to a float32 forces every fractional bit out under
// the default rounding mode, and the default IS round-half-to-even, so
// (t + 2^23) - 2^23 rounds exactly as rintf does. Above 2^23 a float32 is
// already integral and is returned unchanged.
//
// Verified against rintf over 64016 values -- a dense grid through the whole
// interesting range plus every exact .5, both signed zeros, denormals and the
// 2^23 boundary itself -- with zero mismatches. It is an identity, not an
// approximation, which is why it may stand in for the specified rounding.
static inline float round_half_even_f32(float x) {
  const float m = 8388608.0f; /* 2^23 */
  float t = __builtin_fabsf(x);
  float r = (t + m) - m;
  r = __builtin_copysignf(r, x);
  return t >= m ? x : r;
}

void simd_quantize_i8(signed char *__restrict d, const float *__restrict a,
                      float scale, int zero_point, isize n) {
  for (isize i = 0; i < n; i++) {
    float r = round_half_even_f32(a[i] / scale);
    // The intermediate is float, not int: a value far out of range would
    // overflow the conversion, which is undefined behaviour rather than a
    // saturating one, so it is clamped while still floating point.
    float z = r + (float)zero_point;
    z = z < -128.0f ? -128.0f : z;
    z = z > 127.0f ? 127.0f : z;
    d[i] = (signed char)z;
  }
}

void simd_dequantize_i8(float *__restrict d, const signed char *__restrict a,
                        float scale, int zero_point, isize n) {
  for (isize i = 0; i < n; i++) {
    d[i] = (float)((int)a[i] - zero_point) * scale;
  }
}

// The unsigned variant, which is what TFLite and most mobile runtimes use.
void simd_quantize_u8(unsigned char *__restrict d, const float *__restrict a,
                      float scale, int zero_point, isize n) {
  for (isize i = 0; i < n; i++) {
    float r = round_half_even_f32(a[i] / scale);
    float z = r + (float)zero_point;
    z = z < 0.0f ? 0.0f : z;
    z = z > 255.0f ? 255.0f : z;
    d[i] = (unsigned char)z;
  }
}

void simd_dequantize_u8(float *__restrict d, const unsigned char *__restrict a,
                        float scale, int zero_point, isize n) {
  for (isize i = 0; i < n; i++) {
    d[i] = (float)((int)a[i] - zero_point) * scale;
  }
}

// ---------- zigzag ----------
//
// Zigzag maps a signed integer onto an unsigned one so that small magnitudes
// of either sign become small unsigned values: 0, -1, 1, -2, 2 become 0, 1, 2,
// 3, 4. That is what makes a varint of a negative number short, so this is the
// transform protobuf, Avro and every delta-encoded column store applies before
// varint encoding.
//
// encode is (x << 1) ^ (x >> (W-1)) and decode is (u >> 1) ^ -(u & 1).
//
// Both are written entirely in the unsigned domain, for the reason csrc/wrap.h
// gives at length: shifting a negative value left is undefined in C, and an
// optimizer entitled to assume it never happens is entitled to delete the
// branch that handles it. The sign mask is `x < 0 ? ~0 : 0` rather than an
// arithmetic right shift of a signed value, which C leaves
// implementation-defined; clang folds it back to one arithmetic shift, so the
// vector code is the same and the source no longer depends on a guarantee the
// standard does not give.
//
// Both directions are exact and total: every value round-trips, including the
// most negative one, which is the case a naive `-x` formulation gets wrong.
#define ZIGZAG(T, UT, SUF)                                                 \
  void simd_zigzag_encode_##SUF(UT *__restrict d, const T *__restrict a,   \
                                isize n) {                                 \
    for (isize i = 0; i < n; i++) {                                        \
      UT u = (UT)a[i];                                                     \
      UT m = (UT)(a[i] < 0 ? ~(UT)0 : (UT)0);                              \
      d[i] = (UT)((UT)(u << 1) ^ m);                                       \
    }                                                                      \
  }                                                                        \
  void simd_zigzag_decode_##SUF(T *__restrict d, const UT *__restrict a,   \
                                isize n) {                                 \
    for (isize i = 0; i < n; i++) {                                        \
      UT u = a[i];                                                         \
      d[i] = (T)((UT)(u >> 1) ^ (UT)((UT)0 - (UT)(u & 1)));                \
    }                                                                      \
  }

ZIGZAG(signed char, unsigned char, i8)
ZIGZAG(short, unsigned short, i16)
ZIGZAG(int, unsigned int, i32)
ZIGZAG(long long, unsigned long long, i64)

// ---------- varint widths ----------
//
// How many LEB128 bytes each value needs, and how many the whole slice needs.
//
// The *emission* of a varint stream is a serial problem — where value i lands
// depends on the width of every value before it — and no rewriting fixes that;
// it is the same loop-carried address dependency compress.c describes. What
// vectorizes is the question asked first: how wide is each one.
//
// That is worth having on its own. Knowing the exact encoded size before
// writing a byte is what lets an encoder allocate once instead of growing, and
// the per-element widths prefix-summed give every value's offset, which is
// what turns the serial emit into an independent per-value write.
//
// The width is written as a sum of comparisons rather than from a
// leading-zero count. Both are correct; only this one vectorizes everywhere.
// __builtin_clz lowers to llvm.ctlz, which has a vector form on AVX-512 and
// on NEON but not on SSE2 or AVX2, where LLVM scalarizes it lane by lane and
// gives back a kernel slower than the loop it replaced. Four unsigned
// comparisons and three adds are the same on every target here.
static inline unsigned varint_len_u32(u32 x) {
  return 1u + (x >= 0x80u) + (x >= 0x4000u) + (x >= 0x200000u) +
         (x >= 0x10000000u);
}

static inline unsigned varint_len_u64(unsigned long long x) {
  return 1u + (x >= 0x80ull) + (x >= 0x4000ull) + (x >= 0x200000ull) +
         (x >= 0x10000000ull) + (x >= 0x800000000ull) +
         (x >= 0x40000000000ull) + (x >= 0x2000000000000ull) +
         (x >= 0x100000000000000ull) + (x >= 0x8000000000000000ull);
}

#define VARINT_LEN(UT, SUF)                                                \
  void simd_varint_len_##SUF(int *__restrict d, const UT *__restrict a,    \
                             isize n) {                                    \
    for (isize i = 0; i < n; i++) d[i] = (int)varint_len_##SUF(a[i]);      \
  }

VARINT_LEN(u32, u32)
VARINT_LEN(unsigned long long, u64)

// The total is folded through sixty-four-bit lanes rather than the
// thirty-two-bit ones the byte counters in fold.h use, because this
// accumulator holds up to ten times the element count rather than at most one
// per element: a slice large enough to be worth calling this on is already
// within a factor of four of overflowing a u32.
//
// The 8 -> 4 -> 2 -> 1 fold is the library's fixed combine tree, so the answer
// does not depend on the vector length of the machine. Integer addition is
// associative, so the portable reference's plain loop gives the same value —
// unlike the float reductions, where the tree is the contract.
typedef unsigned long long u64x8 __attribute__((ext_vector_type(8), aligned(1)));

#define VARINT_SIZE(UT, SUF)                                               \
  void simd_varint_size_##SUF(isize *__restrict out,                       \
                              const UT *__restrict a, isize n) {           \
    u64x8 acc = 0;                                                         \
    isize i = 0;                                                           \
    for (; i + 8 <= n; i += 8) {                                           \
      u64x8 v;                                                             \
      for (int j = 0; j < 8; j++) v[j] = varint_len_##SUF(a[i + j]);       \
      acc += v;                                                            \
    }                                                                      \
    unsigned long long r[8];                                               \
    *(u64x8 *)r = acc;                                                     \
    for (int w = 4; w >= 1; w /= 2)                                        \
      for (int j = 0; j < w; j++) r[j] += r[j + w];                        \
    isize s = (isize)r[0];                                                 \
    for (; i < n; i++) s += (isize)varint_len_##SUF(a[i]);                 \
    *out = s;                                                              \
  }

VARINT_SIZE(u32, u32)
VARINT_SIZE(unsigned long long, u64)

// ---------- per-channel quantization ----------
//
// One scale and zero point per output channel rather than one per tensor.
//
// This is what real inference uses for weights, and the reason is distribution
// rather than precision: a convolution's output channels are trained
// independently and their weight ranges differ by an order of magnitude or
// more, so a single tensor-wide scale is set by the widest channel and wastes
// most of the int8 range on every other one. Per-channel typically recovers
// one to two bits of effective precision on the narrow channels for no extra
// storage beyond the scale vector.
//
// The layout is the one every runtime uses: elements grouped by channel, with
// `inner` consecutive elements sharing a channel. For a weight tensor laid out
// [out_channels][in_channels * kh * kw] that makes inner the size of one
// filter, which is exactly how the data already sits.
//
// The vectorization is the same elementwise pass as the per-tensor form. What
// changes is that the scale is loop-invariant within a channel rather than
// across the whole call, so the loop is nested and the inner one is what the
// vectorizer sees — unchanged from the tensor-wide kernel, which is the point
// of splitting it this way rather than gathering a scale per element.
#define QUANT_PER_CHANNEL(T, SUF, LO, HI)                                 \
  void simd_quantize_per_channel_##SUF(                                   \
      T *__restrict d, const float *__restrict a,                         \
      const float *__restrict scale, const int *__restrict zero_point,    \
      isize channels, isize inner) {                                      \
    for (isize c = 0; c < channels; c++) {                                \
      float s = scale[c];                                                 \
      float z = (float)zero_point[c];                                     \
      const float *ac = a + c * inner;                                    \
      T *dc = d + c * inner;                                              \
      for (isize i = 0; i < inner; i++) {                                 \
        float r = round_half_even_f32(ac[i] / s);                         \
        float q = r + z;                                                  \
        q = q < (LO) ? (LO) : q;                                          \
        q = q > (HI) ? (HI) : q;                                          \
        dc[i] = (T)q;                                                     \
      }                                                                   \
    }                                                                     \
  }

QUANT_PER_CHANNEL(signed char, i8, -128.0f, 127.0f)
QUANT_PER_CHANNEL(unsigned char, u8, 0.0f, 255.0f)

// The inverse, same layout.
#define DEQUANT_PER_CHANNEL(T, SUF)                                       \
  void simd_dequantize_per_channel_##SUF(                                 \
      float *__restrict d, const T *__restrict a,                         \
      const float *__restrict scale, const int *__restrict zero_point,    \
      isize channels, isize inner) {                                      \
    for (isize c = 0; c < channels; c++) {                                \
      float s = scale[c];                                                 \
      int z = zero_point[c];                                              \
      const T *ac = a + c * inner;                                        \
      float *dc = d + c * inner;                                          \
      for (isize i = 0; i < inner; i++) dc[i] = (float)((int)ac[i] - z) * s; \
    }                                                                     \
  }

DEQUANT_PER_CHANNEL(signed char, i8)
DEQUANT_PER_CHANNEL(unsigned char, u8)

// ---------- fp8 ----------
//
// Two 8-bit float formats, both in the OCP "OFP8" specification and both in
// current use for inference and increasingly for training:
//
//   e4m3  1 sign, 4 exponent, 3 mantissa, bias 7.  Weights and activations.
//   e5m2  1 sign, 5 exponent, 2 mantissa, bias 15. Gradients.
//
// They trade the same eight bits differently: e5m2 has the range of a float16
// and two mantissa bits, e4m3 has three mantissa bits and a range that stops
// at 448.
//
// THE ONE INTEROPERABILITY DECISION, stated rather than left implicit. e4m3
// has two incompatible definitions:
//
//   OCP e4m3 (this one)  no infinity. Exponent 1111 with mantissa 111 is NaN;
//                        every other 1111 encoding is a finite number, which
//                        is what buys the 448 maximum.
//   NVIDIA e4m3fn        same thing, and the name says so — "fn" is
//                        finite-not-nan.
//   A hypothetical IEEE-shaped e4m3 would spend 1111.000 on infinity and cap
//   at 240. Nothing ships it.
//
// So: e4m3 here has NO infinity. An input infinity saturates to +-448 and a
// NaN maps to 0x7f or 0xff. e5m2 IS IEEE-shaped — it has infinities and NaNs
// in the usual places — so the two formats genuinely behave differently at the
// top of their range and that is not a bug in either.
//
// Rounding is to nearest even in both directions, matching every other
// narrowing conversion here and the hardware instructions on the machines that
// have them.
#define F8_ROUND(bits, shift)                                             \
  do {                                                                    \
    unsigned rest_ = (bits) & ((1u << (shift)) - 1u);                     \
    unsigned half_ = 1u << ((shift) - 1);                                 \
    (bits) >>= (shift);                                                   \
    if (rest_ > half_ || (rest_ == half_ && ((bits) & 1u))) (bits)++;      \
  } while (0)

void simd_f8e4m3_to_f32(float *__restrict d, const unsigned char *__restrict a,
                        isize n) {
  for (isize i = 0; i < n; i++) {
    unsigned u = a[i];
    unsigned sign = (u & 0x80u) << 24;
    unsigned exp = (u >> 3) & 0x0fu;
    unsigned man = u & 0x07u;
    if (exp == 0x0fu && man == 0x07u) {
      d[i] = bits_to_float(sign | 0x7fc00000u); /* the only NaN encoding */
    } else if (exp == 0) {
      if (man == 0) {
        d[i] = bits_to_float(sign);
      } else {
        /* Denormal: normalise by hand, as f16_to_f32 does. */
        int e = -6;
        while ((man & 0x08u) == 0) {
          man <<= 1;
          e--;
        }
        man &= 0x07u;
        d[i] = bits_to_float(sign | (unsigned)(e + 127) << 23 | man << 20);
      }
    } else {
      d[i] = bits_to_float(sign | (exp + 127u - 7u) << 23 | man << 20);
    }
  }
}

void simd_f32_to_f8e4m3(unsigned char *__restrict d,
                        const float *__restrict a, isize n) {
  for (isize i = 0; i < n; i++) {
    unsigned u = float_to_bits(a[i]);
    unsigned sign = (u >> 24) & 0x80u;
    unsigned mag = u & 0x7fffffffu;
    if (mag > 0x7f800000u) {
      d[i] = (unsigned char)(sign | 0x7fu); /* NaN */
    } else if (mag >= 0x43e00000u) {
      /* 448 is the largest finite e4m3; there is no infinity to overflow to. */
      d[i] = (unsigned char)(sign | 0x7eu);
    } else if (mag < 0x3a800000u) {
      d[i] = (unsigned char)sign; /* below half the smallest denormal */
    } else if (mag < 0x3c800000u) {
      /* Denormal in e4m3: below 2^-6, the smallest normal. The constant is
         2^-6 and not 2^-4 — an exhaustive round trip over all 256 encodings
         caught that immediately, because every value between the two took the
         denormal path and came back one step too large. */
      int e = (int)(mag >> 23) - 127;
      unsigned m = (mag & 0x7fffffu) | 0x800000u;
      unsigned shift = (unsigned)(-e - 6 + 20);
      F8_ROUND(m, shift);
      d[i] = (unsigned char)(sign | m);
    } else {
      unsigned e = (mag >> 23) - 127u + 7u;
      unsigned m = mag & 0x7fffffu;
      unsigned r = (e << 3) | (m >> 20);
      unsigned rest = m & 0xfffffu;
      if (rest > 0x80000u || (rest == 0x80000u && (r & 1u))) r++;
      /* Rounding can carry into the NaN encoding; saturate instead. */
      d[i] = (unsigned char)(sign | (r > 0x7eu ? 0x7eu : r));
    }
  }
}

void simd_f8e5m2_to_f32(float *__restrict d, const unsigned char *__restrict a,
                        isize n) {
  for (isize i = 0; i < n; i++) {
    unsigned u = a[i];
    unsigned sign = (u & 0x80u) << 24;
    unsigned exp = (u >> 2) & 0x1fu;
    unsigned man = u & 0x03u;
    if (exp == 0x1fu) {
      /* IEEE-shaped: infinity and NaN both live here. */
      d[i] = bits_to_float(sign | 0x7f800000u | (man ? 0x400000u | man << 21 : 0));
    } else if (exp == 0) {
      if (man == 0) {
        d[i] = bits_to_float(sign);
      } else {
        int e = -14;
        while ((man & 0x04u) == 0) {
          man <<= 1;
          e--;
        }
        man &= 0x03u;
        d[i] = bits_to_float(sign | (unsigned)(e + 127) << 23 | man << 21);
      }
    } else {
      d[i] = bits_to_float(sign | (exp + 127u - 15u) << 23 | man << 21);
    }
  }
}

void simd_f32_to_f8e5m2(unsigned char *__restrict d,
                        const float *__restrict a, isize n) {
  for (isize i = 0; i < n; i++) {
    unsigned u = float_to_bits(a[i]);
    unsigned sign = (u >> 24) & 0x80u;
    unsigned mag = u & 0x7fffffffu;
    if (mag > 0x7f800000u) {
      d[i] = (unsigned char)(sign | 0x7fu); /* NaN, quieted */
    } else if (mag >= 0x47700000u) {
      d[i] = (unsigned char)(sign | 0x7cu); /* overflows to infinity */
    } else if (mag < 0x37000000u) {
      d[i] = (unsigned char)sign;
    } else if (mag < 0x38800000u) {
      int e = (int)(mag >> 23) - 127;
      unsigned m = (mag & 0x7fffffu) | 0x800000u;
      unsigned shift = (unsigned)(-e - 14 + 21);
      F8_ROUND(m, shift);
      d[i] = (unsigned char)(sign | m);
    } else {
      unsigned e = (mag >> 23) - 127u + 15u;
      unsigned m = mag & 0x7fffffu;
      unsigned r = (e << 2) | (m >> 21);
      unsigned rest = m & 0x1fffffu;
      if (rest > 0x100000u || (rest == 0x100000u && (r & 1u))) r++;
      d[i] = (unsigned char)(sign | (r > 0x7bu ? 0x7cu : r));
    }
  }
}

// ---------- bit packing ----------
//
// Pack uint32 values into a dense bitstream of `bits` bits each, and unpack
// them again. This is the representation Parquet, Arrow, Lucene and every
// column store use after delta encoding: once the deltas are small, storing
// them in 32 bits each wastes most of the file.
//
// The width is a run-time parameter rather than a template, because a column
// store picks it per block from the actual maximum. That costs a variable
// shift per element, which every target has.
//
// THE SHAPE THAT MAKES THIS VECTORIZE. The obvious loop walks a bit cursor and
// writes across word boundaries, which is a serial dependence through the
// cursor and does not vectorize at all. This one is written the other way
// round: for each OUTPUT word, work out which inputs contribute to it. Every
// output word is then independent, and the loop over them is a plain
// elementwise pass.
//
// The cost is that each output word re-reads up to two inputs, which is
// cheaper than it sounds because they are adjacent and in cache.
//
// bits must be 1..32. Zero would be a degenerate encoding with no output and
// is rejected by the caller rather than special-cased here.
// It takes both lengths. The output is SHORTER than the input — that is the
// entire point of packing — and the generated guard's default is to clamp
// every slice to the shortest, which silently truncated the input to the
// number of output words. Two lengths turn the clamping off; see the note on
// countLens in the emitter, and simd_sum_lanes, which has the same shape for
// the same reason.
void simd_bitpack_u32(unsigned int *__restrict d, const unsigned int *__restrict a,
                      int bits, isize n, isize ndst) {
  // Number of output words: ceil(n*bits / 32).
  isize words = (n * (isize)bits + 31) / 32;
  if (ndst < words) return;
  for (isize w = 0; w < words; w++) {
    // The bit position this output word starts at, and the first input that
    // contributes to it.
    isize startBit = w * 32;
    isize first = startBit / bits;
    unsigned int acc = 0;
    // At most 33 inputs can touch one output word (bits == 1), so the loop is
    // bounded and clang can unroll the common widths.
    for (isize i = first; i < n; i++) {
      isize lo = i * (isize)bits;
      if (lo >= startBit + 32) break;
      isize hi = lo + bits;
      if (hi <= startBit) continue;
      unsigned int v = a[i] & (bits == 32 ? 0xffffffffu : ((1u << bits) - 1u));
      if (lo >= startBit) {
        acc |= v << (unsigned)(lo - startBit);
      } else {
        acc |= v >> (unsigned)(startBit - lo);
      }
    }
    d[w] = acc;
  }
}

// The inverse. Each output element is independent — it reads the one or two
// words its bits straddle — so this is a plain elementwise pass and vectorizes
// where the packing direction does not.
void simd_bitunpack_u32(unsigned int *__restrict d, const unsigned int *__restrict a,
                        int bits, isize n, isize nsrc) {
  if (nsrc < (n * (isize)bits + 31) / 32 + 1) return;
  unsigned int mask = bits == 32 ? 0xffffffffu : ((1u << bits) - 1u);
  for (isize i = 0; i < n; i++) {
    isize lo = i * (isize)bits;
    isize w = lo / 32;
    unsigned shift = (unsigned)(lo % 32);
    unsigned int v = a[w] >> shift;
    // The second word contributes only when the value straddles a boundary.
    // Reading it unconditionally would be simpler and would run off the end of
    // the buffer on the last element.
    if (shift + (unsigned)bits > 32) v |= a[w + 1] << (32u - shift);
    d[i] = v & mask;
  }
}

// ---------- varint decode ----------

// simd_varint_decode_u64: LEB128 varints from src into dst until either
// runs out. One eight-byte load per value instead of a byte loop: the
// continuation bits of the loaded word give the length by count-trailing-
// zeros of their complement, and the payload assembles with SWAR masking
// -- the branch per byte becomes one branch per value. Values longer than
// eight bytes (needing the ninth and tenth) take the byte path; malformed
// input -- a varint that never terminates within ten bytes, or truncation
// -- stops the walk. out[0] is values written, out[1] bytes consumed.
void simd_varint_decode_u64(isize *__restrict out, unsigned long long *__restrict dst,
                            isize dcap, const unsigned char *__restrict src, isize n) {
  isize d = 0, i = 0;
  while (d < dcap && i < n) {
    if (i + 8 <= n) {
      unsigned long long w;
      __builtin_memcpy(&w, src + i, 8);
      unsigned long long cont = w & 0x8080808080808080ull;
      if ((cont & 0x80) == 0) {
        // one byte
        dst[d++] = w & 0x7f;
        i += 1;
        continue;
      }
      unsigned long long stops = ~cont & 0x8080808080808080ull;
      if (stops != 0) {
        int len = (__builtin_ctzll(stops) >> 3) + 1;
        // Gather 7-bit groups from len bytes.
        unsigned long long v = 0;
        for (int k = len - 1; k >= 0; k--)
          v = (v << 7) | ((w >> (8 * k)) & 0x7f);
        dst[d++] = v;
        i += len;
        continue;
      }
    }
    // Slow path: near the end, or a varint longer than eight bytes.
    unsigned long long v = 0;
    int shift = 0, len = 0;
    for (;;) {
      if (i + len >= n || len >= 10) {
        out[0] = d;
        out[1] = i;
        return;
      }
      unsigned char b = src[i + len];
      v |= (unsigned long long)(b & 0x7f) << shift;
      len++;
      if ((b & 0x80) == 0) break;
      shift += 7;
    }
    dst[d++] = v;
    i += len;
  }
  out[0] = d;
  out[1] = i;
}
