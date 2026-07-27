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
