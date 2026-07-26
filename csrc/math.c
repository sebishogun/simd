// Transcendental kernels: exp, log, the trigonometric family and their
// inverses, as straight-line polynomial evaluation.
//
// # Why these cannot call libm
//
// A Plan 9 assembly file has no procedure linkage table, so a kernel that
// emits `call exp` cannot be assembled into this library at all. That rules
// out the obvious implementation and is why every function here is a range
// reduction followed by a polynomial, with no branch that a vector unit
// cannot express as a select. It is also why __builtin_exp and friends are
// absent: C requires them to set errno, so clang lowers them to a call even at
// -O3. Only the __builtin_elementwise_* family, which has no errno contract,
// stays inline — and it covers sqrt, abs, min and max, not the transcendentals.
//
// # The accuracy contract
//
// Rule 6 in package kernel: these guarantee a stated ULP bound rather than
// bit identity, because the polynomial that is correct to 1 ULP in float32 is
// not the one that is correct to 1 ULP in float64, so no single evaluation
// order reproduces both. The bound is measured against the Go standard
// library by TestTranscendentalULP and reported there; it is not asserted from
// theory.
//
// What still holds exactly:
//
//   - The odd functions are exactly odd. sin(-x) is the negation of sin(x)
//     and atan(-x) of atan(x), because each is fitted as x*P(x*x) and the
//     sign rides through untouched.
//   - A NaN in gives a NaN out, and an infinity goes where IEEE 754 says.
//
// What does not, and it is worth being precise about because the algebraic
// kernels do promise it: two tiers agree here only to within the stated
// bound, not bit for bit. Where both tiers have the kernel the evaluation
// order is identical and they do in fact agree exactly, but a tier that could
// not compile one falls back to Go's math, which is a different algorithm.
// The baseline x86-64 tier is the live example: it reaches its constant pools
// with legacy SSE instructions that require 16-byte alignment, so it keeps the
// portable path for every function here.
//
// Measured against glibc rather than assumed: at the points where these
// disagree most with Go's standard library — acos near 1, asin near 1, log2
// just above 1 — the value here is glibc's to the bit, and it is Go that is
// losing digits to a cancelling identity (pi/2 - asin(x) for acos, and a
// Frexp-based log2). See TestTranscendentalULP and TestAcosNearOne.
//
// # Shape rules
//
// No branches on data, only ternaries, which every backend lowers to a select.
// No table lookups: a table is a constant pool, and a constant pool on a RISC
// target is an address built from a high/low instruction pair that this
// generator cannot rewrite. Every constant is either an immediate or a
// literal the compiler can materialize.

#include "goabi.h"
#include "poly.h"

typedef long isize;

// ---------- bit-level helpers ----------
//
// always_inline is not a hint here. A call that survived would be a call into
// nothing, since these are compiled to a bare .text blob with no relocations
// resolved; the generator's verifier rejects a kernel with an undefined
// symbol, but inlining is what makes sure there is never one to reject.

#define AI __attribute__((always_inline)) static inline

AI double bits_to_f64(unsigned long long u) {
  union {
    unsigned long long u;
    double d;
  } v = {.u = u};
  return v.d;
}

AI unsigned long long f64_to_bits(double d) {
  union {
    double d;
    unsigned long long u;
  } v = {.d = d};
  return v.u;
}

AI float bits_to_f32(unsigned int u) {
  union {
    unsigned int u;
    float f;
  } v = {.u = u};
  return v.f;
}

AI unsigned int f32_to_bits(float f) {
  union {
    float f;
    unsigned int u;
  } v = {.f = f};
  return v.u;
}

// pow2_f64 builds 2^k for |k| <= 1023 by writing the exponent field directly.
// Nothing rounds: the result is a power of two, which every binary format
// holds exactly.
AI double pow2_f64(int k) {
  return bits_to_f64((unsigned long long)(k + 1023) << 52);
}

AI float pow2_f32(int k) { return bits_to_f32((unsigned int)(k + 127) << 23); }

// round_f64 rounds to the nearest integer, ties to even, without needing a
// rounding instruction.
//
// Adding and then subtracting 2^52+2^51 forces the significand to drop every
// fractional bit under the current rounding mode, which Go never changes from
// round-to-nearest-even. The alternative — __builtin_roundeven — has no
// instruction on baseline x86-64 and lowers to a libm call there, which would
// lose the whole sse2 tier.
//
// Valid for |x| < 2^51, which every caller here satisfies: exp overflows past
// 1024 and the trigonometric reduction handles large arguments before this.
#define F64_SHIFT 0x1.8p52
#define F32_SHIFT 0x1.8p23f

AI double round_f64(double x) { return (x + F64_SHIFT) - F64_SHIFT; }
AI float round_f32(float x) { return (x + F32_SHIFT) - F32_SHIFT; }

// Splitting a constant into a high part with trailing zero bits and a low
// remainder is what makes Cody-Waite reduction work: k*hi is then exact, so
// the only rounding in x - k*hi - k*lo is in the second subtraction.
#define LN2_HI 0x1.62e42fefa3800p-1
#define LN2_LO 0x1.ef35793c76730p-45
#define LOG2E 0x1.71547652b82fep+0
#define LN2 0x1.62e42fefa39efp-1
#define LOG10_2 0x1.34413509f79ffp-2
#define LOG2_10 0x1.a934f0979a371p+1

#define LN2_HI_F 0x1.62e400p-1f
#define LN2_LO_F 0x1.7f7d1cp-20f
#define LOG2E_F 0x1.715476p+0f
#define LN2_F 0x1.62e430p-1f
#define LOG10_2_F 0x1.344136p-2f
#define LOG2_10_F 0x1.a934f0p+1f

// pi/2 in three pieces, for the trigonometric reduction. Three is what it
// takes to keep the reduced argument accurate out to |x| ~ 1e14; beyond that
// the answer is dominated by how many bits of pi the input itself implies,
// and the kernels defer to the portable path rather than pretend otherwise.
#define PIO2_HI 0x1.921fb54400000p+0
#define PIO2_MID 0x1.0b4611a626331p-34
#define PIO2_LO 0x1.3198a2e037073p-69
#define TWO_OVER_PI 0x1.45f306dc9c883p-1

#define PIO2_HI_F 0x1.920000p+0f
#define PIO2_MID_F 0x1.fb4000p-12f
#define PIO2_LO_F 0x1.4442d2p-24f
#define TWO_OVER_PI_F 0x1.45f306p-1f

#define PI_F 0x1.921fb6p+1f
#define PI 0x1.921fb54442d18p+1
#define PIO2 0x1.921fb54442d18p+0
#define PIO4 0x1.921fb54442d18p-1
#define PIO2_F 0x1.921fb6p+0f
#define PIO4_F 0x1.921fb6p-1f

#define INF bits_to_f64(0x7ff0000000000000ull)
#define QNAN bits_to_f64(0x7ff8000000000000ull)
#define INF_F bits_to_f32(0x7f800000u)
#define QNAN_F bits_to_f32(0x7fc00000u)

// Horner evaluation, written out rather than looped.
//
// POLYn(x, C) evaluates the degree-(n-1) polynomial whose coefficients are
// C_0 through C_(n-1), innermost term first. The nesting is explicit so the
// dependency chain is fixed: a compiler free to reassociate it could produce
// a different answer at a different vector width, which is exactly what the
// tiers-agree-with-each-other property forbids.
#define POLY1(x, C) (C##_0)
#define POLY2(x, C) (C##_0 + (x) * (C##_1))
#define POLY3(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2)))
#define POLY4(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3))))
#define POLY5(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4)))))
#define POLY6(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5))))))
#define POLY7(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6)))))))
#define POLY8(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7))))))))
#define POLY9(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7 + (x) * (C##_8)))))))))
#define POLY10(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7 + (x) * (C##_8 + (x) * (C##_9))))))))))
#define POLY11(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7 + (x) * (C##_8 + (x) * (C##_9 + (x) * (C##_10)))))))))))
#define POLY12(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7 + (x) * (C##_8 + (x) * (C##_9 + (x) * (C##_10 + (x) * (C##_11))))))))))))
#define POLY13(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7 + (x) * (C##_8 + (x) * (C##_9 + (x) * (C##_10 + (x) * (C##_11 + (x) * (C##_12)))))))))))))
#define POLY14(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7 + (x) * (C##_8 + (x) * (C##_9 + (x) * (C##_10 + (x) * (C##_11 + (x) * (C##_12 + (x) * (C##_13))))))))))))))
#define POLY15(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7 + (x) * (C##_8 + (x) * (C##_9 + (x) * (C##_10 + (x) * (C##_11 + (x) * (C##_12 + (x) * (C##_13 + (x) * (C##_14)))))))))))))))
#define POLY16(x, C) (C##_0 + (x) * (C##_1 + (x) * (C##_2 + (x) * (C##_3 + (x) * (C##_4 + (x) * (C##_5 + (x) * (C##_6 + (x) * (C##_7 + (x) * (C##_8 + (x) * (C##_9 + (x) * (C##_10 + (x) * (C##_11 + (x) * (C##_12 + (x) * (C##_13 + (x) * (C##_14 + (x) * (C##_15))))))))))))))))

// ---------- exp ----------

AI double exp_f64(double x) {
  // Reduce to x = k*ln2 + r with |r| <= ln2/2, evaluate e^r, then scale by
  // an exact power of two.
  double k = round_f64(x * LOG2E);
  double r = (x - k * LN2_HI) - k * LN2_LO;
  double p = POLY13(r, EXPP);
  int ki = (int)k;
  // The scale is split in two so the exponent field never has to hold a value
  // it cannot: k reaches 1024 at the overflow boundary and -1075 where the
  // result is denormal, and either would be an invalid exponent on its own.
  // Two halves also make underflow to a denormal come out right, because the
  // first multiply lands in normal range and the second rounds once.
  double s = p * pow2_f64(ki / 2) * pow2_f64(ki - ki / 2);
  s = x > 709.782712893384 ? INF : s;
  s = x < -745.1332191019411 ? 0.0 : s;
  return x != x ? x : s;
}

AI float exp_f32(float x) {
  float k = round_f32(x * LOG2E_F);
  float r = (x - k * LN2_HI_F) - k * LN2_LO_F;
  float p = POLY7(r, EXPPF);
  int ki = (int)k;
  float s = p * pow2_f32(ki / 2) * pow2_f32(ki - ki / 2);
  s = x > 88.72283f ? INF_F : s;
  s = x < -103.97208f ? 0.0f : s;
  return x != x ? x : s;
}

AI double exp2_f64(double x) {
  double k = round_f64(x);
  double r = x - k;
  double p = POLY14(r, EXP2P);
  int ki = (int)k;
  double s = p * pow2_f64(ki / 2) * pow2_f64(ki - ki / 2);
  s = x >= 1024.0 ? INF : s;
  s = x < -1075.0 ? 0.0 : s;
  return x != x ? x : s;
}

AI float exp2_f32(float x) {
  float k = round_f32(x);
  float r = x - k;
  float p = POLY8(r, EXP2PF);
  int ki = (int)k;
  float s = p * pow2_f32(ki / 2) * pow2_f32(ki - ki / 2);
  s = x >= 128.0f ? INF_F : s;
  s = x < -150.0f ? 0.0f : s;
  return x != x ? x : s;
}

// ---------- log ----------
//
// log2_frac returns log2 of x's mantissa reduced to [sqrt(2)/2, sqrt(2)), and
// writes the matching exponent to *kout, so that log2(x) = *kout + return.
//
// Centering the mantissa on 1 rather than on [1,2) is what removes the
// cancellation: for x just below 1 the naive split gives k = -1 and a mantissa
// near 2, whose log2 is near 1, and k + log2(m) is then a difference of two
// numbers of size 1 producing an answer near 0.

AI double log2_frac_f64(double x, double *kout) {
  // A denormal has no exponent to read — its field is zero and its mantissa is
  // not normalised — so scale it into normal range first and take the scale
  // back out of the exponent afterwards. Without this, log of anything below
  // 2^-1022 is nonsense, and the damage is not confined to log: expm1 divides
  // by log(exp(x)), so a large negative argument, where exp underflows to a
  // denormal, produced -1.0027 for a function whose range stops at -1.
  int sub = x < 0x1p-1022 && x > 0.0;
  double xn = sub ? x * 0x1p54 : x;
  unsigned long long u = f64_to_bits(xn);
  // The exponent is taken one higher when the mantissa has already passed
  // sqrt(2), which is what centres m on 1 rather than leaving it in [1, 2).
  //
  // The constant is what a mantissa must be *added to* in order to carry out
  // of the field at that point, so it is 2^52 minus the threshold and not the
  // threshold itself. Getting it the wrong way round leaves m as large as 2,
  // where s = (m-1)/(m+1) reaches 1/3 against a polynomial fitted to 0.172 —
  // measurably wrong, and invisible to any sweep that does not go near 1,
  // because everywhere else k*ln2 dominates and hides it.
  int k = (int)((u + 0x0095f619980c433ull) >> 52) - 1023;
  unsigned long long em = (unsigned long long)((int)(u >> 52) - k) << 52;
  double m = bits_to_f64((u & 0x000fffffffffffffull) | em);
  double s = (m - 1.0) / (m + 1.0);
  *kout = (double)(k - (sub ? 54 : 0));
  return s * POLY8(s * s, LOG2P);
}

AI float log2_frac_f32(float x, float *kout) {
  int sub = x < 0x1p-126f && x > 0.0f;
  float xn = sub ? x * 0x1p30f : x;
  unsigned int u = f32_to_bits(xn);
  int k = (int)((u + 0x004afb0du) >> 23) - 127;
  unsigned int em = (unsigned int)((int)(u >> 23) - k) << 23;
  float m = bits_to_f32((u & 0x007fffffu) | em);
  float s = (m - 1.0f) / (m + 1.0f);
  *kout = (float)(k - (sub ? 30 : 0));
  return s * POLY5(s * s, LOG2PF);
}

// log_special folds in the values IEEE 754 fixes: log of a negative is NaN,
// log(0) is -Inf, log(+Inf) is +Inf, and log of a NaN is that NaN.
#define LOG_SPECIAL(x, v, INFV, NANV)                                     \
  ((x) < 0.0 ? NANV : (x) == 0.0 ? -INFV : (x) == INFV ? INFV : (x) != (x) ? (x) : (v))

AI double log_f64(double x) {
  double k;
  double f = log2_frac_f64(x, &k);
  // ln2 split again: k*LN2_HI is exact, so the sum keeps its low bits.
  double v = k * LN2_HI + (k * LN2_LO + f * LN2);
  return LOG_SPECIAL(x, v, INF, QNAN);
}

AI float log_f32(float x) {
  float k;
  float f = log2_frac_f32(x, &k);
  float v = k * LN2_HI_F + (k * LN2_LO_F + f * LN2_F);
  return LOG_SPECIAL(x, v, INF_F, QNAN_F);
}

AI double log2_f64(double x) {
  double k;
  double f = log2_frac_f64(x, &k);
  return LOG_SPECIAL(x, k + f, INF, QNAN);
}

AI float log2_f32(float x) {
  float k;
  float f = log2_frac_f32(x, &k);
  return LOG_SPECIAL(x, k + f, INF_F, QNAN_F);
}

AI double log10_f64(double x) {
  double k;
  double f = log2_frac_f64(x, &k);
  return LOG_SPECIAL(x, (k + f) * LOG10_2, INF, QNAN);
}

AI float log10_f32(float x) {
  float k;
  float f = log2_frac_f32(x, &k);
  return LOG_SPECIAL(x, (k + f) * LOG10_2_F, INF_F, QNAN_F);
}

// log1p(x) is log(1+x) computed so that small x keeps its precision. Forming
// 1+x first would round away everything below 2^-53 of it.
//
// The trick is due to Goldberg: u = 1+x rounds, but log(u)*x/(u-1) corrects
// for exactly that rounding, because u-1 is the value that was actually added.
AI double log1p_f64(double x) {
  double u = 1.0 + x;
  double d = u - 1.0;
  double v = d == 0.0 ? x : log_f64(u) * (x / d);
  v = x == -1.0 ? -INF : v;
  v = x < -1.0 ? QNAN : v;
  // At +Inf the correction factor is Inf/Inf, so the general form gives NaN.
  v = x == INF ? INF : v;
  return x != x ? x : v;
}

AI float log1p_f32(float x) {
  float u = 1.0f + x;
  float d = u - 1.0f;
  float v = d == 0.0f ? x : log_f32(u) * (x / d);
  v = x == -1.0f ? -INF_F : v;
  v = x < -1.0f ? QNAN_F : v;
  v = x == INF_F ? INF_F : v;
  return x != x ? x : v;
}

// expm1(x) is exp(x)-1, likewise. For small x, exp(x) rounds to 1 and the
// subtraction returns zero; the same correction recovers it.
AI double expm1_f64(double x) {
  double e = exp_f64(x);
  double d = e - 1.0;
  double l = log_f64(e);
  double v = d * (x / l);
  v = d == 0.0 ? x : v;      // exp rounded to 1; the answer is x to first order
  // Below this the answer is -1 to the last bit, and the general form stops
  // being able to say so: exp(x) is a denormal there, carrying far fewer than
  // 53 bits, so log(exp(x)) is no longer x and the correction factor drifts.
  // exp(-37) is under 2^-53, so 1-exp(x) rounds to 1 for every x past it.
  v = x < -37.0 ? -1.0 : v;
  v = e == 0.0 ? -1.0 : v;   // exp underflowed; log of it is -Inf
  v = e == INF ? INF : v;
  return x != x ? x : v;
}

AI float expm1_f32(float x) {
  float e = exp_f32(x);
  float d = e - 1.0f;
  float l = log_f32(e);
  float v = d * (x / l);
  v = d == 0.0f ? x : v;
  v = x < -17.0f ? -1.0f : v;
  v = e == 0.0f ? -1.0f : v;
  v = e == INF_F ? INF_F : v;
  return x != x ? x : v;
}

// ---------- trigonometric ----------
//
// sin and cos share one reduction: x = q*(pi/2) + r with |r| <= pi/4, after
// which the answer is +-sin(r) or +-cos(r) chosen by q mod 4.
//
// pi/2 is subtracted in three pieces so that q*piece is exact for the first
// two. That holds the reduced argument accurate to roughly |x| < 1e14; past
// that, recovering r needs many more bits of pi than a double holds, and the
// kernel returns NaN so the caller's threshold guard is not what decides
// correctness. The exported API documents the range.

#define TRIG_LIMIT 1e14
#define TRIG_LIMIT_F 1e6f

AI double sin_f64(double x) {
  double q = round_f64(x * TWO_OVER_PI);
  double r = ((x - q * PIO2_HI) - q * PIO2_MID) - q * PIO2_LO;
  double u = r * r;
  double s = r * POLY8(u, SINP);
  double c = POLY9(u, COSP);
  long long qi = (long long)q;
  // Bit 0 of q swaps sine for cosine; bit 1 flips the sign.
  double v = (qi & 1) ? c : s;
  v = (qi & 2) ? -v : v;
  double ax = x < 0.0 ? -x : x;
  return (ax > TRIG_LIMIT || ax != ax) ? (x != x ? x : QNAN) : v;
}

AI float sin_f32(float x) {
  float q = round_f32(x * TWO_OVER_PI_F);
  float r = ((x - q * PIO2_HI_F) - q * PIO2_MID_F) - q * PIO2_LO_F;
  float u = r * r;
  float s = r * POLY4(u, SINPF);
  float c = POLY5(u, COSPF);
  int qi = (int)q;
  float v = (qi & 1) ? c : s;
  v = (qi & 2) ? -v : v;
  float ax = x < 0.0f ? -x : x;
  return (ax > TRIG_LIMIT_F || ax != ax) ? (x != x ? x : QNAN_F) : v;
}

AI double cos_f64(double x) {
  // cos(x) = sin(x + pi/2), done by shifting q rather than x so the reduction
  // stays exact.
  double q = round_f64(x * TWO_OVER_PI);
  double r = ((x - q * PIO2_HI) - q * PIO2_MID) - q * PIO2_LO;
  double u = r * r;
  double s = r * POLY8(u, SINP);
  double c = POLY9(u, COSP);
  // cos(q*pi/2 + r) is cos(r), -sin(r), -cos(r), sin(r) as q runs mod 4, which
  // is the sine table shifted by one quadrant. Selecting on q and negating on
  // q+1 spells exactly that, and it stays right for negative q because two's
  // complement makes (-1)&1 == 1.
  long long qi = (long long)q;
  double v = (qi & 1) ? s : c;
  v = ((qi + 1) & 2) ? -v : v;
  double ax = x < 0.0 ? -x : x;
  return (ax > TRIG_LIMIT || ax != ax) ? (x != x ? x : QNAN) : v;
}

AI float cos_f32(float x) {
  float q = round_f32(x * TWO_OVER_PI_F);
  float r = ((x - q * PIO2_HI_F) - q * PIO2_MID_F) - q * PIO2_LO_F;
  float u = r * r;
  float s = r * POLY4(u, SINPF);
  float c = POLY5(u, COSPF);
  int qi = (int)q;
  float v = (qi & 1) ? s : c;
  v = ((qi + 1) & 2) ? -v : v;
  float ax = x < 0.0f ? -x : x;
  return (ax > TRIG_LIMIT_F || ax != ax) ? (x != x ? x : QNAN_F) : v;
}

AI double tan_f64(double x) {
  double q = round_f64(x * TWO_OVER_PI);
  double r = ((x - q * PIO2_HI) - q * PIO2_MID) - q * PIO2_LO;
  double u = r * r;
  double s = r * POLY8(u, SINP);
  double c = POLY9(u, COSP);
  long long qi = (long long)q;
  // An odd quadrant swaps sine and cosine, which turns tan into -cot.
  double v = (qi & 1) ? -c / s : s / c;
  double ax = x < 0.0 ? -x : x;
  return (ax > TRIG_LIMIT || ax != ax) ? (x != x ? x : QNAN) : v;
}

AI float tan_f32(float x) {
  float q = round_f32(x * TWO_OVER_PI_F);
  float r = ((x - q * PIO2_HI_F) - q * PIO2_MID_F) - q * PIO2_LO_F;
  float u = r * r;
  float s = r * POLY4(u, SINPF);
  float c = POLY5(u, COSPF);
  int qi = (int)q;
  float v = (qi & 1) ? -c / s : s / c;
  float ax = x < 0.0f ? -x : x;
  return (ax > TRIG_LIMIT_F || ax != ax) ? (x != x ? x : QNAN_F) : v;
}

// ---------- inverse trigonometric ----------
//
// atan reduces with two identities. |x| > 1 becomes pi/2 - atan(1/x), and what
// is left is folded once more against tan(pi/6) so the polynomial only has to
// cover |r| <= tan(pi/12), which is a third of the original interval and worth
// several degrees.

#define TAN_PIO6 0x1.279a74590331cp-1
#define TAN_PIO6_F 0x1.279a74p-1f
#define TAN_PIO12 0x1.126145e9ecd56p-2
#define TAN_PIO12_F 0x1.126146p-2f
#define PIO6 0x1.0c152382d7366p-1
#define PIO6_F 0x1.0c1524p-1f

AI double atan_f64(double x) {
  double ax = x < 0.0 ? -x : x;
  int inv = ax > 1.0;
  double t = inv ? 1.0 / ax : ax;
  int fold = t > TAN_PIO12;
  // (t - tan(pi/6)) / (1 + t*tan(pi/6)) is tan of the difference of angles.
  double r = fold ? (t - TAN_PIO6) / (1.0 + TAN_PIO6 * t) : t;
  double v = r * POLY10(r * r, ATANP);
  v = fold ? v + PIO6 : v;
  v = inv ? PIO2 - v : v;
  v = ax == INF ? PIO2 : v;
  v = x < 0.0 ? -v : v;
  return x != x ? x : v;
}

AI float atan_f32(float x) {
  float ax = x < 0.0f ? -x : x;
  int inv = ax > 1.0f;
  float t = inv ? 1.0f / ax : ax;
  int fold = t > TAN_PIO12_F;
  float r = fold ? (t - TAN_PIO6_F) / (1.0f + TAN_PIO6_F * t) : t;
  float v = r * POLY5(r * r, ATANPF);
  v = fold ? v + PIO6_F : v;
  v = inv ? PIO2_F - v : v;
  v = ax == INF_F ? PIO2_F : v;
  v = x < 0.0f ? -v : v;
  return x != x ? x : v;
}

// asin uses the polynomial directly for |x| <= 1/2, and for larger arguments
// the half-angle identity asin(x) = pi/2 - 2*asin(sqrt((1-x)/2)), which brings
// the argument back below 1/2. Both branches evaluate; the select picks.
AI double asin_f64(double x) {
  double ax = x < 0.0 ? -x : x;
  double big = __builtin_elementwise_sqrt((1.0 - ax) * 0.5);
  double s = ax > 0.5 ? big : ax;
  double v = s + s * s * s * POLY15(s * s, ASINP);
  v = ax > 0.5 ? PIO2 - 2.0 * v : v;
  v = x < 0.0 ? -v : v;
  return ax > 1.0 ? QNAN : (x != x ? x : v);
}

AI float asin_f32(float x) {
  float ax = x < 0.0f ? -x : x;
  float big = __builtin_elementwise_sqrt((1.0f - ax) * 0.5f);
  float s = ax > 0.5f ? big : ax;
  float v = s + s * s * s * POLY5(s * s, ASINPF);
  v = ax > 0.5f ? PIO2_F - 2.0f * v : v;
  v = x < 0.0f ? -v : v;
  return ax > 1.0f ? QNAN_F : (x != x ? x : v);
}

AI double acos_f64(double x) {
  // acos(x) = pi/2 - asin(x), except that the subtraction loses precision as
  // x approaches 1, where acos goes to zero. There the half-angle branch of
  // asin is reused directly: acos(x) = 2*asin(sqrt((1-x)/2)).
  double ax = x < 0.0 ? -x : x;
  double big = __builtin_elementwise_sqrt((1.0 - ax) * 0.5);
  double s = ax > 0.5 ? big : ax;
  double a = s + s * s * s * POLY15(s * s, ASINP);
  double near1 = 2.0 * a;               // for x in (1/2, 1]
  double nearm1 = PI - 2.0 * a;         // for x in [-1, -1/2)
  double small = PIO2 - (x < 0.0 ? -a : a);
  double v = ax > 0.5 ? (x > 0.0 ? near1 : nearm1) : small;
  return ax > 1.0 ? QNAN : (x != x ? x : v);
}

AI float acos_f32(float x) {
  float ax = x < 0.0f ? -x : x;
  float big = __builtin_elementwise_sqrt((1.0f - ax) * 0.5f);
  float s = ax > 0.5f ? big : ax;
  float a = s + s * s * s * POLY5(s * s, ASINPF);
  float near1 = 2.0f * a;
  float nearm1 = PI_F - 2.0f * a;
  float small = PIO2_F - (x < 0.0f ? -a : a);
  float v = ax > 0.5f ? (x > 0.0f ? near1 : nearm1) : small;
  return ax > 1.0f ? QNAN_F : (x != x ? x : v);
}

// atan2(y, x) is atan(y/x) placed in the right quadrant. The quotient is
// formed only after both are finite and x is nonzero; the remaining cases are
// the axes and the infinities, which IEEE 754 fixes and which are selected in
// rather than branched to.
AI double atan2_f64(double y, double x) {
  double a = atan_f64(y / x);
  double v = x > 0.0 ? a : (y >= 0.0 ? a + PI : a - PI);
  // x == 0: the answer is +-pi/2, or +-0 when y is also zero.
  v = x == 0.0 ? (y > 0.0 ? PIO2 : y < 0.0 ? -PIO2 : (1.0 / x > 0.0 ? y : (y < 0.0 || 1.0 / y < 0.0 ? -PI : PI))) : v;
  v = (x != x || y != y) ? QNAN : v;
  return v;
}

AI float atan2_f32(float y, float x) {
  float a = atan_f32(y / x);
  float v = x > 0.0f ? a : (y >= 0.0f ? a + PI_F : a - PI_F);
  v = x == 0.0f ? (y > 0.0f ? PIO2_F : y < 0.0f ? -PIO2_F : (1.0f / x > 0.0f ? y : (y < 0.0f || 1.0f / y < 0.0f ? -PI_F : PI_F))) : v;
  v = (x != x || y != y) ? QNAN_F : v;
  return v;
}

// ---------- hyperbolic ----------
//
// Each is built from exp, with the argument clamped where the exponential
// would overflow but the function itself is still representable — or where it
// is not, and the answer is an infinity that the clamp produces anyway.

AI double sinh_f64(double x) {
  double e = exp_f64(x);
  double v = 0.5 * (e - 1.0 / e);
  double ax = x < 0.0 ? -x : x;
  // expm1 keeps the small-argument accuracy that e - 1/e throws away.
  double sm = expm1_f64(x);
  v = ax < 1.0 ? 0.5 * (sm + sm / (1.0 + sm)) : v;
  return x != x ? x : v;
}

AI float sinh_f32(float x) {
  float e = exp_f32(x);
  float v = 0.5f * (e - 1.0f / e);
  float ax = x < 0.0f ? -x : x;
  float sm = expm1_f32(x);
  v = ax < 1.0f ? 0.5f * (sm + sm / (1.0f + sm)) : v;
  return x != x ? x : v;
}

AI double cosh_f64(double x) {
  double e = exp_f64(x < 0.0 ? -x : x);
  double v = 0.5 * e + 0.5 / e;
  return x != x ? x : v;
}

AI float cosh_f32(float x) {
  float e = exp_f32(x < 0.0f ? -x : x);
  float v = 0.5f * e + 0.5f / e;
  return x != x ? x : v;
}

AI double tanh_f64(double x) {
  double ax = x < 0.0 ? -x : x;
  // e^(2|x|) - 1 over e^(2|x|) + 1, using expm1 so that small x keeps its
  // leading bits, and saturating where the exponential would overflow but
  // tanh has long since reached 1.
  double e = expm1_f64(2.0 * ax);
  double v = e / (e + 2.0);
  v = ax > 20.0 ? 1.0 : v;
  v = x < 0.0 ? -v : v;
  return x != x ? x : v;
}

AI float tanh_f32(float x) {
  float ax = x < 0.0f ? -x : x;
  float e = expm1_f32(2.0f * ax);
  float v = e / (e + 2.0f);
  v = ax > 10.0f ? 1.0f : v;
  v = x < 0.0f ? -v : v;
  return x != x ? x : v;
}

AI double sigmoid_f64(double x) {
  // 1/(1+e^-x), computed from the side that does not overflow: for x very
  // negative e^-x is huge, so the algebraically equal e^x/(1+e^x) is used.
  double ep = exp_f64(x);
  double en = exp_f64(-x);
  double v = x >= 0.0 ? 1.0 / (1.0 + en) : ep / (1.0 + ep);
  return x != x ? x : v;
}

AI float sigmoid_f32(float x) {
  float ep = exp_f32(x);
  float en = exp_f32(-x);
  float v = x >= 0.0f ? 1.0f / (1.0f + en) : ep / (1.0f + ep);
  return x != x ? x : v;
}

// ---------- cbrt and pow ----------

AI double cbrt_f64(double x) {
  double ax = x < 0.0 ? -x : x;
  // A denormal has no exponent to read, so scale it into normal range first
  // and take the cube root of the scale back out afterwards. 54 is a multiple
  // of three, so the correction is exact.
  int sub = ax < 0x1p-1022 && ax > 0.0;
  double as = sub ? ax * 0x1p54 : ax;
  unsigned long long u = f64_to_bits(as);
  int e = (int)(u >> 52) - 1023;
  // Split the exponent into a multiple of three and a remainder in {0,1,2},
  // so the seed only ever sees a mantissa scaled into [1, 8).
  //
  // The division has to floor, and C's truncates toward zero: for e = -2 that
  // gives a remainder of -2 and a seed argument of 0.25, which is outside the
  // interval the polynomial was fitted on. Biasing by a multiple of three
  // before dividing makes it floor for every exponent a double can hold.
  int e3 = (e + 1074) / 3 - 358;
  int rem = e - 3 * e3;
  double m = bits_to_f64((u & 0x000fffffffffffffull) | 0x3ff0000000000000ull);
  // Scaling by 2^rem arithmetically rather than selecting between three
  // constants. Written as nested ternaries this is a three-way switch, and
  // LLVM builds a jump table for it — which is a constant pool holding code
  // addresses, the one kind this generator cannot relocate, so the whole
  // kernel was dropped on avx512 and left running the portable path.
  m = m * pow2_f64(rem);
  double y = POLY6(m, CBRTP);
  // Three Newton steps. Each triples the correct digits, and the seed is good
  // to about three, so three steps reach well past what a double holds.
  y = y - (y - m / (y * y)) * (1.0 / 3.0);
  y = y - (y - m / (y * y)) * (1.0 / 3.0);
  y = y - (y - m / (y * y)) * (1.0 / 3.0);
  double v = y * pow2_f64(e3) * (sub ? 0x1p-18 : 1.0);
  v = ax == 0.0 ? 0.0 : v;
  v = ax == INF ? INF : v;
  v = x < 0.0 ? -v : v;
  return x != x ? x : v;
}

AI float cbrt_f32(float x) {
  float ax = x < 0.0f ? -x : x;
  int sub = ax < 0x1p-126f && ax > 0.0f;
  float as = sub ? ax * 0x1p30f : ax;
  unsigned int u = f32_to_bits(as);
  int e = (int)(u >> 23) - 127;
  int e3 = (e + 150) / 3 - 50;
  int rem = e - 3 * e3;
  float m = bits_to_f32((u & 0x007fffffu) | 0x3f800000u);
  m = m * pow2_f32(rem);
  float y = POLY5(m, CBRTPF);
  y = y - (y - m / (y * y)) * (1.0f / 3.0f);
  y = y - (y - m / (y * y)) * (1.0f / 3.0f);
  float v = y * pow2_f32(e3) * (sub ? 0x1p-10f : 1.0f);
  v = ax == 0.0f ? 0.0f : v;
  v = ax == INF_F ? INF_F : v;
  v = x < 0.0f ? -v : v;
  return x != x ? x : v;
}

// pow(x, y) is exp2(y * log2(x)), with the product carried in two pieces so
// that a large exponent does not throw away the low bits of the logarithm —
// the same reason exp reduces against a split ln2.
AI double pow_f64(double x, double y) {
  double ax = x < 0.0 ? -x : x;
  double k;
  double f = log2_frac_f64(ax, &k);
  // (k + f) * y, split so the exact part stays exact.
  double hi = k * y;
  double lo = f * y;
  double t = hi + lo;
  double q = round_f64(t);
  double r = (hi - q) + lo;
  double p = POLY14(r, EXP2P);
  int qi = (int)q;
  double v = p * pow2_f64(qi / 2) * pow2_f64(qi - qi / 2);
  v = t >= 1024.0 ? INF : v;
  v = t < -1075.0 ? 0.0 : v;

  // A negative base is only defined for an integral exponent, and the sign of
  // the result is the parity of that integer.
  double yr = round_f64(y);
  int odd = (yr == y) && (((long long)y) & 1);
  v = x < 0.0 ? (yr == y ? (odd ? -v : v) : QNAN) : v;

  // The IEEE special cases, in the order the standard gives them.
  v = y == 0.0 ? 1.0 : v;
  v = x == 1.0 ? 1.0 : v;
  v = (x == 0.0) ? (y < 0.0 ? INF : 0.0) : v;
  v = (ax == INF) ? ((x < 0.0 && yr == y && odd) ? (y > 0.0 ? -INF : -0.0) : (y > 0.0 ? INF : 0.0)) : v;
  v = (y == INF) ? (ax > 1.0 ? INF : ax < 1.0 ? 0.0 : 1.0) : v;
  v = (y == -INF) ? (ax > 1.0 ? 0.0 : ax < 1.0 ? INF : 1.0) : v;
  v = (x != x || y != y) ? ((y == 0.0 || x == 1.0) ? 1.0 : QNAN) : v;
  return v;
}

AI float pow_f32(float x, float y) {
  float ax = x < 0.0f ? -x : x;
  float k;
  float f = log2_frac_f32(ax, &k);
  float hi = k * y;
  float lo = f * y;
  float t = hi + lo;
  float q = round_f32(t);
  float r = (hi - q) + lo;
  float p = POLY8(r, EXP2PF);
  int qi = (int)q;
  float v = p * pow2_f32(qi / 2) * pow2_f32(qi - qi / 2);
  v = t >= 128.0f ? INF_F : v;
  v = t < -150.0f ? 0.0f : v;

  float yr = round_f32(y);
  int odd = (yr == y) && (((long long)y) & 1);
  v = x < 0.0f ? (yr == y ? (odd ? -v : v) : QNAN_F) : v;

  v = y == 0.0f ? 1.0f : v;
  v = x == 1.0f ? 1.0f : v;
  v = (x == 0.0f) ? (y < 0.0f ? INF_F : 0.0f) : v;
  v = (ax == INF_F) ? ((x < 0.0f && yr == y && odd) ? (y > 0.0f ? -INF_F : -0.0f) : (y > 0.0f ? INF_F : 0.0f)) : v;
  v = (y == INF_F) ? (ax > 1.0f ? INF_F : ax < 1.0f ? 0.0f : 1.0f) : v;
  v = (y == -INF_F) ? (ax > 1.0f ? 0.0f : ax < 1.0f ? INF_F : 1.0f) : v;
  v = (x != x || y != y) ? ((y == 0.0f || x == 1.0f) ? 1.0f : QNAN_F) : v;
  return v;
}

// hypot(x, y) is sqrt(x*x + y*y) with the squares taken after scaling by the
// larger magnitude, so that an intermediate cannot overflow for a result that
// is representable, nor underflow to zero for one that is not.
AI double hypot_f64(double x, double y) {
  double ax = x < 0.0 ? -x : x;
  double ay = y < 0.0 ? -y : y;
  double hi = ax > ay ? ax : ay;
  double lo = ax > ay ? ay : ax;
  double t = lo / hi;
  double v = hi * __builtin_elementwise_sqrt(1.0 + t * t);
  v = hi == 0.0 ? 0.0 : v;
  v = (ax == INF || ay == INF) ? INF : v;
  return (x != x || y != y) ? ((ax == INF || ay == INF) ? INF : QNAN) : v;
}

AI float hypot_f32(float x, float y) {
  float ax = x < 0.0f ? -x : x;
  float ay = y < 0.0f ? -y : y;
  float hi = ax > ay ? ax : ay;
  float lo = ax > ay ? ay : ax;
  float t = lo / hi;
  float v = hi * __builtin_elementwise_sqrt(1.0f + t * t);
  v = hi == 0.0f ? 0.0f : v;
  v = (ax == INF_F || ay == INF_F) ? INF_F : v;
  return (x != x || y != y) ? ((ax == INF_F || ay == INF_F) ? INF_F : QNAN_F) : v;
}

// ---------- the exported kernels ----------

#define UNARY_MATH(NAME, T, SUF)                                          \
  void simd_##NAME##_##SUF(T *__restrict d, const T *__restrict a,        \
                           isize n) {                                     \
    for (isize i = 0; i < n; i++) d[i] = NAME##_##SUF(a[i]);              \
  }

#define BINARY_MATH(NAME, T, SUF)                                         \
  void simd_##NAME##_##SUF(T *__restrict d, const T *__restrict a,        \
                           const T *__restrict b, isize n) {              \
    for (isize i = 0; i < n; i++) d[i] = NAME##_##SUF(a[i], b[i]);        \
  }

#define MATH_SET(SUF, T)                                                  \
  UNARY_MATH(exp, T, SUF)                                                 \
  UNARY_MATH(exp2, T, SUF)                                                \
  UNARY_MATH(expm1, T, SUF)                                               \
  UNARY_MATH(log, T, SUF)                                                 \
  UNARY_MATH(log2, T, SUF)                                                \
  UNARY_MATH(log10, T, SUF)                                               \
  UNARY_MATH(log1p, T, SUF)                                               \
  UNARY_MATH(cbrt, T, SUF)                                                \
  UNARY_MATH(sigmoid, T, SUF)                                             \
  UNARY_MATH(sin, T, SUF)                                                 \
  UNARY_MATH(cos, T, SUF)                                                 \
  UNARY_MATH(tan, T, SUF)                                                 \
  UNARY_MATH(asin, T, SUF)                                                \
  UNARY_MATH(acos, T, SUF)                                                \
  UNARY_MATH(atan, T, SUF)                                                \
  UNARY_MATH(sinh, T, SUF)                                                \
  UNARY_MATH(cosh, T, SUF)                                                \
  UNARY_MATH(tanh, T, SUF)                                                \
  BINARY_MATH(pow, T, SUF)                                                \
  BINARY_MATH(atan2, T, SUF)                                              \
  BINARY_MATH(hypot, T, SUF)

MATH_SET(f64, double)
MATH_SET(f32, float)
