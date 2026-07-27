// The Fast tier: the same transcendentals, cheaper and less exact.
//
// Every kernel here is csrc/math.c compiled a second time with two changes and
// no others. That is deliberate — the argument reduction, the reconstruction
// and every special case are *shared source*, so the two tiers cannot drift
// apart in anything except the polynomial, and a fix to one is a fix to both.
//
// What differs:
//
//   1. Shorter fits. SIMD_FAST_POLY points every evaluator at a lower-degree
//      minimax polynomial for the same function on the same interval. The
//      degrees are searched against an error budget rather than chosen, and
//      the budget is a little under one ULP so the rest of the 3.5 is left for
//      the reduction and the reconstruction; see tools/polygen/polygen.py.
//      Most fits drop one or two terms and several drop none, because they
//      cannot inside that budget.
//
//   2. Fused multiply-add. This file is compiled with -ffp-contract=fast where
//      the rest of the library is compiled with it off. Fusing halves the
//      instruction count of a Horner chain and is *more* accurate, not less —
//      but it gives a different answer on a machine that has an FMA than on
//      one that does not, and every other kernel here promises the same bits
//      everywhere. A function named Fast has already given that promise up.
//      That is what it buys, and it is why the flag lives with this file
//      alone; see kernels.Source.ExtraFlags.
//
// What does *not* differ, and this is the line that matters: NaN in gives NaN
// out, the infinities go where IEEE 754 says, and the signed zeros are
// preserved. -ffast-math and -fno-signed-zeros would both buy more and are
// both refused, because they change what the function *means* on those inputs
// rather than how accurately it computes the ordinary ones. viterin/vek
// compiles with -Ofast and its NaN behaviour is undefined as a result; the
// whole point of naming a bound is that it is a bound, not a shrug.
//
// The measured margin is 8% to 18% depending on the architecture, which is
// smaller than it looks like it should be: the polynomial is only about a
// tenth of exp_f64's 166 instructions, and the rest is reduction and
// special-case handling that neither change touches. Where a target does not
// come out ahead the kernel is simply not generated for it, and the accurate
// kernel stands in — a more accurate answer satisfies an upper bound on error.

#define SIMD_FAST_POLY
#define SIMD_TIER fast_

#include "poly.h"
#include "math.c"
