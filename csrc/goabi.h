// Registers the Go runtime owns, taken out of the compiler's hands.
//
// Go keeps the current goroutine's descriptor in a fixed register on every
// architecture here, and reads it without warning: at a preemption check, at
// a stack-growth check, and from the signal handler for asynchronous
// preemption. A kernel that uses that register for a loop counter does not
// crash where it is written — it corrupts whatever the runtime does next.
// The symptom is a panic in unrelated Go code with a nonsense slice header,
// which is exactly how this was found: `slice bounds out of range [::16] with
// capacity 0` inside the portable reference, on s390x, from an arithmetic
// kernel that had already returned.
//
// The C ABI has no idea about any of this. On s390x, r13 is an ordinary
// callee-saved register, so clang uses it freely for long-lived values —
// eighty-eight times in arith.c alone, and not merely to save and restore it.
//
// arm64 and riscv64 are handled with -ffixed on the command line, which is
// cleaner. clang has no -ffixed at all for s390x, loongarch or ppc64, so
// those use a global register variable, which reserves the register for the
// whole translation unit. It is a GCC extension that clang implements for
// exactly this purpose.
//
// The variables are never read or written. Declaring them is the entire point.

#ifndef SIMD_GOABI_H
#define SIMD_GOABI_H

#if defined(__s390x__)
// REGG = R13, per cmd/internal/obj/s390x/a.out.go.
//
// clang accepts this declaration on SystemZ and then allocates r13 anyway —
// measured, 86 uses survived it. It is kept because it states the intent and
// costs nothing, but what actually protects r13 here is the check in package
// verify, which drops any kernel that touches it. That is why the s390x tier
// has visibly fewer kernels than the others.
register void *simd_reserved_g __asm__("r13");

#elif defined(__loongarch64) || defined(__loongarch__)
// REGG = R22, per cmd/internal/obj/loong64/a.out.go.
register void *simd_reserved_g __asm__("$r22");

#elif defined(__PPC64__) || defined(__powerpc64__)
// REGG = R30, and R2 is the TOC pointer, which Go does not set up the way the
// ELFv2 ABI expects.
register void *simd_reserved_g __asm__("r30");
register void *simd_reserved_toc __asm__("r2");
#endif

// SPLAT broadcasts a scalar across every lane of a vector type.
//
// It lives here rather than beside its callers because the obvious spelling is
// wrong and every file needs the right one. A C cast to a vector type does NOT
// broadcast — it puts the scalar in lane 0 and zeroes the rest:
//
//	f32xS id = (f32xS)(1.0f);   // {1, 0, 0, 0, ...}, not {1, 1, 1, 1, ...}
//
// Measured on ppc64le, fifteen of sixteen lanes zero, at every optimisation
// level from -O0 to -O3. It is target-dependent — the same cast broadcasts on
// amd64 and arm64 — which is what makes it dangerous: it passes every test on
// the development machine and is wrong where nobody looks.
//
// A scalar operand in vector *arithmetic* is broadcast, so subtracting a zero
// vector does it. Subtraction and not addition: `(VT){0} + x` gives +0 for
// x = -0, because IEEE says (+0) + (-0) is +0, so an addition-based splat
// silently loses the sign of a negative zero. `x - (VT){0}` is exact for every
// input — -0, +0, NaN, both infinities.
//
// This cost one shipped bug and had three more latent uses waiting. See
// entry 52 of docs/wrong.md.
#define SPLAT(VT, X) ((X) - ((VT){0}))

#endif // SIMD_GOABI_H
