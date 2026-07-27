# Numerical findings

Things measured while building the kernels that were not in the plan, are not obvious, and cost
real time to find. Each is here because it would otherwise be rediscovered.

## 1. The portable reference was not architecture-independent

The library's headline promise is that a number does not change when a program moves to another
machine. That was already broken **below the assembly layer**, in the pure Go reference.

Go's spec permits an implementation to fuse a multiply and a following add into one operation with
a single rounding. `gc` takes that licence on **arm64, ppc64, s390x and riscv64** and declines it on
**amd64**. So `dst[i] = a[i] + b[i]*s`, written the obvious way, returns different bits on a
Graviton than on a laptop.

Measured: 770,224 differing results out of 3,000,000 random `float32` triples for
`a + (b-a)*t` on arm64, and zero on amd64.

The fix is an explicit conversion around the product — `a[i] + T(b[i]*s)` — which the spec says
rounds, and a fusion that would discard that rounding is then forbidden. It works through a type
parameter. Applied at every multiply feeding an add in `internal/ref`, and pinned by
`TestNoFusedMultiplyAdd`.

The kernels never had this problem: clang is invoked with `-ffp-contract=off`.

The same trap catches *tests*. `TestElementwiseMatchesScalarLoop` computed its
expectation as `x + y*2.5` in Go, which fuses on four of the six architectures
and not on the one it was written on, so it passed locally and failed by an ULP
on s390x and riscv64. Any expectation written as a multiply feeding an add
needs the same conversion the reference does.

**This is only visible cross-architecture.** The amd64 test suite passed throughout. It surfaced
the first time the conformance suite ran under `qemu-aarch64`.

## 2. Go's standard library is the less accurate side in four places

Each of these was first seen as a conformance failure and assumed to be a kernel bug. In every
case the kernel matches glibc to the bit and `math` is the one that has drifted.

| Function | Input | This library / glibc | Go `math` | Cause |
|---|---|---|---|---|
| `Log` | `1e-320` | `-736.82724089097394` | `-709.08956571282363` | denormal input |
| `Log2` | `1.0139113857366788` | `0.019931568569323342` | off by 26 ULP | `log(frac)/Ln2 + exp` after `Frexp` adds two numbers near 1 to get an answer near 0 |
| `Acos` | `0.9999` | `0.014142253477512098` | off by 971 ULP | `pi/2 - Asin(x)` subtracts two numbers near `pi/2` |
| `Asin` | `-0.9999` | `-1.5566540733173844` | off by ~7 ULP | `Atan(x/Sqrt(1-x*x))`, and `1-x*x` cancels |

`math.Log(1e-320)` is not a rounding difference — it is wrong by 28 in the result. `ln(1e-320)` is
`-320*ln(10) = -736.827`.

The portable path is a tier: it is what a `-tags purego` build uses everywhere, and what the
baseline x86-64 backend falls back to. So `internal/ref` now computes `Log2`, `Asin` and `Acos`
through identities that do not cancel, using the standard library only where it is accurate.

Consequence for the test harness: a ULP suite is only as good as its reference. Two of these were
hidden by sweeps that never visited the interesting region — a linear sweep of `1e-300 .. 1e300`
spends essentially every point where `k*ln2` dominates and dilutes any mantissa error. Logarithms
need a **geometric** sweep, and anything with a zero in range needs points near it.

## 3. Denormals

`log2_frac` read the exponent field directly, which is zero for a denormal, and the mantissa is not
normalised. So `Log` was nonsense below `2^-1022`.

The damage was not confined to `Log`. `expm1` is computed as `(e-1) * x/log(e)` — Kahan's
correction for the rounding of `e` — so a large negative argument, where `exp(x)` underflows to a
denormal, produced **-1.0027** for a function whose range stops at -1. `cbrt` had the same hole.

Both now scale into normal range first and take the scale back out of the exponent; the scale is a
multiple of three for `cbrt` so the correction stays exact.

Separately, `expm1` now clamps below `x = -37`, where `exp(x) < 2^-53` and the answer is -1 to the
last bit. Past that point `exp(x)` is a denormal carrying far fewer than 53 bits, `log(exp(x))` is
no longer `x`, and the correction factor drifts.

## 4. What "the tiers agree" can and cannot mean

For algebraic kernels it means bit identity, and that is tested.

For transcendentals it cannot, and rule 6 already said so. Two tiers that **both** have the kernel
do agree exactly — the evaluation order is one fixed chain of elementwise operations and a wider
vector only changes how many lanes are in flight. A tier that could not compile one keeps the
portable path, which is a different algorithm. The baseline x86-64 tier is the live example: it
reaches its constant pools with legacy SSE instructions that require 16-byte alignment, so it has
none of these kernels.

`TestTranscendentalTiersAgree` therefore compares only pairs that both generated the kernel, and
compares those exactly. Accuracy of a fallback tier is covered by the ULP bound instead.

## 4a. The register the Go runtime owns

The worst bug in the project, and the one that only emulation could have found.

Go keeps the current goroutine's descriptor in a fixed register on every
architecture here, and reads it without warning — at a preemption check, at a
stack-growth check, and from the signal handler. The C ABI knows nothing about
this and treats the same register as ordinary callee-saved state:

| GOARCH | Go's g register | What clang does about it |
|---|---|---|
| arm64 | x28 | `-ffixed-x28` works |
| riscv64 | x27 | `-ffixed-x27` works |
| loong64 | r22 | no `-ffixed`; a global register variable works |
| ppc64le | r30 (and r2 = TOC) | no `-ffixed`; a global register variable works |
| s390x | r13 | **no mechanism at all** |
| amd64 | R14, X15 | nothing needed — ABI0 leaves them undefined |

Only arm64 was reserved. On s390x, `arith.c` alone used r13 as a scratch
register **86 times**, and nine of its kernels did not even save it — so the
goroutine pointer was simply destroyed by an arithmetic kernel that had
already returned successfully.

The symptom is not a crash at the clobber. It is a panic in unrelated Go code,
later, with a nonsense slice header: `slice bounds out of range [::16] with
capacity 0` inside the portable reference, raised from a function whose input
was a zero-length slice.

clang accepts the global register variable on SystemZ and then allocates the
register anyway. So the last word has to be a check on the generated code, and
package verify now drops any kernel that names a register the runtime owns. It
costs s390x about eighty kernels, which is the right trade: the alternative was
memory corruption that no amount of testing on amd64 would ever have shown.

## 4b. The save area the caller is supposed to provide

The register mismatch above has a twin that is about memory, and it is subtler
because the generated code looks perfectly reasonable.

The s390x ELF ABI puts the register save area **at the caller's stack
pointer**: the caller reserves 160 bytes at the top of its own frame and the
callee stores its registers there. Every non-trivial kernel therefore begins

    stmg %r14, %r15, 112(%r15)

A positive displacement from the stack pointer is memory *above* it, which
belongs to whoever called us. Under C that is fine — the caller promised those
bytes. Under Go nobody promised anything, so the kernel writes sixteen bytes
into the middle of the calling Go function's locals.

arm64, riscv64, loong64 and amd64 do not do this; their callees allocate what
they need below the stack pointer, and the measured count of stores above it is
zero on all four. amd64 additionally has `-mno-red-zone`, which forbids the
one case where it could.

### Why the obvious fix does not work

Declaring a Go frame of 176 bytes puts the save area in the right place — and
breaks the return. The compiled body ends in its own branch to the link
register and never reaches the epilogue that would pop the frame, so control
arrives back in Go with the stack pointer still lowered. The next symptom is
worse than the first: `unexpected return pc`.

Appending an epilogue is not an option either, for the same reason reductions
write through an out-pointer: LLVM lays basic blocks out *after* the return
instruction, so there is no position after the body that always executes.

### What does work

Emit two symbols and call the body rather than falling into it:

    TEXT ·kernel(SB), NOSPLIT, $160-28      // a frame, so the area is ours
        MOVD $ret+24(FP), R2
        MOVD a_base+0(FP), R3
        MOVD a_len+8(FP), R4
        BL   ·kernelBody(SB)
        RET                                  // Go's epilogue restores SP

    TEXT ·kernelBody(SB), NOSPLIT|NOFRAME, $0-0
        WORD $0xebeff070                     // stmg %r14, %r15, 112(%r15)
        ...

The body's save-area writes now land in the trampoline's frame. Its branch to
the link register returns to the trampoline, with the stack pointer exactly as
it found it, and Go's own epilogue then runs. Go stores the link register at
0(SP) and clang's save area starts at offset 48, so the two never overlap.

The cost is one branch-and-link per call, against a kernel that is about to
walk a whole slice. Without it, s390x would have had to drop 196 of its 245
kernels — the alternative that was measured before this was found.

## 4c. The parameter save area is parallel on ppc64le

A third ABI disagreement, and the one that produced the least informative
crash. Under ELFv2 every argument owns a slot in the parameter save area in
declaration order, and a floating-point argument's slot is *skipped* in the
integer sequence rather than reused. So for

    simd_scale_f64(double *d, const double *a, double s, isize n)

ppc64le passes d in r3, a in r4, s in f1 — and n in **r6**, because s owns r5.
The generator assigned integer arguments sequentially and put n in r5, so the
kernel read a garbage length and wrote that many elements. It faulted inside
the runtime's stack allocator, several calls away from anything to do with
this library.

Every other target assigns its integer and floating-point argument registers
independently. s390x reads the same kernel's length from r4, and AAPCS64,
RISC-V and System V all behave the same way. Only ELFv2 does this.

## 4d. The red zone that has no flag, and the parser hole that hid it

ppc64le took four bugs to get right, and the last one had been quietly
weakening every check on that target.

**The protected zone.** ELFv2 gives a leaf function 288 bytes *below* the
stack pointer to use without adjusting it — x86-64's red zone by another name.
Go writes below the stack pointer during signal delivery and stack growth,
which is why amd64 is compiled with `-mno-red-zone`. There is no equivalent
for ppc64le: `-mno-red-zone` is accepted there and merely reduces the count,
from 74 stores to 52. Forty-six percent of the kernels use it, and they must
be rejected.

**Why that took so long to see.** The check that should have caught it found
nothing, because llvm-objdump separates a mnemonic from its operands with a
tab only when the mnemonic is long enough to need one, and with a space
otherwise:

    cmpdi\t7, 0
    std 24, -64(1)

The parser assumed a tab. So `std 24, -64(1)` was read as a *mnemonic* with no
operands at all, and every check that looks at operands — the stack-frame
check, the Go-owned-register check, the register half of the feature gate —
silently matched nothing on ppc64le. That is the third parser hole in this
package, after the s390x `<unknown>` instructions and the internal-label
truncation, and all three had the same shape: a parse that failed quietly and
left a check passing on an empty list.

The lesson is in the code now: every one of those checks fails loudly on an
unparseable line rather than skipping it.



Four bugs, all fixed: the parameter save area slot, a framed assembly function
needing NO_LOCAL_POINTERS, the protected zone above, and the parser hole that
hid the last one. ppc64le ships 87 kernels and passes the full suite under
emulation. The rest of its kernels keep the portable path because they use the
protected zone, which no flag can turn off.

Not attempted: a trampoline like the s390x one. ELFv2 convention is that a
`bl` to a global symbol is followed by a slot the linker may rewrite into a
TOC reload, and in a trampoline the instruction after the `bl` is the `RET`.
Losing that is precisely the "returns to address 1" this target was failing
with, so the trampoline is not safe here even though it is on s390x.

## 5. Constant pools on AArch64

clang reaches a pool with `adrp x11, sym` + `ldr d1, [x11, :lo12:sym]`. `adrp` computes
`page(target) - page(pc)`, which is not known until link time, and re-spelling the pair as Plan 9
instructions needs three instructions where the original is two — and a body that changes length
invalidates every PC-relative branch across it.

`ADR` is the same four bytes as `ADRP` and differs in one bit, but counts **bytes from the
instruction** rather than pages. With the pool appended to the function's own body the distance is
known at generation time. So `adrp` becomes `adr` pointing at the base of the copied section, and
each load's 12-bit offset is filled in with its own position in it. Both edits are in place and
length-preserving.

Pointing the `ADR` at the section base rather than at the individual constant matters: LLVM shares
one `adrp` between several loads at different offsets, and pointing at one of them would silently
give the others the wrong number.

This took arm64 from 22 of 42 transcendental kernels to 40, and is worth roughly 130 more across
s390x (`larl`, byte-granular and the easiest of the three), loong64 (`pcalau12i` → `pcaddu12i`) and
riscv64 (`auipc` hi20/lo12). ppc64le is different — it reaches pools through the TOC pointer in
`r2`, which Go does not set up.

## 6. Do not rely on text alignment

There is a tempting fix for the baseline x86-64 tier: pad the appended pool to a 16-byte offset
within the function and rely on the linker aligning every text symbol — 32 bytes on amd64, per
`cmd/link/internal/amd64.funcAlign`.

Do not. `cmd/link` takes a `-funcalign` flag, so a consumer building with `-ldflags=-funcalign=8`
would get a SIGSEGV out of this library with no plausible way to trace it back. Dropping the kernel
and keeping the portable path costs a few percent on a tier a machine only reaches if it predates
AVX2.

## 6a. The disassembly parser stopped at the first internal label

llvm-objdump prints a local label inside a function in the same shape as a
function header — `0000000000007460 <.L0 >:` — and the parser treated it as
the start of a new function. Everything after the first internal label went
into a phantom entry and was never checked.

That is not cosmetic. The instruction-vs-tier check is the one thing standing
between a mis-gated kernel and a SIGILL on a user's machine, and it was seeing
truncated bodies. It also made 23 loong64 kernels look like tail calls to libm,
because their `ret` happened to sit past a label, and it hid the `leaq` in
cbrt that made the kernel unliftable.

Fixing it moved kernel counts in both directions, which is what a fixed checker
should do.

## 7. Shapes that decide whether LLVM vectorizes

Four cases where the obvious C is the unvectorizable one, all found by reading generated assembly.

| Written as | What LLVM does | Written instead as |
|---|---|---|
| `d[i] = m[i] ? yes[i] : no[i]` | selects between the two **base pointers** and loads once (`csel x11, x2, x3`) — a good scalar strength reduction, and unvectorizable | load both into locals first, leaving a lane-wise blend |
| an accumulator array with a runtime-indexed tail | gives the array an address, spilling all 16 lanes | `ext_vector_type` accumulator with a blended tail |
| a byte reduction ending in a horizontal fold | a `tbl` whose index vector is loaded from `.rodata` | fold by hand through 64-bit lane extracts |
| a three-way `rem` select in `cbrt` | a jump table — a constant pool of code addresses, the one kind that cannot be relocated | `m * pow2(rem)` |

One more that is a diagnosis rather than a shape: `leaq` was being rejected as
"a legacy SSE instruction that requires 16-byte alignment". It is not a load at
all — it computes an address and never dereferences it — and neither are the
scalar `ss`/`sd` forms, which touch four or eight bytes. Only the packed
128-bit legacy forms genuinely need the pool aligned. Correcting that
recovered cbrt on amd64 and four more kernels on the baseline tier.

And two on the numerical side: `__builtin_sqrtf` emits a libm call because C requires it to set
errno, while `__builtin_elementwise_sqrt` does not; and a search written to accumulate over the
whole input vectorizes but was measured **1700x slower** than portable Go on a 256 KiB slice whose
first byte settled the answer. Neither extreme is right — fold whole blocks with vector code and
test the accumulator once per block, which bounds the wasted work at one block.

## The RISC-V constant pool read from four bytes before the function

`auipc`/`ld` addresses a constant in two halves, and the low half cannot be
resolved on its own: `R_RISCV_PCREL_LO12`'s symbol is a label placed on the
`auipc`, because the value it needs depends on where the *high* half was.
`constpool_rv.go` knew that and paired the two by destination register, which
is correct. What it then did was look the pool's base address up with the low
half's own `TargetSection` — and that section is `.text`, where the label is.
The map has no entry for it, so the lookup returned zero and the address came
out as `hi.at` bytes before the function rather than the pool.

For `simd_is_ascii` the load was `ld a4, -4(a4)` after `auipc a4, 0` at offset
4: it read the eight bytes ending at the function's own first instruction. Any
eight bytes decode as a double, which is why nothing crashed. `Exp` on float64
returned `+Inf` at −700, `Sigmoid` was out by 9.0e15 ULP, `Acos` by 7.07e15 at
−1, and their float32 twins were exact — the float32 polynomials fit in
immediates and need no pool at all.

Every riscv64 kernel with a constant pool was affected, and none of it was
visible: the `riscv64` lane runs under an emulator whose CPU has no vector
extension, so the whole backend was skipped as unexecutable and the lane was
green. See D11 in `05-decisions.md` for the general form of that, and
`simdinfo -require-accelerated`, which is now the first thing every emulated
lane runs.
