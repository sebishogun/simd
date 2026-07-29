# Where the obvious answer was wrong

Every entry here is something that a competent person would have assumed, that
was false, and that cost real time to find. They are written as *what was
believed / what was true / how it surfaced*, because that is the only form in
which this kind of knowledge is useful to anyone else.

If you are trying to get compiler output into Plan 9 assembly, or to keep a
vectorized result identical across instruction sets, you will hit most of
these. The background reasoning lives in [`research/`](research/); this file is
the part worth reading.

---

## 1. A register can be reserved by *value*, not by name

**Assumed.** Telling the compiler not to allocate a register is enough to keep
it out of the way. `-ffixed-rN` for the ones that have it, a global register
variable for the ones that don't.

**Actually.** Go's ppc64 ABI requires `r0` to *contain zero* — not merely to be
left alone. `PPC64Ops.go` lowers a struct zeroing to a run of `MOVD R0,(R3)`,
storing the register on the understanding that its contents are the zero it is
writing. Under ELFv2 `r0` is ordinary volatile scratch and LLVM uses it 212
times across these kernels. There is no flag: clang's PowerPC target has no
`-ffixed-rN` at all, and a global register variable for it is accepted and then
silently ignored, exactly as on SystemZ.

**How it surfaced.** Not as a wrong answer. The kernel computes correctly and
returns; the *next* allocation in Go comes back non-zero and the process dies
inside the runtime — an index out of range in the allocator, a nil dereference
in the collector — several calls from anything to do with this library. One
`li r0, -1` in an otherwise empty assembly function reproduces it.

**Fix.** Every return is rewritten into a branch to the end of the body so that
a two-instruction epilogue always runs and restores the register. See
`emit/returns_power.go`.

---

## 2. A green test lane can mean nothing at all

**Assumed.** The riscv64 and loong64 suites passing means those backends work.

**Actually.** The emulator in the image predated the vector extension. Every
accelerated tier was detected as unavailable, skipped, and the suite passed
having executed none of it. It had been in that state for months.

**How it surfaced.** Only when the emulator was upgraded deliberately. The
first run that actually executed those kernels found a segfault in one backend
and wrong answers from every constant-reading kernel in the other.

**Fix.** `simdinfo -require-accelerated` runs before the tests in every
emulated lane and fails if the selected tier is `scalar`. A suite that skipped
everything is indistinguishable from one that tested everything, so the
assertion has to be explicit.

---

## 3. `--mattr` replaces the feature set, it does not add to it

**Assumed.** `llvm-objdump --mattr=+sve2` means "SVE2 as well as the baseline".

**Actually.** It means "SVE2 instead of the default features". Base NEON stops
disassembling.

**How it surfaced.** The first kernel to emit `sdot` disassembled as
`<unknown>`, and the gate-versus-emission check refused to pass an instruction
it could not read — correctly, since an unreadable instruction cannot be proven
to be within the tier. It had been invisible until then because nothing
happened to use a NEON instruction outside the SVE subset.

**Fix.** `--mattr=+sve2,+neon,+dotprod`. Allowing DotProd is not a widening of
the tier: SVE2 implies ARMv9-A, ARMv9-A implies ARMv8.5-A, and DotProd has been
mandatory since ARMv8.4-A, so any CPU reporting SVE2 has `sdot`.

---

## 4. A compiler builtin can compile to nothing

**Assumed.** `__builtin_masked_compress_store(mask, v, dst)` stores something.

**Actually.** Building the mask by assigning into a `_Bool` vector element by
element — `m[j] = keep[j]` — folds the whole mask to `zeroinitializer` at `-O1`
and above. The IR then contains a compress store with an all-zero mask, which
is dead, and the function returns having written nothing. At `-O0` it is
correct, which is the worst possible combination.

**How it surfaced.** The generated function was one instruction: `retq`.

**Fix.** Build the mask from a comparison and convert it:
`__builtin_convertvector(kb != 0, maskType)`. Then it lowers to `vpcompressd`
on AVX-512 and `compact` on SVE2, as intended.

---

## 5. EVEX is not VEX with a longer prefix

**Assumed.** The constant-pool rewriter's VEX handling covers `vmovdqa64`,
since the `pp` field is in the last prefix byte for both.

**Actually.** VEX puts `pp` in the byte three before the displacement; EVEX has
a four-byte prefix whose *third* payload byte holds the mask register and the
vector length, and `pp` lives one byte earlier. Reading the VEX position on an
EVEX instruction lands on `aaa`, which is zero for an unmasked move.

**How it surfaced.** As `pp=0, which is neither 66 nor F3` — and the kernel was
dropped rather than mis-patched, so the symptom was a missing kernel rather
than a wrong answer. `b64Encode` had silently never existed on AVX-512; base64
encoding ran the portable loop on the fastest tier in the library.

**Fix.** Detect the form by its leading byte (`62`/`C4`/`C5`) and confirm EVEX
with the two bits the encoding fixes, rather than assuming.

---

## 6. An accumulator written as an array is an accumulator in memory

**Assumed.** `T acc[16]` with a loop over it is sixteen accumulators in
registers.

**Actually.** A runtime-indexed remainder loop forces the array to have an
address, so all sixteen live on the stack. Measured stack references in the
float64 sum: 16 on NEON, 16 on SVE2, 21 on AVX-512, 33 on SSE2. Every length
that was a multiple of 16 ran fast and every other length was four to five
times slower — for *less* work.

**Actually, part two.** Guarding the remainder without changing the type fixes
amd64 and makes LLVM abandon vectorization entirely on arm64: 220 scalar
instructions and not one vector instruction.

**Fix.** Declare the accumulator as an explicit vector type and blend the
remainder into a second vector. Lanes with no element contribute `+0`, and
`x + 0` is `x` for every finite value, infinity and NaN, so it stays exact.

---

## 7. `-fwrapv` is the cheap way to avoid signed-overflow UB

**Assumed.** Defining signed overflow costs nothing.

**Actually.** It costs 9–11% of vectorization, because the optimizer uses the
undefinedness to prove that induction variables do not wrap.

**Fix.** Compute in the unsigned domain instead, where wrapping is defined by
the standard. Free, and the generated code is identical to the version that
relied on UB.

---

## 8. The reference is architecture-independent because it is written in Go

**Assumed.** A portable Go implementation is the fixed point that the
accelerated tiers are compared against.

**Actually.** Go fuses a multiply into an add on some architectures and not
others, so the *reference itself* produced different bits on different
machines. The thing being used as ground truth was not ground.

**Fix.** The reference's arithmetic is written so that no contraction is
possible, and reductions use a fixed sixteen-lane accumulator with an explicit
combine tree, so a 128-bit and a 512-bit machine reproduce it exactly.

---

## 9. A test written in Go is a neutral judge of a kernel

**Assumed.** A textbook triple loop written in Go is a safe reference to check
a matrix multiply against, because it is obviously correct.

**Actually.** `acc += a[i]*b[j]` in Go may be fused into a single
multiply-add with one rounding, and on arm64 it is. So the reference computed
different bits from the kernel it was checking — and the kernel was the correct
one, since it deliberately does not fuse.

**How it surfaced.** `TestMatMulExactBits` passed on amd64 and failed on
arm64 under emulation. Entry 8 is the same trap one level down, and this test
walked into it anyway, which is the point: knowing about it is not protection.

**Fix.** An explicit conversion rounds to the target type and forbids the
fusion: `acc += T(a[i] * b[j])`. `internal/ref` had always done this; the new
test had not.

---

## 10. Compiler intrinsics are equivalent to hand-written assembly

**Assumed.** Go 1.26's `simd/archsimd` intrinsics compile to the same
instructions a generated kernel does, so the only difference is that intrinsics
inline and assembly does not — which would make intrinsics strictly better below
the call-boundary crossover and equal above it.

**Actually.** Measured on float32 `AddInto`, the assembly is **4.4× faster at
n = 256** and three times faster at n = 128. Intrinsics win only at n ≤ 16, by
about two nanoseconds.

**Why.** Intrinsics are equivalent to *what you write with them*. The idiomatic
Go version is an 8-lane load-add-store loop; the generated kernel is
clang-optimized, uses 512-bit vectors and unrolls. Closing that gap means
hand-writing the unrolled 512-bit loop in Go, once per element type and once
per vector width — which is precisely the work a generator exists to avoid.

**Consequence.** The intrinsics tier was built, measured, and deliberately not
shipped: it would ask consumers to set a `GOEXPERIMENT` a library cannot
require, on amd64 only, for two nanoseconds. See ROADMAP.md.

---

## 11. A comparator is a detail, not a cost

**Assumed.** Expressing "NaN sorts last" with `slices.SortFunc` and a small
comparison function is equivalent to `slices.Sort`, give or take.

**Actually.** It made sorting 1024 float64 take **17.5µs against pdqsort's
6.9µs**. A closure comparator is called once per comparison and cannot be
inlined; `slices.Sort` inlines a bare `<`.

**How it surfaced, and the part worth keeping.** The 2.5x was attributed to the
vectorized partition the sort was built around — the new, complicated,
suspicious thing — and "fixed" by raising the cutoff so the partition ran less
often. That made every size slower, some by 24%, because it removed the levels
that were doing useful work. The wrong cause produced a fix that made things
worse, which is the only reason the real cause got looked for.

The tell was available from the start and was ignored: 17.5µs to sort 1024
float64 is too slow for *any* correct algorithm. That number indicts the
constant factor, not the strategy.

**Fix.** `slices.Sort`, then rotate the leading NaN block to the end — one pass,
and only when a NaN is present at all. With the comparator gone the same sort
beats pdqsort by 19-27% at every large size.

---

## 12. A scripted edit that does not match is an edit that did not happen

**Assumed.** A string replace over a source file either applies or is obviously
wrong.

**Actually.** It fails silently. Registering `Partition` in the portable
reference was done with a scripted replace whose pattern no longer matched after
gofmt reflowed the surrounding lines. Nothing errored. The slot stayed nil.

**Why nothing caught it.** amd64 and arm64 generate a real partition kernel, so
the backend filled the slot and every local test passed — `go test`, `make
verify`, all green. The nil only exists where the kernel does not, which is
every other architecture. `PartitionInto` would have panicked with a nil
dereference for every user on s390x, ppc64le, riscv64, loong64, and on amd64
below AVX-512 or arm64 without SVE2. Not a wrong answer — a crash on the first
call, on the majority of machines.

The emulated s390x lane found it, running the portable path that the two
development architectures both mask.

**Fix, in two places.** The registration, and a nil guard in `PartitionInto` so
a missing kernel can never panic again — which `CompressInto` already had, for
this exact reason, written by the same hand a day earlier and then not repeated
for the slot with identical semantics.

**And the process fix:** a scripted edit must assert that it changed something.
Two other registrations made the same way happened to land, which is luck.

---

## 13. The kernel that reports the wrong answer is the kernel with the bug

**Assumed.** When a differential test says `I8.Mul` returned 9 where it should
have returned 0, the fault is in `I8.Mul`.

**Actually.** Under memory corruption the failure surfaces wherever the
scribbled-on memory is next read, which has nothing to do with where the
scribbling happened. Bisecting the ppc64le TOC rewrite showed `simd_mul_` in
isolation — which contains `simd_mul_i8` — running 1.74 million fuzz executions
clean. The kernel named in the failure was innocent.

**What it cost.** A whole hypothesis was built on that name. The DS-form
displacement patch was the obvious suspect for an `int8` multiply going wrong,
it was investigated first, and excluding all 360 DS relocations changed nothing.

**The general form.** A wrong *value* points at the code that computed it. A
wrong value caused by corruption points nowhere at all, and the two are
indistinguishable from a single test failure. Reproducibility separates them: a
fault that names a different input every run is corruption, and the name in the
report is noise.

**What worked instead.** Bisecting by *how much* was enabled rather than by
which kernel failed. Every subset up to 367 kernels was clean and every set from
419 up crashed, which said scale rather than identity, and pointed at a check
that admits a rare instruction form rather than at any one family.

---

## 14. Associativity is what makes a scan vectorizable

**Assumed.** A prefix scan can be vectorized exactly when its operation is
associative, because then the log-shift regrouping is not observable. Minimum
and maximum are associative, so CumMin and CumMax get the fast path and CumSum
and CumProd cannot.

**Actually.** Associativity makes it *valid*, not *profitable*, and the second
question has a different answer per element type. A scan costs log2(lanes)
combines per block where the serial loop costs one combine per element, so the
combine has to be cheap for the trade to pay:

    CumMin, 1M elements     vectorized   scalar
    int32                       266µs      626µs    2.4x faster
    float64                     877µs      601µs    46% SLOWER

**Why.** Integer minimum is one instruction. IEEE-754-2019 minimum is a
five-operation select chain — NaN propagation and -0 ordering are exactly what
the hardware `minps` does *not* do — so the float scan pays five times as much
at every one of the four steps.

**Fix.** The integer scans ship and the float ones do not. The kernels are left
in the source, compiled and unreferenced, because the measurement is worth being
able to repeat.

**And the near miss.** The first version was slower for *both* types, and the
instruction count said why: sixty vector instructions for a sixteen-element
block. The shift had been written as an elementwise select — `sh[j] = (j >= s)
? v[j-s] : v[j]`, fully unrolled — which clang turns into sixteen inserts per
step rather than one permute. Replacing it with `__builtin_shufflevector`, whose
indices must be source literals rather than merely constant after unrolling,
is what let the integer case win. Had that not been fixed first, the honest
conclusion would have been "scans do not vectorize", and it would have been
wrong.

---

## 15. The standard library is the more accurate side

**Assumed.** Where a kernel and Go's `math` disagree, the kernel is wrong.

**Actually.** In four places measured here, Go's `math` is the less accurate
one. A ULP bound quoted against it is a bound against a moving target, which is
why the bound is *measured and reported every run* rather than asserted from
theory.

---

## 16. `-0.0 < 0.0`

**Assumed.** A sign test can be written as a comparison against zero.

**Actually.** `-0.0 < 0.0` is false. `cbrt(-0)` therefore returned `+0`.

**Fix.** Restore the sign bit explicitly rather than testing the value:
`bits_to_f64(f64_to_bits(v) | (f64_to_bits(x) & signMask))`.

---

## 17. Skipping a zero multiply is a free optimization

**Assumed.** `if (s == 0) continue` in a matrix multiply changes nothing but
speed.

**Actually.** Under IEEE 754, zero times infinity is NaN. Skipping suppresses
it, which disagrees with BLAS, with numpy, and with the standard. It also
cannot survive register blocking: in a tile that test guards a single fused
multiply-add rather than a whole row, so it costs more than the blocking gains
on any matrix that is not mostly zeros.

**Fix.** Removed, from the kernel and the reference together, and pinned by a
test in both the tiled path and the edges.

---

## 18. A rewrite that produces correct instructions is a correct rewrite

**Assumed.** A constant-pool reference can be re-spelled as a Plan 9 mnemonic
naming a `DATA` symbol, letting Go's linker compute the displacement.

**Actually.** The replacement is not the same length as the instruction it
replaces, so every PC-relative branch spanning it now jumps to the wrong place.

**How it surfaced.** A reciprocal kernel whose remainder loop was simply never
entered. No crash, no diagnostic — just untouched output at the tail.

**Fix.** Patch displacements in place and append the pool to the body. Nothing
changes length, so every branch that was correct in the object file still is.

---

## 19. Vectorizing a loop makes it faster

**Assumed.** Any loop turned into vector instructions beats the scalar version.

**Actually.** Four separate counter-examples in this library alone:

- `Count` was 1.9× *slower* than the standard library, because the filter
  rescanned quadratically.
- `CountByte` was 3.5× slower, because it accumulated in 32-bit lanes when the
  data is bytes.
- `IndexNotAny` never vectorized at all, because the loop nest was inside-out.
- The first rewrite of one of these was **6× slower**, because the lane-indexed
  accumulator spilled — see entry 6.

And the structural case: "is any byte set" written as an unconditional
accumulate vectorizes and reads the whole input; written as a search it does
not vectorize and exits early. Measured 3166 ns against 1.8 ns on a 256 KiB
slice whose first byte already settled the answer — a 1700× loss for the
"faster" version.

**Fix.** Measure every one. The thresholds in this library are measured
per-operation and per-type, and several operations defer to the standard
library because it is genuinely better.

---

## 20. A disassembler prints register names

**Assumed.** Checking generated code for a forbidden register is a text search
for its name.

**Actually.** llvm-objdump prints PowerPC registers as bare numbers by default,
so `addi 10, 9, 2` names two registers and a constant and is indistinguishable
from immediates. The check matched nothing at all and the one thing it exists
to catch went through it unseen. On LoongArch the same check needs `$fp`, not
`r22`, because that is the spelling the disassembler chooses.

**Fix.** `-mllvm=-ppc-asm-full-reg-names`, and register names per target
written in the spelling that target's disassembler actually emits.

---

## 21. The benchmark said it got slower

**Assumed.** A regression harness reporting sixteen benchmarks over the
threshold means sixteen regressions.

**Actually.** They were all measurement contention — a `go vet` and a `git
commit` issued in another shell while the run was in flight. `git show --stat`
proved the relevant `.s` files were byte-identical to the baseline, so the
kernels could not have changed. An idle re-run matched the baseline to within
1%: one benchmark that had reported +129% measured 6.83–7.02 ns against a
6.99 ns baseline.

**Fix.** Run nothing at all during a measurement, and when something is
flagged, check whether the generated code changed before believing it.

---

## 22. A test lane that produces no output is running slowly

**Assumed.** The emulated arm64 lane sitting silent for half an hour is qemu
being qemu. The Makefile even says to allow that long.

**Actually.** It was hung, and had been from the start. `go test` runs `go vet`
automatically; under qemu-user the vet subprocess exits without being reaped,
so it becomes a zombie and the parent blocks in `do_wait` forever.

**How it surfaced.** By checking CPU rather than elapsed time. Thirty-two
minutes of wall clock at **0.1% CPU across every qemu process**, with no
compiler children — and `ps` inside the container showing `[vet] <defunct>`.
Slow and stopped look identical from the outside; they do not look at all alike
in a process table.

**Fix.** `-vet=off` in the emulated lanes. Nothing is lost: vet runs natively
in `make verify` and its findings do not depend on GOARCH.

---

## 23. "No space left on device" means the disk is full

**Assumed.** `ENOSPC` means bytes.

**Actually.** `/tmp` had 40 GB free and *zero* free inodes — a tmpfs is created
with a fixed inode count, and an unrelated test suite had consumed all 1,048,576
of them. Nothing could create a file of any size.

**Fix.** `df -i`, not `df -h`.

## 24. Two correct sorts of the same slice hold the same bits

**Wrong.** They hold the same *values*. Bits are a stronger claim, and this
package promises the stronger one — accelerated and portable paths must agree
bit for bit — so the gap is a contract violation and not a curiosity.

A differential test written for the new accelerated `Median` failed on its
first run: at 2047 elements the accelerated path returned `+0` and the
reference returned `-0`. The obvious suspicion was a bug in the new
quickselect. It was not. `-0 < +0` is false and so is `+0 < -0`, so under the
`<` that `slices.Sort` and `cmp.Less` are built on, the two zeros are *equal*.
Which of two equal-comparing values lands in a given position is decided by the
algorithm, and the two paths are different algorithms.

Checking `Sort` directly, on 4096 float64 containing both zeros, the
accelerated and portable results differed in **848 of 4096 positions**. The
property was already there, undocumented, and no test had looked — every
existing `Median` and `Quantile` test ran at `maxLen = 70`, far below the
threshold where the accelerated path even engages, so none of them executed it
at all.

**Fix.** Not a fix — a decision, and a documented one. Forcing agreement means
giving the zeros a total order, which means `slices.SortFunc` with a comparator
closure. That was already measured at 2.5x slower (entry 12), for a
distinction only `math.Signbit` can observe. So the exception is stated at
`Sort`, inherited by `Median` and `Quantile`, and the differential test carves
out exactly `x == 0 && y == 0` and nothing else — NaN, the infinities and every
ordinary value still have to match bit for bit.

The transferable part is the second sentence of the first paragraph. A test
that never crosses a dispatch threshold is testing the fallback and reporting
success.

## 25. "LLVM did not vectorize it" means the loop cannot be vectorized

**Wrong.** It can mean three quite different things, and only one of them is a
property of the loop. Eighteen kernels on riscv64 — log, log2, log10, log1p,
expm1, cbrt, pow, sinh and tanh, in both precisions — sat behind that message
for months looking like a limitation of RVV. They are straight-line polynomial
evaluation with no calls and no branches. Nothing about them resists a vector
unit, and exp and sin, from the same file and the same macro, vectorize fine.

`-Rpass-missed=loop-vectorize` said *"the cost-model indicates that
vectorization is not beneficial"*, which reads like a preference and invites
`#pragma clang loop vectorize(enable)`. Adding it changes the message to *"the
optimizer was unable to perform the requested transformation"* — the pragma
overrides the cost model's *opinion* but not a cost that is missing entirely.

`-Rpass-analysis=loop-vectorize` gives the real answer:

	Recipe with invalid costs prevented vectorization at
	VF=(vscale x 1, vscale x 2, vscale x 4): call to llvm.is.fpclass

Eighteen calls in the IR, one per affected function, all with test mask 128 —
`fcPosInf`. Every one of the nine functions has a special-case chain that must
return +Inf for +Inf, and the RVV backend has no cost entry for a vector
`llvm.is.fpclass`, so the vectorizer discards every width and falls back to
scalar.

**Not fixed, and worth being precise about what was tried.** LLVM canonicalises
*every* spelling of an infinity test into that one intrinsic:

- `x == INF` — folds
- `x > 0x1.fffffep+127f`, the largest finite float — folds
- `bit_cast<unsigned>(x) == 0x7f800000u` — folds
- `(bit_cast<unsigned>(x) << 1) == 0xff000000u` — folds

Nor is it a scalable-vector problem: `-mrvv-vector-bits=zvl` pins RVV to a
fixed width and the count stays at eighteen. clang 22.1.8.

So this is an upstream gap, not something the kernel source can spell around,
and the honest recording of it is the finding. The one route not yet tried is
to stop asking the question about a float at all: load each element twice, once
as `float` for the polynomial and once as `unsigned` for the classification, so
the comparison is an ordinary integer compare on a value LLVM never saw
bitcast. Whether InstCombine sees through the second load is the open question.

The transferable part: when a compiler says a transform is not *beneficial*,
check whether it is instead not *costed*. The two produce the same silence.

## 26. A check that never fires is a check that passes

**Wrong**, and this one cost two kernels and a wrong explanation for each.

`writesOutsideFrame` in tools/simdgen/verify is the guard against a kernel
storing outside its own frame — into the caller's frame above the stack pointer
or the protected zone below it. Both are fatal under Go, and the failure is
delayed and unrecognisable: the kernel passes its own differential tests, and
the process later dies with SIGSEGV inside `runtime.scanstack`, in a garbage
collection, walking the stack of a goroutine that has nothing to do with it.

Two kernels have exactly that signature. `countAnyVSX` on ppc64le, skipped for
months with the cause recorded as unknown. And the whole compress family on
riscv64 — found only because the noCompress list claimed RVV has no compress
instruction, which is false, and removing the entry produced five working-looking
kernels that destroy the heap.

The guard could not have caught either, because on riscv64 it does not run:

- It needs `tgt.StackReg`, and only ppc64le, the three amd64 tiers and loong64
  declare one. **riscv64, arm64 and s390x declare none**, so the check returns
  immediately on all three.
- `storeMnemonics` lists only s390x and ppc64le instructions. So on amd64 and
  loong64, where the check does run, it matches no store and always passes.
- The displacement is found with a regex for a literal `N(sp)`. An RVV vector
  store is `vse32.v v8, (a5)` — register-indirect through an address computed
  from the stack pointer — and is invisible even with the mnemonic listed.

So of six architectures, the check meaningfully covers **one**: ppc64le. Its
comment says "Only the targets whose ABI makes the mistake possible need an
entry; the rest declare no StackReg and are not checked", which reads as a
deliberate narrowing and is really a description of the gap.

**Fix**: attempted, reverted, and the attempt is the more useful half.

Adding the `StackReg` entries, the missing mnemonics and an `[sp, #16]` pattern
for arm64's bracket syntax makes the check run everywhere — and it immediately
drops six kernels on amd64, fifteen on arm64 and four on riscv64. That looks
like twenty-five latent corruptions found. It is not.

The check reads *any* non-negative offset from the stack pointer as the
caller's frame. That is true on ppc64le, where clang leaves the stack pointer
alone in a leaf and spills into the protected zone below it, so a positive
offset really does belong to the caller. It is false on arm64 and amd64, where
clang emits `sub sp, sp, #N` and then spills at positive offsets **inside the
frame it just allocated**. Every one of those is legal, and the check condemns
them all.

So the guard cannot be extended without first tracking frame allocation: find
the prologue's stack adjustment N, and flag only offsets at or above N. Until
that exists, turning the check on elsewhere trades a false negative for
twenty-one false positives and deletes working kernels.

The transferable part is still the title. A conditional guard that silently
no-ops on most of its inputs reports success identically to a guard that ran
and found nothing, and the only way to tell them apart is to make it fail on
purpose. The corollary, learned here: when you do make it fire, check what it
catches before believing it.

## 27. The corruption is on the stack, because the crash is in the stack scanner

**Wrong**, and it took chasing the wrong thing all the way to a working fix to
find out.

Two kernels destroy memory — `countAnyVSX` on ppc64le and the compress family
on riscv64 — and both die identically: SIGSEGV inside `runtime.scanstack`,
during a collection, walking a goroutine that has nothing to do with either.
The stack unwinder crashing on a frame is a very good reason to suspect a
kernel that wrote outside its frame, and entry 26 found a real gap that would
have hidden exactly that: `writesOutsideFrame` runs on one architecture of six.

The riscv64 compress kernels contain **zero references to the stack pointer**.
Not one store, not one load, no `addi sp, sp, -N` — they allocate no frame at
all. They cannot have written outside a frame they do not have.

What they do have is the contract at `compressK`: the store is unconditional
and it is the *pointer* that advances, so dst must have room for the whole of
src. On AVX-512 that is safe because `vpcompressd` writes only the lanes the
mask selects, and on SVE2 because `compact` does the same. LLVM emitted no
`vcompress` for RVV — thirty-one vector instructions and not one of them the
instruction the whole design rests on — so whatever it did emit stores a full
vector at the output cursor. Near the end of dst the cursor plus a vector
length runs off the end of the slice and into whatever the Go allocator put
next. The heap is corrupt; the collector finds it later, somewhere else.

So the crash site named the *victim*, not the culprit. `scanstack` walks stacks,
but the object graph it walks lives in the heap, and a heap header overwritten
by a stray vector store presents as a stack that will not unwind.

**And the heap-overrun explanation does not survive either.** Written above as
though it were established, it is not. The Go side refuses the call unless
`len(dst) >= len(src)`, and inside the loop `k <= i` while `i + 16 <= n`, so
`k + 16 <= n <= len(dst)`: a sixteen-lane store at `dst+k` is in bounds at every
iteration, by construction. The bound holds whatever LLVM lowers the builtin
to, as long as it stores sixteen elements.

So the cause is **unknown**, which is where `countAnyVSX` has been all along, and
this entry has now produced two confident wrong answers about one bug. The
kernels stay excluded on that basis and not on a story.

What is actually established: LLVM emits no `vcompress.vm` for RVV, the kernels
touch no stack, and the sixteen-lane store cannot leave `dst`. What is not:
anything about what does go wrong. The next step is to stop reasoning about it
and instrument — run the riscv64 compress family under a checked allocator, or
bisect the kernel body the way `countAnyVSX` was bisected, rather than produce a
third theory.

The transferable part: a crash in the garbage collector tells you when the
damage was *noticed*, never where it was *done*. Reading it as a location cost
two wrong hypotheses here — one of which produced a plausible, committed,
entirely irrelevant patch.

## 28. Regressions that scale coherently with n are real

**Wrong**, and this is the more dangerous half of entry 21, because the
reasoning that produced it was better than a guess.

`make bench-check` failed with 16 regressions across `AddInt/i16`,
`MulInt/u8`, `SatSub/u8` and one `AddInt/i32`. A note already existed saying a
stray `go vet` had once fabricated exactly 16 regressions here, so the run was
repeated with the machine quiet — but the numbers looked too structured to be
noise and were argued to be real on that basis:

> The regressions are coherent across four orders of magnitude of n within each
> type — i16 Add is 48-107% slower at every one of five sizes. Noise does not
> scale like that; it scatters.

The second run found **five** regressions, all `MinimumInt/u16`, and **not one
of the original sixteen**. Zero overlap. Both sets are noise.

The argument failed because of how the benchmarks are ordered. All the sizes
for one type and operation run consecutively, so a single transient — a
frequency drop, a migration, a noisy neighbour — spans the whole group and
regresses n=16 through n=65536 together. The coherence that was read as
evidence of a real defect **is the signature of a transient**, not evidence
against one.

The real fault is in the check. These bodies run at 6 to 15 ns/op, where
`-count 6` and a flat 25% threshold cannot separate signal from scheduling, so
every run produces a fresh set of plausible-looking regressions.

**Fix**: not the kernels — nothing was wrong with them. The threshold has to
scale with the measurement: more counts, or a wider band below some ns/op, or
comparing distributions rather than two numbers. Until then a `bench-check`
failure means "run it again", which is a check that cannot be trusted to gate
anything.

The transferable part: structure in noise is not evidence of signal when the
sampling has structure too. Ask what the measurement order was before believing
a pattern that follows it.

## 29. hash/adler32 is a scalar byte loop, so there is 12x to win

**Wrong**, and the whole of a day's work rested on it.

Adler-32 looked like the obvious checksum to accelerate. CRC-32 was measured
first and correctly dismissed: `hash/crc32` reaches 19 GB/s because amd64 and
arm64 both have carry-less multiply and Go uses it, so there is nothing there.
A quick probe then put `adler32.Checksum` at 636us for a megabyte — 1.65 GB/s,
a twelvefold gap, and the algorithm vectorises cleanly. That is a good case.

The probe was wrong. The proper benchmark, on the same machine and the same
input size, measures `hash/adler32` at **220us — 4.7 GB/s**. The ad-hoc timing
loop had measured something else: unwarmed, unpinned, and against a different
buffer. The gap it reported was three times too large, and it was the entire
justification.

A kernel was written anyway, and it is worth recording what it cost, because
the algorithm is genuinely vectorisable and the implementation is genuinely
correct:

- The first version summed the vector lanes with a scalar loop inside the block
  loop — the trap `reduce.c` documents — and measured *identical* to the
  standard library at a megabyte: 222.6us against 223.6us.
- The rewrite used the right identity, `sum (L-i)*d[i] = L*sum(d) -
  sum(i*d[i])`, with three vector accumulators reduced once per 5552-byte
  chunk. It also got the prefix weighting backwards: adding the running sum
  into an accumulator before each block gives `sum_b Sb*(B-1-b)`, not
  `sum_b b*Sb`. Both forms agree for one block, so everything up to 63 bytes
  passed and 64 failed.
- Corrected, it lost three of its six tiers and the fallback path — a reference
  doing a modulo per byte — made the benchmark 8.8x *slower* than the standard
  library.

**Fix.** Reverted, all of it. What survives is the measurement: `hash/adler32`
is not a naive loop, it already blocks to NMAX before reducing, and beating it
needs a better kernel than the one that fits this generator's constraints.

The transferable part is the first paragraph. A number from a hand-rolled
timing loop is not a benchmark, and it is worth exactly as much scrutiny as the
result it is being used to justify — more, when it is the thing that makes the
work look worthwhile.

## 30. Measure the gap against the standard library

**Incomplete**, and the missing half wasted most of a task twice in one day.

`bytes.ToLower` measures 1.9 GB/s, against a memory ceiling nearer 30, because
it walks the input as runes to cover the whole of Unicode. That is a real and
correctly measured gap, and ASCII case folding is a trivially vectorisable
elementwise operation. So a kernel was written, wired through the manifest, the
reference and the kernel slots — and every one of those edits collided with
code that was already there. `simd_toupper_ascii` already existed in
`csrc/bytes.c`. The slot already existed in `kernel.Bytes`. `ToLowerASCII` was
already exported.

**`simd.ToLowerASCII` was already running at 18.9 GB/s — ten times the standard
library.** The gap was real, and this library had already closed it.

The same shape had appeared an hour earlier with Adler-32, where the
justification was a mismeasured baseline (entry 29). Together they are one
mistake with two halves: *before* believing there is work to do, measure the
standard library **with the benchmark harness**, and check what this package
already exports. Either check alone is insufficient. The first tells you
whether anyone needs it; the second tells you whether it is already done.

For the record, the text API already covers `Index`, `IndexAll`, `IndexAny`,
`Count`, `CountAny`, `Contains`, `HasPrefix`, `HasSuffix`, `EqualFoldASCII`,
`ToLowerASCII`, `ToUpperASCII`, `ReplaceByte`, `TrimSpaceASCII`, `TrimAny`,
`ValidUTF8`, `IsASCII`, hex and base64 — which is most of a roadmap entry that
was written as though none of it existed.

**Fix.** Reverted, nothing shipped, and the roadmap entry rewritten to name only
what is genuinely absent. The cost of this one was low because the compiler
caught the collision; had the names differed slightly it would have shipped two
implementations of the same thing.

## 31. Renaming a function is a mechanical change

**Wrong** when a function of the same name and signature already exists, which
is precisely when you are renaming to avoid a collision.

`ConvolveFullInto` was added beside the existing `ConvolveInto`. They are
different operations — the existing one is numpy's "valid" mode, returning
`len(sig)-len(ker)+1` elements where the kernel fully overlaps, and the new one
is "full" mode, returning `len(a)+len(b)-1` — so the new function was renamed
to make room.

The rename missed one call site, inside `CorrelateFullInto`, which still said
`ConvolveInto(dst, a, rev)`. **It compiled.** Both functions take
`(dst, a, b []T)` over the same constraint, so the call silently rebound to the
other operation, and correlation returned the valid-mode answer padded with
zeros: `[50 80 0 0]` where `[20 50 80 30]` was wanted — the right numbers,
shifted, with the ends missing.

The compiler cannot help here. A missed rename normally fails to build because
the old name is gone; when the old name still exists and has a compatible
signature, the failure is silent and semantic.

**Fix.** Not just the call site — an assertion in the edit script that no bare
`ConvolveInto(` survives outside the `ConvolveFullInto(` occurrences. The rule
is that any rename made to resolve an ambiguity must be followed by a search
for the *old* name, not a rebuild, because a rebuild is exactly what will not
catch it.

Worth noting how it was found: not by the differential tests, which passed,
but by printing the output of a four-element worked example. The test that
should have caught it asserted a peak position the author had reasoned out
incorrectly, so it failed for the wrong reason and was nearly "fixed" by
adjusting the expectation. Checking a whole short vector against arithmetic
done by hand is both stricter and much harder to talk yourself out of.

## 32. The frame-write check just needs the frame size added

**Wrong four times running**, which is the point of this entry.

Entry 26 found that `writesOutsideFrame` runs on one architecture of six.
Entry 26's own attempted fix was reverted because switching it on flagged
twenty-one legal spills on arm64. The diagnosis then was that it needed to know
the frame the prologue allocated, and skip anything below it — a positive
offset inside your own frame is where a spill belongs.

That diagnosis is right and it is not sufficient. Implemented, with
`stackAdjust` extended to riscv64, ppc64le, s390x and loong64, `StackReg`
declared for the three targets that had none, the RVV and NEON store mnemonics
added, arm64's `[sp, #16]` bracket syntax handled, and pre/post-indexed stores
excluded as allocations rather than overruns — arm64 still lost twenty-one
kernels and riscv64 five.

Four hypotheses, each plausible, each wrong:

1. *Positive offsets below the frame are legal.* True, implemented, insufficient.
2. *clang allocates arm64 frames with a pre-indexed `stp x29, x30, [sp, #-N]!`,
   which the size parser misses.* It does not; the disassembly shows a separate
   `sub sp, sp, #0xf0`.
3. *The displacement is printed in hex and the parser reads decimal.* True of
   `#0xf0`, and fixing it changed nothing.
4. *`reSubSPAA` was spelled `#([0-9]+)` so the hex widening missed it.* Also
   true, also fixed, also changed nothing.

Every rejection is at `sp+0` — a bare `[sp]` with no displacement, fifty-three
of them. So the frame size is still measuring zero on arm64 after all four
fixes, and the reason is not yet known.

**Fix.** Reverted, again. Coverage is back to 772/740/708/392/654/576 and the
check still covers one architecture. What is now known and was not before: the
frame-size idea is necessary but not the whole answer, `stackAdjust` returns
zero for arm64 even with a correct regex against a real `sub sp, sp, #0xf0`,
and the surviving rejections are all zero-displacement stores. The next attempt
should start by printing what `stackAdjust` actually receives as `in.Operands`
for that instruction, rather than by reasoning about what it ought to receive.

The transferable part: four consecutive plausible diagnoses, each confirmed
true in isolation, none of which moved the number. When a fix that must work
does not, the assumption to check is not the fix — it is the measurement you
are using to judge it.

## 33. The corruption was exotic

**Wrong.** It was a stack overflow, and the check that says so had been dead on
that architecture the whole time.

Two kernels destroyed memory in a way that took most of a session to explain:
`countAnyVSX` on ppc64le, and the compress family on riscv64. Both died the
same way — SIGSEGV inside `runtime.scanstack`, during a collection, walking a
goroutine unrelated to either. Two confident explanations were published and
retracted along the way, an out-of-frame stack write (entry 27) and a heap
overrun past `dst` (retracted in the same entry), and the honest conclusion was
that the cause was unknown.

It was not exotic. Every kernel in the riscv64 compress family spills far past
the 512-byte budget a NOSPLIT function is allowed: 640 bytes for the int32
compress, 1216 for float32, 1792 for the 64-bit types, 2032 for the partitions.
A NOSPLIT function that uses four times the stack the runtime guarantees runs
off the end of the goroutine stack. The damage is silent and the collector
finds it later, somewhere else.

The budget check has existed since long before any of this. It never fired on
riscv64 because `stackAdjust` had no case for the architecture and so measured
every frame as zero — and it never fired on arm64, s390x or loong64 either, for
the same reason. Underneath that, `parseInstr` was deleting every arm64
immediate by cutting operands at the first `#`, which on that architecture is
the immediate prefix (entry 32). So the measurement feeding the check was
empty, and the check reported success.

**Fix.** Both are in the two commits that precede this entry. The parser cut
moves to `"# "`; `stackAdjust` learns riscv64, ppc64le, s390x and loong64
prologues and arm64's pre-indexed form; `writesOutsideFrame` reads both
addressing syntaxes and compares against the frame actually allocated. arm64
keeps all 740 slots with zero false positives, and riscv64 loses ten kernels
that were genuinely over budget and shipping.

The transferable part is the arithmetic of it. Four entries in this file — 26,
27, 30 and 32 — were spent on a fault that a working check would have named in
one line. The cost of a check that silently no-ops is not the bug it misses; it
is every hour spent explaining that bug by other means.

## 34. A minimal reproduction proves the trigger is elsewhere

**Wrong**, when the thing you removed to minimise it is the thing that caused
it.

Eighteen riscv64 kernels fail to vectorise because each contains an
`llvm.is.fpclass` the RVV backend cannot cost. Entry 25 established that much.
The suspicion was the special-case chain — `x < 0 ? NaN : x == 0 ? -Inf :
x == Inf ? Inf : x != x ? x : v` — since a positive-infinity test is exactly
what folds into that intrinsic.

So it was extracted into a standalone probe. The identical four-test chain,
compiled for the same target with the same flags, produced **zero**
`is.fpclass` and vectorised to eighteen vector instructions. Reordering it and
rewriting the infinity test as a magnitude comparison did too. Three
formulations, none reproducing the fault, and the reasonable conclusion was
that the trigger lay somewhere else in the inlined body.

It did not. Reading `simd_log_f32` at `-O1` shows one `is.fpclass`, applied
directly to the loaded value, before anything else happens to it. LLVM hoists
the infinity guard to the front of the function — because `log2_frac`, which
the guard protects, produces garbage for an infinite input — and canonicalises
the hoisted guard into the intrinsic.

The probe removed exactly that. Its polynomial was `x * 0.5f + 1.0f`, which is
perfectly well-defined at infinity, so there was nothing to hoist and no guard
to canonicalise. **The minimisation deleted the cause and kept the symptom's
shape.**

**Fix.** Not yet applied; the mechanism now points at making `log2_frac` total
so there is nothing to guard, rather than at rewriting the test that guards it.

The transferable part: a reduction that does not reproduce is evidence about
the reduction, not about the original. Before concluding "the trigger is
elsewhere", check whether the property you simplified away was load-bearing —
here it was the *interaction* between the guard and what it guarded, which
survives in neither half alone.

## 35. There must be some way to spell it that LLVM does not fold

**Wrong**, and after five attempts that is the finding rather than a
complaint.

Eighteen riscv64 kernels will not vectorise because each contains an
`llvm.is.fpclass` the RVV backend cannot cost. Entry 34 identified the
mechanism: LLVM hoists the positive-infinity guard to the top of the function,
because `log2_frac`'s result is meaningless for an infinite input, and
canonicalises the hoisted guard into the intrinsic.

That suggested a fix with a good shape — remove the *test* and let the special
values fall out of arithmetic. Clamp with a minimum, which is one instruction
and classifies nothing, then multiply by a correction factor that is exactly 1
for every finite input, infinite for `+Inf`, and NaN for a NaN:

	float xc = __builtin_elementwise_min(x, 0x1.fffffep+127f);
	v = poly(xc) * (1.0f + (x - xc));

No comparison against infinity anywhere. LLVM produced it anyway:

	%12 = call float @llvm.minnum.f32(float %9, float 0x47EFFFFFE0000000)
	%13 = call i1 @llvm.is.fpclass.f32(float %12, i32 128)

It derived the classification *from the clamp*, because `minnum(x, MAX)` equals
`MAX` exactly when `x >= MAX`, and folded that into the intrinsic.

Five spellings have now been tried and all five fold: the float compare against
infinity, the compare against the largest finite value, an integer comparison
of the bit pattern, a shifted integer comparison, and now a clamp that never
mentions infinity at all. LLVM is not failing to notice these are equivalent —
it is deliberately normalising every form of "is this positive infinity" into
one intrinsic, which is good compiler design and exactly what makes this
unfixable in the source.

**Fix.** Upstream, and only upstream: an RVV cost model entry for
`llvm.is.fpclass`. clang 22.1.8 here. The task becomes tracking that change
rather than looking for a sixth spelling.

The transferable part: when a compiler normalises many forms to one, the number
of remaining forms to try is zero, and the effort is better spent confirming
that than continuing. Five is more than enough evidence; the second or third
should have prompted asking whether the canonicalisation was the point.
