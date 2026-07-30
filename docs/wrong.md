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

## 36. The ppc64le corruption was countAnyVSX

**Wrong**, and the disproof took one control run that the original
fourteen-run bisection never had.

Replaying the FuzzDifferential seed corpus under qemu-ppc64le crashes the
process — SIGURG, then SIGSEGV inside `runtime.sigtrampgo` dereferencing a
garbage g. With countAnyVSX skipped, the first replay batch passed; enabled,
the first batch crashed. One variable, clean flip, case closed — until the
batches were repeated. Enabled crashed 2 of 7. **Skipped crashed 3 of 7.** The
"clean flip" was two coin tosses, and the historic bisection that condemned
countAnyVSX was fourteen tosses read as a proof.

What the controls then established, six batches each:

	kernels registered, preemption on      crashes (~30-40% of batches)
	GOSIMD=scalar (never executed)         6/6 PASS
	GODEBUG=asyncpreemptoff=1              6/6 PASS

So the crash needs a generated kernel *executing* when Go's preemption signal
lands, and the suspect set is exactly the three kernels this fuzz reaches on
ppc64le: addFloat64VSX, minMaxFloat64VSX, validUTF8VSX. Their disassembly
shows no r30 use, no nonzero r0, no pool/R2 repointing — the three invariants
checked so far — which is where the trail currently ends: a three-way
registration bisection is the next step, not a theory.

Found in passing and real regardless: **eleven ppc64le kernels write a nonzero
value to r0 mid-body** — b64Decode, b64Encode, hexDecode, countAny, the three
complex64 reductions, l1diff/sumsqdiff f64, mul3 i8/u8 — relying on the
generator's epilogue `li r0, 0` to restore the ABI's r0-is-zero before
returning. Between first write and epilogue, a signal sees r0 ≠ 0 in an ABI
where Go's compiler and runtime treat r0 as constant zero. None of the eleven
is called by the crashing fuzz, so this is not tonight's crash — it is a
second latent class of the same shape, and it needs a verifier rule: GoOwned
catches registers by *name*; nothing yet checks a register owned by *value*.

The transferable parts: a bisection over a probabilistic failure needs
repeated trials per step, or it converges confidently on an innocent; and
`GOSIMD=scalar` / `asyncpreemptoff` cost two minutes and partitioned the
space better than three hypotheses had — controls first, theories second.

## 37. One long benchmark run gives one machine state

**Wrong** — it gives a gradient.

The regenerated baseline was recorded in a single 85-minute pinned run, and
the validation pass flagged nineteen regressions. The Median rows re-measured
on a quiet machine sat within ±2% of baseline: those flags were contamination
from qemu work sharing the package (a trade knowingly made). But the ParseInts
rows re-measured **30-50% faster than the baseline** — both the simd and the
strconv side. The baseline itself is slow there.

The benchmark files run alphabetically, so an 85-minute sustained run records
the early families on a cold package at full boost and the late families —
parse, sort, text — heat-soaked. Short, core-bound benchmarks late in the
alphabet carry a machine-state floor of tens of percent, which no 25%
threshold and no minimum-estimator can see through, because it is not noise
around a value: it is a different value.

**Fix**: not attempted tonight. The gate needs one of: interleaved
baseline/candidate measurement of the same benchmark back-to-back; an
auto-triage pass that re-runs only flagged rows alone and compares those; or a
per-family thermal anchor. Until then, the protocol stands — a flagged row is
re-measured alone before it is believed — and it caught this exactly as
designed. What must NOT happen is bench-update after a long hot run being
treated as truth for short benchmarks.

## 38. A crash counter counts crashes

**Wrong** — it counts *non-successes*, and the difference voided an evening's
"decisive" result.

To isolate which ppc64le kernel dies under signal load, a hammer program ran
one operation per invocation, and a batch loop counted every run whose last
line was not `OK` as a crash. All three suspects "crashed 6/6, alone". Strong,
clean, wrong: after an unrelated edit, the same binary "crashed" with
preemption off, with signals off, in every configuration — because the guest
was panicking on `os.Args[1]` before touching a kernel. The argument had
stopped surviving the qemu invocation, the panic's last line was not `OK`, and
the counter filed it under crash. Whether the *original* 6/6 rounds measured
kernels or argv is unrecoverable — which is the point: **every hammer-derived
conclusion had to be discarded**, including the ones that might have been
true.

The results that survive are the ones from the untainted harness — the real
fuzz binary, whose flags demonstrably worked because `-test.run` and
`-test.count` visibly did their jobs, and whose crashes left cores with the
right symptoms in them.

**Fix**: the counter now distinguishes outcomes only where the success token
is printed by code that runs *after* the operation under test, and any
experiment series whose harness is found lying is discarded whole, not
patched and partially believed. The transferable rule joins entries 28 and
31: before believing a discriminator, make it fail on purpose in both
directions — a harness that cannot distinguish "kernel crashed" from "harness
crashed" discriminates nothing.

## 39. Only kernels need the FMA fusion barrier

**Wrong.** The *reference* needs it too, and the reference is where it was
missing.

`matMul`'s inner line is `row[j] += T(s * br[j])`, and that conversion is not
decoration: Go permits `x*y + z` to become a fused multiply-add, which keeps
the product at full precision and yields a **different, more accurate** answer
than the multiply-then-add a kernel performs. The conversion forces the
rounding the kernel does.

The new packed reference `matMulPk` was written as `dst[...] += s * bp[...]`.
It passed every amd64 tier — the Go compiler did not fuse there — and failed
on ppc64le at one ULP, `-3.0407238` against `-3.0407236`, in float32 where a
single lost rounding is visible. Both paths were running the *reference* on
that target, so this was reference-versus-reference: two functions that are
supposed to compute identical values, one of which the compiler was allowed to
improve.

The trap is that the bug is invisible on the development machine by
construction. Fusion is a per-architecture compiler choice, so a missing
barrier is a latent disagreement that only appears on the targets that have
FMA and only for types where one rounding shows.

**Fix**: the conversion, plus a comment at the line saying why it is there —
`matMul` had the barrier but not the explanation, which is how the copy lost
it.

## 40. The counter learned its lesson

**Wrong**, twice, in the same night.

Entry 38 recorded a crash counter that filed a guest panic as a kernel crash,
and the fix was to classify outcomes rather than count non-successes. The
replacement counter classified three ways — PASS, signal, unclassified — and
was used to conclude that a rebuilt ppc64le tree "crashed 12 of 12", a
deterministic new failure that looked alarming beside the intermittent one
under investigation.

It was the GEMM test failing. `go test` prints `FAIL`, not `PASS`, and the
signal branch did not match, so twelve ordinary test failures landed in the
bucket labelled *crash* — the same conflation as entry 38, in a counter
written specifically to avoid it, because the third bucket was named
"unclassified" and quietly believed to be empty rather than being *printed*.

**Fix**: the classifier must have no default bucket. Every outcome is named —
PASS, FAIL, signal — and anything unmatched is printed in full, not tallied.
A counter that can absorb an outcome it does not understand will absorb the
one that matters.

## 41. Every emulated lane was passing. None of them read its flags.

**Wrong**, for the entire life of the qemu lanes, and it invalidates a large
part of what this file records about the non-amd64 backends.

The rule from entry 40 — no default bucket — was applied to a fresh ppc64le
crash hunt, and both arms came back clean: 12 of 12 PASS ungated, 12 of 12
under `GOSIMD=scalar`. The classifier was correct this time. The input was
not. `-test.v` on the same binary printed no `=== RUN` lines at all, and
`-test.list '.*'` listed nothing while still taking twenty-eight seconds.

A two-test toy package reproduced it, and a plain Go program named the cause:

```
$ qemu-ppc64le ./argv                      argc=1 argv=["./argv"]
$ qemu-ppc64le ./argv ONE TWO              argc=2 argv=["ONE" "TWO"]
```

The program path is gone. These emulators come from the binfmt images, and
they follow the `binfmt_misc` *preserve-argv[0]* calling convention, in which
the kernel invokes the interpreter as `qemu-ARCH <path> <argv0> <args...>`.
Run by hand, the first real argument is eaten as the guest's `argv[0]` and
everything after it shifts one place left. `flag.Parse` then sees a list whose
first element is not a flag, stops there, and reports nothing wrong: Go's flag
package treats the first non-flag as the start of positional arguments. All
five emulators here behave this way — aarch64, riscv64, s390x, ppc64le,
loongarch64 — while the distribution's `qemu-user` does not, which is why the
one hosted lane that uses apt's build was fine.

**A Go test binary with no flags runs its full default suite and prints
`PASS`.** That is the whole trap. The flags did not error, they evaporated,
and the lane stayed green while testing something other than what it said.
Every `-test.run` selected nothing, every `-test.count` ran once, every
`-test.short` ran long. The "seed replay ×20" reproducer in the ppc64le
dossier was a single full-suite run, which is why its crash rate never
responded to anything done to the seeds.

The worst casualty is the guard against precisely this class of bug. The
Makefile asserted an accelerated tier before trusting a PASS:

```
QEMU_CPU=$(CPU) $(QEMU) $$bin -require-accelerated || exit 1
```

That flag had never once been evaluated. Verified directly — `rv64` with no
vector extension, which must fail:

```
$ QEMU_CPU=rv64 qemu-riscv64 ./simdinfo -require-accelerated
riscv64 tier=scalar available=[scalar]
exit=0                                      # wrong
$ QEMU_CPU=rv64 qemu-riscv64 ./simdinfo ./simdinfo -require-accelerated
riscv64 tier=scalar available=[scalar]
simdinfo: the portable path was selected, ...
exit=1                                      # correct
```

`simdinfo`'s own doc comment says a suite that skips every accelerated tier
"passes, and looks exactly like one that tested them". The check written to
prevent that was in that exact state itself, for the same reason, and could
not have reported it.

**Fix**: pass the binary path twice, so the throwaway lands in the `argv[0]`
slot. And because a silent convention change is what caused this, the padding
is no longer trusted on faith — `simdinfo -argv0-probe` exits 7 and does
nothing else, `make qemu-run-probe` asserts that 7 before any lane runs, and
an emulator that handles argv differently now fails loudly instead of
quietly testing the wrong thing.

The general lesson is narrower than "verify your harness" and worth stating
exactly: **a flag that is dropped is indistinguishable from a flag that had
nothing to say.** Nothing about a green run can tell the two apart, so a
harness must contain at least one flag whose effect is visible in the exit
code — an assertion that fails when it is *not* delivered. Everything else in
this repo asserts on the result of a test; this asserts that the test was the
one asked for.

## 42. The ppc64le crash needed a kernel bisection

**Wrong**. It needed the ABI rule that was already written, and the bisection
would have found nothing, because the culprit was not one kernel.

The dossier had narrowed the intermittent SIGSEGV to three suspects and
prescribed three binaries with one registration removed from each. That
experiment was never run. The r0-by-value rule — reject a ppc64le kernel that
parks a nonzero value in the register Go's ABI defines as constant zero — was
added for a different reason, as a verifier rule rather than a fix, and the
crash stopped.

Which is not evidence by itself, so it was tested directly: two trees, one
with the rule active and one with it disabled so the fifteen offending kernels
register again, run interleaved so that drift in machine state cannot favour
either, classified with no default bucket.

```
FINAL r0on  (violators registered): pass=15 fail=0 signal=5 /20
FINAL r0off (rule active):          pass=20 fail=0 signal=0 /20
```

Five crashes and none, and under the null hypothesis that both arms crash
alike, five of five landing in the arm named in advance is p ≈ 0.03. The
mechanism was already written down and matches exactly: a signal arriving
while r0 holds a nonzero value runs the runtime against an ABI whose zero
register is not zero, which is how a garbage `g` reaches `sigtrampgo` and
faults dereferencing `g.m`.

So the answer was not "which kernel" but "which ABI register, by value rather
than by name". Fifteen kernels were doing it, `validUTF8` among them, which is
why the earlier bisection over three candidates kept reading coin flips: **any
one of fifteen could crash a run, so removing any one of three changed
nothing measurable.** The v0.2.0 bisection that blamed `countAnyVSX` failed
for the same reason, and entry 36 recorded that it was innocent without
recognising why the search itself could not work.

**Fix**: none to make — the rule was the fix. What is worth keeping is the
shape of the error. A bisection assumes one cause. Fifteen independent causes
with a per-run probability each look exactly like noise under it, and the way
out was not a better bisection but a rule that named the *class*.

## 43. ParseFloats needed a kernel

**Wrong**, and the generator said so before a single measurement had to be
argued about. Every target refused it:

```
simd_parse_floats (LLVM did not vectorize it for neon)
simd_parse_floats (LLVM did not vectorize it for sve2)
simd_parse_floats (LLVM did not vectorize it for rvv)
simd_parse_floats (LLVM did not vectorize it for vsx)
simd_parse_floats (LLVM did not vectorize it for vx)
simd_parse_floats (LLVM did not vectorize it for lasx)
simd_parse_floats (needs more argument registers than amd64 has)
```

Both refusals are right. Parsing a float is a dependent scan that has to find
a decimal point and an exponent before it knows what any digit is worth, and
the eligibility branch is the algorithm rather than an aside. The seventh
argument — the resume offset, needed because field offsets are absolute — is
one past what amd64 passes in registers.

The useful part is that the speedup never depended on any of it. It comes from
Clinger's fast path, which is scalar: a mantissa under 2^53 and an exponent
within [-22, 22] make both operands exactly representable, so `mant * 10^exp`
is a single rounding and therefore *the correctly rounded result*. Not close
to strconv — identical to it, which is why this needs no `Fast*` name and no
tolerance in its test.

Shipped as plain Go: 2.3x on two-decimal CSV, 2.0x on scientific, and 0.77x on
17-significant-digit round-trip values, which are 10% eligible and pay for
being scanned twice. That last row is in the doc comment, because a function
whose speed depends on data the caller chose has to say which data.

**Fix**: none. The lesson is that "this library accelerates X" and "X wants a
kernel" are different claims, and the generator refusing to emit one is
evidence about the second, not a problem to route around. Three things have
now been faster without a kernel — the histogram private tables, the checksum
family, and this — and each was found by measuring rather than by assuming the
vector unit was the point.

## 44. Fast* means less accurate

**Wrong** for the scans, and the assumption was mine, carried over from the
transcendentals where it is true — `FastExp` really does trade 1.0 ULP for 3.5.

Task 68 proposed relaxing the promise that a prefix scan matches a naive
serial loop, on the reasoning that the promise is what blocks the vectorized
form. That reasoning is right. What it left implicit is that relaxing it costs
accuracy, and measured against a long-double scan of a million values, it does
the opposite:

```
corpus              serial mean|err|   blocked mean|err|   closer to truth
all ones            0                  0                   tie
mixed sign          4.088e-10          4.034e-10           blocked
uniform positive    2.436e-09          1.891e-09           blocked, 680k/1M
1e16 then ones      5.000e+05          1.000e+00           blocked, 999998/1M
```

The last row is the mechanism in one line. A running accumulator at 1e16
cannot represent a `+1` increment at all — the spacing of doubles there is 2 —
so every one of a million increments is lost. The blocked scan sums sixteen
small values among themselves before merging the large carry once, so they
survive. Blocked summation has O(log n) error growth against a serial
accumulator's O(n); this is textbook and it was simply not applied to the
question.

So the honest statement of what `FastCumSum` trades is **agreement, not
accuracy**. It does not match the loop a caller would write; it is closer to
the answer that loop was approximating. Both halves belong in the doc comment,
because a caller told only the first half will reasonably assume the second.

**Fix**: the doc comment states it and a test asserts it — `moreAccurateThanSerial`
fails if the total error ever stops being lower, which also catches the scan
silently degrading into the serial loop.

## 45. The integer scans would benefit too

**Wrong**, and this is the same measurement cutting the other way.

Two's-complement addition and multiplication are associative, so the log-shift
scan is *exactly* the serial loop for integers — no Fast prefix needed, no
contract touched. That looked like the free half of the task. Measured against
the Go loop actually shipped, at four million elements:

```
int32 product   2509 us -> 1230 us   2.04x   shipped
int64 product   2596 us -> 3884 us   0.67x   not shipped
int32 sum        980 us -> 1082 us   0.91x   not shipped
int64 sum       1441 us -> 2165 us   0.67x   not shipped
```

Three of the four lose. The rule that explains all of them, and the float min
and max scans this file already recorded as 46% slower, is not associativity —
every one of these has it — but **the latency of the serial combine**. A scan
replaces n dependent combines with log2(L) per block of L, so it wins only
where the serial chain is latency-bound. Integer addition has one-cycle
latency: the serial loop already issues an element per cycle, there is no
stall to fill, and the scan's shuffle chain plus the lane extract that carries
the running value between blocks is pure added cost. int64 product loses for a
different reason — there is no 64-bit vector multiply below AVX-512DQ, so the
"vector" form is emulated.

Only int32 product wins, because a 32-bit multiply has three-cycle latency and
a real vector instruction.

**Fix**: ship the one that wins. The general lesson is that "this operation is
associative" answers whether a scan is *correct* and says nothing about
whether it is *faster*, and the second question has to be asked per operation
and per type — four types of one operation gave three different answers here.

## 46. LoongArch branches are already resolved, like RISC-V's

**Wrong**, and this one shipped an infinite loop into 23 kernels before a test
lane hung on it.

The generator refused 23 loong64 kernels with "references .L0, which is not a
constant pool". `.L0` turned out to be a local branch label in `.text`, reached
by `R_LARCH_B16` and `R_LARCH_B26` relocations, not a pool at all — and RISC-V
and AArch64 branches were already on the self-relative list two lines above:

```go
case "R_RISCV_BRANCH", "R_RISCV_JAL", ...:
case "R_AARCH64_JUMP26", "R_AARCH64_CONDBR19", ...:
```

so adding the LoongArch spellings looked like completing an obvious omission.
It recovered 23 kernels and every check stayed green.

Then the emulated lane, which had finished in minutes an hour earlier, sat at
42 minutes of CPU on one package. Bisecting the tests found exactly one
hanging — `TestAppendUTF16`, whose only accelerated loong64 dependency was
`simd_widen_u8_u16`, one of the 23. Disassembling it says the rest:

```
73fc: blez $a2, 0 <simd_widen_u8_u16>
7404: bgeu $a2, $a3, 12 <simd_widen_u8_u16+0x14>
740c: b    0 <simd_widen_u8_u16+0x10>       <- displacement 0: branches to itself
```

On RISC-V and AArch64 the assembler resolves an internal branch and leaves the
relocation for the linker's relaxation pass, so the displacement in the
instruction is already right and copying the bytes is safe. **On LoongArch it
does not.** clang emits the branch with a displacement of zero and expects the
linker to supply it. Copied verbatim, `b 0` is a branch to itself.

The premise was "these relocations mean the same thing on every target because
they have the same shape". They do not. A relocation type says what the linker
should compute; it says nothing about whether the assembler already did.

**Fix**: reverted. Those kernels keep the portable path and the rejection
stands — it was never a gap to route around, it was the generator correctly
declining to copy code it cannot relocate. `R_LARCH_ALIGN` and `R_LARCH_RELAX`
stay skipped, because those genuinely are markers and patch nothing; they
account for five of the 23, which is why the count settled at 659 rather than
back at 654.

Two things made this cheap instead of expensive, and both were built earlier
the same night. The emulated lane runs the real suite rather than a smoke
test, so an infinite loop showed up as a hang instead of as a kernel nobody
exercised; and entry 41's argv fix is what made that lane execute the tests it
claimed to. A verification lane that had been silently passing empty runs
would have shipped this.

**And the code was fragile independently of the bug.** `AppendUTF16` advances
by whatever the ASCII-run scan returns, so a kernel returning zero makes no
progress and spins. The kernel was wrong here, but a loop whose termination
depends on a kernel returning a positive number should not be written that
way; see the guard added with this entry.

## 47. A general n-ary closure combinator is worth building

**Wrong.** It is slower than the loop the caller would have written without
it, at every size, in both of its plausible forms.

The idea was `ZipInto(dst, f, srcs...)`, so a caller could express a fused
expression this library has never heard of and still make one pass over
memory. The known objection was that a closure call per element defeats
vectorization; the assumption worth testing was that one pass with a closure
still beats two passes without one. Measured on `dst = a*b + c`, nanoseconds:

```
n           handwritten   composed   zipVariadic   zipFixedArity   fused kernel
1024              728.7      74.72          2222            1163          52.59
262144           195722      86528        590483          310461          60707
4194304         4377370    5133919       8856742         5446108        3597582
```

`zipVariadic` is the honest general form and is 2.0x to 3.0x slower than a
plain Go loop. `zipFixedArity` removes the per-element argument slice, which is
the most generous version anyone would build, and is still 1.24x to 1.6x
slower than the same plain loop. The one-pass advantage is real at four
million elements — `composed` does two passes and loses to `handwritten`
there — and it is not nearly enough to pay for the closure.

Meanwhile the fused kernel is 13.9x faster than the handwritten loop at n=1024
and 3.2x at 262144, because it is one pass *and* vectorized.

So the combinator would occupy the worst corner of the design: slower than
doing nothing, and far slower than the thing it sits next to. The value it was
meant to add — one pass — is available only by not calling a closure.

**Fix**: not shipped. The guidance instead is the one the numbers support:
reach for the fused catalogue, and where the expression is not in it, write the
loop. A note at [AddAll] says so with these numbers, because "why is there no
general Zip" is a reasonable question to have answered in the package rather
than in a commit message.

The narrower lesson is about which assumption to test. That a closure costs
something was never in doubt; the claim actually load-bearing was that saving
a memory pass would outweigh it. That is the one that needed a number, and it
was off by more than an order of magnitude at small n.

## 48. A bench-check regression means the code got slower

**Wrong**, and every one of the flagged regressions after a night of work
turned out to be something else.

`make bench-check` compares against a recorded baseline. After the work in
entries 41–47 it reported 21 regressions, several above +70%. Two things were
already suspicious: some were in arms this work cannot touch — `impl=go`,
`impl=naive`, `impl=std` are pure Go and standard library — and a second run
flagged 25, of which only 4 were in the first list. Sixteen and twenty-one
respectively were noise, which is what the "run it twice, trust the
intersection" rule in `testdata/bench/README.md` exists for.

Re-measuring the 4 survivors with ten samples and the minimum estimator
dropped another: `Compress/n=65536/p=0.50/impl=simd` came back at 12058
against a baseline of 11953, +0.9%.

The remaining three looked real, persisted across every re-run, and included a
**pure Go** arm. So the next check was whether the code had changed at all —
and `indexByteAVX512`, `countByteAVX512` and `countSeqAVX512` are byte-for-byte
identical to their pre-work versions, at the same line of the same file.

The decisive test is the one that should have been first: **build the old tree
and run it now.**

```
                                        baseline   old tree, now   new tree, now
Compress/n=65536/p=0.90/impl=go            19648           27674           25147
TextCount/needle=zebra/n=1048576/simd      11744           16775           16826
TextIndexByte/n=4096/impl=simd             22.23           30.49           30.35
```

The old tree is as slow or slower. There is no regression: the machine is
about 35% slower for these than when the baseline was taken — a machine that
has been under load for eleven hours, whose boost behaviour is not what it was
at hour one, even with the governor pinned to `performance`.

**Fix**: none needed, and the baseline is deliberately NOT updated — recording
today's numbers would bake a degraded machine state into the reference. What
is added is the protocol: a baseline diff nominates *candidates*, and the only
thing that convicts is an A/B against the actual old code at the same moment
on the same machine. A worktree at the previous commit costs a minute and
answers it outright.

The failure this avoids is the expensive kind. Three plausible, reproducible,
double-confirmed "regressions" would have sent someone hunting for a cause
inside code that had not changed a byte.

## 49. Performance on the other architectures cannot be verified without them

**Wrong**, mostly — and I had already been corrected once for the same shape of
claim.

Task 64 sat open for a long time saying performance verification "needs real
hardware", on the reasoning that qemu-user emulates instruction semantics and
not timing. That reasoning is correct and the conclusion drawn from it was too
broad. What needs the hardware is *wall-clock throughput on that hardware*.
What does not:

- **Whether the kernel is vectorized at all.** Already checked, by the
  generator, on every target.
- **How many cycles the instruction stream should take.** `llvm-mca` is LLVM's
  own scheduling model — the same tables the compiler uses to order
  instructions — and it will model any target LLVM has tables for. It ships
  with clang, which this repository already requires.

`make perf-model` now reports cycles per element for each kernel against the
same kernel compiled without vectorization, on amd64, arm64, ppc64le and
s390x. Nothing measured below 1.2x, and the shape is what it should be:
speedups track the vector width for the arithmetic kernels and run far ahead
of it for the byte kernels, where a scalar loop does one byte at a time.

The model is checked rather than assumed. Against measured amd64, comparing
the avx512 and avx2 tiers — the same C, two widths, which is the comparison it
is being trusted to make across architectures:

```
Add float64, n=1024      model 2.00x   measured 1.79x   12% high
AddInt int32, n=4096     model 1.98x   measured 1.89x    5% high
```

Consistently optimistic and close, which is what a model with no memory system
should be. Absolute cycle counts are optimistic by about a factor of two for
the same reason, so the tool says to read ratios and not cycles.

Two targets stay unmodelled, both for stated reasons rather than for want of
trying. riscv64's RVV has a boot-time vector length and the kernels are
written to be length-agnostic, so a number would be a number for a machine
nobody named. loong64 has no scheduling tables in LLVM at all — it can be
assembled and not scheduled — which is an upstream gap and not something to
work around.

**Fix**: the tool, and the narrower claim. What genuinely still needs hardware
is wall-clock under a real memory system, and macOS and Windows, which have
zero OS-specific source in this library and now build and vet clean for
darwin/amd64, darwin/arm64, windows/amd64, windows/arm64 and freebsd/amd64.
The residual risk there is `x/sys/cpu` feature detection, which is the whole
of the OS-dependent surface.

The lesson is the one from the last time: "needs hardware" is a conclusion that
deserves the same scepticism as any other. Ask what *exactly* needs it, and
then check whether the rest can be had another way. Most of it could.

## 50. That '#' bug is fixed

**Wrong.** I wrote it again, in a new tool, within an hour of writing the entry
that describes it.

Entry 62's root cause was a disassembly parser that cut each line at the first
`#` to strip comments, which on AArch64 deletes every immediate — `[x10, #-16]`
becomes `[x10,` — and silently disabled the frame and stack checks on that
architecture for months.

`perfmodel` extracts a loop body from clang's assembly output and strips
comments. It cut at the first `#`. The first arm64 run produced:

```
ldp  q0, q3, [x10,
subs x12, x12,
```

and llvm-mca rejected all seven kernels with "unknown token in expression".
Twenty minutes after committing a comment that says this exact thing is what
entry 62 was about.

`#` is a comment on x86, PowerPC, SystemZ and LoongArch, and an immediate
prefix on AArch64. There is no convention to rely on, which is why the fix is
a per-target field rather than a smarter regular expression — the second time
this has been solved, and the first time it has been solved somewhere the next
person will see it before writing the parser rather than after.

**Fix**: `modelTarget.hashIsComment`, false for exactly one architecture, with
the reason at the field. The general point is that knowing a bug intimately is
not protection against writing it: the parser and the extractor were written
months apart by the same reasoning from the same wrong default, and the second
one had the first one's post-mortem open in another file.

## 51. The scan kernels were verified everywhere

**Wrong.** They were verified on amd64, purego, every amd64 tier, riscv64 and
loong64, and not on the three architectures that only `make test-cross`
reaches. On ppc64le, `FastCumProd` returned **zero for every element after the
first**, and it shipped that way.

It was found by the report tool written afterwards, on its first full run, in
the lane I had not re-run since adding the kernels.

The fault is the scan's identity vector reading as all-zeros, and the evidence
is clean:

- every scan built with a **nonzero** identity fails — `FastCumProd` on both
  float types and the exact integer `CumProd`;
- `FastCumSum`, built from the same macro with an identity of **0.0**, passes.

So `FastCumSum` on ppc64le is currently right for the wrong reason: it reads
zeros, and zero is what it wanted. The shuffle masks in the same pool are read
correctly — or the sum would fail too — which makes this a specific offset
fault rather than the pool being generally broken.

**Fix**: skipped on ppc64le, not repaired. The portable path stands in and is
correct, and the last generator change made on exactly this kind of reasoning
shipped an infinite loop the same night (entry 46). A second guess at the
constant-pool arithmetic, hours later, is not what this deserves.

Two things worth taking from it. **A test lane that is not run is not a test
lane**: `test-cross` is the only cover for arm64, s390x and ppc64le, it takes
three minutes, and I ran `test-gates`, `test-riscv64` and `test-loong64`
instead because those were the ones I had been debugging. And **an identity
element is a terrible canary**: a bug that zeroes it is invisible in every
operation whose identity is already zero, so the sum kernel would have gone on
passing for as long as anyone cared to look.

## 52. Casting a scalar to a vector type broadcasts it

**Wrong**, and it is the root cause of entry 51 rather than the constant-pool
theory recorded there.

```c
f32xS id = (f32xS)(1.0f);   // {1, 0, 0, 0, ...}, NOT {1, 1, 1, 1, ...}
```

A C cast to a vector type puts the scalar in lane 0 and zeroes the rest.
Measured on ppc64le: fifteen of sixteen lanes zero, at every optimisation
level from -O0 to -O3.

What makes it dangerous is that it is **target-dependent**. The same cast
broadcasts correctly on amd64, arm64, riscv64 and loong64, so it passes every
test on the development machine and is wrong only where nobody is looking.

Entry 51 blamed the constant pool, on the reasoning that `FastCumSum` passed
and `FastCumProd` failed while sharing a macro. That reasoning was sound and
the conclusion was not, because the same evidence fits both explanations: a
sum's identity is 0.0, so a splat that yields `{0,0,...}` is *correct by
accident*, and a product's is 1.0, so the same bug is total. What settled it
was three measurements, each cheap and each one I should have reached for
sooner:

1. Link clang's own object with `lld` and run it under qemu, with no
   generator involved: **same eleven wrong lanes**. That exonerated this
   repository's assembly entirely.
2. Compile the kernel at `-O0`: **still wrong**. Not an optimiser bug.
3. Test the primitives alone — the shuffle, then the cast. The shuffle was
   correct. The cast was not.

**Fix**: `SPLAT(VT, X)` in `csrc/goabi.h`, because a scalar operand in vector
*arithmetic* is broadcast even though a cast is not.

It has to be **subtraction**, which was the second bug in one macro. The first
attempt was `((VT){0}) + (X)`, and for `X = -0` that gives `+0`, because IEEE
says `(+0) + (-0)` is `+0` — so the running carry of a scan silently lost the
sign of its zero. The conformance suite caught it on amd64 within minutes,
`got -0 want 0`, which is the whole argument for keeping ±0 in the adversarial
generator. `(X) - ((VT){0})` is exact for every input.

Three more uses were latent in `bytes.c` — `(u8xB)set[s]` and
`(u8xB)(u8)0xFF` in the `IndexAny`, `CountAny` and `NotAny` family. None of
them ships on ppc64le today, for unrelated ABI reasons, so none had failed;
they were wrong and unreached. All four sites now go through `SPLAT`.

The lesson is not "read the spec more carefully". It is that when two kernels
built from one macro disagree, the difference between them is the evidence,
and the *cheapest* experiment that splits the hypotheses should come first. I
spent far longer on a constant-pool theory that a two-minute test of the cast
would have killed.

---

## 53. A `GLOBL` that declares more bytes than the `DATA` directives write

**Assumed.** If the constant pool emitter has a bug, the build will say so.
Plan 9 assembly is picky about almost everything else, and a pool that does not
add up sounds like the kind of thing an assembler refuses.

**True.** `GLOBL sym<>(SB), RODATA|NOPTR, $436` reserves 436 bytes and the
assembler zero-fills whatever no `DATA` directive covers. The emitter wrote
eight-byte directives in a loop bounded by `i+8 <= len(d)`, so a pool whose
length was not a multiple of eight silently lost its last four bytes. Nothing
failed to build. The constant that lived there simply read as `0.0f`.

46 of 223 ppc64le pools were truncated — every one whose length ended in a
`.rodata.cst4` section with an odd number of 4-byte constants. That number is
alarming and the honest one is smaller: only **two** kernels actually loaded
from the missing four bytes. The rest of the tails were constants LLVM had
emitted into a shared pool for a kernel that this target rejects for unrelated
ABI reasons, so they were dead. Both live ones were wrong:

| kernel | missing constant | what it did |
|---|---|---|
| `simd_quantize_u8` | `255.0f`, the upper clamp | read `0.0f`, so `min(z, 0)` clamped the whole scalar tail to zero |
| `simd_fast_hypot_f32` | `NaN` | returned `0` where it owed NaN |

**How it surfaced.** `TestQuantizeUint8` on ppc64le: `n=17 ... i=16 ... got 0
want 27`. One wrong element in seventeen, on one architecture.

What made this one quick, after entry 52 made the opposite mistake, was
splitting the hypothesis before theorising. Link clang's own object with `lld`,
run it under qemu, no generator involved: **zero wrong lanes**. That is the
inverse of the FastCumProd result and it says the generator is at fault rather
than the C. From there the diff is mechanical — clang emits 129 instructions,
the generator emits 129 `WORD`s, so look for a *changed* instruction rather
than a missing one. There were six changes, all constant-pool displacements,
all correct; and one of them pointed at `0x1b0` in a pool whose last `DATA`
line was `+0x1a8`.

**The generalisation is the check, not the fix.** The fix is four lines: step
down through the directive widths Plan 9 has instead of assuming eight divides
the length. The check is `TestPoolRenderCoversEveryByte`, which reads the
rendered directives back and reconstructs the pool the way the assembler
would, for every length from 1 to 40 and for the four sizes that occur here.
Verified by reverting the emitter and watching it fail — a test for a silent
bug is worth exactly nothing until you have seen it go red.

---

## 54. Two special-value tests, neither of which checked what the other did

**Assumed.** `simd_fast_hypot_f32` returning 0 instead of NaN would be caught,
because there is a test whose entire purpose is that a NaN comes back a NaN.

**True.** There were two such tests and between them they had three holes.
`TestFastSpecialValues` checked NaN, the infinities *and* the sign of a zero —
against `set.F64` only, reading its list of functions from `unaryCases()`.
`TestTranscendentalSpecialValues` covered the accurate tier and checked NaN and
the infinities but **not the sign of a zero at all**. So:

- no float32 kernel was ever checked, by either;
- `Pow`, `Atan2` and `Hypot` were checked by neither, being binary;
- `Asinh`, `Acosh`, `Atanh`, `Erf` and `Erfc` were checked by neither, not
  being in `unaryCases()`.

`fast_hypot_f32` was outside on both axes at once, which is why a constant that
read `0.0` instead of `NaN` shipped.

**What was actually broken.** Widening the sweep found four defects in
`csrc/math.c` on the first run, on every tier and both architectures, none of
them related to the pool bug and all of them in source that *both* tiers
compile — so the accurate kernels were wrong too:

```
Asinh(-0)   = +0     want -0
Erf(-0)     = +0     want -0
Erf(0)      = 1e-9   want 0
Pow(-0, 1)  = +0     want -0
Pow(-0, -1) = +Inf   want -Inf
```

Three of the five are one mistake: `x < 0 ? -v : v` to put the sign back on a
function computed from `|x|`. `-0.0 < 0.0` is false, so a negative zero comes
out positive. **This was already known and already written down** — the comment
on `cbrt_f64` explains the trap, names it, and even reasons about which other
functions are safe from it — but the reasoning was never applied to `asinh` and
`erf`, which have exactly the shape it describes. A comment is not a check.

`Erf(0) = 1e-9` is a different mistake: the Abramowitz and Stegun coefficients
sum to `0.999999999` rather than to 1, so `1 - tail` is not zero at the origin.
That is well inside the 1.5e-7 absolute bound the form promises, and still the
wrong answer for a value C99 F.10.5.1 specifies exactly. `pow(-0, y)` is the
third: the `x == 0.0` branch is selected by both zeros and never looked at the
sign bit.

**The fix that matters is structural.** The two tests are now one file,
`internal/conformance/special_test.go`, driven by a table keyed on the accurate
slot name with the Fast slot derived from it, run over both widths and both
arities. Alongside it `TestSpecialValueTable` walks `kernel.Ops` by reflection
and fails if any `Fast*` field is absent from that table or from a
`notPointwise` set that has to state where it is covered instead. Coverage is
enumerated rather than chosen, because choosing is what produced the holes.

---

## 55. Fusing two kernels made it slower

**Assumed.** Hamming distance is `popcount(a ^ b)`, both halves already exist
here as kernels, and fusing them into one pass has to win. Chaining costs a
destination buffer the size of the input and three passes over memory —
write the xor, read it back, read it again — where the fused form makes one.
The memory traffic argument is sound and it is the reason the operation was
put in the catalogue at all.

**True.** The first version was *slower than the chain it replaced*, and not
marginally:

| n | Xor + PopCount | fused, hand-written | fused, `COUNT_FOLD` |
|---|---|---|---|
| 64 | 8.9 ns | 8.5 | **6.5** |
| 1024 | 36.7 | 68.2 | **34.2** |
| 65536 | 2718 | 4292 | **2084** |

At n = 1024 the obvious fused loop was **1.76x slower** than not fusing.

**Why.** The loop was written the way the operation reads:

```c
isize s = 0;
for (isize i = 0; i < n; i++) s += __builtin_popcount((unsigned)(a[i] ^ b[i]));
```

A single 64-bit running total is a serial dependence with a widening on every
element. The vectorizer can only serve it by widening each byte to 64 bits and
doing a horizontal add per iteration, which is four times the vector work per
input byte and a reduction that cannot pipeline.

`simd_popcount` next door does not do that, and had not for a long time:
`COUNT_FOLD` keeps sixteen 32-bit lane accumulators and folds them once at the
end. The chain was beating the fused version because *half of the chain was
already using the better reduction* and the fused version had thrown it away.
Rewriting the same expression through `COUNT_FOLD` turned a 1.76x loss into a
1.07x-1.36x win, and the C is now shorter than what it replaced:

```c
void simd_hamming_u8(isize *out, const u8 *a, const u8 *b, isize n) {
  COUNT_FOLD(__builtin_popcount((unsigned)(a[p] ^ b[p])))
}
```

`COUNT_FOLD` and not `COUNT_BYTES`, incidentally: popcount of a byte is 0..8
rather than a predicate, so a byte-lane accumulator would wrap after 32 blocks
where that macro assumes 255.

**The lesson is about where the reasoning stopped.** Counting memory passes was
correct and it was not sufficient: it compared the two shapes and never asked
what the *accumulator* did in either. A fused kernel is not automatically
faster than an unfused one — it is faster only if it keeps everything the
unfused path was already doing right. The existing kernel beside it was the
specification for that, and reading it first would have skipped the whole
detour.

There is a gate here that did not fire, which is worth stating: nothing in this
repository fails a build for a kernel that is *correct but slower than the
composition it replaces*. `make perf-model` models one kernel, not two against
each other. The only thing that caught this was writing the benchmark against
the chain rather than against a scalar loop, which is the comparison a caller
would actually face.

---

## 56. The portable path is not portable if you write a multiply-add

**Assumed.** `internal/ref` is plain Go over plain slices. Whatever else varies
between architectures, *that* is the fixed point — it is the thing every kernel
is checked against, and the reason a `-tags purego` build is a meaningful
fallback. A one-line arithmetic expression in it cannot be
architecture-dependent.

**True.** Go's specification permits an implementation to fuse a floating-point
multiply and add into a single FMA, with one rounding instead of two. The gc
compiler does this on **riscv64, loong64, arm64 and ppc64**, and does not on
amd64. The kernels compile with `-ffp-contract=off` and never fuse anywhere.

So a reference function written the way the operation reads:

```go
dst[i] = (a[i]+shift)/denom*gamma[i] + beta[i]
```

agrees with its kernel on amd64 and disagrees on four of the six architectures
this library targets. Not by a rounding mode or an edge case — by one ULP on
ordinary inputs, `878.1303` where the kernel says `878.1304`.

**How it surfaced.** `make test-riscv64` and `make test-loong64`, both failing
`TestLayerNormInto` while amd64, arm64 and s390x passed, and while
`internal/conformance` passed *on the failing machines* — because that suite
compares the kernel against `ref` on one machine at a time, and here both sides
were the same wrong Go expression. What caught it was a test that compared
against an independently-computed expectation.

**The fix is one conversion**, and it was already in this repository:

```go
t := (a[i] + shift) / denom
dst[i] = T(t*gamma[i]) + beta[i]
```

An explicit conversion to `T` rounds to `T`, and a rounding the fusion would
have to discard is a fusion the compiler may not perform. `dotFloat` has taken
this precaution since it was written, `internal/ref/fusion_test.go` exists to
check it for `AddScaled`, `Lerp` and `Dot`, and the comment on rule 4 explains
why. None of that helped, because none of it is reached by *new* code.

So `LayerNorm` is now in `fusion_test.go` alongside the other three, and that
test was confirmed to fail against the unfixed version on riscv64 before being
kept — a guard nobody has watched go red is not a guard.

**What generalises.** Two things, and the second is the uncomfortable one.

The first: any new arithmetic in `internal/ref` containing a multiply feeding
an add needs the conversion. That is a rule, it is mechanical, and it now has a
test.

The second: **the differential suite cannot catch this class of bug at all.**
It compares kernel against reference on a single machine, so a defect that
moves both sides together is invisible to it, however many architectures it is
run on. Only a cross-architecture *comparison of results*, or a test with an
independently-derived expected value, sees it. Every emulated lane in this
repository is currently the former kind of check. That is worth knowing about
the verification, not just about this bug.
