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

Re-confirmed 2026-08-08 when the columnar work made prefix sums tempting
again: a standalone u64 log-shift scan on avx512 with the current clang,
against the serial loop, four million elements, minimum of seven --
1,559 against 1,389 us, 0.89x, outputs bit-identical. The one-cycle add
still leaves nothing to fill. The scan kernel stays unwritten.

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

---

## 57. The vector type is slower than the slice API it was meant to beat

**Assumed.** The one place this library loses is small inputs: below the
dispatch threshold a call into assembly costs about 1.4 ns and cannot be
inlined, so a plain Go loop wins. Go 1.26's `simd/archsimd` compiles to the
instruction with *no call at all*, so a vector type built on it should win that
band outright — and, being inline, should be at least competitive everywhere
else.

**True.** It wins one point of one curve. float32 addition, ns per call:

| n | plain Go | slice API | `MapFloat32x8` |
|---|---|---|---|
| 4 | **3.07** | 4.82 | 4.95 |
| 8 | 5.48 | 5.57 | **4.11** |
| 16 | 11.07 | **5.89** | 6.15 |
| 32 | 19.41 | **6.13** | 8.89 |
| 64 | 33.44 | **6.11** | 15.73 |
| 256 | 162.9 | **9.22** | 54.99 |

At n = 8 it is 1.3× the slice API. By n = 256 it is **six times slower**, and
the slice API has been winning since n = 16.

**Why**, separated by measurement rather than guessed:

| n = 256 | ns |
|---|---|
| `MapFloat32x8` (closure, 8 wide) | 55.0 |
| hand-written, no closure, 8 wide | 31.7 |
| hand-written, no closure, 16 wide | 26.4 |
| this library's generated kernel | 9.2 |

The closure is the larger cost — 23 of the 55 ns. **Go does not inline through
a function-value parameter**, so every block pays a call, which is precisely
the overhead the vector type was supposed to remove. Widening the block from 8
to 16 lanes buys another 5 ns.

And the residue is the interesting part: even the best hand-written `archsimd`
loop, no closure and full width, is **2.9× slower than the generated kernel**.
Intrinsics are not automatically as good as a compiled kernel. Clang unrolls,
schedules across iterations and hoists the bounds checks; a Go loop over
`archsimd` does one block per iteration with a slice expression each time.

**What this changes.** Nothing about the code — the helpers are correct and
they stay — but everything about what they claim. The doc comment now leads
with the table and says the niche exactly: *an expression the catalogue does
not have*, at any size, or *eight to sixteen elements*. For anything the
catalogue has, the slice API is faster and shorter. A doc comment claiming the
closure "is inlined by the compiler when it is a simple expression" was written
before the measurement and was simply wrong; it now says the opposite, with the
number.

**The lesson is the same one as entry 55, arriving from the other direction.**
There the mistake was counting memory passes and not asking what the
accumulator did. Here it was reasoning about call overhead and not asking what
the *rest* of the generated code does. Both times the argument was locally
correct and the conclusion was wrong, and both times ten minutes with a
benchmark that compared against the real alternative would have said so
immediately.

---

## 58. internal/fastpath: measured, and rejected

**Assumed.** Every generated kernel is guarded — below a per-operation element
count the guard calls the portable scalar loop, because a Go-to-assembly call
costs a fixed ~1.4 ns and cannot be inlined. Go 1.26's intrinsics have no call
at all, so routing that fallback through `simd/archsimd` instead of
`internal/ref` should win the whole band for free. The plan estimated 3.74 ns
against 5.61 at n = 16.

**True.** The package was built, made bit-identical to `ref` over an
adversarial corpus, benchmarked against `ref`, and was **faster**:

| n | `ref` | `fastpath` |
|---|---|---|
| 4 | 2.96 | 2.74 |
| 8 | 5.56 | 4.98 |
| 16 | 10.82 | 6.42 |
| 32 | 19.42 | 9.39 |

Then it was wired into the guards and A/B'd against the previous tree through
the **public API**, which is the only path that exists in a real program:

| n | before | after |
|---|---|---|
| 4 | 4.50 | 5.49 |
| 8 | 5.50 | **8.04** |
| 16 | 5.49 | 5.44 |
| 256 | 9.05 | 9.47 |

A 46% regression at n = 8, in the band the package existed to speed up.

**Why the package's own benchmark was wrong.** It called
`fastpath.AddFloat32(dst, x, y)` with a concrete `[]float32`, which the
compiler inlines and specialises. The generated guard cannot: `ref.Add` is a
generic wrapper around a simple loop that inlines *into* the guard, while any
fastpath function is a real call — a length check, a block loop and a method
expression — that does not. At n = 8 the total is about 5 ns, so one
non-inlined call is most of it.

The first version was worse still. It mirrored `ref`'s generic signatures with
a type switch on `any(dst)`, which boxes the slice header on every call: 8.73
ns at n = 8. Making the functions concrete and per-type recovered 0.7 ns of
that and left the rest.

**And the band does not exist anyway.** The fallback runs only below the
threshold, which is 16 for elementwise operations. At 16 and above the
*generated kernel* runs, and it beats archsimd there — 5.44 against 6.42 —
because clang unrolls and schedules across iterations. So the interval where an
archsimd fallback could help is n < 16, and that is exactly where a
non-inlined call costs more than the vector work saves. The two windows do not
overlap.

**Rejected**, on the same footing as CRC32 and Adler32 in task #57: measured,
lost, recorded rather than implied. The package is deleted rather than left
unwired, because dead code rots and the knowledge belongs here.

**The lesson is about which benchmark to believe**, and it is the third time in
this file. Entry 55: counted memory passes, did not ask what the accumulator
did. Entry 57: reasoned about call overhead, did not ask what the rest of the
generated code did. Here: benchmarked the component in isolation, did not ask
what the *call site* looks like. Each time the local argument was correct and
the conclusion was wrong, and each time the fix was to measure the thing a
caller actually runs — which for a library means through the exported function,
against the tree as it was before.

---

## 59. The 338 vectorization refusals are not one problem, and forcing them is a trap

**Assumed.** There are zero `#pragma clang loop` directives in this repository
and several hundred kernels that LLVM declines to vectorize. Some fraction of
those must be the cost model being conservative, and `vectorize(enable)` would
recover them.

**True.** The refusals sort into four kinds, and only the last is even a
candidate. Counted by kernel rather than estimated:

| kernels refusing | why |
|---|---|
| `scatter` 65, `gather` 31, `reverse` 30, `transpose` 8, `tile` 10 | the instruction does not exist on that target |
| `dot`, `mul3`, `mul4`, `prod` — the `i64`/`u64` instantiations | no 64×64 vector multiply on NEON, VSX or z/Arch |
| `abs_u8`, `abs_u16`, `abs_u32`, `abs_u64` | absolute value of an *unsigned* type is the identity; there is nothing to vectorize |
| the rest | `the cost-model indicates that vectorization is not beneficial` |

The plan for this work guessed "~60 scatter, ~30 reverse, 114 float/fast_* as
the plausible wins". The first two were right and the third was not: the large
groups are all missing-instruction cases, and `gather` — 31 kernels, the second
largest — was not in the estimate at all.

**The trap, demonstrated rather than predicted.** `simd_dot_i64` on NEON, which
refuses because ARMv8-A has no 64-bit vector multiply. Compiled twice, once
with `#pragma clang loop vectorize(enable)`:

| | vector instructions | total instructions |
|---|---|---|
| without | 0 | 36 |
| with | 6 | **47** |

Forcing worked, in the sense that vector instructions appeared. Here is what
they are:

```asm
mul   x14, x15, x14      ; scalar 64-bit multiply
mul   x13, x13, x15      ; scalar
fmov  d2, x14            ; move the scalar result into a vector lane
mov   v2.d[1], x12       ; insert the other lane
add   v0.2d, v2.2d, v0.2d
```

The multiplies are still scalar. The `fmov` and `mov v.d[1]` exist only to
reassemble the scalar results into a vector register so that the *add* can be
one. Eleven extra instructions to make a loop look vectorized.

**And this repository's gate would pass it.** `package verify` checks that a
kernel contains vector instructions, which this does — `add v0.2d` is as vector
as anything. The check exists to catch a kernel that failed to vectorize at
all, and it cannot tell that apart from one that vectorized into scalar work
plus packing.

**Not applied.** Nothing is gained on the three categories where the
instruction is missing, and on those the pragma actively produces something
slower that the existing gate would wave through. The cost-model group is the
only place worth trying, it is the smallest, and each candidate needs both
`make perf-model` and a real benchmark before it ships — which is the same bar
`internal/fastpath` failed in entry 58.

The useful output of this investigation is the categorisation, not a change:
**150 of the refusals are a missing instruction and will still be refused on
the day someone tries again**, so the number to quote is not 338.

---

## 60. An exhaustive test of the kernel says nothing about the reference

**Assumed.** fp8 has 256 encodings. A test that widens all 256, narrows them
back, and checks every one returns its own encoding is not a sample — it is a
proof. Nothing else is needed.

**True.** That test passed while `internal/ref` had **three** separate defects,
and it passed *because* it was thorough: 256 elements is well over the
threshold, so every call went to the generated kernel and the reference was
never executed. The exhaustive test proved the kernel right and said nothing
about the thing the kernel is checked against.

The conformance differential found all three in three runs:

| | kernel | ref |
|---|---|---|
| `F32ToF8E4M3(0)` | `0x00` | `0x30`, which is 0.5 |
| `F32ToF8E5M2(NaN)` | `0x7f` | `0x7d` |
| `F32ToF8E5M2(57345)` | `0x7b` | infinity |

The first is the sharpest. `math.Frexp(0)` returns `(0, 0)`, so zero fell
through to the general path, which synthesised an exponent field of `bias-1`
and produced 0.5. Zero is the one value every float format agrees on and the
easiest to lose in a path written for the general case.

The third is a real specification error rather than a slip. `ref` overflowed to
infinity above the largest finite value, 57344. Round to nearest does not: the
threshold is the *midpoint* between 57344 and the next power of two, 61440, so
57345 rounds back down to 57344. Writing "overflow above the maximum" is the
obvious thing and it is wrong by 4096.

**And the kernel was right all three times**, which is worth saying because the
usual assumption runs the other way. The C was written from the bit layout; the
reference was written through `math.Frexp` and `math.Ldexp` deliberately, so
that the two could not share a derivation. That is what made the differential
informative — and it is also why the reference, being the less mechanical of
the two, is where the mistakes were.

**The lesson is which direction a test points.** A test through the public API
exercises whatever the dispatcher selected, which on a large input is never the
reference. To test the reference you have to call it, and the only thing that
does that systematically is the differential — or a `-tags purego` run, which
is why that lane exists and why it is not optional.

---

## 61. The sse2 alignment skips: the objection on record is right about the wrong thing

**Assumed** (by the comment on `checkPatchable`). The 170 sse2 kernels that
refuse because a legacy SSE instruction needs a 16-byte-aligned memory operand
cannot be recovered, because the tempting fix is to pad the appended pool to a
16-byte offset and rely on the linker aligning every text symbol to 32 —
which is not a guarantee, since `cmd/link` takes `-funcalign`, and a consumer
building with `-ldflags=-funcalign=8` would get a SIGSEGV out of this library
with no plausible way to trace it.

**That reasoning is correct, and it is about the wrong strategy.** It applies
to appending the pool *inside the TEXT symbol*, which is what the amd64 path
does today. It does not apply to a separate `DATA`/`GLOBL` symbol, which is
what the ppc64le path already emits — and the linker aligns those by size, not
by `funcAlign`. Measured:

```
GLOBL simdpool<>(SB), RODATA|NOPTR, $32

  default            pool address 0x4d0880, %16 = 0, %32 = 0
  -funcalign=8       pool address 0x4ca880, %16 = 0, %32 = 0
```

Still 16-byte aligned under the exact flag the comment is about, because
`-funcalign` aligns **text**, not data.

**So the path is open, and it is not short.** Three things have to hold, and
only the first is established:

1. *The pool is aligned.* Yes, as above.
2. *The reference can be expressed.* On amd64 the pool is reached RIP-relative,
   and a raw instruction encoding cannot carry a relocation. The instruction
   would have to be emitted as a Plan 9 mnemonic — `MULPS simdpool<>+0x20(SB),
   X3` — so the assembler emits the relocation and the linker fills the
   displacement. Go's assembler does know these mnemonics, and ppc64le already
   proves the mixed approach works: it emits `MOVD $simdconst<>(SB), R2` as a
   mnemonic beside a body of `WORD`s.
3. *The lengths must match exactly.* This is the risk. The body is raw
   encodings with branch displacements already computed, so if Go's assembler
   encodes `MULPS sym(SB), X3` in a different number of bytes than clang's
   `mulps 0x1234(%rip), %xmm3` — a different REX choice, a different prefix
   order — every branch after it is wrong, silently. Both *should* be
   `0F 59` + ModRM(mod=00, rm=101) + disp32, but "should" is what this file
   exists to disbelieve, and it has to be verified per mnemonic across the
   dozen or so that occur.

**Not attempted beyond this.** What is recorded is that the reason on file for
not trying is answering a different question than the one that matters, and
that step 3 is the actual work: an encoding-equivalence check per mnemonic,
with the branch displacements re-verified, before a single kernel changes. The
prize is real — 170 kernels, the largest addressable bucket in the repository
and the whole reason the baseline x86-64 tier is thin — and so is the failure
mode, which is a silently wrong branch target rather than a build error.

**Superseded by entry 67, which did it.** Step 3 held for every mnemonic that
occurs, and the baseline tier went from 673 kernels to 789.

---

## 62. A streaming sum that overwrites its accumulators is wrong by one bit

**Assumed.** A resumable sum needs to carry the sixteen lane accumulators
across a chunk boundary, because element i must land in lane i%16 whatever the
chunking. Get the lane bookkeeping right — a head to finish the partly-filled
block, whole blocks through a kernel, a tail — and the answer must match the
whole-slice reduction.

**True, and one step short.** The lane bookkeeping was right and the answer
still differed in the last bit. The kernel wrote its partial sums into `dst`
rather than adding to them, so the streaming form computed

	(chunk1 lane j) + (chunk2 lane j)

where the whole-slice reduction computes one running sum per lane. That is
`(a+b)+(c+d)` against `((a+b)+c)+d`, floating-point addition is not
associative, and the difference is exactly the thing the API promises does not
happen: an answer that depends on where the chunks fell.

The fix is one line — seed the accumulator from `dst` instead of from zero —
and it has to be in the kernel *and* the reference, or the differential catches
it instead of the caller.

**What caught it is the part worth keeping.** Every uniform chunking passed:
1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 63, 100, 4096. So did every size of
input. The failure needed *ragged* chunks — sizes drawn from 1 to 40, none a
multiple of sixteen, so that the per-chunk partials landed on lanes in a
pattern that never repeated. One test, and without it this ships.

And it had to be a **bit-equality** test. The discrepancy was one ULP on a
sum of five thousand terms. Any tolerance anyone would write passes it, and the
library would have had a streaming API whose result changed with the reader's
buffer size — silently, and only for some inputs.

**Two smaller things on the way**, both from the generated guard rather than
the arithmetic:

The guard clamped the input to the output. `dst` here is a fixed sixteen-lane
window, not an array parallel to `a`, and the emitter's default is
`n := min(len(dst), len(a))` — so a 4096-element chunk summed its first
sixteen elements. Declaring two lengths in `CArgs` turns the clamping off; the
mechanism already existed for `Diff`, whose output is one element shorter than
its input.

And a local `[16]T` handed to a kernel through a function pointer escapes, so
`Add` allocated once per call. Accumulating straight into the struct's array
removes both the allocation and the extra add.

---

## 63. RVV refuses the transcendentals on the cost model, and will not be talked out of it

**Assumed.** Entry 59 sorted the vectorization refusals and found that the only
category worth trying is the cost-model one — where LLVM *can* vectorize and
decides not to. The accurate transcendentals on RVV are exactly that: 24
refusals, and asking clang why gives

	the cost-model indicates that vectorization is not beneficial

every time, with no mention of a missing instruction. RVV has every float
operation these need. So `#pragma clang loop vectorize(enable)` should recover
them, and 24 kernels is worth a one-line macro change.

**True.** Adding the pragma to `UNARY_MATH` and recompiling for RVV produces
**22 instances** of

	loop not vectorized: the optimizer was unable to perform the requested
	transformation; the transformation might be disabled or specified as part
	of an unsupported transformation ordering

`vectorize(enable)` is a request, not an instruction. Where entry 59's
`dot_i64` case took the request and produced scalarised vector code — the bad
outcome — these decline it outright. The cost model is not the only thing
saying no; something later in the pipeline is, and the pragma has no answer for
it.

So the two failure modes of forcing are now both on record, and neither is
"it works":

| | what forcing does |
|---|---|
| missing instruction (`dot_i64`, NEON) | emits scalar work plus lane packing, 36 → 47 instructions, and the repo's vector-instruction gate passes it |
| cost model then a later pass (transcendentals, RVV) | refuses, with a warning |

**What is left is what the task actually asked for: intrinsics.** And the
constraint that makes it a project rather than an afternoon is not the
arithmetic, it is rule 6's neighbour — the *accurate* transcendentals promise
the same bits on every tier. A hand-written RVV version has to reproduce
clang's exact evaluation order for the polynomial and the reduction, at a
vector length that is not known until run time. That is achievable and it is a
different kind of work from everything else in this repository, which gets its
cross-tier identity for free by compiling one source everywhere.

Not attempted. Recorded so the next person does not spend the afternoon on the
pragma first.

## 64. A vectorized sliding window loses to a deque, and the crossover is on the window

**The belief.** `RollingMin` over a window of *w* is w-1 elementwise minima,
and elementwise minima are what this library is fastest at, so the vectorized
version wins the way every other elementwise family does.

**Half true, and the half that is false is the half people ask for.**

The textbook sliding-window minimum is a monotonic deque: two amortized
comparisons per element, *independent of w*. The kernel here does w-1 passes
over the output — asymptotically worse — but each pass is contiguous and
vectorized, and they are tiled so the block being accumulated stays in L1
across all of them. So the kernel does sixteen windows per instruction where
the deque does one, and which wins is a race between a constant factor of the
vector width and a linear factor of w.

Measured, one million float64, minimum of three runs:

| window | kernel | deque | |
|---|---|---|---|
| 4 | 0.65 ms | 8.35 ms | **12.8x** |
| 8 | 1.35 | 8.90 | 6.6x |
| 16 | 2.79 | 8.62 | 3.1x |
| 32 | 5.65 | 8.44 | 1.5x |
| 64 | 11.2 | 8.33 | **0.75x** |
| 256 | 44.7 | 8.21 | **0.18x** |

The deque's column barely moves, which is the whole point: it does not care
about w. The crossover is just above 32 — four times the eight float64 lanes an
AVX-512 register holds — and past it the kernel loses without limit.

**Why the library ships the kernel anyway and does not switch.** Three reasons,
in the order they were considered and the order they matter.

The O(n log w) sparse-table formulation removes the problem entirely: doubling
the window each pass needs `ceil(log2 w) + 1` passes instead of w-1, nine
rather than 255 at w=256. It was not taken because the doubling table needs
`n - 2^j + 1` entries at step j, which is longer than the `n - w + 1` outputs
`dst` holds. It needs scratch the size of the input, and this library allocates
nothing. Requiring the caller to pass a scratch slice was rejected as a worse
API than a documented limit.

The deque itself was written and deleted. It needs an index ring proportional
to the window, which would have made this the only allocating operation in the
library. And it is subtle in a way that testing does not catch: IEEE minimum
propagates NaN, and "pop the back while it is worse than the new element" does
nothing when neither operand orders — a plain deque holds the NaN, never
reports it, and returns a neighbouring value instead. Correcting that needs a
separate scan tracking the *leftmost* NaN in the window, because a chain of
IEEE minima yields that NaN's payload rather than any NaN's. That is a third
implementation of the same semantics, in a library whose central promise is
that every implementation of an operation agrees bit for bit.

So the limit is documented on `RollingMinInto` with the table above, and the
answer to "what if my window is 256" is: write the deque, it is fifteen lines.
A library that shipped only this path and called itself SIMD would be five
times slower than a hand-written loop while implying otherwise, which is what
`testdata/bench/README.md` exists to prevent.

**What is on record.** The vector formulation, measured, with the window where
it stops paying. Not "rolling minimum is not vectorizable" — it is, and below a
window of 48 it is twelve times faster.

## 65. XXH64 vectorizes, and on the tier most machines run it is slower

**The belief.** XXH64's bulk loop has four independent accumulators, which is
exactly the shape a vector unit wants. Four lanes of one register instead of
four scalar chains.

**LLVM agrees, does it, and it does not help.**

The stripe loop itself cannot vectorize — each stripe's accumulators depend on
the last, which is a real serial dependence — so the only question is whether
SLP turns the four independent *lanes* into vector operations. It does, from
`-march=x86-64-v3` up. What comes out per 32-byte stripe:

| target | multiplies emitted | |
|---|---|---|
| `x86-64` | 8 `imulq` | no vectorization, the four chains interleave |
| `x86-64-v3` | 6 `vpmuludq` | there is no 64-bit vector multiply below AVX-512DQ, so each one costs three |
| `x86-64-v4` | 2 `vpmullq` | native, plus one `vprolq` for the rotate |

llvm-mca, `-mcpu=znver5`, cycles per stripe:

| target | cycles/stripe | against scalar |
|---|---|---|
| `x86-64` | 8.0 | — |
| `x86-64-v3` | 10.5 | **0.76x** |
| `x86-64-v4` | 6.0 | 1.33x |

So AVX2 — the tier the overwhelming majority of x86 machines select — is
**slower than not vectorizing at all**, and for the reason `csrc/scan.c`
already records for the int64 product: the 64-bit multiply has to be emulated
from three 32-bit ones. This is entry 59's trap arriving by a different route.
Nobody wrote a pragma; the vectorizer did it unprompted and the result would
have passed the has-vector-instructions gate.

And the 1.33x on AVX-512 is not the win it looks like either. 32 bytes per 8
cycles is 4 bytes per cycle, which at this machine's clock is already past what
DRAM delivers — the gain exists only while the input is in L1 or L2, and a hash
is usually called on data that is not.

**What would be worth building instead.** XXH3, not XXH64. It was designed for
SIMD after the fact that XXH64 was not: eight accumulators rather than four, and
a mixing step built from 32x32→64 multiplies, which is `vpmuludq` — one
instruction on SSE2 and everything after it, no emulation. That is a genuine
project with an exacting specification, and it is not this one.

**Recorded, not attempted.** The measurement is the point: XXH64 is not a
missing operation, it is a measured loss on the tier that matters.

## 66. Edit distance is not a loop, and the fast version is not SIMD

**The belief.** Levenshtein distance is a dynamic program over a table, and a
table is a loop nest, so it vectorizes.

**Two things are true and they point away from each other.**

The DP does vectorize, along the anti-diagonal: every cell on a diagonal depends
only on the two diagonals before it, so a diagonal is independent. That is a
real formulation. It also needs scratch proportional to the shorter input, and a
stride that is not one — the diagonal walks the table at a slant, so every load
is a gather or a shuffle. This library allocates nothing and has spent entry 59
establishing what gather-dependent kernels cost. It would be a rewrite rather
than a loop, and the payoff is against a DP that is already O(nm).

The fast implementation of edit distance is not the vectorized DP at all. It is
Myers' bit-parallel algorithm, which packs the DP's state into machine words and
computes 64 cells per word operation — O(nm/64), so roughly sixty-four times
faster than the DP the vector version is trying to speed up by eight. It uses
addition, and, or, xor and shift, all scalar, and a vector version of it needs a
carry to propagate between lanes, which is exactly what vector adds do not do.

So the honest position is: the operation people want is Myers, Myers is scalar,
and a vectorized DP would be a slower answer with more machinery. The place SIMD
genuinely helps is *many* comparisons at once — one pattern against a batch of
candidates, one bit-parallel state per lane — which is a different API and a
different task.

**Recorded, not attempted**, and recorded specifically so the next person does
not vectorize the DP and then discover Myers.

## 67. The sse2 alignment skips, closed: 673 kernels to 789

Entry 61 ended "not attempted beyond this", having established one of three
things the fix needed. All three now hold, and the baseline x86-64 tier went
from **673 kernels to 789**, with **no alignment refusals left at all**.

**The three preconditions.**

1. *A separate RODATA symbol is 16-byte aligned.* Measured in entry 61, and it
   survives `-ldflags=-funcalign=8` because `-funcalign` aligns text, not data.

2. *Go's assembler can express the reference.* It can:
   `MULPS simdpool<>+0x20(SB), X3` assembles, and the linker fills the
   displacement.

3. *The replacement is exactly as long as what it replaces.* This was the open
   one, and it is the whole safety argument — a replacement of a different
   length moves every PC-relative branch after it, silently. Every mnemonic
   that occurs was assembled both ways and compared:

   ```
   Go     MULPS simdpool<>+0x20(SB), X3   ->  0f 59 1d <disp32>
   clang  mulps 0x1234(%rip), %xmm3       ->  0f 59 1d <disp32>
   ```

   Byte-identical, including the mandatory `66` prefix and the REX byte that
   `xmm8`-`xmm15` require, and in the same order.
   `TestRespelledEncodingsMatchClang` now does this for all forty entries
   against both register ranges, on every run, so the table cannot rot.

**What it took, and three ways to be wrong that the first version was.**

The mnemonic table is not the interesting part. These were:

*Sections concatenated without padding.* A kernel's pool is often several ELF
sections, and appending the second straight after the first put a
16-byte-aligned constant at `pool+0xa9f`. `fastSigmoidFloat64SSE2` segfaulted
the moment the differential suite ran it. **Caught by the conformance
differential**, which is what it is for.

*A trailing immediate changes the addend.* A PC-relative displacement is
measured from the end of the *whole* instruction, so `cmpnltps` — which carries
a one-byte predicate after the displacement — is emitted with an addend of −5,
not −4. `objfile` adds back a fixed +4, leaving the target offset one byte
short. One byte short of a 16-byte boundary is exactly the misalignment these
instructions fault on, so this looked identical to the first bug and was not.
**Caught by an assertion added after the first bug** — that the computed pool
offset is 16-byte aligned — which turned a segfault into a build error naming
the instruction.

*Not everything that reads a pool needs alignment.* `cmpltss` and `movd` went
into the re-spelling table because they were being refused, but they read four
bytes and have no 128-bit alignment requirement at all. They belong in
`unalignedLoads` with the other scalar forms, and demanding a 16-byte-aligned
offset from them failed the assertion for two kernels. **Caught by the same
assertion**, which is the argument for adding assertions that state what you
believe rather than only what you fear.

**A note on the dead code this replaced.** `emit/respell.go` had implemented
exactly this idea before, was found to break branch displacements, and was left
in the tree — unreferenced, with a table of load forms and a doc comment
describing a mechanism nothing used. It is deleted. The distinction it never
made is the one that matters: the technique is sound, and what makes it sound
is the length check, not the mnemonic table.

**What is left on this tier.** Nothing to do with alignment. The remaining sse2
gaps are the ones every tier has — LLVM declining to vectorize a kernel at all.

## 68. The pragma does work on NEON, does nothing on RVV, and the control is what said so

Entry 63 recorded that `#pragma clang loop vectorize(enable)` on the
transcendentals produced **22 instances** of "unable to perform the requested
transformation", and concluded that intrinsics were the only route left.

**Half of that is now wrong and half of it is still right.**

The error message does not reproduce on clang 22.1.8. There are none. What the
vectorizer says instead, for thirteen functions at both widths, is:

	remark: the cost-model indicates that vectorization is not beneficial

which is a different claim entirely — not *cannot*, but *will not*. And
`vectorize(enable)` exists precisely to overrule a cost model.

**On NEON and SVE2 it overrules it, and the result is real.** Five float64
functions — acosh, atanh, cbrt, log1p, sinh — plus their five `Fast` twins, on
two tiers: **20 kernels**, arm64 going from 1594 to 1614. Checked against the
trap entry 59 describes rather than assumed:

| check | result |
|---|---|
| genuine vector code, not scalarised | 43-87 vector-lane ops per body, **zero** per-lane extract/insert |
| results unmoved | the differential and the ULP suite pass on arm64 |
| worth doing | llvm-mca, neoverse-v2, `cbrt_f64`: **131 cycles/element scalar, 16 vectorized** |

There is nothing for the trap to catch here, and the reason is worth stating:
these bodies are an inlined polynomial applied elementwise, so every operation
in them is plain floating-point arithmetic that every target has. Entry 59's
failure mode needs a *missing* instruction — scatter, or the 64-bit vector
multiply — and there is not one.

**On RVV it changes nothing at all, and I nearly reported otherwise.**

The first measurement was: compile with the pragma, count the remarks, find 26
loops vectorized, conclude the pragma had unlocked them. That number is
correct and the conclusion was not, because there was no control. Compiling the
*unmodified* source with the identical command gives:

	without the pragma  26 vectorized loops
	with the pragma     26 vectorized loops

The 26 are the ones that already vectorized. The generator's kernel count for
riscv64 is unchanged at 801, which is the same fact arriving through a
different door — and had the count been the only thing looked at, the wrong
conclusion would have been caught anyway. It was the missing control that made
a wrong answer *plausible*, which is the more dangerous state.

So entry 63's conclusion survives for RVV in substance while being wrong in its
evidence: the transcendentals there still need hand-written intrinsics, and the
reason is not the message that entry recorded.

**What is left, and what it does *not* need.** Thirteen accurate
transcendentals at two widths on RVV — 26 kernels: acosh, asinh, atanh, cbrt,
expm1, log, log2, log10, log1p, pow, sinh, tanh.

It was written here first that these need generator plumbing that does not
exist, on the grounds that the pipeline compiles one portable C source per
kernel and has no concept of a per-target intrinsic source. **That is wrong**,
and it is wrong in the direction that makes work look impossible rather than
merely large. The pipeline already branches per target inside a single file —
`csrc/compress.c` picks its lane count with `#if defined(__riscv_v)` — and
intrinsics work the same way. Compiled with the generator's exact command:

```c
#if defined(__riscv_v)
#include <riscv_vector.h>
  size_t vl = __riscv_vsetvl_e32m1(n - i);
  vfloat32m1_t v = __riscv_vle32_v_f32m1(a + i, vl);
#endif
```

assembles to real `vsetvli`, `vle32`, `vfmul` and `vse32`. No plumbing, no new
flags, no separate source.

So the remaining cost is entirely the numerics: each function is its polynomial
written against RVV's length-agnostic intrinsics, holding the 1.0 ULP bound the
accurate tier promises. Large, and with nothing in the way.

Not a correctness hole either way; an unbuilt kernel is not registered and the
portable Go implementation runs.

## 69. The RVV transcendentals were never a cost-model problem. One intrinsic cost them all

Entry 63 concluded these needed hand-written intrinsics. Entry 68 upheld that
for RVV while correcting its evidence. Both were looking at the wrong thing,
and the actual cause takes two lines to fix.

**What the vectorizer was really saying.** The remarks visible when compiling
the whole of `csrc/math.c` are about the cost model, which is what sent two
entries down that path. Compile **one function** instead — the same file with
`MATH_SET` reduced to a single `UNARY_MATH(log, ...)` — and the real message
appears:

	Recipe with invalid costs prevented vectorization at
	VF=(vscale x 1, vscale x 2, vscale x 4): call to llvm.is.fpclass

Not a cost model declining a bad trade. RVV has **no cost at all** for
`llvm.is.fpclass`, so the vectorizer cannot price the recipe and abandons the
loop. No pragma overrules that, which is exactly why entry 68's pragma
experiment moved nothing on RVV — the right answer for the wrong reason.

**Where the intrinsic came from.** From the denormal test, in `log2_frac_f64`
and `cbrt_f64` and their float32 twins:

```c
int sub = x < 0x1p-1022 && x > 0.0;
```

Clang recognises "positive, nonzero, and below the smallest normal" as a class
query and emits `llvm.is.fpclass(x, 128)` — class 128 being *positive
subnormal*. It is a good canonicalisation everywhere except a target that
cannot cost it.

**The fix.** Ask the same question of the bits, where it is an ordinary
unsigned comparison:

```c
unsigned long long ub = f64_to_bits(x);
int sub = ub > 0ull && ub < 0x0010000000000000ull;
```

Exactly equivalent, and it can be checked case by case rather than trusted:
`+0` gives bits 0 and fails the first test; `-0` and every negative gives bits
at or above the sign bit and fails the second; NaN and infinity sit above
`0x7ff0...` and fail it too; a positive subnormal is precisely a nonzero
pattern with a zero exponent field, which is the interval named.

**What it bought.** riscv64 **801 kernels to 849**, and *no accurate
transcendental refuses on any target any more* — log, log2, log10, log1p,
expm1, sinh, tanh, pow, asinh, acosh, atanh and cbrt all vectorize on RVV.
The lane that runs them is `make test-riscv64`, which boots qemu with
`v=true`; `make test-gates` deliberately runs riscv64 *without* the extension
and would have reported PASS while executing none of this.

**What it cost.** Two ppc64le kernels, `pow_f32` and `fast_pow_f32`. The new
integer sequence gave the register allocator a different shape and it chose
`r0`, which Go's ppc64le ABI defines as constant zero — so the existing check
dropped them, exactly as it should. 592 kernels to 590. Net across all
targets: 6,623 to 6,669.

**The lesson worth keeping.** Two entries concluded "this needs intrinsics"
from remarks emitted while compiling a 1,000-line file, where every kernel
reports against the same macro line and the interesting message is buried under
a hundred repetitions of an uninteresting one. Isolating a single function cost
about two minutes and produced a different answer. A diagnostic that cannot
name which function it is talking about is not evidence about any of them.

## 70. Forcing vectorization everywhere segfaulted ppc64le, and the kernel count said nothing

Entry 68 applied `#pragma clang loop vectorize(enable)` to the transcendental
macros and reported +20 arm64 kernels, checked against entry 59's trap. The
checks it ran were the right ones. It ran them **on arm64 only**, and the
pragma was applied to every target.

**What happened.** ppc64le's conformance suite died with

	signal: segmentation fault (core dumped)

and it died in `TestFastTranscendentalULP`, which **passes when run on its
own**. That combination — a fault that only appears after other tests have run,
in a test that is fine in isolation — is the signature this repository already
documents for a kernel that corrupts state and returns: the damage lands
somewhere else entirely, later.

**Why nothing caught it earlier.** Three reasons, and each is worth keeping.

The kernel count did not move. ppc64le reported 592 kernels before the pragma
and 592 after, so the emission summary — which is the first thing anyone reads
after a generator change — was identical. Forcing did not add or drop kernels
there; it changed the code inside them.

The verification that did run was on the target that benefited. arm64 gained
kernels, so arm64 was where the vector-instruction count, the extract/insert
check and the ULP suite were pointed. A target that gains nothing from a change
still receives it.

And the cross lane that would have caught it was killed by a tool timeout
mid-ppc64le on the run that mattered, then reported green on an earlier tree.
A lane interrupted is not a lane passed, and the log looked the same either way
until the platform lines were counted.

**The fix is what entry 59 actually asked for.** Not "do not force" — force per
target, and verify per target. `FORCE_VECTORIZE` is now defined only for
`__aarch64__` and `__riscv_v`, the two where the gain was measured and the
tests were run:

	arm64    +20 kernels, 1594 -> 1614
	riscv64   +4 kernels, 845 -> 849, but only once entry 69's
	          is.fpclass fix removed the real blocker
	ppc64le   no change, and no segfault

**What to take from it.** An emission summary that does not move is not evidence
that nothing moved. The generator reports how many kernels a target got, not
what is in them, and forcing changes the second without touching the first. The
only thing that distinguishes those two states is running the tests on that
target.

## 71. Packing B before a parallel matrix multiply, which loses at every size

**Assumed.** Packing and parallelism win in the same regime, so combining them
should compound. Packing rearranges B into a layout the kernel can stream, and
`MatMulIntoScratch` already switches to it once B stops fitting in cache
(entry: `gemmPackCliff`, 1 MiB). Parallelism wins at large sizes for the same
reason — there is enough work to hide the overhead. B is read by every worker,
so packing it once and sharing it looked like the obvious shape, and
`MatMulParallelIntoScratch` was written and tested before it was measured.

**True.** It is slower than the unpacked parallel path at every size tried,
float64 square, 32-thread Zen 5, microseconds:

	n      parallel   parallel+packed
	128          40               128
	256          73               129
	512         483               594
	1024       3173              3586

Not marginally, and not only where packing was expected to lose. At n=1024,
where B is 8 MiB and far past the cliff that makes packing pay in the serial
kernel, packing still costs 13%.

**Why.** The cliff is about one core's cache. Splitting by output row gives each
worker its own slice of A and of the destination while B is shared, so the
working set per core is already smaller than the serial kernel's, and the part
that was thrashing is the part that got divided. Packing then buys reuse that
the row split has already bought, and charges a full pass over B plus the
allocation to hold it.

**What was done.** `MatMulParallelIntoScratch` was deleted rather than shipped
with a caveat. An exported function that is slower than the one beside it at
every measured size is a trap with documentation attached, and the next person
to find it would have to redo this measurement to know not to use it.
`MatMulParallelInto` is the whole parallel API.

**What to take from it.** Two optimisations that win for the same stated reason
can be the same optimisation. Packing and row-splitting both exist to keep the
working set in cache, so applying both does not halve the misses twice — the
first one to run takes the win and the second one only pays. Worth measuring
the combination rather than reasoning about it, which is cheaper than it sounds:
this took one benchmark and would have shipped a slower function otherwise.

## 72. Accelerating ger to speed up an LU, which it does not

**Assumed.** gonum's LU was no faster with an accelerated BLAS because the
routines it spends its time in were not accelerated. Counting BLAS calls across
`mat` and `lapack/gonum` put `Dger` high on the list, and LAPACK's `Dgetf2` is
visibly built out of it. So a `ger` kernel should move a factorisation.

A kernel was written, and it works: `RankOneInto` is 8.2x the portable path at
n=64 and 2.4x at n=2048. LU did not move at all — same wall-clock as stock
gonum at 256 and 512 square, both directions inside noise across two runs.

**True, and the first answer found was not the whole one.**

`Dgetf2` calls `Dger` with the x operand a *column* of the working matrix, so
`incX` is `lda`, and with the target a trailing submatrix, so its leading
dimension is the parent's width rather than n. The kernel takes neither a
stride nor a leading dimension — six arguments is the SysV limit and those are
what got dropped — so every call was rejected by the guard and delegated.

Rewriting it as one accelerated axpy per row lifts both restrictions and made
things *worse*: 12% slower at 256, 7% faster at 512. Row length in a
factorisation shrinks by one per column, so most calls are short, and a per-row
dispatch against gonum's own per-row SSE2 axpy only pays once the row covers
the call. Measured crossover is 256 elements — 0.20x at 16, 0.72x at 64, 0.91x
at 128, 1.07x at 256, 1.50x at 1024 — and with that threshold in place LU
returns to parity.

That explained the panel factorisation and should have been checked against the
rest, because `Dgetrf` is blocked and the panel is not where the time is. The
blocked path calls

	bi.Dgemm(NoTrans, NoTrans, m-j-jb, n-j-jb, jb, -1, a[...], lda, ...)

with **alpha of -1** and every operand a submatrix window. The gemm fast path
requires alpha of 1 and natural leading dimensions, so it rejects those too —
and `Dtrsm`, the other half of the blocked step, is not accelerated at all.
Every accelerated routine is bypassed inside a decomposition, and `ger` was
only the one that was looked at first.

**What to take from it.** A call-frequency count says where the calls are, not
where the time is, and nothing about whether they have a shape the fast path
can take. LAPACK works almost entirely on submatrix windows with a scaling
factor, which is precisely the shape a first-cut guard excludes as "not the
common case" — it is not the common case in application code and it is nearly
the only case inside a decomposition.

The kernel is worth having for a contiguous rank-1 update, which is what it now
claims and all it claims. Making decompositions faster is a different piece of
work: widening the gemm guard to accept a leading dimension and a general
alpha and beta, and writing `trsm`. Neither is hard; both were assumed
unnecessary.

## 73. A correctness fix that cost 75%, and went two days unnoticed

**Assumed.** Fixing the sign of zero in `pow` was a small change. C99 F.10.4.4
requires `pow(-0, 3)` to be `-0` and `pow(-0, -3)` to be `-Inf`, the kernel
returned `+0` and `+Inf`, and the fix is three lines:

	int negzero_odd = odd && (f64_to_bits(x) >> 63) != 0;
	v = (x == 0.0) ? (y < 0.0 ? (negzero_odd ? -INF : INF)
	                          : (negzero_odd ? -0.0 : 0.0))
	               : v;

Correct, tested, committed. Nobody ran the benchmark gate afterwards.

**True.** It costs **75%**. `Pow` at n=4096 went from 12.6 µs to 22.0 µs, and
the same at 256 and 65536, and for both the accurate and the `Fast` tier —
seven benchmarks, all between +72% and +75%.

The reason is where the code sits. Those three lines are not on a rare branch;
they are inside the vectorized loop, so the bit extract and the two nested
selects execute for every element whether or not any element is a zero. A
scalar implementation would predict the branch and pay nothing on ordinary
data. A vector one has no branch to predict.

**How it surfaced.** `make bench-check` against the recorded baseline, run for
an unrelated reason — the machine happened to be idle. It flagged the seven
`Pow` benchmarks in two consecutive full runs with the deltas agreeing to
within 0.3%, which is what separates a regression from noise: the same run also
flagged `SatSub` once and `MulInt` once, and neither reproduced.

**What was done.** Nothing, to the code. The kernel is right and the cost is
what being right costs here; the alternative is a `pow` that violates the
standard on a case the accuracy contract explicitly covers. The baseline was
re-recorded with this entry as the reason.

**What to take from it.** Two things. A correctness fix is a performance change
and needs the same gate — the assumption that it was too small to measure is
the whole of the mistake. And the benchmark gate only works if it is run: this
sat for two days across a dozen commits, and would have sat indefinitely,
because nothing fails when a benchmark is merely slower.

Worth noting what the same run said in the other direction: fourteen benchmarks
were *faster* than the baseline by 25–52%, `Median` and `ParseInts` among them.
A stale baseline hides improvements exactly as well as it hides regressions.

## 74. The portable spelling of a movemask is not a movemask

**What was believed.** That turning a vector comparison into one bit per lane
could be written the way everything else in `csrc` is written — as the plain
loop LLVM is expected to recognise:

```c
u64 bits = 0;
for (int j = 0; j < 64; j++) bits |= (u64)(hit[j] != 0) << j;
```

This is the same reasoning that produced `POPCOUNT_AT` in `compress.c`, and the
comment there is explicit about why: "the byte sum is what every target
vectorizes". Extending it to a bit-gather was the obvious next step.

**What happened.** It was compiled and read before being committed to, which is
the only reason it is an entry here and not a shipped kernel. Against
`-march=x86-64-v4` the loop version produces a stack frame, a saved base
pointer, a 288-byte stack allocation, a spill of both halves of the vector to
memory and a scalar loop over them. The bit-cast version:

```
vpcmpeqb (%rsi,%r8), %zmm0, %k0
kmovq    %k0, (%rdi,%r10,8)
```

Two instructions per sixty-four bytes.

**What was done.** `__builtin_bit_cast(unsigned long long, m)` on a `_Bool`
ext_vector, wrapped in the `STORE_MASK_BITS` macro so the reason is recorded
next to the code that depends on it.

**What to take from it.** The "write it plainly and let LLVM see it" rule that
holds across the rest of this repository has an edge: it holds for arithmetic
LLVM has a vectorizer for, and a lane-to-bit gather is not that — there is no
loop shape it reconstructs into a movemask. Where the target has a single
instruction for the whole operation, the builtin that names it is not premature.
Compiling both and reading the output took two minutes.

## The has-a-zero-byte trick does not say where the zero byte is

`(x - lo) &^ x & hi` is the standard test for "does this word contain a zero
byte", and it is correct for that question, which is nearly always the question.
It is not correct for "which bytes are zero". The subtraction borrows across
byte boundaries, so a zero byte can set the flag on its neighbour.

That distinction does not matter anywhere this expression was already used --
`OR_ANY` reduces the whole accumulator to one bit and never looks at a lane --
and it is the whole answer for the mask builders, which write one bit per byte
into a buffer that a parser then indexes. A bit in the wrong place is a wrong
answer, not a slow one.

Rewriting `ref.MaskBits`, `MaskBitsAny`, `MaskBitsAny4` and `MaskBitsLess` word
at a time with the familiar expression produced masks that were wrong for
`"\x00"` at byte 1, for every byte above 0x7f under `MaskBitsLess`, and for
whitespace next to a match. `GOSIMD=scalar` in `make test-tiers` caught all of
it; the conformance differential did not, because it works on inputs above the
dispatch threshold, where the kernels answer and the portable path is never
reached.

The exact forms:

	zero:  ^(((x & lo7) + lo7) | x | lo7)     no byte of (x&lo7)+lo7 exceeds 0xfe
	less:  ^((x&^hi | hi) - lo*c) &^ x & hi   0x80 + low7 - c cannot go negative

Both replace a borrow that crosses bytes with one that cannot.

**The rule.** An expression that reduces to one bit and an expression that
answers per lane are different expressions, and the well-known one is usually
the first. Check which question the caller is asking before reusing it -- and
run the tier lane, because the portable path is the one the differential suite
cannot see.


## The README's function count was right and the check on it was wrong

The README says "473 exported functions ... the function count is for an
ordinary build; the `goexperiment.simd` vector type adds four more."

Counted from outside:

	go doc . | grep -c '^func '                    469
	GOEXPERIMENT=simd go doc . | grep -c '^func '  473

which reads as an obvious defect: 473 is the experiment's count, the ordinary
build has 469, and the sentence has them the wrong way round. The difference is
four, the sentence says the vector type adds four, and `diff` on the two
listings names them -- MapFloat32x8, MapFloat64x4, ZipFloat32x8, ZipFloat64x4.
Every piece fits.

It is wrong. `go doc` prints a type's constructors indented under the type, so
`grep '^func '` at column zero misses them, and this package has exactly four:
NewFFTPlan, NewRFFTPlan, NewMultiSearcher, NewRK4Workspace. The ordinary build
has 473 exported functions. The README is right, and
TestReadmeCountsAreCurrent has been checking it all along -- it parses the
package with the goexperiment tags stripped and asserts the number, which is why
it passed while the "defect" was being written up.

Two unrelated groups of four, and the coincidence made a false explanation fit
better than the true one. A task had already been filed against this repository
and the number had already been deleted from simdjson's README as unverifiable.
Both undone.

**The rule.** When a measurement disagrees with a claim, the measurement is a
suspect too -- especially an ad-hoc one built out of grep, and most of all when
the story it tells is tidy. There was a test in the tree that answered this
question properly and it was passing; running it first would have cost nothing.
Check whether the thing is already checked before concluding it is not.

## CRC32C by PCLMUL fold, against the standard library's three streams

The fold-by-four kernel with a crc32-instruction drain reaches 20 GB/s
on amd64/avx512. hash/crc32's Castagnoli assembly reaches 37: it runs
three independent crc32 instruction streams and recombines them, and
the instruction's three-cycle latency pipelines across streams in a way
one folded stream does not beat. Minimum of three, quiet machine, 1 KB
to 1 MB. Below ~128 bytes the kernel wins (16.8 against 13.2 GB/s at
64 bytes) because the fold never runs there -- that is the plain
eight-byte instruction loop against stdlib's setup overhead.

Kept, documented as losing: the function is the portable specification
and the small-input path, and the doc comment sends bulk hashing to the
standard library by name. Adler-32 from the same file ships the other
verdict -- 1.8x stdlib everywhere past a kilobyte.

## CRC32C multi-stream / wider fold: attempted, and why it stays a non-battle

The question after Adler's 7.2x win: could a wider fold beat stdlib's 37
GB/s on bulk CRC32C, the way explicit multi-lane beat autovectorized
Adler? Attempted fold-by-8 (eight 128-bit PCLMUL accumulators, 128
bytes/iteration, twice fold-by-4's parallel chains). It failed the
correctness check: the fold constants for a 128-byte distance are derived
values (x^1024 and x^1088 mod the reflected Castagnoli poly), not
guessable, and a self-calibrating reflected mod-pow to recover them did
not reproduce even the known-good fold-by-4 constants -- the exact
bit-reflection convention is the classic GF(2) rabbit hole.

But the deciding fact is hardware, not the constants. The crc32
instruction runs on a single execution port at 1/cycle -- 8 bytes/cycle,
about 40 GB/s at this clock. stdlib's three independent crc32 streams
already saturate that port, which is precisely why it uses three and
stops; more crc32 streams cannot beat a saturated port. So the entire
crc32-instruction avenue is capped at stdlib's level -- there is no win
there by out-streaming. The only ceiling-breaker is VPCLMULQDQ (AVX-512
carry-less multiply on the separate vector ports), which adds AVX-512
constant derivation on top of the fold-constant problem for an upside
bounded by stdlib already sitting at ~37 of ~40 GB/s.

Verdict unchanged and now understood at the port level: bulk CRC32C on
amd64 goes to the standard library (the doc comment says so by name), our
kernel wins small inputs and is the portable spec. Same shape as the
integer prefix-sums (0.67x, not shipped) -- attempted, measured, the
instruction economics settle it, and the entry is the deliverable.
Adler-32 shipped the opposite verdict from the same file only because
stdlib's Adler is scalar Go with no hardware instruction to lose to.

## SortInto's scratch does not cover duplicate extraction

`SortInto` was documented as allocating nothing, but its allocation test used
one random slice and did not reach the duplicate-skew recovery path. Rebuilding
a 4,096-element float64 input from three distinct values before every measured
call reports exactly two allocations per run. The existing random case reports
zero.

The instructions identify both allocations. `extractEqual` contains:

	CALL runtime.makeslice(SB)

That call creates the `[]bool` mask used to find and compress elements unequal
to the pivot. The three-value input reaches `extractEqual` twice. The caller's
`[]T` scratch stores partition output; it does not provide storage for the mask.

The resulting contract is narrower: `SortInto` avoids `Sort`'s unconditional
element-scratch allocation, but duplicate-skew recovery can allocate one mask
for each equal run it extracts. The active API documentation now states that
exception. Historical changelog text remains a record of the implementation at
the time of each release.

## Every complex kernel was generated, tested, and dispatched to by nothing

**Believed.** The complex surface -- `DotComplex`, `DotComplexConj`,
`ScaleComplex`, `SumComplex`, the arithmetic set, and the ComplexParts half
that produces real slices -- runs on the generated assembly like the rest of
the library. nine `internal/<arch>/complex_*.s` files exist -- sse2, avx2, avx512, neon,
sve2, rvv, vsx, vx, lasx, across six architectures -- `registerComplex<TIER>`
installs them, and `TestInventoryCoversEveryGroup` asserts that `C64`,
`C128`, `C64Parts` and `C128Parts` all carry declared kernels. All of that
was true.

**Actually.** None of it was reachable from a consumer. `complexOps[C]()`
returned `&refBase.C64` / `&refBase.C128` *directly*, with no per-tier
overlay, and `complexParts` did the same:

```go
case complex128:
    return any(&refBase.C128).(*kernel.Complex[C])   // complex.go:33, before
```

The dispatch tables the runtime indexes carried no complex entry at all --
`grep -c Complex dispatch_tables_amd64.go` was **0**, against ten
`opsXXByTier` tables for the real element types -- because
`tools/simdgen/emit/dispatch_gen.go` had a `numericElem` map and no complex
equivalent, so `numericGroupsIn` skipped every complex group and emitted
nothing for them. The generated complex assembly was reachable only from
`internal/<arch>/sets_gen_<arch>.go`, whose own generated header says "Tests
call this; consumers never do".

The test that should have caught it is `TestDeclaredKernelsAreWired`, not
`TestInventoryCoversEveryGroup` -- the first version of this entry named the
wrong one. `TestDeclaredKernelsAreWired` checks the runtime tables
(`allFlatTables`) for exactly three groups, `Bytes`, `Convert` and `Mask`,
and sends everything else -- numeric AND complex -- through `archSets()`,
the test-only aggregator. So the numeric groups have the same nominal hole;
they were saved by having real tables, not by being checked. Nothing walked
the tables the runtime actually uses for a non-flat group.

Confirmed by disassembly, which is the only thing that would have shown it.
The symbol survives only where the call is not inlined -- the root package's
own test binary inlines it and prints nothing, so this is the
`internal/benchmarks` binary:

```
$ go test -c -o /tmp/bench.test ./internal/benchmarks
$ go tool objdump -s 'simd\.DotComplex\[' /tmp/bench.test
  complex.go:31  LEAQ github.com/sebishogun/simd.refBase+13560(SB), AX
```

Empirically, from a consumer module calling `simd.DotComplex`: 0 complex
kernel symbols linked before, 24 after. A consumer calling only `simd.Sum`
links 0 in both, so the per-operation dead-code elimination survives.

**How it surfaced.** Not from this repository. A downstream evaluation
(simdblas Task 4, complex Level 1 against gonum) measured `Zdotu`/`Zdotc`
2.2x-3.1x slower than gonum at n <= 100000, 1.65x-1.86x at n = 1e6. Not
against "a plain Go loop", as the first version of this entry said: gonum's
unit-increment complex dot calls `c128.DotuUnitary`/`DotcUnitary`, which are
hand-written AVX assembly on amd64. Losing to it is not by itself proof of
anything. The tell was that `GOSIMD=scalar` and the detected tier gave
identical timings.

**Cost, measured.** `GOSIMD=scalar` against `avx512` in one binary --
one build, so the 8.3% layout floor does not apply; there is no second
layout to differ. Complete grid, minimum of six alternating passes,
reproduced twice. **amd64/avx512 only**: no other architecture was measured,
and the per-tier ratios elsewhere are not these.

The machine was at load 5-12 for this, against the project rule of load
under 1. The wall-clock column is therefore the weaker evidence; the
instruction column below it is layout- and load-independent and is what the
conclusion rests on.

| operation | scalar ns/op | avx512 ns/op | ratio |
|---|---|---|---|
| Dot c128 n=1024 | 749.7 | 101.5 | 7.39x |
| Dot c128 n=65536 | 47156 | 15459 | 3.05x |
| Dot c128 n=1048576 | 805434 | 408702 | 1.97x |
| DotConj c128 n=1024 | 890.1 | 102.5 | 8.68x |
| DotConj c128 n=65536 | 55570 | 15488 | 3.59x |
| DotConj c64 n=1024 | 1249 | 97.31 | 12.8x |
| DotConj c64 n=65536 | 83114 | 6614 | 12.6x |
| DotConj c64 n=1048576 | 1276125 | 158120 | 8.07x |
| Sum c128 n=1024 | 470.1 | 43.43 | 10.8x |
| Sum c128 n=65536 | 29002 | 5611 | 5.17x |
| Sum c128 n=1048576 | 480519 | 115280 | 4.17x |
| DotConj c128 n=1048576 | 918445 | 420455 | 2.18x |

Instructions retired per element, two-point slope so setup cancels, which
is the number that does not move with load:

| operation | scalar | avx512 | ratio |
|---|---|---|---|
| Sum c128 n=1048576 | 12577259 | 1040671 | 12.1x |
| Dot c128 n=1048576 | 32070536 | 2627180 | 12.2x |
| DotConj c64 n=1048576 | 44563734 | 2030498 | 22.0x |

(The three `DotConj/c128` cells have no instruction row: `go test -bench`
cannot exclude the deeper `impl=naive` sub-benchmark, so selecting them
also runs it and its instructions scale with `-benchtime` too.)

**When, exactly.** Not "since the kernels were written" -- the first version
of this entry said that and it is wrong. The kernels landed in `ba09d24` and
dispatched correctly through `active.C64`/`active.C128`. The break came with
`02c258a`, "Per-operation dispatch: the linker keeps only what a program
calls", which replaced `active` with `refBase` plus per-tier tables and
emitted no complex tables:

```
$ git show v1.13.0:complex.go | grep 'active\.C64'   -> return any(&active.C64)...
$ git show v1.14.0:complex.go | grep 'refBase\.C64'  -> return any(&refBase.C64)...
$ git tag --contains 02c258a --sort=creatordate | head -1
v1.14.0
```

So the affected releases are **v1.14.0 through v1.20.0** -- seven tags. A
consumer pinned to v1.13.0 or earlier was never affected. The table
mechanism itself arrived in that commit, so the missing `complexElem` was
never an omission from an older design; it was a group the new design did
not carry over.

**The repository's own baseline held the evidence.** `testdata/bench/amd64.txt`
was recorded at `f6c72f9`, before the regression, with
`BenchmarkComplexReduce/Dot/c128/n=1024-8` at 101.2-101.9 ns/op -- a
benchmark that calls the public `simd.DotComplex`. The portable path for
that shape is 749.7 ns/op, 7x against `make bench-check`'s 25% threshold.
Whether the gate was run at `02c258a` is not recoverable from here; the
baseline was never updated, which is why it is still pre-regression and
still a working detector.

**Fix.** `dispatch_gen.go` gained `complexElem`/`partsElem` and an
`emitTierTable` that writes `cplx<Group>ByTier` and `parts<Group>ByTier`
alongside the existing `ops<Group>ByTier`; `dispatch.go` gained a
`groupCache[G any]` (opsCache is constrained to `Number`, and neither
`kernel.Complex` nor `kernel.ComplexParts` is `kernel.Ops`) with an
`overlayAny`; `complexOps`/`complexParts` now route through it. The
regenerated tables are **additive only** -- 0 deleted lines across all seven
`dispatch_tables_*.go`, 479 added.

`dispatch_complex_test.go` is the test that would have caught it: it compares
the function pointer `DotComplex` actually reaches against the reference
set's, which is the question a caller asks and the inventory walk does not.

## The three-way partition may already be built, and the 34% has not been re-measured

ROADMAP.md records one losing case for `Sort`: few distinct values at 16384,
by 34% against `slices.Sort`, with "**the fix is a three-way partition**" and a
note that it "needs a second kernel". `docs/plans/2026-08-13-simd-production.md`
Task 1 carries the same plan.

Reading `sort.go` before writing the kernel: `extractEqual` already does the
three-way split. When the split comes back skewed past `sortSkewLimit`, it
takes the run equal to the pivot out of the recursion entirely — which is what
a below/equal/above partition buys — and it does so with kernels this package
already ships (`EqualScalarInto`, `CountTrue`, `NotMask`, `CompressInto`), so
no second kernel was needed for that part. Its own comment says so.

That does not close the task, and the reason is that nothing here has been
measured on a quiet machine. Two attempts:

    load 2.2-2.5, min of 3 x 300:   simd 28,531 ns   slices 23,000 ns   1.24x
    load 12.0,    min of 5 x 400:   simd 117,359 ns  slices 79,546 ns   noise

The second set is unusable — the same benchmark moved by 4x between them, and
three review agents were running. The first set is not quiet either. What can
be said: the gap at that shape is real and still there, and 1.24x is not 1.34x.
What cannot be said: whether it is 24%, whether the difference from 34% is the
`extractEqual` work landing, or whether either number survives a quiet machine.

So no number in ROADMAP.md was changed. Overwriting a measured claim with a
worse-conditioned one is not a correction. What the task needs first is the
measurement under load average below 1, and then a profile — because if the
residual is the `copy(a, scratch[:len(a)])` after every partition rather than
the split itself, a three-way kernel does not address it and would be built for
a cause nobody checked.

## A forgotten reference wiring shipped past the whole normal-lane suite

**Probed, 2026-08-15.** `Sqrt: sqrt[T]` deleted from `floatOps` in
`internal/ref/ref.go` — one entry, the shape of a wiring somebody forgets when
adding a kernel.

| lane | result |
|---|---|
| `TestDeclaredKernelsAreWired`, normal | **PASS** (verified with `-v` that it ran) |
| `TestDeclaredKernelsAreWired`, purego | **PASS** |
| whole suite, normal — all 14 packages | **PASS** |
| whole suite, purego | FAIL, `TestInPlaceMatchesInto` |

So the completeness gate that exists for exactly this did not see it, and the
only thing that did was a functional test in a lane nobody runs by default.

The cause is scope. `TestDeclaredKernelsAreWired` walks `backend.Inventory`
against the dispatch tables and `archSets()` — the GENERATED side. `ref.Set()`
is never in the subject, and its doc comment says so plainly: "The subject is
`active` ... and not `ref.Set()`. That is not a shortcut, it is the only
correct choice", because ref leaves every Fast slot nil until a backend
registers. That reasoning is right about Fast slots and was taken to cover the
whole reference.

The reference is not optional. It is the live fallback on architectures with
no backend, in `purego` builds, and below every kernel's element threshold —
so a nil there is a nil call in the small-n path of an operation that works on
a big slice.

`TestDeclaredKernelsAreWiredInTheReference` is the other half: every
non-`Fast` entry in `backend.Inventory` must be non-nil in `refBase`. 853
today. The `Fast` exclusion is the same one the older test's comment describes,
for the same reason.

**Also settled, and against a plan.** `docs/plans/…-simd-production.md`'s item
1 proposes generating the `internal/ref` ops-table entry and the
`kernel.Ops` struct field from the manifest, and justifies it as removing "the
two easiest things to forget — one of which caused the `-tags purego` panic
that `TestDispatchTableComplete` now guards". Half of that is wrong: a
forgotten `kernel.Ops` FIELD cannot ship, because the generator emits code
referencing it and the build fails; a forgotten `ref` WIRING shipped past
fourteen packages, as above. The gate is the fix; codegen would be
convenience, and it would move the numerical contract — which lives in the
`kernel.Ops` field comments — into a manifest.

## `make fuzz` fuzzed one of its three targets and exited 0

Measured 2026-08-16, by sweeping the family for fuzz targets nothing runs.

The recipe named two targets and the comment above it opened "Two fuzz
targets". There are three, and only ONE of them was being fuzzed:

| target | what happened |
|---|---|
| `FuzzKernelsAgainstReference` | fuzzed, correctly |
| `FuzzDifferential` | named, and run in the ROOT package while the target lives in `internal/tests/arrays` |
| `FuzzJSONMasksMatchesSeparateCalls` | never named; only its seed corpus ever ran, under `go test` |

The middle row is the one worth the entry:

```
$ go test -run '^$' -fuzz FuzzDifferential -fuzztime 3s .
PASS
ok  	github.com/sebishogun/simd	0.002s
EXIT=0
```

**`go test -fuzz X` in a package with no target called X exits 0.** It does not
error, it does not warn, it prints `PASS`. So half of `make fuzz` had been
reporting success for a step that fuzzed nothing, and the only tell was a
duration of two milliseconds where sixty seconds were asked for.

Targets are discovered per package now — `go test -list '^Fuzz'` asks the
compiler, and a target can only be fuzzed in the package it was found in, so
the wrong-package shape is unreachable rather than merely fixed. The recipe
also fails outright if it ends up running nothing at all, because a discovery
loop over an empty list is the same silent green one level up.

**The same sweep, across the family.** Ten repositories. `simdlogs` already
discovered its targets and its workflow says why: "A hand-maintained list is
how a new target gets written and never run." `simdhttp` hand-listed three of
four and is fixed (its own entries 15 and 16 — the missing one had been named
in a verification document since v1.2.0 and never written). `simdcsv` (3
targets) and `simdparquet` (13) have no fuzz recipe at all; `simdjson` names 11
of 14. Those are recorded here and not yet fixed.

**The shape.** A gate that cannot fail is this family's most persistent defect,
and every instance so far has been a test or an assertion. This is the same
thing at the build level: a Makefile recipe whose command succeeds while doing
nothing, in the repository the others depend on.

## 75. The archsimd small-n win depends on inlining, and a package cannot inline

`docs/research/08-goexperiment-simd-small-n.md` measured `GOEXPERIMENT=simd`
intrinsics against the portable loop at −19.2% (n=8), −26.8% (16) and −31.7%
(32) instructions, and concluded: build it, elementwise first. The plan put the
implementation in a new `internal/fastpath` package that every generated guard
would call below its threshold.

Built, bit-identity-gated, and measured. `perf stat -e instructions:u`,
2,000,000 iterations, interleaved, three rounds at n=8 and two at the rest,
disjoint ranges throughout:

| | ref | archsimd | |
|---|---|---|---|
| f32 n=8 | 124.06M | 155.85M | **+25.6%** |
| f32 n=9 | 136.12M | 288.08M | **+111.6%** |
| f32 n=16 | 220.20M | 228.13M | **+3.6%** |
| f32 n=17 | 232.14M | 360.57M | **+55.3%** |
| f32 n=32 | 412.03M | 371.91M | −9.7% |
| f64 n=8 | 124.17M | 228.07M | **+83.7%** |
| f64 n=16 | 220.05M | 372.34M | **+69.2%** |

It loses across the whole band the guards actually use, and wins only at f32
n≥32 — above where any guard falls back.

**The cause is inlining, and the research document had already named the risk
without measuring it.** `-gcflags=-m`:

```
fastpath_ref.go:16:6:  can inline AddFloat32          <- the one-line forward to ref
fastpath_simd.go:38:36: inlining call to archsimd.LoadFloat32x8Slice
                        (no "can inline AddFloat32")  <- the loop cannot
```

The portable fallback is a one-line forward, so the guard inlines it and pays
nothing to reach it. The archsimd version is a loop, and Go does not inline a
function containing one — so it is a real call, and at n=8 that call is most of
the work. The record's reproducer was a direct inlined loop and said so: *"It is
a direct call, not the guarded one … the end-to-end figure will be smaller than
19–32%."* Smaller was the wrong word. It is negative.

**The record's table measured only exact multiples of the vector width, and that
hid the worst of it.** n=9 and n=17 — one full vector plus a one-element tail,
which is the shape the guard band mostly sees — cost +111.6% and +55.3%. A table
of 8/16/32 cannot show a tail cost.

**float64 loses everywhere in the band**, and for a structural reason: 4 lanes
means n=8 is two vectors and n=16 is four, so the call is amortised over half as
much arithmetic as f32 at the same length. The research document measured f32
only.

Reverted. What the next attempt has to do differently is emit the intrinsics
**inside the generated backend**, in the same package as the guard, where they
can inline into it — not behind a package boundary. Until something does that,
the measured answer for the guard band is that `internal/ref` is the faster
fallback.

The bit-identity gate that came with it did pass, at every length 0–40 over
IEEE specials, on both lanes: elementwise archsimd IS bit-identical to `ref`,
which was the other thing that had to be true. That half of the record stands.

## 76. Forced vectorization: the bucket the pragma is for is the bucket the contract forbids

The plan's item 5 proposed trying `#pragma clang loop vectorize(enable)` per
(kernel, target) and keeping measured winners, over "346 refusals: ~60
`scatter_*`, ~30 `reverse_*`, 114 float/`fast_*` — the plausible wins".
Measured, before writing any pragma. Every number in that sentence is wrong,
and the investigation ends somewhere else.

**The inventory.** `make check-emission`, clang 22.1.8: **336** (kernel, tier)
refusals over **132** distinct kernels, not 346.

| tier | vsx | neon | sse2 | vx | sve2 | lasx | rvv | avx2 | avx512 |
|---|---|---|---|---|---|---|---|---|---|
| refusals | 81 | 80 | 56 | 36 | 33 | 23 | 14 | 10 | 3 |

By operation, as (kernel, tier) pairs: `scatter` 65, `gather` 31, `reverse` 30,
`tile` 10, `reversebits` 10 — the permute/scatter family is **162 of 336, 48%**,
and each needs an instruction the target does not have. `gather` at 31 pairs is
larger than `reverse` and the plan did not mention it.

**There is no float bucket.** `fast_*` refusals: **zero**. Float-typed
refusals: 59 of 336, and on the three amd64 tiers only **2** — both
`f8e*_to_f32`, which are integer bit manipulation. The refused
`prod`/`dot`/`sumsq*`/`sumsqdev`/`sumsqdiff` entries are all `i64`/`u64`:
64-bit integer reductions, blocked by a missing multiply (SSE2 has no
`pmullq`), not by anything a pragma reaches.

**The pragma's own bucket is ten loops.** `-Rpass-analysis=loop-vectorize` over
all of `csrc` at sse2, by reason:

| reason | loops |
|---|---|
| instruction return type cannot be vectorized | 1929 |
| loop control flow is not understood | 186 |
| value that could not be identified as reduction is used outside the loop | 128 |
| potentially faulting load | 58 |
| store instruction cannot be vectorized | 47 |
| **cannot prove it is safe to reorder floating-point operations** | **10** |

Ten. And clang's own remark names the remedy: *"allow reordering by specifying
'#pragma clang loop vectorize(enable)' before the loop or by providing the
compiler option '-ffast-math'"*. The pragma there is not a cost-model override,
it is **permission to reassociate** — which changes the answer. The numerical
contract in `internal/kernel/kernel.go` is the reason `-ffp-contract=off` is in
`commonFlags` at all, and the repository already has the correct home for a
kernel that trades it: the `simd_fast_*` prefix. So the one bucket the pragma is
designed for is the one bucket that cannot take it without becoming a different
operation.

The ten are `csrc/numeric.c:121,127,133,139` (the `CONVOLVE` accumulation, four
instantiations), `csrc/numeric.c:399,404`, and `csrc/gemm.c:142` (`GEMM`, four).

**And those same kernels ship scalar today.** `simd_convolve_f32`,
`simd_correlate_f64` and their siblings are refused on **zero** tiers — they
pass the emission gate — yet:

| tier | simd_convolve_f32 | packed arithmetic | scalar arithmetic |
|---|---|---|---|
| sse2 | 119 instrs | **0** | 10 (`mulss`, `addss`) |
| avx2 | 140 instrs | **0** | 10 (`vmulss`, `vaddss`) |
| avx512 | 141 instrs | **0** | 10 |

Its only vector instructions on sse2 are 2 `movups` and 2 `xorps` — a move and
a register zero. `dispatch_tables_amd64.go` carries
`Convolve: amd64.ConvolveFloat32SSE2` and `ConvolveFloat32AVX2`, so a caller on
amd64 dispatches into an assembly kernel that multiplies one element at a time,
paying the dispatch and threshold machinery for it.

**The gate cannot see this, structurally.** `verify.vectorWidth` returns >0 for
any amd64 instruction whose operands mention `%xmm`/`%ymm`/`%zmm` — and every
scalar float instruction on x86-64 uses those registers. `vmulss %xmm0, %xmm1,
%xmm2` counts as vectorized. So `RequireVector`, whose comment says it exists
because "dispatching to it would run scalar code under a name promising
otherwise", cannot report a scalar float kernel on amd64 at all. Its 69 amd64
refusals are integer kernels, necessarily.

**Counted properly, by the generator.** The first pass here was a shell scan
over every symbol in each object, which included helpers that are not kernels
and used a cruder mnemonic match; it reported 26 at sse2 and 8 at avx2. Those
numbers are superseded. `verify.arithKind` now classifies each instruction as
packed arithmetic, scalar arithmetic or neither — moves and the `xorps
%xmm0,%xmm0` zeroing idiom are neither, which is the point — and `make
check-emission` prints `SCALAR-ONLY` for any kernel that does arithmetic with
none of it in lanes:

| tier | instances |
|---|---|
| amd64/avx512 | 9 |
| amd64/sse2 | 8 |
| amd64/avx2 | 8 |
| arm64/sve2 | 8 |
| arm64/neon | 5 |

**38 instances over 13 distinct kernels**: `polyeval`, `convolve`, `correlate`,
`movavg`, `minr` (f32 and f64 each), `rolling_min_f64`, `rolling_max_f64`,
`dtoa_f64`. It is **not an x86 quirk** — arm64 has 13 of the 38, and there the
gate is blind for the same reason in a different spelling: `fmul s0, s1, s2` is
scalar and `fmul v0.4s, v1.4s, v2.4s` is packed, and `vectorWidth` reads the
register file, not the arrangement.

The classifier runs on amd64 and arm64 only, and `Report.ArithKnown` says so:
on the other four architectures a zero count means "not measured", and
`ScalarOnly()` returns false there rather than reporting a finding it did not
make.

**What this leaves.** Not a pragma sweep. Two separate things:

1. `vectorWidth` on amd64 should not count a scalar-form instruction (`*ss`,
   `*sd`) as vector, and `RequireVector` should ask "does this kernel do
   arithmetic, and is none of it packed" rather than "is any register wide".
   A permute kernel does no arithmetic and must still pass.
2. Fixing the gate DROPS the eight, falling back to `internal/ref`. Whether
   that is faster than the scalar assembly is unmeasured: both are scalar, and
   the assembly costs a call the guard would otherwise inline. That is the
   benchmark this needs, and the machine was not quiet (load average 1.17, two
   review agents building) when this was written, so no number is offered.

The transferable part is the same one as entry 25: a gate that asks a proxy
question answers the proxy question. "Contains an instruction that touches a
vector register" is not "vectorized", and on the one architecture that runs
natively here the two are never the same for float code.

## 77. Blocking the outer loop is what vectorizes a reduction kernel, and it is bit-identical

Entry 76 found `polyeval`, `convolve`, `correlate` and `movavg` shipping scalar
on every tier while passing the emission gate, and left the question of what to
do about it open. The answer is not a pragma and not a fallback: it is the loop
nest.

**LLVM's loop vectorizer only ever vectorizes the INNERMOST loop.** Every one of
these kernels is an outer loop over independent outputs wrapped around an inner
reduction with a runtime trip count:

```c
for (isize i = 0; i < n; i++) {         // independent — but not innermost
  T acc = c[nc - 1];
  for (isize k = nc - 2; k >= 0; k--)   // innermost — a serial dependency
    acc = acc * xv + c[k];
  d[i] = acc;
}
```

So the vectorizer looked at the Horner recurrence, could not break it without
reassociating, and gave up — and `polyeval`'s own comment claimed the opposite:
"The outer loop over elements is independent, which is what vectorizes."

**Blocking swaps which loop is innermost without touching any element's
arithmetic.** A fixed-size accumulator array, elements in tiles of 16:

```c
for (; i + 16 <= n; i += 16) {
  T acc[16];
  for (b) acc[b] = c[nc - 1];
  for (k) { T ck = c[k]; for (b) acc[b] = acc[b] * x[i + b] + ck; }
  for (b) d[i + b] = acc[b];
}
```

Every output still evaluates the same coefficients in the same order, so this is
**bit-identical**, not close. That distinction is the whole point: reassociating
is exactly what `#pragma clang loop vectorize(enable)` would have done and what
the numerical contract forbids. Nothing is allocated — 16 doubles is 128 bytes
against the 512-byte kernel frame budget — and the tail runs the original nest.

| | packed arithmetic before | after (sse2) | after (avx2) |
|---|---|---|---|
| `simd_polyeval_f32` | 0 | 24 | 12 |
| `simd_polyeval_f64` | 0 | 16 | 24 |
| `simd_convolve_f32` | 0 | 8 | 12 |
| `simd_correlate_f64` | 0 | 16 | 8 |
| `simd_movavg_f32` | 0 | 16 | 12 |
| `simd_movavg_f64` | 0 | 16 | 16 |

The ten scalar instructions that remain in each are the tail loop.

`SCALAR-ONLY` went from **13 kernels to 5** (`dtoa_f64`, `minr_f32/f64`,
`rolling_min_f64`, `rolling_max_f64`). And loong64 gained two kernels, 728 →
730: `simd_movavg_f32` and `simd_movavg_f64` were refused there with "LLVM did
not vectorize it for lasx" and now emit, so the total moved 6,931 → 6,933.

**The bit-identity claim is tested, and the test was checked for being
vacuous.** `internal/conformance` compares each tier against `ref` at eighteen
length pairs, eleven of which are at or above the guard's threshold of 16 — so
the SIMD path is genuinely reached. Poisoning the blocked path only
(`acc[b] = acc[b]*x[i+b] + ck + 1`) reddens it at `dst=16 x=16 coeffs=2 i=0`,
the first blocked case, which is what says the comparison sees the new code
rather than the fallback.

**Not measured: whether it is faster.** It executes fewer instructions per
element and the arithmetic is in lanes, but the machine was not quiet (load
average 2.27, two review agents building) and no wall-clock or counter number
is offered. What is established is that the kernels now do their arithmetic in
lanes and produce the same bits, which is the precondition for the speed
question rather than an answer to it.

`movavg` also rules out the obvious alternative: a sliding window (subtract the
element leaving, add the one arriving) is O(n) instead of O(n·w) and gives a
DIFFERENT number, because floating-point addition does not cancel.
