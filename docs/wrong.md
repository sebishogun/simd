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

## 14. The standard library is the more accurate side

**Assumed.** Where a kernel and Go's `math` disagree, the kernel is wrong.

**Actually.** In four places measured here, Go's `math` is the less accurate
one. A ULP bound quoted against it is a bound against a moving target, which is
why the bound is *measured and reported every run* rather than asserted from
theory.

---

## 15. `-0.0 < 0.0`

**Assumed.** A sign test can be written as a comparison against zero.

**Actually.** `-0.0 < 0.0` is false. `cbrt(-0)` therefore returned `+0`.

**Fix.** Restore the sign bit explicitly rather than testing the value:
`bits_to_f64(f64_to_bits(v) | (f64_to_bits(x) & signMask))`.

---

## 16. Skipping a zero multiply is a free optimization

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

## 17. A rewrite that produces correct instructions is a correct rewrite

**Assumed.** A constant-pool reference can be re-spelled as a Plan 9 mnemonic
naming a `DATA` symbol, letting Go's linker compute the displacement.

**Actually.** The replacement is not the same length as the instruction it
replaces, so every PC-relative branch spanning it now jumps to the wrong place.

**How it surfaced.** A reciprocal kernel whose remainder loop was simply never
entered. No crash, no diagnostic — just untouched output at the tail.

**Fix.** Patch displacements in place and append the pool to the body. Nothing
changes length, so every branch that was correct in the object file still is.

---

## 18. Vectorizing a loop makes it faster

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

## 19. A disassembler prints register names

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

## 20. The benchmark said it got slower

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

## 21. A test lane that produces no output is running slowly

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

## 22. "No space left on device" means the disk is full

**Assumed.** `ENOSPC` means bytes.

**Actually.** `/tmp` had 40 GB free and *zero* free inodes — a tmpfs is created
with a fixed inode count, and an unrelated test suite had consumed all 1,048,576
of them. Nothing could create a file of any size.

**Fix.** `df -i`, not `df -h`.
