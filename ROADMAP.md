# Roadmap

What is not built yet, in the order it is worth building, with the reason each
one is where it is.

Nothing here is a promise of a date. It is a statement of what the gaps are, so
that a reader can tell the difference between a thing that was decided against
and a thing that has not been reached.

---

## Kernels

### ~~compress / expand / filter~~ — done in v0.1.0

`CompressInto`, `ExpandInto` and `FilterInto` ship, and `IndexAll` is built on
the first of them. Accelerated on AVX-512 (`vpcompressd`) and SVE2 (`compact`);
portable on the other seven, permanently, because they have no compress
instruction and the operation cannot be vectorized without one.

`ExpandInto` is portable everywhere and will stay that way. Compression's
serial half is the store, which a compress instruction fixes; expansion's is
the load, which it does not — clang emits the same scalar loop for it on both
of the targets that have the instruction.

### ~~n-ary and variadic operations~~ — done after v0.1.0

`AddAll` and `MulAll` take any number of slices and make a single pass. Arity-3
and arity-4 kernels on every architecture; longer calls fold in groups of four,
so a sixteen-way sum is four passes rather than fifteen. Arity stops at four
because a five-source kernel needs seven pointer arguments and System V passes
six integers in registers.

The element type is enforced by the compiler — the parameter is `...[]T`, so
mixing types will not build. Lengths remain a run-time property, because a Go
slice carries its length in its header rather than in its type; the work is
bounded by the shortest slice, as everywhere else here.

Still open: the **other half of what was asked for**, a general n-ary
combinator taking a Go closure. That is easy to write and hard to make fast —
a closure call per element defeats vectorization entirely — so it needs a
design where the common shapes dispatch to real kernels and the rest is honest
about being a scalar loop. `FilterInto` is the same problem already solved once
that way.

### ~~sort / argsort~~ — done after v0.1.0

`Sort`, `SortInto`, `Argsort`, `PartitionInto` and `SortedIndex` ship. A
quicksort in Go around a compress-based partition kernel, accelerated on
AVX-512 and SVE2 and portable elsewhere. Against `slices.Sort` on float64 it is
19-27% faster at every size from 16K elements up, and even at 1024.

The one case that loses is few distinct values at 16384, by 34%: the pivot
equals much of the range, the split is lopsided, and the skew guard hands the
range to pdqsort after paying for one partition, which at that size does not
earn its cost back. **The fix is a three-way partition** — splitting into
below / equal / above so duplicates are consumed at each level instead of being
pushed to one side. That needs a second kernel and is the obvious next step
here.

---

## Backends

### ~~ppc64le: repoint the TOC prologue at an appended pool~~ — done

**Shipped.** 281 kernels became 468, and 592 as the kernel set grew. The
approach is the one sketched here before it was written: emit the pool as a Go
`DATA`/`GLOBL` symbol, put `MOVD $pool<>(SB), R2` in the generator's prologue,
overwrite clang's two global-entry instructions with NOPs — length-preserving,
which is not optional — and recompute each `R_PPC64_TOC16_HA/LO` immediate as
an offset from the pool base, minding the HA/LO sign adjustment.

What the sketch had wrong is worth keeping. It assumed the obstacle was
Power9's lack of PC-relative data addressing. It was not: Go's own assembler
materialises a symbol address in two instructions with no TOC involvement, so
the pool never needed to be reached PC-relatively at all.

The eleven kernels ppc64le still declines are a different wall each: `r30` and
`r2` are registers Go owns, and seventeen more write above `r1` into the
caller's frame, which would need a SaveArea and a trampoline — and the ELFv2
trampoline is unsafe here, because a `bl` to a global symbol is followed by a
slot the linker may rewrite into a TOC reload, and in a trampoline that slot
is the `RET`. That decision is recorded at `target.go`.

### s390x

413 kernels, and the missing ones are missing because clang uses `r13`,
which is where Go keeps the current goroutine. There is no `-ffixed-r13` for
SystemZ; the global register variable is accepted and silently ignored. Four
routes were probed and all four are dead, so this is upstream — recorded here
so that its absence is not mistaken for an oversight.

An earlier version of this file said 614 and the README said 650. Both were
wrong: 614 was the count before the r13 rule was enforced, and 650 double-
counted registrations and their wrappers. The number has in fact risen
monotonically, 325 to 413, and never fell.

### amd64 sse2: closed — 673 kernels became 789

**Shipped.** The baseline x86-64 tier used to decline around 170 kernels
because a legacy SSE instruction with a memory operand requires a
16-byte-aligned address and has no unaligned form of the same length. It was
the largest addressable bucket in the repository. There are now no alignment
refusals on any tier.

The reason on file for not attempting it — the comment on `checkPatchable` —
rejected padding the appended pool to a 16-byte offset, since that relies on the
linker aligning text symbols and `-ldflags=-funcalign=8` would then SIGSEGV.
Correct, and about appending the pool *inside* the TEXT symbol. It does not
apply to a separate `DATA`/`GLOBL` symbol, which the linker aligns by size:

```
GLOBL simdpool<>(SB), RODATA|NOPTR, $32
  default        0x4d0880   %16 = 0
  -funcalign=8   0x4ca880   %16 = 0
```

So the pool moved to RODATA and the reading instruction became a Plan 9
mnemonic — `MULPS simdpool<>+0x20(SB), X3` — which works only because Go's
assembler emits byte-for-byte what clang does, at the same length, so every
branch displacement in the body stays correct. That equivalence is now
`TestRespelledEncodingsMatchClang`, run for all forty mnemonics against both
register ranges rather than assumed.

[`docs/wrong.md`](docs/wrong.md) entry 61 has the reasoning and entry 67 the
outcome, including the three ways the first version was wrong and which
mechanism caught each.

### The wall the remaining item shares

Three of the four items that used to sit behind this wall are through it.
Sorted-set intersection and difference shipped, because the tile turned out to
need only constant shuffles; the varint widths shipped, with the emission
itself recorded as serial rather than pending; and the sse2 emission is done.

And the fourth is through it too, by a route nobody expected. **The RVV and
NEON transcendentals needed no intrinsics at all.** NEON's five float64
refusals were a cost model declining a good trade, which a pragma overrules;
RVV's thirteen were `llvm.is.fpclass`, emitted from a denormal test written as
a float comparison, which the vectorizer cannot cost on that target and which
an equivalent test on the bits avoids entirely. No accurate transcendental
refuses on any target now. See docs/wrong.md entries 68, 69 and 70 — the last
being the ppc64le segfault that forcing caused on a target nobody had checked.

That is not a difficulty ranking, it is a different kind of work. Everything in
this repository gets its cross-architecture identity for free by compiling one
source everywhere and checking the results against each other. The first kernel
written per-target gives that up, and the differential suite becomes the only
thing standing between a subtly different evaluation order and a shipped
inconsistency. Worth doing; worth deciding to do deliberately.

For the transcendentals specifically, the pragma route is closed rather than
untried: `#pragma clang loop vectorize(enable)` on the RVV refusals produces 22
instances of "the optimizer was unable to perform the requested
transformation". Entry 63.

## Tiers

### A `GOEXPERIMENT=simd` tier — half shipped, half measured and rejected

**Both halves are now settled, and they went opposite ways.**

**Shipped: the vector type.** `vec.go`, behind `//go:build goexperiment.simd &&
amd64`, aliases `simd/archsimd`'s types so every one of its methods is
available through one import, adds `Lanes[T]()` so a caller need not guess the
width, and wraps the loop-plus-tail shape that hand-written SIMD gets wrong.
`vec_stub.go` is what every other configuration compiles: `Lanes` returning 0
and `HasVectorType` false, and deliberately nothing else — a `F32x8` aliased to
something that is not a vector would compile everywhere and be fast in one
place, with nothing saying which.

`archsimd` is **amd64 only** in Go 1.26, verified rather than assumed: all ten
of its implementation files are `_amd64.go`, `GOARCH=arm64` fails with
`undefined: archsimd.LoadFloat32x8Slice`, and its own doc says "It currently
supports AMD64". So five architectures get the stub.

`make verify` now runs `test-vec`, because a build tag nothing exercises is the
vacuously-green lane [`docs/wrong.md`](docs/wrong.md) entry 41 warns about.

**Rejected: routing the small-n fallback through it.** This was the plan, the
prize looked like 2 ns in the band below the dispatch threshold, and the
package was built, made bit-identical to `internal/ref` over an adversarial
corpus, and benchmarked against it — where it won at every size, up to 2.1×.

Then it was A/B'd through the **public API** against the previous tree, which
is the only path a real program takes:

| n | before | after |
|---|---|---|
| 4 | 4.50 ns | 5.49 |
| 8 | 5.50 | **8.04** |
| 16 | 5.49 | 5.44 |
| 256 | 9.05 | 9.47 |

A 46% regression at n = 8. The package's own benchmark measured a call shape
that does not exist: it passed a concrete `[]float32` the compiler inlines,
where a generated guard cannot. And the band does not exist anyway — the
fallback runs only below the threshold, which is 16, and at 16 and above the
generated kernel already beats archsimd (5.44 against 6.42), because clang
unrolls and schedules across iterations. The two windows do not overlap.

Deleted rather than left unwired. [`docs/wrong.md`](docs/wrong.md) entry 58 has
the whole chain, including the earlier generic version that boxed a slice
header on every call.

**What this changes about v2.** The vector type is already here and already
structured for it: delete two build tags and bump the module path. What v2 does
*not* get is the fastpath, and the reason is measured rather than deferred.

### Why the small-n band is the only opening

Go 1.26 shipped SIMD intrinsics behind `GOEXPERIMENT=simd`, and they are
targeted at the language proper in 1.27.

They cannot replace this library — they do not reach SVE2 or RVV, they cover
amd64 and arm64 only, and requiring a consumer to set a `GOEXPERIMENT` is not
something a library can do. But there is one band where they beat assembly and
always will: **below the call boundary**.

An assembly kernel cannot be inlined. It pays ~1.4 ns at the floor and 50–65
cycles once `VZEROUPPER` and register save/restore are counted, and that cost
is fixed rather than proportional to `n`. At n = 4096 it is a rounding error.
At n = 32 it is most of the runtime — which is why the dispatcher currently
gives up below its threshold and runs a scalar Go loop instead. An inlined
intrinsic pays none of it.

So the plan is additive and measured:

- Build the tier now, behind the build tag, against the same `kernel.Ops`
  contract every other backend implements, so dispatch selects it the same way.
- Benchmark it against **both** the assembly tier and the scalar fallback at
  n = 4, 8, 16, 32, 64, 128, 256. Only the sizes where it wins get wired up.
  The same rule as every other tier in this library: a measurement decides.
- The bit-identity contract still binds. Go's intrinsics have their own
  rounding and NaN behaviour, and they get differentially tested against the
  portable reference like everything else.

When the intrinsics are in the language and need no flag, re-run that
benchmark and move every operation where the intrinsic wins — keeping assembly
everywhere it does not, which on current evidence is everywhere above n ≈ 64
and every architecture Go's intrinsics do not cover.

---

## Releases

Tags so far are `v0.1.0`, `v0.1.1` and `v0.2.0`. Two later numbers are spoken
for, and the reasons are different in kind.

### v1.0.0 — when the gaps close

Not a date and not a feature count. `v1.0.0` says the API is stable and the
numerical contract is one you can build on, so the bar is the things that would
otherwise force a breaking change or a correction later:

1. **Every operation accelerated on amd64, arm64 and riscv64**, or documented at
   its declaration as permanently portable with the reason. The list is now
   `EMA` and the float `CumSum` and `CumProd`, all serial through their own
   output, and it should only shrink by explanation, never by omission. It has
   shrunk twice by explanation: integer `CumProd` is accelerated and exact,
   because two's-complement multiplication is associative and the log-shift
   grouping is therefore not observable; and the float scans gained opt-in
   `FastCumSum` and `FastCumProd`, which drop agreement with a naive loop
   while keeping bit-identity across tiers. See entries 44 and 45 of
   docs/wrong.md — including the measurement that says the integer *sums*
   should stay portable, since a one-cycle add leaves no latency to hide.
   *Nearly there:* the riscv64 transcendental gap is closed as upstream (an
   LLVM cost-model entry for `llvm.is.fpclass`; five source spellings ruled
   out, entry 35 of docs/wrong.md) and re-opens itself when the LLVM version
   changes.
2. ~~The frame-write and stack-budget checks running on every architecture.~~
   **Done.** The root cause was the disassembly parser deleting every arm64
   immediate; with that fixed the checks run on all six architectures, keep
   all 740 arm64 slots with zero false positives, and immediately found ten
   riscv64 kernels shipping over budget.
3. **The two corruptions explained** — the riscv64 compress family is done: a
   stack overflow, 640–2032 bytes against the 512-byte NOSPLIT budget, fixed
   by per-target lane counts and back in service. `countAnyVSX` on ppc64le
   remains open; with the budget check now working there, re-bisecting it is
   the next step.
4. ~~A threshold meta-test.~~ **Done**, and it found four more uncovered
   kernels on its first run, which now have tests.
5. **Verified on real hardware** — now for *wall-clock only*, and the list has
   shrunk twice by doing rather than by waiting.

   The correctness half is done under emulation, including `make test-gates`,
   which runs the suite on a CPU that lacks the vector extension — the one
   configuration `-cpu max` can never produce.

   The instruction-stream half is done too, by `make perf-model`: llvm-mca
   over each kernel's inner loop against the same kernel compiled without
   vectorization, on amd64, arm64, ppc64le and s390x. Nothing modelled below
   1.2x, and the model is checked against measured amd64 to within 5–12% on
   the avx512-versus-avx2 comparison. riscv64 is excluded because RVV's vector
   length is a boot-time property; loong64 because LLVM has no scheduling
   tables for it. See entry 49 of docs/wrong.md.

   The library also has **zero OS-specific source** and builds and vets clean
   for darwin/amd64, darwin/arm64, windows/amd64, windows/arm64 and
   freebsd/amd64. The entire OS-dependent surface is `x/sys/cpu` feature
   detection.

   What genuinely still needs metal: wall-clock under a real memory system,
   which is what dominates a whole-slice kernel at large n and is exactly what
   a model with no memory system cannot tell you. Every GB/s figure in this
   repo is amd64-only, and either those numbers get measured elsewhere or
   every figure gets marked amd64-only before the tag. Apple Silicon would
   cover macOS and arm64 wall-clock at once.

Anything not on that list — FFT, the DSP set, checksums, the rest of the C99
math tail — is a feature and ships in a minor release. Features do not gate a
1.0; contract and verification do.

### v2.0.0 — reserved for Go's intrinsics

Held deliberately empty. Go 1.26 shipped `simd/archsimd` behind
`GOEXPERIMENT=simd`; when the intrinsics are in the language and need no build
flag, the measurements in the Tiers section get re-run and anywhere an
intrinsic beats the assembly, the intrinsic wins. That is a change to what the
package compiles to on every architecture at once and to its minimum Go
version, which is a major-version change even if not one line of the API moves.

Reserving the number now means the intervening releases can stay minor without
anyone having to decide later whether a backend swap counts as breaking.

## Verification

### Real hardware

Everything outside amd64 is verified under emulation. That proves semantics and
proves nothing about timing, so every per-architecture threshold outside amd64
is currently a guess carried over from a machine that does not resemble it.

The thresholds are the part that is actually wrong without this — a kernel that
is correct under qemu is correct on the metal, but a crossover measured on
nothing is a number with no evidence behind it.
