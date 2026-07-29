# Changelog

## Unreleased

### Versioning

The release plan is now written down in ROADMAP.md rather than assumed. `v1.0.0`
is gated on five things, of which the largest is that the frame-write and
stack-budget checks currently run on one architecture of six — features are not
on the list. `v2.0.0` is reserved for when Go's intrinsics leave
`GOEXPERIMENT=simd` and the tier measurements get re-run against them.

### Complex reductions

`SumComplex`, `DotComplex` and `DotComplexConj` had no kernel on any
architecture. They now have one on every tier but s390x.

| n = 65536 | accelerated | portable | naive loop |
|---|---|---|---|
| `DotConj` complex64 | 6.89us | 80.5us | |
| `DotConj` complex128 | 15.6us | 55.7us | 155us |
| `Sum` complex128 | 6.08us | 29.4us | |

Ten times the loop a caller writes by hand, which is the comparison that
matters: `sum(a[i] * conj(b[i]))` is the inner product of signal processing and
Go had no fast way to spell it.

The sixteen accumulators the numerical contract requires are held as two
vectors of eight rather than one of sixteen. For `complex128` the obvious
spelling needs six 128-byte vectors live at once, which measured 1120 bytes of
spill against a 512-byte budget and lost the kernel on every amd64 tier.
Splitting each accumulator into halves changes no arithmetic — lane `k%16`
still receives element `k`, and `CombineTree`'s first step *is* adding the two
halves — and it fits everywhere, including SSE2.

### NaN and predicate helpers

`IsNaNInto`, `IsInfInto`, `IsFiniteInto`, `SignInto`, `CountNaN`, `AnyNaN`,
`NanSum` and `NanMean`.

None of them adds a kernel and none needs one: every question reduces to a
comparison the library already vectorizes, so all eight are accelerated on
every architecture at once. `IsNaN` is `NotEqualInto` of a slice against itself
— the IEEE definition read literally, since NaN is the only value not equal to
itself, with no bit-masking and no special case for the payload.

`IsFinite` is deliberately not "not infinite": a NaN fails the comparison
against `+Inf` as unordered rather than as less-than, so it needs a strict `<`,
which excludes both. `sign(NaN)` is NaN rather than zero, and `NanMean` returns
the surviving count alongside the mean, because an average over three points
out of a thousand is not an average.

### Fixed: the regression gate could not gate anything

`benchcheck` compared the *median* of six samples against a flat 25% threshold,
on bodies running at 6 to 15 ns/op. Two consecutive runs on an idle machine,
same commit and same binary, reported sixteen regressions and then five, with
**zero overlap** — every one a transient that reached the median.

It now compares *minimums*, because benchmark interference is one-sided: a
frequency drop or a migration can only make a run slower, so the samples are
the true cost plus a non-negative contaminant and the minimum is the
maximum-likelihood estimate of the true cost. It cannot hide a real regression,
since a real regression raises every sample including the best one. Sixteen of
the twenty-one false positives disappear.

### Four operations leave the portable path on x86

`HexDecode`, `Base64Decode`, `Median` and `Quantile` were four of the eight
operations still running plain Go on amd64. Each was blocked on something
different, and only one of the four was blocked on what the roadmap said.

**HexDecode** genuinely needed the two-value return. It reports a count and a
validity flag, and the generator's result slot held one value, so it was
portable on every architecture for a reason that had nothing to do with the
hardware. The kernel validates a block without branching and only decodes one
that is wholly valid, dropping to a scalar tail to find the exact offset of a
bad character. Against `encoding/hex` at 1 MiB: **44.2us versus 660us**, 15x,
at 22 GB/s.

**Base64Decode** returns one value and was never blocked on that at all. It was
excluded for spilling 576 bytes on AVX2 and 704 on AVX-512, past the 512-byte
budget a NOSPLIT function has. Four bytes in and three out makes LLVM build a
shuffle tree whose cost grows faster than the vector width, and left alone it
picked a width that spilled. Pinning that width — 64 where `__AVX512F__` is
defined, 32 elsewhere, because 64 everywhere would cost the AVX2 and ppc64le
tiers — gives **43.7us at 1 MiB against 894us portable and 399us for
`encoding/base64`**: 20.5x and 9.1x.

**Median and Quantile** now run a quickselect around the accelerated partition.
float64, taking the median of nine runs:

| n | accelerated | portable | via `slices.Sort` |
|---|---|---|---|
| 4096 | 10.4us | 15.0us | 119us |
| 65536 | 157us | 762us | 3.80ms |
| 1048576 | 3.12ms | 13.4ms | 76.7ms |

New `MedianInto` and `QuantileInto` take the scratch buffer from the caller and
allocate nothing, in the same relationship `SortInto` has to `Sort`.

That leaves `EMA`, `CumSum` and `CumProd` as the only operations still portable
on amd64, and all three are permanently so: each is serial through its own
output, and the contract forbids the reassociation that would break the
dependency.

### Fixed

- **Signed zero in `Sort` is now documented.** The accelerated and portable
  paths can place `-0` and `+0` differently — 848 of 4096 positions on a slice
  containing both — because `-0 < +0` is false and the two therefore tie under
  the `<` that defines the order. Every such output is a correct sort and every
  differing pair is `==`. Making them agree needs a comparator closure, already
  measured at 2.5x slower. `Median` and `Quantile` inherit the caveat.

## v0.2.0 — 2026-07-28

### ppc64le: 281 kernels become 468

The largest coverage gain in the library, and it came from discarding a
constraint that was never real.

clang reaches its constants on ppc64le through `r2`, the TOC pointer, which Go
does not maintain for these objects. Power9 has no PC-relative data addressing,
so reaching an appended pool appeared to require either `bcl`/`mflr` — which
clobbers the link register and wants a save slot in a protected zone the kernels
already use down to −256 bytes — or a dependency on whatever `r2` held on entry,
which is unsafe under `-shared`.

Neither is necessary. **Go's own assembler materialises a symbol address in two
instructions with no TOC involvement**, because Go builds non-PIE and the
address is a link-time constant. That was settled by building and running a
probe under emulation rather than by reasoning about it.

So the pool becomes a standalone `GLOBL`, `R2` is pointed at it with one `MOVD`
in the prologue, clang's two global-entry instructions are replaced with nops in
place, and every TOC16 immediate is rewritten as an offset from the pool base.
Two other checks were in the way, and both were wrong rather than conservative:
`.TOC.` was counted as an undefined *call* when it is a linker-defined data
anchor, and `r2` was rejected outright when its only uses here are clang's own
prologue and reads.

One kernel of 469 corrupts memory with the rewrite enabled and is not
registered, so `CountAny` keeps the portable path it already had on this
architecture. It was bisected to `countAnyVSX` in fourteen runs and its
addressing then verified correct by hand, so the fault is something else about
that kernel — see the note at its skip.

Verified on emulated ppc64le with 3.5 million clean fuzz executions.

### Also

- **Sorting**: `Sort`, `SortInto`, `Argsort`, `PartitionInto`, `SortedIndex`.
- **N-ary arithmetic**: `AddAll`, `MulAll`.

## v0.1.1 — 2026-07-28

**Fixes a crash.** `PartitionInto` dereferenced a nil function pointer on every
architecture without a hardware compress instruction — s390x, ppc64le, riscv64,
loong64, and amd64 below AVX-512 or arm64 without SVE2. That is most machines,
and it was a panic on the first call rather than a wrong answer. Anyone on
v0.1.0 who touches the sorting API should take this.

The cause was two things at once: the portable implementation was never
registered in the reference, because a scripted edit stopped matching after
gofmt reflowed the surrounding lines and failed silently; and `PartitionInto`
called through the slot unguarded, although `CompressInto` — the same
arrangement — already had the guard. Both are fixed, and the emulated s390x
lane is what found it. See entry 12 of [docs/wrong.md](docs/wrong.md).

### Added since v0.1.0

- **Sorting**: `Sort`, `SortInto` (allocation-free), `Argsort`, `PartitionInto`
  and `SortedIndex`. A quicksort around a compress-based partition, 19–27%
  faster than `slices.Sort` above 16K elements on float64. NaN sorts last, as
  in `Median` and `Quantile`, which differs from `slices.Sort`.
- **N-ary arithmetic**: `AddAll` and `MulAll` over any number of slices in a
  single pass, with the element type enforced by the compiler.

## v0.1.0 — 2026-07-28

The first tagged version. Everything below is new; there was nothing before it.

### What this is

SIMD-accelerated slice operations for Go on every architecture with a vector
unit, without cgo. 309 exported functions, 5,247 generated kernels across nine
targets: amd64 sse2/avx2/avx512 (1664), arm64 neon/sve2 (1121), s390x vx (614),
riscv64 rvv (558), loong64 lasx (506), ppc64le vsx (281).

### What is covered by compatibility

The **exported API of the root package** — names, signatures and documented
semantics — from this tag onward.

Explicitly **not** covered, and it matters here more than it usually would:

- **Which kernels exist on which architecture.** Coverage is uneven and will
  move as ABI problems are solved. A function whose kernel is missing on a
  target runs the portable implementation, so this is a performance property,
  never a correctness one.
- **The measured numbers.** They are from one machine and will differ on
  yours.
- **`internal/`, `tools/` and `csrc/`.** The generator, the reference and the
  kernel sources are implementation.

### The accuracy contract

Every operation is bit-identical on every instruction set, including for NaN
payloads, ±Inf, ±0 and denormals. Two exceptions, both opt-in by name: the
transcendentals guarantee a stated ULP bound rather than bit identity, and
`Fast*` promises 3.5 ULP and gives up agreement between architectures.

### Known limits

- **Nothing outside amd64 has run on real hardware.** Every other architecture
  is verified under emulation, which proves semantics and proves nothing about
  timing.
- **ppc64le (281 kernels) and s390x (614) are partial**, both because clang
  uses a register the Go runtime owns and no compiler flag stops it. ppc64le
  additionally reaches its constants through the TOC pointer, which Go does not
  maintain for these objects.
- **`Compress` and `IndexAll` are accelerated on AVX-512 and SVE2 only**, and
  permanently so. Those are the two instruction sets with a compress
  instruction, and the operation cannot be vectorized without one: where each
  element lands depends on how many earlier ones matched, which is a real
  loop-carried dependency rather than a compiler shortcoming. The other seven
  targets run the portable loop, which is what their compilers would emit
  anyway.
- **`HexDecode` is portable everywhere**, because it returns two values where
  the generator's result slot holds one.
- **Not yet built:** sort/argsort, and packed-panel
  cache blocking above the GEMM microkernel. See [ROADMAP.md](ROADMAP.md).

### Where to start

[`example_test.go`](example_test.go) has a runnable example for every operation
— checked by `go test`, so none of them can drift — and
[`docs/examples/`](docs/examples/) has complete programs. The README opens with
a table indexed by what you are trying to do rather than by what the operation
is called.

[`docs/wrong.md`](docs/wrong.md) is the twenty-two things that turned out not to
be true, which is the part of this project most worth borrowing.

### One deliberate semantic choice worth stating

`MatMul` does not skip zeros in `a`. An earlier draft did, and it is not the
free optimization it looks like: under IEEE 754 a zero times an infinity is a
NaN, and skipping suppresses it. BLAS does not skip, numpy does not skip, and
the standard says what the answer is — so neither does this. It is also what
makes the register-blocked microkernel possible, since in a tile that test
would guard a single fused multiply-add rather than a whole row.

### On Go's own SIMD intrinsics

Go 1.26 shipped them behind `GOEXPERIMENT=simd`, targeted at the language in
1.27. They do not replace this library and are not a competitor to it: they
cover amd64 and arm64 only, they do not reach SVE2 or RVV, and a library cannot
require its consumers to set a `GOEXPERIMENT`.

They do win in one band, and will keep winning there. An assembly kernel cannot
be inlined, so it pays a fixed call boundary — ~1.4 ns at the floor, 50–65
cycles once `VZEROUPPER` and register save/restore are counted. Above n ≈ 128
that is a rounding error. Below n ≈ 64 it is most of the runtime, which is why
the dispatcher gives up and runs a scalar loop there. An inlined intrinsic pays
none of it.

The plan is therefore additive: build the tier behind the experiment flag,
benchmark it against both the assembly tier and the scalar fallback, and wire
up only the sizes where it wins. When it needs no flag, re-measure and adopt it
wherever the number says so. The bit-identity contract binds it like every
other tier.

### Measured on a Ryzen AI MAX+ 395 (Zen 5, AVX-512)

Against the portable Go build, integer and saturating arithmetic, geomean over
the set: **−86% time, +593% throughput**. `SatAdd` on int8 at n=4096 is −99.1%.

Against `bytes` and `strings` — the harder comparison, since `bytealg` is
assembly on four of the six architectures — geomean **+186%**. `LastIndex` at
n=4096 is +8309%; `IndexAny` at 1 MiB is +1084%.

Against `encoding/base64`: −42% to −63%.

`MatMulInto` against the naive kernel it replaced: **−60% to −86%** depending
on size, reaching 260 GFLOP/s on f32 — about 90% of this core's single-thread
AVX-512 peak. `GemvInto` is new, and is bit-identical to `Dot` per row.

`CompressInto` against the scalar filter loop, geomean **−51%**; at 1 M
elements and 50% match density, **−93%** (1.29 GiB/s → 19.3 GiB/s).

`Fast` against accurate: `FastSin` −45%, `FastExp` −43%.
