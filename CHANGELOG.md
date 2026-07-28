# Changelog

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
unit, without cgo. 302 exported functions, 4,744 generated kernels across nine
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

[`docs/wrong.md`](docs/wrong.md) is the twenty-one things that turned out not to
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
