# Changelog

## v0.1.0 — unreleased

The first tagged version. Everything below is new; there was nothing before it.

### What this is

SIMD-accelerated slice operations for Go on every architecture with a vector
unit, without cgo. 298 exported functions, 4,721 generated kernels across nine
targets: amd64 sse2/avx2/avx512, arm64 neon/sve2, riscv64 rvv, s390x vx,
loong64 lasx, ppc64le vsx.

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
- **`IndexAll` and `HexDecode` are portable on every architecture.** The first
  needs a compress primitive that is not yet written; the second returns two
  values where the generator's result slot holds one.
- **Not yet built:** compress/expand/filter, sort/argsort, a blocked GEMM
  microkernel, `Gemv`.

### Measured on a Ryzen AI MAX+ 395 (Zen 5, AVX-512)

Against the portable Go build, integer and saturating arithmetic, geomean over
the set: **−86% time, +593% throughput**. `SatAdd` on int8 at n=4096 is −99.1%.

Against `bytes` and `strings` — the harder comparison, since `bytealg` is
assembly on four of the six architectures — geomean **+186%**. `LastIndex` at
n=4096 is +8309%; `IndexAny` at 1 MiB is +1084%.

Against `encoding/base64`: −42% to −63%.

`Fast` against accurate: `FastSin` −45%, `FastExp` −43%.
