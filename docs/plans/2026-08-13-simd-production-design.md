# simd production design record

> **Status:** a design record for the *existing* production library and the
> governance future work must meet. Nothing here describes a new shipped API
> or a change to the current v1.20.0 contract — every item below is either
> already true of the shipped tree (with the source that makes it true) or a
> rule for work that is not yet built. Future implementation work is staged
> separately in [`2026-08-13-simd-production.md`](2026-08-13-simd-production.md).

**Goal:** one page that states the production architecture of
`github.com/sebishogun/simd` as it ships, and the evidence bar a new kernel
or a contract change must clear.

## 1. Shipped v1 package boundary

The module is `github.com/sebishogun/simd`, one package `simd` of
whole-slice primitives: arithmetic, reductions, text and byte scanning,
sorting and sets, columnar codecs, matrix/DSP, and random fills. The v1.20.0
tree exports the function and generated-kernel counts that
[`docs/platforms.md`](../platforms.md) states — the documentation tests hold
them to the sources — over the nine tier targets, with a portable Go path
covering every operation.

Non-goals, stated in the README and enforced by the layout: not a BLAS,
tensor library, dataframe, or autodiff framework; no ownership of data
models, allocation policy, or control flow; no cgo; no C toolchain for
consumers; no JIT; no build tag required for the ordinary build; one
transitive dependency (`golang.org/x/sys`, CPU feature detection).

## 2. Portable reference fallback

`internal/ref` is the specification, not a fallback: every generated kernel
is differentially tested against it, and where the two disagree the kernel
is wrong (unless the bug is in the reference — which has happened). The
reference is ordinary Go linked into every consumer regardless; the
per-tier assembly is what the static dispatch tables keep out. The `purego`
build tag removes the assembly entirely and exercises the same portable
implementations used for per-operation fallback.

## 3. Generated, committed assembly

Kernels are written once in C under `csrc/`, compiled per instruction set
by the generator in `tools/` (a separate module), verified, and committed
as Plan 9 assembly under `internal/<arch>/`. Consumers need no C toolchain.
`make codegen` regenerates; `make check-emission` is the dry-run inventory
that also refuses instructions outside the tier gating each file. A kernel
that fails verification is dropped with the reason recorded — that is the
expected state on the ABI-constrained targets, and a declined kernel is a
documented tradeoff, never a correctness gap.

## 4. Platform and tier model

Nine accelerated tier targets across six architectures (sse2/avx2/avx512,
neon/sve2, rvv, vsx, vx, lasx). Correctness and wall-clock are separate
claims: correctness is proven per tier by the differential suite, under
emulation where the host cannot run it; published wall-clock figures are
amd64 measurements, and tiers marked *unmeasured* carry no speed claim.
Selection is `internal/cpu` + `dispatch.go`: the strongest tier with tables
for this build wins, `GOSIMD` pins an exact tier (can only select down),
`SIMD_DISABLE` masks tiers out, and the portable path is always available.
ABI walls (s390x `r13`, ppc64le `r30`/`r2`/`r0`, loong64 `$fp`, frame
writes) are recorded per architecture in `docs/platforms.md` and in the
roadmap, not silently worked around.

## 5. API stability

For the life of v1, every exported function keeps its name, signature, and
meaning, and so does the numerical contract: the six rules in
`internal/kernel/kernel.go` (elementwise bit identity, integer reductions,
the fixed 16-accumulator float reduction shape, no FMA in `Dot`, `Fast*`
bounds, transcendental ULP bounds). CHANGELOG.md states exactly what
compatibility covers and excludes. A breaking change is a v2 decision, and
v2.0.0 is reserved in ROADMAP.md for Go's intrinsics.

## 6. Allocation, sizing, aliasing, and length conventions

Kernel calls do not allocate; `Into` forms are caller-owned fast paths with
explicit output/workspace. Convenience functions that return slices, build
plans, or grow append destinations can allocate and document it.
Elementwise operations process the shortest slice; operations whose output
length differs from the input declare two lengths in the manifest so the
guard cannot clamp silently; output-shaped operations state exact capacity
in their comments. Partial overlap is operation-specific and the function
comment is the contract. Errors are counts and negative results
(`LZ4BlockDecode` returns −1 on malformed input), plus documented panics
where Go semantics demand them (zero-divisor division). The implementation
level is `docs/lld/api-and-memory.md`; the user reference is `docs/api.md`.

## 7. Operation-level dispatch and link behavior

Each exported function reads its own static table (`dispatch_tables_<arch>.go`)
at a tier index assigned once at init. Static tables are what let the
linker drop every operation a program never calls, assembly included; a
consumer using three operations carries three operations' kernels. The
reference base is always linked; per-element-type partials are merged
lazily so a program that never touches a type never links its kernels. The
only type switch lives in `dispatch.go`, keeping every exported function a
one-liner.

## 8. Evidence required for a new kernel

A kernel ships only with all five forms of evidence, per
`docs/kernels.md` and the verification gates:

1. **Source** — the C under `csrc/` plus a manifest entry in
   `tools/simdgen/kernels/kernels.go`.
2. **Disassembly** — the generated Plan 9 committed under
   `internal/<arch>/`, statically checked (instructions within tier, stack
   budget, reserved registers, respelled encodings) by `internal/asmcheck`
   and `make check-emission`.
3. **Correctness** — differential tests against `internal/ref` in
   `internal/conformance/`, plus fuzz over adversarial bit patterns.
4. **Cross-tier** — the suite per tier on the host (`make test-tiers`) and
   under emulation (`make test-cross`, `test-riscv64`, `test-loong64`,
   `test-gates`), each lane asserting an accelerated tier was selected.
5. **Benchmark** — `bench_*_test.go` comparing against what a caller would
   otherwise write, kept honest by `make bench-check` against the stored
   baseline; a kernel that does not beat the chain it replaces does not
   ship.

Documentation must land with the kernel: the API table, the guides, and the
source-backed counts that `internal/tests/docs` enforces.

## 9. Governance

- ROADMAP.md is the canonical roadmap; nothing in it is shipped until it is
  in code, tests, and changelog, and nothing in it is a date promise.
- `docs/wrong.md` records measurements that argued against changes; a
  finding that cost a measurement belongs there whether or not code changed,
  and entries are never rewritten.
- Future work is planned in `docs/plans/` and executed task-by-task with
  the executing-plans workflow; this record is the contract that work must
  not violate.
- A change to any of the properties above — the module boundary, the
  fallback, the generation pipeline, the tier model, the API promise, the
  dispatch/link behavior — is an architecture change and must be recorded
  here or in `docs/research/` before it is built.
