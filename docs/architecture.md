# Architecture

What the repository is made of and how a call reaches machine code. The
[README](../README.md) is the front page; this page is the map. `docs/lld/`
holds the low-level details this page points at.

## What ships

One Go module, `github.com/sebishogun/simd`, containing one package `simd`:
the public slice API. At the v1.20.0 tip it exports 493 functions over
6,931 generated kernels for nine tier targets, plus a portable Go path that
covers every operation on every target. The current state of the shipped
library is the design record in
[`docs/plans/2026-08-13-simd-production-design.md`](plans/2026-08-13-simd-production-design.md).

Consumers `go get` the module and nothing else: no cgo, no C toolchain, no
JIT, and no build tag for the ordinary build. The generated assembly is
committed to the repository. The only runtime dependency is
`golang.org/x/sys`, for CPU feature detection.

## Components

| Component | Location | Role |
|---|---|---|
| public API | root package, one file per topic | exported functions over slices |
| dispatch tables | `dispatch_tables_<arch>.go` (generated) | one static table per operation, indexed by the selected tier |
| kernel contract | `internal/kernel/kernel.go` | the `Ops`/`Bytes`/`Convert`/`Complex`/`Set` shapes and the six numerical rules |
| portable reference | `internal/ref/` | the specification every kernel is differentially tested against |
| CPU detection | `internal/cpu/` | tier detection via `x/sys/cpu`, `GOSIMD`/`SIMD_DISABLE` handling |
| generated kernels | `internal/<arch>/*.s` (generated) | Plan 9 assembly, committed |
| generator | `tools/simdgen/` | C → object → verified Plan 9 assembly, tables, thresholds |
| kernel sources | `csrc/*.c` | the C that is the source of truth for every fast path |
| conformance | `internal/conformance/` | differential suite: every tier against the reference and against each other |
| asmcheck | `internal/asmcheck/` | static checks on the committed assembly |
| benchmarks | `bench_*_test.go`, `internal/benchmarks/` | measurements against what a caller would otherwise write |
| public tests | `internal/tests/<topic>/` | the public-API suite, including the documentation gates |

## Call path

```
caller
  └ exported wrapper (package simd)
      └ static per-operation table  (dispatch_tables_<arch>.go)
          └ generated guard         (threshold check, length clamp, RefWhen)
              ├ generated kernel     (internal/<arch>/*.s)
              └ internal/ref         (portable implementation)
```

- The tier is selected once during package initialization
  (`internal/cpu` + `dispatch.go`) and never changes; `tierIdx` is written
  once and read without synchronization.
- Each exported function indexes its own static table, so the linker keeps
  only the operations a program actually calls — assembly included. A binary
  using three operation families does not retain all 6,931 kernels.
- The guard is generated per operation. Below the per-kernel element
  threshold it calls the portable reference directly, because crossing into
  assembly costs about 1.4 ns and cannot be inlined. It also clamps to the
  shortest slice and evaluates `RefWhen` conditions before the kernel runs.
- Numeric groups reach their kernels through a per-element-type merge of the
  reference with a tier's partial `Ops`, built lazily on first use
  (`opsCache` in `dispatch.go`). The only type switch lives there, so the
  public wrappers stay one-liners and a caller that never touches a type
  never links its kernels.
- The reference itself (`refBase`) is a few hundred kilobytes of ordinary Go
  and is linked into every consumer regardless; the per-tier assembly is what
  the tables keep out.

The low-level details are in
[`docs/lld/kernels-and-dispatch.md`](lld/kernels-and-dispatch.md).

## Generator source of truth

```
csrc/*.c  +  manifest (tools/simdgen/kernels/kernels.go)
      │ clang -O3 --target=<tier>   (nine targets)
      ▼
  object file
      │ tools/simdgen: verify (reserved registers, frame budget,
      │                 tier-correct instructions) then emit
      ▼
internal/<arch>/<name>_<tier>_<arch>.s        ← committed Plan 9
internal/<arch>/..._register_...go            ← guard wrappers
dispatch_tables_<arch>.go                     ← per-operation tables
kernel_thresholds_<arch>.go                   ← per-operation thresholds
```

- **The C is the source; the assembly is the output.** Every generated `.s`
  names the C file it came from and the target it was built for, and none of
  them should be edited by hand — `make codegen` regenerates them.
- `make check-emission` is the dry run: it reports per-target emitted/skipped
  counts and the reason for every skip without needing the assembly committed.
- The generator lives in its own module (`tools/go.mod`) so it never becomes
  a dependency of anyone using the library. Regenerating needs clang and
  `llvm-objdump`; consuming does not.
- A kernel that fails verification is not an error: it is dropped and the
  portable implementation stands in, with the reason recorded by
  `check-emission`. Shipping a kernel on five of six targets is normal.

Full pipeline and platform detail:
[`docs/lld/generation-and-platforms.md`](lld/generation-and-platforms.md),
[`docs/platforms.md`](platforms.md), and the add-a-kernel walkthrough in
[`docs/kernels.md`](kernels.md).

## Package and module boundaries

- `github.com/sebishogun/simd` — the public module. Root package `simd`
  (public API) plus `internal/*` (kernel contract, reference, detection,
  tests). `internal/` packages are not importable outside the module.
- `tools/` — a nested module containing the generator, `benchcheck`, and
  `perfmodel`. Not imported by the library.
- `csrc/` — C sources compiled by the generator, not by Go. There is no cgo
  anywhere; the assembly is the Go toolchain's input.
- `cmd/simdinfo`, `cmd/site` — development tools in the main module
  (`simdinfo` reports the selected tier; `site` runs the tutorial
  comparisons live).
- `testdata/bench/` — recorded benchmark baselines per GOARCH.
  `testdata/hardware/` — one report per machine that has run on real
  silicon.
- `docs/` — references, guides, records, and plans; `docs/wrong.md` and
  `docs/research/` are historical and are not rewritten.

## Claims policy

The repository claims no cgo, no C toolchain for consumers, no JIT, and no
build tags for the ordinary build — the source that supports those claims is
the committed assembly, the `backend_purego.go`/`backend_asm.go` build
constraints, and the Makefile. The optional `purego` build tag and the
amd64-only `GOEXPERIMENT=simd` vector type (`vec.go`) are the documented
exceptions. Anything that would add a new build-time or runtime mechanism to
the ordinary build is a change to the architecture, not a kernel addition,
and belongs in the production design record before it belongs in code.
