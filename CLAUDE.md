# Working on this project

## Disassemble first, always

Before proposing a cause for anything slow, before writing a variant, before
reading a profile delta — **build it and read the instructions**.

```
go test -c -o /tmp/x.test .
go tool objdump -s 'pkg\.functionName' /tmp/x.test | less
```

Use gdb when a breakpoint or a live register is needed, and both together when
that helps. Go compiles in seconds; there is no cost to looking, and every guess
that skips it costs a build-measure-revert cycle and risks a wrong conclusion
landing in `docs/wrong.md` as fact.

What the disassembly says that nothing else does:

- **Register pressure.** A large stack frame with the loop counter or a flag
  spilled and reloaded per iteration. No performance counter reports this. It
  was the real cause of an 18% gap that three rounds of counter-guessing —
  cache footprint, key copying, per-field bookkeeping — all missed.
- **Whether a bounds check was eliminated**, and whether an index multiply is
  a shift or a multiply.
- **Whether a call was inlined**, and whether `append(b, s...)` became inline
  stores or a `memmove` call.
- **Which branch the compiler laid out as fallthrough.**

## Benchmarks

The code-layout noise floor here is **8.3%**. Anything smaller cannot be told
from nothing by wall-clock, and more samples do not help — layout noise is
per-build, not per-run. When a change is expected to be worth less than that:

- compare **instructions retired** and **cycles** with `perf stat -e
  instructions:u,cycles:u`, which are layout-independent;
- and read the disassembly, which is the only thing that explains *why*.

A/B builds must be **interleaved** in one session and compared on the minimum,
never across sessions. Run the machine quiet: wait for load average under 1.

**Never pipe a gate through `tail`** (or anything else) without `pipefail`:
the pipe reports the last command's status and the failure vanishes. This has
now laundered a red fuzz run, a red README gate, and two red bench-check runs
into green exits. Run gates bare, or `set -o pipefail` first.

## The record

`docs/wrong.md` in each repository holds measurements that argued against
changes, including changes that were then reverted. A finding that cost a
measurement belongs there whether or not any code changed — the entry is the
deliverable. It is a historical record: never rewrite entries to read like
current reference pages.

## Product boundary and non-goals

This repository ships one Go package of whole-slice SIMD primitives for
numeric, text, and columnar work: runtime-selected generated kernels plus a
portable Go path for every operation. It is not a BLAS, tensor library,
dataframe, or autodiff framework; ownership, data models, and control flow
stay with the caller. No cgo, no C toolchain for consumers, no JIT, no build
tags required for the ordinary build — the optional `purego` tag exists for
portable-only builds and auditing. The one transitive dependency is
`golang.org/x/sys`, for CPU feature detection. Projects built on the library
live in their own repositories.

## Current shipped status

The current release is **v1.20.0**; the exported-function and
generated-kernel counts are the machine-checked ones in README.md and
docs/platforms.md. The v1 API is stable — every exported
function keeps its name, signature, and meaning, and so does the numerical
contract in `internal/kernel/kernel.go`. The README describes shipped v1
behavior only.

## Required reading

For a task in this repository, read in this order before touching anything:

1. `README.md`
2. `docs/architecture.md`
3. `docs/lld/api-and-memory.md`
4. `docs/lld/kernels-and-dispatch.md`
5. `docs/lld/generation-and-platforms.md`
6. `ROADMAP.md`
7. `docs/verification.md`
8. `docs/wrong.md`
9. `docs/plans/2026-08-13-simd-production-design.md`
10. `docs/plans/2026-08-13-simd-production.md`

Then `docs/api.md` before touching the public API, `docs/kernels.md` before
adding a kernel, `docs/runner.md` before touching CI, and `docs/platforms.md`
before making a platform claim. The rules in this file
and `AGENTS.md` agree; when they conflict, the stricter one wins.

## Package, ownership, and compatibility rules

- The root package `simd` is the public API, one file per topic. Exported
  names are a v1 promise: no renaming, repurposing, or re-signing, and no
  change to the numerical contract.
- `internal/kernel` is the contract every backend implements; `internal/ref`
  is the portable specification every kernel is tested against.
- `csrc/` is the kernel source of truth; `internal/<arch>/*.s` and the
  generated `dispatch_tables_<arch>.go` / `kernel_thresholds_<arch>.go` are
  output of `make codegen`, never hand-edited. `tools/` is a separate module
  that must never become a consumer dependency.
- Documentation counts are test-enforced, and local links are test-enforced
  for the active document set listed in `internal/tests/docs/links_test.go`
  (README, CONTRIBUTING, ROADMAP, and the main references/guides). Links in
  other documents — the LLDs, plans, and records — are not machine-gated:
  check them by hand. Keep every count and every local link true either way.

## Verification and release gates

Before any commit: `go test ./...`, `make verify`, `make check-emission`,
`git diff --check`. A release additionally needs `make test-cross`,
`make test-riscv64`, `make test-loong64`, `make test-gates`, `make fuzz`,
and `make bench-check`, with the docs gates green and every stable tag in
CHANGELOG.md. All commands and roles are listed in `docs/verification.md`.

## Roadmap-not-shipped rule

ROADMAP.md is the canonical roadmap; nothing in it is shipped until it is in
the code, the tests, and the changelog. Do not write prose that makes an open
item sound built, and do not create a parallel roadmap. Future work lives in
`docs/plans/`.
