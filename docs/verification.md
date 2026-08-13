# Verification

Every gate in this repository, what it runs, and when it applies. The
[README](../README.md#testing) gives the two-minute version; this page is the
complete list. Command roles come from the Makefile, the CI workflow
(`.github/workflows/ci-local.yml`), and the documentation tests.

## The gates

| Command | What it runs | Role |
|---|---|---|
| `go test ./...` | the full suite on the host's default tier | the first gate; ~3–4 minutes baseline |
| `make verify` | `fmt-check`, `vet`, `test`, `test-purego`, `test-vec`, `test-tiers` | the CI entry gate; everything a normal commit needs |
| `make check-emission` | `tools/simdgen -n`, the generator dry run | per-target emitted/skipped inventory with skip reasons; also the mechanical no-instruction-outside-its-tier check |
| `go test ./internal/tests/docs` | documentation gates (below) | counts, links, API names, changelog tags |
| `make test-purego` | `go test -tags purego` | the portable-only build must be self-consistent |
| `make test-vec` | `GOEXPERIMENT=simd go test` | the amd64-only vector type compiles and passes; skipped where the toolchain lacks the experiment |
| `make test-tiers` | one `go test` per tier the CPU supports (`GOSIMD=<tier>`) | catches a kernel correct on AVX-512 and wrong on SSE2 |
| `make test-race` | `go test -race` | full suite under the race detector |
| `make fuzz` | `FuzzDifferential` (public API) and `FuzzKernelsAgainstReference` (every kernel of every tier vs the reference), `FUZZTIME` seconds each | adversarial bit patterns: signalling NaNs, denormals, exponent boundaries |
| `make test-cross` | arm64, s390x, ppc64le under docker + qemu, hermetic (`--network=none`, `CGO_ENABLED=0`, `-vet=off`, `-short`), `simdinfo -require-accelerated` first | the correctness half of the non-amd64 backends; three bugs shipped past a green amd64 suite and only appeared here — two memory corruptions (a kernel clobbering the goroutine register, kernels writing into the caller's frame through a save area) and a reference that computed different bits on architectures where Go fuses a multiply into an add |
| `make test-riscv64` / `make test-loong64` | qemu-user lanes (recent emulators, explicit `-cpu`), `-require-accelerated` | the two architectures the docker lane cannot reach |
| `make test-gates` | riscv64 with **no** vector extension, no acceleration assertion | the fallback must still pass; the SIGILL mis-gate case |
| `make bench-check` | benchmarks pinned to one L3 domain, `-count 6`, compared with the stored baseline for this GOARCH | fails on anything more than 25% slower; `make bench-update` re-records the baseline and the commit must say why |
| `make hardware-report` / `make hardware-bench` | run the suite on real silicon and write `testdata/hardware/<goos>-<goarch>-<tier>.md` | the real-hardware record; a failing run is the more useful report and the target still writes it |
| `make perf-model` | llvm-mca over each kernel's inner loop (needs clang) | instruction-throughput evidence for architectures this machine cannot time; a model, not a measurement |
| `make codegen` / `make gen-thresholds` | regenerate backends/thresholds | not gates by themselves, but a regeneration must be reviewed and the counts kept in sync |

The emulated lanes are load-bearing, not optional extra assurance: the qemu
argv0 convention (`qemu <path> <argv0> <flags...>`) swallows guest flags, so
`qemu-run-probe` asserts a flag actually reaches the guest before any lane
trusts a PASS, and `-require-accelerated` asserts an accelerated tier was
selected.

## The documentation gates

`internal/tests/docs` binds the prose to the tree:

- every count in the README and `docs/platforms.md` (exported functions,
  generated kernels per architecture, `docs/wrong.md` entries, the status
  version vs the newest CHANGELOG heading, the Go requirement vs go.mod);
- every local link in the active documents resolves;
- every `simd.Name` in the README, tutorial, API guide, and `docs/guide/`
  is an exported function;
- every stable semver tag has an exact `## <tag>` CHANGELOG section.

These run under `go test ./...` like everything else. A documentation-only
change is not done until they pass.

## Measurement discipline

- **Disassemble first, always.** Before proposing a cause for anything slow,
  before writing a variant, before reading a profile delta — build it and
  read the instructions:
  `go test -c -o /tmp/x.test . && go tool objdump -s 'pkg\.functionName' /tmp/x.test`.
  Use gdb when a breakpoint or a live register is needed. Register pressure,
  bounds-check elimination, inlining, and branch layout are only visible in
  the instructions.
- **The code-layout noise floor is 8.3%.** Anything smaller cannot be told
  from nothing by wall-clock, and more samples do not help — layout noise is
  per-build, not per-run. For expected gains below that, compare
  `perf stat -e instructions:u,cycles:u` (layout-independent) and read the
  disassembly.
- **A/B builds are interleaved in one session** and compared on the
  minimum, never across sessions. Run the machine quiet: wait for load
  average under 1.
- **Run gates bare.** Never pipe a gate through `tail` (or anything else)
  without `pipefail`: the pipe reports the last command's status and the
  failure vanishes. This has laundered a red fuzz run, a red README gate,
  and two red bench-check runs into green exits. Run gates bare, or
  `set -o pipefail` first.

## Release gates

A release is the verification set plus the release-specific checks, all
green on the tag commit:

1. `make verify`
2. `make test-cross`, `make test-riscv64`, `make test-loong64`, `make
   test-gates`
3. `make fuzz`
4. `make bench-check` (against the recorded baseline for the release
   machine's GOARCH)
5. `go test ./internal/tests/docs` — including every stable tag present in
   CHANGELOG.md
6. `make check-emission` — counts matching `docs/platforms.md`
7. `git diff --check` — no whitespace errors

The CI workflow runs the same set (`make verify`, the three emulated lanes,
`bench-check`, `fuzz`) on the self-hosted runner; see
[`docs/runner.md`](runner.md). The v1.20.0 release notes and the
publication steps are recorded in
[`docs/plans/2026-08-09-documentation-v1.20-release.md`](plans/2026-08-09-documentation-v1.20-release.md).

## What verification is and is not

Emulation proves execution and semantics, not timing; where the platform
table says *unmeasured*, no wall-clock claim applies. A green suite that
skipped every accelerated tier is indistinguishable from one that tested
them — which is exactly why the emulated lanes assert acceleration before
they believe a PASS, and why a real-hardware report (including a failing
one) is worth more than any amount of emulation.
