# LLD: Generation and platforms

The C-to-assembly pipeline and what each platform/tier claim means. The
source-backed inventory, skip counts, and ABI limits live in
[`docs/platforms.md`](../platforms.md); the add-a-kernel walkthrough is
[`docs/kernels.md`](../kernels.md).

## Generator inputs

The generator (`tools/simdgen`, its own module) takes three inputs:

1. **Kernel sources**: `csrc/*.c`, one file per group (`arith.c`, `bytes.c`,
   `sort.c`, …), with `fold.h`, `minmax.h`, `poly.h`, and the ABI wrapper
   headers. Kernels are plain C functions: `__restrict` pointers, signed
   loop counters, no calls, no data-dependent early exits.
2. **The manifest**: `tools/simdgen/kernels/kernels.go`, one entry per
   kernel mapping the C symbol to the Go name, the `kernel.Set` field, the
   reference function, Go parameters vs C arguments (`base`/`lenOf`/`val`/
   `out`), threshold, `RefWhen`, `UnclampedDst`, `AllowScalar`, and
   `SkipOn` architecture lists.
3. **Target definitions**: `tools/simdgen/target/`, one per tier —
   clang target triple, CPU features, and the Go assembler constraints for
   that architecture.

`make check-emission` is the dry run: it reports what each target would
emit or skip and the reason for every skip, and it refuses instructions
outside the tier that gates the file. `make codegen` actually regenerates.

## What the generator emits

For each accepted kernel, per tier:

- `internal/<arch>/<group>_<tier>_<arch>.s` — Plan 9 assembly, committed.
  Every `.s` names the C file it came from and the target it was built for;
  a `// func name(args)` marker precedes each logical kernel (s390x pairs
  every callable entry with a `Body` trampoline, and the counts count the
  pair once).
- `internal/<arch>/<group>_register_<tier>_<arch>.go` — the guard wrappers:
  threshold check, length clamp, `RefWhen`, and the fallthrough to the
  named `ref` function.
- `dispatch_tables_<arch>.go` — one static table per operation, indexed by
  tier, which is what makes unused operations and their assembly
  droppable by the linker.
- `kernel_thresholds_<arch>.go` — the per-operation element thresholds,
  generated from the same manifest the guards come from, with a test
  holding them to agreement.

The generator also verifies every object before emitting: reserved
registers, caller-frame writes, stack budget, instruction tier, and (on
ppc64le) the pool-prologue and TOC handling. A kernel that fails is
dropped with the reason recorded, and the portable implementation stands
in — that is the expected state for a large fraction of kernels on the
four ABI-constrained targets.

## Regeneration workflow

```
make codegen          # regenerate every backend and the thresholds (needs clang, llvm-objdump)
make check-emission   # dry-run inventory: emitted/skipped counts and reasons
make gen-thresholds   # thresholds alone, no clang needed
```

- Never edit a generated file by hand; regenerate and commit the diff.
- After a regeneration that changes counts, `docs/platforms.md` must match
  the sources — the documentation tests enforce it.
- Review the generated diff as part of the change: a `.s` diff that is
  larger than the C change it came from is a codegen warning sign, and
  `TestRespelledEncodingsMatchClang` pins the mnemonic respellings.

## Architecture directories and tier support

| GOARCH | tiers | directory |
|---|---|---|
| amd64 | sse2, avx2, avx512 | `internal/amd64/` |
| arm64 | neon, sve2 | `internal/arm64/` |
| riscv64 | rvv | `internal/riscv64/` |
| ppc64le | vsx | `internal/ppc64le/` |
| s390x | vx | `internal/s390x/` |
| loong64 | lasx | `internal/loong64/` |

The per-architecture kernel/skipped counts and the dominant skip reasons
(`r13`, `r30`/`r2`/`r0`, `$fp`, frame writes, LLVM refusals) are in
[`docs/platforms.md`](../platforms.md). Each accepted kernel is gated on
the tier it was built for; `GOSIMD`/`SIMD_DISABLE`/`Tier()` report the
selection, and `simdinfo -require-accelerated` is what the emulated lanes
assert before accepting a pass.

## Real hardware versus emulation

- Correctness: every generated tier is differentially tested against the
  portable reference, under emulation where the host cannot run it. The
  emulated lanes run a CPU with the target's vector extension switched on
  and assert an accelerated tier was selected (`-require-accelerated`),
  because a suite that skipped every accelerated tier is green and reads
  like one that tested them.
- `make test-gates` runs the suite on a CPU *without* the vector extension,
  which is where a mis-gated kernel shows up as SIGILL — the one
  configuration `-cpu max` can never produce.
- Wall-clock: published figures are amd64 measurements. Where the platform
  table says *unmeasured*, no speed claim in the repository applies to that
  tier. `make perf-model` (llvm-mca over each inner loop) is evidence about
  instruction throughput, not elapsed time.
- Real-hardware runs are committed per machine in `testdata/hardware/`
  (`make hardware-report`); failures are the more useful reports.

## purego behavior

The `purego` build tag removes the generated assembly entirely
(`backend_purego.go` forces tier 0). It exercises the same portable
implementations used for per-operation fallback and serves three audiences:
people auditing what actually executes, people on a toolchain or platform
where the assembly cannot be trusted, and the differential tests, which
confirm the reference is self-consistent before anything is compared
against it.

## Reviewing generated diffs

A kernel change lands as a C diff plus the generated output. Checklist:

1. `make codegen`, then read the `.s` diff — the instructions, not just the
   counts.
2. `make check-emission` — the skip reasons for the architectures/tiers
   that declined.
3. `go test ./...` and `make verify` — correctness on the host's tiers.
4. `make test-cross`, `make test-riscv64`, `make test-loong64`, `make
   test-gates` — the emulated tiers.
5. `make bench-check` — the performance gate against the stored baseline;
   update the baseline only with a stated reason.
6. Keep `docs/platforms.md` counts and the kernel-coverage table in step
   with the sources; the documentation tests will refuse drift.
