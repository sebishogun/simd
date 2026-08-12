# Platforms and generated kernels

The public API is portable Go with optional generated assembly selected at
process startup. A missing kernel never removes an operation: dispatch uses the
portable implementation for that operation and tier.

## Runtime tiers

| architecture | accelerated tiers | correctness | wall-clock |
|---|---|---|---|
| amd64 | sse2, avx2, avx512 | **real hardware** | **real hardware** |
| arm64 | neon | **real hardware** | unmeasured |
| arm64 | sve2 | emulation | unmeasured |
| riscv64 | rvv | emulation | unmeasured |
| ppc64le | vsx | emulation | unmeasured |
| s390x | vx | emulation | unmeasured |
| loong64 | lasx | emulation | unmeasured |

Correctness and timing are separate claims. The differential suite executes
every generated tier against the portable reference and against other tiers.
Emulation can prove that an instruction stream runs and returns the expected
bits; it does not model a processor pipeline, cache hierarchy, or memory
system. Where the table says *unmeasured*, no wall-clock claim in this
repository applies to that tier.

The reports behind the real-hardware cells are committed in
[`testdata/hardware/`](../testdata/hardware/). To add one, run
`make hardware-report` and follow [the contribution guide](../CONTRIBUTING.md#reporting-a-hardware-run).

## Selection and fallback

CPU features are detected once during package initialization. Each operation
reads its own static dispatch table, so the linker can discard operations and
assembly that a program never references. A program that calls three operation
families does not retain every kernel in the repository.

The strongest generated tier supported by the current CPU is selected. If a
particular operation has no kernel at that tier, its generated guard calls the
portable implementation. `GOSIMD` can pin a tier for testing and benchmarking;
`SIMD_DISABLE` can mask tiers out. [`Tier`](https://pkg.go.dev/github.com/sebishogun/simd#Tier),
[`AvailableTiers`](https://pkg.go.dev/github.com/sebishogun/simd#AvailableTiers),
and [`Describe`](https://pkg.go.dev/github.com/sebishogun/simd#Describe) report
the resulting selection.

The `purego` build tag removes generated assembly entirely and exercises the
same portable implementations used for per-operation fallback.

## Kernel coverage

The default build exposes 493 exported functions and the repository contains
6,931 generated kernels across nine accelerated tier targets. Counts below are
logical callable kernels, not the extra `Body` trampolines required by the
s390x ABI. `make check-emission` regenerates the inventory as a dry run.

The skipped column counts manifest slots the generator declined with a stated
reason. Both columns sum across an architecture's tiers, so one operation
emitted for AVX2 and declined for SSE2 appears once in each column.

| | kernels | skipped | dominant reason for skips |
|---|---|---|---|
| amd64 (sse2/avx2/avx512) | 2609 | 101 | LLVM declined to vectorize for a tier |
| arm64 (neon/sve2) | 1681 | 127 | LLVM declined to vectorize for a tier |
| riscv64 (rvv) | 891 | 15 | LLVM declined to vectorize |
| loong64 (lasx) | 728 | 174 | `$fp` ownership and unresolved `.L0` references |
| ppc64le (vsx) | 602 | 300 | `r30`/`r2`, `r0`, frame writes, then LLVM refusals |
| s390x (vx) | 420 | 482 | `r13`, which the Go runtime owns |

These numbers describe generated code, not API coverage. A declined kernel is
slower on that target and never a correctness gap.

## ABI limits

Generated objects begin as C compiled for each target, then the generator
checks and translates them into Go assembly. A C compiler is free to use
registers and stack conventions that Go reserves, so accepting every object
would corrupt the runtime.

- **s390x:** clang frequently allocates `r13`; Go keeps the current goroutine
  there. SystemZ has no working equivalent of `-ffixed-r13` for this pipeline,
  so those kernels are rejected. Accepted s390x kernels use a callable
  trampoline plus a `Body` symbol; the coverage table counts the pair once.
- **ppc64le:** Go owns `r30` and uses `r2` in ways a foreign ELFv2 object cannot
  assume. A nonzero write to `r0` is also rejected because Go's ABI treats it
  as constant zero, including if a signal interrupts the kernel. Writes into
  the caller's frame require a save-area design and are rejected today.
- **loong64:** `$fp` is runtime-owned. Some clang output also addresses `.L0`
  labels that are not liftable constant pools, or leaves branch displacement
  patching to a linker model the generator does not use.

Every accepted kernel is checked for reserved-register use, writes outside its
frame, stack budget, and instructions newer than the tier that gates it. The
conformance suite then executes it against the portable reference.

## Throughput away from amd64

Published wall-clock benchmark numbers are amd64 measurements. For arm64,
ppc64le, and s390x, `make perf-model` runs `llvm-mca` over each inner loop and
compares it with the same source compiled without vectorization. The model is
L1-resident, single-core, and has no real memory system; it is evidence about
instruction throughput, not elapsed time for a whole-slice operation.

RISC-V vector length is a boot-time property, so one static schedule cannot
model RVV generally. LLVM has no usable LoongArch scheduling model for this
purpose. Both remain correctness-tested under emulation and wall-clock
unmeasured.

## Optional Go vector type

The ordinary slice API requires no experiment. On **amd64 only**, Go 1.26's
`GOEXPERIMENT=simd` build enables the escape hatch in `vec.go`, which aliases
the standard library's experimental `simd/archsimd` types and adds four map/zip
helpers. Every other build gets `HasVectorType == false` and `Lanes` returning
zero.

This is for an expression absent from the slice catalog or for the narrow band
where an inlined vector operation avoids the assembly call boundary. It does
not replace the generated slice kernels, which measure faster once a batch is
large enough to cross their dispatch threshold.

## Operating systems

There is no OS-specific kernel source. The OS-dependent surface is CPU feature
detection through `golang.org/x/sys/cpu`. The repository builds and vets for
darwin/amd64, darwin/arm64, windows/amd64, windows/arm64, and freebsd/amd64 in
addition to its Linux execution lanes.

The generated assembly is committed. Consumers need neither clang nor a C
toolchain; those are development dependencies used by `make codegen` and
`make check-emission`.
