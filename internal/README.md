# internal — implementation

Nothing in here is importable from outside the module, and nothing in here is
covered by the compatibility promise in [CHANGELOG.md](../CHANGELOG.md). The
public API is the `simd` package at the repository root.

## Generated assembly

| directory | tiers | generated from |
|---|---|---|
| [`amd64/`](amd64) | sse2, avx2, avx512 | [`csrc/`](../csrc) |
| [`arm64/`](arm64) | neon, sve2 | [`csrc/`](../csrc) |
| [`riscv64/`](riscv64) | rvv | [`csrc/`](../csrc) |
| [`ppc64le/`](ppc64le) | vsx | [`csrc/`](../csrc) |
| [`s390x/`](s390x) | vx | [`csrc/`](../csrc) |
| [`loong64/`](loong64) | lasx | [`csrc/`](../csrc) |

Each holds `.s` files — Plan 9 assembly, committed so that consumers need no C
toolchain — and the `.go` files declaring them. Every one opens with the
generator, the C source it came from, and the target it was built for. **Do not
edit them by hand**; run `make codegen`.

## Everything else

| package | what it is |
|---|---|
| [`ref/`](ref) | the portable Go implementation. Every kernel is differential-tested against it, and it runs wherever a kernel could not be generated. |
| [`kernel/`](kernel) | the dispatch table, and the numerical contract — the fixed sixteen-accumulator reduction tree lives here. |
| [`cpu/`](cpu) | feature detection and tier selection, including the `GOSIMD` override. |
| [`backend/`](backend) | wiring that picks the accelerated or portable path at startup. |
| [`conformance/`](conformance) | the differential suite: every tier against `ref`, and tier against tier. |
| [`asmcheck/`](asmcheck) | static assertions on the committed assembly. Parses text, needs no toolchain, catches mis-gated instructions. |
| [`perf/`](perf) | llvm-mca throughput modelling for architectures this machine cannot time. |
| [`benchmarks/`](benchmarks) | every benchmark. See its own README. |
