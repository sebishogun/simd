# LLD: Kernels and dispatch

How the CPU tier is chosen, how one operation reaches its kernel, what the
linker keeps, and the rules the kernel code itself must obey. The user-facing
summary is [`docs/platforms.md`](../platforms.md); the add-a-kernel walkthrough
is [`docs/kernels.md`](../kernels.md).

## CPU detection

- Detection lives in [`internal/cpu`](../../internal/cpu/) and uses
  `golang.org/x/sys/cpu` — the only runtime dependency of the module. It
  runs once during package initialization (`cpu.init` reads `GOSIMD` and
  `SIMD_DISABLE`; `dispatch.init` sets `tierIdx` once, never again, so reads
  need no synchronization).
- Nothing in detection executes a SIMD instruction. A package-level
  initializer that runs a vector instruction would execute before the
  dispatch check that is supposed to guard it — the mechanism behind
  go-highway#69's SIGILL on non-AVX2 CPUs.
- `scalar` is always the first available tier; tier lists are weakest-first
  and architecture-specific (see the `Tier` constants in `cpu.go`).

## Overrides

| Mechanism | Effect |
|---|---|
| `GOSIMD=<tier>` | pins an exact tier. It can only select down: an unknown tier, or one this CPU lacks, falls back to `scalar` with a `Reason` rather than crashing. A pinned benchmark must not silently substitute a neighbour tier. |
| `SIMD_DISABLE=<tier,...>` | masks tiers out of consideration. `scalar` can never be masked. |
| `-tags purego` | `puregoOnly` forces tier 0: the whole assembly surface is excluded and the reference is the only implementation. |
| `simdinfo` / `Tier()` / `AvailableTiers()` / `Describe()` | report the outcome (`Describe` includes `forced`, `disabled`, `backend`, and `reason` fields). |

`make test-tiers` runs the suite once per tier the current CPU supports by
iterating `simdinfo -tiers`.

## Tier selection and per-operation dispatch

- `tierIndexFor` picks the strongest tier this build has tables for among
  the ones the CPU supports. The backend tier can be lower than the CPU
  tier (a machine with AVX-512 whose build carries no AVX-512 kernels gets
  AVX2, not portable Go).
- Each exported function is a one-liner through its own static table
  (`dispatch_tables_<arch>.go`), indexed at `tierIdx`:
  `func Add[T Number](a, b []T) { ops[T]().Add(a, a, b) }`.
- The numeric groups use `ops[T]()`: a single type switch in `dispatch.go`
  selects the per-element-type cache, which overlays the reference with the
  tier's partial `Ops` struct lazily on first use (one `reflect` pass per
  element type, never on the hot path).
- A **generated guard** sits between the table and the kernel. It checks the
  per-operation element threshold (`KernelThreshold` is the public view;
  the tables are generated from the same manifest and a test holds the two
  to agreement), clamps to the minimum slice length, evaluates `RefWhen`,
  and otherwise calls the kernel or the named `ref` function.
- Where an operation has no kernel on a tier — declined at generation time
  (`make check-emission` prints the reason) or nil by design (`Compress` is
  nil everywhere but AVX-512 and SVE2, `ExpandInto` is portable everywhere)
  — the guard calls the portable implementation. A missing kernel is slower,
  never a correctness gap.

## Scalar fallback and thresholds

- Below a per-kernel element count, crossing into assembly costs more than
  the arithmetic saves: a Go-to-assembly call is a fixed ~1.4 ns and can
  never be inlined. The threshold belongs to the kernel (generated into
  `kernel_thresholds_<arch>.go`), is in the operation's own length units,
  and must be measured rather than guessed.
- The reference base (`refBase` in `dispatch.go`) is the complete portable
  `kernel.Set` with fast fallbacks filled in: the fallback for per-operation
  misses and the direct target for operations with no kernel anywhere.

## Linker and dead-code behavior

- Dispatch tables are **static composite literals** of exported guards. Any
  computed entry would force an init function, and anything reachable from
  init is linked into every binary that imports the package. Static
  per-operation tables let the linker drop every operation a program never
  calls, assembly included: a consumer using three operations carries three
  operations' kernels, not all 6,931.
- The reference is ordinary Go linked into every consumer regardless (a few
  hundred kilobytes); the per-tier assembly is what the tables keep out.
- Per-element-type partials keep liveness scoped: the float32 partials are
  referenced only from the float32 instantiation of `ops`, so a program that
  never touches float32 never links its float32 kernels.

## Hot-loop rules

These are the properties that make a kernel fast and safe; `docs/kernels.md`
is the how-to, this is the list:

- Kernels do not allocate, do not call (no libc, no libm — `sqrtf` is a
  call and is rejected; `__builtin_elementwise_sqrt` is an instruction), and
  have no data-dependent early exits except where the search kernels
  deliberately declare one.
- No interfaces, no indirection in the kernel path: backends are plain
  structs of function fields (`kernel.Set`), selected by a static table
  index, and the one `reflect` pass runs once per element type at first
  use. This is what keeps per-operation dead-code elimination intact.
- The six numerical rules in `internal/kernel/kernel.go` bind every tier:
  bit identity for elementwise work (rule 1), integer reductions (rule 2),
  the fixed 16-accumulator `SumLanes`/`CombineTree` shape for float
  reductions regardless of hardware width (rule 3), no FMA contraction in
  `Dot` (rule 4), documented `Fast*` bounds (rule 5), and ULP bounds for
  transcendentals (rule 6). `-ffast-math`/`-Ofast` are never used.
- Per-element operations follow Go semantics where they differ from C:
  shifts at or above the element width, `LeadingZeros(0)`, saturating
  `SatAdd`/`SatSub` at the type limits.
- Fused operations exist to make one memory pass instead of several; a
  kernel that does not beat the chain it replaces is a benchmark failure,
  not a style question.

## ABI expectations

- Generated objects begin as C compiled for each target, so the generator
  enforces what a foreign compiler would otherwise violate: no `r13`
  (s390x, Go's goroutine register), no `r30`/`r2` use and no nonzero `r0`
  writes (ppc64le), no `$fp` use (loong64), no writes into the caller's
  frame, and a 512-byte NOSPLIT stack budget.
- System V amd64 passes six integer arguments in registers; the generator
  declines kernels that need more (three in, three out and a length is
  seven — split the kernel).
- On s390x every accepted kernel pairs its callable entry with a `Body`
  trampoline; the coverage tables count the pair once.
- Every accepted kernel is statically checked (instructions within the tier
  that gates its file — no EVEX in an `_avx2.s`; stack and reserved
  registers; the respelled-encodings test holds forty mnemonics byte-for-byte
  against clang), then differentially executed by the conformance suite on
  every tier the host can run.

## Bounds checks and disassembly checks

- The generated guard performs the length and `RefWhen` checks; the kernel
  does not re-check (the manifest's two-length rule decides which slice is
  authoritative). The conformance suite runs every tier against the
  reference, including adversarial and fuzzed inputs.
- `internal/asmcheck` inspects the committed assembly itself; `make
  check-emission` refuses instructions outside a file's tier. Together they
  are the mechanical SIGILL prevention.
- For performance work the rule is disassemble first: `go test -c -o
  /tmp/x.test . && go tool objdump -s 'pkg\.functionName' /tmp/x.test`.
  Register pressure, bounds-check elimination, inlining, and branch layout
  are only visible in the instructions; see `docs/verification.md` for the
  measurement rules that go with it.
