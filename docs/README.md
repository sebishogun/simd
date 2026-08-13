# Documentation

## Using the package

| Read | Purpose |
|---|---|
| [`tutorial.md`](tutorial.md) | **Start here.** Batch work, choose a vector-friendly data layout, reuse buffers, and fuse memory passes. |
| [`api.md`](api.md) | API conventions, allocation and sizing rules, supported types, and the operation catalog by task. |
| [`guide/`](guide) | Task guides for arrays, text, search, encodings, signals, and matrices. |
| [`platforms.md`](platforms.md) | Runtime tiers, source-backed kernel counts, fallback behavior, hardware verification, and ABI limits. |
| [`examples/`](examples) | Complete runnable programs. Shorter checked examples render on pkg.go.dev. |

## Contributing

| Read | Purpose |
|---|---|
| [`kernels.md`](kernels.md) | Add a kernel end to end, including generation, dispatch, testing, and architecture traps. |
| [`runner.md`](runner.md) | Verification runner and cross-architecture execution details. |
| [`comparison/`](comparison) | Reproduce the reduction comparison without adding another SIMD library to the main module. |

## Production and engineering

For anyone changing the library or auditing how it is built:

| Read | Purpose |
|---|---|
| [`architecture.md`](architecture.md) | Component map, call path, generator source of truth, package and module boundaries. |
| [`lld/api-and-memory.md`](lld/api-and-memory.md) | LLD: operation families, allocation, sizing, aliasing, and length contracts. |
| [`lld/kernels-and-dispatch.md`](lld/kernels-and-dispatch.md) | LLD: CPU detection, tier selection, dispatch, linking, and hot-loop rules. |
| [`lld/generation-and-platforms.md`](lld/generation-and-platforms.md) | LLD: the C-to-assembly pipeline, tier support, and what each platform claim means. |
| [`verification.md`](verification.md) | Every gate, what it runs, when it applies, and the measurement discipline. |
| [`plans/2026-08-13-simd-production-design.md`](plans/2026-08-13-simd-production-design.md) | Production design record: the shipped contract and the evidence bar for new kernels. |
| [`plans/2026-08-13-simd-production.md`](plans/2026-08-13-simd-production.md) | Staged implementation plan for the remaining roadmap work. |
| [`../AGENTS.md`](../AGENTS.md) | Agent and contributor rules: boundaries, gates, and records. |

Suggested order for someone changing the library: README, architecture, the
three LLDs, ROADMAP, verification, wrong.md, then the production design
record and plan.

## Engineering record

These files preserve decisions and measurements from the time they were made;
they are not rewritten to read like current reference pages.

| Read | Purpose |
|---|---|
| [`wrong.md`](wrong.md) | Changes and assumptions that measurement rejected, including work that was deleted. |
| [`research/`](research) | Longer design research; `05-decisions.md` is the decision record. |
| [`plans/`](plans) | Completed and active implementation plans. |
| [`assets/`](assets) | Repository badges and images. |
