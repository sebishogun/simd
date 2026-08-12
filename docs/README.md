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

## Engineering record

These files preserve decisions and measurements from the time they were made;
they are not rewritten to read like current reference pages.

| Read | Purpose |
|---|---|
| [`wrong.md`](wrong.md) | Changes and assumptions that measurement rejected, including work that was deleted. |
| [`research/`](research) | Longer design research; `05-decisions.md` is the decision record. |
| [`plans/`](plans) | Completed and active implementation plans. |
| [`assets/`](assets) | Repository badges and images. |
