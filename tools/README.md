# tools — the code generator

A **separate Go module**, on purpose: it depends on things a consumer of the
library must never inherit, and keeping it out of the main module's dependency
graph is the reason `go get github.com/sebishogun/simd` pulls in nothing but
`golang.org/x/sys`.

| | |
|---|---|
| `simdgen/` | compiles [`csrc/`](../csrc) per target with clang and lifts the encoded bytes into Plan 9 assembly under [`internal/`](../internal). |
| `benchcheck/` | compares benchmark output against the recorded baseline in [`testdata/bench/`](../testdata/bench). |
| `goat/` | the object-file reader simdgen is built on. |

```
make codegen         # regenerate every backend (needs clang and llvm-objdump)
make check-emission  # dry run: what each target would emit or skip, and why
```

Needed only to change a kernel. [`docs/kernels.md`](../docs/kernels.md) is the
guide.
