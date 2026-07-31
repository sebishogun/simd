# csrc — the kernels, in C

**This is the source of truth for every accelerated code path.** Nothing here
is compiled at install time and nothing here needs a C toolchain to *use* the
library — these files are compiled ahead of time by the generator, and the
resulting assembly is committed under [`internal/`](../internal).

One file per family: `arith.c`, `bytes.c`, `reduce.c`, `math.c` and so on.
They are written to be auto-vectorized by clang rather than to use intrinsics,
which is what lets one file serve nine instruction sets.

## Editing these

A change here does nothing until the assembly is regenerated:

```
make codegen         # recompile every target (needs clang and llvm-objdump)
make check-emission  # dry run: what each target would emit or skip, and why
```

Then `make verify`, because a kernel that compiles is not a kernel that is
correct. [`docs/kernels.md`](../docs/kernels.md) is the guide to adding one,
including the traps that have already been paid for.

## What the compiler is allowed to do

`-ffp-contract=off` throughout: clang must not fuse a multiply and an add into
an FMA, because the result would differ from the portable Go implementation and
from other architectures. The `Fast*` family is the exception and opts in by
name.
