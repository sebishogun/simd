# Contributing

## The one thing to know first

This library's entire claim is that **the answer does not depend on the
machine**. Every operation returns the same bits on a 128-bit vector unit and a
512-bit one, on six architectures, and under `-tags purego`. Operations that
trade that away are named `Fast*` and say what they traded.

So the bar for a change is not "it passes the tests". It is "it cannot make two
machines disagree". Most of what follows is in service of that.

The six rules are in the package comment of
[`internal/kernel/kernel.go`](internal/kernel/kernel.go). Read them before
changing anything numerical.

## Adding an operation

[`docs/kernels.md`](docs/kernels.md) is the worked guide: the eight files, what
the manifest fields mean, the verification a new kernel must pass, and the
traps that already cost someone a day each.

## Building

The generated assembly is **committed**. You do not need clang to build or use
this library — only to change a kernel.

```
go test ./...                 # that is the whole thing, if you are not touching csrc
```

To regenerate you need clang 20 or newer with the targets built in, which is
the default for the LLVM project's own releases and for Arch, Debian and
Homebrew:

```
make codegen                  # regenerate every kernel for all nine tiers
git diff --stat internal/      # confirm the diff is only what you changed
```

If regenerating produces a diff in kernels you did not touch, stop and find out
why before committing it.

## Verifying

```
make verify                   # fmt, vet, test, purego, every amd64 tier
make test-cross               # arm64, s390x, ppc64le — docker + qemu
make test-riscv64
make test-loong64
make test-gates               # the same binaries on a CPU with no vector unit
```

`make test-gates` is the one people skip and the one that catches the failure
this library was designed against. Every other lane runs a CPU with every
feature switched on, so a kernel gated on a feature it does not actually
require is selected and runs fine — and fails as SIGILL on a machine that lacks
it. In production, at the first call.

Two ways to get a meaningless green, both of which have happened here:

- **Editing the tree while `make test-cross` runs.** It mounts the working
  directory into the container, so the result describes a half-written tree.
- **Piping it through `tail`.** The pipeline exits with `tail`'s status, which
  is 0 no matter what make did. Redirect to a file and check `$?`.

`loong64` needs a hand-fetched qemu 10.x; the Makefile prints how to get one if
yours is too old.

## Benchmarks

Nothing lands on a performance claim that was measured once. See
[`testdata/bench/README.md`](testdata/bench/README.md), and:

- Compare against **what a caller would otherwise write** — the plain Go loop,
  or the sequence of existing calls being replaced. A fused kernel measured
  against a scalar loop rather than against the chain it replaces looked like a
  win while being 1.76× slower than not fusing.
- Take the **minimum** of several runs, not the mean, and run the comparison
  twice. A number from a busy laptop is worse than no number.
- `make bench-check` compares against the committed baseline.

## Documentation

Doc comments here carry the numerical contract, so they are part of the change
rather than a follow-up. If an operation is exact, say so; if it carries a
bound, state the bound; if it is `Fast*`, say what it trades.

[`docs/wrong.md`](docs/wrong.md) records things that were believed, were false,
and cost real time — written as *what was assumed / what was true / how it
surfaced*. If you lose an afternoon to something, that is where it goes. It is
the most-read file in the repository for a reason.

## Style

- Match the surrounding code. Comment density here is high on purpose: the
  comments explain *why the obvious thing is wrong*, which is the part that
  does not survive in a diff.
- Explain the tradeoff, not the syntax. `// increment i` is noise;
  `// r0 is constant-zero by ABI, so a signal landing here poisons the runtime`
  is the reason the line exists.
- No new dependencies. The library has none outside the standard library and
  `golang.org/x/sys` for CPU detection, and that is a feature.

## What gets declined

A kernel that cannot be generated for a target is **not** a failure — the
portable path stands in, and `make check-emission` records the reason. Shipping
on five of six architectures is normal and is stated rather than hidden.

What is not acceptable is a kernel that is accepted and wrong. Everything in
the traps section of [`docs/kernels.md`](docs/kernels.md) is an instance of
that, and every one of them was green somewhere before it was caught.
