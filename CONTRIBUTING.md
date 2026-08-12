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

## Cutting a release

The order matters, and getting it wrong is not recoverable.

```
make verify && make test-cross          # and the qemu lanes
git push origin main                    # FIRST
git fetch origin && git rebase origin/main
git tag -a vX.Y.Z -F - <<'EOF'
...
EOF
git push origin vX.Y.Z                  # only now
```

**Tag after main is pushed and rebased, never before.** A tag names a commit. If
you tag, then rebase — which is what happens when the release-badge bot has
pushed while you were working — the tag is left pointing at the commit the
rebase replaced. v1.1.0 and v1.2.0 are both like that: the Go code in them is
byte-identical to main and they resolve from the proxy perfectly well, but the
commits they name are not ancestors of any branch, so `git describe` reports the
wrong version.

**They were not fixed, and should not be.** Moving a published tag changes the
commit and therefore the module zip, and the Go checksum database has already
recorded a hash for that version. A caller who has fetched it once gets a
checksum mismatch, which fails hard and looks like a supply-chain attack. A
cosmetic history oddity is very much the cheaper of the two.

The same rule covers the badge: the bot commits to main *after* a tag is pushed,
so the tagged tree always has the previous version's badge in it. That is
expected and is the only file that differs between a tag and its equivalent
commit on main.

## Reporting a hardware run

**This is the contribution the library most needs, and it requires knowing
nothing about the internals.**

Every architecture is tested under emulation on every change. That proves
semantics — a wrong answer gets caught — and it proves nothing about timing,
because qemu does not model a pipeline. It also cannot catch a chip that
implements an instruction differently from the emulator, an errata, or a
scalable vector length nobody configured.

Five tiers have never run on real silicon: **arm64 sve2**, **riscv64 rvv**,
**ppc64le vsx**, **s390x vx** and **loong64 lasx**. amd64 and arm64 neon are
verified for correctness; amd64 is the only one with a wall-clock figure. If
you have one of those machines, this is one command:

```
make hardware-report
```

That runs `simdinfo`, the accelerated suite and the portable suite, and writes
`testdata/hardware/<goos>-<goarch>-<tier>.md` with everything already filled
in. It does not abort when the suite fails, because that is the report worth
having.

Open a pull request adding that file. If you would rather not, or cannot push
from the machine that ran it, paste the output into
[an issue](https://github.com/sebishogun/simd/issues/new/choose) and someone
else will file it — the point is the data, not the paperwork.
[`TEMPLATE.md`](testdata/hardware/TEMPLATE.md) is the same report written by
hand, for when running make is not an option.

Three things worth saying plainly:

**A failing run is the more valuable one.** Do not clean it up, do not try to
fix it first, and do not decide it is your machine's fault. Three
memory-corruption bugs in this library's history were invisible on amd64; the
next one is likelier to be found by a stranger's board than by any test here.

**"Tests pass" on its own is a complete report.** It moves a row of the
[platform verification table](docs/platforms.md#runtime-tiers) from
*emulation* to *real hardware*. Two rows have moved that way so far and five
have not; none of them will move without evidence.

**Benchmarks are optional and have rules.** If you send timing, the machine has
to be quiet and it has to be `-count 6` or more, because the minimum is what
gets used — benchmark noise is one-sided, so the fastest run is the one least
interfered with. A number from a busy laptop is worse than no number, which is
[entry 48 of `docs/wrong.md`](docs/wrong.md) and cost twenty-one phantom
regressions to learn.

## What gets declined

A kernel that cannot be generated for a target is **not** a failure — the
portable path stands in, and `make check-emission` records the reason. Shipping
on five of six architectures is normal and is stated rather than hidden.

What is not acceptable is a kernel that is accepted and wrong. Everything in
the traps section of [`docs/kernels.md`](docs/kernels.md) is an instance of
that, and every one of them was green somewhere before it was caught.
