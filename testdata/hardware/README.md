# Hardware reports

One directory per machine that has run this library on real silicon, so that
the verification table in the README is a record of what happened rather than
an assertion.

Every architecture here is tested under emulation on every change, which proves
semantics and nothing about timing. qemu does not model a pipeline. So a single
run on a real chip is worth more than any amount of emulated coverage — and for
five of the seven tiers, nobody has done one.

## What is in a report

A file named `<goos>-<goarch>-<tier>.md`, for example `linux-riscv64-rvv.md`,
containing:

- the output of `go run ./cmd/simdinfo`, which names the tier that was actually
  selected,
- whether `go test ./...` passed, and the failure if it did not,
- the CPU, as specifically as you can say — model name, or the SoC,
- optionally, benchmark output.

That is it. A report saying only "this CPU, this tier, tests pass" is a useful
report; it is the difference between "we think this works" and "it worked once,
here, on this".

## Why a failing run is the more valuable one

Three memory-corruption bugs in this library's history were invisible on amd64
and only appeared under emulation: kernels clobbering the register Go keeps the
current goroutine in, kernels writing into the caller's frame, and a reference
that computed different bits where Go fuses a multiply into an add.

Emulation caught those because it executes. It cannot catch a CPU that
implements an instruction differently from the emulator, an errata, or a
scalable vector length no emulator was configured for. Those need silicon, and
if one of them is live, a failing report is how it gets found.

Send the failure. Do not clean it up first.

## The tiers that need one

As of this writing: **arm64 sve2** (Graviton 3 or 4, or any SVE2 chip),
**riscv64 rvv** (a board with the V extension, not just RV64GC),
**ppc64le vsx**, **s390x vx**, and **loong64 lasx**.

amd64 and arm64 NEON are covered. Another report on either is still welcome —
a different microarchitecture is a different machine — but the five above are
where a single run changes what this library can honestly claim.
