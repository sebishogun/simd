# Benchmark baselines

One file per GOARCH, holding raw `go test -bench` output. `make bench-check`
compares a fresh run against the file for the current architecture and fails on
a regression past its threshold.

## How these were recorded

    make bench-update

which runs the suite pinned to cores 0-7 at `BENCH_COUNT` repetitions and
overwrites the baseline. **Run nothing else while it measures.** A compile
burst lands on whichever benchmarks happen to be sampling and is
indistinguishable from a real regression.

## The machine

    kernel     7.1.4-arch1-1
    cpu        AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, 32 threads
    governor   performance
    pinned to  cores 0-7 (BENCH_PIN), builds on 8-15,24-31

This is recorded because it was not, for the previous baseline, and a figure
without the conditions that produced it cannot be reproduced or argued with.
The governor matters most: on `powersave` the same suite measures differently
enough to manufacture regressions on its own.

## Reading a failure

`benchcheck` compares the **minimum** of each benchmark's samples, not the
median. Interference is one-sided — nothing makes a kernel finish faster than
it can — so the samples are the true cost plus a non-negative contaminant, and
the minimum is the maximum-likelihood estimate of the true cost.

That was not always so, and the reason it changed is worth knowing before
trusting a failure: comparing medians, two consecutive runs on an idle machine
at the same commit reported sixteen regressions and then five, **with zero
overlap**. Every one was noise. See entry 28 of docs/wrong.md.

So: a single failure means run it again. Only a regression that appears in
both runs is real. Never run `make bench-update` to clear a failure — it
records whichever transient was in flight and makes the problem permanent.

## Sample count

The baseline must be recorded at the same `BENCH_COUNT` the check will use.
More samples give a lower minimum, so a three-sample baseline compared against
a six-sample run is biased toward the new run looking faster, and hides real
regressions.
