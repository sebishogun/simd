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

## When it flags something twice

The intersection rule above catches transients. It does not catch a machine
that is simply slower than it was when the baseline was recorded, and that
looks exactly like a real regression: it reproduces, it survives the
intersection, and it survives ten samples with the minimum estimator.

After the work of 2026-07-30, three flags did all of that, and one of them was
in an `impl=go` arm — pure Go, untouched. The generated kernels behind the
other two, `indexByteAVX512` and `countSeqAVX512`, were byte-for-byte
identical to their previous versions at the same line of the same file.

**So the third step, after "run it twice" and "re-measure the survivors", is
to build the old tree and run it now:**

```
git worktree add --detach /tmp/old <commit-before-the-work>
cd /tmp/old && taskset -c 0-7 go test -run '^$' -count 10 -bench '<the flagged ones>' .
git worktree remove --force /tmp/old
```

That run said 30.49 ns where the current tree said 30.35 and the baseline said
22.23 — the old code was *slower* than the new code on the same machine at the
same minute. The machine had lost about 35% on those benchmarks over eleven
hours of sustained load, with the governor still pinned to `performance`.

A baseline diff nominates candidates. Only an A/B against the actual old code,
at the same moment, convicts. It costs a minute and it is the difference
between fixing a regression and hunting for one in code that has not changed.

And do not run `make bench-update` to make it go away — that records the
degraded machine as the reference and hides the next real regression behind
it. See entry 48 of docs/wrong.md.
