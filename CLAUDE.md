# Working on this project

## Disassemble first, always

Before proposing a cause for anything slow, before writing a variant, before
reading a profile delta — **build it and read the instructions**.

```
go test -c -o /tmp/x.test .
go tool objdump -s 'pkg\.functionName' /tmp/x.test | less
```

Use gdb when a breakpoint or a live register is needed, and both together when
that helps. Go compiles in seconds; there is no cost to looking, and every guess
that skips it costs a build-measure-revert cycle and risks a wrong conclusion
landing in `docs/wrong.md` as fact.

What the disassembly says that nothing else does:

- **Register pressure.** A large stack frame with the loop counter or a flag
  spilled and reloaded per iteration. No performance counter reports this. It
  was the real cause of an 18% gap that three rounds of counter-guessing —
  cache footprint, key copying, per-field bookkeeping — all missed.
- **Whether a bounds check was eliminated**, and whether an index multiply is
  a shift or a multiply.
- **Whether a call was inlined**, and whether `append(b, s...)` became inline
  stores or a `memmove` call.
- **Which branch the compiler laid out as fallthrough.**

## Benchmarks

The code-layout noise floor here is **8.3%**. Anything smaller cannot be told
from nothing by wall-clock, and more samples do not help — layout noise is
per-build, not per-run. When a change is expected to be worth less than that:

- compare **instructions retired** and **cycles** with `perf stat -e
  instructions:u,cycles:u`, which are layout-independent;
- and read the disassembly, which is the only thing that explains *why*.

A/B builds must be **interleaved** in one session and compared on the minimum,
never across sessions. Run the machine quiet: wait for load average under 1.

**Never pipe a gate through `tail`** (or anything else) without `pipefail`:
the pipe reports the last command's status and the failure vanishes. This has
now laundered a red fuzz run, a red README gate, and two red bench-check runs
into green exits. Run gates bare, or `set -o pipefail` first.

## The record

`docs/wrong.md` in each repository holds measurements that argued against
changes, including changes that were then reverted. A finding that cost a
measurement belongs there whether or not any code changed — the entry is the
deliverable.
