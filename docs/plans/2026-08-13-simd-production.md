# simd production implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining open items recorded in ROADMAP.md — the sort three-way partition, the measured `GOEXPERIMENT=simd` small-n tier, real-hardware thresholds away from amd64, and the general n-ary closure combinator — each as a TDD task with evidence, without changing the shipped v1.20.0 contract.

**Architecture:** Every task follows the repository's existing pipeline: C kernel in `csrc/` → manifest in `tools/simdgen/kernels/` → generated Plan 9 assembly and dispatch tables → portable reference in `internal/ref/` → differential conformance. Each task starts with a failing test or failing measurement gate, ships only with the five forms of evidence (source, disassembly, correctness, cross-tier, benchmark) required by the production design record, and records what measurement rejects in `docs/wrong.md`.

**Tech Stack:** Go 1.26.5, clang + llvm-objdump (codegen), Plan 9 assembly, the generator in `tools/` (separate module), qemu-user and docker+qemu emulation lanes, `perf stat`, llvm-mca.

**Scope note:** `docs/plans/kernels-backlog.md` records the kernel backlog #207–#213 as complete and released in v1.16.0–v1.20.0; its remaining items (the #205 projects: simdcbor, simdparquet, simdmsgpack, and the rest) live in their own repositories and are deliberately not staged here. This plan stages only the open work that belongs to this repository: the ROADMAP items above.

---

### Task 1: Three-way sort partition kernel

The sort section of ROADMAP.md names the one remaining sort gap: few distinct values at 16384 loses to `slices.Sort` by 34%, because the pivot equals much of the range, the split is lopsided, and the skew guard hands the range to pdqsort after paying for one partition. The recorded fix is a three-way partition — below / equal / above — so duplicates are consumed at each level.

**Files:**
- Modify: `csrc/sort.c` (add the three-way partition kernel beside `simd_partition_*`)
- Modify: `tools/simdgen/kernels/kernels.go` (manifest entry)
- Modify: `internal/kernel/kernel.go` (dispatch field with the numerical contract comment; this file is not generated)
- Modify: `internal/ref/sort.go` (portable implementation, ops-table entry, exported entry point)
- Modify: `sort.go` (public wrapper; the exact name/signature follows the `PartitionInto` conventions and is decided in Step 1, not assumed)
- Modify: `sort_test.go`, `internal/conformance/` (differential)
- Create: `bench_sort3_test.go` or extend `bench_*_test.go` (benchmark including the few-distinct corpus)
- Modify: `docs/platforms.md` (per-architecture kernel counts change wherever the new kernel emits; the docs tests hold them to the sources)
- Test: `internal/tests/docs` counts and API-table rows stay in sync

**Step 1: Write the failing differential and benchmark tests**

Add the conformance entry for the new kernel against `internal/ref`, and a
benchmark whose corpus includes the failing shape: few distinct values at
16384 elements, plus the sizes where sort currently wins (1024, 16K, 64K).

**Step 2: Run them to verify they fail**

Run: `go test ./internal/conformance -run <new>`

Expected: FAIL — no such kernel in `kernel.Set`, no manifest entry.

**Step 3: Implement the kernel through the eight-file path**

Follow `docs/kernels.md` in order: C in `csrc/sort.c` (signed `isize`
counters, `__restrict` pointers, fold macros), manifest entry with the
correct `CArgs`/`lenOf` handling (three-way partition writes two output
regions), contract comment in `internal/kernel/kernel.go` (exact under rule
2 for integer types; float ordering follows the existing partition rules),
portable reference, public wrapper, conformance, benchmark.

Run: `make codegen && make check-emission`

Expected: PASS; the new kernel emitted on the tiers that can express the
partition (AVX-512, SVE2, and RVV have a compress instruction; the portable
path is the permanent answer elsewhere, as with `Compress`).

**Step 4: Wire it into the sort path and verify**

Replace the skew-guard fallback in `sort.go` that currently hands the
few-distinct case to pdqsort after one partition, so duplicates are consumed
by the three-way split instead.

Run: `go test ./... && make verify`

Expected: PASS.

Run: `make bench-check`

Expected: PASS, with the few-distinct 16384 case no longer a 34% loss; the
existing 19–27% wins at 1024+ must not regress. If a tier loses, keep the
portable path there and record the measurement in `docs/wrong.md`.

**Step 5: Commit**

```bash
git add csrc/sort.c tools/simdgen/kernels/kernels.go internal/kernel/kernel.go \
        internal/ref/sort.go sort.go sort_test.go internal/conformance \
        bench_sort3_test.go docs/wrong.md docs/platforms.md
git commit -m "feat: three-way partition for few-distinct sorts"
```

---

### Task 2: Measured `GOEXPERIMENT=simd` small-n tier

ROADMAP.md's Tiers section records both halves of the history: the vector
type shipped (`vec.go`, amd64-only), and the small-n fast path through it
was built, benchmarked, and **rejected** — a 46% regression at n=8 through
the public API (`docs/wrong.md` entry 58). The standing plan is additive and
measured: build the tier behind the build tag against the same `kernel.Ops`
contract, benchmark n = 4, 8, 16, 32, 64, 128, 256 against both the assembly
tier and the scalar fallback, and wire only the sizes where it wins. The
same rule as every other tier: a measurement decides.

**Files:**
- Create: a new backend under `internal/` behind `//go:build goexperiment.simd` implementing `kernel.Ops` (the exact directory follows where the Go 1.26 `simd/archsimd` import lives; the `vec.go` build constraint is the precedent)
- Modify: `dispatch.go` (tier wiring so dispatch selects it like any other backend)
- Create: `bench_intrin_*_test.go` (n = 4…256, versus assembly tier and `GOSIMD=scalar`)
- Modify: `internal/conformance/` (bit-identity differential against `internal/ref` — the contract still binds)
- Modify: `docs/wrong.md` (the A/B outcome, win or loss)
- Modify: `ROADMAP.md` only to mark the settled outcome after the measurement

**Step 1: Write the failing differential test**

Differential test for the new tier against the portable reference.

**Step 2: Run it to verify it fails**

Run: `GOEXPERIMENT=simd go test ./internal/conformance`

Expected: FAIL — the tier does not exist yet.

**Step 3: Build the tier behind the build tag**

Implement the backend against `kernel.Ops`; do not wire dispatch yet.

Run: `GOEXPERIMENT=simd go test ./... && make verify`

Expected: PASS (the new backend compiles under the experiment and the
default build is unchanged).

**Step 4: Benchmark every size before wiring anything**

Run: `go test -run '^$$' -bench . -count 6 -shuffle=on ./internal/benchmarks` on a quiet machine, interleaved A/B builds, minimum compared, `perf stat -e instructions:u,cycles:u` where the wall-clock gap is under the 8.3% layout-noise floor.

**Step 5: Wire only the sizes that win**

Wire dispatch for exactly the sizes the measurement supports; everywhere
else keeps the assembly tier or the scalar fallback. Re-run Steps 2 and 4
through the public API (the mistake entry 58 records was benchmarking a call
shape no real program uses).

Run: `GOEXPERIMENT=simd go test ./... && make verify && make bench-check`

Expected: PASS.

**Step 6: Record and commit**

```bash
git add internal/<new> dispatch.go bench_intrin_*_test.go internal/conformance docs/wrong.md
git commit -m "feat: GOEXPERIMENT=simd tier at the measured sizes"
```

If no size wins, commit only the differential coverage and the `docs/wrong.md` entry — the record is the deliverable.

---

### Task 3: Real-hardware thresholds away from amd64

ROADMAP.md's Verification section states that every per-architecture
threshold outside amd64 is a guess carried over from a machine that does not
resemble it, and that a crossover measured on nothing is a number with no
evidence behind it. Thresholds are per-kernel element counts generated from
the manifest into `kernel_thresholds_<arch>.go`.

**Files:**
- Modify: `tools/simdgen/kernels/kernels.go` (per-architecture threshold overrides, once measured)
- Regenerate: `kernel_thresholds_<arch>.go` via `make gen-thresholds`
- Modify: `testdata/hardware/` (the real-silicon runs the numbers come from)
- Modify: `docs/wrong.md` (measurements that moved or confirmed thresholds)
- Modify: `docs/platforms.md` only if a claim about thresholds appears there

**Step 1: Gather the evidence**

Run `make hardware-report` and `make hardware-bench` on real arm64 NEON
silicon, then campaign for the unmeasured tiers (sve2, rvv, vsx, vx, lasx)
per CONTRIBUTING.md — one real run is more useful than any amount of
emulation, and failures are useful data.

**Step 2: Write the failing meta-test for the candidate table**

Extend the threshold meta-test (the one that first found four uncovered
kernels) so each architecture's table carries its measurement provenance;
run it before regenerating.

**Step 3: Measure the crossover on the metal**

For each operation, interleaved A/B of kernel vs reference across the
threshold band on the real machine, minimum compared, quiet machine.

**Step 4: Regenerate and verify**

Run: `make gen-thresholds && make check-emission && make verify`

Expected: PASS with the new tables.

**Step 5: Commit**

```bash
git add tools/simdgen/kernels/kernels.go kernel_thresholds_*.go testdata/hardware docs/wrong.md
git commit -m "feat: measured thresholds for <arch>"
```

---

### Task 4: General n-ary closure combinator

ROADMAP.md's n-ary section records the shipped half (`AddAll`, `MulAll`,
arity 3–4 kernels, folding beyond) and the open half: a general combinator
taking a Go closure. The roadmap's own constraint: a closure call per
element defeats vectorization entirely, so it needs a design where the
common shapes dispatch to real kernels and the rest is honest about being a
scalar loop — `FilterInto` is the same problem solved once that way.

**Files (design step):**
- Create: `docs/research/07-nary-closure.md` (the decision record: which shapes dispatch, which stay scalar, the measured call cost)

**Files (implementation step, once the design is fixed):**
- Modify: `internal/ref/nary.go` (portable implementation)
- Modify: the public n-ary file (wrapper; name/signature per the design)
- Create: `bench_*_test.go` (closure loop vs the equivalent handwritten loop)

**Step 1: Write the design record with the measurement gate**

The record must state the dispatch shapes and the scalar contract, and
benchmark the closure-loop cost per element so the design decision is
evidence-backed. Commit it alone.

**Step 2: Write the failing tests**

Differential test for the scalar path and for each shape that dispatches to
a real kernel; benchmark vs the handwritten loop the combinator replaces.

**Step 3: Implement the decided shape**

Follow `docs/kernels.md` for the dispatch shapes; the scalar rest follows
the `FilterInto` precedent.

**Step 4: Verify**

Run: `go test ./... && make verify && make bench-check`

Expected: PASS. A shape that does not beat the handwritten loop stays out of
the dispatch set and is recorded in `docs/wrong.md`.

**Step 5: Commit**

```bash
git add internal/ref/nary.go nary.go bench_*_test.go docs/research/07-nary-closure.md docs/wrong.md
git commit -m "feat: n-ary closure combinator"
```

---

### Final task: full verification and review

**Step 1:** Run: `go test ./... && make verify && make check-emission && git diff --check`

Expected: PASS, bare (no piped `tail` without `pipefail`).

**Step 2:** Run the release set: `make test-cross && make test-riscv64 && make test-loong64 && make test-gates && make fuzz && make bench-check`

Expected: PASS.

**Step 3:** Review the diff: `git diff HEAD~<n> --stat`, and confirm every
generated file matches its source (no hand-edited assembly), every count in
`docs/platforms.md` matches the sources, and every new measurement that
argued against a change is in `docs/wrong.md`.

**Step 4:** Commit any verification-only corrections and report.

---

## Follow-on: production-readiness ledger (appended 2026-08-24)

The tasks above are historical records: their IDs and text are preserved and
never renumbered or rewritten for status. Follow-on production-readiness work
is recorded here, one task ID at a time, with the seven-state vocabulary
(`open`, `staged`, `in-progress`, `blocked`, `evidence-complete`, `shipped`,
`rejected`). Every row below starts `open`. A transition is an edit to this
table (plus the changelog or `docs/wrong.md` for `shipped`/`rejected`);
`rejected` is terminal without a documented reopen condition.

| ID | state | work | evidence | exit |
|---|---|---|---|---|
| `SIMD-CORR-01` | open | fuzz differential corrective: review the fuzz targets in `internal/conformance/` against the differential contract (NaN payloads, IEEE specials, adversarial lengths); fix or record | bare fuzz run with timeout, green, or a `docs/wrong.md` entry | corrective slice green |
| `SIMD-CORR-02` | open | timeout corrective: every test and fuzz invocation in the verification set carries an explicit timeout | `docs/verification.md` lists each command with its timeout; one bare green run | corrective slice green |
| `SIMD-CORR-03` | open | docs/count/claim corrective: reconcile `docs/platforms.md` counts, README claims, and ROADMAP status against sources; resolve ROADMAP's "door closed twice" sentence against `docs/wrong.md` entry 80; reconstruct or correct entry 80's unexplained 1,080-case denominator, then correct the dependent arithmetic (388,352 minus 1,080 leaves 387,272, not 387,271) and its attribution of the four-line objdump excerpt (the `src1 = a` / `src1 = b` operand-order annotations in the masked-tail listing); reconcile entry 81 and the fuzz source comment's historical "11 seeds" label against the current three `f.Add` seeds; resolve entry 81's stale `f.Fatalf` wording against the final `f.Skip` guard | docs tests green; each named contradiction corrected against code or its owning record; historical measurements explicitly labeled with their measurement-time corpus | corrective slice green |
| `SIMD-R5-01` | open | real-silicon threshold evidence: thresholds away from amd64 measured on real hardware; emulated lanes are not real-silicon evidence | recorded measurements with provenance | R5 maintained-production evidence |
| `SIMD-R5-02` | open | patch/upgrade cycle and observed release automation: one patch cycle with the regression suite green and the release automation observed end to end | patch release plus green suite | R5 |
| `SIMD-R5-03` | open | remaining roadmap kernel work: the sort three-way partition and the general n-ary closure combinator, each with its evidence bar from Tasks 1 and 4 above | per-task gates or `docs/wrong.md` rejections | roadmap items closed or rejected |
| `SIMD-R5-04` | open | C++/Rust workload-gap decisions: compare supported whole-slice workloads with the named C++ and Rust peers; classify material gaps as in-boundary work, future work, or rejected with evidence | the workload matrix below plus the decision record | gaps classified without feature-count parity |

**Not a task row, by design:** the measured `GOEXPERIMENT=simd` small-n tier.
Entries 58, 75, and 79 close it on measured speed; the ROADMAP records the
condition for re-running it. Entry 80 records a source-level bit-identity limit
and adds no speed rejection. It is not reopened or listed as open here.

### v1/R5 exit

The existing production release remains current: the shipped v1 line
(v1.21.1) is the supported release and nothing in this ledger changes that.
R5 requires real-silicon/current evidence (SIMD-R5-01), an observed
patch/upgrade cycle with release automation (SIMD-R5-02), and the open
roadmap work closed or rejected (SIMD-R5-03, SIMD-R5-04), with the
corrective slice (SIMD-CORR-01..03) shipped.

### Workload matrix (competitive work)

Columns: `workload | this repo | peer | oracle-or-basis | gate`.

No external library is an oracle in this repository. The differential
contract binds to `internal/ref` and the numerical contract in
`internal/kernel/kernel.go`; the plain Go loop in `docs/kernels.md` is the
local performance reference. C++ and Rust SIMD libraries are
workload and performance peers, never behavioral oracles, named only where
this repository's own research docs name them: Google Highway, xsimd,
SIMDe, and Rust portable-simd (`docs/research/02-codegen-pipelines.md`).
Figures are quoted from sources
with provenance, never invented; a gap is a workload, an allocation, a
dispatch, or a measurement, never a feature count.

Every comparison is gated on the five forms of the production design
record: workload stated before measurement, compatibility promise,
zero-allocation path, runtime dispatch engagement, and interleaved-A/B
benchmark discipline on a quiet host (minimum compared, bench-check
baseline, `-benchmem`, `perf stat` below the 8.3% layout floor).

| workload | this repo | peer | oracle-or-basis | gate |
|---|---|---|---|---|
| elementwise float add/mul, whole slice, n above the dispatch threshold | `internal/ref` and the scalar loop | Google Highway, xsimd, Rust portable-simd elementwise loops | none - `internal/ref` differential | quiet-host interleaved A/B; bench-check baseline; dispatch-engagement assertion |
| whole-slice reductions (Sum) | `internal/ref` | Google Highway, Rust portable-simd reduce | none - `internal/ref` differential | same |
| compress/expand/filter (CompressInto, ExpandInto, FilterInto) | `internal/ref` | Google Highway compress, SIMDe | none - `internal/ref` differential | same; per-tier correctness differential |
| byte scanning (IndexByte, Equal, HammingDistance, mask scans) | `internal/ref` | Google Highway, SIMDe | none - `internal/ref` differential | same |
| sort/argsort/partition at 16K+ (few-distinct corpus included) | `internal/ref` | none named in this repo's docs; chosen under SIMD-R5-04 and recorded with the workload | none - `internal/ref` differential | same; few-distinct corpus per Task 1 |
