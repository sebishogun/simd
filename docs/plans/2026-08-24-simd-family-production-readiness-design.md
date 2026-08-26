# SIMD family production readiness: coordination design record

> **Status:** a design record for coordinating production readiness across the
> eleven-repository SIMD family. Nothing here describes new shipped behavior in
> any repository. Current state is described only as it is recorded in each
> owning repository's own documents; everything this record stages is future
> work to be executed per-repo through the existing executing-plans workflow.
> Per-repo implementation work is staged per Section 7 in each repository's
> own documents.
>
> **Scope of this document:** the coordination index, task-state vocabulary,
> competitive-performance policy, dirty-work preservation, per-repo document
> strategy, the first implementation wave, and the review/verification
> workflow. It is one file in one repository; it is deliberately not a
> parallel roadmap for any of the eleven.
>
> **File naming.** This record is `2026-08-24-simd-family-production-
> readiness-design.md`. The family index it stages is a separate file,
> `2026-08-24-simd-family-production-readiness.md`, created in wave 0. The
> `-design` suffix distinguishes the record from the index throughout this
> document.

**Goal:** one non-canonical family coordination index that lets any agent or
contributor tell, for any of the eleven repositories, what production-readiness
work is open, what state it is in, where the authoritative task list lives, and
which gates gate it - without duplicating any per-task status that the owning
repository already records.

**Architecture:** each owning repository keeps its canonical roadmap and its
dated production design and plan. This record adds (a) a family index that
links those documents and states shared gates only, (b) a task-state vocabulary
that maps onto each repository's local statuses without rewriting them, (c)
governance files for the one repository that lacks them, and (d) recorded
first-wave task IDs whose owning-ledger state is `open`. No new parallel
roadmap is created anywhere.

**Tech Stack:** Markdown only. New files: the family index in GO_SIMD
(`docs/plans/2026-08-24-simd-family-production-readiness.md`), seven
simdmetrics governance and plan files (Section 7), and the simdparquet
follow-on readiness plan plus the simdjson v1-readiness plan (Section 7).
Edits: every canonical roadmap, the Production task management section in each
repository's `AGENTS.md`/`CLAUDE.md` (Section 8), per-repo follow-on status
sections where the repository's rules permit (Section 7), and the simdlogs
ledger (`docs/release-readiness.md` and `docs/roadmap.md`).

---

## 1. Goals

1. **One index, not one roadmap.** A reader can find every repository's
   production design, production plan, roadmap, and verification document from
   one place, and can see the shared family gates and the release waves without
   reading eleven roadmaps.
2. **Canonical truth stays local.** Every per-task status, every shipped claim,
   and every measurement lives in the owning repository. The index never
   restates them.
3. **A single task-state vocabulary.** The seven states in Section 5 map onto
   every repository's local statuses, so cross-repo status is comparable and an
   agent never has to guess what "in progress" or "done" means in another repo.
   The mapping never rewrites historical task text or IDs.
4. **A single repository-maturity taxonomy.** R0-R5 classifies product
   maturity separately from task status and is sourced to each repository's
   own evidence.
5. **Production readiness is evidence-gated.** Work is staged with task IDs,
   status, and evidence requirements; a task changes state only on the evidence
   its owning repo's gates define.
6. **Competitive work is gated on workload, compatibility, allocation,
   dispatch, and benchmark evidence** - not on feature-count parity with any
   other library.
7. **Dirty work is preserved.** Measurements, rejected variants, and
   decisions recorded in `docs/wrong.md`, completed plans, and uncommitted
   user changes are historical records: never rewritten, never deleted, never
   reverted, always linked.
8. **The first wave is corrective and stabilization work** in three
   repositories (Section 9), each item as a documented task with a status.

## 2. Non-goals

1. **Not a new roadmap.** ROADMAP.md in GO_SIMD and simdmetrics,
   `docs/roadmap.md` in the other nine repositories, remains canonical for that
   repository. The index states no dates, no promises, and no per-task status.
2. **Not a build system or CI.** No workflows, no runner changes, no release
   machinery in this documentation work. Shared gates are *listed* in the
   index; they are *executed* by each repo's own gates.
3. **Not an umbrella module.** No go.mod changes, no dependency edges between
   family repositories, no release train with lockstep versions. Each
   repository keeps its own release cadence; the index records waves as
   ordering information only.
4. **Not a committee or ownership layer.** The primary agent reviewing family
   documentation (Section 12) reviews diffs and claims; it does not approve
   per-repo implementation work or hold any repository's gate.
5. **Not a feature-parity tracker.** Competitive gaps are recorded as workload
   and evidence gaps (Section 6), never as "library X has N more functions".
6. **Not shipped claims.** This documentation task creates no pushes, tags, or
   release claims in any repository (Section 13). Drafting and review are
   commit-free; the user separately authorized final preservation commits on
   2026-08-26 after review and verification.
7. **Not a fork of the execution discipline.** The existing rules in each
   repo's `AGENTS.md`/`CLAUDE.md` - disassemble first, the 8.3% layout floor,
   interleaved A/B, bare gates, `docs/wrong.md` - are adopted by reference and
   not restated per-repo here.

## 3. Information architecture

```
GO_SIMD/
  docs/plans/2026-08-24-simd-family-production-readiness-design.md   (this record)
  docs/plans/2026-08-24-simd-family-production-readiness.md          (the index; created in wave 0)
  AGENTS.md / CLAUDE.md                      (+ Production task management section)
simdblas/ ... simdvec/                       (each own repo, own docs)
  docs/roadmap.md                            canonical roadmap (GO_SIMD, simdmetrics: ROADMAP.md)
  docs/plans/2026-08-13-*-production*.md      historical design/plan records (preserved, never renumbered)
  <follow-on status sections or follow-on plans per Section 7>
  docs/verification.md                       gates
  docs/wrong.md                              historical measurement record
  AGENTS.md / CLAUDE.md                      (+ Production task management section)
```

**The family index** (`GO_SIMD/docs/plans/2026-08-24-simd-family-
production-readiness.md`) contains, and only contains:

1. **Taxonomy** - the eleven repositories, one line each: module path, one-line
   product boundary, R0-R5 maturity and evidence date, link to the repo's
   canonical roadmap, production design, production plan, verification
   document, and `docs/wrong.md`.
2. **Release waves** - ordering information from Section 9 and the per-repo
   plans: which repositories are in the first wave, what their release exits
   are, and the rule that no repository's release is blocked by another's.
3. **Links and shared gates** - the common family gates: differential
   conformance against the portable reference, bench-check with stored
   baselines, race/vet, cross-architecture tiers, no-panic/no-hang limits,
   fuzz with timeouts, release verification sets per repo.
4. **Status legend** - the seven states of Section 5 and the transition rule.
5. **Quiet-host protocol** - the coordination rule from Section 6: one primary
   coordinator, no concurrent agents/builds/tests during measurement, fixed
   source identity, load below 1 before and throughout, repository-native load
   gates, bounded waits and benchmark timeouts, recorded CPU/governor/affinity
   provenance, contaminated-run rejection, and restoration of anything paused.
   The index records the protocol; wave 0 does not execute it.

The index never contains per-task status, never restates a roadmap item, and
never states a speed claim. Where it links a claim (for example a README
benchmark), it links, it does not quote.

**Per-repo production plans.** Historical `2026-08-13-*` plans are records:
their task IDs and text are preserved and never renumbered, rewritten, or
edited for status. Follow-on work is staged where each repository's rules
permit, per Section 7: appended status/follow-on sections for GO_SIMD,
simdblas, simdcsv, simdvec, simdhttp, simdcbor, and simdimage; separate
2026-08-24 readiness plans for simdjson and simdparquet (whose historical
plans stay untouched); the `docs/release-readiness.md` and `docs/roadmap.md`
ledger for simdlogs (whose completed plan remains a record); and new
2026-08-24 design/plan files for simdmetrics. Every canonical roadmap gains
the high-level open work and a link to its owning ledger, without copying
per-task status. New plan files are created only where Section 7 names them.

## 4. Ownership of truth

| Truth | Owned by | Recorded in |
|---|---|---|
| What is built and shipped | each repository | its own code, tests, changelog |
| What is not built and why | each repository | its own ROADMAP / `docs/roadmap.md` |
| Numerical/API contract | each repository | its own design record and LLDs |
| Task status | each repository | its own ledger per Section 7: plan follow-on sections, follow-on plans, or `docs/release-readiness.md` |
| Measurements and rejections | each repository | its own `docs/wrong.md` |
| Family taxonomy and waves | GO_SIMD only | `docs/plans/2026-08-24-simd-family-production-readiness.md` |
| Repository maturity | each repository supplies evidence; GO_SIMD indexes it | dated R0-R5 line in the family index linked to owning evidence |
| Task-state vocabulary | this record | this file, mapped into per-repo AGENTS/CLAUDE per Section 8 |

Conflicts are resolved by ownership: where the index and a repository disagree,
the repository's own document wins and the index is corrected (Section 11).
Nothing in the index is authoritative about a repository's content; the index
is a link collection plus shared vocabulary.

## 5. Shared vocabularies

### 5.1 Repository maturity

Repository maturity and task state answer different questions. R0-R5 describes
the product boundary at a dated snapshot; it never implies that every roadmap
task is closed:

| Level | Meaning |
|---|---|
| `R0` | Unsecured: ownership, corruption, resource-bound, or ABI risks prevent production use |
| `R1` | Prototype: a useful vertical slice exists, but core format/API/interop work remains |
| `R2` | Feature-complete alpha: intended boundary exists, but compatibility and release evidence are incomplete |
| `R3` | Release candidate: boundary and gates are defined; remaining work is release evidence, stabilization, or rehearsal |
| `R4` | Production release: a stable release exists and its documented production gates passed |
| `R5` | Maintained production: R4 plus a completed patch/upgrade cycle, observed release automation, and current reproducible evidence |

The 2026-08-24 evidence snapshot used to seed the index is:

| Repository | Maturity | Basis |
|---|---|---|
| GO_SIMD | `R4` released; current dirty workspace `R3` | v1.21.1 exists; corrective fuzz/timeout/docs changes are uncommitted |
| simdblas | `R3` | v1.1.0 identity and documentation/release evidence need reconciliation |
| simdcsv | `R3` | v0.2.1 exists; v1 contracts and complete gates remain |
| simdvec | `R3` | v0.2.0 exists; API consolidation and v1 evidence remain |
| simdjson | `R2` | v0.7.0 is released; the v0.8 candidate is a large dirty stabilization tree |
| simdhttp | `R2` | intended parser/router surface exists; root fuzz and first-release evidence remain |
| simdcbor | `R2` | codec surface exists; representation, limits, interoperability, and first release remain |
| simdparquet | `R1` | useful structural subset exists; typed values, dictionary/index, codec, and interop work remain |
| simdimage | `R0` | untracked FFmpeg/runtime implementation still has ownership, callback, ABI, and platform risks |
| simdmetrics | `R0` | large dirty implementation lacks repository governance and complete corruption/resource contracts |
| simdlogs | `R3` | phases 0-10 are local; quiet evidence, observed CI, soak, and release rehearsal remain |

The family index copies these levels only as a dated, linked snapshot. A later
change updates a level only when the owning repository records the evidence;
the index cannot promote a repository by itself.

### 5.2 Task-state vocabulary and transitions

Seven states, used in every per-repo ledger (plan follow-on sections,
follow-on plans, release-readiness documents) and nowhere else; status never
lives in the index:

| State | Meaning | Entry requires |
|---|---|---|
| `open` | Recorded in the owning ledger, not selected for execution | task ID and evidence bar recorded in the owning ledger |
| `staged` | Selected for execution with scope, steps, and prerequisites ready | executing plan or equivalent scoped work record linked from the ledger |
| `in-progress` | A task is being executed | executing-plans session started on the task |
| `blocked` | Work stopped by an external dependency | named blocker (toolchain, hardware, upstream, approval) |
| `evidence-complete` | Work done; evidence gathered; gates pending or being run | measurements/artifacts recorded |
| `shipped` | In code, tests, changelog, and docs | all owning-repo gates green; released or commit landed |
| `rejected` | Measured against and declined | measurement recorded in `docs/wrong.md` |

**Transition rules:**

1. Every transition is recorded by editing the owning ledger's status line
   and, where the transition is to `shipped` or `rejected`, the changelog or
   `docs/wrong.md` respectively. A transition with no edit did not happen.
   Historical task text and IDs are never edited for status; the status is an
   appended line or section.
2. `open -> staged` is the only way recorded work becomes executable; a task
   is never executed directly from `open`.
3. `in-progress -> blocked` names the blocker and the condition that clears it.
   `blocked -> in-progress` is allowed when the condition clears; the blocker
   line is kept as history.
4. `evidence-complete -> shipped` requires every gate the owning repo's
   verification document lists for that task, run bare.
5. `rejected` is **terminal**: the task is closed by the measurement. It may
   return to `open` only through a **documented reopen condition** - a
   paragraph in the owning `docs/wrong.md` entry stating what new evidence
   would reopen it (new ISA, new input class, new workload measurement).
   Without that paragraph, a rejected task stays rejected.
6. No task may be in two states. The index is never edited for status.

New task IDs use a repository and stream prefix, such as `SIMD-CORR-01`,
`JSON-V1-04`, and `LOGS-V1-02`. Historical `Task N` IDs remain unchanged;
the stream prefix makes every follow-on ID unambiguous without renumbering
history.

## 6. Competitive-performance policy

Competitive work - comparing this family to other libraries, or closing a gap
another library exposes - is gated on five forms of evidence, matching the
existing kernel-evidence discipline in the simd production design record:

1. **Workload-gated.** The comparison is over concrete workloads (a corpus, a
   shape, a request mix), stated before measurement. A gap that no workload in
   the repo exercises is a roadmap note, not a task.
2. **Compatibility-gated.** The other library's behavior is a *contract* only
   where this family promises compatibility (for example stdlib-compatible
   marshal, net/http differential, BLAS delegation). Outside promised
   compatibility, the other library is a workload generator, not an oracle.
3. **Allocation-gated.** A competitive claim counts allocations on the same
   per-request/per-record path this family's own rules require: zero
   allocations per element, caller-supplied buffers, sized slices. A win that
   allocates is not a win.
4. **Dispatch-gated.** Where SIMD kernels are involved, the dispatch must be
   shown to reach the kernel at runtime for the compared shape (the tier
   assertion in the existing conformance suites); a benchmark that never
   reaches the kernel compares the wrong thing.
5. **Benchmark-gated.** Interleaved A/B in one session, minimum compared,
   quiet machine, the 8.3% layout floor, `perf stat -e instructions:u,cycles:u`
   below it, `-benchmem` above it. Where a repo has a stored-baseline gate
   (bench-check), it runs bare and green before and after.

### Quiet-host coordination

The primary agent coordinates every benchmark window on this machine. This is
part of the benchmark task, not an external blocker:

1. Finish or pause every other agent, test, build, code-generation, and
   benchmark command before the window. Never run repository work in parallel
   with a measurement. Unknown user-owned processes are not killed; if they
   keep the host busy, wait with a bounded timeout or ask before stopping them.
2. Record source identity (commit plus dirty state), CPU model and tier, Go
   version, kernel, logical CPU count, `GOMAXPROCS`, CPU affinity, governor, and
   the one-, five-, and fifteen-minute load averages beside the result.
3. Require the one-minute load average to remain below 1 before and throughout
   every publishable run. Use the repository's existing gate where present:
   simdlogs `requireQuiet`, simdjson `benchcheck -maxload 1` (overriding the
   tool's current default of 4), and GO_SIMD's pinned-core benchmark harness.
   The simdlogs gate skips rather than fails above the threshold, so
   `SIMDLOGS_BENCH_NOISY` remains unset and the evidence must show that the
   benchmark executed rather than skipped. A wait loop is always bounded and
   every benchmark command carries an explicit timeout.
4. Measure a fixed source state. Clean revisions use a detached worktree;
   dirty candidates use an immutable recorded snapshot of the exact modified
   and untracked inputs rather than a clean checkout that omits the candidate.
5. Keep the compared variants interleaved in one window and use equal sample
   counts. If load rises to 1 or higher, another workload starts, affinity or
   governor changes, or the fixed state changes, discard the affected run and
   restart only after the host is quiet again.
6. Restore every service or setting explicitly paused for the window, even
   after timeout or failure. The coordinator reports the commands, provenance,
   accepted runs, and discarded runs in the owning repository's evidence
   record; temporary output does not become a published baseline by default.

**Feature-count parity is explicitly not a gate.** A competitor having more
functions, more formats, or more encodings is not a gap by itself; the gap is
a workload, an allocation, a dispatch, or a benchmark result. Competitive
sections of per-repo docs that count features are converted to workload and
evidence language during the first wave.

**Rejected competitive work follows the `rejected` state of Section 5**: the
measurement that rejects it is recorded in the owning `docs/wrong.md` with a
reopen condition.

## 7. Per-repo document strategy (all eleven)

Common to every repository: keep all existing documents; update the canonical
roadmap with the high-level production-readiness gaps and a link to the owning
ledger; add the Production task management section to `AGENTS.md` and
`CLAUDE.md` (Section 8); stage follow-on work per the repository's row below;
never create a parallel roadmap. Roadmaps do not duplicate per-task status.
Historical plans are preserved with their task IDs and text intact; follow-on
sections and plans append, they never renumber or rewrite.

| Repo | Current docs | First-wave action |
|---|---|---|
| `github.com/sebishogun/simd` (GO_SIMD) | Full set: README, architecture, LLDs, ROADMAP, plans 08-13, verification, wrong | Append a status/follow-on section to `2026-08-13-simd-production.md` (no renumbering); create `docs/plans/2026-08-24-simd-family-production-readiness.md` (the index); corrective slice tasks (Section 9.1) |
| `simdblas` | Full set incl. plans 08-13 | Append status/follow-on section without renumbering; index entry |
| `simdcsv` | Full set incl. plans 08-13 | Append status/follow-on section without renumbering; index entry |
| `simdvec` | Full set incl. plans 08-13 | Append status/follow-on section without renumbering; index entry |
| `simdjson` | Full set incl. plans 08-13 | Historical plan stays untouched; create `docs/plans/2026-08-24-simdjson-v1-readiness.md` for v0.8 stabilization then the v1 compatibility/release path (Section 9.2) |
| `simdhttp` | Full set incl. plans 08-13 | Append status/follow-on section without renumbering; index entry |
| `simdcbor` | Full set incl. plans 08-13 | Append status/follow-on section without renumbering; index entry |
| `simdparquet` | Full set incl. plans 08-13 | Historical plan stays **untouched**; create a new follow-on readiness plan `docs/plans/2026-08-24-simdparquet-readiness.md`; index entry |
| `simdimage` | Full set incl. plans 08-13 | Append status/follow-on section without renumbering; index entry |
| `simdmetrics` | `docs/design.md`, `docs/wrong.md`, README; no AGENTS/CLAUDE, no ROADMAP, no plans | **Missing-governance repository.** Create only: `AGENTS.md`, `CLAUDE.md`, `ROADMAP.md`, `docs/verification.md`, `CHANGELOG.md`, `docs/plans/2026-08-24-simdmetrics-production-design.md`, `docs/plans/2026-08-24-simdmetrics-production.md`. All new files dated 2026-08-24 - nothing backdated. `docs/design.md` remains the architecture source and is linked, not duplicated as `docs/architecture.md`. `docs/wrong.md` is retained. Index entry |
| `simdlogs` | Full set incl. plans 08-13 + ecosystem docs; `docs/release-readiness.md` exists | No historical plan edit. Wave 0 creates the task ledger in `docs/release-readiness.md` and `docs/roadmap.md`; `docs/plans/2026-08-13-simdlogs-production.md` remains the completed implementation record, and `docs/plans/2026-08-13-simd-family-production-documentation.md` remains the superseded historical family-documentation record. The GO_SIMD index now owns family coordination. Wave tasks in Section 9.3; index entry |

Each row also updates that repository's canonical roadmap as described above.
The simdmetrics production design derives from its current `docs/design.md`
(shipped surface, non-goals, evidence bar); the staged plan follows the family
template with task IDs, status, and evidence. Its `docs/wrong.md` is retained
and linked, never rewritten.

Where a repository may append a follow-on section, the section is appended
below the historical task text - existing task IDs and prose are not rewritten
into the new status vocabulary; the status vocabulary maps onto the existing
tasks without changing them.

## 8. Production task management section (AGENTS.md / CLAUDE.md)

Every repository's `AGENTS.md` and `CLAUDE.md` (created for simdmetrics) gains
a short section, identical in substance across the family and using each
repository's actual document paths:

- **Local authority.** The repo's canonical roadmap - `ROADMAP.md` in GO_SIMD
  and simdmetrics, `docs/roadmap.md` in the other nine - is canonical; the
  dated production plan and ledger in `docs/plans/` (or
  `docs/release-readiness.md` for simdlogs) is the only staging area;
  `docs/wrong.md` is the only record of rejections. The family index in
  `github.com/sebishogun/simd` (`docs/plans/2026-08-24-simd-family-
  production-readiness.md`) is a link collection and never overrides local
  truth.
- **One task ID at a time.** Work is executed one task at a time, identified
  by its ID from the local ledger (for example `SIMD-CORR-01`,
  `JSON-V1-04`). The stream prefix distinguishes new work from historical
  `Task N` IDs. A session touching implementation work names its task ID in
  its first message; a session without a task ID touches no implementation
  files.
- **State transitions.** The seven states of Section 5 map onto the repo's
  local statuses without rewriting historical tasks: each existing task is
  assigned a state by an appended status line, and the transition rules of
  Section 5 apply. The terminal-rejection rule holds: a transition is an edit
  in the local ledger (and changelog or `docs/wrong.md` for
  `shipped`/`rejected`).
- **Gate rule.** Before any commit: the repo's gate set from its
  `docs/verification.md`, run bare (no `tail` without `pipefail`), with
  explicit timeouts; a hung test binary is a leak alarm, not a retry
  candidate.

## 9. First implementation wave

Wave 0 (documentation) and the first implementation wave are strictly
separated: wave 0 is this documentation task (Section 13). The first
*implementation* wave is three repositories, in this order; each task carries
its ID, status, and evidence in the owning repository's ledger per Section 7.
No repository's tasks block another's. A task that requires a commit, push,
tag, or release stops at the preceding evidence checkpoint until the user
separately requests that operation; inclusion in this wave is not permission
to perform it.

### 9.1 `simd` (GO_SIMD): corrective fuzz / timeouts / docs slice

- **SIMD-CORR-01** - fuzz corrective: review the fuzz targets in
  `internal/conformance/` against the differential contract (NaN payloads,
  IEEE specials, adversarial lengths); fix or record; evidence is the fuzz
  run with timeout, bare, green or a `docs/wrong.md` entry.
- **SIMD-CORR-02** - timeout corrective: every test and fuzz invocation in the
  verification set carries an explicit timeout; evidence is the verification
  document listing each command with its timeout and one bare green run.
- **SIMD-CORR-03** - docs corrective: reconcile `docs/platforms.md` counts,
  README claims, and ROADMAP status against sources via the docs tests,
  including the authoritative ledger row's named contradictions: ROADMAP's
  "door closed twice" sentence; entry 80's unexplained 1,080-case denominator,
  dependent arithmetic, and four-line objdump attribution; and entry 81's
  historical 11-seed label plus stale `f.Fatalf` wording against the current
  three-seed corpus and final `f.Skip` guard; no claim edits without the
  machine-checked counts.
Each task moves through the states of Section 5. The task-specific evidence
moves a row to `evidence-complete`; `shipped` additionally requires the repo's
standard gates (go test with timeout, make verify, make check-emission), bare.

### 9.2 `simdjson`: v0.8 stabilization, then v1 compatibility/release path

The v0.8 candidate exists today as the dirty working tree (the in-place
uncommitted state covering `register.go`, `differential_test.go`,
`skip_validation_test.go`, the `bench/` tools, untracked snapshot JSONs, and
modified figures). The wave preserves that tree in full and organizes it; it
is the current state, not a starting point to be rebuilt.

- **JSON-V1-01** - preserve and organize the dirty v0.8 candidate: inventory
  the in-place tree (every modified and untracked file) into the stabilization
  plan, preserving every user change; evidence is the inventory recorded in
  the plan.
- **JSON-V1-02** - commit-independent stabilization plan: each stabilization
  task is executable from the dirty-tree state as-is, whether or not the
  candidate is committed first; no task's precondition is "a clean checkout".
  Evidence is the plan's per-task preconditions.
- **JSON-V1-03** - current snapshot/figure regeneration: coordinate a quiet
  host window under Section 6, then regenerate the benchmark snapshots
  (`docs/bench/compare-*.json`) and figures from the recorded dirty candidate
  with provenance (machine, commit plus dirty-state identity, harness, date);
  evidence is the regenerated snapshots and their provenance record.
- **JSON-V1-04** - explicit deltas: record exactly what the candidate changes
  in `RegisterEncoder` behavior, error taxonomy, and stdlib compatibility
  against the previous release; evidence is the delta record in the plan.
- **JSON-V1-05** - full gates: run the repository's full functional gate set
  bare with explicit timeouts, then run `benchcheck -maxload 1` against the
  stored baseline in a Section 6 quiet window. A failure is diagnosed with
  same-window interleaved A/B; `bench-update` is never used to clear a failure
  and is allowed only after a recorded expected-change reason, equal sample
  counts, and a second quiet `benchcheck`. Evidence is the commands and green
  results.
- **JSON-V1-06** - v0.8: the tag and changelog entry; evidence is the tag and
  changelog.
- **JSON-V1-07** - compatibility freeze and patch-cycle evidence: after v0.8,
  the compatibility surface is frozen and holds through at least one patch
  cycle (the regression suite green across a patch release) before the v1.0
  path proceeds; evidence is the patch-cycle run.
- **JSON-V1-08** - workload-backed ecosystem-gap decision: compare the
  supported boundary against the named Go, C++, and Rust peers using concrete
  corpora and operations; classify each material gap as a v1 blocker,
  post-v1 work, or rejected with evidence. The roadmap records the decisions,
  not feature counts; implementation requires its own approved task.

### 9.3 `simdlogs`: benchmark provenance / CI / soak / release blockers to v1.0

Phases 0-10 of `docs/plans/2026-08-13-simdlogs-production.md` are committed
locally and are not open. Wave 0 creates the task ledger in
`docs/release-readiness.md` and `docs/roadmap.md`; it does not exist there yet.
The remaining work staged into that ledger is:

- **LOGS-V1-01** - quiet benchmark provenance: the primary agent coordinates
  a quiet host window under Section 6, then re-measures the published tables
  under `requireQuiet` with a machine/commit record, per the benchmark
  contract; evidence is the corrected tables and their provenance.
- **LOGS-V1-02** - push/observed CI: push local main and observe the CI run
  of the documented gate set; evidence is the observed CI result.
- **LOGS-V1-03** - long fuzz/soak: fuzz and soak runs beyond the current
  soak, with durations and exit criteria from `docs/release-readiness.md`;
  evidence is the completed runs.
- **LOGS-V1-04** - stale docs/comments corrective: reconcile stale
  documentation and comments against shipped code; evidence is the diff and
  the green doc gates.
- **LOGS-V1-05** - bounded-ingest decision: the bounded-ingest open decision
  is taken with measurement and recorded - implemented, or rejected with a
  reopen condition in `docs/wrong.md`; evidence is the decision record.
- **LOGS-V1-06** - release rehearsal: run the release path end-to-end without
  tagging; evidence is the rehearsal run.
- **LOGS-V1-07** - v1.0: the full production exit, phases 0-10 plus the
  release gate set green; evidence is the release gates green.
- **LOGS-V1-08** - workload-backed ecosystem-gap decision: reconcile
  `docs/vl-parity.md` and the ecosystem documents against concrete ingest,
  query, storage, recovery, and operations workloads from VictoriaLogs, Loki,
  Elasticsearch/OpenSearch, ClickHouse, and Vector. Classify each material
  gap as a v1 blocker, post-v1 work, or rejected with evidence; no feature is
  added merely for parity.

`LOGS-IO-01` is recorded outside the first implementation wave with state
`open`; `deferred` and `non-v1-blocking` are scope notes, not task states. It is
not rejected and not a v1 blocker. Queries are mmap-backed and the
durable ingest path retains file, directory, and manifest sync barriers. First
instrument the ingest stages. Prototype io_uring only if explicit I/O waits are
at least 30% of durable-ingest time; retain it only for a repeatable improvement
of at least 20% in end-to-end throughput or p99 latency while durability and
conformance gates stay green. Without that evidence it remains deferred and no
`docs/wrong.md` measurement entry is created.

Wave ordering is informational: no repository's tasks block another's, and no
task waits on another repository's tag.

## 10. Error and conflict handling

1. **Index/repo disagreement.** The repository's own document wins. The index
   is corrected in a docs change; the disagreement and its resolution are
   noted in the index's diff, never in the repo's history.
2. **Status disagreement.** Two documents claiming different states for one
   task: the owning ledger's status line is truth; the other document (index,
   README, changelog) is wrong and fixed. `shipped` requires the changelog
   entry; a changelog without a ledger status is a missing edit, not a claim.
3. **Claim disagreement.** A README claim contradicted by the code or tests is
   a bug in the README (or the claim's evidence), and both the corrective task
   and the measurement go through the owning repo's process - never silently
   rewritten.
4. **Rejected-task disagreement.** A task marked `rejected` in `docs/wrong.md`
   and still listed as open in a roadmap is a documentation bug; the roadmap
   gains the rejection reference, the wrong.md entry keeps its reopen
   condition, and the task does not return to `open` unless the condition is
   met and recorded.
5. **Merge conflicts in extended documents.** Follow-on edits happen in place
   in the owning repository, one repository at a time, on the documents
   Section 7 names. Conflicts resolve by preserving the newer main content and
   any user changes, then appending the follow-on section; nothing is reverted.
6. **Dirty working trees.** Current state in simdjson, simdparquet,
   simdimage, simdmetrics, and GO_SIMD is the dirty tree, not a clean
   worktree; a docs-only clean worktree cannot represent it. Wave-0 edits
   there are surgical: read the exact diff first (`git status --short`, the
   full diff, and the untracked inventory), touch only the files Section 7
   names for that repository, and preserve every user change. No revert,
   reset, stash, checkout, or cleanup, in any repository. Corrective tasks
   for dirty work are staged in the owning ledger, never performed as
   drive-by edits.
7. **Unclear ownership.** A document or claim that cannot be assigned to an
   owning repository is reported in the review (Section 12) and parked in the
   index as unresolved until the owning repo claims it.

## 11. Verification

Two scopes, kept separate:

1. **This design record.** Creating this record runs no gates: no builds, no
   tests, no link checks by machine. Its verification is the review workflow
   (Section 12) and the claim checks below, done by inspection.
2. **Wave-0 document implementation.** Every per-repository document change
   runs that repository's document, link, and test gates with explicit
   timeouts, bare: the docs-count tests, the link tests, `go test` with a
   timeout where the repo's verification document requires it, and
   `git diff --check`. No benchmarks, no codegen, no release gates - the
   wave is documentation. For simdmetrics, wave 0 first derives
   `docs/verification.md` from commands that already exist in its `scripts/`,
   `.github/workflows/ci.yml`, and package tests; primary review confirms the
   inventory before those documented gates run. The result is not used to
   invent an easier gate after seeing a failure.

Then, by inspection:

3. **Link check.** Every link in the index resolves to the stated file in the
   stated repository, checked by hand (the index is not in the machine-gated
   link set of any repo).
4. **Claim check.** Every current-state sentence in the index and in per-repo
   follow-on documents is sourced to a file in the owning repo (code, tests,
   docs). Un-sourced sentences are marked as planned.
5. **Gate inventory.** The shared-gates section of the index lists only gates
   that exist in the linked `docs/verification.md` of a repository; a gate
   that does not exist there is not listed.
6. **Status vocabulary check.** Every status line in a per-repo ledger uses
   one of the seven states of Section 5, every `rejected` line has a reopen
   condition, and no historical task ID or text was changed.
7. **Wave check.** Wave-0 artifacts contain no implementation change and no
   release claim; drafting/review preceded the separately authorized final
   preservation commits; per-repo diffs touched only the documents the
   per-repo strategy (Section 7) names; user changes in dirty trees are
   bit-preserved (verified by diff review, never by reset).

## 12. Review workflow

1. **Drafting.** Per-repo documentation (roadmap updates, follow-on sections,
   the simdjson and simdparquet follow-on plans, simdmetrics governance files,
   index draft) is drafted by
   DeepSeek agents, one repository per agent, with a strict file scope: the
   exact files Section 7 names for that repository, and nothing else. In the
   five dirty repositories (simdjson, simdparquet, simdimage, simdmetrics,
   GO_SIMD) drafting happens in place: the agent reads the exact current diff
   before editing, preserves every user change, and never reverts, resets,
   stashes, checks out, or cleans up. The task-state vocabulary and Section
   7's strategy are in the agent prompt.
2. **Primary-agent review.** The primary agent reviews every diff before it
   is accepted: links resolve, current-state claims are source-backed,
   shipped-versus-planned language is separated, status lines use the
   vocabulary without touching historical IDs or text, only the scoped files
   changed, and user changes are preserved. A diff failing any of these
   returns to the drafting agent with the specific defect.
3. **Cross-repo consistency review.** After all diffs, one review pass checks
   the family as a whole: every repo appears in the index once with correct
   links; every repo's AGENTS/CLAUDE has the Section 8 section; no repo
   except simdmetrics lacks governance files; no rejected task appears as
   open anywhere; no duplicate task status exists in the index.
4. **Interleaved review of claims.** Where two repos' documents make the same
   claim (dependency versions, shared contract), the claims are checked
   against each other and against the sources in the same pass.
5. **Review record.** Review findings and their resolutions are reported in
   the terminal, per repository, with file paths; no artifact document is
   produced (the repository record convention applies).

## 13. What this documentation task does and does not do

**Does:** create this record; create the family index at
`docs/plans/2026-08-24-simd-family-production-readiness.md`; append per-repo
follow-on status sections where Section 7 permits; update every canonical
roadmap; create the simdjson and simdparquet follow-on readiness plans; edit
the simdlogs ledger in
`docs/release-readiness.md` and `docs/roadmap.md`; add the Section 8 section
to per-repo AGENTS/CLAUDE; create the simdmetrics governance files
(AGENTS.md, CLAUDE.md, ROADMAP.md, docs/verification.md, CHANGELOG.md, and
the 2026-08-24 design/plan pair); draft per-repo docs via DeepSeek agents and
review them per Section 12; run the owning repository's document, link, and
test gates with explicit timeouts for the wave-0 implementation.

**Does not:** push, tag, or release in any repository; create this record under
any gate (the design-record creation runs none); run benchmarks,
codegen, or release gates; modify Go source, generated files, go.mod/go.sum,
Makefiles, workflows, or benchmark artifacts; create a parallel roadmap;
state a date promise; restate per-task status in the index; mark any task
`shipped`; revert, reset, stash, checkout, or clean up any working tree;
touch a document Section 7 does not name for that repository.

**Execution amendment (2026-08-26):** after drafting and review completed, the
user separately authorized final preservation commits for the current state of
each repository. That authorization does not include push, tag, publish, or
release, and it does not make any open ledger item shipped.

Current state and future work are separated throughout: this record, the
index, and the follow-on documents describe current state only where a linked
owning-repo document (or the dirty tree itself) states it, and everything
staged here is labeled future work to be executed through the owning
repository's executing-plans workflow.
