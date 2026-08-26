# SIMD family production readiness: family index

This is the family coordination index for the eleven SIMD-family
repositories. It is a **noncanonical link collection**: every repository's
canonical roadmap, per-task status, shipped claim, and measurement live in
that repository's own documents. This file states no per-task status, no
roadmap item, no speed claim, and no benchmark number; where a claim is
relevant it is linked, not quoted.

It was created in wave 0 of the family documentation plan
(`docs/plans/2026-08-24-simd-family-documentation-implementation.md`) from the
maturity snapshot in the coordination design record
(`docs/plans/2026-08-24-simd-family-production-readiness-design.md`,
Sections 5.1 and 6). Levels below are a dated snapshot: the index cannot
promote a repository; a level changes only when the owning repository records
the evidence.

## 1. Taxonomy

All eleven repositories, once each. Maturity levels and their bases are the
2026-08-24 snapshot from the design record Section 5.1, confirmed against each
repository's own roadmap during wave 0. Links use each repository's `main`
branch.

| Repository (module path) | Product boundary (one line) | Maturity, 2026-08-24 | Links |
|---|---|---|---|
| `github.com/sebishogun/simd` | Whole-slice SIMD primitives for numeric, text, and columnar work; runtime-selected generated kernels plus a portable Go path | R4 released (v1.21.1); the 2026-08-24 workspace snapshot was R3 with corrective fuzz/timeout/docs changes | [roadmap](https://github.com/sebishogun/simd/blob/main/ROADMAP.md) - [design](https://github.com/sebishogun/simd/blob/main/docs/plans/2026-08-13-simd-production-design.md) - [ledger](https://github.com/sebishogun/simd/blob/main/docs/plans/2026-08-13-simd-production.md#follow-on-production-readiness-ledger-appended-2026-08-24) - [verification](https://github.com/sebishogun/simd/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simd/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdblas` | BLAS linear-algebra routines (gemm/gemv/syrk/complex level-1) as guard+delegate over simd kernels | R3 - v1.1.0 identity and documentation/release evidence need reconciliation | [roadmap](https://github.com/sebishogun/simdblas/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdblas/blob/main/docs/plans/2026-08-13-simdblas-production-design.md) - [ledger](https://github.com/sebishogun/simdblas/blob/main/docs/plans/2026-08-13-simdblas-production.md#follow-on-ledger) - [verification](https://github.com/sebishogun/simdblas/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdblas/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdcsv` | SIMD CSV parser with encoding/csv delegation on the quoted/short spans where compatibility is promised | R3 - v0.2.1 exists; v1 contracts and complete gates remain | [roadmap](https://github.com/sebishogun/simdcsv/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdcsv/blob/main/docs/plans/2026-08-13-simdcsv-production-design.md) - [ledger](https://github.com/sebishogun/simdcsv/blob/main/docs/plans/2026-08-13-simdcsv-production.md#follow-on-ledger) - [verification](https://github.com/sebishogun/simdcsv/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdcsv/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdvec` | Exact flat vector search over float32 embeddings - one `simd.GemvParallelInto` matrix-vector product per query | R3 - v0.2.0 exists; API consolidation and v1 evidence remain | [roadmap](https://github.com/sebishogun/simdvec/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdvec/blob/main/docs/plans/2026-08-13-simdvec-production-design.md) - [ledger](https://github.com/sebishogun/simdvec/blob/main/docs/plans/2026-08-13-simdvec-production.md#the-v1-production-ledger) - [verification](https://github.com/sebishogun/simdvec/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdvec/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdjson` | encoding/json-compatible SIMD JSON library; the drop-in bar is the promised compatibility surface | R2 - v0.7.0 is released; the v0.8 candidate is a large dirty stabilization tree | [roadmap](https://github.com/sebishogun/simdjson/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdjson/blob/main/docs/plans/2026-08-13-simdjson-production-design.md) - [ledger](https://github.com/sebishogun/simdjson/blob/main/docs/plans/2026-08-24-simdjson-v1-readiness.md#5-ledger-json-v1-0108) - [verification](https://github.com/sebishogun/simdjson/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdjson/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdhttp` | net/http-compatible HTTP parser/router over SIMD; the server loop stays outside the boundary unless design evidence moves it in | R2 - intended parser/router surface exists; root fuzz and first-release evidence remain | [roadmap](https://github.com/sebishogun/simdhttp/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdhttp/blob/main/docs/plans/2026-08-13-simdhttp-production-design.md) - [ledger](https://github.com/sebishogun/simdhttp/blob/main/docs/plans/2026-08-13-simdhttp-production.md#follow-on-production-readiness-ledger-appended-2026-08-24) - [verification](https://github.com/sebishogun/simdhttp/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdhttp/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdcbor` | RFC 8949 CBOR codec with a JSON-shaped shipped API adapter (`Unmarshal`, `Marshal`, `Skip`) | R2 - codec surface exists; representation, limits, interoperability, and first release remain | [roadmap](https://github.com/sebishogun/simdcbor/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdcbor/blob/main/docs/plans/2026-08-13-simdcbor-production-design.md) - [ledger](https://github.com/sebishogun/simdcbor/blob/main/docs/plans/2026-08-13-simdcbor-production.md#production-readiness-ledger) - [verification](https://github.com/sebishogun/simdcbor/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdcbor/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdparquet` | Parquet reader/writer over SIMD (RLE/bitpacking, pages, bloom, indexes) with mandatory Arrow interoperability | R1 - useful structural subset exists; typed values, dictionary/index, codec, and interop work remain | [roadmap](https://github.com/sebishogun/simdparquet/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdparquet/blob/main/docs/plans/2026-08-13-simdparquet-production-design.md) - [ledger](https://github.com/sebishogun/simdparquet/blob/main/docs/plans/2026-08-24-simdparquet-readiness.md#ledger) - [verification](https://github.com/sebishogun/simdparquet/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdparquet/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdimage` | Image/media decode/encode over an FFmpeg runtime ABI seam plus owned frame/pixel-format types; what is owned versus the runtime is part of the boundary record | R0 - untracked FFmpeg/runtime implementation still has ownership, callback, ABI, and platform risks | [roadmap](https://github.com/sebishogun/simdimage/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdimage/blob/main/docs/plans/2026-08-13-simdimage-production-design.md) - [ledger](https://github.com/sebishogun/simdimage/blob/main/docs/plans/2026-08-13-simdimage-production.md#follow-on-ledger) - [verification](https://github.com/sebishogun/simdimage/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdimage/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdmetrics` | VictoriaMetrics-compatible time-series metrics database; pure Go, zero external dependencies | R0 - large dirty implementation lacks repository governance and complete corruption/resource contracts | [roadmap](https://github.com/sebishogun/simdmetrics/blob/main/ROADMAP.md) - [design](https://github.com/sebishogun/simdmetrics/blob/main/docs/plans/2026-08-24-simdmetrics-production-design.md) - [ledger](https://github.com/sebishogun/simdmetrics/blob/main/docs/plans/2026-08-24-simdmetrics-production.md#ledger) - [verification](https://github.com/sebishogun/simdmetrics/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdmetrics/blob/main/docs/wrong.md) |
| `github.com/sebishogun/simdlogs` | Logs-only database with authenticated tenancy, static application-level sharding, and the documented Elasticsearch subset; VictoriaLogs-compatible LogsQL | R3 - phases 0-10 are local; quiet evidence, observed CI, soak, and release rehearsal remain | [roadmap](https://github.com/sebishogun/simdlogs/blob/main/docs/roadmap.md) - [design](https://github.com/sebishogun/simdlogs/blob/main/docs/plans/2026-08-13-simdlogs-production-design.md) - [ledger](https://github.com/sebishogun/simdlogs/blob/main/docs/release-readiness.md#task-ledger) - [verification](https://github.com/sebishogun/simdlogs/blob/main/docs/verification.md) - [wrong](https://github.com/sebishogun/simdlogs/blob/main/docs/wrong.md) |

## 2. Release waves

Waves are ordering information only. Each repository keeps its own release
cadence; no repository's release is blocked by another's, and no task waits on
another repository's tag. Per-task state lives in the owning ledger, never
here.

- **Wave 0 (documentation, this wave).** Created this index and, in every
  repository, the production-readiness roadmap summaries, follow-on ledgers
  or readiness plans, and the production task-management sections. It records
  the quiet-host protocol (Section 5); it does not execute it.
- **First implementation wave.** Three repositories, in this order, each with
  its release exit stated only as ordering information and linked to its
  owning ledger:
  - `github.com/sebishogun/simd` - the corrective slice (fuzz differential,
    explicit timeouts, docs/count/claim reconciliation), the exit being the
    corrective slice green; separate R5 evidence work follows later. Ledger:
    [simd production follow-on](https://github.com/sebishogun/simd/blob/main/docs/plans/2026-08-13-simd-production.md#follow-on-production-readiness-ledger-appended-2026-08-24).
    GO_SIMD R5 evidence work is **not** part of the first implementation wave.
  - `github.com/sebishogun/simdjson` - v0.8 stabilization from the recorded
    dirty candidate, then the v1 compatibility/release path (compatibility
    freeze held through a patch cycle before v1.0). Ledger:
    [simdjson v1 readiness](https://github.com/sebishogun/simdjson/blob/main/docs/plans/2026-08-24-simdjson-v1-readiness.md).
  - `github.com/sebishogun/simdlogs` - v1 release evidence: the full
    production exit (phases 0-10 complete, plus the release gate set green).
    Ledger: [simdlogs task ledger](https://github.com/sebishogun/simdlogs/blob/main/docs/release-readiness.md#task-ledger).
    The deferred io_uring decision is **not** part of the first implementation
    wave.

All other repositories' production-readiness work is recorded only in their
own ledgers and is not part of the first implementation wave.

## 3. Links and shared gates

Every shared gate below exists in the linked verification document of the
repositories named for it. A gate that does not exist in a linked
`docs/verification.md` is not listed. The exact commands for each repository
are in that repository's verification document.

| Shared gate | Defined in the linked verification docs of |
|---|---|
| Differential conformance against a reference or a promised compatibility oracle | GO_SIMD (portable reference `internal/ref`), simdblas (gonum), simdcsv (encoding/csv overlap), simdvec (naive reference), simdjson (encoding/json), simdhttp (net/http route differential), simdcbor (RFC 8949 vectors), simdparquet (reference decoder plus Arrow/parquet-mr golden files), simdmetrics (pinned VictoriaMetrics binary and Prometheus PromQL), simdlogs (VictoriaLogs parity). simdimage has no behavioral oracle: FFmpeg is an ABI provider and the named libraries are workload and performance peers |
| Bench-check against stored baselines | GO_SIMD, simdjson, simdhttp, simdcbor, simdlogs, simdmetrics (hand-run quiet-bench with two-run agreement). simdblas and simdvec define benchmark harnesses without a committed stored-baseline gate; simdparquet's verification document states its bench-check is an output helper, not a regression gate |
| Race and vet | All eleven: `go test -race` and `go vet` are in every linked verification document |
| Cross-architecture tiers | All eleven: GO_SIMD (emulated lanes plus per-tier runs), simdblas, simdcsv, simdvec, simdjson (support and release matrix), simdhttp (cross-arch and tiers), simdcbor (race and cross-arch), simdparquet, simdimage (support matrix and hardware gates), simdmetrics (six-GOARCH cross build and vet), simdlogs |
| No-panic/no-hang limits | simdhttp, simdcbor, simdparquet, simdimage, simdmetrics, simdlogs, simdjson, simdcsv (fuzz and malformed-input surfaces). simdblas and simdvec verification documents define no fuzz or no-panic/no-hang gate |
| Fuzz with explicit timeouts | GO_SIMD, simdcsv, simdjson, simdhttp, simdcbor, simdparquet, simdimage, simdmetrics, simdlogs |
| Per-repository release verification sets | GO_SIMD, simdblas, simdcsv, simdvec, simdjson, simdcbor, simdparquet, simdimage, simdmetrics, simdlogs. simdhttp's first-release gate set is staged in its ledger, not yet defined in its verification document |

## 4. Status legend

Seven states, used in every per-repository ledger and nowhere else. Status
never lives in this index; a transition is an edit in the owning ledger, plus
the changelog or `docs/wrong.md` for `shipped` or `rejected`.

- `open` - recorded in the owning ledger, not selected for execution.
- `staged` - selected for execution with scope, steps, and prerequisites
  ready.
- `in-progress` - a task is being executed.
- `blocked` - work stopped by an external dependency; the blocker is named.
- `evidence-complete` - work done and evidence gathered; gates pending or
  being run.
- `shipped` - in code, tests, changelog, and docs, with the owning-repo gates
  green.
- `rejected` - measured against and declined, recorded in `docs/wrong.md`.
  `rejected` is **terminal**: a task returns to `open` only through a
  documented reopen condition in the owning `docs/wrong.md` entry. Without
  that paragraph, a rejected task stays rejected.

`open -> staged` is the only way recorded work becomes executable; a task is
never executed directly from `open`. No task may be in two states. Historical
task IDs and text are never edited for status.

## 5. Quiet-host protocol

Condensed from the coordination design record Section 6. The primary agent
coordinates every benchmark window on this machine; this is part of the
benchmark task, not an external blocker. Wave 0 records the protocol; the
first implementation wave executes it.

1. **Quiet-window coordination.** Finish or pause every other agent, test,
   build, code-generation, and benchmark command before the window. Never run
   repository work in parallel with a measurement. Unknown user-owned
   processes are not killed; if they keep the host busy, wait with a bounded
   timeout or ask before stopping them.
2. **Provenance record.** Record source identity (commit plus dirty state),
   CPU model and tier, Go version, kernel, logical CPU count, `GOMAXPROCS`,
   CPU affinity, governor, and the one-, five-, and fifteen-minute load
   averages beside the result.
3. **Load below 1.** Require the one-minute load average below 1 before and
   throughout every publishable run, using each repository's existing gate
   where present: simdlogs `requireQuiet`, simdjson `benchcheck -maxload 1`
   (overriding the tool's current default of 4), and GO_SIMD's pinned-core
   benchmark harness. The simdlogs gate skips rather than fails above the
   threshold, so `SIMDLOGS_BENCH_NOISY` stays unset and the evidence must show
   the benchmark executed rather than skipped. A wait loop is always bounded
   and every benchmark command carries an explicit timeout.
4. **Fixed source state.** Measure a fixed source state: clean revisions use
   a detached worktree; dirty candidates use an immutable recorded snapshot
   of the exact modified and untracked inputs rather than a clean checkout
   that omits the candidate.
5. **Discard and rerun.** Keep the compared variants interleaved in one
   window with equal sample counts. If load rises to 1 or higher, another
   workload starts, affinity or governor changes, or the fixed state changes,
   discard the affected run and restart only after the host is quiet again.
6. **Restore after.** Restore every service or setting explicitly paused for
   the window, even after timeout or failure. The coordinator reports the
   commands, provenance, accepted runs, and discarded runs in the owning
   repository's evidence record; temporary output does not become a published
   baseline by default.
