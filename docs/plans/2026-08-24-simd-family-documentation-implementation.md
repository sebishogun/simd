# SIMD Family Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute wave 0 of the family production-readiness design - per-repository documentation in all eleven SIMD-family repositories (governance files, roadmap summaries, ledgers, follow-on plans, agent-file sections) plus the GO_SIMD family index - as Markdown-only, user-change-preserving edits, with drafting and review commit-free.

**Architecture:** One fresh DeepSeek writer per repository, strict file scope per design Section 7, drafted in place against the exact current diff. GO_SIMD's writer does its local governance first and is resumed after all other repositories pass review to build the family index from final links. Every drafting task ends in a primary-review checkpoint instead of a commit; verification is serialized afterwards; a cross-repo review and a primary final review close the documentation wave; Task 18 performs the separately authorized preservation commits.

**Tech Stack:** Markdown; git status/diff for scope control and preservation checks; per-repo Go/make documentation gates with explicit timeouts; DeepSeek writers for drafting; no benchmarks, codegen, release gates, or quiet-machine windows in this wave.

---

## Ground rules (whole plan)

1. **No commits during drafting or review.** Every normal commit step in Tasks 1-17 is replaced by a **PRIMARY-REVIEW CHECKPOINT** (defined in each task). On 2026-08-26 the user separately authorized Task 18's final preservation commits after verification. Still prohibited in every repository: push, tag, release, reset, revert, stash, checkout, clean, deletion, and fetch.
2. **Preserve every user change.** The dirty trees (GO_SIMD, simdjson, simdparquet, simdimage, simdmetrics) are the current state. Writers read the exact diff before editing and touch only the files their task names. Task 1 captures the baseline in the session; every review checkpoint compares against it. Scoped dirty-roadmap edits preserve every pre-existing hunk and append new ones; they never rewrite an old hunk.
3. **Wave 0 is documentation only.** No benchmarks, no codegen, no release gates, no quiet-machine windows. The quiet-host protocol is *recorded* into the family index (Task 14) for later simdjson/simdlogs implementation waves; it is not executed here.
4. **No fabricated claims, no unverified numbers.** Every current-state sentence is source-backed (code, tests, or a linked doc); everything staged is labeled future work. This plan states no unverified numerical claim; writers quote a figure only after reading it in the source with its provenance.
5. **Oracle versus peer, explicitly.** A differential target is an **oracle** only where compatibility is promised: encoding/json drop-in (simdjson), encoding/csv delegation (simdcsv), gonum differential (simdblas), net/http differential (simdhttp), RFC 8949 vectors (simdcbor), Prometheus PromQL semantics and VictoriaMetrics API parity (simdmetrics), Arrow/parquet-mr cross-language (simdparquet). C++, Rust, and other Go libraries are **workload and performance peers**, never behavioral oracles. Feature-count parity is never a gate; a gap is a workload, an allocation, a dispatch, or a measurement.
6. **Timeouts on every test/build.** Every `go test`/`go build`/`make` invocation is wrapped in shell `timeout` AND carries the native `-timeout` flag where Go accepts it. A hung binary is a leak alarm, not a retry candidate. Gates run bare - never piped through `tail` without `set -o pipefail`.
7. **Writers.** One fresh DeepSeek writer per repository; no writer delegates; repositories may be drafted concurrently because they are independent; the GO_SIMD writer runs in two disjoint scopes (Task 3 local, Task 14 index). Every writer prompt embeds the CORE TENETS block from GO_SIMD `CLAUDE.md` verbatim, the timeout rule verbatim, and this plan's ground rules.
8. **Task-ID uniqueness.** IDs below are unique family-wide. Writers never renumber, rewrite, or restate historical `Task N` IDs or text; follow-on sections and ledgers append.
9. **ASCII only.** This plan and every new writer-authored document use ASCII only: no unicode dashes, arrows, multiplication signs, or section signs (write "-", "->", "x", "Section N").

---

## Task 1: Baseline inventory of all eleven repositories

**Files:** none. The baseline is captured in the session, not in files.

**Step 1: Capture the per-repository baseline in the session**

For each repo `R` in `GO_SIMD simdblas simdcsv simdvec simdjson simdhttp simdcbor simdparquet simdimage simdmetrics simdlogs` under `/home/sebishogun/Work/Development`:

1. Run `git -C /home/sebishogun/Work/Development/$R status --short` through the Bash tool and keep the full printed output in the session notes.
2. Run `git -C /home/sebishogun/Work/Development/$R diff` (full) for the dirty repositories and read the printed output.
3. For the untracked trees, list paths via the glob tool over the repository (e.g. `internal/**`, `media/**` for simdimage) or the printed `--short` output; no shell file writes, no redirection, no `awk`.
4. Record each repo's HEAD via `git -C ... log --oneline -3`.

Expected outcome: the exact inventories below (captured 2026-08-24; re-verify at execution time and correct any drift before writing):

| Repo | HEAD | State |
|---|---|---|
| GO_SIMD | `21b0415` | 8 modified: `AGENTS.md`, `CLAUDE.md`, `Makefile`, `README.md`, `ROADMAP.md`, `docs/plans/2026-08-13-simd-production.md`, `docs/wrong.md`, `internal/conformance/fuzz_test.go`. 2 untracked: `docs/plans/2026-08-24-simd-family-production-readiness-design.md` (the design record), `docs/plans/2026-08-24-simd-family-documentation-implementation.md` (this plan). The Section 8 paragraphs, follow-on ledger, and roadmap summary already exist in the dirty baseline; verify them and do not append duplicates. Neither untracked record is touched by a writer. |
| simdblas | `924f399` | clean |
| simdcsv | `0f56a11` | clean |
| simdvec | `d725897` | clean |
| simdjson | `53ceffb` | 34 modified: `AGENTS.md`, `CLAUDE.md`, `Makefile`, `README.md`, `bench/coldstart_test.go`, `bench/easyjson_rows_test.go`, `bench/marshalindent_row_test.go`, `bench/readme_test.go`, `bench/scale_test.go`, `bench/shapes_test.go`, `bench/small_test.go`, `bench/sml_rows_test.go`, `bench/stream_test.go`, `bench/text_test.go`, `docs/architecture.md`, `docs/cpp-baseline.md`, `docs/lld/marshal-and-codegen.md`, `docs/lld/value-and-ownership.md`, `docs/roadmap.md`, `docs/verification.md`, `docs/wrong.md`, `jsontestsuite_test.go`, `marshal.go`, `marshal_test.go`, `parallel_index_test.go`, `register.go`, `register_test.go`, `steady_test.go`, `syntaxerror_test.go`, `tools/benchchart/main.go`, `tools/benchchart/map.go`, `tools/benchchart/map_test.go`, `tools/benchrunner/main.go`, `tools/benchrunner/parse_test.go`. 10 untracked, exactly: `bench/crosslib_test.go`, `bench/goccyvalid_test.go`, `bench/selectivity_test.go`, `differential_test.go`, `docs/bench/compare-2026-08-22.json`, `docs/bench/compare-2026-08-23.json`, `docs_counts_test.go`, `skip_validation_test.go`, `tools/benchchart/harness_test.go`, `tools/benchchart/render_test.go`. Note: `skip_validation_test.go` and `docs_counts_test.go` are **untracked**, not modified. `git diff --numstat` reports `register.go` +213/-10 and `marshal.go` +319/-66. |
| simdhttp | `9f23757` | clean |
| simdcbor | `326beae` | clean |
| simdparquet | `1cc1fef` | 35 modified: `Makefile`, `README.md`, `bloom.go`, `bloom_test.go`, `corpus_bitpacked_test.go`, `corpus_compress_test.go`, `corpus_pages_test.go`, `docs/architecture.md`, `docs/lld/file-writer.md`, `docs/verification.md`, `docs/wrong.md`, `encoding/dict.go`, `encoding/dict_test.go`, `encoding/page_test.go`, `errors.go`, `errors_test.go`, `filter.go`, `filter_test.go`, `format/footer_test.go`, `format/indexes_test.go`, `indexes.go`, `indexes_test.go`, `interop_test.go`, `reader.go`, `reader_bench_test.go`, `reader_fuzz_test.go`, `reader_limits_test.go`, `reader_test.go`, `schema_test.go`, `seam_errors_test.go`, `tools/fetch-parquet-testing.sh`, `tools/parquet-testing.sha256`, `tools/record-footers.py`, `writer.go`, `writer_test.go`. 25 untracked, exactly: `bloom_refusal_test.go`, `comment_counts_test.go`, `corpus_baddata_test.go`, `corpus_counts_test.go`, `docs_fuzz_test.go`, `encoding/deltabytes_happy_test.go`, `encoding/dictencode_test.go`, `enum_names_test.go`, `exported_readback_test.go`, `filter_noindex_test.go`, `indexes_bound_test.go`, `reader_pagecontent_test.go`, `reader_recordstart_test.go`, `writer_dict.go`, `writer_dict_test.go`, `writer_dict_wire_test.go`, `writer_identity_test.go`, `writer_indexes.go`, `writer_indexes_test.go`, `writer_offsetindex_test.go`, `writer_rowgroup_meta_test.go`, `writer_signedzero_test.go`, `writer_stats_types_test.go`, `writer_transaction_test.go`, `writer_wireclaims_test.go`. |
| simdimage | `8b69b5f` | 4 modified: `Makefile`, `docs/wrong.md`, `go.mod`, `go.sum`. Untracked: `internal/` (ffmpegabi tables + loader, abiprobe) and `media/` (rational, packet, frame, stream types + ffmpeg runtime). |
| simdmetrics | `6d6a41b` | 25 modified: `README.md`, `docs/wrong.md`, `internal/api/cluster.go`, `internal/api/cluster_metadata.go`, `internal/api/cluster_metadata_test.go`, `internal/api/joinbody_guards_test.go`, `internal/api/labelbytes_test.go`, `internal/api/promapi.go`, `internal/api/scanlimit_test.go`, `internal/api/server.go`, `internal/api/sketches.go`, `internal/api/vendors.go`, `internal/chunk/int64_format_test.go`, `internal/otlp/otlp.go`, `internal/otlp/otlp_proto.go`, `internal/otlp/otlp_proto_test.go`, `internal/prompb/prompb.go`, `internal/prompb/prompb_test.go`, `internal/promql/engine.go`, `internal/snappy/bounds_test.go`, `internal/snappy/snappy.go`, `internal/tsdb/rawlength_test.go`, `internal/vmnative/block.go`, `internal/vmnative/stream.go`, `internal/vmnative/zstd.go`. 25 untracked: `internal/api/body_status_test.go`, `internal/api/cluster_claim_compose_test.go`, `internal/api/cluster_lookback_test.go`, `internal/api/cluster_merge_alloc_test.go`, `internal/api/cluster_merge_test.go`, `internal/api/cluster_ownership_test.go`, `internal/api/cluster_stability_test.go`, `internal/api/compressed_body_test.go`, `internal/api/decompressionbounds_test.go`, `internal/api/otlp_attr_bound_test.go`, `internal/api/sketch_packed_test.go`, `internal/api/sketches_alloc_test.go`, `internal/api/truncated_proto_test.go`, `internal/otlp/otlp_depth_test.go`, `internal/otlp/otlp_json_test.go`, `internal/otlp/otlp_walk_census_test.go`, `internal/otlp/race_slack_race_test.go`, `internal/otlp/race_slack_test.go`, `internal/tsdb/append_dedupe_test.go`, `internal/vmnative/bounds_test.go`, `internal/vmnative/testdata/hufflit4_a.raw`, `internal/vmnative/testdata/hufflit4_b.raw`, `internal/vmnative/testdata/hufflit4_sf1.zst`, `internal/vmnative/testdata/hufflit4_sf2.zst`, `internal/vmnative/testdata/hufflit4_sf3.zst`. Note: `bench.test` (11 MB, repo root) is **tracked**; its removal is staged as `METR-CORR-01` (Task 12), never performed in wave 0. |
| simdlogs | `47bc3a9` | clean |

**Execution reclassification (2026-08-26):** the table remains the immutable
pre-wave capture. Source-backed review corrections expanded three originally
clean repositories beyond their initial four-file writer scopes: simdvec now
has eight wave files (the four Task 6 files plus README, architecture, the
index/search LLD, and the production-design snapshot annotation); simdhttp has
seven (the four Task 8 files plus README, architecture, and verification); and
simdcbor has eleven (the four Task 9 files plus README, architecture, decoder
and encoder LLDs, the production-design snapshot annotation, verification, and
historical-record annotations in `docs/wrong.md`). These are reviewed wave
outputs, not unexplained user changes; Tasks 6, 8, and 9 below carry the amended
scopes. No implementation file was added to those scope expansions.

**Step 2: Baseline acceptance**

The printed status/diff is the baseline. At every review checkpoint, re-run status/diff and compare by hand: every pre-existing hunk must be present unchanged (a scoped dirty-roadmap edit preserves old hunks and appends new ones), and no path outside the task's scope may appear. No baseline snapshots are written to /tmp or anywhere else.

---

## Task 2: Writer dispatch protocol

**Files:** none. This task fixes the prompt and content skeleton every writer task (3-13) uses, then dispatches Tasks 3-13.

**Step 1: Fix the common content skeleton** (every writer task must produce these five artifacts in its scoped documents):

1. **Roadmap summary** - appended to the repo's canonical roadmap (`ROADMAP.md` in GO_SIMD and simdmetrics; `docs/roadmap.md` in the other nine): a short "Production readiness" section with (a) current maturity level and 2026-08-24 snapshot date, source-backed; (b) the open high-level work as bullets; (c) a link to the owning ledger. It never copies per-task status, never states dates or promises.
2. **Ledger table** - columns exactly: `ID | state | work | evidence | exit`. Rows are the repo's stream-prefixed IDs from its task below. `state` uses only the seven-state vocabulary: `open`, `staged`, `in-progress`, `blocked`, `evidence-complete`, `shipped`, `rejected`. Wave-0 rows start `open` (recorded, not staged) unless the task says otherwise. `evidence` names the gate or artifact that closes the task. `exit` is the release exit line.
3. **v1 exit** - the repo's release exit, one line, sourced to the repo's own docs (each repo's exit is fixed in its task below; simdlogs uses the full production exit, phases 0-10 plus release gates).
4. **Competitor workload matrix** - table: `workload | this repo | peer | oracle-or-basis | gate`. Rows are concrete corpora/shapes/request-mixes only; no feature counts, no parity claims, no invented numbers. The oracle column is populated only where ground rule 5's compatibility promise exists; C++/Rust/other-Go entries are peers. Figures are quoted only after the writer reads them in the source with provenance.
5. **Task-management paragraph** - the Section 8 content (Step 2), with the repo's actual paths.

**Step 2: Fix the Section 8 paragraph** (appended to both `AGENTS.md` and `CLAUDE.md` in every repo; created with them in simdmetrics):

- **Local authority.** `<roadmap path>` is canonical; the ledger at `<ledger path>` is the only staging area; `docs/wrong.md` is the only record of rejections. The family index at `github.com/sebishogun/simd`'s `docs/plans/2026-08-24-simd-family-production-readiness.md` is a link collection and never overrides local truth; it never duplicates per-task status.
- **One task ID at a time.** Work executes one task at a time by its ID from the local ledger (for example `<PREFIX>-01`). A session touching implementation work names its task ID in its first message; without one it touches no implementation files.
- **State transitions.** Seven states (`open/staged/in-progress/blocked/evidence-complete/shipped/rejected`); a transition is an edit in the local ledger (plus changelog or `docs/wrong.md` for `shipped`/`rejected`); `rejected` is terminal without a documented reopen condition; historical task text and IDs are never edited for status.
- **Gate rule.** Before any commit: the gate set from `<verification.md path>`, run bare with explicit timeouts; a hung test binary is a leak alarm, not a retry candidate.

**Step 3: Dispatch**

Dispatch Tasks 3-13, one fresh DeepSeek writer per repository, all in parallel (repositories are independent). Each writer prompt contains: the CORE TENETS block verbatim (GO_SIMD `CLAUDE.md`), the timeout rule verbatim, ground rules 1-9 of this plan, the common content skeleton (Step 1), the Section 8 paragraph (Step 2), the per-repo scope from its task, and the instruction to read the repo's exact current diff before editing. No writer may delegate or spawn sub-writers.

---

## Task 3: GO_SIMD local governance (writer GO_SIMD-A - part 1 of 2)

**Maturity:** `R4` released (v1.21.1); current dirty workspace `R3` (design Section 5.1 snapshot). **Boundary:** a Go package of whole-slice SIMD primitives for numeric, text, and columnar work, runtime-selected generated kernels plus a portable Go path. **Oracles/peers:** the portable reference `internal/ref` and the handwritten scalar loop (`docs/kernels.md`) are the workload baselines; no external library is an oracle; relevant C++ and Rust SIMD libraries (for example Google Highway and Rust `wide`/portable-simd, named only if the writer finds them in the repo's docs) are workload comparators, never oracles.

**Task groups** (all rows in the ledger table, all `open`):

- `SIMD-CORR-01` fuzz corrective - review the fuzz targets in `internal/conformance/` against the differential contract (NaN payloads, IEEE specials, adversarial lengths); fix or record. Evidence: bare fuzz run with timeout, green, or a `docs/wrong.md` entry. Exit: corrective slice green.
- `SIMD-CORR-02` timeout corrective - every test and fuzz invocation in `docs/verification.md` carries an explicit timeout. Evidence: verification doc lists each command with its timeout and one bare green run. Exit: corrective slice green.
- `SIMD-CORR-03` docs corrective - reconcile `docs/platforms.md` counts, README claims, and ROADMAP status against sources via the docs tests, including the named contradictions in the authoritative ledger row: ROADMAP's "door closed twice" sentence; entry 80's unexplained 1,080-case denominator, dependent arithmetic, and four-line objdump attribution; and entry 81's historical 11-seed label plus stale `f.Fatalf` wording against the current three-seed corpus and final `f.Skip` guard. No claim edits without the machine-checked counts. Evidence: `go test ./internal/tests/docs` green. Exit: corrective slice green.
- `SIMD-R5-01` real-silicon threshold evidence - thresholds away from amd64 measured on real hardware (emulated lanes do not count as real-silicon evidence). Evidence: recorded measurements with provenance; exit: `R5` maintained-production evidence.
- `SIMD-R5-02` patch/upgrade-cycle evidence - one patch cycle with the regression suite green and release automation observed. Evidence: patch release plus green suite; exit: `R5`.
- `SIMD-R5-03` remaining roadmap kernel closure - the sort three-way partition and general n-ary closure combinator, each with its evidence bar from the production plan. Evidence: per-task gates or `docs/wrong.md` rejections; exit: roadmap items closed or rejected. The measured `GOEXPERIMENT=simd` small-n tier is not a task row: entries 58, 75, and 79 close it on measured speed, while the ROADMAP records the condition for re-running it. Entry 80 records a source-level bit-identity limit and is not an additional speed rejection.
- `SIMD-R5-04` cross-ecosystem workload decision - compare supported whole-slice workloads with the named C++ and Rust peers; classify material gaps as in-boundary work, future work, or rejected with evidence. Evidence: workload matrix and decisions. Exit: gaps classified without feature-count parity.

**Files:**
- Modify: `/home/sebishogun/Work/Development/GO_SIMD/ROADMAP.md` (append the roadmap summary only; the dirty edits stay untouched, old hunks preserved, new section appended)
- Modify: `/home/sebishogun/Work/Development/GO_SIMD/AGENTS.md`, `/home/sebishogun/Work/Development/GO_SIMD/CLAUDE.md` (Section 8 paragraph)
- Modify: `/home/sebishogun/Work/Development/GO_SIMD/docs/plans/2026-08-13-simd-production.md` (append a status/follow-on section *below* the "Final task: full verification and review" - the SIMD-CORR-01..03 + SIMD-R5-01..04 ledger table; no renumbering, no edits to existing task text)
- Do NOT touch: `Makefile`, `README.md`, `docs/wrong.md`, `internal/conformance/fuzz_test.go` (dirty, preserved), the design record and this plan (untracked, preserved), the index (Task 14).

**Step 1: Read the current state.** Run `git status --short` and `git diff` (full) in the repo; read `ROADMAP.md`, `docs/verification.md`, `docs/plans/2026-08-13-simd-production.md`, `AGENTS.md`, `CLAUDE.md`. Expected: the Task 1 8-modified + 2-untracked dirty inventory; the Section 8 paragraphs, follow-on ledger, and roadmap summary are already present as baseline content; the plan's task format runs from "Task 1: Three-way sort partition kernel" through "Final task".

**Step 2: Verify and correct the roadmap summary, ledger table, and v1 exit.** The existing roadmap summary names the three corrective streams (fuzz, timeouts, docs counts) and the R5 evidence streams (real silicon, patch cycle, remaining roadmap kernels, ecosystem workloads) as open high-level work with the ledger link; the existing ledger table carries the seven task rows, all `open`, each with work/evidence/exit as above. Correct source-backed defects in place; do not append a second summary or ledger. The rejected small-n experiment stays outside the open ledger and retains its existing rejection reference. No benchmark or shipped claims are added.

**Step 3: Verify the existing Section 8 paragraph** in `AGENTS.md` and `CLAUDE.md`, with GO_SIMD's paths (`ROADMAP.md`, `docs/plans/2026-08-13-simd-production.md`, `docs/verification.md`, `docs/wrong.md`) and the index path as the local file `docs/plans/2026-08-24-simd-family-production-readiness.md`. Correct it in place if needed; do not append a duplicate.

**Step 4: Check.** Run `git -C /home/sebishogun/Work/Development/GO_SIMD diff --check`. Expected: clean.

**Step 5: PRIMARY-REVIEW CHECKPOINT (replaces commit).** The primary agent: (1) re-run `git status --short` and `git diff`, compare against the reverified Task 1 baseline - wave-0 governance edits remain within the **four** scoped files; `Makefile`, `README.md`, `docs/wrong.md`, `internal/conformance/fuzz_test.go` remain byte-identical to that baseline; no duplicate Section 8, ledger, or roadmap-summary section exists; (2) links in the new text resolve; (3) current/planned language is separated; (4) historical task IDs/text are untouched; (5) no commit. A defect returns to writer GO_SIMD-A; then re-run Steps 4-5.

---

## Task 4: simdblas (writer simdblas)

**Maturity:** `R3` - v1.1.0 identity and documentation/release evidence reconciliation. **Boundary:** BLAS linear-algebra routines (gemm/gemv/syrk/complex level-1) as guard+delegate over simd kernels. **Oracles/peers:** gonum is the differential oracle at both correctness bars (per `AGENTS.md`); OpenBLAS, BLIS, MKL, and Rust/C++ BLAS bindings are workload and performance peers, never oracles.

**Task groups** (ledger rows, all `open`):

- `BLAS-V1-01` release identity - the local and remote v1.1.0 tag, latest GitHub release v1.0.1, origin/main lag, changelog, README, and agent files describe the same tree and publication state. Evidence: local/remote tag state plus GitHub release state, changelog, and docs agree; docs gates green. Exit: release evidence complete.
- `BLAS-V1-02` complex thresholds and runtime engagement - complex64/complex128 fast-path thresholds measured and shown engaged at runtime. Evidence: quiet-host measurements with provenance plus engagement tests in the `TestFastPathActuallyRuns` style. Exit: complex rows measured.
- `BLAS-V1-03` NaN/signed-zero and panic parity - differential parity with gonum for NaN payloads, signed zero, and exact panic behavior on invalid arguments. Evidence: differential tests at both bars. Exit: parity tests green.
- `BLAS-V1-04` quiet threshold measurements - level-1/2/3 thresholds re-measured on a quiet host under the repo's benchmark contract. Evidence: provenance records. Exit: thresholds recorded.
- `BLAS-V1-05` R5 patch cycle - full gate set green across a patch cycle. Evidence: patch release plus green gates. Exit: `R5` evidence.
- `BLAS-V1-06` cross-ecosystem workload decision - compare supported BLAS workloads with OpenBLAS, BLIS, MKL, and Rust/C++ peers; classify material gaps as v1 blockers, post-v1 work, or rejected with evidence. Evidence: workload matrix and decisions. Exit: gaps classified.

**Competitor workload matrix rows:** gemm/gemv/syrk shapes from the plan; complex level-1 routine shapes; invalid-argument shapes for panic parity. Figures only from the plan/`docs/wrong.md`, quoted by the writer after reading them.

**Files:**
- Modify: `/home/sebishogun/Work/Development/simdblas/docs/roadmap.md` (roadmap summary)
- Modify: `/home/sebishogun/Work/Development/simdblas/docs/plans/2026-08-13-simdblas-production.md` (append follow-on section with the ledger table below the existing lose-or-record table; no renumbering)
- Modify: `/home/sebishogun/Work/Development/simdblas/AGENTS.md`, `/home/sebishogun/Work/Development/simdblas/CLAUDE.md` (Section 8)
- Modify: `/home/sebishogun/Work/Development/simdblas/README.md` (review correction: distinguish the local and remote v1.1.0 tag from the latest GitHub release v1.0.1)

**Steps:** (1) read `docs/roadmap.md`, the plan, `docs/verification.md`, `AGENTS.md`; confirm clean tree. (2) Write roadmap summary + ledger (`BLAS-V1-01..06`, all `open`) + v1 exit (R5 after reconciled v1.1.0 identity, green gates, and patch-cycle evidence) + workload matrix (rows above; gonum oracle column, others peers). (3) Append Section 8 with paths `docs/roadmap.md`, `docs/plans/2026-08-13-simdblas-production.md`, `docs/verification.md`, `docs/wrong.md`; apply the source-backed README tag/publication correction found by primary review. (4) `git diff --check` - expected clean. (5) **PRIMARY-REVIEW CHECKPOINT**: diff vs baseline (only the five scoped files), links resolve, IDs unique, historical instructions explicitly marked superseded rather than deleted, no commit.

---

## Task 5: simdcsv (writer simdcsv)

**Maturity:** `R3` - v0.2.1 exists; v1 contracts and complete gates remain. **Boundary:** a SIMD CSV parser with encoding/csv delegation on the quoted/short spans where compatibility is promised. **Oracles/peers:** `encoding/csv` is the oracle only for the delegated compatibility surface (the delegation records in `docs/wrong.md` define it); the Rust `csv` crate, Arrow CSV readers, and C++ fast CSV parsers are workload and performance peers, never oracles.

**Task groups** (ledger rows, all `open`):

- `CSV-V1-01` dependency identity - v0.2.1, the Go version, and the simd version facts pinned in the README status section and docs. Evidence: docs state the pinned facts; link checks green. Exit: identity reconciled.
- `CSV-V1-02` FieldsPerRecord and malformed-input contracts - `FieldsPerRecord` behavior and the malformed-input contract hardened and documented. Evidence: contract tests plus verification entries. Exit: contracts green.
- `CSV-V1-03` bounded-reader decision - the bounded/streaming reader option evaluated with measurement; adopted, or rejected with a reopen condition in `docs/wrong.md`; the already-exported `ErrRecordTooLarge` name and stale prototype comment are swept with the decision. Evidence: decision record. Exit: decision recorded.
- `CSV-V1-04` double-copy measurement - the double-copy cost on the hot path measured; removed, or recorded in `docs/wrong.md`. Evidence: measurement record. Exit: measurement recorded.
- `CSV-V1-05` v1 freeze - the full CI/fuzz/cross-arch/bench gate set green. Evidence: gates green. Exit: v1 release.
- `CSV-V1-06` cross-ecosystem workload decision - compare concrete CSV workloads with Rust csv, Arrow CSV, and C++ peers; classify material gaps as v1 blockers, post-v1 work, or rejected with evidence. Evidence: workload matrix and decisions. Exit: gaps classified.

**Competitor workload matrix rows:** all-quoted spans, quoted rows, short spans, large-row-count files - figures quoted from `docs/wrong.md` by the writer after reading them, never invented.

**Files:**
- Modify: `/home/sebishogun/Work/Development/simdcsv/docs/roadmap.md`
- Modify: `/home/sebishogun/Work/Development/simdcsv/docs/plans/2026-08-13-simdcsv-production.md` (append follow-on ledger)
- Modify: `/home/sebishogun/Work/Development/simdcsv/AGENTS.md`, `/home/sebishogun/Work/Development/simdcsv/CLAUDE.md` (Section 8)
- Modify: `/home/sebishogun/Work/Development/simdcsv/docs/verification.md` (review correction: describe the actual per-target seed corpus)

**Steps:** (1) read roadmap, plan, verification, `docs/wrong.md` delegation records, AGENTS; clean tree. (2) Summary + ledger (`CSV-V1-01..06`, `open`) + v1 exit + workload matrix. (3) Section 8 and the source-backed seed-corpus correction found by primary review. (4) `git diff --check` clean. (5) **PRIMARY-REVIEW CHECKPOINT** (five-file scope, links, IDs, historical preservation, no commit).

---

## Task 6: simdvec (writer simdvec)

**Maturity:** `R3` - v0.2.0 exists; API consolidation and v1 evidence remain. **Boundary:** exact flat vector search over float32 embeddings - one `simd.GemvParallelInto` matrix-vector product per query. **Peers:** Faiss Flat (`IndexFlatIP`), USearch exact search, hnswlib brute-force flat, and Rust/Gonum vector peers are workload comparators; there is no oracle (no compatibility promise). ANN remains excluded unless recall/latency evidence reopens it (the int8 rejection in `docs/wrong.md` is the standing record).

**Task groups** (ledger rows, all `open`):

- `VEC-V1-01` v0.2 release identity - docs, changelog, and tie semantics (which row wins on duplicate scores) stated and tested. Evidence: docs plus tie tests. Exit: identity reconciled.
- `VEC-V1-02` consolidate duplicate paths - the duplicate top-K and Euclidean code paths merged into one. Evidence: one path; full suite green. Exit: consolidation green.
- `VEC-V1-03` contracts - scale/memory, scratch/capacity, persistence-corruption, and concurrency contracts documented with tests. Evidence: contract tests plus LLD entries. Exit: contracts green.
- `VEC-V1-04` v1 freeze - full gate set green. Evidence: gates. Exit: v1 release.
- `VEC-V1-05` cross-ecosystem workload decision - compare exact-search workloads with Faiss Flat, USearch exact, hnswlib brute force, and Rust/Gonum peers; classify material gaps as v1 blockers, post-v1 work, or rejected with evidence. Evidence: workload matrix and decisions. Exit: gaps classified. ANN stays excluded unless recall/latency evidence satisfies its reopen condition.
- `VEC-V1-06` fresh performance evidence - run exact-search scale, allocation, and memory benchmarks for single-query, batch-query, and index-build workloads in a coordinated quiet-host window; verify the SIMD dispatch used by each claimed fast path. Evidence: provenance, `-benchmem`, dispatch assertions, and interleaved peer comparisons. Exit: current performance claims are reproducible.

**Competitor workload matrix rows:** single query, batch queries (the roadmap's batch-search row), index build, memory at scale - figures from `docs/roadmap.md`/`docs/wrong.md` only, quoted by the writer after reading them.

**Files:**
- Modify: `/home/sebishogun/Work/Development/simdvec/docs/roadmap.md`
- Modify: `/home/sebishogun/Work/Development/simdvec/docs/plans/2026-08-13-simdvec-production.md` (append follow-on ledger below the settled-decisions table)
- Modify: `/home/sebishogun/Work/Development/simdvec/AGENTS.md`, `/home/sebishogun/Work/Development/simdvec/CLAUDE.md` (Section 8)
- Modify: `/home/sebishogun/Work/Development/simdvec/README.md`, `/home/sebishogun/Work/Development/simdvec/docs/architecture.md`, `/home/sebishogun/Work/Development/simdvec/docs/lld/index-and-search.md`, `/home/sebishogun/Work/Development/simdvec/docs/plans/2026-08-13-simdvec-production-design.md` (source-backed review corrections and historical-snapshot annotation)

**Steps:** (1) read roadmap, plan, `docs/wrong.md` (int8 rejection), AGENTS; clean tree. (2) Summary + ledger (`VEC-V1-01..06`, `open`) + v1 exit + workload matrix. (3) Section 8 plus the source-backed public-surface, architecture, LLD, and historical-snapshot corrections found during review. (4) `git diff --check` clean. (5) **PRIMARY-REVIEW CHECKPOINT** over the eight scoped files; historical text is annotated rather than deleted.

---

## Task 7: simdjson (writer simdjson)

**Maturity:** `R2` - v0.7.0 released; the v0.8 candidate is the large dirty tree. **Boundary:** an encoding/json-compatible SIMD JSON library; the drop-in bar (byte- and error-identical, vendored stdlib suites, differential fuzzers) is the promised compatibility surface. **Oracles/peers:** `encoding/json` is the oracle **only within that drop-in promise**; goccy/go-json, sonic, fastjson, segmentio, minio/simdjson-go, C++ simdjson, and Rust serde_json are workload and performance peers, never behavioral oracles.

**Task groups:** `JSON-V1-01..08` exactly as design Section 9.2, restated here with evidence and exits:

- `JSON-V1-01` preserve and organize the dirty v0.8 candidate - the Task 1 inventory (34 modified + 10 untracked, exact lists above) is recorded into the plan; every user change preserved. Evidence: the inventory section in the readiness plan. Exit: candidate organized.
- `JSON-V1-02` commit-independent stabilization plan - each stabilization task is executable from the dirty-tree state as-is, whether or not the candidate is committed first; no task's precondition is "a clean checkout". Evidence: the plan's per-task preconditions. Exit: plan executable.
- `JSON-V1-03` current snapshot/figure regeneration - a Section 6 quiet-host window in the implementation wave, then regenerate the benchmark snapshots (`docs/bench/compare-*.json`) and figures from the recorded dirty candidate with provenance (machine, commit plus dirty-state identity, harness, date). Evidence: regenerated snapshots and provenance record. Exit: snapshots regenerated.
- `JSON-V1-04` explicit deltas - exactly what the candidate changes in `RegisterEncoder` behavior (the differential harness covers eight positions: top level, struct field, slice element, array element, map value, pointer target, inside an interface, map key; `RegisterEncoder` does not cover the map-key position, which encoding/json's `TextMarshaler` does), error taxonomy, and stdlib compatibility against the previous release. Evidence: the delta record in the plan. Exit: deltas recorded.
- `JSON-V1-05` full gates - the repository's full functional gate set bare with explicit timeouts, then `benchcheck -maxload 1` against the stored baseline in a Section 6 quiet window (the tool's default refuses above load 4, per `docs/verification.md`; `bench-update` is never used to clear a failure and is allowed only after a recorded expected-change reason, equal sample counts, and a second quiet `benchcheck`). Evidence: commands and green results. Exit: gates green.
- `JSON-V1-06` v0.8 - the tag and changelog entry. Evidence: tag and changelog. Exit: v0.8 released.
- `JSON-V1-07` compatibility freeze and patch-cycle evidence - after v0.8 the compatibility surface is frozen and holds through at least one patch cycle (regression suite green across a patch release) before the v1.0 path proceeds. Evidence: the patch-cycle run. Exit: freeze proven.
- `JSON-V1-08` workload-backed ecosystem-gap decision - compare the supported boundary against the named Go, C++, and Rust peers using concrete corpora and operations; classify each material gap as a v1 blocker, post-v1 work, or rejected with evidence; the roadmap records the decisions, not feature counts; implementation requires its own approved task. Evidence: the decision record. Exit: decisions recorded.

**Dirty-state detail the writer must record in the readiness plan (JSON-V1-01 inventory, verified by Task 1):** the candidate code (`register.go` +213/-10, `marshal.go` +319/-66, untracked `differential_test.go` and `skip_validation_test.go`, untracked `docs_counts_test.go`), the bench/tools tree (`bench/` 10 modified plus untracked `crosslib_test.go`, `goccyvalid_test.go`, `selectivity_test.go`; `tools/benchchart/` 3 modified plus untracked `harness_test.go`, `render_test.go`; `tools/benchrunner/` 2 modified), the untracked snapshot JSONs (`docs/bench/compare-2026-08-22.json`, `compare-2026-08-23.json`), and the dirty docs (`README.md`, `docs/architecture.md`, `docs/cpp-baseline.md`, `docs/lld/marshal-and-codegen.md`, `docs/lld/value-and-ownership.md`, `docs/roadmap.md`, `docs/verification.md`, `docs/wrong.md`, `AGENTS.md`, `CLAUDE.md`, `Makefile`, 6 root tests). Known stale/wrong figures: the dirty README diff adjusts several competitive rows and the dirty `docs/cpp-baseline.md` carries new uncommitted C++ measurements whose load-average range is above the quiet threshold - a provenance gap. The readiness plan lists these as evidence requirements for JSON-V1-03/05 without restating the numbers; the writer quotes any figure only after reading the dirty diff with its provenance.

**Files:**
- Create: `/home/sebishogun/Work/Development/simdjson/docs/plans/2026-08-24-simdjson-v1-readiness.md` (the v1-readiness plan; contains the JSON-V1-01..08 ledger, the JSON-V1-01 inventory, the Section 6 quiet-host reference, the gates, the v1 exit, and the workload matrix)
- Modify: `/home/sebishogun/Work/Development/simdjson/docs/roadmap.md` (append roadmap summary + ledger link; dirty edits preserved as old hunks, new section appended)
- Modify: `/home/sebishogun/Work/Development/simdjson/AGENTS.md`, `/home/sebishogun/Work/Development/simdjson/CLAUDE.md` (Section 8)
- Do NOT touch: `docs/plans/2026-08-13-simdjson-production.md` (historical plan stays untouched), every dirty code/test/bench/tool/snapshot file listed above.

**Steps:** (1) Read the exact diff: `git status --short`, full `git diff`, and the untracked inventory; read the historical plan's structure, `docs/verification.md` gate table, `docs/roadmap.md`, `AGENTS.md`. (2) Create the readiness plan per the task groups above; the v1 exit is v1.0 with the compatibility freeze held through a patch cycle. (3) Append the roadmap summary and Section 8 to `AGENTS.md`/`CLAUDE.md` (paths: `docs/roadmap.md`, `docs/plans/2026-08-24-simdjson-v1-readiness.md`, `docs/verification.md`, `docs/wrong.md`). (4) `git diff --check` clean. (5) **PRIMARY-REVIEW CHECKPOINT**: only the four scoped files have wave-0 deltas; every dirty file outside `docs/roadmap.md`, `AGENTS.md`, and `CLAUDE.md` is byte-identical to the Task 1 baseline; those three scoped dirty files retain every old hunk and only append the new sections; the JSON-V1-01 inventory matches the baseline inventory exactly; links resolve; IDs unique; no commit.

---

## Task 8: simdhttp (writer simdhttp)

**Maturity:** `R2` - intended parser/router surface exists; root fuzz and first-release evidence remain. **Boundary:** a net/http-compatible HTTP parser/router over SIMD; the server loop stays outside the boundary unless design evidence moves it in. **Oracles/peers:** `net/http` is the differential oracle where compatibility is promised (`FuzzParseAgainstNetHTTP`); fasthttp, llhttp, picohttpparser, hyper, and httparse are workload and performance peers, never oracles.

**Task groups** (ledger rows, all `open`):

- `HTTP-V1-01` root differential-fuzz contract - `FuzzParseAgainstNetHTTP` seed pinned with the duplicate-Host gap fix (the `docs/wrong.md` record). The lane is currently red by design locally; a red run is read, not piped. Evidence: fuzz run green or the recorded red-by-design state plus a verification entry. Exit: fuzz contract green.
- `HTTP-V1-02` reader progress - semicolon handling and repeated `0, nil` reads must make progress. Evidence: progress tests green. Exit: progress proven.
- `HTTP-V1-03` adversarial router shapes - the router corpus extended with pathological shapes. Evidence: corpus tests green. Exit: corpus green.
- `HTTP-V1-04` strictness inventory and docs - strictness decisions inventoried and documented. Evidence: strictness doc plus verification entries. Exit: inventory complete.
- `HTTP-V1-05` first release - the release gate set green. Evidence: gates. Exit: first release.
- `HTTP-V1-06` cross-ecosystem workload decision - compare parser and router workloads with fasthttp, llhttp, picohttpparser, hyper, and httparse; classify material gaps as v1 blockers, post-v1 work, or rejected with evidence. Evidence: workload matrix and decisions. Exit: gaps classified. A server loop remains outside the boundary without a separately approved design.
- `HTTP-V1-07` fresh performance evidence - run parser/router allocation, throughput, and adversarial-shape benchmarks in a coordinated quiet-host window; verify runtime SIMD engagement for every SIMD-backed claim. Evidence: provenance, `-benchmem`, dispatch assertions, and interleaved peer comparisons. Exit: current performance claims are reproducible.

**Competitor workload matrix rows:** router-match corpus, parse-vs-net/http differential shapes, adversarial shapes - from the test/fuzz surface; no invented numbers.

**Files:**
- Modify: `/home/sebishogun/Work/Development/simdhttp/docs/roadmap.md`
- Modify: `/home/sebishogun/Work/Development/simdhttp/docs/plans/2026-08-13-simdhttp-production.md` (append follow-on ledger)
- Modify: `/home/sebishogun/Work/Development/simdhttp/AGENTS.md`, `/home/sebishogun/Work/Development/simdhttp/CLAUDE.md` (Section 8)
- Modify: `/home/sebishogun/Work/Development/simdhttp/README.md`, `/home/sebishogun/Work/Development/simdhttp/docs/architecture.md`, `/home/sebishogun/Work/Development/simdhttp/docs/verification.md` (source-backed parser/router and fuzz-seed review corrections)

**Steps:** (1) read roadmap, plan, verification, `docs/wrong.md` (duplicate-Host record), AGENTS; clean tree. (2) Summary + ledger (`HTTP-V1-01..07`, `open`) + v1 exit + workload matrix. (3) Section 8 plus the source-backed parser/router, architecture, and fuzz-seed corrections found during review. (4) `git diff --check` clean. (5) **PRIMARY-REVIEW CHECKPOINT** over the seven scoped files.

---

## Task 9: simdcbor (writer simdcbor)

**Maturity:** `R2` - codec surface exists; representation, limits, interoperability, and first release remain. **Boundary:** an RFC 8949 CBOR codec with a JSON-shaped shipped API adapter (`Unmarshal`, `Marshal`, `Skip`). **Oracles/peers:** RFC 8949 test vectors are the conformance oracle; fxamacker/cbor is an interop peer; QCBOR, TinyCBOR, serde_cbor, and ciborium are workload and performance peers, never oracles.

**Task groups** (ledger rows, all `open`):

- `CBOR-V1-01` byte-string representation - the bytes-versus-string representation decision recorded with tests. Evidence: decision record plus tests. Exit: representation fixed.
- `CBOR-V1-02` public limits and typed errors - limits (depth, size, recursion) and the error taxonomy as a public contract. Evidence: contract tests plus verification entries. Exit: contract green.
- `CBOR-V1-03` Marshal adapter and dead path - the adapter's dead path removed or justified. Evidence: tests green. Exit: path resolved.
- `CBOR-V1-04` RawNext UTF-8 - `RawNext` validates UTF-8. Evidence: vector tests. Exit: validated.
- `CBOR-V1-05` linear indefinite text - indefinite-length text handled in linear time. Evidence: vectors. Exit: linear.
- `CBOR-V1-06` reader progress - repeated `0, nil` reads make progress. Evidence: progress tests. Exit: progress proven.
- `CBOR-V1-07` diagnostics and docs. Evidence: docs plus link checks. Exit: docs green.
- `CBOR-V1-08` first release - vectors/interoperability/race/fuzz/cross-arch/bench gates green (the plan's Task 11 gate list plus `docs/verification.md` rules). Evidence: gates. Exit: first release.
- `CBOR-V1-09` cross-ecosystem workload decision - compare supported encode/decode workloads with fxamacker, QCBOR, TinyCBOR, serde_cbor, and ciborium; classify material gaps as v1 blockers, post-v1 work, or rejected with evidence. Evidence: workload matrix and decisions. Exit: gaps classified.

**Competitor workload matrix rows:** RFC 8949 vectors, fxamacker interop round-trips, indefinite-length shapes - no invented numbers.

**Files:**
- Modify: `/home/sebishogun/Work/Development/simdcbor/docs/roadmap.md`
- Modify: `/home/sebishogun/Work/Development/simdcbor/docs/plans/2026-08-13-simdcbor-production.md` (append follow-on ledger)
- Modify: `/home/sebishogun/Work/Development/simdcbor/AGENTS.md`, `/home/sebishogun/Work/Development/simdcbor/CLAUDE.md` (Section 8)
- Modify: `/home/sebishogun/Work/Development/simdcbor/README.md`, `/home/sebishogun/Work/Development/simdcbor/docs/architecture.md`, `/home/sebishogun/Work/Development/simdcbor/docs/lld/decoder.md`, `/home/sebishogun/Work/Development/simdcbor/docs/lld/encoder.md`, `/home/sebishogun/Work/Development/simdcbor/docs/plans/2026-08-13-simdcbor-production-design.md`, `/home/sebishogun/Work/Development/simdcbor/docs/verification.md`, `/home/sebishogun/Work/Development/simdcbor/docs/wrong.md` (source-backed codec-surface and historical-record annotations found during review)

**Steps:** (1) read roadmap, plan (Task 11 gate list), verification, AGENTS; clean tree. (2) Summary + ledger (`CBOR-V1-01..09`, `open`) + v1 exit (first release with the full-codec gates) + workload matrix. (3) Section 8 plus the source-backed codec-surface and historical-record annotations found during review. (4) `git diff --check` clean. (5) **PRIMARY-REVIEW CHECKPOINT** over the eleven scoped files; historical task instructions remain verbatim with current notes appended.

---

## Task 10: simdparquet (writer simdparquet)

**Maturity:** `R1` - useful structural subset; typed values, dictionary/index, codec, and interop work remain. **Boundary:** a Parquet reader/writer over SIMD (RLE/bitpacking, pages, bloom, indexes) with mandatory Arrow interoperability. **Oracles/peers:** Apache Arrow and parquet-mr are the cross-language oracles where interop is promised (golden files, `tools/fetch-parquet-testing.sh`); parquet-go, Arrow C++/Rust, DuckDB, and Polars are workload and performance peers, never behavioral oracles.

**Task groups** (ledger rows, all `open`):

- `PARQ-01` typed values/records coverage - the typed-value and record surface completed. Evidence: typed round-trip tests. Exit: typed surface green.
- `PARQ-02` dictionary/index completion - dictionary and index work completed, atomic with the dirty tree's in-progress state (it is the current state, not a rebuild). Evidence: dictionary/index tests plus the corpus gates. Exit: dictionary/index green.
- `PARQ-03` Snappy/Zstd codecs and logical metadata - codec coverage and logical metadata handling completed. Evidence: golden files plus corpus gates. Exit: codecs green.
- `PARQ-04` bounds and error taxonomy - reader/writer bounds and the error taxonomy as a public contract. Evidence: bounds tests plus verification entries. Exit: contract green.
- `PARQ-05` mandatory Arrow interoperability plus a second implementation - interop against Arrow and at least one second implementation (parquet-mr or Arrow C++/Rust). Evidence: cross-language golden files. Exit: interop green.
- `PARQ-06` concurrency and file benchmarks on two architectures. Evidence: benchmarks with provenance. Exit: measured.
- `PARQ-07` first release - the release gate set green. Evidence: gates. Exit: first release.
- `PARQ-08` cross-ecosystem workload decision - compare supported file, scan, and interoperability workloads with parquet-go, Arrow C++/Rust, parquet-mr, DuckDB, and Polars; classify material gaps as first-release blockers, future work, or rejected with evidence. Evidence: workload matrix and decisions. Exit: gaps classified.

**Dirty-state detail:** the Task 1 pre-governance source snapshot is 35 modified + 25 untracked (exact lists above, including `writer_dict.go` and `writer_indexes.go`). The readiness plan records that snapshot separately from the four later governance artifacts; nothing is reverted or rebuilt.

**Files:**
- Create: `/home/sebishogun/Work/Development/simdparquet/docs/plans/2026-08-24-simdparquet-readiness.md` (the follow-on readiness plan with the PARQ-01..08 ledger and the dirty inventory)
- Modify: `/home/sebishogun/Work/Development/simdparquet/docs/roadmap.md` (roadmap summary - clean file, append)
- Modify: `/home/sebishogun/Work/Development/simdparquet/AGENTS.md`, `/home/sebishogun/Work/Development/simdparquet/CLAUDE.md` (Section 8)
- Modify: `/home/sebishogun/Work/Development/simdparquet/README.md` (review correction: current reader/writer boundary and working-plan link; preserve all unrelated dirty hunks)
- Modify: `/home/sebishogun/Work/Development/simdparquet/docs/verification.md`, `/home/sebishogun/Work/Development/simdparquet/docs/lld/file-writer.md`, `/home/sebishogun/Work/Development/simdparquet/docs/lld/encodings-and-pages.md`, `/home/sebishogun/Work/Development/simdparquet/docs/lld/compression-indexes-and-filtering.md` (review corrections: actual truncation stride, fuzz locations, and checksum policy; preserve unrelated dirty hunks)
- Do NOT touch: `docs/plans/2026-08-13-simdparquet-production.md` (historical plan stays **untouched**), and every dirty file above except the two reviewed README paragraphs named here.

**Steps:** (1) read the exact diff and untracked inventory, the historical plan's verification summary, `docs/verification.md`, `AGENTS.md`. (2) Create the readiness plan: goal (typed values, then dictionary/index, then codec/interop, then first release), PARQ-01..08 ledger (all `open`), dirty inventory section, v1 exit, workload matrix (bitpacked corpus, compressed corpus, page seams, Arrow interop round-trip - sourced to the dirty tests' names, no invented numbers). (3) Roadmap summary + Section 8; correct README's stale current-boundary and plan-index paragraphs plus the reviewed truncation-stride, fuzz-location, and checksum-policy prose while preserving every unrelated hunk. (4) `git diff --check` clean. (5) **PRIMARY-REVIEW CHECKPOINT**: only the scoped files carry wave-0 deltas; unrelated dirty hunks remain byte-identical to baseline; historical plan untouched.

---

## Task 11: simdimage (writer simdimage)

**Maturity:** `R0` - the untracked FFmpeg/runtime implementation still has ownership, callback, ABI, and platform risks. **Boundary:** image/media decode/encode over an FFmpeg runtime ABI seam plus owned frame/pixel-format types; what is owned here versus the runtime is part of the boundary record. **Oracles/peers:** FFmpeg is the ABI provider, not an oracle; libvips, OpenCV, FFmpeg/libswscale, the Rust `image` crate, and Go's `image`/`image/png` stdlib are workload and performance peers, never behavioral oracles.

**Task groups** (ledger rows, all `open`):

- `IMG-01` dirty-tree preservation and product boundary - the Task 1 inventory recorded; the product boundary stated (owned media types vs FFmpeg runtime via the ABI seam). Evidence: inventory plus boundary record. Exit: boundary recorded.
- `IMG-02` lifecycle contracts - AVIO reentrant close and registry-leak behavior. Evidence: lifecycle tests. Exit: lifecycle green.
- `IMG-03` callback concurrency contract - callback invocation from multiple goroutines. Evidence: concurrency tests. Exit: contract green.
- `IMG-04` FFmpeg platform matrix - per-FFmpeg-major/per-architecture cells. Evidence: matrix cells green per `docs/verification.md` (no claim without the matrix cell). Exit: matrix populated.
- `IMG-05` ABI reproducibility and security - table checksums and reproducible loader probes. Evidence: reproducibility checks. Exit: ABI reproducible.
- `IMG-06` SIMD dispatch and integration benchmarks - dispatch shown to reach kernels at runtime, plus integration benchmarks. Evidence: engagement tests plus benchmarks with provenance. Exit: dispatch proven.
- `IMG-07` first release. Evidence: release gates. Exit: first release.
- `IMG-08` cross-ecosystem workload decision - compare supported decode, encode, conversion, and pipeline workloads with libvips, OpenCV, FFmpeg/libswscale, Rust image, and Go image; classify material gaps as first-release blockers, future work, or rejected with evidence. Evidence: workload matrix and decisions. Exit: gaps classified.

**Competitor workload matrix rows:** the verification doc's codec/container/filter cells - none claimed green without the cell; figures only from `docs/wrong.md`/verification, quoted after reading.

**Files:**
- Modify: `/home/sebishogun/Work/Development/simdimage/docs/roadmap.md`
- Modify: `/home/sebishogun/Work/Development/simdimage/docs/plans/2026-08-13-simdimage-production.md` (append follow-on ledger)
- Modify: `/home/sebishogun/Work/Development/simdimage/AGENTS.md`, `/home/sebishogun/Work/Development/simdimage/CLAUDE.md` (Section 8)
- Do NOT touch: the untracked `internal/` and `media/` trees, `Makefile`, `docs/wrong.md`, `go.mod`, `go.sum`.

**Steps:** (1) read the exact diff and untracked inventory (ffmpegabi tables/loader, abiprobe, media types + ffmpeg runtime), roadmap, verification, AGENTS. (2) Summary + ledger (`IMG-01..08`, `open`) + v1 exit (first production release after ownership, lifecycle, callback, ABI, platform, dispatch, integration, and release gates are green) + workload matrix. (3) Section 8. (4) `git diff --check` clean. (5) **PRIMARY-REVIEW CHECKPOINT**: only the four scoped files changed; the untracked trees and the four dirty files byte-identical to baseline.

---

## Task 12: simdmetrics (writer simdmetrics - the missing-governance repository)

**Maturity:** `R0` - large dirty implementation lacks repository governance and complete corruption/resource contracts. **Boundary:** a VictoriaMetrics-compatible time-series metrics database that beats it on disk; pure Go, zero external dependencies. **Oracles/peers:** VictoriaMetrics is the API-parity target and oracle where parity is promised (the VM prod binaries are gitignored; parity tests run against the staged binary when present, skipping cleanly when absent); Prometheus is the oracle for PromQL semantics (MetricsQL is a PromQL superset); Mimir/Thanos, InfluxDB, ClickHouse, and relevant Rust peers are workload and performance peers, never behavioral oracles. `docs/design.md` stays the architecture source - linked, never duplicated as `docs/architecture.md`.

**Task groups** (ledger rows, all `open`):

- `METR-GOV-01` `AGENTS.md` + `CLAUDE.md`. Evidence: Section 8 present in both. Exit: governance in place.
- `METR-GOV-02` `ROADMAP.md`. Evidence: roadmap states shipped surface from `docs/design.md` only, non-goals, and the open work with the ledger link. Exit: roadmap live.
- `METR-GOV-03` `docs/verification.md` - derived from what already exists: `scripts/quiet-bench.sh`, `.github/workflows/ci.yml` (gofmt, vet, `go test -timeout 120s ./...`, `go test -race -timeout 300s ./...`, whitespace, fuzz smoke with discovered targets, the VictoriaMetrics compatibility job, and the six-GOARCH cross-build/vet matrix), and the package tests (`internal/{api,bench,chunk,lineproto,otlp,prompb,promql,snappy,tsdb,vmnative}`, `cmd/simdmetrics`), including the separate `SIMDMETRICS_SHAPES=1` / `TestEndpointShapes` body-shape, parameter-honouring, and path-prefix probes. No gate is invented; a gate that does not exist in those sources is not documented. Evidence: primary review confirms the inventory before the gates run (Task 15). Exit: verification doc live.
- `METR-GOV-04` `CHANGELOG.md` - no tagged release exists; an "Unreleased" section only; no invented entries. Evidence: changelog present. Exit: changelog live.
- `METR-GOV-05` `docs/plans/2026-08-24-simdmetrics-production-design.md` - derives from `docs/design.md` (shipped surface, non-goals, evidence bar) and the family design record. Evidence: design record present. Exit: design recorded.
- `METR-GOV-06` `docs/plans/2026-08-24-simdmetrics-production.md` - the staged plan carrying this ledger. Evidence: plan present. Exit: plan live.
- `METR-CORR-01` tracked `bench.test` removal - the tracked 11 MB `bench.test` binary at the repo root is removed in the implementation wave, not in wave 0. Evidence: removal commit with `git diff --check` clean. Exit: artifact gone.
- `METR-CORR-02` split dirty increments - the large dirty implementation split into per-area increments (api, otlp, prompb, snappy, vmnative, tsdb) staged task-by-task. Evidence: each increment's gates. Exit: dirty tree landed incrementally.
- `METR-CORR-03` zstd/all decoder bounds - zstd and every decoder's bounds contracts (the dirty `internal/vmnative` work). Evidence: bounds tests plus verification entries. Exit: bounds green.
- `METR-CORR-04` cluster identity collision/ownership - identity collision and ownership contracts. Evidence: contract tests. Exit: contracts green.
- `METR-CORR-05` TSDB crash/upgrade - crash recovery and upgrade-path contracts. Evidence: crash/upgrade tests. Exit: contracts green.
- `METR-CORR-06` mandatory PromQL differential - differential testing against Prometheus for the PromQL subset, and against the VM binary for the API surface. Evidence: differential suites green. Exit: differential green.
- `METR-CORR-07` measure SIMD or remove claims - `docs/design.md` records the simd-kernel wiring as a pending measured optimization; the README's claims about it are either measured on a quiet host or removed. Evidence: measurement record or claim removal. Exit: claims sourced.
- `METR-CORR-08` docs/comparisons/runner - the docs set, competitor comparisons in workload language, and the CI runner completed. Evidence: docs gates plus an observed runner run. Exit: docs/runner green.
- `METR-REL-01` compatibility and first release - the compatibility surface frozen and the first release gated. Evidence: release gates green. Exit: first release.

**Files - create exactly seven under `/home/sebishogun/Work/Development/simdmetrics/`; the dated plan pair uses 2026-08-24 and nothing is backdated:**
1. `AGENTS.md` - governance + Section 8 (paths: `ROADMAP.md`, `docs/plans/2026-08-24-simdmetrics-production.md`, `docs/verification.md`, `docs/wrong.md`)
2. `CLAUDE.md` - same substance
3. `ROADMAP.md` - canonical roadmap: shipped surface from `docs/design.md` only, non-goals, open high-level work (governance completion, corruption/resource contracts, disk-beats-VM evidence) with the ledger link
4. `docs/verification.md` - derived per METR-GOV-03
5. `CHANGELOG.md` - per METR-GOV-04
6. `docs/plans/2026-08-24-simdmetrics-production-design.md` - per METR-GOV-05
7. `docs/plans/2026-08-24-simdmetrics-production.md` - per METR-GOV-06, carrying the full ledger table (`METR-GOV-01..06`, `METR-CORR-01..08`, `METR-REL-01`, all `open`), the v1 exit, and the competitor workload matrix (rows: remote_write ingest, query_range, label scans, disk bytes/sample - figures only from `docs/design.md`/`docs/wrong.md`, quoted with source after reading)

**Retain and do not touch:** `docs/design.md`, `docs/wrong.md`, the dirty `README.md`, all dirty Go files and untracked tests, the tracked `bench.test`.

**Steps:** (1) read `git status --short` + full diff + untracked inventory, `docs/design.md`, `docs/wrong.md`, `README.md`, `scripts/quiet-bench.sh`, `.github/workflows/ci.yml`, `go.mod`. (2) Create the seven files per the above; every current-state sentence sourced to `docs/design.md`, `docs/wrong.md`, or the dirty README. (3) `git diff --check` clean. (4) **PRIMARY-REVIEW CHECKPOINT**: exactly seven new files; `docs/design.md`/`docs/wrong.md`/README/dirty Go files/`bench.test` byte-identical to baseline; verification-doc gate inventory confirmed against scripts/, ci.yml, tests (this confirmation precedes Task 15's run); no commit.

---

## Task 13: simdlogs (writer simdlogs)

**Maturity:** `R3` - phases 0-10 are committed locally (clean tree, HEAD `47bc3a9`); quiet evidence, observed CI, soak, and release rehearsal remain. **Boundary:** a logs-only database with authenticated tenancy, static application-level sharding, and the documented Elasticsearch subset; VictoriaLogs-compatible LogsQL. **Oracles/peers:** VictoriaLogs is the parity oracle where compatibility is promised (`docs/vl-parity.md`); Loki, Elasticsearch/OpenSearch, ClickHouse, and Vector are workload and performance peers, never behavioral oracles.

**v1 exit (correction):** the **full production exit** - phases 0-10 complete (already committed locally), including the static-cluster contract and release artifacts, plus the release gate set green. Every v1 exit reference below uses this; nothing in this plan references a phases 0-7 exit.

**Task groups** (ledger rows, all `open`):

- `LOGS-V1-01` quiet benchmark provenance - `docs/release-readiness.md`'s standing blocker: the published tables predate `requireQuiet` and carry no machine/commit record; the gate now covers `perops_test.go` too; `SIMDLOGS_BENCH_NOISY` must stay unset and the evidence must show the benchmark executed rather than skipped (the gate skips above load 1). Implementation wave only, under the Section 6 protocol. Evidence: corrected tables with provenance. Exit: blocker cleared.
- `LOGS-V1-02` push/observed CI - the workflows are authored and their YAML parses, but none has been observed running (that is how the `release.yml` dry-run bug was found and corrected); push requires the user's permission and the task stops at the evidence checkpoint until asked. Evidence: observed CI result. Exit: CI observed.
- `LOGS-V1-03` long fuzz/soak - fuzz campaigns beyond the current short runs and the soak modes beyond what has passed (`scripts/soak.sh`: dev and release modes, durations and exit criteria from `docs/release-readiness.md` and `docs/verification.md`). Evidence: completed runs. Exit: soak/fuzz evidence complete.
- `LOGS-V1-04` stale docs/comments corrective - the stale source comments and records (the `es.go` package comment, the `scale_test.go` header comment, the duplicated entry-37 heading in `docs/wrong.md`, and stale roadmap task references) reconciled; source changes remain a code task staged in the ledger, never a docs-session drive-by. Evidence: the diff plus the green doc gates. Exit: stale items fixed and record references unambiguous.
- `LOGS-V1-05` bounded-ingest decision - the open bounded-ingest decision taken with measurement; implemented, or rejected with a reopen condition in `docs/wrong.md`. Evidence: the decision record. Exit: decision recorded.
- `LOGS-V1-06` release rehearsal - the release path run end-to-end without tagging (the `scripts/release-check.sh` artifact is the gate). Evidence: the rehearsal run. Exit: rehearsal green.
- `LOGS-V1-07` v1.0 - the full production exit: phases 0-10 plus the release gate set green. Evidence: the release gates green. Exit: v1.0.
- `LOGS-V1-08` workload-backed ecosystem-gap decision - `docs/vl-parity.md` and the ecosystem documents reconciled against concrete ingest, query, storage, recovery, and operations workloads from VictoriaLogs, Loki, Elasticsearch/OpenSearch, ClickHouse, and Vector; each material gap classified as a v1 blocker, post-v1 work, or rejected with evidence; no feature is added merely for parity. Evidence: the decision record. Exit: decisions recorded.
- `LOGS-IO-01` io_uring decision - **open/deferred, non-v1-blocking** (no measurement supports `rejected`; nothing goes into `docs/wrong.md` without a measurement): queries are mmap; durable ingest has file/directory/manifest sync barriers. First instrument the ingest stages to measure explicit I/O waits; prototype an io_uring path only if explicit I/O waits are at least 30% of durable-ingest time; retain it only for a repeatable at least 20% end-to-end throughput or p99 improvement while the durability and conformance gates remain green; otherwise leave it deferred. Evidence: the instrumented measurement or the deferred decision record. Exit: decision recorded with measurement.

**Files:**
- Modify: `/home/sebishogun/Work/Development/simdlogs/docs/release-readiness.md` (append the task ledger with the rows above; keep the existing Gates/Blockers text untouched)
- Modify: `/home/sebishogun/Work/Development/simdlogs/docs/roadmap.md` (roadmap summary: open release-blocker work with the ledger link)
- Modify: `/home/sebishogun/Work/Development/simdlogs/AGENTS.md`, `/home/sebishogun/Work/Development/simdlogs/CLAUDE.md` (Section 8; ledger path = `docs/release-readiness.md`)
- Modify: `/home/sebishogun/Work/Development/simdlogs/README.md` (review correction: historical labels and provenance caveats only)
- Modify: `/home/sebishogun/Work/Development/simdlogs/docs/verification.md`, `/home/sebishogun/Work/Development/simdlogs/docs/lld/cluster.md` (review correction: source-backed router-merge status and the working AGENTS/CLAUDE sync command)
- Do NOT touch: `docs/plans/2026-08-13-simdlogs-production.md` (completed implementation record - no edit), `docs/plans/2026-08-13-simd-family-production-documentation.md` (the superseded historical family-documentation record - **explicitly preserved, body untouched**; the ledger links it as superseded by the GO_SIMD index).

**Steps:** (1) read `docs/release-readiness.md`, `docs/roadmap.md`, the production plan's exit sections, `docs/verification.md` (soak section), `AGENTS.md`. (2) Append the ledger: the rows above (all `open`, including `LOGS-IO-01` as open/deferred), the v1 exit (full production exit, phases 0-10 plus release gates), and the competitor workload matrix (rows from `docs/vl-parity.md` and the ecosystem docs - quoted, not invented). (3) Roadmap summary + Section 8; apply the source-backed router-status, sync-command, and historical-provenance corrections found by primary review. (4) `git diff --check` clean. (5) **PRIMARY-REVIEW CHECKPOINT**: only the seven scoped files changed; both historical plan files byte-identical; ledger rows all `open`; `LOGS-IO-01` labeled open/deferred with its instrumentation precondition; no commit.

---

## Task 14: Family index (writer GO_SIMD-B - part 2 of 2, after Tasks 3-13 pass review)

**Files:**
- Create: `/home/sebishogun/Work/Development/GO_SIMD/docs/plans/2026-08-24-simd-family-production-readiness.md`

**Step 1: Read the final state.** All eleven repos' post-wave docs must exist (Tasks 3-13 review checkpoints passed). Read the final `docs/roadmap.md`/`ROADMAP.md`, ledgers, and verification docs to extract the *final* links and the shared-gate names. Expected: every ledger's task IDs and every roadmap's ledger link are final.

**Step 2: Write the index.** It contains, and only contains:
1. **Taxonomy** - eleven rows: module path, one-line product boundary, R0-R5 level from design Section 5.1's 2026-08-24 snapshot with its basis, and links to the repo's canonical roadmap, production design, production plan/ledger, verification doc, and `docs/wrong.md`. Levels are a dated, linked snapshot - the index cannot promote a repository.
2. **Release waves** - wave 0 (this documentation wave) and the first implementation wave (the GO_SIMD corrective slice, simdjson v0.8 stabilization and v1 evidence, and simdlogs v1 release evidence), stated as ordering information with links to the owning ledgers only; no task IDs or per-task statuses are copied into the index, and no repository's release is blocked by another's. GO_SIMD R5 work and deferred `LOGS-IO-01` are not part of the first implementation wave.
3. **Links and shared gates** - only gates that exist in the linked verification docs: differential conformance vs the portable reference, bench-check with stored baselines, race/vet, cross-architecture tiers, no-panic/no-hang limits, fuzz with timeouts, per-repo release verification sets. A gate that does not exist in a linked `docs/verification.md` is not listed.
4. **Status legend** - the seven states and the transition rule (a transition is an edit in the owning ledger; `rejected` is terminal without a reopen condition).
5. **Quiet-host protocol summary** - condensed from design Section 6 for the later simdjson/simdlogs implementation waves: quiet-window coordination by the primary agent, provenance record (source identity, CPU/tier, Go version, kernel, GOMAXPROCS, affinity, governor, load averages), load-below-1 requirement with the existing gates (`requireQuiet`, `benchcheck -maxload 1`, GO_SIMD's pinned-core harness), fixed-source-state measurement, bounded waits with explicit timeouts, restore-after, discard-and-rerun rules. The index records the protocol; this wave does not execute it.

The index never contains per-task status, never restates a roadmap item, never states a speed claim, and never quotes a benchmark - it links. Current-state sentences are source-backed; un-sourced sentences are marked planned. ASCII only.

**Step 3: Check.** Run `git -C /home/sebishogun/Work/Development/GO_SIMD diff --check`. Expected: clean.

**Step 4: PRIMARY-REVIEW CHECKPOINT.** Primary agent checks: every link resolves to the stated file in the stated repository (by hand); all eleven repos present once; R0-R5 rows match design Section 5.1; no per-task status, no speed claims; shared gates all exist in linked verification docs; quiet-host protocol present; the index is the only file this task creates (GO_SIMD's wave-0 file set is exactly: the four Task-3 files, the Task-14 index, and the two pre-existing untracked records); no commit.

---

## Task 15: Serialized wave-0 verification (primary agent; no writers running)

Run strictly serially - one repo at a time - to avoid machine pressure. Every `go`/`make` invocation carries shell `timeout` AND native `-timeout` where Go accepts it; gates run bare (never piped through `tail` without `set -o pipefail`); a hung binary is a leak alarm, not a retry candidate. No benchmarks, no codegen, no release gates, no quiet windows.

| Repo | Commands (serialized) | Expected |
|---|---|---|
| GO_SIMD | `timeout 600 go test -timeout 540s ./internal/tests/docs`; `git diff --check` | PASS; clean |
| simdblas | manual link check over every `.md` touched; `git diff --check` | links resolve; clean |
| simdcsv | link check per `AGENTS.md` (internal links + the two named external links); `git diff --check` | resolve; clean |
| simdvec | link check; `git diff --check` | resolve; clean |
| simdjson | link check of the new readiness plan (AGENTS rule); `git diff --check` | resolve; clean |
| simdhttp | Markdown checks per `AGENTS.md` item 5 (links, dead references, trailing whitespace); `git diff --check` | clean |
| simdcbor | link check; `git diff --check` | resolve; clean |
| simdparquet | link check; `git diff --check` | resolve; clean |
| simdimage | link check; `git diff --check` | resolve; clean |
| simdmetrics | FIRST confirm the verification inventory (Task 12's review did): gates exist in `scripts/quiet-bench.sh`, `.github/workflows/ci.yml`, package tests. Then `timeout 300 go test -timeout 120s ./...`; `git diff --check` | PASS; clean |
| simdlogs | `timeout 300 go test -timeout 120s ./internal/tests/docs`; `git diff --check` | PASS; clean |
| index (GO_SIMD) | manual link check of the index (not in any machine-gated link set); `git diff --check` | resolve; clean |

**Step 1:** Run the table top to bottom, one repo at a time, recording each output. Any failure is diagnosed (systematic-debugging), fixed by the owning writer in its scoped files only, and re-run; the result is never used to invent an easier gate.

**Step 2:** Re-run Task 1's baseline comparison per repo. Expected: the only changes since baseline are the wave-0 scoped files; all dirty user changes byte-preserved.

---

## Task 16: Cross-repo consistency review (fresh DeepSeek reviewer)

One review pass over the family as a whole, per design Sections 12.3-12.4. Checks: every repo appears in the index once with correct links; every repo's AGENTS/CLAUDE carries the Section 8 section; no repo except simdmetrics lacks governance files; no rejected task appears as open anywhere; no duplicate task status in the index; interleaved claims (dependency versions, shared contracts) checked against each other and their sources in the same pass; oracle columns used only where a compatibility promise exists. Findings are reported in the terminal with file paths; defects return to the owning writer, then re-run the affected Task 15 commands.

---

## Task 17: Primary-agent final review and report

**Step 1: Final review.** Walk every diff one last time against the acceptance conditions: exact file scope per design Section 7 and this plan; current/planned language separated; task-ID uniqueness family-wide; historical text/IDs preserved; dirty changes preserved (old hunks intact, new sections appended); no commit before Task 18 and no push/tag/release/fetch/reset/revert/stash/checkout/clean anywhere; wave-0 artifacts contain no implementation change; no unverified numerical claim anywhere.

**Step 2: Report in the terminal.** Per repository: files changed, review findings and resolutions, verification output. No artifact, dashboard, or report file is produced (the repository record convention applies). State the wave's completion and that the first implementation wave (GO_SIMD corrective, simdjson v0.8 to v1, simdlogs v1.0) is staged in the ledgers, to be executed through each owning repo's executing-plans workflow with the quiet-host protocol.

**Step 3: Design consistency.** Confirm the design record and this plan both state the simdlogs full production exit (phases 0-10 plus release gates), the same first-wave boundary, and the same five permitted family-index sections (taxonomy, release waves, shared gates/links, status legend, quiet-host protocol) before reporting completion.

---

## Task 18: User-authorized preservation commits

On 2026-08-26 the user separately authorized commits for the complete current
state of all eleven repositories after review and bounded verification. This is
a preservation checkpoint, not permission to push, tag, publish, or release,
and it does not move any ledger row to `shipped`.

For each repository, inspect `git status`, the full diff, and recent commit
style; run its bounded pre-commit gates; stage the complete intended current
tree (including the preserved pre-wave source work the user asked to commit);
review the staged diff and file inventory; then create one repository-local
checkpoint commit whose message describes that repository's actual contents.
No secret, ignored artifact, or unrelated external path is staged. Finish with
`git status --short --branch` in all eleven repositories and report every commit
ID and any gate that could not be run.
