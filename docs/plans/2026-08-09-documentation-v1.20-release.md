# Documentation and v1.20.0 Release Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the oversized, partially stale documentation with a source-backed technical README and focused references, then publish the existing v1.20.0 tag as the latest GitHub release.

**Architecture:** Keep the root README as the technical front page, including representative performance evidence and the project ecosystem. Move the large operation catalog and backend inventory to `docs/api.md` and `docs/platforms.md`, and move the existing documentation tests with them. Treat Go doc comments as public documentation, preserve historical records, and derive current facts from Go declarations, generated assembly, hardware reports, tags, and related repositories.

**Tech Stack:** Go 1.25, Go AST and filesystem-based documentation tests, Markdown, generated Plan 9 assembly, Git, and GitHub CLI.

---

### Task 1: Add failing integrity checks for the new documentation layout

**Files:**
- Modify: `internal/tests/docs/root_test.go`
- Modify: `internal/tests/docs/readme_examples_test.go`
- Modify: `internal/tests/docs/readme_facts_test.go`
- Modify: `internal/tests/docs/docs_guide_test.go`
- Create: `internal/tests/docs/links_test.go`
- Create: `internal/tests/docs/changelog_test.go`

**Step 1: Add a general document reader**

Add this helper to `internal/tests/docs/root_test.go` and make `readReadme` call
it:

```go
func readDoc(t *testing.T, rel string) string {
	t.Helper()
	src, err := os.ReadFile(path(rel))
	if err != nil {
		t.Fatalf("reading %s: %v", rel, err)
	}
	return string(src)
}
```

Add the required `os` and `testing` imports.

**Step 2: Point operation-catalog checks at `docs/api.md`**

In `readme_examples_test.go`, read `docs/api.md`, rename
`readmeTableFunctions` to `operationTableFunctions`, and update failure messages
to say "operation catalog" rather than "README table". Keep the lower bound of
50 parsed operations so a broken parser cannot pass vacuously.

**Step 3: Point platform checks at `docs/platforms.md`**

In `readme_facts_test.go`:

- keep exported-function count and current-version checks on `README.md`;
- read `docs/platforms.md` for the kernel total and coverage rows;
- read `docs/platforms.md` in `TestContributingMatchesVerificationTable`;
- rename test and error text where it now describes the platform reference.

Add a source-backed count check that parses architecture rows and counts `TEXT`
declarations in committed assembly:

```go
func generatedKernelCount(t *testing.T, arch string) int {
	t.Helper()
	names, err := filepath.Glob(path(filepath.Join("internal", arch, "*.s")))
	if err != nil {
		t.Fatalf("listing %s assembly: %v", arch, err)
	}
	if len(names) == 0 {
		t.Fatalf("no generated assembly found for %s", arch)
	}
	n := 0
	text := regexp.MustCompile(`(?m)^TEXT `)
	for _, name := range names {
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("reading %s: %v", name, err)
		}
		n += len(text.FindAll(src, -1))
	}
	return n
}
```

Compare each documented kernel count with this value, not only with the sum of
the same Markdown table.

**Step 4: Expand API-reference checks to all active user guides**

Refactor `TestGuidesNameRealFunctions` to check:

```text
README.md
docs/api.md
docs/tutorial.md
docs/guide/README.md
docs/guide/*.md
```

Keep the `simd.Name` parser and the non-vacuous lower bound.

**Step 5: Add local-link validation**

Create `links_test.go`. Check local Markdown links in the active documents
listed above plus `CONTRIBUTING.md`, `ROADMAP.md`, `docs/README.md`,
`docs/kernels.md`, and `docs/examples/README.md`. Ignore `http:`, `https:`,
`mailto:`, and anchor-only links. Strip an anchor from a local target, resolve
it relative to the source document, and require `os.Stat` to succeed. Do not
scan historical plans, research notes, hardware reports, or `docs/wrong.md`.

**Step 6: Add tag-to-changelog validation**

Create `changelog_test.go`. Run `git -C <repoRoot> tag --list 'v*'`; skip only
when the checkout has no Git metadata. For every stable semver tag, require an
exact `## <tag>` heading in `CHANGELOG.md`. This test must report the six
currently missing versions: v1.6.0, v1.7.0, v1.7.1, v1.8.0, v1.9.0, and
v1.9.1.

**Step 7: Run the documentation tests and confirm the intended failures**

Run: `go test ./internal/tests/docs`

Expected: FAIL because `docs/api.md` and `docs/platforms.md` do not exist and
because six stable tags have no changelog heading.

**Step 8: Commit the failing checks**

```bash
git add internal/tests/docs
git commit -m "test: bind documentation to current sources"
```

### Task 2: Create the task-oriented API reference

**Files:**
- Create: `docs/api.md`
- Modify: `README.md`
- Test: `internal/tests/docs/readme_examples_test.go`
- Test: `internal/tests/docs/docs_guide_test.go`

**Step 1: Create `docs/api.md` from the current operation catalog**

Give the page this structure:

```markdown
# API guide

## Conventions
## Allocation and workspace
## Lengths, overlap, and malformed input
## Find an operation by task
## Supported element types
## v1.19 and v1.20 additions
```

Move the full `| I want to... | Call |` table from `README.md` without dropping
rows. Preserve backticked exported names so the example and API-name checks keep
working.

**Step 2: Correct the cross-cutting API rules**

State these rules narrowly enough to match the implementation:

- Plain arithmetic names generally mutate their first slice; `Into` generally
  takes caller-owned output or workspace.
- Kernel and `Into` paths are designed not to allocate.
- Convenience functions that return slices, create plans or workspaces, or grow
  append-style destinations can allocate and document that behavior.
- Elementwise operations usually process the shortest slice, but output-shaped
  operations can require exact capacity, return a count, do nothing on invalid
  dimensions, or reject malformed input. Their function documentation is the
  contract.
- Partial overlap is operation-specific; do not claim every destination may
  alias every input.

Name representative allocating APIs: `Sort`, `Median`, `Quantile`, `TopK`,
`BottomK`, `Histogram`, `Bincount`, `FFT`, `RFFT`, plan/workspace constructors,
and append-style functions when capacity is insufficient.

**Step 3: Document the v1.19 and v1.20 groups**

Add concise entries for:

- data movement: `RunLengthDecodeInt32`, `RandFillU64`,
  `MergeSortedUint32`, `Interleave2`, `Deinterleave2`, and
  `Transpose8x8Bytes`;
- columnar codecs: the width-specialized `BitUnpackInto` path,
  `VarintDecode`, `HashUint64`, `Bitshuffle`, and `Unbitshuffle`.

Describe observable contracts, not implementation slogans. Link each group to
the relevant guide.

**Step 4: Remove the moved table and duplicated catalog prose from README**

Replace them with a short task-to-guide table and a direct link to
`docs/api.md`.

**Step 5: Run focused tests**

Run: `go test ./internal/tests/docs -run 'TestReadmeTableHasExamples|TestGuidesNameRealFunctions'`

Expected: PASS.

**Step 6: Commit**

```bash
git add README.md docs/api.md internal/tests/docs
git commit -m "docs: move the operation catalog into an API guide"
```

### Task 3: Create the source-backed platform reference

**Files:**
- Create: `docs/platforms.md`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `testdata/hardware/README.md`
- Test: `internal/tests/docs/readme_facts_test.go`

**Step 1: Count the current generated kernels**

Run: `make check-emission`

Expected: PASS and a per-target emitted/skipped summary. Compare it with the
counts obtained from committed `TEXT` declarations by the new test. Resolve any
difference before editing prose.

**Step 2: Build `docs/platforms.md`**

Move and tighten these README sections:

- platform support and real-hardware status;
- generated-kernel coverage and skip counts;
- fallback behavior;
- ABI limitations on s390x, ppc64le, and loong64;
- OS support;
- the optional `goexperiment.simd` escape hatch.

Correct the optional vector type to amd64-only, matching
`//go:build goexperiment.simd && amd64` in `vec.go`. Make clear that the ordinary
slice API needs no experiment and remains available on all supported targets.

**Step 3: Keep a compact README platform summary**

Retain the correctness/wall-clock matrix and a link to `docs/platforms.md`, but
remove the long kernel-coverage and ABI-wall discussion from the root README.

**Step 4: Keep contribution requests synchronized**

Update `CONTRIBUTING.md` and `testdata/hardware/README.md` only if the
source-backed platform table changes which tiers lack real-hardware reports.

**Step 5: Run focused tests**

Run: `go test ./internal/tests/docs -run 'TestReadmeCountsAreCurrent|TestContributingMatchesVerificationTable'`

Expected: PASS with 493 exported functions and every platform count matching
committed assembly.

**Step 6: Commit**

```bash
git add README.md docs/platforms.md CONTRIBUTING.md testdata/hardware/README.md internal/tests/docs
git commit -m "docs: separate platform coverage from the README"
```

### Task 4: Rewrite the README as a technical front page

**Files:**
- Modify: `README.md`
- Modify: `simd.go:1-79`
- Modify: `numeric.go:5-15,140-142`

**Step 1: Invoke the copy-editing skill**

Use the available copy-editing workflow before rewriting existing prose. Keep
technical claims and measurements intact while cutting repetition and broad
claims the implementation does not support.

**Step 2: Rebuild the README in this order**

```text
Title and badges
What it is
Install and requirements
Quick start
API model and allocation boundary
When SIMD pays
Accuracy contract
Operation areas and documentation map
Performance
Platform summary
Built on simd
Repository and development links
Status and license
```

Keep the performance machine, baselines, representative tables, caveats, and
`go run ./cmd/site` reproduction path. Remove repeated explanations already in
the tutorial or focused references. Target roughly 300 to 350 lines, but prefer
complete technical meaning over a hard line count.

**Step 3: Correct stale or conflicting README facts**

At minimum:

- change the competitor table from 457 to 493 exported functions or avoid a
  count in that row and link the source-backed current count;
- remove the claim that no function allocates;
- remove the claim that one pair of naming conventions covers every API;
- change optional vector-type availability from amd64 and arm64 to amd64 only;
- retain Go 1.25+, no cgo, one transitive dependency, and per-operation linker
  elimination;
- distinguish bit-identical core operations from documented transcendental and
  `Fast*` contracts without claiming every exported helper has one identical
  rule;
- preserve the amd64-only qualification on all measured speed figures.

**Step 4: Align package documentation with the same contract**

Rewrite the package overview in `simd.go` so it describes the common in-place
and `Into` shapes without presenting them as universal. Replace "This package
never allocates" with the caller-owned fast-path/convenience-helper distinction.

In `numeric.go`, replace "none of these allocate" and "the only function in the
package that allocates" with statements scoped to the step routines and
`NewRK4Workspace`.

**Step 5: Format and test package documentation**

Run: `gofmt -w simd.go numeric.go`

Run: `go test ./internal/tests/docs && go test .`

Expected: PASS.

**Step 6: Commit**

```bash
git add README.md simd.go numeric.go
git commit -m "docs: make the README a technical front page"
```

### Task 5: Bring active guides and navigation through v1.20

**Files:**
- Modify: `docs/README.md`
- Modify: `docs/tutorial.md`
- Modify: `docs/guide/README.md`
- Modify: `docs/guide/arrays.md`
- Modify: `docs/guide/search.md`
- Modify: `docs/guide/encoding.md`
- Modify: `docs/examples/README.md`
- Modify: `simd.go:432-476`

**Step 1: Update the documentation map**

Make `docs/README.md` identify the intended reading order and include
`api.md` and `platforms.md`. Distinguish user guides, contributor references,
and historical records.

**Step 2: Correct allocation guidance in active prose**

In the tutorial and guide index, replace package-wide zero-allocation claims
with the fast-path/convenience distinction used in `docs/api.md`. In the arrays
guide, replace "all four hundred and fifty operations" with wording that does
not drift and retain exact allocation notes where they are function-specific.

In `RollingMinInto` documentation and the search guide, change "the only
allocating operation in the library" to a scoped statement: an internal deque
would require workspace proportional to the window and would break this
function's caller-owned-storage shape.

**Step 3: Add the v1.19 workflows**

In `docs/guide/search.md`, document `MergeSortedUint32` and revise the old
statement that vector merge machinery has not been built. Keep union absent,
but explain that merge preserves both inputs and does not deduplicate.

In `docs/guide/encoding.md`, add run-length decode, byte-plane
interleave/deinterleave, and 8x8 transpose where they fit the columnar pipeline.

**Step 4: Add the v1.20 workflows**

In `docs/guide/encoding.md`:

- explain that `BitUnpackInto` selects width-specialized kernels internally;
- show `VarintDecode` with `n, consumed` and resumable slicing;
- show `HashUint64` for bulk numeric keys and state that one string belongs to
  `hash/maphash`;
- show a `Bitshuffle`/compress/`Unbitshuffle` pipeline and the 64-byte multiple
  requirement.

Use snippets that name real exported functions and match their sizing rules.

**Step 5: Run documentation and example tests**

Run: `go test ./internal/tests/docs ./docs/examples/...`

Expected: PASS.

**Step 6: Commit**

```bash
git add docs simd.go
git commit -m "docs: cover the v1.19 and v1.20 workflows"
```

### Task 6: Restore missing changelog history and refresh the roadmap

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `ROADMAP.md`
- Test: `internal/tests/docs/changelog_test.go`

**Step 1: Reconstruct the missing tagged sections**

Insert these sections between v1.10.0 and v1.5.0, newest first:

- v1.9.1: word-aligned `JSONMasks` regions and exported `MaskWords`;
- v1.9.0: `JSONMasks`, five classifications from one pass;
- v1.8.0: `JSONCopyRun`, scan and copy fused;
- v1.7.1: portable mask builders changed to word-at-a-time with corrected bit
  positions;
- v1.7.0: `IndexAnyOrLess`;
- v1.6.0: full UTF-8 grammar checked by predecessor-aligned vector comparisons.

Use the public commit messages at `8091271`, `bbc9baf`, `88237bd`, `ea000ae`,
`51e3ebc`, and `eb6997e`. Preserve their measured qualifications and avoid
adding conclusions not present in those commits.

**Step 2: Refresh active roadmap facts**

Change the current v1 line from v1.13.0 to v1.20.0. Update current ppc64le and
s390x counts from the source-backed platform table. Separate historical count
progressions from current totals so later readers can tell which is which.

Do not rewrite completed plans or historical decision text merely because an
old count was correct at the time.

**Step 3: Run the changelog and link tests**

Run: `go test ./internal/tests/docs -run 'TestStableTagsHaveChangelogSections|TestLocalMarkdownLinks'`

Expected: PASS.

**Step 4: Commit**

```bash
git add CHANGELOG.md ROADMAP.md internal/tests/docs
git commit -m "docs: restore v1.6 through v1.9 release history"
```

### Task 7: Audit and expand the SIMD ecosystem section

**Files:**
- Modify: `README.md`

**Step 1: Inventory public related repositories**

Run:

```bash
gh repo list sebishogun --limit 100 --json name,description,url,isPrivate,pushedAt,latestRelease
```

Expected public dependents: `simdblas`, `simdjson`, `simdcsv`, `simdvec`,
`simdhttp`, `simdcbor`, `simdparquet`, `simdimage`, and `simdlogs`.

**Step 2: Check each repository's current evidence**

Read each repository's README, `go.mod`, benchmark records, and latest release.
Confirm that it imports `github.com/sebishogun/simd`, which primitives it uses,
and which performance claims are supported. Do not turn a repository
description into a stronger claim.

**Step 3: Replace "Built on this" with an audited ecosystem table**

Split released libraries from public projects without releases, or mark release
state in one compact table. For each project give one sentence on its workload,
the key simd primitive or pipeline, and a qualified end-to-end result when the
project contains one.

Keep this section in the root README as requested.

**Step 4: Run documentation tests**

Run: `go test ./internal/tests/docs`

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update the libraries built on simd"
```

### Task 8: Run full verification and review the documentation diff

**Files:**
- Modify only files required by failures found in this task

**Step 1: Run formatting and focused documentation gates**

Run: `gofmt -w internal/tests/docs/*.go simd.go numeric.go`

Run: `go test ./internal/tests/docs`

Expected: PASS.

**Step 2: Run the full Go suite**

Run: `go test ./...`

Expected: PASS.

**Step 3: Run repository verification**

Run: `make verify`

Expected: PASS. Run this command bare, not through a pipe.

**Step 4: Re-run generated inventory**

Run: `make check-emission`

Expected: PASS and counts matching `docs/platforms.md`.

**Step 5: Inspect the final diff and repository status**

Run: `git diff HEAD~7 --check`

Run: `git status --short`

Expected: no whitespace errors and no unintended files.

**Step 6: Commit any verification-only corrections**

```bash
git add <only-corrected-files>
git commit -m "docs: fix documentation verification findings"
```

Skip this commit if verification required no changes.

### Task 9: Prepare and publish the existing v1.20.0 GitHub release

**Files:**
- Create temporarily outside the repository: `/tmp/opencode/simd-v1.20.0-notes.md`

**Step 1: Verify immutable release inputs**

Run: `git ls-remote --tags origin refs/tags/v1.20.0`

Expected SHA: `09ff317a7fd4f1313b8cd811d7038552573f8b3a`.

Run: `gh release view v1.20.0`

Expected before publication: not found. If it exists, stop rather than creating
a duplicate or editing it without approval.

**Step 2: Verify checks for the tagged commit**

Use `gh api` or `gh run list --commit 09ff317...` to confirm required checks on
the tag commit are green. The current-tree verification in Task 8 does not
replace tag-commit status.

**Step 3: Draft cumulative release notes**

Write `/tmp/opencode/simd-v1.20.0-notes.md` with this structure:

```markdown
v1.20 adds the columnar codec tier: width-specialized bit unpacking,
wide-load varint decoding, bulk uint64 hashing, and bitshuffle round trips.

## v1.20 highlights
## Since the last GitHub release (v1.2.0)
## Compatibility and requirements
## Verification and platform scope
## Full changelog
```

The cumulative section should group v1.3-v1.13 text/JSON work, v1.14 dispatch
and binary-size work, v1.15 columnar validity, and v1.16-v1.20 checksums,
formatting, decoding, data movement, and codecs. State Go 1.25+, no cgo, v1 API
compatibility, and that published wall-clock figures are amd64 measurements.
Link `CHANGELOG.md` on GitHub.

**Step 4: Review the exact release payload**

Read the notes file, confirm every named function exists, and compare its claims
with `CHANGELOG.md` and the tag commit. Confirm release title
`simd.go v1.20.0`, non-draft, non-prerelease, latest.

**Step 5: Publish**

Run:

```bash
gh release create v1.20.0 --verify-tag --title "simd.go v1.20.0" --notes-file /tmp/opencode/simd-v1.20.0-notes.md --latest
```

Expected: a GitHub release URL.

**Step 6: Verify the published release**

Run:

```bash
gh release view v1.20.0 --json name,tagName,isDraft,isPrerelease,isLatest,url
```

Expected: tag `v1.20.0`, title `simd.go v1.20.0`, draft false, prerelease false,
latest true.

### Task 10: Final repository check

**Files:**
- None unless an unexpected problem is found

**Step 1: Check status and recent commits**

Run: `git status --short`

Run: `git log --oneline -10`

Expected: clean working tree and the planned documentation commits only.

**Step 2: Report the result**

Provide the release URL, changed documentation map, corrected technical facts,
test commands and outcomes, and whether local documentation commits remain
unpushed. Do not claim a push unless one was explicitly performed.
