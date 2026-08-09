# Documentation and v1.20.0 release design

## Goal

Make the repository documentation a readable technical guide to the current
v1.20.0 implementation, preserve historical records, and publish the existing
v1.20.0 tag as the latest GitHub release.

## Audience

The primary reader is a Go developer deciding whether and how to use the
package. Secondary readers are contributors adding kernels, maintainers
checking platform coverage, and users evaluating the related SIMD libraries.

## Documentation structure

The root README is the technical entry point rather than the complete manual.
It will retain:

- package purpose and boundaries;
- installation and Go version;
- a compilable quick start;
- in-place and `Into` conventions, allocation, sizing, and dispatch behavior;
- accuracy guarantees;
- representative performance evidence and reproduction commands;
- a compact platform summary;
- an audited list of libraries built on this package;
- links for users and contributors.

The target is roughly 300 to 350 lines. Detailed material moves into focused
active documents:

- `docs/api.md` contains the task-oriented operation catalog, supported types,
  API conventions, and the v1.19/v1.20 additions.
- `docs/platforms.md` contains instruction-set tiers, real-hardware status,
  kernel coverage, portable fallbacks, and backend ABI limits.
- `docs/README.md` maps readers to the README, tutorial, guides, references,
  examples, and contributor material.

The tutorial and five task guides remain the main usage documentation. They
will be checked against the implementation and expanded where v1.19 and v1.20
introduced workflows that are not yet covered.

## Historical boundary

Research notes, completed plans, hardware reports, old changelog entries, and
`docs/wrong.md` remain records of what was known when they were written. They
change only for broken navigation or a sentence that explicitly claims to
describe the current tree.

The missing changelog sections for v1.6.0 through v1.9.1 will be reconstructed
from their public tags and tag-to-tag commit ranges. Adding missing release
records does not rewrite existing records.

## Sources of truth

Each current claim will be checked against the closest executable or versioned
source:

- exported names and counts: default-build Go declarations;
- behavior, sizing, aliasing, and fallback: public implementations, dispatch
  paths, and tests;
- kernel counts and tier coverage: generated dispatch inventories and emission
  checks;
- Go version and dependencies: `go.mod`;
- real-hardware qualification: `testdata/hardware/`;
- performance: committed benchmark records and measured release evidence;
- related libraries: their current source and release state;
- release history: public tags and tag-to-tag commit ranges.

No new performance claim will be inferred from implementation shape alone.

## Verification design

Documentation checks will follow content moved out of the README:

- the operation table and example coverage check follow `docs/api.md`;
- platform and kernel-count checks follow `docs/platforms.md`;
- active-document API references must name exported identifiers;
- local documentation links must resolve;
- each published stable tag must have a changelog heading;
- current-version text must agree with the latest changelog section;
- complete example programs must compile.

The documentation package, full Go test suite, repository verification target,
and relevant generated-inventory checks run before publication.

## Release design

`v1.20.0` already exists publicly and resolves to commit `09ff317`. It will not
be moved or recreated. The GitHub release will:

- use the existing `v1.20.0` tag;
- be latest, non-draft, and non-prerelease;
- lead with the v1.20 columnar-codec additions;
- summarize v1.3 through v1.20 because the previous GitHub release is v1.2.0;
- state the Go 1.25 minimum, no-cgo deployment, v1 API compatibility, and
  architecture qualification;
- link the full changelog.

Immediately before publication, the workflow must confirm the remote tag still
resolves to `09ff317`, no v1.20.0 GitHub release exists, and required checks
pass. Any mismatch stops publication.
