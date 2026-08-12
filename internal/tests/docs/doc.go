// Package docs checks the repository's active documentation against the tree:
// the counts the documents carry (exported functions and current version in
// the README, generated kernels in docs/platforms.md), the identifiers they
// name, the tiers CONTRIBUTING asks for runs on, that every operation in the
// docs/api.md catalog has a runnable example, that local links resolve, and
// that every release tag has a CHANGELOG section.
//
// It sits three levels down, so every path it opens is relative to repoRoot
// rather than to the working directory.
package docs
