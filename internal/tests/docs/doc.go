// Package docs checks the repository's own documentation against the tree:
// the counts in the README, the identifiers it names, the tiers CONTRIBUTING
// asks for runs on, and that every operation in the README's index has a
// runnable example.
//
// It sits three levels down, so every path it opens is relative to repoRoot
// rather than to the working directory.
package docs
