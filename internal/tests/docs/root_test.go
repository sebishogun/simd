package docs

import "path/filepath"

// repoRoot is where these checks read from. The tests used to live in the
// repository root and opened "README.md" directly; they now sit three
// directories down, and `go test` runs each package in its own directory.
const repoRoot = "../../.."

func path(rel string) string { return filepath.Join(repoRoot, rel) }
