package docs

import (
	"os"
	"path/filepath"
	"testing"
)

// repoRoot is where these checks read from. The tests used to live in the
// repository root and opened "README.md" directly; they now sit three
// directories down, and `go test` runs each package in its own directory.
const repoRoot = "../../.."

func path(rel string) string { return filepath.Join(repoRoot, rel) }

// readDoc reads a document from the repository, failing the test if it is
// missing or unreadable: a document these tests bind to the sources has to
// exist before any of its claims can be checked.
func readDoc(t *testing.T, rel string) string {
	t.Helper()
	src, err := os.ReadFile(path(rel))
	if err != nil {
		t.Fatalf("reading %s: %v", rel, err)
	}
	return string(src)
}
