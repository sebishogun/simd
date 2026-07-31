package docs

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Every simd.Something named in a prose guide has to be a function that
// exists.
//
// The guides in docs/guide are fragments rather than whole programs — a
// snippet says `simd.Add(a, b)` and lets the surrounding prose say what a and
// b are, which is the right shape for documentation and the wrong shape for a
// compiler. So they cannot simply be built.
//
// What can be checked is the part that actually rots: a function gets renamed,
// and the guide goes on naming the old one. Nothing fails, the snippet is
// quietly wrong, and the first person to notice is someone who copied it.
func TestGuidesNameRealFunctions(t *testing.T) {
	dir := path(filepath.Join("docs", "guide"))
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("reading %s: %v", dir, err)
	}

	exported := exportedFunctions(t)
	if len(exported) < 100 {
		t.Fatalf("only %d exported functions found; the parse has lost the "+
			"package rather than the package having shrunk", len(exported))
	}

	ref := regexp.MustCompile(`simd\.([A-Z][A-Za-z0-9]*)`)
	checked := 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		src, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("reading %s: %v", e.Name(), err)
		}
		for _, m := range ref.FindAllStringSubmatch(string(src), -1) {
			checked++
			if !exported[m[1]] {
				t.Errorf("%s names simd.%s, which the package does not export",
					e.Name(), m[1])
			}
		}
	}
	if checked == 0 {
		t.Fatal("no simd.* references found in docs/guide; either the guides " +
			"moved or this test stopped looking at them")
	}
}

// exportedFunctions lists the package's exported top-level functions, read
// from the source rather than through reflection, because a package cannot
// enumerate its own functions at run time.
func exportedFunctions(t *testing.T) map[string]bool {
	t.Helper()
	entries, err := os.ReadDir(path("."))
	if err != nil {
		t.Fatalf("reading the package directory: %v", err)
	}
	decl := regexp.MustCompile(`(?m)^func ([A-Z][A-Za-z0-9]*)`)
	out := map[string]bool{}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(path(name))
		if err != nil {
			t.Fatalf("reading %s: %v", name, err)
		}
		for _, m := range decl.FindAllStringSubmatch(string(src), -1) {
			out[m[1]] = true
		}
	}
	return out
}
