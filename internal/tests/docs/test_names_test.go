package docs

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// A document naming a test is claiming a guarantee is enforced.
//
// It is the same promise as a link and it rots the same way: a test is
// renamed, the document goes on naming the old one, and a reader who greps
// for it finds nothing and cannot tell whether the guarantee was dropped or
// the name moved. Both are worse than no sentence.
//
// The subject is activeDocs — the set links_test.go checks, for the reason
// stated there. CHANGELOG.md, docs/wrong.md, docs/plans and docs/research are
// historical or external: a changelog entry describing what a release did is
// accurate about that release even after a rename, and docs/research quotes
// benchmark names from other people's blog posts (`BenchmarkNoopAsm` is
// mmcloughlin's, not this repository's). Gating those would force history to
// be rewritten to satisfy a checker, which is the thing docs/wrong.md exists
// to prevent.
func TestActiveDocsNameRealTests(t *testing.T) {
	declared := declaredTestFuncs(t)
	if len(declared) == 0 {
		t.Fatal("no test functions were found in the tree, so this gate measures nothing")
	}

	cited := regexp.MustCompile("`((?:Test|Fuzz|Benchmark|Example)[A-Za-z0-9_]+)`")
	checked := 0
	for _, doc := range testNameDocs() {
		b, err := os.ReadFile(filepath.Join(repoRoot, doc))
		if err != nil {
			t.Fatalf("%s: %v", doc, err)
		}
		for i, line := range strings.Split(string(b), "\n") {
			for _, m := range cited.FindAllStringSubmatch(line, -1) {
				checked++
				if !declared[m[1]] {
					t.Errorf("%s:%d names %s, which no test in this repository "+
						"declares: the guarantee on that line is prose",
						doc, i+1, m[1])
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("no test names were cited in any active document, so this gate " +
			"measures nothing -- if the citations were deliberately removed, " +
			"delete this test rather than leaving it green over nothing")
	}
	t.Logf("%d citations across %d documents, all declared", checked, len(testNameDocs()))
}

// testNameDocs is activeDocs plus the two documents that describe the SUITE
// itself, which is where most citations naturally are.
//
// Not by widening activeDocs: CLAUDE.md describes that set as what the LINK
// gate covers -- README, CONTRIBUTING, ROADMAP and the main references and
// guides -- and says the LLDs, plans and records are checked by hand. This
// gate has its own subject rather than quietly changing the meaning of that
// sentence. The LLD citations were checked by hand on 2026-08-16 and all
// resolve.
func testNameDocs() []string {
	return append(append([]string(nil), activeDocs...),
		"docs/verification.md",
		"internal/tests/README.md",
	)
}

// declaredTestFuncs is every Test/Fuzz/Benchmark/Example function in the tree.
func declaredTestFuncs(t *testing.T) map[string]bool {
	t.Helper()
	decl := regexp.MustCompile(`(?m)^func ((?:Test|Fuzz|Benchmark|Example)[A-Za-z0-9_]*)\(`)
	out := map[string]bool{}
	err := filepath.WalkDir(repoRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if n := d.Name(); n == ".git" || n == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, "_test.go") {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, m := range decl.FindAllStringSubmatch(string(b), -1) {
			out[m[1]] = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}
