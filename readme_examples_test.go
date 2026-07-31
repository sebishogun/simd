package simd_test

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// The README's "which function do I want" table says of its entries:
//
//	Every one of these has a runnable example in example_test.go, checked on
//	every build.
//
// It was not true. 100 of the 109 entries had no example at all, and the claim
// had been in the file long enough that nobody was going to check it by hand.
// A promise about documentation rots exactly like a comment about code, and
// for the same reason: nothing fails when it stops being true.
//
// So this fails instead. Add a row to the table and you have to write the
// example; rename an operation and the table has to follow.
func TestReadmeTableHasExamples(t *testing.T) {
	readme, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatalf("reading README.md: %v", err)
	}

	named := readmeTableFunctions(string(readme))
	if len(named) < 50 {
		// The table is the point of the test. If the parse stops finding it —
		// because the heading moved, say — every assertion below becomes
		// vacuous and the test would go on passing while checking nothing.
		t.Fatalf("only %d operations parsed out of the README table; the parse "+
			"has probably lost track of the table rather than the table having "+
			"shrunk", len(named))
	}

	have := exampleNames(t)
	var missing []string
	for _, fn := range named {
		if !have[fn] {
			missing = append(missing, fn)
		}
	}
	if len(missing) > 0 {
		t.Errorf("the README table names %d operations with no runnable example: %s\n"+
			"Either write Example%s, or take the row out of the table — the claim "+
			"under it says every entry has one.",
			len(missing), strings.Join(missing, ", "), missing[0])
	}
}

// readmeTableFunctions returns the operations named in backticks in the
// "which function do I want" table, skipping the `Into` in its prose row.
func readmeTableFunctions(readme string) []string {
	start := strings.Index(readme, "| I want to…")
	if start < 0 {
		return nil
	}
	end := strings.Index(readme[start:], "\n\n")
	if end < 0 {
		end = len(readme) - start
	}
	table := readme[start : start+end]

	re := regexp.MustCompile("`([A-Z][A-Za-z0-9]*)`")
	seen := map[string]bool{}
	var out []string
	for _, m := range re.FindAllStringSubmatch(table, -1) {
		name := m[1]
		// "the same name with `Into`" is prose about the convention, not an
		// operation anyone can call.
		if name == "Into" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	return out
}

// exampleNames collects the operation each Example function documents, taking
// ExampleFoo and ExampleFoo_bar both to be about Foo.
func exampleNames(t *testing.T) map[string]bool {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("reading the package directory: %v", err)
	}
	re := regexp.MustCompile(`(?m)^func Example([A-Za-z0-9]*)`)
	have := map[string]bool{}
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		src, err := os.ReadFile(e.Name())
		if err != nil {
			t.Fatalf("reading %s: %v", e.Name(), err)
		}
		for _, m := range re.FindAllStringSubmatch(string(src), -1) {
			if m[1] != "" {
				have[m[1]] = true
			}
		}
	}
	return have
}
