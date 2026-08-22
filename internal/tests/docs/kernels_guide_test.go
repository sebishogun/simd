package docs

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// docs/kernels.md's worked example names things that have to exist.
//
// The guide is the answer to "the operation I need isn't in the 403", and its
// centre is one `spec.Kernel` literal with eight fields, each annotated with
// what it is for. A reader copies that literal. If a field is renamed the
// guide goes on naming the old one, and the person following it gets a
// compile error in generated code they did not write and cannot easily read
// -- which is the moment the guide exists to prevent.
//
// The link gate and the test-name gate already cover this document. Neither
// looks inside a code block.
//
// THE FIELD LIST IS READ FROM THE DOCUMENT, not written here. A list in this
// file would have to be kept in step with the guide by hand, which is the
// thing being gated, one level up.
func TestTheKernelGuidesManifestFieldsExist(t *testing.T) {
	doc := readDoc(t, filepath.Join("docs", "kernels.md"))

	// The one block that shows a manifest entry, from `spec.Kernel{` to the
	// fence that ends it.
	block := regexp.MustCompile("(?s)```go\\s*\nspec\\.Kernel\\{(.*?)\n```").FindStringSubmatch(doc)
	if block == nil {
		t.Fatal("docs/kernels.md no longer contains a ```go block starting " +
			"`spec.Kernel{`. That literal is the guide's worked example and " +
			"this gate reads it; if the example moved, repoint the gate rather " +
			"than leaving it green over nothing")
	}

	key := regexp.MustCompile(`(?m)^\s*([A-Z][A-Za-z0-9]*):`)
	var named []string
	for _, m := range key.FindAllStringSubmatch(block[1], -1) {
		named = append(named, m[1])
	}
	if len(named) < 5 {
		t.Fatalf("the worked example names %d fields (%v); it documented eight, "+
			"so the extraction has lost the block rather than the block having "+
			"shrunk", len(named), named)
	}

	fields := structFields(t, filepath.Join("tools", "simdgen", "spec", "spec.go"), "Kernel")
	if len(fields) < 5 {
		t.Fatalf("spec.Kernel parsed to %d fields; the parse has lost the "+
			"struct", len(fields))
	}
	for _, n := range named {
		if !fields[n] {
			t.Errorf("docs/kernels.md's worked example sets spec.Kernel.%s, "+
				"which the struct does not declare. Someone following the guide "+
				"writes a manifest entry that does not compile", n)
		}
	}

	// AND THE TWO HAND-WRITTEN WIRINGS THE EXAMPLE POINTS AT.
	//
	// `RefFunc` and `Field` are the two files `docs/plans` wants generated
	// precisely because they are the two easiest things to forget when adding a
	// kernel -- forgetting the ref wiring shipped past the whole normal lane
	// once. The guide names a REAL kernel for its example, so both ends of it
	// can be checked rather than described.
	lit := regexp.MustCompile(`(?m)^\s*(RefFunc|Field):\s*"([A-Za-z0-9]+)"`)
	seen := 0
	for _, m := range lit.FindAllStringSubmatch(block[1], -1) {
		seen++
		switch m[1] {
		case "RefFunc":
			if !refExports(t)[m[2]] {
				t.Errorf("the guide's example declares RefFunc %q, which "+
					"internal/ref does not export. The guard would fall back to "+
					"a function that does not exist", m[2])
			}
		case "Field":
			if !kernelOpsFields(t)[m[2]] {
				t.Errorf("the guide's example declares Field %q, which "+
					"internal/kernel does not declare on any kernel group. That "+
					"is the wiring a new kernel is most often missing", m[2])
			}
		}
	}
	if seen != 2 {
		t.Fatalf("found %d of the example's RefFunc/Field literals, want 2; "+
			"the two assertions above then covered less than they claim", seen)
	}
}

// structFields is the field names of one struct in one file, multi-name
// declarations (`Group, Field string`) included -- which is how two of the
// eight are declared, so a parser that took one name per line would report
// the guide as wrong about fields that exist.
//
// Parsed rather than imported: `tools/` is a separate module and must never
// become a consumer dependency, which is a rule in CLAUDE.md.
func structFields(t *testing.T, rel, name string) map[string]bool {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), path(rel), nil, 0)
	if err != nil {
		t.Fatalf("parsing %s: %v", rel, err)
	}
	out := map[string]bool{}
	ast.Inspect(f, func(n ast.Node) bool {
		ts, ok := n.(*ast.TypeSpec)
		if !ok || ts.Name.Name != name {
			return true
		}
		st, ok := ts.Type.(*ast.StructType)
		if !ok {
			return true
		}
		for _, fld := range st.Fields.List {
			for _, id := range fld.Names {
				out[id.Name] = true
			}
		}
		return false
	})
	return out
}

// refExports is every exported function internal/ref declares.
func refExports(t *testing.T) map[string]bool {
	t.Helper()
	return declaredIn(t, filepath.Join("internal", "ref"),
		regexp.MustCompile(`(?m)^func ([A-Z][A-Za-z0-9]*)`))
}

// kernelOpsFields is every function-typed field on any group in
// internal/kernel -- the `Field` half of a manifest entry.
func kernelOpsFields(t *testing.T) map[string]bool {
	t.Helper()
	return declaredIn(t, filepath.Join("internal", "kernel"),
		regexp.MustCompile(`(?m)^\t([A-Z][A-Za-z0-9]*)\s+func\(`))
}

func declaredIn(t *testing.T, dir string, re *regexp.Regexp) map[string]bool {
	t.Helper()
	out := map[string]bool{}
	matches, err := filepath.Glob(filepath.Join(path(dir), "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range matches {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllStringSubmatch(string(b), -1) {
			out[m[1]] = true
		}
	}
	if len(out) == 0 {
		t.Fatalf("no declarations matched in %s; the pattern has stopped "+
			"finding them and every lookup against this set would pass or fail "+
			"for the wrong reason", dir)
	}
	return out
}
