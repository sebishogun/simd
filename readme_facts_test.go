package simd_test

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The README carries counts: how many functions are exported, how many kernels
// are generated, how many entries docs/wrong.md has, which version is current.
// Every one of them had drifted. The function count said 461 against an actual
// 457, the wrong.md count said twenty-three against an actual 70, the status
// section said v1.0.0 two releases after v1.0.0, and the coverage table's own
// rows disagreed with the ppc64le figure quoted three paragraphs below it.
//
// None of that is discoverable by reading, which is the problem: a number in
// prose looks equally true whether or not anyone has checked it since it was
// written. So the numbers are checked here, against the tree rather than
// against a copy of themselves.
func TestReadmeCountsAreCurrent(t *testing.T) {
	readme := readReadme(t)

	t.Run("exported functions", func(t *testing.T) {
		claimed := singleInt(t, readme, `(\d+) exported functions`)
		if actual := exportedFuncCount(t); claimed != actual {
			t.Errorf("README claims %d exported functions; the package has %d.\n"+
				"Update the count in the coverage section.", claimed, actual)
		}
	})

	t.Run("wrong.md entries", func(t *testing.T) {
		claimed := singleInt(t, readme, `records (\d+) things`)
		src, err := os.ReadFile("docs/wrong.md")
		if err != nil {
			t.Fatalf("reading docs/wrong.md: %v", err)
		}
		actual := len(regexp.MustCompile(`(?m)^## `).FindAllString(string(src), -1))
		if actual < 10 {
			t.Fatalf("only %d entries parsed from docs/wrong.md; the heading format "+
				"has probably changed and this check has gone vacuous", actual)
		}
		if claimed != actual {
			t.Errorf("README says docs/wrong.md records %d things; it has %d entries.",
				claimed, actual)
		}
	})

	// The coverage table lists kernels per architecture and a total in the
	// paragraph above it. They are written at different times and drifted
	// apart before; adding a row is exactly when the total is forgotten.
	t.Run("kernel total matches the table", func(t *testing.T) {
		claimed := singleInt(t, readme, `and ([\d,]+) generated kernels`)

		rows := regexp.MustCompile(`(?m)^\| [a-z0-9]+ \([a-z0-9/]+\) \| (\d+) \| (\d+) \|`)
		matches := rows.FindAllStringSubmatch(readme, -1)
		if len(matches) < 6 {
			t.Fatalf("parsed %d rows out of the coverage table, expected one per "+
				"architecture; the table format has changed and this check no "+
				"longer verifies anything", len(matches))
		}
		sum := 0
		for _, m := range matches {
			n, err := strconv.Atoi(m[1])
			if err != nil {
				t.Fatalf("kernel count %q is not a number: %v", m[1], err)
			}
			sum += n
		}
		if claimed != sum {
			t.Errorf("README states %d kernels in total; its own table sums to %d.",
				claimed, sum)
		}
	})

	// The status section names a version. It is the first thing a reader
	// checks and the last thing anyone remembers to bump, so tie it to the
	// newest CHANGELOG heading, which a release cannot skip.
	t.Run("status version matches the changelog", func(t *testing.T) {
		claimed := singleString(t, readme, `\*\*(v\d+\.\d+\.\d+)\.\*\* The API is stable`)

		src, err := os.ReadFile("CHANGELOG.md")
		if err != nil {
			t.Fatalf("reading CHANGELOG.md: %v", err)
		}
		latest := regexp.MustCompile(`(?m)^## (v\d+\.\d+\.\d+)`).FindStringSubmatch(string(src))
		if latest == nil {
			t.Fatal("no version heading found in CHANGELOG.md")
		}
		if claimed != latest[1] {
			t.Errorf("README status says %s; the newest CHANGELOG section is %s.",
				claimed, latest[1])
		}
	})
}

// Prose in the README backticks things that are not functions: register names,
// environment variables, and shorthand for a family whose members all carry a
// suffix. Everything else in backticks is a promise that the reader can call
// it, and `Diff` sat in that position for months naming an operation that only
// ever existed as DiffInto.
var readmeNonAPI = map[string]bool{
	"Into":         true, // the naming convention, not a function
	"Fast":         true, // the tier prefix
	"Gemv":         true, // prose for GemvInto
	"MatMul":       true, // prose for MatMulInto
	"Quantize":     true, // prose for the QuantizeInt8 family
	"GOEXPERIMENT": true, // environment variable
	"ENOSPC":       true, // errno
	"P":            true, // AArch64 predicate registers
	"Z":            true, // AArch64 scalable vector registers
	"R2":           true, // ppc64le TOC pointer
}

func TestReadmeNamesRealFunctions(t *testing.T) {
	readme := readReadme(t)
	exported := exportedNames(t)

	re := regexp.MustCompile("`([A-Z][A-Za-z0-9]*)`")
	matches := re.FindAllStringSubmatch(readme, -1)
	if len(matches) < 50 {
		t.Fatalf("only %d backticked names found in the README; the parse has "+
			"lost the file rather than the file having lost its API references",
			len(matches))
	}

	seen := map[string]bool{}
	var unknown []string
	for _, m := range matches {
		name := m[1]
		if seen[name] || readmeNonAPI[name] {
			continue
		}
		seen[name] = true
		if !exported[name] {
			unknown = append(unknown, name)
		}
	}
	if len(unknown) > 0 {
		t.Errorf("the README names %d identifiers that the package does not export: %s\n"+
			"Either fix the name, or add it to readmeNonAPI with a comment saying "+
			"what it is.", len(unknown), strings.Join(unknown, ", "))
	}
}

// withoutGoexperiments drops the goexperiment.* tags the toolchain adds when
// GOEXPERIMENT is set, leaving the tags an unconfigured build would have.
func withoutGoexperiments(tags []string) []string {
	out := make([]string, 0, len(tags))
	for _, tag := range tags {
		if !strings.HasPrefix(tag, "goexperiment.") {
			out = append(out, tag)
		}
	}
	return out
}

func readReadme(t *testing.T) string {
	t.Helper()
	src, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatalf("reading README.md: %v", err)
	}
	return string(src)
}

// singleInt pulls one number out of the README, failing if the pattern does
// not match exactly once — a pattern that stops matching would otherwise turn
// the assertion into a no-op.
func singleInt(t *testing.T, readme, pattern string) int {
	t.Helper()
	s := singleString(t, readme, pattern)
	n, err := strconv.Atoi(strings.ReplaceAll(s, ",", ""))
	if err != nil {
		t.Fatalf("%q is not a number: %v", s, err)
	}
	return n
}

func singleString(t *testing.T, readme, pattern string) string {
	t.Helper()
	m := regexp.MustCompile(pattern).FindAllStringSubmatch(readme, -1)
	if len(m) != 1 {
		t.Fatalf("pattern %q matched %d times in the README, want exactly 1; "+
			"the sentence it checks has been reworded and the check no longer "+
			"verifies anything", pattern, len(m))
	}
	return m[0][1]
}

// exportedFuncCount counts exported top-level functions in the package,
// excluding methods, which the README's figure has never included.
func exportedFuncCount(t *testing.T) int {
	t.Helper()
	n := 0
	forEachExportedDecl(t, func(d ast.Decl) {
		fn, ok := d.(*ast.FuncDecl)
		if ok && fn.Recv == nil && fn.Name.IsExported() {
			n++
		}
	})
	return n
}

// exportedNames collects every exported top-level identifier: functions,
// types, constants and variables.
func exportedNames(t *testing.T) map[string]bool {
	t.Helper()
	out := map[string]bool{}
	forEachExportedDecl(t, func(d ast.Decl) {
		switch decl := d.(type) {
		case *ast.FuncDecl:
			if decl.Recv == nil && decl.Name.IsExported() {
				out[decl.Name.Name] = true
			}
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if s.Name.IsExported() {
						out[s.Name.Name] = true
					}
				case *ast.ValueSpec:
					for _, id := range s.Names {
						if id.IsExported() {
							out[id.Name] = true
						}
					}
				}
			}
		}
	})
	return out
}

// forEachExportedDecl visits the declarations of the package as an ordinary
// `go get` sees it.
//
// Two things make that harder than it looks. parser.ParseDir ignores build
// constraints entirely, so it pulls in vec.go — which exists only under
// goexperiment.simd — alongside the vec_stub.go that replaces it. And
// build.Default inherits the ambient GOEXPERIMENT, so the same count comes out
// as 457 under `make test` and 461 under the goexperiment.simd lane that `make
// verify` also runs. The README documents one library, so it documents the
// build everyone gets, and the experiment is stripped from the context here
// rather than the number being allowed to depend on the lane.
func forEachExportedDecl(t *testing.T, visit func(ast.Decl)) {
	t.Helper()
	ctx := build.Default
	ctx.ToolTags = withoutGoexperiments(ctx.ToolTags)

	pkg, err := ctx.ImportDir(".", 0)
	if err != nil {
		t.Fatalf("resolving the package for the default build context: %v", err)
	}
	if len(pkg.GoFiles) == 0 {
		t.Fatal("the build context matched no Go files in the package directory")
	}

	fset := token.NewFileSet()
	for _, name := range pkg.GoFiles {
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parsing %s: %v", name, err)
		}
		for _, d := range f.Decls {
			visit(d)
		}
	}
}
