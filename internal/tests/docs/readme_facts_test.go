package docs

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The documentation carries counts: how many functions are exported, how many
// kernels are generated, how many entries docs/wrong.md has, which version is
// current. Every one of them had drifted. The function count said 461 against
// an actual 457, the wrong.md count said twenty-three against an actual 70,
// the status section said v1.0.0 two releases after v1.0.0, and the coverage
// table's own rows disagreed with the ppc64le figure quoted three paragraphs
// below it.
//
// None of that is discoverable by reading, which is the problem: a number in
// prose looks equally true whether or not anyone has checked it since it was
// written. So the numbers are checked here, against the tree rather than
// against a copy of themselves. The function count, wrong.md count and status
// version live in the README; the kernel counts live in docs/platforms.md.
func TestDocumentedCountsAreCurrent(t *testing.T) {
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
		src, err := os.ReadFile(path("docs/wrong.md"))
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

	// The coverage table in docs/platforms.md lists kernels per architecture
	// and a total in the paragraph near it. They are written at different
	// times and drifted apart before; adding a row is exactly when the total
	// is forgotten.
	t.Run("kernel total matches the table", func(t *testing.T) {
		platforms := readDoc(t, "docs/platforms.md")
		claimed := singleInt(t, platforms, `([\d,]+) generated kernels`)

		sum := 0
		for _, row := range coverageRows(t, platforms) {
			sum += row.kernels
		}
		if claimed != sum {
			t.Errorf("docs/platforms.md states %d kernels in total; its own table "+
				"sums to %d.", claimed, sum)
		}
	})

	// The table's kernel counts are written by hand; the assembly they
	// describe is not. The generator marks every logical kernel in the .s
	// files, so every documented count has to equal the number of markers
	// across the architecture's sources.
	t.Run("documented kernels match the sources", func(t *testing.T) {
		platforms := readDoc(t, "docs/platforms.md")

		for _, row := range coverageRows(t, platforms) {
			actual := generatedKernels(t, row.arch)
			if row.kernels != actual {
				t.Errorf("docs/platforms.md documents %d kernels for %s; "+
					"internal/%s/*.s declares %d.\nRegenerate the "+
					"platform numbers from the sources rather than copying them "+
					"forward.", row.kernels, row.arch, row.arch, actual)
			}
		}
	})

	// The status section names a version. It is the first thing a reader
	// checks and the last thing anyone remembers to bump, so tie it to the
	// newest CHANGELOG heading, which a release cannot skip.
	t.Run("status version matches the changelog", func(t *testing.T) {
		claimed := singleString(t, readme, `\*\*(v\d+\.\d+\.\d+)\.\*\* The API is stable`)

		src, err := os.ReadFile(path("CHANGELOG.md"))
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

// archKernels is one row of the coverage table: an architecture and the
// kernel count the documentation claims for it.
type archKernels struct {
	arch    string
	kernels int
}

// coverageRows parses the per-architecture kernel table out of
// docs/platforms.md.
func coverageRows(t *testing.T, platforms string) []archKernels {
	t.Helper()
	rows := regexp.MustCompile(`(?m)^\| ([a-z0-9]+) \([a-z0-9/]+\) \| (\d+) \| (?:\d+) \|`)
	matches := rows.FindAllStringSubmatch(platforms, -1)
	if len(matches) < 6 {
		t.Fatalf("parsed %d rows out of the coverage table, expected one per "+
			"architecture; the table format has changed and this check no "+
			"longer verifies anything", len(matches))
	}
	out := make([]archKernels, 0, len(matches))
	for _, m := range matches {
		n, err := strconv.Atoi(m[2])
		if err != nil {
			t.Fatalf("kernel count %q is not a number: %v", m[2], err)
		}
		out = append(out, archKernels{arch: m[1], kernels: n})
	}
	return out
}

// generatedKernels counts the logical kernels across internal/<arch>/*.s.
// The generator precedes every exported assembly declaration with a
// `// func name(args)` marker, one per logical kernel, which is what the
// documentation's kernel counts mean. Counting TEXT declarations instead
// would double on s390x, where every logical kernel pairs its callable entry
// with a *Body TEXT helper.
//
// The files are read at the top level of each architecture directory: the
// generator lays them out flat and os.ReadDir does not descend. If a future
// layout change moves the assembly into subdirectories, the missing-files
// failure below is what reports it.
func generatedKernels(t *testing.T, arch string) int {
	t.Helper()
	dir := path(filepath.Join("internal", arch))
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("reading %s: %v", dir, err)
	}
	marker := regexp.MustCompile(`(?m)^// func `)
	files, n := 0, 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".s") {
			continue
		}
		files++
		src, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("reading %s: %v", filepath.Join(dir, e.Name()), err)
		}
		n += len(marker.FindAllString(string(src), -1))
	}
	if files == 0 {
		t.Fatalf("no .s files found in %s; the architecture layout has changed "+
			"and this check no longer verifies anything", dir)
	}
	return n
}

// CONTRIBUTING asks strangers for runs on the tiers that have never executed on
// real silicon, and names them. That list is the verification table in
// docs/platforms.md restated in prose, so it drifts the moment the table moves
// — and it had: correcting the arm64 NEON row left CONTRIBUTING still saying
// five tiers and still omitting NEON from the list, which would have told
// someone with the one machine most likely to be reading that their run was
// not wanted.
//
// A report landing is exactly when both change, so they are tied together here.
func TestContributingMatchesVerificationTable(t *testing.T) {
	platforms := readDoc(t, "docs/platforms.md")

	// Rows of the platform table whose correctness column is not real hardware.
	// Each row may list several tiers for one architecture.
	rows := regexp.MustCompile(`(?m)^\| ([a-z0-9]+) \| ([a-z0-9, ]+) \| ([^|]+) \|`)
	matches := rows.FindAllStringSubmatch(platforms, -1)
	if len(matches) < 6 {
		t.Fatalf("parsed %d rows out of the platform table, expected one per "+
			"architecture; the table format has changed and this check no longer "+
			"verifies anything", len(matches))
	}

	unverified := map[string]bool{}
	for _, m := range matches {
		arch, tiers, correctness := m[1], m[2], strings.TrimSpace(m[3])
		if arch == "architecture" { // the header row matches the same shape
			continue
		}
		if strings.Contains(correctness, "real hardware") {
			continue
		}
		for _, tier := range strings.Split(tiers, ",") {
			unverified[arch+" "+strings.TrimSpace(tier)] = true
		}
	}
	if len(unverified) == 0 {
		t.Fatal("no unverified tiers parsed from the platform table; either every " +
			"row now claims real hardware, in which case CONTRIBUTING should stop " +
			"asking, or the parse has broken")
	}

	src, err := os.ReadFile(path("CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("reading CONTRIBUTING.md: %v", err)
	}
	contributing := string(src)

	// The count opens a sentence in CONTRIBUTING, so it is capitalised there.
	claimed := strings.ToLower(singleString(t, contributing,
		`(\w+) tiers have never run on real silicon`))
	if want := numberWord(len(unverified)); claimed != want {
		t.Errorf("CONTRIBUTING says %q tiers have never run on real silicon; the "+
			"docs/platforms.md table lists %d.", claimed, len(unverified))
	}

	for tier := range unverified {
		if !strings.Contains(contributing, "**"+tier+"**") {
			t.Errorf("docs/platforms.md lists %q as unverified, but CONTRIBUTING "+
				"does not name it among the tiers it asks for runs on.", tier)
		}
	}

	// testdata/hardware/README.md states the same fact a third time, for the
	// person who arrives at the directory rather than at the contributing
	// guide. It was stale for the same reason and by the same amount.
	dir, err := os.ReadFile(path("testdata/hardware/README.md"))
	if err != nil {
		t.Fatalf("reading testdata/hardware/README.md: %v", err)
	}
	stated := strings.ToLower(singleString(t, string(dir),
		`for\s+(\w+) of the seven tiers, nobody has done one`))
	if want := numberWord(len(unverified)); stated != want {
		t.Errorf("testdata/hardware/README.md says %q of the seven tiers have no "+
			"report; the docs/platforms.md table lists %d.", stated, len(unverified))
	}
}

func numberWord(n int) string {
	words := []string{"zero", "one", "two", "three", "four", "five", "six",
		"seven", "eight", "nine", "ten"}
	if n < len(words) {
		return words[n]
	}
	return strconv.Itoa(n)
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
	return readDoc(t, "README.md")
}

// singleInt pulls one number out of a document, failing if the pattern does
// not match exactly once — a pattern that stops matching would otherwise turn
// the assertion into a no-op.
func singleInt(t *testing.T, doc, pattern string) int {
	t.Helper()
	s := singleString(t, doc, pattern)
	n, err := strconv.Atoi(strings.ReplaceAll(s, ",", ""))
	if err != nil {
		t.Fatalf("%q is not a number: %v", s, err)
	}
	return n
}

func singleString(t *testing.T, doc, pattern string) string {
	t.Helper()
	m := regexp.MustCompile(pattern).FindAllStringSubmatch(doc, -1)
	if len(m) != 1 {
		t.Fatalf("pattern %q matched %d times, want exactly 1; the sentence it "+
			"checks has been reworded and the check no longer verifies anything",
			pattern, len(m))
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

	pkg, err := ctx.ImportDir(path("."), 0)
	if err != nil {
		t.Fatalf("resolving the package for the default build context: %v", err)
	}
	if len(pkg.GoFiles) == 0 {
		t.Fatal("the build context matched no Go files in the package directory")
	}

	fset := token.NewFileSet()
	for _, name := range pkg.GoFiles {
		// GoFiles are bare names relative to the directory that was imported.
		f, err := parser.ParseFile(fset, path(name), nil, 0)
		if err != nil {
			t.Fatalf("parsing %s: %v", name, err)
		}
		for _, d := range f.Decls {
			visit(d)
		}
	}
}
