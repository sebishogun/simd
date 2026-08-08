// Package thresholds holds the generated threshold tables and the generated
// guards to agreement.
//
// The tables (kernel_thresholds_<arch>.go) and the guards (the `if len < N`
// in internal/<arch>/*_register_*.go) are generated from the same manifest,
// so they cannot disagree TODAY. What this test prevents is tomorrow: a guard
// regenerated after a manifest change while the tables are stale, or the
// reverse, leaves KernelThreshold telling callers a number the guard does not
// use -- which is exactly the silent cross-repository drift the API exists to
// end. Both artifacts are parsed from disk, for all six architectures, on
// whatever machine runs the test.
package thresholds

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

var goarches = []string{"amd64", "arm64", "riscv64", "s390x", "ppc64le", "loong64"}

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "kernel_thresholds.go")); err != nil {
		t.Fatalf("repo root not found from test directory: %v", err)
	}
	return root
}

// tableFor parses one generated kernel_thresholds_<arch>.go.
func tableFor(t *testing.T, root, arch string) map[string]int {
	t.Helper()
	src, err := os.ReadFile(filepath.Join(root, "kernel_thresholds_"+arch+".go"))
	if err != nil {
		t.Fatalf("threshold table for %s: %v", arch, err)
	}
	out := map[string]int{}
	for _, m := range regexp.MustCompile(`"(\w+)":\s*(\d+),`).FindAllStringSubmatch(string(src), -1) {
		n, _ := strconv.Atoi(m[2])
		out[m[1]] = n
	}
	if len(out) < 100 {
		t.Fatalf("only %d entries parsed from the %s table; the format has changed and this test has gone vacuous", len(out), arch)
	}
	return out
}

// guardRe matches a generated guard's head: the function name and the first
// length comparison inside it. Guards are exported now -- the dispatch
// tables name them in static composite literals -- so the head is an
// uppercase name with no suffix. Elementwise guards clamp first -- `n :=
// min(len(dst), len(a), ...)` -- so one such line is allowed before the if.
var guardRe = regexp.MustCompile(`func ([A-Z]\w+)\([^)]*\)[^{]*\{\s*\n(?:\s*n := [^\n]*\n)?\s*if [^\n{<]*< (\d+)`)

func TestGuardsMatchTables(t *testing.T) {
	root := repoRoot(t)
	totalGuards := 0
	for _, arch := range goarches {
		table := gonameThresholds[arch]
		if len(table) < 500 {
			t.Fatalf("%s: only %d go names in the generated fixture; regenerate with simdgen/thresholds", arch, len(table))
		}
		files, err := filepath.Glob(filepath.Join(root, "internal", arch, "*_register_*.go"))
		if err != nil || len(files) == 0 {
			t.Fatalf("%s: no register files found (%v)", arch, err)
		}
		for _, f := range files {
			src, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			for _, m := range guardRe.FindAllStringSubmatch(string(src), -1) {
				name, num := m[1], m[2]
				name = strings.ToLower(name[:1]) + name[1:]
				goName, want, ok := lookup(name, table)
				if !ok {
					t.Errorf("%s: guard %s matches no goName in the manifest fixture", filepath.Base(f), name)
					continue
				}
				n, _ := strconv.Atoi(num)
				if want != n {
					t.Errorf("%s: guard %s uses %d; the manifest says %s = %d on %s",
						filepath.Base(f), name, n, goName, want, arch)
				}
				totalGuards++
			}
		}
		// The public per-arch table must exist and parse; it is written by the
		// same generator run as the fixture above, so presence and size is the
		// staleness check that matters for it.
		tableFor(t, root, arch)
	}
	// The regexps going quietly stale must fail loudly: this repository has
	// thousands of guards, and a low count means the parse broke, not that
	// the guards left.
	if totalGuards < 4000 {
		t.Fatalf("only %d guards parsed across six architectures; the guard format has changed and this test has gone vacuous", totalGuards)
	}
	t.Logf("%d guards checked against the manifest", totalGuards)
}

// lookup strips the tier suffix from a guard's base name by trying ever
// shorter prefixes against the manifest's exact goNames, longest first --
// sumsqUint8VX reaches sumsqUint8 before anything shorter, so a goName that
// is a prefix of another cannot steal its guards.
func lookup(name string, table map[string]int) (string, int, bool) {
	for l := len(name); l > 0; l-- {
		if n, ok := table[name[:l]]; ok {
			return name[:l], n, true
		}
	}
	return "", 0, false
}
