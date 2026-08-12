package docs

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// A link in a living document is a promise about the tree, and it rots the
// same way the counts do: a file moves and every document that pointed at the
// old path goes on being wrong, silently, until a reader clicks. The fix is
// the same as for the counts — check the promise against the tree.
//
// Only the active set is scanned: the documents a reader is meant to reach
// from the README. Historical material — docs/plans, docs/research, the
// hardware reports, docs/wrong.md — is allowed to link to paths that no
// longer exist; rewriting history to satisfy a link checker would destroy its
// point.
var activeDocs = []string{
	"README.md",
	"CONTRIBUTING.md",
	"ROADMAP.md",
	"docs/README.md",
	"docs/api.md",
	"docs/platforms.md",
	"docs/tutorial.md",
	"docs/kernels.md",
	"docs/examples/README.md",
	"docs/guide/README.md",
}

func TestLocalLinksResolve(t *testing.T) {
	files := append([]string{}, activeDocs...)

	dir := path(filepath.Join("docs", "guide"))
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("reading %s: %v", dir, err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".md") && e.Name() != "README.md" {
			files = append(files, filepath.Join("docs", "guide", e.Name()))
		}
	}

	link := regexp.MustCompile(`\[([^\]]*)\]\(([^)]+)\)`)
	// Code is not prose: fenced blocks and inline spans can contain text that
	// only looks like a link, such as Go's `simd.F[T](args)` signatures.
	fence := regexp.MustCompile("(?ms)^`{3}[^\\n]*\\n.*?^`{3}[ \\t]*$")
	span := regexp.MustCompile("`[^`\\n]*`")

	checked := 0
	for _, rel := range files {
		src := readDoc(t, rel)
		src = fence.ReplaceAllString(src, "")
		src = span.ReplaceAllString(src, "")
		for _, m := range link.FindAllStringSubmatch(src, -1) {
			target := strings.TrimSpace(m[2])
			target = strings.TrimPrefix(target, "<")
			target = strings.TrimSuffix(target, ">")
			lower := strings.ToLower(target)
			if target == "" || strings.HasPrefix(target, "#") ||
				strings.HasPrefix(lower, "http:") || strings.HasPrefix(lower, "https:") ||
				strings.HasPrefix(lower, "mailto:") {
				continue
			}
			if i := strings.IndexAny(target, "#?"); i >= 0 {
				target = target[:i]
			}
			target = strings.TrimSpace(target)
			if target == "" { // the link was only an anchor or query
				continue
			}
			checked++
			abs := path(filepath.Join(filepath.Dir(rel), target))
			if _, err := os.Stat(abs); err != nil {
				t.Errorf("%s links to %q, which does not resolve from that "+
					"document: %v", rel, m[2], err)
			}
		}
	}
	if checked < 25 {
		t.Fatalf("only %d local links checked across %d documents; the parser "+
			"has lost the links rather than the documents having lost them",
			checked, len(files))
	}
}
