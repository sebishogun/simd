package docs

import (
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// A release tag is a promise that the version shipped, and the CHANGELOG is
// where the reader finds out what shipped in it. Tags exist whose release
// notes never made the file — v1.6.0 through v1.9.1 shipped while the
// CHANGELOG jumped from v1.5.0 to v1.10.0 — and nothing noticed, because
// nothing compared the two lists. This does.
//
// The check skips when git metadata is unavailable (a source tarball, say):
// without the tags there is nothing to compare the CHANGELOG against.
func TestChangelogCoversReleaseTags(t *testing.T) {
	git, err := exec.LookPath("git")
	if err != nil {
		t.Skip("git is not available; cannot enumerate release tags")
	}
	root, err := filepath.Abs(repoRoot)
	if err != nil {
		t.Fatalf("resolving %s: %v", repoRoot, err)
	}
	out, err := exec.Command(git, "-C", root, "tag", "--list", "v*").Output()
	if err != nil {
		t.Skipf("git metadata is unavailable (%v); cannot enumerate release tags", err)
	}

	// Only stable tags bind the CHANGELOG; pre-releases are free to be
	// documented by their stable successor.
	stable := regexp.MustCompile(`^v(\d+)\.(\d+)\.(\d+)$`)
	type version struct {
		tag                 string
		major, minor, patch int
	}
	var tags []version
	for _, line := range strings.Split(string(out), "\n") {
		m := stable.FindStringSubmatch(strings.TrimSpace(line))
		if m == nil {
			continue
		}
		v := version{tag: m[0]}
		v.major, _ = strconv.Atoi(m[1])
		v.minor, _ = strconv.Atoi(m[2])
		v.patch, _ = strconv.Atoi(m[3])
		tags = append(tags, v)
	}
	if len(tags) == 0 {
		t.Fatal("git returned no stable release tags; the tag parse has broken " +
			"rather than the repository having no releases")
	}
	// `git tag --list` orders lexically, which puts v1.10.0 before v1.6.0;
	// sort by number so the report reads in release order regardless.
	sort.Slice(tags, func(i, j int) bool {
		a, b := tags[i], tags[j]
		if a.major != b.major {
			return a.major < b.major
		}
		if a.minor != b.minor {
			return a.minor < b.minor
		}
		return a.patch < b.patch
	})

	changelog := readDoc(t, "CHANGELOG.md")
	headings := map[string]bool{}
	heading := regexp.MustCompile(`(?m)^## (v\d+\.\d+\.\d+)(?:\s|$)`)
	for _, m := range heading.FindAllStringSubmatch(changelog, -1) {
		headings[m[1]] = true
	}
	if len(headings) < 10 {
		t.Fatalf("only %d version headings parsed from CHANGELOG.md; the heading "+
			"format has changed and this check has gone vacuous", len(headings))
	}

	var missing []string
	for _, v := range tags {
		if !headings[v.tag] {
			missing = append(missing, v.tag)
		}
	}
	if len(missing) > 0 {
		t.Errorf("CHANGELOG.md has no \"## <tag>\" section for %d release tags: %s\n"+
			"Every tagged release needs a section, even a one-line one — the tag "+
			"says the version exists and the CHANGELOG is where that is explained.",
			len(missing), strings.Join(missing, ", "))
	}
}
