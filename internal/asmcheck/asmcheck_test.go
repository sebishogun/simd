// Package asmcheck reads the committed assembly and asserts the things about
// it that no runtime test can reach.
//
// Everything here is a property of the *bytes* rather than of the answers a
// kernel gives. A kernel that violates one of these is not wrong when called;
// it is correct, returns, and leaves the machine in a state the Go runtime
// does not expect, so the failure lands in unrelated code some time later.
// That is the class of bug this package exists for, and each check below
// corresponds to one that actually shipped.
//
// The tests run on any host: they parse text, they do not execute anything,
// and they need no clang. That is deliberate — the generator's own checks
// require a cross toolchain and so only run when somebody regenerates, while
// these run in every `go test ./...`.
package asmcheck

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// texts splits an assembly file into (symbol, body) pairs, where body is every
// line up to the next TEXT.
func texts(t *testing.T, path string) map[string][]string {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	head := regexp.MustCompile(`^TEXT ·(\w+)\(SB\)`)
	out := map[string][]string{}
	name := ""
	for line := range strings.SplitSeq(string(src), "\n") {
		if m := head.FindStringSubmatch(line); m != nil {
			name = m[1]
			out[name] = nil
			continue
		}
		if name == "" {
			continue
		}
		if line = strings.TrimSpace(line); line != "" && !strings.HasPrefix(line, "//") {
			out[name] = append(out[name], line)
		}
	}
	return out
}

func files(t *testing.T, dir string) []string {
	t.Helper()
	got, err := filepath.Glob(filepath.Join("..", dir, "*.s"))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatalf("no assembly found under internal/%s; the backend is meant to be "+
			"committed, so an empty directory means the generator wrote nowhere", dir)
	}
	return got
}

// TestPPC64LERestoresR0 checks that every ppc64le kernel ends with the
// epilogue that puts r0 back, and that nothing can reach a return without
// passing through it.
//
// Go's ppc64 ABI reserves r0 by value: abi-internal.md gives it "Zero value"
// as its meaning on call, on return and in the body, and the compiler lowers a
// memory clear to a run of "MOVD R0,(Rn)" on the understanding that the
// register holds the zero it is storing. clang has no such notion — r0 is
// volatile scratch under ELFv2 — and there is no -ffixed-rN on its PowerPC
// target to tell it otherwise, so the generator rewrites each of a body's own
// returns into a branch to the end and puts the restore there.
//
// The symptom when this is missing is not a wrong result. It is the next
// allocation in Go coming back non-zero, and the process dying inside the
// allocator or the collector with an index out of range, nowhere near this
// library.
func TestPPC64LERestoresR0(t *testing.T) {
	for _, path := range files(t, "ppc64le") {
		for name, body := range texts(t, path) {
			if len(body) < 2 {
				t.Errorf("%s: %s has no body", filepath.Base(path), name)
				continue
			}
			last, prev := body[len(body)-1], body[len(body)-2]
			if last != "RET" || prev != "MOVD $0, R0" {
				t.Errorf("%s: %s ends with %q, %q; want %q, %q — every kernel must "+
					"restore r0, which Go requires to hold zero",
					filepath.Base(path), name, prev, last, "MOVD $0, R0", "RET")
			}
		}
	}
}

// TestPPC64LEHasNoBareReturn checks that no branch-to-link-register survives in
// a body, which is the other half of the same contract: a bclr the generator
// failed to rewrite is a path that leaves the kernel without running the
// epilogue, and it would leave no trace in the file's tail.
func TestPPC64LEHasNoBareReturn(t *testing.T) {
	for _, path := range files(t, "ppc64le") {
		for name, body := range texts(t, path) {
			for i, w := range words(t, body) {
				// Primary opcode 19, extended opcode 16: the whole bclr family,
				// conditional and not.
				if w>>26 == 19 && (w>>1)&0x3ff == 16 {
					t.Errorf("%s: %s has a bclr at word %d (%#08x); it would return "+
						"without restoring r0", filepath.Base(path), name, i, w)
				}
			}
		}
	}
}

// TestNoArchitectureIsEmpty guards against the failure that looks like success:
// a backend that generated nothing still builds, still passes every runtime
// test, and silently runs the portable path everywhere.
func TestNoArchitectureIsEmpty(t *testing.T) {
	for _, arch := range []string{"amd64", "arm64", "loong64", "ppc64le", "riscv64", "s390x"} {
		n := 0
		for _, path := range files(t, arch) {
			n += len(texts(t, path))
		}
		if n == 0 {
			t.Errorf("internal/%s has assembly files but not one TEXT symbol", arch)
		}
		t.Logf("%-8s %3d kernels", arch, n)
	}
}

// words decodes the WORD directives of a body into instructions. ppc64le is
// little-endian and fixed width, so a WORD is exactly one instruction and the
// value in the directive is the instruction as the manual writes it.
func words(t *testing.T, body []string) []uint32 {
	t.Helper()
	var out []uint32
	for _, line := range body {
		v, ok := strings.CutPrefix(line, "WORD $0x")
		if !ok {
			continue
		}
		n, err := strconv.ParseUint(v, 16, 32)
		if err != nil {
			t.Fatalf("undecodable directive %q: %v", line, err)
		}
		out = append(out, uint32(n))
	}
	return out
}
