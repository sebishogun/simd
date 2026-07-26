// Package compile drives clang.
package compile

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/target"
)

// Result is one compiled object file.
type Result struct {
	// ObjPath is the object file on disk.
	ObjPath string
	// Command is the full clang invocation, recorded so the generated file
	// can state exactly how it was produced.
	Command string
	// ClangVersion identifies the compiler, for the same reason.
	ClangVersion string
}

// Clang is a configured compiler.
type Clang struct {
	// Path is the clang binary. Defaults to "clang" on PATH.
	Path string
	// TempDir is where object files are written.
	TempDir string
}

// New returns a Clang writing intermediates under dir.
func New(clangPath, dir string) (*Clang, error) {
	if clangPath == "" {
		clangPath = "clang"
	}
	if _, err := exec.LookPath(clangPath); err != nil {
		return nil, fmt.Errorf("compile: %s not found: %w", clangPath, err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("compile: %w", err)
	}
	return &Clang{Path: clangPath, TempDir: dir}, nil
}

// Version returns the clang version string.
func (c *Clang) Version() (string, error) {
	out, err := exec.Command(c.Path, "--version").Output()
	if err != nil {
		return "", fmt.Errorf("compile: clang --version: %w", err)
	}
	if line, _, ok := strings.Cut(string(out), "\n"); ok {
		return strings.TrimSpace(line), nil
	}
	return strings.TrimSpace(string(out)), nil
}

// Object compiles src for tgt and returns the object file.
func (c *Clang) Object(src string, tgt target.Target) (*Result, error) {
	ver, err := c.Version()
	if err != nil {
		return nil, err
	}
	base := strings.TrimSuffix(filepath.Base(src), filepath.Ext(src))
	obj := filepath.Join(c.TempDir, fmt.Sprintf("%s_%s_%s.o", base, tgt.Arch, tgt.Tier))

	args := append(tgt.Flags(), "-c", src, "-o", obj)
	cmd := exec.Command(c.Path, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("compile %s for %s: %w\n%s", src, tgt, err, stderr.String())
	}
	// A warning here is usually the compiler telling us it could not do
	// something we depend on, so it is surfaced rather than swallowed.
	if s := strings.TrimSpace(stderr.String()); s != "" {
		fmt.Fprintf(os.Stderr, "clang warnings for %s %s:\n%s\n", src, tgt, s)
	}
	return &Result{
		ObjPath:      obj,
		Command:      c.Path + " " + strings.Join(args, " "),
		ClangVersion: ver,
	}, nil
}

// Assembly compiles src for tgt to assembly text.
//
// The generator does not need this to emit anything — the object file carries
// the bytes — but the text is kept alongside the encodings as a comment, so a
// reviewer can read what the machine code actually is without reaching for a
// disassembler.
func (c *Clang) Assembly(src string, tgt target.Target) (string, error) {
	args := append(tgt.Flags(), "-S", src, "-o", "-")
	cmd := exec.Command(c.Path, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("compile -S %s for %s: %w\n%s", src, tgt, err, stderr.String())
	}
	return stdout.String(), nil
}
