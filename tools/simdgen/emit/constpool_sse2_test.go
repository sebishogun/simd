package emit

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The check entry 61 in docs/wrong.md said had to exist before a single kernel
// changed.
//
// Re-spelling a pool read as a Plan 9 mnemonic is safe only because the
// replacement is exactly as long as the encoding it replaces. If it were not,
// every PC-relative branch spanning it would move — and the symptom, recorded
// in the header of constpool.go from the first attempt at this, is not a crash
// but a remainder loop that is silently never entered.
//
// So this assembles every entry in respellable twice, once with Go's assembler
// and once with clang, and compares the bytes. Not the displacement, which is
// a relocation in one and a placeholder in the other, but everything else:
// prefixes, opcode, ModRM, immediate, and above all the length.
//
// Both register ranges are covered. xmm8-xmm15 need a REX prefix, so an
// assembler that emitted it in a different position — or not at all — would
// produce a different length for exactly half the register file, which is the
// kind of thing that shows up in one kernel out of forty.
func TestRespelledEncodingsMatchClang(t *testing.T) {
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Skip("clang is not installed; it is required to generate kernels at all")
	}
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Skip("objdump is not installed")
	}

	names := make([]string, 0, len(respellable))
	for m := range respellable {
		names = append(names, m)
	}
	sort.Strings(names)

	// Two registers per mnemonic: one below xmm8 and one above, so the REX
	// case is covered for every entry rather than for a sample.
	regs := []int{3, 11}

	dir := t.TempDir()
	got, err := assembleWithGo(t, dir, names, regs)
	if err != nil {
		t.Fatalf("assembling with Go: %v", err)
	}
	want, err := assembleWithClang(t, clang, dir, names, regs)
	if err != nil {
		t.Fatalf("assembling with clang: %v", err)
	}

	// A disassembler that returned nothing would make every comparison below
	// vacuous, which is the failure mode this test would otherwise hide.
	if n := len(names) * len(regs); len(got) != n {
		t.Fatalf("Go emitted %d instructions, expected %d (%d mnemonics x %d registers)",
			len(got), n, len(names), len(regs))
	}
	if len(got) != len(want) {
		t.Fatalf("Go emitted %d instructions, clang emitted %d", len(got), len(want))
	}
	for i := range got {
		g, w := got[i], want[i]
		which := fmt.Sprintf("%s -> X%d", names[i/len(regs)], regs[i%len(regs)])
		if len(g) != len(w) {
			t.Errorf("%s: Go emitted %d bytes, clang %d — a re-spelling of a "+
				"different length moves every branch after it\n  go:    % x\n  clang: % x",
				which, len(g), len(w), g, w)
			continue
		}
		// The four displacement bytes differ by construction: Go leaves a
		// relocation, clang a zero placeholder. Everything else must match.
		if !sameIgnoringDisp(g, w) {
			t.Errorf("%s: encodings differ outside the displacement\n  go:    % x\n  clang: % x",
				which, g, w)
		}
	}
}

// sameIgnoringDisp compares two encodings of the same instruction, treating the
// four-byte displacement as a wildcard. It is located from the end: the trailing
// immediate, if any, is one byte, and the displacement is the four before it.
func sameIgnoringDisp(a, b []byte) bool {
	if len(a) != len(b) || len(a) < 5 {
		return false
	}
	// Find the displacement by scanning for the ModRM byte that selects the
	// RIP-relative form; it is immediately before the four displacement bytes.
	for i := range a {
		inDisp := false
		for start := 0; start+4 <= len(a); start++ {
			if start > 0 && a[start-1]&0xc7 == 0x05 && (start+4 == len(a) || start+5 == len(a)) {
				if i >= start && i < start+4 {
					inDisp = true
				}
			}
		}
		if inDisp {
			continue
		}
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func assembleWithGo(t *testing.T, dir string, names []string, regs []int) ([][]byte, error) {
	t.Helper()
	mod := filepath.Join(dir, "goasm")
	if err := os.MkdirAll(filepath.Join(mod, "cmd"), 0o755); err != nil {
		return nil, err
	}
	write := func(name, body string) error {
		return os.WriteFile(filepath.Join(mod, name), []byte(body), 0o644)
	}
	if err := write("go.mod", "module goasm\n\ngo 1.26\n"); err != nil {
		return nil, err
	}
	if err := write("probe.go", "package goasm\n\nfunc Probe()\n"); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(mod, "cmd", "main.go"),
		[]byte("package main\n\nimport \"goasm\"\n\nfunc main() { goasm.Probe() }\n"), 0o644); err != nil {
		return nil, err
	}

	var b strings.Builder
	b.WriteString("#include \"textflag.h\"\n\nGLOBL pool<>(SB), RODATA|NOPTR, $256\n")
	for i := range 32 {
		fmt.Fprintf(&b, "DATA pool<>+0x%03x(SB)/8, $0x0102030405060708\n", i*8)
	}
	b.WriteString("\nTEXT ·Probe(SB), NOSPLIT|NOFRAME, $0-0\n")
	for _, m := range names {
		f := respellable[m]
		for _, r := range regs {
			if f.imm >= 0 {
				fmt.Fprintf(&b, "\t%s pool<>+0x20(SB), X%d, $%d\n", f.plan9, r, f.imm)
			} else {
				fmt.Fprintf(&b, "\t%s pool<>+0x20(SB), X%d\n", f.plan9, r)
			}
		}
	}
	b.WriteString("\tRET\n")
	if err := write("probe_amd64.s", b.String()); err != nil {
		return nil, err
	}

	bin := filepath.Join(dir, "goprobe")
	cmd := exec.Command("go", "build", "-o", bin, "./cmd")
	cmd.Dir = mod
	cmd.Env = append(os.Environ(), "GOARCH=amd64", "GOOS=linux")
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("go build: %v\n%s", err, out)
	}
	return disassemble(bin, "goasm.Probe")
}

func assembleWithClang(t *testing.T, clang, dir string, names []string, regs []int) ([][]byte, error) {
	t.Helper()
	var b strings.Builder
	b.WriteString("\t.text\n\t.globl probe\nprobe:\n")
	for _, m := range names {
		f := respellable[m]
		for _, r := range regs {
			// clang spells the predicate in the mnemonic, which is where the
			// table's imm came from in the first place.
			fmt.Fprintf(&b, "\t%s pool(%%rip), %%xmm%d\n", m, r)
			_ = f
		}
	}
	b.WriteString("\tretq\n\t.section .rodata\n\t.p2align 4\npool:\t.zero 256\n")
	src := filepath.Join(dir, "clang.s")
	if err := os.WriteFile(src, []byte(b.String()), 0o644); err != nil {
		return nil, err
	}
	obj := filepath.Join(dir, "clang.o")
	if out, err := exec.Command(clang, "-c", src, "-o", obj).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("clang: %v\n%s", err, out)
	}
	return disassemble(obj, "")
}

var reBytes = regexp.MustCompile(`^\s+[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)`)

// disassemble returns the encoding of each instruction in a symbol, in order,
// stopping at the return.
//
// objdump wraps a long encoding onto a continuation line with no mnemonic, so
// bytes are accumulated until a line that has one.
func disassemble(path, sym string) ([][]byte, error) {
	out, err := exec.Command("objdump", "-d", path).Output()
	if err != nil {
		return nil, err
	}
	var (
		instrs  [][]byte
		cur     []byte
		started = sym == ""
	)
	flush := func() {
		if len(cur) > 0 {
			instrs = append(instrs, cur)
			cur = nil
		}
	}
	for _, line := range strings.Split(string(out), "\n") {
		if sym != "" && strings.Contains(line, "<"+sym) && strings.HasSuffix(strings.TrimSpace(line), ":") {
			started = true
			continue
		}
		if !started {
			continue
		}
		m := reBytes.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		hasMnemonic := strings.Contains(strings.TrimSuffix(line, "\n"), "\t") &&
			len(strings.Split(strings.TrimRight(line, "\n"), "\t")) > 2
		var bs []byte
		for _, f := range strings.Fields(m[1]) {
			v, err := strconv.ParseUint(f, 16, 8)
			if err != nil {
				return nil, err
			}
			bs = append(bs, byte(v))
		}
		if hasMnemonic {
			flush()
			cur = bs
			if strings.Contains(line, "\tret") {
				flush()
				return instrs[:len(instrs)-1], nil
			}
		} else {
			cur = append(cur, bs...)
		}
	}
	flush()
	return instrs, nil
}
