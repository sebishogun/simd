package emit

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/target"
)

// Re-spelling relocated instructions.
//
// Almost every instruction is emitted as raw bytes, which is safe because a
// function's internal branches are PC-relative and the whole body moves as a
// unit. An instruction carrying a relocation is the exception: it refers to
// something outside the function — in practice a floating-point constant pool
// — through a displacement that was only going to be correct at the address
// the linker chose. Copying those bytes would leave it pointing at whatever
// now sits at that offset.
//
// So the constant is lifted into a Go DATA symbol and the one instruction that
// reads it is written out as a real Plan 9 mnemonic referring to that symbol,
// letting Go's own linker compute the displacement. Everything around it stays
// raw bytes.
//
// This is the only part of the generator that has to understand x86 encoding,
// and it deliberately understands as little as possible: a small table of load
// forms, and a refusal with a precise diagnostic for anything else. A
// generator that guesses at an instruction it does not know would produce code
// that assembles, runs, and is wrong.

// respelled is one instruction rewritten to reference a Go symbol.
type respelled struct {
	Text    string // the Plan 9 instruction
	Comment string // the original, for the reader
}

var reATTReg = regexp.MustCompile(`%([xyz])mm([0-9]+)`)

// amd64LoadForms maps an AT&T mnemonic that loads from memory to the Plan 9
// mnemonic to emit in its place.
//
// The aligned moves map deliberately to their unaligned counterparts. Go's
// linker does not promise 32- or 64-byte alignment for a DATA symbol, and
// VMOVAPD on a misaligned address faults. VMOVUPD accepts any alignment and
// computes exactly the same result, so the substitution costs nothing on any
// CPU made this decade and removes a crash that would only appear once the
// linker happened to lay the symbol out badly.
var amd64LoadForms = map[string]string{
	"vmovapd":   "VMOVUPD",
	"vmovaps":   "VMOVUPS",
	"vmovdqa":   "VMOVDQU",
	"vmovdqa32": "VMOVDQU32",
	"vmovdqa64": "VMOVDQU64",
	"vmovupd":   "VMOVUPD",
	"vmovups":   "VMOVUPS",
	"vmovdqu":   "VMOVDQU",

	// Scalar loads. A broadcast of a constant often lowers to a scalar load
	// followed by a shuffle, so these turn up wherever a kernel multiplies by
	// a literal.
	"vmovss": "VMOVSS",
	"vmovsd": "VMOVSD",

	// Broadcast-from-memory forms. LLVM reaches for these when a kernel needs
	// the same constant in every lane, which rounding does for its 0.5.
	"vmovddup":     "VMOVDDUP",
	"vbroadcastss": "VBROADCASTSS",
	"vbroadcastsd": "VBROADCASTSD",
	"vpbroadcastd": "VPBROADCASTD",
	"vpbroadcastq": "VPBROADCASTQ",

	"movapd": "MOVUPD",
	"movaps": "MOVUPS",
	"movupd": "MOVUPD",
	"movups": "MOVUPS",
	"movdqa": "MOVOU",
	"movdqu": "MOVOU",
	"movsd":  "MOVSD",
	"movss":  "MOVSS",
}

// respellAMD64 rewrites a RIP-relative load so it reads a Go symbol instead.
//
// The instruction must have exactly two operands, a RIP-relative memory
// reference and a register, which is what a constant-pool load looks like.
// Anything else — an arithmetic instruction with a memory operand, say — is
// refused rather than guessed at.
func respellAMD64(mnemonic, operands, sym string, off uint64) (respelled, error) {
	plan9, ok := amd64LoadForms[mnemonic]
	if !ok {
		return respelled{}, fmt.Errorf(
			"no Plan 9 spelling for %q; it reads a constant pool and cannot be "+
				"emitted as raw bytes. Add it to amd64LoadForms if it is a plain load, "+
				"or restructure the kernel so the constant is materialised in a register",
			mnemonic)
	}
	parts := splitOperands(operands)
	if len(parts) != 2 {
		return respelled{}, fmt.Errorf(
			"%s has %d operands (%q); only a two-operand load from a constant pool "+
				"is handled", mnemonic, len(parts), operands)
	}
	if !strings.Contains(parts[0], "(%rip)") {
		return respelled{}, fmt.Errorf(
			"%s: the relocated operand is %q, not a RIP-relative memory reference",
			mnemonic, parts[0])
	}
	dst, err := plan9Register(parts[1])
	if err != nil {
		return respelled{}, fmt.Errorf("%s: %w", mnemonic, err)
	}
	// AT&T and Plan 9 both write source before destination, so the order is
	// already right.
	ref := fmt.Sprintf("%s<>(SB)", sym)
	if off != 0 {
		ref = fmt.Sprintf("%s<>+%d(SB)", sym, off)
	}
	return respelled{
		Text:    fmt.Sprintf("%s %s, %s", plan9, ref, dst),
		Comment: fmt.Sprintf("%s %s", mnemonic, operands),
	}, nil
}

// plan9Register converts an AT&T register name to Plan 9's spelling.
func plan9Register(op string) (string, error) {
	op = strings.TrimSpace(op)
	if m := reATTReg.FindStringSubmatch(op); m != nil {
		return strings.ToUpper(m[1]) + m[2], nil
	}
	return "", fmt.Errorf("cannot spell register operand %q in Plan 9 syntax", op)
}

// splitOperands splits an AT&T operand list on commas that are not inside
// parentheses, so that "(%rax,%rbx,4), %ymm0" yields two operands rather than
// three.
func splitOperands(s string) []string {
	var (
		out   []string
		depth int
		cur   strings.Builder
	)
	for _, r := range s {
		switch r {
		case '(':
			depth++
		case ')':
			depth--
		case ',':
			if depth == 0 {
				out = append(out, strings.TrimSpace(cur.String()))
				cur.Reset()
				continue
			}
		}
		cur.WriteRune(r)
	}
	if t := strings.TrimSpace(cur.String()); t != "" {
		out = append(out, t)
	}
	return out
}

// respell dispatches to the architecture's rewriter.
func respell(arch target.Arch, mnemonic, operands, sym string, off uint64) (respelled, error) {
	switch arch {
	case target.AMD64:
		return respellAMD64(mnemonic, operands, sym, off)
	}
	return respelled{}, fmt.Errorf(
		"lifting constant pools is not implemented for %s. On this architecture a "+
			"constant address is built from a HI20/LO12 instruction pair, so both "+
			"halves have to be rewritten together", arch)
}
