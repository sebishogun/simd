// Package verify checks compiled kernels before they are turned into
// committed assembly.
//
// These checks exist because of what the surveyed libraries actually shipped,
// not out of caution in the abstract. See
// docs/research/03-competitive-analysis.md.
//
//   - Gate versus emission. ajroetker/go-highway#68 emitted EVEX-prefixed
//     AVX-512 instructions into its AVX2 code path and crashed with SIGILL on
//     an AMD EPYC 7763. tphakala/simd#197 and #196 are the same bug in a
//     different library. AVX-512 alone has 21 feature flags; nobody keeps that
//     straight by hand, so it is checked mechanically here.
//
//   - Frame pointer. kelindar/simd#5 generated functions that clobbered BP
//     without restoring it, which silently breaks stack unwinding, tracebacks,
//     profiling and garbage-collector stack scanning. It shipped that way for
//     a long time because nothing visibly failed.
//
//   - Stack growth. A NOSPLIT function that moves the stack pointer skips the
//     check that would have grown the goroutine stack, so a deep enough call
//     runs off the end of it.
//
//   - Vectorization. If LLVM quietly failed to vectorize a loop, the result is
//     scalar code in a file named avx512 — correct, dispatched to as though it
//     were fast, and slower than the portable Go it replaced. Nothing else
//     would ever report this.
//
// Disassembly comes from llvm-objdump rather than a decoder written here.
// It ships with clang, which is already required, and it is LLVM's own
// decoder, so it knows every instruction the compiler can emit — including
// the AVX-512 and SVE encodings a third-party decoder typically does not.
package verify

import (
	"bufio"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/target"
)

// Instr is one disassembled instruction.
type Instr struct {
	Offset   uint64
	Raw      []byte
	Mnemonic string
	Operands string
}

// Report is the outcome of verifying one function.
type Report struct {
	Func string
	// MaxFeature is the highest CPU feature any instruction requires.
	MaxFeature target.Feature
	// FeatureWitness is the instruction that established MaxFeature, so a
	// failure names the actual offender rather than just the level.
	FeatureWitness Instr
	// StackBytes is the largest stack-pointer adjustment observed.
	StackBytes int
	// VectorInstrs is how many instructions operate on vector registers.
	VectorInstrs int
	// TotalInstrs is the instruction count.
	TotalInstrs int
	// WidestVector is the widest vector register class seen, in bits.
	WidestVector int
	// BodyEnd is the byte offset of the function's return instruction, which
	// is where the copied body must stop.
	//
	// The compiled function ends by returning, and that return would run
	// before any epilogue appended after it — so a kernel with a result would
	// return before its result was ever stored. Truncating here and emitting
	// Go's own RET instead is what makes a value-returning kernel work at all.
	BodyEnd uint64
	// Returns is the offset of every return instruction found.
	Returns []uint64
	// Problems are correctness failures. A kernel with any of these must not
	// ship: it would be miscompiled, mis-gated, or corrupt the runtime.
	Problems []string
	// Unsupported records that this target simply cannot express the kernel —
	// most often because LLVM would not vectorize it, or because the operation
	// needs an instruction the tier does not have. That is not a bug, it is a
	// gap, and the right response is to keep the portable implementation
	// rather than to fail the build.
	//
	// With 86 kernels across 9 targets there are hundreds of combinations, and
	// most gaps are one-line facts about an instruction set. Enumerating them
	// by hand in the manifest would be a maintenance burden that silently goes
	// stale; detecting them is both accurate and self-updating.
	Unsupported string
}

// OK reports whether the function passed every correctness check.
func (r Report) OK() bool { return len(r.Problems) == 0 }

// Usable reports whether the kernel should be generated at all.
func (r Report) Usable() bool { return r.OK() && r.Unsupported == "" }

// Options tunes the checks.
type Options struct {
	// ObjdumpPath is the llvm-objdump binary. Defaults to "llvm-objdump".
	ObjdumpPath string
	// MaxStackBytes is the largest stack adjustment tolerated in a NOSPLIT
	// function. Go's own nosplit budget is around 800 bytes for a whole call
	// chain; a kernel should be using far less than that or it is doing
	// something a kernel should not.
	MaxStackBytes int
	// RequireVector fails a kernel whose body contains no vector instruction
	// at all, which means LLVM did not vectorize it.
	RequireVector bool
}

// DefaultOptions are the settings used by the generator.
func DefaultOptions() Options {
	return Options{ObjdumpPath: "llvm-objdump", MaxStackBytes: 256, RequireVector: true}
}

// Object disassembles an object file and verifies every named function in it.
func Object(objPath string, tgt target.Target, funcs []string, opt Options) ([]Report, error) {
	r, _, err := ObjectWithDisasm(objPath, tgt, funcs, opt)
	return r, err
}

// ObjectWithDisasm is Object, and also returns the disassembly.
//
// The emitter needs it to re-spell instructions that reference a constant
// pool. Producing it once here rather than shelling out to llvm-objdump a
// second time keeps the two views of the object guaranteed identical.
func ObjectWithDisasm(objPath string, tgt target.Target, funcs []string, opt Options) ([]Report, map[string][]Instr, error) {
	if opt.ObjdumpPath == "" {
		opt.ObjdumpPath = "llvm-objdump"
	}
	byFunc, err := disassemble(objPath, tgt, opt.ObjdumpPath)
	if err != nil {
		return nil, nil, err
	}
	reports := make([]Report, 0, len(funcs))
	for _, name := range funcs {
		instrs, ok := byFunc[name]
		if !ok {
			return nil, nil, fmt.Errorf("verify: %s not found in the disassembly of %s", name, objPath)
		}
		reports = append(reports, checkFunc(name, instrs, tgt, opt))
	}
	return reports, byFunc, nil
}

func checkFunc(name string, instrs []Instr, tgt target.Target, opt Options) Report {
	r := Report{Func: name, TotalInstrs: len(instrs)}

	for _, in := range instrs {
		if f, w := featureOf(in, tgt.Arch); f > r.MaxFeature {
			r.MaxFeature, r.FeatureWitness = f, in
			_ = w
		}
		if bits := vectorWidth(in, tgt.Arch); bits > 0 {
			r.VectorInstrs++
			if bits > r.WidestVector {
				r.WidestVector = bits
			}
		}
		if n := stackAdjust(in, tgt.Arch); n > r.StackBytes {
			r.StackBytes = n
		}
		if isReturn(in, tgt.Arch) {
			r.Returns = append(r.Returns, in.Offset)
		}
	}

	// 1. Gate versus emission.
	if r.MaxFeature > tgt.MinFeature {
		r.Problems = append(r.Problems, fmt.Sprintf(
			"requires %s but the file is gated on %s — this is the SIGILL bug: "+
				"at +0x%x  %s %s",
			r.MaxFeature, tgt.MinFeature, r.FeatureWitness.Offset,
			r.FeatureWitness.Mnemonic, r.FeatureWitness.Operands))
	}

	// 2. Frame pointer, amd64 only. Elsewhere the C ABI already treats the
	// frame pointer as callee-saved with a balanced save and restore.
	if tgt.Arch == target.AMD64 {
		if why, bad := clobbersBP(instrs); bad {
			r.Problems = append(r.Problems, fmt.Sprintf(
				"clobbers BP without restoring it (%s) — breaks stack unwinding, "+
					"tracebacks, profiling and GC stack scanning. Compile this kernel "+
					"with -fno-omit-frame-pointer so the save and restore are balanced.",
				why))
		}
	}

	// 3. Stack growth.
	if r.StackBytes > opt.MaxStackBytes {
		r.Problems = append(r.Problems, fmt.Sprintf(
			"adjusts the stack pointer by %d bytes, over the %d-byte limit for a "+
				"NOSPLIT function — it would skip the stack-growth check. Reduce the "+
				"kernel's live values or split it up.",
			r.StackBytes, opt.MaxStackBytes))
	}

	// 4. Everything decoded.
	//
	// An instruction llvm-objdump prints as <unknown> is one this package
	// cannot classify, so it cannot be checked against the tier the file is
	// gated on. Skipping it quietly would leave exactly the hole the gate
	// check exists to close — and it did: s390x prints every vector
	// instruction as <unknown> unless told --mcpu=z14, which made a correctly
	// vectorized kernel look entirely scalar.
	if n := countUndecoded(instrs); n > 0 {
		r.Problems = append(r.Problems, fmt.Sprintf(
			"%d instruction(s) could not be disassembled, so they cannot be checked "+
				"against the %s gate. Add the right --mcpu or --mattr to the target's "+
				"DisasmFlags.", n, tgt.Tier))
	}

	// 5. The function returns at all. Multiple returns are fine — the body is
	// copied whole and nothing is appended after it — but a kernel that never
	// returns would run off the end of its own code into whatever the linker
	// placed next.
	if len(r.Returns) == 0 {
		// Usually a tail call: the target lacks the instruction, so clang
		// jumped to a libm symbol instead of returning. Either way the body
		// cannot be copied safely, and the portable implementation stands in.
		r.Unsupported = "no return instruction, probably a tail call to libm"
	}

	// 6. Vectorization actually happened. If it did not, this target cannot
	// usefully accelerate the kernel — dispatching to it would run scalar code
	// under a name promising otherwise — so the kernel is dropped rather than
	// failing the build.
	if opt.RequireVector && r.VectorInstrs == 0 {
		r.Unsupported = "LLVM did not vectorize it for " + tgt.Tier
	}
	return r
}

// countUndecoded counts instructions llvm-objdump could not decode.
func countUndecoded(instrs []Instr) int {
	n := 0
	for _, in := range instrs {
		if strings.HasPrefix(in.Mnemonic, "<unknown") || in.Mnemonic == "" {
			n++
		}
	}
	return n
}

// isReturn reports whether the instruction returns from the function.
func isReturn(in Instr, arch target.Arch) bool {
	switch arch {
	case target.AMD64:
		return in.Mnemonic == "retq" || in.Mnemonic == "ret"
	case target.ARM64, target.RISCV64, target.LOONG64:
		return in.Mnemonic == "ret"
	case target.PPC64LE:
		return strings.HasPrefix(in.Mnemonic, "blr")
	case target.S390X:
		return strings.HasPrefix(in.Mnemonic, "br ") || in.Mnemonic == "br"
	}
	return false
}

// isPadding reports whether the instruction is alignment filler, which
// compilers place after the return and which the symbol size can include.
func isPadding(in Instr, arch target.Arch) bool {
	m := in.Mnemonic
	return strings.HasPrefix(m, "nop") || m == "ud2" || m == "hint" ||
		strings.HasPrefix(m, "<unknown")
}

// ---------- feature classification ----------
//
// Two independent signals are used and either one is enough to condemn an
// instruction, because a check that can be fooled is worse than no check.
//
// On amd64 the register class in the operands gives it away — %zmm or a %k
// mask register means AVX-512, %ymm means AVX — and separately the EVEX
// prefix byte 0x62 identifies an AVX-512 encoding regardless of what the
// operands look like. In 64-bit mode 0x62 has no other meaning, since the
// BOUND instruction it used to be is invalid there. Measured on a real object
// file: 12 EVEX-prefixed instructions in the AVX-512 build, none in the AVX2
// or SSE2 builds.
//
// On arm64 an SVE instruction names a Z register or a P predicate, neither of
// which NEON has.

var (
	reZMM  = regexp.MustCompile(`%zmm[0-9]+`)
	reYMM  = regexp.MustCompile(`%ymm[0-9]+`)
	reXMM  = regexp.MustCompile(`%xmm[0-9]+`)
	reMask = regexp.MustCompile(`%k[0-7]\b`)

	reSVEZ = regexp.MustCompile(`\bz[0-9]+\.[bhsdq]`)
	reSVEP = regexp.MustCompile(`\bp[0-9]+(\.[bhsdq]|/[zm])`)
	reNEON = regexp.MustCompile(`\bv[0-9]+\.[0-9]*[bhsdq]`)
)

func featureOf(in Instr, arch target.Arch) (target.Feature, string) {
	switch arch {
	case target.AMD64:
		if len(in.Raw) > 0 && in.Raw[0] == 0x62 {
			return target.FeatAVX512, "EVEX prefix"
		}
		switch {
		case reZMM.MatchString(in.Operands) || reMask.MatchString(in.Operands):
			return target.FeatAVX512, "512-bit or mask register"
		case reYMM.MatchString(in.Operands):
			return target.FeatAVX2, "256-bit register"
		case strings.HasPrefix(in.Mnemonic, "v") && reXMM.MatchString(in.Operands):
			return target.FeatAVX, "VEX-encoded"
		case reXMM.MatchString(in.Operands):
			return target.FeatSSE2, "128-bit register"
		}
		return target.FeatBase, ""
	case target.ARM64:
		switch {
		case reSVEZ.MatchString(in.Operands) || reSVEP.MatchString(in.Operands):
			return target.FeatSVE2, "SVE register or predicate"
		case reNEON.MatchString(in.Operands):
			return target.FeatNEON, "NEON register"
		}
		return target.FeatBase, ""
	}
	// The remaining architectures have a single vector tier each, so any
	// vector instruction is exactly at the gate and nothing can exceed it.
	if vectorWidth(in, arch) > 0 {
		return featureForArch(arch), "vector register"
	}
	return target.FeatBase, ""
}

func featureForArch(a target.Arch) target.Feature {
	switch a {
	case target.RISCV64:
		return target.FeatRVV
	case target.S390X:
		return target.FeatVX
	case target.LOONG64:
		return target.FeatLASX
	case target.PPC64LE:
		return target.FeatVSX
	}
	return target.FeatBase
}

var (
	reVecRVV = regexp.MustCompile(`\bv[0-9]+\b`)
)

// vectorWidth returns the width in bits of the widest vector register the
// instruction touches, or 0 if it touches none.
func vectorWidth(in Instr, arch target.Arch) int {
	switch arch {
	case target.AMD64:
		switch {
		case reZMM.MatchString(in.Operands):
			return 512
		case reYMM.MatchString(in.Operands):
			return 256
		case reXMM.MatchString(in.Operands):
			return 128
		}
	case target.ARM64:
		switch {
		case reSVEZ.MatchString(in.Operands):
			// Scalable: the real width is a property of the machine, not the
			// encoding. Report the architectural minimum.
			return 128
		case reNEON.MatchString(in.Operands):
			return 128
		}
	case target.RISCV64:
		if strings.HasPrefix(in.Mnemonic, "v") && reVecRVV.MatchString(in.Operands) {
			return 128
		}
	case target.S390X, target.LOONG64, target.PPC64LE:
		if strings.HasPrefix(in.Mnemonic, "v") || strings.HasPrefix(in.Mnemonic, "xv") ||
			strings.HasPrefix(in.Mnemonic, "lxv") || strings.HasPrefix(in.Mnemonic, "stxv") {
			return 128
		}
	}
	return 0
}

// ---------- frame pointer ----------

var (
	rePushBP = regexp.MustCompile(`^push[ql]?$`)
	reBPDst  = regexp.MustCompile(`%rbp\s*$`)
)

// clobbersBP reports whether the function uses the frame pointer as a scratch
// register without saving it.
//
// The bug this exists for is kelindar/simd#5, where generated code put a value
// in BP and never restored it, silently breaking stack unwinding, tracebacks,
// profiling and garbage-collector stack scanning.
//
// The test is deliberately not "pushes equal pops". A function with two exit
// paths pushes BP once at entry and pops it once on each path, so a static
// count reads 1 push and 2 pops while every actual execution is balanced. The
// first version of this check rejected perfectly correct kernels on exactly
// that basis.
//
// What matters is whether the register is saved at all. If the function
// establishes a frame — clang does whenever -mstackrealign is in effect — then
// BP is pushed on entry and popped before every return, and any use in between
// is the frame pointer doing its job. If BP is written with no push anywhere,
// it is being used as scratch and the old value is gone.
func clobbersBP(instrs []Instr) (string, bool) {
	pushes, writes := 0, 0
	for _, in := range instrs {
		switch {
		case rePushBP.MatchString(in.Mnemonic) && strings.Contains(in.Operands, "%rbp"):
			pushes++
		case strings.HasPrefix(in.Mnemonic, "pop"):
			// Restoring it is never the problem.
		case reBPDst.MatchString(in.Operands) &&
			!strings.HasPrefix(in.Mnemonic, "cmp") && !strings.HasPrefix(in.Mnemonic, "test"):
			// AT&T puts the destination last, so %rbp at the end is a write.
			writes++
		}
	}
	if writes > 0 && pushes == 0 {
		return fmt.Sprintf("%d write(s) and no push to save it", writes), true
	}
	return "", false
}

// ---------- stack ----------

var (
	reSubRSP  = regexp.MustCompile(`^\$(?:0x([0-9a-f]+)|([0-9]+)),\s*%rsp$`)
	reSubSPAA = regexp.MustCompile(`^sp,\s*sp,\s*#([0-9]+)$`)
)

// stackAdjust returns how far the instruction moves the stack pointer down.
func stackAdjust(in Instr, arch target.Arch) int {
	switch arch {
	case target.AMD64:
		if in.Mnemonic != "subq" && in.Mnemonic != "subl" {
			return 0
		}
		m := reSubRSP.FindStringSubmatch(strings.TrimSpace(in.Operands))
		if m == nil {
			return 0
		}
		if m[1] != "" {
			n, _ := strconv.ParseInt(m[1], 16, 64)
			return int(n)
		}
		n, _ := strconv.Atoi(m[2])
		return n
	case target.ARM64:
		if in.Mnemonic != "sub" {
			return 0
		}
		if m := reSubSPAA.FindStringSubmatch(strings.TrimSpace(in.Operands)); m != nil {
			n, _ := strconv.Atoi(m[1])
			return n
		}
	}
	return 0
}

// ---------- disassembly ----------

var reFuncHeader = regexp.MustCompile(`^([0-9a-f]+)\s+<([^>]+)>:$`)

// parseInstr reads one llvm-objdump instruction line.
//
// The format is not uniform across architectures, which a single regular
// expression handles badly:
//
//	x86     "       0: 48 85 c9        \ttestq\t%rcx, %rcx"    byte pairs
//	arm64   "       0: f100047f        \tcmp\tx3, #0x1"        one 32-bit word
//
// Splitting on the offset and then on tabs copes with both, and with the
// fixed-width encodings of riscv64, loong64 and ppc64le, which print like
// arm64. Getting this wrong is not harmless: an unparsed line is an
// unverified instruction, and the checks in this package are the only thing
// standing between a mis-gated kernel and a SIGILL in production.
func parseInstr(line string) (Instr, bool) {
	head, rest, ok := strings.Cut(line, ":")
	if !ok {
		return Instr{}, false
	}
	off, err := strconv.ParseUint(strings.TrimSpace(head), 16, 64)
	if err != nil {
		return Instr{}, false
	}
	// After the offset comes the raw encoding, then a tab, then the mnemonic,
	// then a tab, then the operands.
	fields := strings.Split(rest, "\t")
	if len(fields) < 2 {
		return Instr{}, false
	}
	in := Instr{Offset: off, Raw: parseRaw(strings.TrimSpace(fields[0]))}
	in.Mnemonic = strings.TrimSpace(fields[1])
	if in.Mnemonic == "" {
		// Keep it rather than dropping it: an unparsed line is an unverified
		// instruction, and countUndecoded needs to see it.
		in.Mnemonic = "<unknown>"
	}
	if len(fields) > 2 {
		ops := strings.Join(fields[2:], " ")
		// Drop llvm-objdump's trailing "# imm = 0x..." annotation, which would
		// otherwise be mistaken for an operand.
		if i := strings.Index(ops, "#"); i >= 0 {
			ops = ops[:i]
		}
		in.Operands = strings.TrimSpace(ops)
	}
	return in, true
}

// parseRaw decodes the encoding field into bytes in memory order.
func parseRaw(field string) []byte {
	if field == "" {
		return nil
	}
	if strings.Contains(field, " ") {
		// Byte pairs, already in memory order.
		var out []byte
		for _, tok := range strings.Fields(field) {
			v, err := strconv.ParseUint(tok, 16, 8)
			if err != nil {
				return out
			}
			out = append(out, byte(v))
		}
		return out
	}
	// A single fixed-width word, printed as a value. The architectures that
	// print this way are all little-endian here, so the low byte comes first
	// in memory.
	v, err := strconv.ParseUint(field, 16, 64)
	if err != nil {
		return nil
	}
	n := len(field) / 2
	out := make([]byte, n)
	for i := range out {
		out[i] = byte(v >> (8 * i))
	}
	return out
}

// disassemble runs llvm-objdump and groups instructions by function.
func disassemble(objPath string, tgt target.Target, objdump string) (map[string][]Instr, error) {
	args := append([]string{"-d", objPath}, tgt.DisasmFlags...)
	out, err := exec.Command(objdump, args...).Output()
	if err != nil {
		return nil, fmt.Errorf("verify: %s -d %s: %w", objdump, objPath, err)
	}

	byFunc := map[string][]Instr{}
	cur := ""
	var base uint64
	sc := bufio.NewScanner(strings.NewReader(string(out)))
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if m := reFuncHeader.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
			// llvm-objdump prints addresses relative to the section, while the
			// object file's relocations are relative to the symbol. Everything
			// downstream matches the two up by offset, so normalise here to the
			// symbol-relative form rather than leaving two coordinate systems
			// in play.
			base, _ = strconv.ParseUint(m[1], 16, 64)
			cur = m[2]
			byFunc[cur] = nil
			continue
		}
		if cur == "" {
			continue
		}
		if in, ok := parseInstr(line); ok {
			in.Offset -= base
			byFunc[cur] = append(byFunc[cur], in)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("verify: reading disassembly: %w", err)
	}
	return byFunc, nil
}

// Summary renders a one-line description of a report, for the generator's log.
func (r Report) Summary() string {
	return fmt.Sprintf("%-28s %3d instrs, %3d vector (%d-bit), feature=%s, stack=%dB",
		r.Func, r.TotalInstrs, r.VectorInstrs, r.WidestVector, r.MaxFeature, r.StackBytes)
}
