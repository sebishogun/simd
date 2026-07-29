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
	// function.
	//
	// The real limit is the linker's: abi.StackNosplitBase is 800 bytes for a
	// whole nosplit call chain, and cmd/link/internal/ld/stackcheck.go rejects
	// anything over it. These kernels are leaves and make no calls, so the
	// chain is one deep and the budget is nearly all theirs. This check is not
	// that check — it exists to fail here, with the kernel's name and a clear
	// reason, rather than at link time in a consumer's build.
	//
	// 512 leaves comfortable headroom under 800 while accommodating the widest
	// transcendental, which spills around 288 bytes on the AVX2 tier. A kernel
	// wanting materially more than this is doing something a kernel should
	// not, and the message says so.
	MaxStackBytes int
	// RequireVector fails a kernel whose body contains no vector instruction
	// at all, which means LLVM did not vectorize it.
	RequireVector bool
}

// DefaultOptions are the settings used by the generator.
func DefaultOptions() Options {
	return Options{ObjdumpPath: "llvm-objdump", MaxStackBytes: 512, RequireVector: true}
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
	//
	// A kernel that spills more than the budget is unusable on this target,
	// not evidence that something is wrong: register pressure is a property
	// of the tier, and the baseline x86-64 has eight vector registers where
	// avx512 has thirty-two. So this drops the kernel and keeps the portable
	// path, like every other capability limit. Reporting it as a hard failure
	// took a whole target down over one kernel.
	if r.StackBytes > opt.MaxStackBytes {
		r.Unsupported = fmt.Sprintf(
			"spills %d bytes, over the %d-byte budget for a NOSPLIT function on this tier",
			r.StackBytes, opt.MaxStackBytes)
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

	// 5. No instruction touches a register the Go runtime owns.
	//
	// This is the backstop, and it is needed because neither mechanism that
	// should have prevented it is reliable: clang has no -ffixed for SystemZ,
	// and it accepts the global register variable in csrc/goabi.h there
	// without honouring it — measured, 86 uses of r13 survived the
	// declaration. Nine kernels in arith.c alone used Go's goroutine pointer
	// as a scratch register and did not even save it, which is not a crash at
	// the point of the clobber but corruption of whatever the runtime does
	// next.
	//
	// A save-and-restore pair is allowed through: it leaves the register
	// intact for the caller, and it is how a C prologue treats any
	// callee-saved register.
	if reg, ok := usesGoRegister(instrs, tgt); ok {
		// r2 on ppc64le is the one exemption, and a narrow one. Every kernel
		// that reads a constant there sets r2 itself in clang's ELFv2
		// global-entry prologue and then uses it only as the base of its own
		// constant loads — it never reads the caller's value. The emitter
		// repoints those loads at a pool appended to the body and replaces the
		// prologue, so the finished kernel does not depend on Go's r2 either;
		// see emit/constpool_power.go.
		//
		// The exemption is conditional on the pattern being exactly that. A
		// kernel using r2 for anything else is still rejected, which is what
		// tocPatternOnly checks.
		if !(tgt.Arch == target.PPC64LE && reg == "r2" && tocPatternOnly(instrs)) {
			r.Unsupported = fmt.Sprintf("uses %s, which the Go runtime owns", reg)
		}
	}

	// 6. Nothing is written outside the kernel's own stack frame.
	//
	// Memory above the callee's stack pointer belongs to the caller. Two ABIs
	// here have the callee write there anyway — s390x puts its register save
	// area at the caller's stack pointer, and ELFv2 has the callee save the
	// link register at 16(r1) — and under Go that is the calling function's
	// locals. The symptom is a slice header with a length past its capacity,
	// panicking somewhere else entirely, or a return to address 1.
	//
	// Targets that do this declare a SaveArea, and the kernel is emitted as a
	// framed trampoline calling the body, so the writes land in the
	// trampoline's own frame.
	//
	// Below the stack pointer is just as bad and has no such remedy. x86-64's
	// red zone is the familiar case, and -mno-red-zone is why amd64 is safe;
	// ppc64le's ELFv2 has a 288-byte "protected zone" with the same meaning
	// and no flag that turns it off — -mno-red-zone is accepted there and
	// merely reduces the count. Go writes below the stack pointer during
	// signal delivery and stack growth, so a kernel that keeps anything there
	// is corrupted by the runtime rather than the other way round. Forty-six
	// percent of ppc64le kernels do, which is why that target is only
	// partially accelerated.
	//
	// This is what makes both a decision rather than an assumption. It is
	// also the check that would have caught the first ppc64le measurement,
	// which looked at one source file and concluded there was nothing to
	// contain.
	if tgt.StackReg != "" {
		if off, ok := writesOutsideFrame(instrs, tgt); ok {
			where := "the caller's frame; the target needs a SaveArea"
			if off < 0 {
				where = "below the stack pointer, where the Go runtime writes"
			}
			r.Unsupported = fmt.Sprintf("writes to %s%+d, which is %s",
				tgt.StackReg, off, where)
		}
	}

	// 7. The function returns at all. Multiple returns are fine — the body is
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
// storeMnemonics are the store instructions of each architecture, for the
// above-the-stack-pointer check. Only the targets whose ABI makes the mistake
// possible need an entry; the rest declare no StackReg and are not checked.
var storeMnemonics = map[string]bool{
	// s390x
	"stmg": true, "stm": true, "stg": true, "st": true, "sty": true,
	"std": true, "ste": true, "vst": true, "vstm": true,
	// ppc64le shares std; add the rest of its family
	"stw": true, "stb": true, "sth": true, "stfd": true, "stfs": true,
	"stxv": true, "stxvd2x": true, "stvx": true, "stmw": true,
}

// writesOutsideFrame reports whether any instruction stores outside the
// kernel's own frame, and at what displacement from the stack pointer.
//
// A non-negative displacement is the caller's frame; a negative one is the
// red or protected zone. Both are forbidden under Go, and a SaveArea excuses
// only the first, because it moves the stack pointer down far enough that the
// positive offsets land in the kernel's own frame.
func writesOutsideFrame(instrs []Instr, tgt target.Target) (int, bool) {
	re := regexp.MustCompile(`(-?\d+)\(` + regexp.QuoteMeta(tgt.StackReg) + `\)`)
	for _, in := range instrs {
		if !storeMnemonics[in.Mnemonic] || frameAlloc[in.Mnemonic] {
			continue
		}
		for _, m := range re.FindAllStringSubmatch(in.Operands, -1) {
			n, err := strconv.Atoi(m[1])
			if err != nil {
				continue
			}
			if n < 0 {
				// Below the stack pointer, which is fatal on every target but
				// the one whose ABI reserves a zone there that Go provably
				// does not touch. See target.Target.ProtectedZone.
				if -n <= tgt.ProtectedZone {
					continue
				}
				return n, true
			}
			if tgt.SaveArea == 0 {
				return n, true
			}
		}
	}
	return 0, false
}

// frameAlloc are the store-and-update instructions that allocate a frame: they
// write at the new stack pointer rather than below the old one, so they are
// not a violation.
var frameAlloc = map[string]bool{"stdu": true, "stwu": true}

// saveRestore reports whether an instruction is a bulk register save or
// restore rather than a use. These move a range of registers to or from the
// frame and leave every one of them as it was.
func saveRestore(m string) bool {
	switch m {
	case "stmg", "lmg", "stm", "lm", // s390x
		"stp", "ldp", // arm64
		"stmw", "lmw", "std", "ld": // ppc64
		return true
	}
	return false
}

// tocPatternOnly reports whether every mention of r2 is part of the TOC
// addressing the emitter knows how to rewrite: the two global-entry
// instructions that set it, and reads that use it as a base.
//
// A write to r2 anywhere else means the kernel is doing something this
// generator has not accounted for, and it stays rejected.
func tocPatternOnly(instrs []Instr) bool {
	for i, in := range instrs {
		if !mentionsRegister(in.Operands, "r2") {
			continue
		}
		// The first two instructions are the global-entry prologue, which is
		// allowed to write r2 because the emitter replaces both.
		if i < 2 {
			continue
		}
		// Everywhere else r2 may only be read. It is read when it appears
		// anywhere but the first operand, which on PowerPC is the destination.
		ops := strings.Split(in.Operands, ",")
		if len(ops) > 0 && mentionsRegister(ops[0], "r2") {
			return false
		}
	}
	return true
}

// usesGoRegister reports whether any instruction names a register the Go
// runtime owns, and which one.
func usesGoRegister(instrs []Instr, tgt target.Target) (string, bool) {
	if len(tgt.GoOwned) == 0 {
		return "", false
	}
	for _, in := range instrs {
		if saveRestore(in.Mnemonic) {
			continue
		}
		for _, reg := range tgt.GoOwned {
			if mentionsRegister(in.Operands, reg) {
				return reg, true
			}
		}
	}
	return "", false
}

// mentionsRegister reports whether an operand list names a register, matching
// whole names only: r2 must not match r22, and x27 must not match x270.
func mentionsRegister(ops, reg string) bool {
	for i := 0; i+len(reg) <= len(ops); i++ {
		if ops[i:i+len(reg)] != reg {
			continue
		}
		if i > 0 && isRegNameByte(ops[i-1]) {
			continue
		}
		if j := i + len(reg); j < len(ops) && isRegNameByte(ops[j]) {
			continue
		}
		return true
	}
	return false
}

func isRegNameByte(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '_'
}

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
	// llvm-objdump separates the mnemonic from its operands with a tab only
	// when the mnemonic is long enough to need one; short ones get a space:
	//
	//	cmpdi\t7, 0
	//	std 24, -64(1)
	//
	// Reading the whole of the second field as the mnemonic therefore leaves
	// "std 24, -64(1)" as a mnemonic with no operands, which silently
	// disables every check that looks at operands — the stack-frame check,
	// the Go-owned-register check, and the register half of the feature gate.
	// It matched nothing on ppc64le for exactly that reason.
	var inlineOps string
	if i := strings.IndexAny(in.Mnemonic, " \t"); i >= 0 {
		inlineOps = strings.TrimSpace(in.Mnemonic[i+1:])
		in.Mnemonic = in.Mnemonic[:i]
	}
	if in.Mnemonic == "" {
		// Keep it rather than dropping it: an unparsed line is an unverified
		// instruction, and countUndecoded needs to see it.
		in.Mnemonic = "<unknown>"
	}
	if len(fields) > 2 || inlineOps != "" {
		ops := strings.TrimSpace(inlineOps + " " + strings.Join(fields[2:], " "))
		// Drop llvm-objdump's trailing "# imm = 0x..." annotation, which would
		// otherwise be mistaken for an operand.
		//
		// The cut is at "# " and not at "#", and the space is load-bearing.
		// On arm64 the hash is the immediate PREFIX -- sub sp, sp, #0xf0 --
		// so cutting at the bare hash deleted every immediate the
		// architecture has. stackAdjust then measured every arm64 frame as
		// zero, which made every ordinary spill look like a write into the
		// caller's frame, and four separate attempts to fix the frame check
		// failed against a parser that was never handing it the numbers. The
		// annotation always has a space after the hash; an immediate never
		// does.
		if i := strings.Index(ops, "# "); i >= 0 {
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
			// A local label inside a function is printed in exactly this
			// shape — "0000000000007460 <.L0 >:" — and treating it as the
			// start of a new function silently truncates the one it is
			// inside. That is not a cosmetic bug: everything after the first
			// internal label goes unchecked, including by the feature gate
			// that exists to keep an AVX-512 instruction out of an AVX2 file.
			// It cost 23 loong64 kernels, which looked like tail calls to
			// libm because their return happened to sit past a label.
			//
			// Local labels are the ones beginning with a dot; a kernel is
			// never named that way.
			if strings.HasPrefix(m[2], ".") {
				continue
			}
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
