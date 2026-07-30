// Command perfmodel estimates kernel throughput on architectures this machine
// cannot execute.
//
// # Why this exists
//
// Every performance number in this repository was measured on amd64, because
// that is the only architecture here that runs at native speed. qemu-user
// emulates instruction semantics and not timing, so no emulated lane says
// anything about whether an arm64, ppc64le, s390x or loong64 kernel is fast —
// only that it is correct. That left the central claim of a SIMD library
// unverified on five of its six targets.
//
// llvm-mca closes most of that gap without the hardware. It is LLVM's own
// machine model: the same scheduling tables the compiler uses to order
// instructions, driven forward over a block to report cycles, throughput and
// port pressure for a named CPU. It ships with clang, which this repository
// already requires.
//
// # What it does
//
//	perfmodel                       every target, the default kernel list
//	perfmodel -kernels simd_add_f32,simd_mul_f64
//	perfmodel -arch arm64
//
// For each (architecture, tier, kernel) it compiles the C to assembly, finds
// the innermost vector loop, runs llvm-mca over it, and divides by the number
// of elements the loop handles per iteration to give cycles per element. It
// then does the same with vectorization disabled, and reports the ratio.
//
// # What it is not
//
// A model, not a measurement. It assumes every operand is in L1, models one
// core with no memory system behind it, and knows nothing about frequency,
// cache misses or bandwidth — which is exactly what dominates a whole-slice
// kernel at large n. So a good number here means "the instruction stream is
// what it should be", not "this will be fast on your data".
//
// That is still the question worth answering. The failure mode this repository
// was built against is a kernel that is scalar code in a file named avx512, or
// vectorized so badly it loses to the portable path — and both show up here as
// a ratio near or below one.
//
// # Whether to believe it
//
// The model was checked against real measurements on the one architecture this
// machine can time, comparing the avx512 and avx2 tiers — the same C, the same
// kernel, two widths, which is exactly the comparison it is being trusted to
// make across architectures. Sizes are L1-resident, since that is the regime
// it models.
//
//	Add float64, n=1024      model 2.00x   measured 1.79x   12% high
//	AddInt int32, n=4096     model 1.98x   measured 1.89x    5% high
//
// Consistently optimistic and close, which is what a model with no memory
// system should be. Absolute cycles are optimistic by roughly a factor of two
// for the same reason, so read the ratios and not the cycle counts.
//
// The scalar column is NOT the library's portable Go path — it is the same C
// with vectorization switched off, which is the right baseline for "did
// vectorizing this help" and the wrong one for "is this faster than Go". Go's
// compiler does not auto-vectorize and inserts bounds checks, so it is slower
// again; the amd64 benchmarks measure that comparison properly.
//
// riscv64 is deliberately absent. RVV's vector length is a boot-time property
// of the machine, the kernels are written to be length-agnostic, and llvm-mca
// would have to be told a VLEN to model anything. A number for one VLEN would
// be a number for a machine nobody specified.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// modelTarget is one architecture and tier, with the CPU llvm-mca should
// model. The CPUs are real parts a user would plausibly deploy on rather than
// the oldest thing that runs the instructions.
type modelTarget struct {
	arch, tier string
	clangArgs  []string
	mtriple    string
	mcpu       string
	// storeWidth maps a store mnemonic to the bytes it writes. This is how
	// elements-per-iteration is derived: an elementwise kernel writes exactly
	// one element per element consumed, so the bytes stored in one trip of the
	// loop, divided by the element size, is the trip's element count.
	storeWidth map[string]int
	// hashIsComment says whether '#' starts a comment on this target.
	//
	// It is false for exactly one architecture and that one matters: AArch64
	// spells an immediate `#-16`, so treating '#' as a comment deletes every
	// offset and addressing mode in the file. This repository has already paid
	// for that once — it is the root cause in entry 62 of docs/wrong.md, where
	// the same assumption in the disassembly parser silently disabled the
	// frame checks on arm64 for months — and this tool reproduced it within an
	// hour of being written. Hence a field rather than a constant.
	hashIsComment bool
}

var targets = []modelTarget{
	{
		arch: "amd64", tier: "avx512",
		clangArgs: []string{"--target=x86_64-linux-gnu", "-march=x86-64-v4",
			"-mprefer-vector-width=512"},
		mtriple: "x86_64-linux-gnu", mcpu: "znver4",
		storeWidth: map[string]int{"vmovups": 0, "vmovupd": 0, "vmovdqu64": 0,
			"vmovdqu32": 0, "movl": 4, "movq": 8, "movb": 1, "movw": 2,
			"movss": 4, "movsd": 8, "vmovss": 4, "vmovsd": 8},
		hashIsComment: true,
	},
	{
		arch: "amd64", tier: "avx2",
		clangArgs: []string{"--target=x86_64-linux-gnu", "-march=x86-64-v3"},
		mtriple:   "x86_64-linux-gnu", mcpu: "znver3",
		storeWidth: map[string]int{"vmovups": 0, "vmovupd": 0, "vmovdqu": 0,
			"movl": 4, "movq": 8, "movb": 1, "movw": 2,
			"movss": 4, "movsd": 8, "vmovss": 4, "vmovsd": 8},
		hashIsComment: true,
	},
	{
		arch: "arm64", tier: "neon",
		clangArgs: []string{"--target=aarch64-linux-gnu", "-march=armv8-a+simd"},
		mtriple:   "aarch64-linux-gnu", mcpu: "neoverse-v2",
		storeWidth:    map[string]int{"stp": 0, "str": 0, "strb": 1, "strh": 2},
		hashIsComment: false, // AArch64 immediates are #-16
	},
	{
		arch: "ppc64le", tier: "vsx",
		clangArgs: []string{"--target=powerpc64le-linux-gnu", "-mcpu=power9"},
		mtriple:   "powerpc64le-linux-gnu", mcpu: "pwr9",
		storeWidth: map[string]int{"stxv": 16, "stxvd2x": 16, "stxvw4x": 16,
			"stw": 4, "std": 8, "stb": 1, "sth": 2, "stfs": 4, "stfd": 8,
			// The update forms, which is what a scalar loop uses: the
			// pointer increment is folded into the store.
			"stwu": 4, "stdu": 8, "stbu": 1, "sthu": 2, "stfsu": 4, "stfdu": 8,
			"stwx": 4, "stdx": 8, "stbx": 1, "sthx": 2},
		hashIsComment: true,
	},
	{
		arch: "s390x", tier: "vx",
		clangArgs: []string{"--target=s390x-linux-gnu", "-march=z14"},
		mtriple:   "s390x-linux-gnu", mcpu: "z14",
		storeWidth: map[string]int{"vst": 16, "st": 4, "stg": 8, "stc": 1,
			"sth": 2, "ste": 4, "std": 8, "sty": 4},
		hashIsComment: true,
	},
	{
		arch: "loong64", tier: "lasx",
		clangArgs: []string{"--target=loongarch64-linux-gnu", "-march=la464"},
		mtriple:   "loongarch64-linux-gnu", mcpu: "la464",
		storeWidth: map[string]int{"xvst": 32, "vst": 16, "st.w": 4, "st.d": 8,
			"st.b": 1, "st.h": 2, "fst.s": 4, "fst.d": 8},
		hashIsComment: true,
	},
}

// commonFlags mirrors the generator's, minus the parts that only matter for
// extraction. -ffp-contract=off in particular changes the instruction stream,
// so modelling without it would model a kernel this repository does not ship.
var commonFlags = []string{
	"-O3", "-ffreestanding", "-fno-builtin", "-fno-builtin-memset",
	"-fno-builtin-memcpy", "-fno-jump-tables", "-fno-math-errno",
	"-fno-asynchronous-unwind-tables", "-fno-exceptions", "-fomit-frame-pointer",
	"-fno-stack-protector", "-ffp-contract=off",
}

// kernel names the C function and the size of one element of its output, which
// is what turns bytes-per-iteration into elements-per-iteration.
type kernel struct {
	name string
	src  string
	elem int
}

var defaultKernels = []kernel{
	{"simd_add_f32", "arith.c", 4},
	{"simd_add_f64", "arith.c", 8},
	{"simd_mul_f32", "arith.c", 4},
	{"simd_abs_f32", "arith.c", 4},
	{"simd_add_i32", "arith.c", 4},
	{"simd_to_lower_ascii", "bytes.c", 1},
	{"simd_bit_xor", "bytes.c", 1},
}

var (
	reInnerHeader = regexp.MustCompile(`^(\.?L[A-Za-z0-9_.$]+):.*This Inner Loop Header: Depth=1`)
	reLabel       = regexp.MustCompile(`^(\.?L[A-Za-z0-9_.$]+):`)

	reAArch64Q = regexp.MustCompile(`\bq[0-9]+`)
	reAArch64D = regexp.MustCompile(`\bd[0-9]+`)
	reAArch64S = regexp.MustCompile(`\bs[0-9]+`)
	reAArch64X = regexp.MustCompile(`\bx[0-9]+`)
	reAArch64W = regexp.MustCompile(`\bw[0-9]+`)
)

// modellable probes whether llvm-mca has scheduling tables for this target,
// by asking it to model a single nop. A target it cannot schedule is reported
// once with the reason rather than once per kernel with the same error.
func (t modelTarget) modellable() bool {
	cmd := exec.Command("llvm-mca", "--mtriple="+t.mtriple, "--mcpu="+t.mcpu)
	cmd.Stdin = strings.NewReader("\tnop\n")
	return cmd.Run() == nil
}

func main() {
	arch := flag.String("arch", "", "only this architecture")
	names := flag.String("kernels", "", "comma-separated kernel names")
	csrc := flag.String("csrc", "../csrc", "path to the C sources")
	verbose := flag.Bool("v", false, "print each extracted loop")
	flag.Parse()

	ks := defaultKernels
	if *names != "" {
		want := map[string]bool{}
		for _, n := range strings.Split(*names, ",") {
			want[strings.TrimSpace(n)] = true
		}
		ks = nil
		for _, k := range defaultKernels {
			if want[k.name] {
				ks = append(ks, k)
			}
		}
		if len(ks) == 0 {
			fmt.Fprintln(os.Stderr, "perfmodel: no known kernel matched -kernels")
			os.Exit(2)
		}
	}

	fmt.Println("Modelled with llvm-mca. Cycles per element, lower is better.")
	fmt.Println("The ratio is against the same kernel compiled without vectorization,")
	fmt.Println("so it is the speedup the instruction stream should deliver from L1.")
	fmt.Println()

	bad := 0
	for _, t := range targets {
		if *arch != "" && t.arch != *arch {
			continue
		}
		if !t.modellable() {
			fmt.Printf("%s/%s  NOT MODELLED: llvm-mca has no scheduling model for "+
				"%s.\n    LLVM can assemble LoongArch and cannot say how long it "+
				"takes; there is no\n    itinerary or scheduling class for la464 "+
				"upstream. Nothing to work around\n    here — the tables have to "+
				"exist first.\n\n", t.arch, t.tier, t.mcpu)
			continue
		}
		fmt.Printf("%s/%s  (llvm-mca -mcpu=%s)\n", t.arch, t.tier, t.mcpu)
		for _, k := range ks {
			vec, err := modelOne(t, k, *csrc, true, *verbose)
			if err != nil {
				fmt.Printf("  %-22s %s\n", k.name, err)
				continue
			}
			scl, err := modelOne(t, k, *csrc, false, *verbose)
			if err != nil {
				fmt.Printf("  %-22s vector %.3f c/elem, scalar unavailable (%v)\n",
					k.name, vec, err)
				continue
			}
			ratio := scl / vec
			flag := ""
			if ratio < 1.2 {
				flag = "  <-- the vector form is not pulling its weight"
				bad++
			}
			fmt.Printf("  %-22s vector %6.3f   scalar %6.3f   %5.2fx%s\n",
				k.name, vec, scl, ratio, flag)
		}
		fmt.Println()
	}
	if bad > 0 {
		fmt.Printf("%d kernel/target pairs modelled below 1.2x. That is a number to "+
			"explain, not necessarily a bug: a byte kernel that is already "+
			"memory-shaped can be correct and unexciting here.\n", bad)
	}
}

// modelOne compiles one kernel, extracts its innermost loop and models it.
func modelOne(t modelTarget, k kernel, csrc string, vectorize, verbose bool) (float64, error) {
	args := append([]string{}, t.clangArgs...)
	args = append(args, commonFlags...)
	if !vectorize {
		// The baseline: same source, same flags, no vector unit. Unrolling is
		// off too, so one trip of the scalar loop is exactly one element and
		// the comparison needs no second correction.
		args = append(args, "-fno-vectorize", "-fno-slp-vectorize", "-fno-unroll-loops")
	}
	args = append(args, "-S", csrc+"/"+k.src, "-o", "-")
	out, err := exec.Command("clang", args...).Output()
	if err != nil {
		return 0, fmt.Errorf("clang: %w", err)
	}

	loop, err := innerLoop(string(out), k.name, t.hashIsComment)
	if err != nil {
		return 0, err
	}
	if verbose {
		fmt.Printf("    --- %s %s/%s vectorize=%v ---\n%s\n", k.name, t.arch, t.tier, vectorize, loop)
	}

	elems := elementsPerIteration(loop, t, k)
	if elems == 0 {
		return 0, fmt.Errorf("could not tell how many elements one trip handles")
	}
	cycles, err := mcaCycles(loop, t)
	if err != nil {
		return 0, err
	}
	return cycles / float64(elems), nil
}

// innerLoop returns the body of the innermost loop of fn, using clang's own
// "This Inner Loop Header" annotation to find it and the branch back to that
// label to end it.
func innerLoop(asm, fn string, hashComments bool) (string, error) {
	lines := strings.Split(asm, "\n")
	start := -1
	for i, l := range lines {
		if strings.HasPrefix(l, fn+":") {
			start = i
			break
		}
	}
	if start < 0 {
		return "", fmt.Errorf("not in the assembly")
	}
	var label string
	var body []string
	collecting := false
	for _, l := range lines[start:] {
		if strings.HasPrefix(l, ".Lfunc_end") {
			break
		}
		if !collecting {
			if m := reInnerHeader.FindStringSubmatch(l); m != nil {
				label, collecting = m[1], true
			}
			continue
		}
		// A new label ends the block: the loop did not branch back, so this
		// was not the shape assumed.
		if m := reLabel.FindStringSubmatch(l); m != nil && m[1] != label {
			break
		}
		t := strings.TrimSpace(l)
		if t == "" || strings.HasPrefix(t, "//") || strings.HasPrefix(t, "#") ||
			strings.HasPrefix(t, ".") || strings.HasSuffix(t, ":") {
			continue
		}
		// Strip trailing comments, which llvm-mca does not need and which
		// differ between architectures.
		if i := strings.Index(t, "//"); i >= 0 {
			t = strings.TrimSpace(t[:i])
		}
		if hashComments {
			if i := strings.Index(t, "#"); i >= 0 {
				t = strings.TrimSpace(t[:i])
			}
		}
		body = append(body, "\t"+t)
		if strings.Contains(l, label) && isBranch(t) {
			return strings.Join(body, "\n"), nil
		}
	}
	if len(body) == 0 {
		return "", fmt.Errorf("no vectorized inner loop found")
	}
	return strings.Join(body, "\n"), nil
}

func isBranch(s string) bool {
	m, _, _ := strings.Cut(s, " ")
	m, _, _ = strings.Cut(m, "\t")
	for _, p := range []string{"b", "j", "cb", "bne", "beq", "blt", "bge", "bdnz", "brc", "brctg"} {
		if m == p || strings.HasPrefix(m, p) {
			return true
		}
	}
	return false
}

// elementsPerIteration counts the bytes stored by one trip and divides by the
// element size. An elementwise kernel writes one element out per element in,
// so this is exact — and it is why the kernel list is elementwise.
func elementsPerIteration(loop string, t modelTarget, k kernel) int {
	bytes := 0
	for _, l := range strings.Split(loop, "\n") {
		f := strings.Fields(strings.TrimSpace(l))
		if len(f) == 0 {
			continue
		}
		w, ok := t.storeWidth[f[0]]
		if !ok {
			continue
		}
		if w > 0 {
			bytes += w
			continue
		}
		// Width comes from the register class rather than the mnemonic.
		bytes += widthFromOperands(strings.Join(f[1:], " "), f[0])
	}
	if bytes == 0 {
		return 0
	}
	return bytes / k.elem
}

// widthFromOperands sizes a store whose mnemonic does not say how wide it is:
// x86's vmovups and AArch64's str/stp, where the register names carry it.
func widthFromOperands(ops, mnemonic string) int {
	switch {
	case strings.Contains(ops, "%zmm"):
		return 64
	case strings.Contains(ops, "%ymm"):
		return 32
	case strings.Contains(ops, "%xmm"):
		return 16
	}
	// AArch64 register classes. w and x appear in the scalar baseline, where
	// the store goes through a general-purpose register rather than a vector
	// one; without them the baseline cannot be sized and the comparison is
	// lost for every integer and byte kernel.
	per := 0
	switch {
	case reAArch64Q.MatchString(ops):
		per = 16
	case reAArch64D.MatchString(ops):
		per = 8
	case reAArch64S.MatchString(ops):
		per = 4
	case reAArch64X.MatchString(ops):
		per = 8
	case reAArch64W.MatchString(ops):
		per = 4
	default:
		return 0
	}
	if mnemonic == "stp" {
		return per * 2 // a pair store writes two registers
	}
	return per
}

var reTotalCycles = regexp.MustCompile(`Total Cycles:\s+(\d+)`)

// mcaCycles runs llvm-mca over the loop and returns cycles per iteration.
func mcaCycles(loop string, t modelTarget) (float64, error) {
	const iters = 100
	cmd := exec.Command("llvm-mca",
		"--mtriple="+t.mtriple, "--mcpu="+t.mcpu,
		"--iterations="+strconv.Itoa(iters))
	cmd.Stdin = strings.NewReader(loop + "\n")
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		// llvm-mca's own diagnostic names the instruction it could not parse,
		// which is the only useful thing to report here.
		msg := strings.TrimSpace(stderr.String())
		if i := strings.IndexByte(msg, '\n'); i > 0 {
			msg = msg[:i]
		}
		return 0, fmt.Errorf("llvm-mca: %s", msg)
	}
	m := reTotalCycles.FindSubmatch(out)
	if m == nil {
		return 0, fmt.Errorf("llvm-mca reported no cycle count")
	}
	n, err := strconv.Atoi(string(m[1]))
	if err != nil {
		return 0, err
	}
	return float64(n) / iters, nil
}
