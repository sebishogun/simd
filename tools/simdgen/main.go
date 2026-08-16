// Command simdgen compiles the C kernels and emits Plan 9 assembly plus the
// matching Go declarations.
//
//	go run ./tools/simdgen -out internal
//	go run ./tools/simdgen -arch arm64 -tier sve2 -v
//
// # What it does
//
// For each (architecture, instruction-set tier) it compiles the C source with
// clang, checks the object file against every rule in package verify, lifts
// the function bodies out as raw instruction encodings, and writes a Plan 9
// assembly file with an ABI prologue plus a Go file declaring the functions.
//
// The output is committed. Consumers of this library run `go get` and need no
// C toolchain; clang is a contributor's dependency, which is why this whole
// tree lives in its own Go module.
//
// # Why raw encodings
//
// Go's assembler only parses Plan 9 syntax, and its arm64 dialect knows 66
// vector mnemonics — no floating-point vector arithmetic at all, and not one
// SVE instruction. Emitting WORD directives sidesteps the mnemonic table
// entirely, which is the only way SVE2 can reach a Go binary without cgo.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/astcheck"
	"github.com/sebishogun/simd/tools/simdgen/compile"
	"github.com/sebishogun/simd/tools/simdgen/emit"
	"github.com/sebishogun/simd/tools/simdgen/kernels"
	"github.com/sebishogun/simd/tools/simdgen/objfile"
	"github.com/sebishogun/simd/tools/simdgen/spec"
	"github.com/sebishogun/simd/tools/simdgen/target"
	"github.com/sebishogun/simd/tools/simdgen/verify"
)

func main() {
	var (
		// The generator lives in its own Go module under tools/, so it is run
		// as `cd tools && go run ./simdgen`. Paths are resolved against the
		// repository root, which is that module's parent.
		root     = flag.String("root", "..", "repository root")
		outDir   = flag.String("out", "internal", "generated packages, relative to -root")
		tmpDir   = flag.String("tmp", "tools/build/tmp", "intermediate object files, relative to -root")
		archFlag = flag.String("arch", "", "only this GOARCH (default: all)")
		tierFlag = flag.String("tier", "", "only this tier (default: all)")
		clangBin = flag.String("clang", "clang", "clang binary")
		objdump  = flag.String("objdump", "llvm-objdump", "llvm-objdump binary")
		verbose  = flag.Bool("v", false, "report every kernel verified")
		dryRun   = flag.Bool("n", false, "verify and report, but write nothing")
	)
	// -only restricts generation to one C source, which is how a
	// cross-architecture failure gets bisected down to the kernels that cause
	// it. Emulated test runs are slow enough that guessing is more expensive
	// than measuring.
	onlySrc := flag.String("only", "", "generate only this csrc file (e.g. csrc/arith.c)")
	flag.Parse()
	sources = kernels.All
	if *onlySrc != "" {
		var keep []kernels.Source
		for _, s := range kernels.All {
			if s.Path == *onlySrc {
				keep = append(keep, s)
			}
		}
		if len(keep) == 0 {
			fmt.Fprintf(os.Stderr, "simdgen: no source named %q\n", *onlySrc)
			os.Exit(1)
		}
		sources = keep
	}

	abs := func(p string) string {
		if filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(*root, p)
	}
	if err := run(abs(*outDir), abs(*tmpDir), *root, *archFlag, *tierFlag,
		*clangBin, *objdump, *verbose, *dryRun); err != nil {
		fmt.Fprintf(os.Stderr, "simdgen: %v\n", err)
		os.Exit(1)
	}
}

// forEmitInstrs narrows the disassembly to what the emitter needs: instruction
// boundaries, to locate the opcode byte of a constant-pool reference, and the
// mnemonic, to know whether that reference needs an alignment patch.
func forEmitInstrs(ins []verify.Instr) []emit.Instr {
	out := make([]emit.Instr, len(ins))
	for i, in := range ins {
		out[i] = emit.Instr{Offset: in.Offset, Mnemonic: in.Mnemonic}
	}
	return out
}

// lastSkipped carries the kernels dropped by the most recent build, so the
// caller can report them. The generator is single-threaded.
var lastSkipped []string

// sources is the manifest actually generated, which -only can narrow to one
// file for bisecting.
var sources = kernels.All

func run(outDir, tmpDir, root, archFlag, tierFlag, clangBin, objdumpBin string, verbose, dryRun bool) error {
	// Validate the manifest before touching a compiler, so a typo in a
	// declaration is an immediate, precise error.
	for _, src := range sources {
		for _, k := range src.Kernels {
			if err := k.Validate(); err != nil {
				return err
			}
		}
	}

	targets := selectTargets(archFlag, tierFlag)
	if len(targets) == 0 {
		return fmt.Errorf("no targets match arch=%q tier=%q", archFlag, tierFlag)
	}

	cc, err := compile.New(clangBin, tmpDir)
	if err != nil {
		return err
	}
	clangVer, err := cc.Version()
	if err != nil {
		return err
	}
	fmt.Printf("simdgen: %s\n", clangVer)

	// Confirm the C source still says what the manifest claims it says,
	// before anything is compiled for a specific target. A signature that has
	// drifted produces a prologue loading arguments from the wrong registers,
	// which no compiler or assembler would object to.
	for _, src := range sources {
		sigs, err := astcheck.Parse(clangBin, filepath.Join(root, src.Path), nil)
		if err != nil {
			return err
		}
		if errs := astcheck.Check(src.Kernels, sigs); len(errs) > 0 {
			var b strings.Builder
			fmt.Fprintf(&b, "%s disagrees with the manifest:\n", src.Path)
			for _, e := range errs {
				fmt.Fprintf(&b, "  - %v\n", e)
			}
			return fmt.Errorf("%s", b.String())
		}
	}
	fmt.Printf("  manifest matches the C source\n")

	opt := verify.DefaultOptions()
	opt.ObjdumpPath = objdumpBin
	opt.ScalarOK = map[string]bool{}
	for _, src := range sources {
		for _, k := range src.Kernels {
			if k.AllowScalar {
				opt.ScalarOK[k.CName] = true
			}
		}
	}

	var failures []string
	// present[arch][tier] holds the CNames emitted for that pair, and
	// regFuncs[arch][tier] the per-file registration functions, both feeding
	// the dispatch-table and Sets emission after the build loop.
	present := map[string]map[string]map[string]bool{}
	regFuncs := map[string]map[string][]string{}
	for _, src := range sources {
		stem := strings.TrimSuffix(filepath.Base(src.Path), filepath.Ext(src.Path))
		for _, tgt := range targets {
			emitted, err := build(cc, src, tgt, root, outDir, opt, verbose, dryRun)
			if err != nil {
				failures = append(failures, fmt.Sprintf("%s %s: %v", src.Path, tgt, err))
				fmt.Printf("  %-14s %-8s FAILED\n", tgt.Arch, tgt.Tier)
				continue
			}
			fmt.Printf("  %-14s %-8s %d kernels\n", tgt.Arch, tgt.Tier, len(emitted))
			if len(lastSkipped) > 0 {
				fmt.Printf("  %-14s %-8s %d not supported here: %s\n", "", "",
					len(lastSkipped), strings.Join(lastSkipped, ", "))
			}
			arch, tier := string(tgt.Arch), tgt.Tier
			if present[arch] == nil {
				present[arch] = map[string]map[string]bool{}
				regFuncs[arch] = map[string][]string{}
			}
			if present[arch][tier] == nil {
				present[arch][tier] = map[string]bool{}
			}
			for _, k := range emitted {
				present[arch][tier][k.CName] = true
			}
			if len(emitted) > 0 {
				regFuncs[arch][tier] = append(regFuncs[arch][tier],
					"register"+strings.ToUpper(stem[:1])+stem[1:]+strings.ToUpper(tier))
			}
		}
	}
	if !dryRun {
		if err := emitDispatch(outDir, targets, present, regFuncs); err != nil {
			return err
		}
	}
	// The inventory is a property of the manifest rather than of any target,
	// so it is written once, after every target has had its chance to fail.
	// It is what lets the main module assert that a declared kernel is wired
	// into both the dispatch table and the reference set; see emit.Inventory.
	if !dryRun {
		inv := emit.Inventory(allKernels(), emit.Provenance{Command: "go run ./tools/simdgen"})
		path := filepath.Join(outDir, "backend", "inventory.go")
		if err := os.WriteFile(path, []byte(inv), 0o644); err != nil {
			return err
		}
	}

	if len(failures) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "%d target(s) failed:\n", len(failures))
		for _, f := range failures {
			fmt.Fprintf(&b, "\n%s\n", f)
		}
		return fmt.Errorf("%s", b.String())
	}
	return nil
}

// build compiles, verifies and emits one source file for one target,
// returning the kernels that survived for it.
func build(cc *compile.Clang, src kernels.Source, tgt target.Target, root, outDir string,
	opt verify.Options, verbose, dryRun bool) ([]spec.Kernel, error) {

	res, err := cc.Object(filepath.Join(root, src.Path), tgt, src.ExtraFlags...)
	if err != nil {
		return nil, err
	}

	obj, err := objfile.Open(res.ObjPath)
	if err != nil {
		return nil, err
	}
	defer obj.Close()

	// Any undefined symbol is fatal: Plan 9 assembly has no procedure linkage
	// table, so a call out of the object can never be resolved. This is what
	// -ffreestanding and -fno-builtin are guarding against, and checking it
	// here means a regression in those flags surfaces immediately.
	// Undefined symbols are checked per extracted function rather than for the
	// object as a whole, below: a kernel this target cannot express may well
	// have lowered to a libm call, and that is only fatal if the kernel is one
	// being generated.

	// Retarget the kernel names for this tier. Two tiers of one architecture
	// are compiled into the same package, so they cannot share a symbol.
	var tiered []spec.Kernel
	var cNames []string
	var skipped []string
	for _, k := range src.Kernels {
		if k.Skips(string(tgt.Arch), tgt.Tier) {
			skipped = append(skipped, k.GoName)
			continue
		}
		k.GoName = k.GoName + tierSuffix(tgt.Tier)
		tiered = append(tiered, k)
		cNames = append(cNames, k.CName)
	}

	if len(tiered) == 0 {
		return nil, nil
	}

	reports, disasm, err := verify.ObjectWithDisasm(res.ObjPath, tgt, cNames, opt)
	if err != nil {
		return nil, err
	}
	var problems []string
	unusable := map[string]string{}
	for _, r := range reports {
		if verbose {
			fmt.Printf("      %s\n", r.Summary())
		}
		// A kernel that does arithmetic and none of it in lanes is printed
		// whether or not the run is verbose, because it is exactly what
		// RequireVector exists to catch and exactly what it cannot see: on
		// amd64 every scalar float instruction lives in %xmm, so the "does any
		// instruction touch a vector register" test passes for scalar code.
		// Eight kernels ship this way today -- convolve, correlate, movavg and
		// polyeval in both precisions -- and docs/wrong.md entry 76 has the
		// disassembly. Not fatal and not dropped: falling back to internal/ref
		// instead is a behaviour change that wants a benchmark first.
		if r.ScalarOnly() {
			fmt.Printf("      SCALAR-ONLY %s: %d scalar arithmetic instructions, "+
				"no packed ones\n", r.Func, r.ScalarArith)
		}
		for _, p := range r.Problems {
			problems = append(problems, fmt.Sprintf("%s: %s", r.Func, p))
		}
		if r.Unsupported != "" {
			unusable[r.Func] = r.Unsupported
		}
	}
	if len(problems) > 0 {
		return nil, fmt.Errorf("verification failed:\n  - %s", strings.Join(problems, "\n  - "))
	}

	// Drop kernels this target cannot express, and any that reach a symbol the
	// object does not define — a libm call that survived because the target
	// lacks the instruction. The backend keeps the portable implementation for
	// those, so there is never a hole, only a slower path.
	kept := tiered[:0]
	for _, k := range tiered {
		if why, bad := unusable[k.CName]; bad {
			skipped = append(skipped, k.CName+" ("+why+")")
			continue
		}
		fn, err := obj.Func(k.CName)
		if err != nil {
			return nil, err
		}
		if undef := obj.UndefinedRefs(fn); len(undef) > 0 {
			skipped = append(skipped, fmt.Sprintf("%s (calls %v)", k.CName, undef))
			continue
		}
		insns := forEmitInstrs(disasm[k.CName])
		if ok, why := emit.CanLift(fn, insns, tgt); !ok {
			skipped = append(skipped, fmt.Sprintf("%s (%s)", k.CName, why))
			continue
		}
		// A signature needing more argument registers than this target has is
		// a capability limit, so the kernel is dropped and the portable path
		// kept. Asking the emitter is the only way to find out, because the
		// count depends on how the C arguments map onto the registers.
		if _, err := emit.LayoutPrologue(k, tgt); errors.Is(err, emit.ErrTooManyArgs) {
			skipped = append(skipped, fmt.Sprintf("%s (needs more argument registers than %s has)",
				k.CName, tgt.Arch))
			continue
		}
		kept = append(kept, k)
	}
	tiered = kept
	lastSkipped = skipped
	if len(tiered) == 0 {
		return nil, nil
	}

	fns := map[string]*objfile.Func{}
	for _, k := range tiered {
		fn, err := obj.Func(k.CName)
		if err != nil {
			return nil, err
		}
		fns[k.CName] = fn
	}

	stem := strings.TrimSuffix(filepath.Base(src.Path), filepath.Ext(src.Path))
	prov := emit.Provenance{
		ClangVersion: res.ClangVersion,
		Command:      res.Command,
		Source:       src.Path,
		Stem:         stem,
	}
	forEmit := map[string][]emit.Instr{}
	for name, ins := range disasm {
		forEmit[name] = forEmitInstrs(ins)
	}
	asm, err := emit.Asm(tiered, fns, forEmit, tgt, prov)
	if err != nil {
		return nil, err
	}
	pkg := string(tgt.Arch)
	stub := emit.Stub(tiered, tgt, pkg, prov)

	if dryRun {
		return tiered, nil
	}

	dir := filepath.Join(outDir, pkg)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(dir, stem+tgt.Suffix()+".s"), []byte(asm), 0o644); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(dir, stem+tgt.Suffix()+".go"), []byte(stub), 0o644); err != nil {
		return nil, err
	}
	reg := emit.Backend(tiered, tgt, pkg, prov)
	if err := os.WriteFile(filepath.Join(dir, stem+"_register"+tgt.Suffix()+".go"), []byte(reg), 0o644); err != nil {
		return nil, err
	}
	return tiered, nil
}

func selectTargets(archFlag, tierFlag string) []target.Target {
	var out []target.Target
	for _, t := range target.All {
		if archFlag != "" && string(t.Arch) != archFlag {
			continue
		}
		if tierFlag != "" && t.Tier != tierFlag {
			continue
		}
		out = append(out, t)
	}
	return out
}

// tierSuffix turns a tier name into a Go identifier suffix: sse2 becomes SSE2,
// sve2 becomes SVE2, avx512 becomes AVX512.
func tierSuffix(tier string) string { return strings.ToUpper(tier) }

// allKernels flattens the manifest. It reads `sources` rather than
// kernels.All so that a -source filter narrows the inventory the same way it
// narrows the build, which keeps a partial regeneration from silently
// dropping declarations.
func allKernels() []spec.Kernel {
	var out []spec.Kernel
	for _, src := range sources {
		out = append(out, src.Kernels...)
	}
	return out
}

// emitDispatch writes the per-operation dispatch layer: one tables file in
// the root package per architecture built this run, the reference-only
// fallback, and the per-arch Sets aggregator the tests use. Narrowed runs
// (-arch, -only) skip it: a dispatch file emitted from a partial manifest
// would silently drop operations.
func emitDispatch(outDir string, targets []target.Target,
	present map[string]map[string]map[string]bool,
	regFuncs map[string]map[string][]string) error {

	if len(sources) != len(kernels.All) {
		fmt.Printf("  dispatch tables not regenerated: -only narrows the manifest\n")
		return nil
	}
	all := allKernels()
	rootDir := filepath.Dir(outDir)
	prov := emit.Provenance{Command: "go run ./tools/simdgen"}

	// Tier order per arch comes from target.All, which is the runtime's
	// order too.
	tierOrder := map[string][]string{}
	var archOrder []string
	for _, t := range target.All {
		a := string(t.Arch)
		if len(tierOrder[a]) == 0 {
			archOrder = append(archOrder, a)
		}
		tierOrder[a] = append(tierOrder[a], t.Tier)
	}

	built := map[string]bool{}
	for _, t := range targets {
		built[string(t.Arch)] = true
	}

	for _, arch := range archOrder {
		if !built[arch] {
			continue
		}
		root := emit.RootDispatch(arch, tierOrder[arch], present[arch], all, prov)
		if err := os.WriteFile(filepath.Join(rootDir, "dispatch_tables_"+arch+".go"),
			[]byte(root), 0o644); err != nil {
			return err
		}
		sets := emit.ArchSets(arch, tierOrder[arch], regFuncs[arch], prov)
		if err := os.WriteFile(filepath.Join(outDir, arch, "sets_gen_"+arch+".go"),
			[]byte(sets), 0o644); err != nil {
			return err
		}
		tst := emit.RootDispatchTest(arch, tierOrder[arch], all, prov)
		if err := os.WriteFile(filepath.Join(rootDir, "dispatch_tables_"+arch+"_test.go"),
			[]byte(tst), 0o644); err != nil {
			return err
		}
	}

	// The fallback is a property of the whole target list, not of one run's
	// narrowing, so it is only written on a full run.
	if len(built) == len(archOrder) {
		fb := emit.RootDispatchFallback(archOrder, all, prov)
		if err := os.WriteFile(filepath.Join(rootDir, "dispatch_tables_fallback.go"),
			[]byte(fb), 0o644); err != nil {
			return err
		}
		fbt := emit.RootDispatchFallbackTest(archOrder, all, prov)
		if err := os.WriteFile(filepath.Join(rootDir, "dispatch_tables_fallback_test.go"),
			[]byte(fbt), 0o644); err != nil {
			return err
		}
	}
	return nil
}
