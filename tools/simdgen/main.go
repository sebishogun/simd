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

	var failures []string
	for _, src := range sources {
		for _, tgt := range targets {
			n, err := build(cc, src, tgt, root, outDir, opt, verbose, dryRun)
			if err != nil {
				failures = append(failures, fmt.Sprintf("%s %s: %v", src.Path, tgt, err))
				fmt.Printf("  %-14s %-8s FAILED\n", tgt.Arch, tgt.Tier)
				continue
			}
			fmt.Printf("  %-14s %-8s %d kernels\n", tgt.Arch, tgt.Tier, n)
			if len(lastSkipped) > 0 {
				fmt.Printf("  %-14s %-8s %d not supported here: %s\n", "", "",
					len(lastSkipped), strings.Join(lastSkipped, ", "))
			}
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

// build compiles, verifies and emits one source file for one target.
func build(cc *compile.Clang, src kernels.Source, tgt target.Target, root, outDir string,
	opt verify.Options, verbose, dryRun bool) (int, error) {

	res, err := cc.Object(filepath.Join(root, src.Path), tgt)
	if err != nil {
		return 0, err
	}

	obj, err := objfile.Open(res.ObjPath)
	if err != nil {
		return 0, err
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
		return 0, nil
	}

	reports, disasm, err := verify.ObjectWithDisasm(res.ObjPath, tgt, cNames, opt)
	if err != nil {
		return 0, err
	}
	var problems []string
	unusable := map[string]string{}
	for _, r := range reports {
		if verbose {
			fmt.Printf("      %s\n", r.Summary())
		}
		for _, p := range r.Problems {
			problems = append(problems, fmt.Sprintf("%s: %s", r.Func, p))
		}
		if r.Unsupported != "" {
			unusable[r.Func] = r.Unsupported
		}
	}
	if len(problems) > 0 {
		return 0, fmt.Errorf("verification failed:\n  - %s", strings.Join(problems, "\n  - "))
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
			return 0, err
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
		return 0, nil
	}

	fns := map[string]*objfile.Func{}
	for _, k := range tiered {
		fn, err := obj.Func(k.CName)
		if err != nil {
			return 0, err
		}
		fns[k.CName] = fn
	}

	prov := emit.Provenance{
		ClangVersion: res.ClangVersion,
		Command:      res.Command,
		Source:       src.Path,
	}
	forEmit := map[string][]emit.Instr{}
	for name, ins := range disasm {
		forEmit[name] = forEmitInstrs(ins)
	}
	asm, err := emit.Asm(tiered, fns, forEmit, tgt, prov)
	if err != nil {
		return 0, err
	}
	pkg := string(tgt.Arch)
	stub := emit.Stub(tiered, tgt, pkg, prov)

	if dryRun {
		return len(tiered), nil
	}

	dir := filepath.Join(outDir, pkg)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 0, err
	}
	stem := strings.TrimSuffix(filepath.Base(src.Path), filepath.Ext(src.Path))
	if err := os.WriteFile(filepath.Join(dir, stem+tgt.Suffix()+".s"), []byte(asm), 0o644); err != nil {
		return 0, err
	}
	if err := os.WriteFile(filepath.Join(dir, stem+tgt.Suffix()+".go"), []byte(stub), 0o644); err != nil {
		return 0, err
	}
	reg := emit.Backend(tiered, tgt, pkg, prov)
	if err := os.WriteFile(filepath.Join(dir, stem+"_register"+tgt.Suffix()+".go"), []byte(reg), 0o644); err != nil {
		return 0, err
	}
	return len(tiered), nil
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
