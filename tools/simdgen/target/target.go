// Package target describes each (architecture, instruction-set tier) the
// generator builds for: the clang invocation, the C calling convention, and
// the rules the emitted assembly has to respect.
//
// The flags here are load-bearing, not incidental. Several were established by
// measurement and each one prevents a specific, silent failure; the comments
// say which. See docs/research/02-codegen-pipelines.md.
package target

import "fmt"

// Arch is a Go GOARCH value.
type Arch string

const (
	AMD64   Arch = "amd64"
	ARM64   Arch = "arm64"
	RISCV64 Arch = "riscv64"
	S390X   Arch = "s390x"
	LOONG64 Arch = "loong64"
	PPC64LE Arch = "ppc64le"
)

// Target is one architecture and instruction-set tier.
type Target struct {
	Arch Arch
	// Tier matches the name used by internal/cpu, so a generated file's build
	// tag and the runtime check that guards it cannot drift apart.
	Tier string

	// Triple is the clang target triple.
	Triple string
	// MArch are the architecture-selection flags.
	MArch []string

	// InstrWidth is the fixed instruction width in bytes, or 0 for a
	// variable-width encoding.
	InstrWidth int

	// Directives are the data directives this architecture's assembler
	// accepts, widest first. They are not the same everywhere and assuming
	// they are does not fail gracefully: Go's s390x assembler has no QUAD and
	// its ppc64le assembler has no BYTE, so emitting either is a hard error
	// from the assembler rather than a wrong result.
	Directives []Directive

	// BigEndian reports the byte order. It decides how instruction bytes are
	// packed into a directive: WORD $0x11223344 places 11 22 33 44 in memory
	// on a big-endian target and 44 33 22 11 on a little-endian one, so
	// packing the wrong way round silently produces a body of garbage
	// instructions. s390x is the one big-endian target here.
	BigEndian bool

	// IntArgs are the C ABI integer and pointer argument registers, in order,
	// spelled the way Go's assembler spells them.
	IntArgs []string
	// FloatArgs are the C ABI floating-point argument registers, in order.
	FloatArgs []string
	// IntResult and FloatResult are where the C ABI returns a value.
	IntResult, FloatResult string

	// Mov* are the Plan 9 mnemonics that move a value of each width between
	// the argument frame and a register.
	//
	// These genuinely differ per architecture and the differences are traps.
	// On riscv64 MOVD is a *float* move and the 64-bit integer move is plain
	// MOV; on loong64 the 64-bit move is MOVV; on arm64 the float moves are
	// FMOVS and FMOVD while riscv64 spells them MOVF and MOVD. Using the wrong
	// one is not a subtle bug — the assembler rejects it outright — but only
	// if the table is right, so it lives here next to the rest of the ABI.
	MovPtr, MovInt32, MovByte string
	MovFloat32, MovFloat64    string

	// AddrOf computes the address of an argument-frame slot into a register.
	// amd64 spells this LEAQ; the RISC-style architectures use their ordinary
	// move with a $ prefix on the operand.
	AddrOf string

	// Reserved are registers the Go runtime owns, which clang must be told
	// not to allocate, spelled the way -ffixed wants them. Empty where clang
	// has no such flag for the target — s390x, loongarch and ppc64 have none
	// — in which case csrc/goabi.h declares a global register variable
	// instead, which reserves it for the whole translation unit.
	Reserved []string

	// SaveArea is how many bytes of Go frame the kernel must reserve for a
	// register save area the C ABI expects its *caller* to have provided.
	//
	// This is the one place the two conventions disagree about memory rather
	// than registers, and it is not survivable. The s390x ELF ABI puts a
	// 160-byte save area at the caller's stack pointer and the callee writes
	// its registers there: every reduction kernel starts with
	// "stmg %r14, %r15, 112(%r15)". Go provides no such area, so those eight
	// bytes land in the middle of the calling Go function's frame. The symptom
	// is a slice header with a length larger than its capacity, panicking
	// somewhere else entirely.
	//
	// Declaring a Go frame of at least that size moves the stack pointer down
	// far enough that the writes land in the kernel's own frame. Zero means
	// the callee allocates everything it needs, which is the case on arm64,
	// riscv64, loong64 and amd64 — the last because -mno-red-zone already
	// forbids writing below the stack pointer.
	SaveArea int

	// GoOwned are the same registers spelled the way the disassembler prints
	// them, for the backstop check in package verify.
	//
	// Neither mechanism above is guaranteed: clang accepts the global register
	// variable on s390x and then allocates the register anyway. So the last
	// word is a check on the generated code, because a kernel that uses Go's
	// goroutine pointer for a loop counter does not fail where it is written.
	// It corrupts whatever the runtime touches next, and the panic surfaces in
	// unrelated Go code with a nonsense slice header.
	GoOwned []string

	// DisasmFlags are passed to llvm-objdump so it can decode this target's
	// instructions.
	//
	// Without them the disassembler falls back to a baseline CPU and prints
	// <unknown> for anything newer — which on s390x meant every vector
	// instruction, so the verifier concluded a perfectly good kernel was not
	// vectorized at all. Worse, an instruction that cannot be decoded cannot
	// be checked against the tier it is gated on, which is precisely the hole
	// the gate check exists to close.
	DisasmFlags []string

	// MinFeature is the CPU feature every instruction in this tier is allowed
	// to require. The verifier rejects anything above it — the check that
	// mechanically prevents the SIGILL class of bug that go-highway and
	// tphakala both shipped.
	MinFeature Feature
}

// Directive is a Plan 9 data directive and the number of bytes it emits.
type Directive struct {
	Name  string
	Width int
}

var (
	// dirWord is every fixed-width RISC target here: four bytes per
	// instruction and nothing else needed.
	dirWord = []Directive{{"WORD", 4}}
	// dirAMD64 covers a variable-width encoding down to single bytes.
	dirAMD64 = []Directive{{"QUAD", 8}, {"LONG", 4}, {"BYTE", 1}}
	// dirS390X has no QUAD or LONG, and instructions are 2, 4 or 6 bytes, so
	// the byte directive is genuinely needed.
	dirS390X = []Directive{{"WORD", 4}, {"BYTE", 1}}
)

// Feature is a CPU capability level, ordered so that a higher value implies
// everything below it within the same architecture.
type Feature int

const (
	FeatBase Feature = iota // the architecture baseline

	// amd64
	FeatSSE2
	FeatAVX
	FeatAVX2
	FeatAVX512

	// arm64
	FeatNEON
	FeatSVE
	FeatSVE2

	// everything else, one level each for now
	FeatRVV
	FeatVX
	FeatLSX
	FeatLASX
	FeatVSX
)

var featureNames = map[Feature]string{
	FeatBase: "base", FeatSSE2: "sse2", FeatAVX: "avx", FeatAVX2: "avx2",
	FeatAVX512: "avx512", FeatNEON: "neon", FeatSVE: "sve", FeatSVE2: "sve2",
	FeatRVV: "rvv", FeatVX: "vx", FeatLSX: "lsx", FeatLASX: "lasx", FeatVSX: "vsx",
}

func (f Feature) String() string {
	if s, ok := featureNames[f]; ok {
		return s
	}
	return fmt.Sprintf("Feature(%d)", int(f))
}

// commonFlags are passed for every target.
//
// Each of these prevents something specific:
//
//   - -ffreestanding, -fno-builtin, -fno-builtin-memset: stop clang turning a
//     loop into a call to memset or memcpy. Plan 9 assembly has no PLT, so any
//     call to an undefined symbol is fatal. Measured over an eight-kernel
//     suite, these bring the undefined-symbol count to zero on every target.
//   - -fno-jump-tables: a jump table lands in .rodata and is referenced by
//     absolute address, which would not survive being lifted into a Go symbol.
//   - -fno-asynchronous-unwind-tables, -fno-exceptions, -fno-rtti: strip
//     .eh_frame and the CFI directives, which have no meaning here.
//   - -fomit-frame-pointer: frees a register; Go's frame pointer convention is
//     handled by the emitted prologue, not by clang's.
//   - -mllvm -inline-threshold=1000: kernels are written as small helpers
//     calling each other; anything not inlined becomes a fatal call.
//
// Note what is *absent*: -ffast-math and -Ofast. They would let the compiler
// reassociate reductions and assume no NaN, which breaks the numerical
// contract in package kernel. viterin/vek compiles with -Ofast and that is
// exactly why its NaN behaviour is undefined.
var commonFlags = []string{
	"-O3",
	"-ffreestanding", "-fno-builtin", "-fno-builtin-memset", "-fno-builtin-memcpy",
	"-fno-jump-tables",
	// Without this, sqrt and friends must set errno, so clang emits a call to
	// libm instead of the instruction — and a call is fatal here. errno is not
	// observable from Go, so nothing is lost.
	"-fno-math-errno",
	"-fno-asynchronous-unwind-tables", "-fno-exceptions", "-fno-rtti",
	"-fomit-frame-pointer", "-fno-stack-protector",
	"-mllvm", "-inline-threshold=1000",
	"-ffp-contract=off",
}

// sysvIntArgs is the SysV AMD64 integer argument sequence.
var sysvIntArgs = []string{"DI", "SI", "DX", "CX", "R8", "R9"}

// aapcsIntArgs is the AArch64 AAPCS integer argument sequence.
var aapcsIntArgs = []string{"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}

// All is every target the generator can build.
var All = []Target{
	{
		Arch: AMD64, Tier: "sse2", Triple: "x86_64-linux-gnu",
		// x86-64 baseline already includes SSE2, so this tier needs no
		// runtime check at all — which is why gonum gets away with having no
		// dispatch layer whatsoever.
		MArch:      []string{"-march=x86-64"},
		InstrWidth: 0,
		Directives: dirAMD64,
		IntArgs:    sysvIntArgs,
		FloatArgs:  []string{"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"},
		IntResult:  "AX", FloatResult: "X0",
		AddrOf: "LEAQ", MovPtr: "MOVQ", MovInt32: "MOVL", MovByte: "MOVBLZX",
		MovFloat32: "MOVSS", MovFloat64: "MOVSD",
		MinFeature: FeatSSE2,
	},
	{
		Arch: AMD64, Tier: "avx2", Triple: "x86_64-linux-gnu",
		// x86-64-v3 is AVX2 + FMA + BMI, which is what the avx2 tier gates on.
		MArch:      []string{"-march=x86-64-v3"},
		InstrWidth: 0,
		Directives: dirAMD64,
		IntArgs:    sysvIntArgs,
		FloatArgs:  []string{"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"},
		IntResult:  "AX", FloatResult: "X0",
		AddrOf: "LEAQ", MovPtr: "MOVQ", MovInt32: "MOVL", MovByte: "MOVBLZX",
		MovFloat32: "MOVSS", MovFloat64: "MOVSD",
		MinFeature: FeatAVX2,
	},
	{
		Arch: AMD64, Tier: "avx512", Triple: "x86_64-linux-gnu",
		// -mprefer-vector-width=512 is required, not optional: without it LLVM
		// caps at 256 bits even on x86-64-v4. Measured on one kernel, zero zmm
		// registers appear without the flag and 74 appear with it.
		MArch:      []string{"-march=x86-64-v4", "-mprefer-vector-width=512"},
		InstrWidth: 0,
		Directives: dirAMD64,
		IntArgs:    sysvIntArgs,
		FloatArgs:  []string{"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"},
		IntResult:  "AX", FloatResult: "X0",
		AddrOf: "LEAQ", MovPtr: "MOVQ", MovInt32: "MOVL", MovByte: "MOVBLZX",
		MovFloat32: "MOVSS", MovFloat64: "MOVSD",
		MinFeature: FeatAVX512,
	},
	{
		Arch: ARM64, Tier: "neon", Triple: "aarch64-linux-gnu",
		MArch:      []string{"-march=armv8-a"},
		InstrWidth: 4,
		Directives: dirWord,
		IntArgs:    aapcsIntArgs,
		FloatArgs:  []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"},
		IntResult:  "R0", FloatResult: "F0",
		AddrOf: "MOVD", MovPtr: "MOVD", MovInt32: "MOVW", MovByte: "MOVBU",
		MovFloat32: "FMOVS", MovFloat64: "FMOVD",
		// R27 and R28 are reserved by Go's compiler and linker; R18 is the OS
		// platform register. Unlike x86, clang honours -ffixed for these.
		Reserved:   []string{"x28", "x27", "x18"},
		GoOwned:    []string{"x28", "x27", "x18"},
		MinFeature: FeatNEON,
	},
	{
		Arch: ARM64, Tier: "sve2", Triple: "aarch64-linux-gnu",
		// The differentiator. Go's assembler cannot encode a single SVE
		// instruction, and upstream has deferred scalable vectors with no
		// design, so raw encodings are the only route. Conveniently this is
		// also the cleanest target measured: zero relocations, zero undefined
		// symbols.
		MArch:      []string{"-march=armv9-a+sve2"},
		InstrWidth: 4,
		Directives: dirWord,
		IntArgs:    aapcsIntArgs,
		FloatArgs:  []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"},
		IntResult:  "R0", FloatResult: "F0",
		AddrOf: "MOVD", MovPtr: "MOVD", MovInt32: "MOVW", MovByte: "MOVBU",
		MovFloat32: "FMOVS", MovFloat64: "FMOVD",
		Reserved:    []string{"x28", "x27", "x18"},
		GoOwned:     []string{"x28", "x27", "x18"},
		DisasmFlags: []string{"--mattr=+sve2"},
		MinFeature:  FeatSVE2,
	},
	{
		Arch: RISCV64, Tier: "rvv", Triple: "riscv64-linux-gnu",
		// Deliberately rv64gv, not rv64gcv: the compressed extension makes
		// some instructions two bytes wide, so a function can end on a 2-byte
		// boundary — and Go's riscv64 assembler has no BYTE directive to
		// express the remainder, only WORD. Dropping compression keeps every
		// instruction four bytes and every function a whole number of them.
		//
		// -mno-relax because linker relaxation is meaningless here: nothing
		// links this object, and the R_RISCV_RELAX hints it emits are not
		// fixups at all, just markers saying a sequence *could* be shortened.
		MArch: []string{"-march=rv64gv", "-mno-relax"},
		// X27 is Go's goroutine pointer (REG_G in cmd/internal/obj/riscv).
		// clang accepts -ffixed for it, unlike s390x, loongarch and ppc64.
		Reserved:   []string{"x27"},
		GoOwned:    []string{"s11", "x27"},
		InstrWidth: 4,
		Directives: dirWord,
		IntArgs:    []string{"X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17"},
		FloatArgs:  []string{"F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17"},
		IntResult:  "X10", FloatResult: "F10",
		AddrOf: "MOV", MovPtr: "MOV", MovInt32: "MOVW", MovByte: "MOVBU",
		MovFloat32: "MOVF", MovFloat64: "MOVD",
		DisasmFlags: []string{"--mattr=+v"},
		MinFeature:  FeatRVV,
	},
	{
		Arch: S390X, Tier: "vx", Triple: "s390x-linux-gnu",
		MArch: []string{"-march=z14"},
		// No Reserved: clang has no -ffixed for SystemZ at all, and it accepts
		// the global register variable in csrc/goabi.h without honouring it.
		// The backstop below is what actually keeps r13 safe here.
		GoOwned:    []string{"r13"},
		SaveArea:   160, // the zSeries ABI register save area
		InstrWidth: 0,
		Directives: dirS390X,
		BigEndian:  true,
		IntArgs:    []string{"R2", "R3", "R4", "R5", "R6"},
		FloatArgs:  []string{"F0", "F2", "F4", "F6"},
		IntResult:  "R2", FloatResult: "F0",
		AddrOf: "MOVD", MovPtr: "MOVD", MovInt32: "MOVW", MovByte: "MOVBZ",
		MovFloat32: "FMOVS", MovFloat64: "FMOVD",
		DisasmFlags: []string{"--mcpu=z14"},
		MinFeature:  FeatVX,
	},
	{
		Arch: LOONG64, Tier: "lasx", Triple: "loongarch64-linux-gnu",
		MArch: []string{"-march=la464"},
		// Reserved by the global register variable in csrc/goabi.h, which
		// clang does honour here.
		GoOwned:    []string{"r22"},
		InstrWidth: 4,
		Directives: dirWord,
		IntArgs:    []string{"R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"},
		FloatArgs:  []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"},
		IntResult:  "R4", FloatResult: "F0",
		AddrOf: "MOVV", MovPtr: "MOVV", MovInt32: "MOVW", MovByte: "MOVBU",
		MovFloat32: "MOVF", MovFloat64: "MOVD",
		DisasmFlags: []string{"--mattr=+lasx"},
		MinFeature:  FeatLASX,
	},
	{
		Arch: PPC64LE, Tier: "vsx", Triple: "powerpc64le-linux-gnu",
		GoOwned: []string{"r30", "r2"},
		// No SaveArea: measured, these kernels write nothing above the stack
		// pointer, so there is nothing for a trampoline to contain.
		// The one target with a non-trivial relocation model: constants are
		// reached through the TOC, which needs its own handling. Scheduled
		// last for that reason.
		MArch:      []string{"-mcpu=power9"},
		InstrWidth: 4,
		Directives: dirWord,
		IntArgs:    []string{"R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"},
		FloatArgs:  []string{"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"},
		IntResult:  "R3", FloatResult: "F1",
		AddrOf: "MOVD", MovPtr: "MOVD", MovInt32: "MOVW", MovByte: "MOVBZ",
		MovFloat32: "FMOVS", MovFloat64: "FMOVD",
		// No -ffixed here: clang's ppc64 target does not accept -ffixed-rN,
		// and the ABI already reserves what the Go runtime needs. r2 is the
		// TOC pointer and clang treats it as fixed of its own accord.
		DisasmFlags: []string{"--mcpu=pwr9"},
		MinFeature:  FeatVSX,
	},
}

// Flags returns the full clang invocation for the target.
func (t Target) Flags() []string {
	f := append([]string{"--target=" + t.Triple}, t.MArch...)
	f = append(f, commonFlags...)
	for _, r := range t.Reserved {
		f = append(f, "-ffixed-"+r)
	}
	if t.Arch == AMD64 {
		// Go writes below the stack pointer during signal delivery and stack
		// growth, so the SysV 128-byte red zone would be corrupted underneath
		// it. This is not optional.
		//
		// Note there is no -ffixed-r14 here: clang rejects it on x86, and it
		// is not needed. abi-internal.md:424 states that R14 and X15 are
		// undefined in ABI0, so assembly may clobber both — and 28 files in
		// the standard library already do.
		f = append(f, "-mno-red-zone", "-mstackrealign")
	}
	return f
}

// BuildTag is the //go:build constraint for a generated file.
func (t Target) BuildTag() string {
	return fmt.Sprintf("%s && !purego", t.Arch)
}

// Suffix is the filename suffix for a generated file, chosen so Go's own
// build constraints on the filename agree with the build tag.
func (t Target) Suffix() string { return fmt.Sprintf("_%s_%s", t.Tier, t.Arch) }

// Find returns the target with the given architecture and tier.
func Find(arch Arch, tier string) (Target, bool) {
	for _, t := range All {
		if t.Arch == arch && t.Tier == tier {
			return t, true
		}
	}
	return Target{}, false
}

// ForArch returns every tier of one architecture, weakest first.
func ForArch(arch Arch) []Target {
	var out []Target
	for _, t := range All {
		if t.Arch == arch {
			out = append(out, t)
		}
	}
	return out
}

func (t Target) String() string { return string(t.Arch) + "/" + t.Tier }
