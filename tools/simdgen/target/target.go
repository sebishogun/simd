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

	// The narrow integer moves, which widen to a full register on the way in.
	//
	// These exist because the narrow element types pass their scalar arguments
	// to C as a 64-bit integer rather than as their own width. That looks
	// wasteful and is the only portable choice: how a narrow argument is
	// extended into its register is a property of each ABI, and they disagree.
	// Both RISC-V and LoongArch require an *unsigned* 32-bit argument to be
	// sign-extended to 64 bits, where every other target here zero-extends it;
	// an implementation that guessed one way would be right on five targets
	// and silently wrong on two, for values above 2^31 only.
	//
	// Declaring the C parameter as long long removes the disagreement — a
	// 64-bit integer argument occupies its whole register everywhere — and
	// moves the widening to this side, where the Go type says unambiguously
	// which of the two it is. Signed types sign-extend, unsigned zero-extend,
	// and the kernel casts back down to its element type.
	//
	// MovByte is the unsigned byte load and serves uint8 as well.
	MovInt8, MovInt16, MovUint16, MovUint32 string

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

	// FloatArgsUseIntSlot says that a floating-point argument also consumes an
	// integer register slot, even though it is passed in a float register.
	//
	// This is the ELFv2 parameter save area: every argument owns a slot in it,
	// in declaration order, and a float's slot is simply skipped in the GPR
	// sequence rather than reused. So for
	//
	//	simd_scale_f64(double *d, const double *a, double s, isize n)
	//
	// ppc64le passes d in r3, a in r4, s in f1 — and n in **r6**, because s
	// owns r5. Assigning n to r5 hands the kernel a garbage length, which it
	// then writes that many elements through. It faulted inside the runtime's
	// stack allocator, several calls away from anything to do with this.
	//
	// Every other target here assigns its integer and floating-point argument
	// registers independently: s390x reads the same kernel's length from r4,
	// and AAPCS64, RISC-V and System V all do the same.
	FloatArgsUseIntSlot bool

	// StackReg is the stack pointer as the disassembler prints it, for the
	// above-the-stack-pointer check in package verify. Empty where the ABI
	// makes that mistake impossible, in which case the check is skipped.
	StackReg string

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

	// ProtectedZone is how many bytes below the stack pointer this ABI
	// reserves for the callee, and that this platform's Go runtime provably
	// does not touch.
	//
	// Zero everywhere but ppc64le, and it is not a licence to write below SP
	// in general — on amd64 the equivalent is the SysV red zone, which Go
	// *does* clobber, which is why -mno-red-zone is mandatory there and this
	// field stays zero.
	//
	// ELFv2 reserves 288 bytes and clang uses them for every leaf that spills:
	// 163 kernels here, nearly half the target, and there is no flag that
	// stops it. -fno-omit-frame-pointer reduces it to 108 and -mno-red-zone
	// makes it *worse* — 256 — so the choice was to reject those kernels or to
	// establish whether the writes are actually unsafe.
	//
	// They are not, on this platform, and the reasoning is specific rather
	// than hopeful. A kernel here is NOSPLIT so nothing grows the stack under
	// it; it makes no call, so nothing else can; it is hand-written assembly,
	// which the runtime never selects as an asynchronous preemption point
	// (isAsyncSafePoint returns false for any function with FuncFlagAsm), so
	// no stack copy happens while it runs; and Go installs a signal stack per
	// M, so a signal handler does not use the goroutine's stack at all.
	//
	// Measured as well as argued: eight goroutines writing a sentinel 128 and
	// 256 bytes below SP, spinning, and reading it back — 160,000 calls under
	// forced garbage collection and with the CPU profiler running so SIGPROF
	// fires inside the window — clobbered none of them.
	//
	// The deepest write clang emits across these sources is -256, inside the
	// 288 the ABI promises. Writes *above* the stack pointer are a different
	// thing entirely and stay rejected: those land in the caller's frame, on
	// top of its saved link register, and that is the "returns to address 1"
	// this target was failing with before.
	ProtectedZone int

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

	// Epilogue is Plan 9 assembly to run on the way out of every kernel, for
	// the one target that needs it.
	//
	// Nothing can normally be appended to a compiled body: LLVM lays basic
	// blocks out after the return instruction, so there is no position at the
	// end that is guaranteed to execute. ppc64le needs one anyway, because
	// Go's ABI there reserves r0 by value — it holds zero at every point in
	// every function, and the compiler stores it as the zero when it clears
	// memory — while clang treats r0 as ordinary scratch and there is no
	// -ffixed-rN on its PowerPC target to say otherwise.
	//
	// Setting this makes emit rewrite the body's returns into branches to the
	// end of it, which turns "after the body" into a position that always
	// runs. See returns_ppc64.go, which is where the reasoning lives and where
	// the encodings are.
	Epilogue []string

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
		Arch: PPC64LE, Tier: "vsx", Triple: "powerpc64le-linux-gnu",
		// r2 is the TOC pointer, r13 the TLS pointer and r30 the current
		// goroutine, all of which Go requires unchanged across a call.
		//
		// r0 is not in this list even though Go reserves it too, because
		// rejecting every kernel that touches it would reject most of them:
		// it is volatile scratch under ELFv2 and LLVM uses it 212 times
		// across these sources. It is restored by Epilogue instead.
		//
		// The names are r-prefixed and the stack register is "r1" because of
		// -ppc-asm-full-reg-names below. Without that flag llvm-objdump
		// prints PowerPC registers as bare numbers, which are indistinguishable
		// from immediates — "addi 10, 9, 2" names two registers and a
		// constant — so this check matched nothing at all and the one thing
		// it exists to catch went through it unseen.
		GoOwned:             []string{"r2", "r13", "r30"},
		StackReg:            "r1",
		ProtectedZone:       288,
		Epilogue:            []string{"MOVD $0, R0"},
		FloatArgsUseIntSlot: true,
		// No SaveArea, deliberately, even though seven of 246 kernels do save
		// the link register at 16(r1) — the caller's frame under ELFv2.
		//
		// The trampoline that fixes this on s390x is not safe here. ELFv2
		// convention is that a bl to a global symbol is followed by a slot the
		// linker may rewrite into a TOC reload, and the instruction after the
		// bl in a trampoline is the RET. Losing that RET is exactly the
		// "returns to address 1" this target was failing with.
		//
		// So those seven kernels are rejected by the writes-above-the-stack-
		// pointer check in package verify and keep the portable path. Seven
		// out of 246 is a far better trade than a call convention that is
		// almost right.
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
		MovInt8: "MOVB", MovInt16: "MOVH", MovUint16: "MOVHZ", MovUint32: "MOVWZ",
		MovFloat32: "FMOVS", MovFloat64: "FMOVD",
		// No -ffixed here: clang's ppc64 target does not accept -ffixed-rN at
		// all, and a global register variable for one is accepted and then
		// ignored. r2 and r13 are reserved by the ABI and clang treats them
		// as fixed of its own accord; r30 is callee-saved, so a kernel that
		// wants it saves it first and is rejected by the writes-outside-the-
		// frame check; r0 is handled by Epilogue.
		//
		// -ppc-asm-full-reg-names makes llvm-objdump print r3 rather than 3.
		// PowerPC is the only target here where a register operand and an
		// immediate are the same text without it, which defeats every check
		// in package verify that reads operands by name.
		DisasmFlags: []string{"--mcpu=pwr9", "-mllvm=-ppc-asm-full-reg-names"},
		MinFeature:  FeatVSX,
	},
	{
		Arch: AMD64, Tier: "sse2", Triple: "x86_64-linux-gnu",
		StackReg: "%rsp",
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
		MovInt8: "MOVBQSX", MovInt16: "MOVWQSX", MovUint16: "MOVWQZX", MovUint32: "MOVLQZX",
		MovFloat32: "MOVSS", MovFloat64: "MOVSD",
		MinFeature: FeatSSE2,
	},
	{
		Arch: AMD64, Tier: "avx2", Triple: "x86_64-linux-gnu",
		StackReg: "%rsp",
		// x86-64-v3 is AVX2 + FMA + BMI, which is what the avx2 tier gates on.
		MArch:      []string{"-march=x86-64-v3"},
		InstrWidth: 0,
		Directives: dirAMD64,
		IntArgs:    sysvIntArgs,
		FloatArgs:  []string{"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"},
		IntResult:  "AX", FloatResult: "X0",
		AddrOf: "LEAQ", MovPtr: "MOVQ", MovInt32: "MOVL", MovByte: "MOVBLZX",
		MovInt8: "MOVBQSX", MovInt16: "MOVWQSX", MovUint16: "MOVWQZX", MovUint32: "MOVLQZX",
		MovFloat32: "MOVSS", MovFloat64: "MOVSD",
		MinFeature: FeatAVX2,
	},
	{
		Arch: AMD64, Tier: "avx512", Triple: "x86_64-linux-gnu",
		StackReg: "%rsp",
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
		MovInt8: "MOVBQSX", MovInt16: "MOVWQSX", MovUint16: "MOVWQZX", MovUint32: "MOVLQZX",
		MovFloat32: "MOVSS", MovFloat64: "MOVSD",
		MinFeature: FeatAVX512,
	},
	{
		Arch: ARM64, Tier: "neon", Triple: "aarch64-linux-gnu",
		StackReg:   "sp",
		MArch:      []string{"-march=armv8-a"},
		InstrWidth: 4,
		Directives: dirWord,
		IntArgs:    aapcsIntArgs,
		FloatArgs:  []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"},
		IntResult:  "R0", FloatResult: "F0",
		AddrOf: "MOVD", MovPtr: "MOVD", MovInt32: "MOVW", MovByte: "MOVBU",
		MovInt8: "MOVB", MovInt16: "MOVH", MovUint16: "MOVHU", MovUint32: "MOVWU",
		MovFloat32: "FMOVS", MovFloat64: "FMOVD",
		// R27 and R28 are reserved by Go's compiler and linker; R18 is the OS
		// platform register. Unlike x86, clang honours -ffixed for these.
		Reserved:   []string{"x28", "x27", "x18"},
		GoOwned:    []string{"x28", "x27", "x18"},
		MinFeature: FeatNEON,
	},
	{
		Arch: ARM64, Tier: "sve2", Triple: "aarch64-linux-gnu",
		StackReg: "sp",
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
		MovInt8: "MOVB", MovInt16: "MOVH", MovUint16: "MOVHU", MovUint32: "MOVWU",
		MovFloat32: "FMOVS", MovFloat64: "FMOVD",
		Reserved: []string{"x28", "x27", "x18"},
		GoOwned:  []string{"x28", "x27", "x18"},
		// --mattr replaces the feature set rather than adding to it, so naming
		// only sve2 leaves the disassembler unable to read base NEON. That was
		// invisible until a kernel emitted one: the compress family is the
		// first here to use sdot, and it disassembled as <unknown>, which the
		// gate check correctly refused to pass rather than assume harmless.
		//
		// +dotprod is safe to allow at this tier and is not a widening of it.
		// The tier is gated at run time on HasSVE2, and SVE2 is an ARMv9-A
		// feature; ARMv9-A implies ARMv8.5-A, and DotProd has been mandatory
		// since ARMv8.4-A. A CPU that reports SVE2 therefore has sdot. The
		// same does not hold for i8mm or bf16, which stay out.
		DisasmFlags: []string{"--mattr=+sve2,+neon,+dotprod"},
		MinFeature:  FeatSVE2,
	},
	{
		Arch: RISCV64, Tier: "rvv", Triple: "riscv64-linux-gnu",
		StackReg: "sp",
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
		MovInt8: "MOVB", MovInt16: "MOVH", MovUint16: "MOVHU", MovUint32: "MOVWU",
		MovFloat32: "MOVF", MovFloat64: "MOVD",
		DisasmFlags: []string{"--mattr=+v"},
		MinFeature:  FeatRVV,
	},
	{
		Arch: S390X, Tier: "vx", Triple: "s390x-linux-gnu",
		StackReg: "%r15",
		MArch:    []string{"-march=z14"},
		// No Reserved: clang has no -ffixed for SystemZ at all, and it accepts
		// the global register variable in csrc/goabi.h without honouring it.
		// The backstop below is what actually keeps r13 safe here.
		//
		// Re-verified on clang 22.1.8 (2026-07): -ffixed-r13 is rejected for
		// this target (the driver's "did you mean -ffixed-r19/-ffixed-a5"
		// suggestions come from other targets' flag tables); -mllvm
		// -help-hidden lists no SystemZ reserve option; and under forced GPR
		// pressure a file-scope `register long g __asm__("r13")` changes the
		// count of r13 uses not at all — eighteen with, eighteen without. A
		// save/restore trampoline is not a route either: async preemption
		// delivers SIGURG at any instruction, the signal path reads g from
		// r13, and a kernel that has borrowed it at that moment hands the
		// runtime a garbage g. The ~380 excluded slots stay excluded until
		// upstream clang gains SystemZ register reservation.
		GoOwned:    []string{"r13"},
		SaveArea:   160, // the zSeries ABI register save area
		InstrWidth: 0,
		Directives: dirS390X,
		BigEndian:  true,
		IntArgs:    []string{"R2", "R3", "R4", "R5", "R6"},
		FloatArgs:  []string{"F0", "F2", "F4", "F6"},
		IntResult:  "R2", FloatResult: "F0",
		AddrOf: "MOVD", MovPtr: "MOVD", MovInt32: "MOVW", MovByte: "MOVBZ",
		MovInt8: "MOVB", MovInt16: "MOVH", MovUint16: "MOVHZ", MovUint32: "MOVWZ",
		MovFloat32: "FMOVS", MovFloat64: "FMOVD",
		DisasmFlags: []string{"--mcpu=z14"},
		MinFeature:  FeatVX,
	},
	{
		Arch: LOONG64, Tier: "lasx", Triple: "loongarch64-linux-gnu",
		MArch: []string{"-march=la464"},
		// r22 is the current goroutine. The global register variable in
		// csrc/goabi.h does not reserve it: clang accepts the declaration for
		// every spelling of the register and allocates it anyway, exactly as
		// on SystemZ, and LoongArch has no -ffixed-rN at all. So the only
		// defence is this check, and the kernels that lose it keep the
		// portable path.
		//
		// The spelling is "$fp" because that is what llvm-objdump prints —
		// LoongArch's ABI name for r22, since the frame pointer is what it is
		// when a frame pointer is wanted and a callee-saved general register
		// otherwise. Written "r22", as it was, this check matched nothing and
		// the clobber shipped: 616 uses across these sources, every one of
		// them fatal the moment a signal arrives, because loong64's sigtramp
		// reads g from the register rather than from thread-local storage
		// unless cgo is in play.
		GoOwned: []string{"$fp"},
		// Probed on clang 22.1.8 (2026-07): LoongArch has no -ffixed family
		// (-ffixed-x22 is AArch64's and is "unsupported for target"), and a
		// file-scope register variable bound to $r22/$fp is accepted and
		// ignored — five fp-class uses in a GPR-pressure probe with the
		// declaration and five without. Same conclusion as s390x's r13: the
		// $fp exclusions stand until upstream clang gains reservation for it.
		StackReg:   "$sp",
		InstrWidth: 4,
		Directives: dirWord,
		IntArgs:    []string{"R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"},
		FloatArgs:  []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"},
		IntResult:  "R4", FloatResult: "F0",
		AddrOf: "MOVV", MovPtr: "MOVV", MovInt32: "MOVW", MovByte: "MOVBU",
		MovInt8: "MOVB", MovInt16: "MOVH", MovUint16: "MOVHU", MovUint32: "MOVWU",
		MovFloat32: "MOVF", MovFloat64: "MOVD",
		DisasmFlags: []string{"--mattr=+lasx"},
		MinFeature:  FeatLASX,
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
