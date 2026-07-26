# Getting SIMD machine code into Go without cgo

*State of the art as of July 2026, plus feasibility measurements taken on this machine
(clang 22.1.8, Go 1.26.2, Zen 5 host).*

## The measured feasibility result

**One C source, compiled per target with `-O3 -ffreestanding -fno-builtin`, produces good
vectorized code on every architecture we care about:**

| Target | Result |
|---|---|
| `x86-64-v4` | `vfmadd213ps`, 4× unrolled automatically. |
| `armv9-a+sve2` | NEON `fmla v1.4s` for the fixed loop, **and LLVM auto-vectorized the plain scalar tail loop into scalable SVE** (`ptrue`, `cnth`, `ld1w {z0.s}, p0/z`). |
| `rv64gcv` | Real RVV: `vsetvli`, `vle32.v`, `vfmacc.vv`, with a proper VLA tail loop. |

**LLVM auto-vectorizes plain scalar C loops into the variable-length ISAs (SVE, RVV).** Go's
compiler never will. This is the entire justification for a clang-fed pipeline, and it means
kernels can be written *once*, as ordinary scalar C, rather than once per ISA.

### Required flags, and why

| Flag | Why |
|---|---|
| `-mno-red-zone` (amd64) | **Mandatory.** Go writes below SP during signal delivery and stack growth; the SysV 128-byte red zone will be corrupted. |
| `-mstackrealign` (amd64) | SIMD alignment inside the borrowed frame. |
| `-ffreestanding -fno-builtin` | Eliminates libcalls. **Measured: zero undefined symbols on every target.** |
| `-fno-builtin-memset` | Scalar remainder loops otherwise get pattern-matched into a `memset` call, which is fatal. |
| `__attribute__((ext_vector_type(N), aligned(1)))` | Without `aligned(1)`, clang emits `vmovaps` (aligned) and **faults on unaligned Go slices**. Measured: 0 aligned moves with it, 40 unaligned. |
| `-mprefer-vector-width=512` | Without it LLVM caps at 256-bit even on `x86-64-v4`. Measured: **0 zmm without, 74 with**. |
| `-ffixed-x28 -ffixed-x27 -ffixed-x18` (arm64) | `x27`/`x28` are reserved by Go's compiler and linker (`doc/asm.html:850`); `x18` is the OS platform register. All three accepted and effective (x18 usage 2 → 0). |
| `-fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti` | Strips `.eh_frame`/`.cfi_*`. |
| **NOT `-ffast-math` / `-Ofast`** | Breaks NaN/Inf semantics. Use `-ffp-contract=fast` and per-kernel `#pragma clang fp`. This is vek's worst bug — see [03-competitive-analysis.md](03-competitive-analysis.md). |

**amd64 register note:** clang **rejects `-ffixed-r14`** on x86, and does emit `r14`. This turns
out not to matter — `abi-internal.md:424` says *"In ABI0, these are undefined, so transitions
from ABIInternal to ABI0 can ignore these registers"* for both `R14` (goroutine pointer) and
`X15` (zero register). 28 stdlib `*_amd64.s` files already clobber R14. See
[04-abi-and-overhead.md](04-abi-and-overhead.md).

### Extraction hazards, measured over an 8-kernel suite

Kernels: `add`, `iadd`, `sum`, `dot`, `max`, `abs`, `count`, `exp`.

| Target | Relocations | Undefined syms | Notes |
|---|---|---|---|
| **arm64** | **0** | **0** | Constants materialized inline. **Cleanest target.** |
| s390x | 2 (`R_390_PC32DBL`) | 0 | Trivial. |
| riscv64 | 14, all `R_RISCV_BRANCH` | 0 | Internal/self-relative — no fixups needed. |
| amd64 | 12 (`R_X86_64_PC32` → `.LCPI*` in `.rodata.cst4`) | 0 | **Confined to 2 kernels** (`abs`, `exp`) — the ones with float constants. `add`/`sum`/`dot`/`max`/`count` have **none**. |
| **ppc64le** | `R_PPC64_TOC16_HA/LO` + **undefined `.TOC.`** | 1 | **The only messy target.** Needs TOC handling. |

So the extractor has exactly one hard problem — rewriting PC-relative references to a local
constant pool into references to Go `DATA`/`GLOBL` symbols — and it only arises on some targets,
for some kernels.

### SVE extraction is a clean byte blob

An SVE2 kernel with the constraints above: **0 relocations, 0 undefined symbols**, 296 bytes of
`.text`, containing real `ptrue`/`ld1w`/`st1w`/`fmla`. It drops into a Go `.s` as `WORD $0x...`
with nothing to fix up.

**The one ISA Go's assembler cannot spell is also the easiest one to extract.**

## Tool survey

| Tool | Status | Arches | Verdict |
|---|---|---|---|
| **[gorse-io/goat](https://github.com/gorse-io/goat)** | **Active** — v0.2.1, 2026-06-18, Apache-2.0 | **amd64, arm64, riscv64, loong64, s390x, ppc64le** | **Build on this.** |
| [minio/c2goasm](https://github.com/minio/c2goasm) | **ARCHIVED** 2021-11 | amd64 only | Dead. |
| [minio/asm2plan9s](https://github.com/minio/asm2plan9s) | **ARCHIVED** 2022-10 | amd64 only | Dead. |
| [kelindar/gocc](https://github.com/kelindar/gocc) | 2026-03 | amd64, arm64 | Strictly less capable: **max 4 arguments**, all 64-bit, no C++. |
| [cloudwego/asm2asm](https://github.com/cloudwego/asm2asm) | 2025-08 | amd64, arm64 | ByteDance-internal (drives sonic, base64x). Undocumented but battle-tested. |
| [avo](https://github.com/mmcloughlin/avo) | **Caretaker mode** | **x86 only** | Assembler front-end, not a compiler. Escape hatch only. |

### goat — how it works

Read from `internal/translate.go`, `internal/amd64/parser.go`, `internal/data.go`:

1. **Signature extraction.** Runs `clang -Xclang -ast-dump=json -fsyntax-only` and walks the JSON
   AST to collect function declarations. **This auto-generates the Go stub file** with
   `//go:noescape` and the correct signature. (c2goasm made you hand-write it — a real source of
   `go vet` asmdecl mismatches.)
2. **Compile.** clang with `-mllvm -inline-threshold=1000 -fno-asynchronous-unwind-tables
   -fno-exceptions -fno-rtti -fno-builtin`; amd64 adds `-mno-red-zone -mstackrealign`; ppc64le
   gets `-finline-limit=1000 -ffixed-r0 -ffixed-r30`.
3. **Dual parse.** Parses both the `.s` text (mnemonics, labels, `rip` references) *and*
   `objdump` of the `.o` (raw encoding bytes), pairing them line by line.
4. **Emit Plan 9.** `TEXT ·name(SB), $frame-argsize`, then a prologue mapping `(FP)` slots to the
   C ABI registers (amd64: `DI, SI, DX, CX, R8, R9` + `X0..X7`; arm64: `R0..R7`), then the body
   as `QUAD $0x… / LONG $0x… / WORD $0x… / BYTE $0x…` with the disassembly in a comment.
5. **Two escapes from raw bytes:** (i) conditional and unconditional jumps become **real Plan 9
   branches to real labels**, so the Go assembler does offset fixups and the code stays
   relocatable; (ii) `leaq sym(%rip), %reg` → `LEAQ sym<>(SB), REG`.
6. **Constants.** Collected from the compiler's data sections and emitted as
   `DATA sym<>+0x000(SB)/8, $0x…` + `GLOBL sym<>(SB), 8, $len` (flag 8 = `RODATA`).
   `.ascii`/`.asciz` decoded.

**Because it emits raw encodings, it is not limited by Go's assembler ISA coverage.** That is how
SVE and SME code gets into Go today, and it is exactly what
[01-go-asm-isa-coverage.md](01-go-asm-isa-coverage.md) requires.

### goat's known limitations (its README says *"potentially BUGGY code generation"*)

- **No call statements except inlined functions.** Hard wall — Plan 9 asm has no PLT.
- **`__builtin_expf`/`__builtin_sqrtf` compile to `bl expf` — a call — which is fatal.**
  ⇒ transcendentals must be inline polynomial evaluation. See SLEEF below.
- Argument types limited upstream to `int64_t/long/float/double/_Bool`/pointer.
  go-highway's fork extends to `int32_t, uint32_t, uint64_t, float16_t`.
- Return types: `void`, `int64_t`, `long`, `float`, `double`, `_Bool`.
- Parser fragility — these all break it: `static inline` helper definitions; single-line
  `if (x) { y; }`; `union` type-punning; `int arr[4] = {a,b,c,d}` with variable initializers.
- Output directory must differ from the source directory.
- Cross-compilation has friction despite `-t`, `--target-os`, `--sysroot`, `-I`.

**Our plan: fork it, harden it, upstream fixes.** go-highway already vendors and extends it,
which both validates the approach and gives a second patch set to learn from.

## Rust: evaluated and rejected

**Verdict: C with clang.** This is a blocker list, not a preference.

| Hazard | Why Rust breaks | clang equivalent |
|---|---|---|
| **Red zone** | Go writes below SP. **rustc has no stable `-mno-red-zone`** — only `-C target-feature=-red-zone` (fragile LLVM feature) or nightly `-Z no-redzone`. | `-mno-red-zone` ✅ |
| **Stack probes** | Emits `__rust_probestack` calls for frames > 4 KiB. **Any call is fatal.** | not emitted |
| **`memcpy`/`memset`** | Rust moves large values via `memcpy` **far more eagerly than C**. | `-fno-builtin` ✅ (measured: 0 undefined syms) |
| **Panics** | Every slice index / bounds check calls into `core`'s panic machinery. | n/a |
| **Unwind tables** | `.eh_frame`/`.cfi_*` to strip. | `-fno-asynchronous-unwind-tables` ✅ |
| **Mangling** | **Rust 1.97 turns on v0 mangling by default on stable.** | n/a |
| **Reserved registers** | No `-ffixed-xN` equivalent. | `-ffixed-x28 -ffixed-x27 -ffixed-x18` ✅ verified |
| **Tooling** | **No `rust2goasm` exists.** Every tool in this space parses **clang** output. | 6-arch tool exists today |

And the reason you would want Rust — portability — **is not there**. `core::simd` is *still*
nightly-only (feature gate `portable_simd`,
[rust#86656](https://github.com/rust-lang/rust/issues/86656), gating list
[portable-simd#364](https://github.com/rust-lang/portable-simd/issues/364)) as of **Rust 1.97,
July 2026**, and it does **not** target SVE/RVV — Rust's actual SVE work is a *separate*
`#[repr(scalable)]` type, precisely because fixed-`N` cannot be retrofitted. On stable Rust you
write `core::arch` per-ISA intrinsics, exactly what you would write in C.

### The linking alternative is also dead

Filippo Valsorda's [rustgo](https://words.filippo.io/rustgo/) (2017) achieved 5.11 ns/call vs
cgo's 73.6 ns by linking a `#![no_std]` staticlib and hand-writing an ABI trampoline. It is dead:

1. `//go:binary-only-package` was **removed in Go 1.12**.
2. `//go:cgo_import_static` outside cgo-generated code is **rejected by the toolchain**.
3. The proposal to relax that — [golang/go#75473](https://github.com/golang/go/issues/75473),
   filed 2025-09-15 — was **closed the next day**. Cherry Mui's objection is the definitive
   statement of Go-team policy: *"on most platforms, we don't have a general mechanism to call a
   C function from Go without using cgo… If we do this, it probably would encourage user to write
   hacky unsafe code for that. That doesn't seem like a good idea."*
4. `//go:linkname` pull-only usage has been progressively locked down since Go 1.23
   ([#67401](https://github.com/golang/go/issues/67401)).

Useful nugget from that thread: Ian Lance Taylor asked what actually breaks without
`cgo_import_static`, and the answer was *nothing* — `//go:linkname` alone suffices with the
**internal** linker. The real barrier is getting the foreign `.o` into the package archive, for
which the `go` command offers no supported hook.

**Re-evaluation trigger:** our extractor consumes **object files**, not C, so it is
source-language agnostic. Adding a Rust front-end later costs a build rule, not a redesign.

## C/C++ kernel-source libraries

### SLEEF — the answer for transcendentals

[github.com/shibatch/sleef](https://github.com/shibatch/sleef), Boost license, v3.9.0 (2025-03).
Vectorized libm: `sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh exp exp2
exp10 expm1 log log2 log10 log1p pow cbrt sqrt hypot erf erfc tgamma lgamma`, in **1.0-ULP** and
**3.5-ULP (`_u35`)** variants. Already ported per-ISA: SSE2/SSE4/AVX/FMA4/AVX2/AVX512F,
NEON/**SVE**, VSX, ZVECTOR, **RVV**, WASM, plus scalar `purec`.

**Why it fits this pipeline exactly:** SLEEF kernels are straight-line polynomial evaluation with
**no function calls**, and constants that are mostly immediates. That directly dodges goat's
`__builtin_expf → bl expf` wall.

Caveats: the non-finite/special-case paths do branch (goat handles branches correctly, so this is
fine); `_u35` variants are cheaper and branchier. **`#include` the inline headers and force
inlining — do NOT link `libsleef`, and do NOT use `-fveclib=SLEEF`**, both of which produce calls.

**SLEEF removed its Go language support (~v2.40) and no Go port exists today.** This is greenfield.

### Others evaluated

- **[Google Highway](https://github.com/google/highway)** — 27 targets (`SSE2 SSSE3 SSE4 AVX2
  AVX3 AVX3_DL AVX3_ZEN4 AVX3_SPR AVX10_2 NEON NEON_BF16 SVE SVE2 SVE_256 SVE2_128 RVV WASM
  WASM_EMU256 Z14 Z15 PPC8 PPC9 PPC10 LSX LASX EMU128 SCALAR`), used in Chromium, Firefox,
  TensorFlow, NumPy, JPEG XL. **Not directly extractable**: templates + "inline everything"
  (10–100× penalty without inlining, so there is no `Add<float,4>` symbol to extract),
  `foreach_target.h` re-inclusion compiles N copies per TU, and runtime-dispatch statics become
  data relocations no converter can express. Tellingly, **go-highway reimplemented Highway's API
  in Go rather than extract from it.** Best used as a *design* reference.
- **[xsimd](https://github.com/xtensor-stack/xsimd)** — simpler C++ than Highway, and its
  "burden" of one TU per target compiled with different flags is *exactly* the shape a transpiler
  wants. Worth a spike.
- **[SIMDe](https://github.com/simd-everywhere/simde)** — write `_mm256_*` once, compile
  anywhere. Cheap cross-ISA strategy, but locks you into Intel's mental model with no way to
  express scalable-width SVE natively, and translation quality is uneven for shuffle-heavy code.

## What real Go libraries do today

| Library | Technique | amd64 | arm64 |
|---|---|---|---|
| klauspost/compress | **avo**, incl. the Honeycomb arm64-lowering fork | AVX2/BMI2/AVX-512 | mix of hand-written and avo-lowered |
| segmentio/asm | avo (amd64) + **hand-written Plan 9** (arm64) | SSE→AVX-512 | hand-written NEON |
| zeebo/xxh3 | avo + hand-written NEON | SSE/AVX2/AVX-512 | hand-written |
| minio/simdjson-go | **c2goasm** from simdjson C++ | AVX2 + AVX-512 | **none** |
| bytedance/sonic | clang → **asm2asm** → runtime loader/JIT | yes | hand-written + asm2asm |
| cloudwego/base64x | clang → **asm2asm** | AVX2/SSE | — |

**Nobody has a single-source cross-arch pipeline.** The universal pattern is:

```
foo.go            // API + dispatch
foo_amd64.go/.s   // avo, or clang→c2goasm/asm2asm/goat
foo_arm64.go/.s   // hand-written Plan 9 NEON
foo_default.go    // pure Go
```

plus a `noasm`/`purego` build tag and a CPU-feature check. go-highway is the first serious attempt
to break that pattern, and it is eight months old.

### avo, specifically

A Go DSL that is an *assembler front-end*, not a compiler: you write `VPADDD` against virtual
registers and avo does register allocation, frame layout, `(FP)` argument loading, and emits `.s`
plus a matching `.go` stub (so `go vet` asmdecl passes). **x86 only.** AVX-512 arrived in v0.4.0
(Nov 2021); v0.6.0 (Jan 2024) added GFNI, VAES, VNNI, VPCLMULQDQ, VPOPCNTDQ, BITALG, VBMI2.

**Maintenance:** last tagged release v0.6.0, **2024-01-07** — no release in 2.5 years. Repo
activity is real but **every commit since Dec 2024 is a bot commit**; the last human maintainer
commit is Dec 2024. Substantive community PRs sit unmerged for years. **Caretaker mode** — pin
your version, expect to fork.

**arm64 development worth watching:** [PR #486](https://github.com/mmcloughlin/avo/pull/486) (Liz
Fong-Jones, Honeycomb, July 2026) adds an experimental arm64 *printer* that mechanically lowers
the already-register-allocated x86 instruction stream to arm64. **Scalar integer only — no NEON,
no SVE, no autovectorization.** It is shipping in production in `klauspost/compress`
(`zstd/seqdec_arm64.s`, via `replace github.com/mmcloughlin/avo => github.com/honeycombio/avo`),
reporting ~1.5× over pure Go with zero NEON.

## Things that do not exist

- No LLVM-IR → Plan 9 tool. The LLVM-adjacent Go work (`gollvm`, TangoLLVM) is *alternative Go
  compilers*, not a way to get foreign IR into `gc`-compiled Go.
- No `rust2goasm`.
- Nothing new appeared in response to the Go 1.26 simd experiment except go-highway.

## Sources

- [gorse-io/goat](https://github.com/gorse-io/goat) · [AVX512 in Golang via C Compiler](https://gorse.io/posts/avx512-in-golang)
- [mmcloughlin/avo](https://github.com/mmcloughlin/avo) · [PR #486 arm64](https://github.com/mmcloughlin/avo/pull/486) · [honeycombio/avo](https://github.com/honeycombio/avo)
- [minio/c2goasm](https://github.com/minio/c2goasm) · [kelindar/gocc](https://github.com/kelindar/gocc) · [cloudwego/asm2asm](https://github.com/cloudwego/asm2asm)
- [rustgo](https://words.filippo.io/rustgo/) · [golang/go#75473](https://github.com/golang/go/issues/75473) · [golang/go#67401](https://github.com/golang/go/issues/67401)
- [rust-lang/portable-simd#364](https://github.com/rust-lang/portable-simd/issues/364) · [Rust 1.97.0](https://blog.rust-lang.org/2026/07/09/Rust-1.97.0/) · [State of SIMD in Rust 2025](https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d)
- [SLEEF](https://github.com/shibatch/sleef) · [Highway](https://github.com/google/highway) · [xsimd](https://github.com/xtensor-stack/xsimd) · [SIMDe](https://github.com/simd-everywhere/simde)
