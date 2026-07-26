# Architecture decisions

*Each row states the decision, the reason, and what would make us revisit it. Evidence lives in
the sibling documents.*

## The decisions

### D1 — Assembly-first, no `GOEXPERIMENT` dependency

**Decision.** Kernels are pre-compiled assembly committed to the repo, dispatched by runtime CPU
detection. Go's `simd`/`simd/archsimd` intrinsics are *not* the foundation.

**Why.**
- The public unit of work is a whole slice, so *n* is large and the ~1.4 ns call boundary
  amortizes to ~0.01% at n=64K ([04](04-abi-and-overhead.md) §3).
- On the loop body LLVM beats Go's backend structurally: **Go's compiler does not unroll loops**,
  has no comparable scheduler, and inserts bounds checks.
- SVE/SVE2 and RVV are unreachable any other way — Go's assembler cannot express SVE at all, and
  upstream deferred scalable vectors with *"no concrete design"* ([01](01-go-asm-isa-coverage.md) §arm64).
- `GOEXPERIMENT=simd` is a build-time flag **consumers** must set. A library cannot require it.
- The intrinsics API already broke once between Go 1.26 and 1.27 and is outside the Go 1
  compatibility promise.

**Cost, stated honestly.** At n < ~64 an inlinable intrinsic is up to 5.7× faster than the best
possible assembly. Mitigated by D6 (thresholds).

**Revisit when.** [golang/go#78979](https://github.com/golang/go/issues/78979) drops the
GOEXPERIMENT gate *and* the API stabilizes. Then add a `goexperiment.simd` tier behind a build tag
to win the small-*n* band — an optimization, not a re-foundation. Structure backends so a
`foo_amd64.s` can be swapped for `foo_amd64_simd.go` behind the same internal interface.

### D2 — Kernels written in C, autovectorized first

**Decision.** One plain scalar (or generic-vector) C source per kernel. LLVM vectorizes it per
target. Drop to hand-written per-ISA intrinsics only where LLVM's output measures badly.

**Why.** Verified on this machine: the same C source produced `vfmadd213ps` on `x86-64-v4`, NEON
`fmla` *and* auto-vectorized scalable **SVE** on `armv9-a+sve2`, and real **RVV**
(`vsetvli`/`vfmacc.vv`) on `rv64gcv` ([02](02-codegen-pipelines.md) §1). Adding an architecture
becomes a build-matrix row rather than a new hand-written implementation — which is what stops
this becoming pehringer/simd, where ~100 hand-written `.s` files still left arm64 NEON with no
Div, Max, Min, or Xor.

**Revisit when.** A specific kernel family measures materially worse than hand-written
intrinsics. The decision is per-kernel, not global.

### D3 — Not Rust

**Decision.** Rejected as the kernel source language.

**Why** — a blocker list, not a preference ([02](02-codegen-pipelines.md) §Rust):
- **rustc has no stable `-mno-red-zone`.** Go writes below SP during signal delivery and stack
  growth; the SysV red zone gets corrupted. This alone is disqualifying.
- Emits `__rust_probestack` calls for frames > 4 KiB; moves large values via `memcpy` far more
  eagerly than C; every slice index calls into panic machinery. **Any call is fatal** — Plan 9
  asm has no PLT.
- Rust 1.97 turns on v0 mangling by default on stable.
- **No `rust2goasm` exists.** Every tool in this space parses clang output.
- The reason to want Rust — portability — **is not there**: `core::simd` is still nightly-only
  after five years and does **not** target SVE/RVV. On stable you write `core::arch` per-ISA
  intrinsics, exactly as in C.

The linking alternative is also closed: `//go:binary-only-package` removed in Go 1.12,
`//go:cgo_import_static` rejected outside cgo, and
[golang/go#75473](https://github.com/golang/go/issues/75473) closed the day after filing.

**Revisit when.** Never on portability grounds. The extractor consumes **object files**, not C,
so a Rust front-end remains addable as a build rule if some other reason appears.

### D4 — Fork `gorse-io/goat` as the codegen tool

**Decision.** Fork it; harden it; upstream fixes.

**Why.** Active (v0.2.1, 2026-06), Apache-2.0, already targets **all six** of our architectures,
auto-generates Go stubs from the clang AST (eliminating a whole class of `go vet` asmdecl
mismatches), and — critically — **emits raw instruction encodings**, so it is not limited by Go's
assembler ISA coverage. That is the only way to reach SVE. Alternatives: c2goasm/asm2plan9s are
archived; gocc caps at 4 arguments; avo is x86-only and in caretaker mode (last human commit
Dec 2024).

**Cost.** Its README says *"potentially BUGGY code generation"*, and it has real parser
fragilities ([02](02-codegen-pipelines.md) §goat limitations). Mitigated by D7 and by the
differential harness. go-highway already runs it in production, which both validates it and gives
us a second patch set to learn from.

### D5 — Whole-slice kernel API; no public vector type

**Decision.** The public API is `Add(dst, a, b []T)`, never `Vec.Add(Vec) Vec`.

**Why.**
- In assembly a public vector type costs a call boundary **per operation** and loses to scalar Go
  — coregex measured 50–65 cycles per crossing, 96% of runtime, and an AVX2 build **4× slower
  than SSSE3** because `VZEROUPPER` fired per call.
- Any type encoding a lane count is unimplementable on SVE/RVV where vector length is a boot-time
  property. Highway's own docs: *"A class wrapper is incompatible with the scalable vectors
  introduced by Arm SVE"* — compilers forbid wrapping sizeless types in classes. C++26
  `std::simd` silently emits fixed 128-bit NEON on SVE hardware; NumPy NEP 54 migrated to Highway
  specifically to reach SVE/RVV; Go itself split `archsimd` (fixed) from `simd` (scalable).

**Known weakness.** `d = a*b + c` as three kernel calls is three memory passes. Mitigated by an
explicit **fused combinator** catalogue (`AxpyInto`, `MulAddInto`, `DotWith`), which is exactly
what vek lacks.

### D6 — Per-op, per-type benchmarked thresholds

**Decision.** Below a measured element count, the dispatcher runs an inlined scalar Go loop and
never crosses the boundary.

**Why.** vek loses to plain Go at n=4 (3.427 ns vs 2.174 ns) and has an open issue about it. The
crossover is ~16 elements for a simple binary op, and even at n=100 you only get half the
asymptotic win. Thresholds are **benchmarked per op and per type, not guessed** — the boundary
cost is fixed but the per-element work is not.

### D7 — Gate-vs-emission verification as a CI gate

**Decision.** CI disassembles every generated `.s` and asserts **no instruction requires a CPU
feature above the tier the file is gated on** (no EVEX in an `_avx2.s`, no AVX2 in an `_avx.s`).
Plus a BP-preservation check.

**Why.** This is the single most common failure mode in the surveyed libraries — four live SIGILL
bugs across two projects (go-highway #68/#69/#67, tphakala #197/#196). It is invisible on the
developer's machine and only manifests on a user's older CPU in production. Vigilance does not
scale; **AVX-512 alone has 21 feature flags.** Upstream is heading the same way with the proposed
`//cpu:requires` directive and `cpu` vet check
([golang/go#76175](https://github.com/golang/go/issues/76175)).

Corollary rule: **no package-level `var` may execute a SIMD instruction** — go-highway#69 was
AVX2 broadcast constants in variable initialization, running before the dispatch check.

### D8 — Bit-identical results across all tiers; opt-in `Fast*` variants

**Decision.** Default operations produce byte-for-byte identical results on every tier, including
NaN/±Inf/±0/denormals. Reductions use one fixed tree order that the `purego` reference also uses,
so all paths agree **by construction**. Separately named `Fast*` operations document their
deviation and ULP bound.

**Why.** vek's worst bug is that its SIMD body and scalar tail disagree on NaN, so results change
with slice length ([vek#11](https://github.com/viterin/vek/issues/11)). Users hit this in
production long before reading the docs.

**Concretely.** Never `-Ofast` or `-ffast-math`; use `-ffp-contract=fast` and per-kernel
`#pragma clang fp`. `Sum` is identical everywhere; `FastSum` reassociates with a documented bound.
`Exp` is 1.0 ULP (SLEEF `u10`); `FastExp` is 3.5 ULP (SLEEF `u35`).

### D9 — All six goat targets plus SVE2, both op domains

**Decision.** amd64 (sse2/avx2/avx512), arm64 (neon/**sve2**), riscv64 (rvv), s390x (vx/vxe),
loong64 (lsx/lasx), ppc64le (vsx). Numeric and bytes/bits domains developed in parallel.

**Why affordable.** D2 makes each target a build-matrix row, not new source. The real cost is
**testing, not authoring**, and it is front-loaded: the full matrix is proven on a *single* kernel
before the op count multiplies it.

**Known risks and their handling** — see below.

### D10 — Codegen in a nested module

**Decision.** `tools/go.mod` holds the forked goat and the build drivers, exactly as
[segmentio/asm](https://github.com/segmentio/asm) puts avo in `build/go.mod`.

**Why.** Consumers run `go get` and inherit nothing — no clang, no goat, no avo. Generated `.s`
and stubs are committed.

## Risk register

| Risk | Severity | Handling |
|---|---|---|
| goat's *"potentially BUGGY code generation"* | High | Fork it. D7's gate-vs-emission and BP checks are the mechanical safety net; the differential harness catches semantic errors. go-highway runs it in production. |
| **ppc64le TOC relocations** (`R_PPC64_TOC16_HA/LO` + undefined `.TOC.`) — the only target with a non-trivial relocation model | Medium | Scheduled last within the matrix phase. goat already passes `-ffixed-r0 -ffixed-r30` for ppc64le, so prior art exists. **Acceptable to ship ppc64le at a reduced tier, or scalar-only, if TOC proves intractable** — it must not block the other five. |
| Scalable targets (SVE2, RVV) have runtime-variable vector length | Medium | One shared multi-VL harness, built once, used by both. Validate at VL = 128/256/512/1024. VLA bugs surface *only* when VL changes. |
| qemu matrix time (6 arches × tiers × VLs) | Low | Full matrix nightly; per-PR runs host-native amd64 tiers plus one arm64 lane. |
| Both op domains at once doubles surface per phase | Low | The two domains are independent and share all infrastructure; they proceed in parallel rather than in sequence. |
| Raw-encoded (`WORD $0x…`) bodies are opaque to `go vet` asmdecl and to Go's disassembler | Medium | goat emits real Plan 9 branches to real labels (so control flow stays relocatable) and keeps the disassembly in trailing comments. asmdecl still checks the *signature*, which is what it is for. D7 covers the body. |
| avo is in caretaker mode, if we need it as an escape hatch | Low | Pin the version; the Honeycomb fork is where arm64 work is happening. It is an escape hatch, not a dependency. |

## What this library owns that nothing else does

1. **arm64 SVE/SVE2** — deferred upstream with no design, unreachable via Go mnemonics, and the
   highest-value ISA on Graviton 3/4 and Neoverse V1/V2.
2. **A real numeric kernel set on arm64 at all** — gonum has *zero* arm64 assembly, and vek says
   it never will.
3. **riscv64 RVV, s390x VX, loong64, ppc64le** — no upstream intrinsics planned.
4. **Vectorized transcendentals in Go** — SLEEF removed its Go support and no port exists.
5. **Works for every consumer** — no `GOEXPERIMENT`, plain `go get`.

## Non-goals

- A portable vector type. Go 1.27's `simd` package is that, upstream owns it, and it can inline —
  we cannot compete and should not try. See D5.
- wasm. Go's assembler has zero SIMD128 opcodes and no assembler path will ever exist; Go 1.27
  intrinsics are the only route, which contradicts D1.
- cgo, `//go:linkname` into C symbols, or shipping `.so` files. All three are either unsupported,
  fragile, or defeat the purpose.
