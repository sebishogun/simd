# Existing Go SIMD libraries: what they do, and what they got wrong

*Surveyed July 2026. The purpose of this document is not scorekeeping — it is to enumerate the
specific failure modes this library must be designed against. Those are in §3.*

## 1. The libraries

### viterin/vek — the one this project was started to improve on

[github.com/viterin/vek](https://github.com/viterin/vek) · 203★ · MIT · last push 2025-09-06 ·
**19 commits total, with a 2-year gap between Aug 2023 and Aug 2025.**

**API.** Whole-slice functions in three variants per op: `Add(x,y)`, `Add_Inplace(x,y)`,
`Add_Into(dst,x,y)`. Two packages — `vek` (float64) and `vek32` (float32) — **duplicated rather
than generic.** Ops cover arithmetic, aggregates (Sum/CumSum/Prod/Mean/Median/Quantile), distance
(Dot/Norm/Cosine/Manhattan), MatMul, special functions (Sqrt/Pow/Round/Floor/Ceil/Sin/Cos/Exp/Log),
comparisons, and `[]bool` boolean ops.

**Toolchain** (from its own `asm/README.md`): hand-written C++ against
[vectorclass/VCL](https://github.com/vectorclass/version2) → clang `-Ofast -mfma -mavx2` → a
custom Python script (`asm/asm2avo.py`) transliterating AT&T asm into avo builder calls → **then
hand-patch the output.** Verbatim: *"Fix potential issues with the output. Data sections need to
be added manually and moves changed to unaligned access."* Requires a local Compiler Explorer
instance. **Not reproducible.**

**Coverage: amd64 + AVX2 + FMA only.** One asm directory, `asm/_avx2/`. No AVX-512, no SSE tier,
no arm64. The gate is:

```go
var UseAVX2 bool = cpu.X86.HasAVX2 && cpu.X86.HasFMA && runtime.GOOS != "darwin"
```

**Acceleration is hard-disabled on macOS** — even on Intel Macs with AVX2 — because the
maintainer has no machine to test on. On arm64 vek is **pure Go**, and
[#12](https://github.com/viterin/vek/issues/12) confirms *"there are no plans to add SIMD
acceleration."*

**Its limitations, concretely:**

| Problem | Evidence |
|---|---|
| **`-Ofast` ⇒ not IEEE-754.** NaN/Inf behavior undefined; precision differs from `math`. | README caveat, added only in the **final commit (2025-09-06)** after a user complained |
| **Accelerated and non-accelerated paths give different answers for NaN.** `Eq`/`Neq` disagree depending on `SetAcceleration()`. | [#11](https://github.com/viterin/vek/issues/11) — failures occur only when NaN lands in the **scalar tail block**: the vector body and the tail have *different semantics* |
| Tail-handling correctness bug | commit `fix: Any(x) where len(x) > 256 && len(x)%32 > 0` (Aug 2025) — classic remainder-loop bug |
| **Slower than plain Go on small slices** | [#6](https://github.com/viterin/vek/issues/6): n=4 → SIMD 3.427 ns vs Go 2.174 ns. Crossover ≈ n=16. |
| Global mutable state | `vek.SetAcceleration(bool)` is process-wide — not composable, racy |
| Bus factor 1 | 19 commits, 2-year dormancy |

Its dependency [viterin/partial](https://github.com/viterin/partial) (12★, last push 2023-08) is
effectively frozen; vek uses it only for `Median`/`Quantile`.

### gonum — the biggest gap in the Go ecosystem

[gonum.org/v1/gonum](https://github.com/gonum/gonum) · 8,419★ · last push 2026-07-21 · **the only
genuinely maintained library here.**

BLAS kernels live in `internal/asm/f64` and `f32`: `AxpyUnitary/Inc/IncTo/UnitaryTo`,
`DotUnitary/DotInc`, `DdotUnitary`, `ScalUnitary/Inc`, `Ger`, `GemvN`, `GemvT`, `L1Norm`,
`L2Norm(Inc/Dist)`, `LinfNorm`, `Sum`, `Add`, `AddConst`, `Div`, `CumSum`, `CumProd`, `AbsSum`,
plus `c64`/`c128`. Stub declarations in `stubs_amd64.go` (`//go:build !noasm && !gccgo && !safe`)
mirrored by `stubs_noasm.go`.

**How it's implemented: hand-written Plan 9 assembly.** No avo, no c2goasm. `dot_amd64.s` credits
loop unrolling copied from `math/big/arith_amd64.s`.

**Coverage: amd64 only, and SSE2-only at that.** Mnemonics actually present:

```
f64/dot_amd64.s        : ADDPD MULPD MOVUPD UNPCKHPD MOVHPD MOVLPD
f64/axpyunitary_amd64.s: ADDPD MULPD MOVUPS SHUFPD
f32/dotunitary_amd64.s : ADDPS MULPS HADDPS PXOR
```

**Zero `V`-prefixed instructions in the entire repository.** `VMULPD`: 0 hits. `VFMADD`: 0 hits.
**No AVX, no AVX2, no FMA, no AVX-512. And zero arm64 assembly anywhere** — every Apple Silicon,
Graviton, and Ampere user runs gonum 100% scalar.

Stuck at 128-bit SSE2 (2 float64 lanes) since 2015: ~2× theoretical, against 8× for AVX-512.
**This is the single largest opportunity for the present library.**

### ajroetker/go-highway — closest prior art

[github.com/ajroetker/go-highway](https://github.com/ajroetker/go-highway) · 115★ · created
2025-12-24, pushed 2026-07-08. *"Write SIMD once, run everywhere… Like Google Highway, but for
Go."* No cgo.

Hybrid backend: amd64 AVX2/AVX-512 via Go 1.26 `simd/archsimd`; arm64 NEON/SVE/SME via `hwy/asm`
with **GoAT-generated assembly**. Its `cmd/hwygen` is a full generator with its own IR, a C
emitter, and target modes. It vendors GoAT at `hwy/goat` and extends its type support to
`int32_t, uint32_t, uint64_t, float16_t`.

Headline claim: **117× for GELU F32 at n=1024 on M4 Max**, comparing bulk-`asm` mode against
per-vector `GoSimd` calls. That is not SIMD width — that is the call boundary.

**Its bug history is the checklist for §3:**

- [#68](https://github.com/ajroetker/go-highway/issues/68) — AVX2 codegen emitted
  **EVEX-prefixed (AVX-512) instructions** into the `_avx2.gen.go` path → **SIGILL on AMD EPYC
  7763**. Affected *all* transcendentals because they shared a generator.
- [#69](https://github.com/ajroetker/go-highway/issues/69) — **AVX2 broadcast constants executed
  before CPU detection** → SIGILL on non-AVX2 CPUs. Package-level `var` initialization ran AVX2
  before the dispatch check.
- [#67](https://github.com/ajroetker/go-highway/issues/67) — illegal instruction `0xc000001d` on
  Windows amd64.
- [#72](https://github.com/ajroetker/go-highway/issues/72) (open) — StreamVByte NEON kernels
  return zeros for sub-4-byte encodings on linux/arm64. Tail handling again.

### The rest

| Library | ★ | Last push | Impl | Arches | Killer limitation |
|---|---|---|---|---|---|
| [kelindar/simd](https://github.com/kelindar/simd) | 34 | 2026-03 | C → clang autovec → [gocc](https://github.com/kelindar/gocc) | amd64 AVX2, arm64 NEON, Apple | **[#5](https://github.com/kelindar/simd/issues/5): generated functions clobber `BP` without restoring it.** BP is callee-save in the Go ABI — breaks stack unwinding, tracebacks, profiling, GC scanning. Root cause [avo#156](https://github.com/mmcloughlin/avo/issues/156). Only 7 ops; `Div` isn't actually accelerated (≈1.2×). Codegen script is **commented out** ([#7](https://github.com/kelindar/simd/issues/7)) so the asm is not reproducible. |
| [pehringer/simd](https://github.com/pehringer/simd) | 136 | 2025-07 | **~100 hand-written `.s` files** | amd64 SSE→AVX512VL, arm64 NEON | The clearest empirical argument against hand-writing ops×types×ISAs: **arm64 NEON is missing Div, Max, Min, Xor entirely**; `DivInt32`/`DivInt64` exist on no arch; `MaxInt64`/`MinInt64`/`MulInt64` require AVX-512VL or nothing. No reductions, no transcendentals, no masks. Good idea worth stealing: QEMU cross-testing via `make test_arm64`. |
| [segmentio/asm](https://github.com/segmentio/asm) | 926 | 2022 (real; later pushes are dependabot) | avo (amd64) + **hand-written Plan 9** (arm64), in a **separate nested Go module** | amd64 SSE→AVX-512, arm64 NEON | Dead upstream — issues unanswered since 2022. Byte algorithms only, no float math. Most functions unexported ([#91](https://github.com/segmentio/asm/issues/91)); breaks `go vendor` ([#73](https://github.com/segmentio/asm/issues/73)). **Steal the two-module design** (`build/go.mod` holds avo) so consumers never inherit the codegen deps. |
| [tphakala/simd](https://github.com/tphakala/simd) | 17 | 2026-07 | hand asm, runtime tiers | SSE2→AVX-512, NEON+FP16 | Broadest type coverage seen anywhere (float64/32/**16**, int32/16/8, complex128/64) and domain kernels (FFT/STFT, convolution, resampling). **Steal `SIMD_DISABLE` env var to mask CPU features** — excellent for tier benchmarking and dodging AVX-512 downclocking. But 14 open issues incl. 2 live SIGILL gate bugs: [#197](https://github.com/tphakala/simd/issues/197) (`f64.Reciprocal` uses AVX2 `VBROADCASTSD` under an AVX-no-FMA gate), [#196](https://github.com/tphakala/simd/issues/196). |
| [alivanz/go-simd](https://github.com/alivanz/go-simd) | 202 | 2025-05 | C intrinsics + **`//go:linkname` into C symbols** to bypass cgo's trampoline | **ARM NEON only** | *"No cgo overhead," not "no cgo"* — still needs a C toolchain at build time. `//go:linkname` into C symbols is fragile and explicitly unsupported by the Go team. 1:1 with ACLE intrinsic names (`VaddlS8`), no slice-level API at all ([#4](https://github.com/alivanz/go-simd/issues/4)). **Do not copy.** |
| [gorgonia/vecf64](https://github.com/gorgonia/vecf64), [vecf32](https://github.com/gorgonia/vecf32) | 24/13 | 2023-06 | hand `.s` per op per ISA | amd64 SSE/AVX | **SIMD selected by build tag (`-tags avx`), not runtime dispatch** — the default `go build` is pure Go, and any binary you ship targets exactly one ISA. Parent [gorgonia](https://github.com/gorgonia/gorgonia) (5.9k★) last pushed 2024-08; de facto abandoned. |
| [ollama](https://github.com/ollama/ollama) | 177k | 2026-07 | **cgo → ggml/llama.cpp**, per-ISA `.so`, `dlopen` at runtime | everything | The industrial precedent for "one binary, many ISAs" — but it is a cgo/CMake answer, not a Go one. Kernels run in a separate subprocess over HTTP. |

Also worth knowing: `klauspost/compress`, `zeebo/xxh3`, `minio/simdjson-go` (2,037★, c2goasm,
amd64 only), `bytedance/sonic` (9,557★, the most sophisticated Go asm project alive),
`cloudwego/base64x`, `grailbio/base/simd`, `templexxx/reedsolomon`.

## 2. Go's own SIMD support

| Release | Date | What landed |
|---|---|---|
| Go 1.26 | Feb 2026 | `simd/archsimd` under `GOEXPERIMENT=simd`, **amd64 only**, 128/256/512-bit fixed-width types |
| Go 1.27 | rc2 now; GA ~Aug 2026 | **arm64 NEON + wasm SIMD128** added to `archsimd`; **amd64 API breaking-changed**; **new portable size-agnostic `simd` package** ([#78902](https://github.com/golang/go/issues/78902)) |

The portable `simd` API: `Float32s` etc. with a **runtime** `.Len()`, `LoadFloat32sPart` /
`StorePart` for tails, `VectorBitSize()`, `Emulated()`, `ToArch()`/`FromArch()` escape hatches to
`archsimd`, and `GODEBUG=simd=<size>` to override. All vectors in a program share one bit length,
fixed for the process. **The compiler does multi-versioning and dispatch internally, so intrinsics
inline** — which is the whole point.

[#78979](https://github.com/golang/go/issues/78979) proposes dropping the GOEXPERIMENT gate on
amd64. [#76175](https://github.com/golang/go/issues/76175) (Austin Clements, **on hold**) proposes
a `//cpu:requires <feature>` directive plus a **`cpu` vet check** doing flow analysis to prove
every intrinsic call is dominated by a successful feature check — motivated by the observation
that *AVX-512 alone has 21 feature flags*. That is the upstream answer to every SIGILL bug above.

**Scalable vectors (SVE, SVE2, RVV) are explicitly future work with no design yet.** archsimd
will not cover them for years.

**The catch for a library:** `GOEXPERIMENT=simd` is a build-time env var your **consumers** must
set. You cannot ship a library that silently requires it.

## 3. The five mistakes to design against

Every library above made at least one.

### 3.1 The CPU-feature gate does not match the instructions actually emitted → SIGILL

Four live examples across two libraries (go-highway #68, #69, #67; tphakala #197, #196). This is
the most common and most damaging failure mode, it is invisible on the developer's machine, and
it only manifests on a user's older CPU in production.

**Mitigation:** a CI step that **disassembles every generated `.s` and asserts no instruction
requires a feature above the tier the file is gated on.** Mechanical, not vigilance-based.

### 3.2 Go ABI violations from generated assembly

`BP` clobbered without restore (kelindar/simd #5, root cause avo #156). Breaks stack unwinding,
tracebacks, profiling, and GC scanning — and does so *non-deterministically*, far from the cause.

**Mitigation:** an explicit BP-preservation check in the generator, plus `go vet` asmdecl.

### 3.3 Different numerical semantics between the SIMD body, the scalar tail, and the pure-Go fallback

vek #11 is canonical: results change depending on whether NaN lands in the vector block or the
remainder block. vek's `Any` bug and go-highway #72 are the same seam.

**Mitigation:** a hard contract — bit-identical across all tiers including NaN/Inf/±0/denormals —
and differential tests that specifically hammer that seam: every length in `0..4*VL+1`, every
start offset `0..63`, adversarial values.

### 3.4 Hand-writing ops × types × ISAs until coverage becomes swiss cheese

pehringer/simd is the proof: ~100 hand-written `.s` files and arm64 NEON still has no Div, Max,
Min, or Xor.

**Mitigation:** one autovectorized C source per kernel; each target is a build-matrix row.

### 3.5 Crossing the Go/asm boundary per tiny op instead of per slice

[coregx/coregex#120](https://github.com/coregx/coregex/issues/120): a Teddy SIMD regex prefilter
processing 16 bytes per assembly call, ~375,000 calls per 6 MB scan. Measured **50–65 cycles per
boundary crossing** (call + `VZEROUPPER` + register save/restore); profiling attributed **96% of
runtime** to it; **8.6× slower than Rust overall** (32 ms vs 3.7 ms). Their conclusion verbatim:
*"100% of the gap is in SIMD function call overhead."*

The killer detail: **their AVX2 implementation was 4× slower than their SSSE3 one**, because
`VZEROUPPER` (4–35 cycles depending on CPU) fires on every call and swamps AVX2's 2× throughput.

**Mitigation:** the public unit of work is a whole loop over a whole slice, never a vector
operation. Plus a benchmarked per-op threshold below which no call happens at all.

## 4. Where the crossover is

vek [#6](https://github.com/viterin/vek/issues/6), AMD Ryzen 7 7840HS, float32 add:

| n | SIMD ns/op | pure Go ns/op | winner |
|---|---|---|---|
| 4 | 3.427 | **2.174** | Go |
| 16 | **7.580** | 8.581 | ~tie |
| 32 | **4.937** | 15.82 | SIMD 3.2× |
| 64 | **8.695** | 33.63 | SIMD 3.9× |

pehringer/simd (Ryzen 7 7840U): n=100 → 5.06×; n=300 → 9.59×; n=800 → 10.31×. Even at n=100 you
get only half the asymptotic win. On M1 Pro, ~4× flat.

**Rule: below ~16 elements do not cross the boundary; 16–64 is marginal; ≥128 is near-asymptotic.**
Thresholds must be per-op and per-type, **benchmarked, not guessed**.

## 5. API shape: fixed-width vs scalable vs whole-slice

| | Fixed-width `Vec<T,N>` | Scalable | Whole-slice kernels |
|---|---|---|---|
| Lane count known at | compile time | runtime | n/a (hidden) |
| **SVE / RVV compatible** | **No — fundamental** | Yes | **Yes** |
| Tail handling | manual, per caller — bug farm | masks / partial load | once, inside the kernel |
| Runtime ISA dispatch | needs external machinery | natural | natural |
| Fusion / composability | best (user fuses in registers) | good | **worst — a memory pass per op** |
| Call-overhead exposure | none *if inlined*, fatal in asm | none *if inlined*, fatal in asm | amortized over n |

**Any type whose name or generic parameter encodes a lane count is unimplementable where the
vector length is a boot-time property.** The evidence is unusually strong:

- **Highway's own [comparison doc](https://github.com/google/highway/blob/master/g3doc/std_simd_comparison.md)**:
  *"A class wrapper is incompatible with the scalable vectors introduced by Arm SVE and
  standardized by RISC-V V"* — compilers **forbid wrapping sizeless types in classes**. Highway's
  answer is a zero-sized **tag** (`ScalableTag<T>`) separate from the vector, with `Lanes(d)`
  returning a runtime value on SVE/RVV.
- **C++26 `std::simd` degrades silently**: on aarch64 with SVE it compiles but emits fixed-width
  128-bit NEON with manual unrolling — assembly ~3× longer, **SVE completely unused**. And
  `native_simd<int>` on an AVX2 machine returns size **4, not 8**, because widening would break
  ABI.
- **Rust `core::simd`** is still nightly after five years and does not support SVE/RVV; Rust's
  actual SVE work is a *separate* `#[repr(scalable)]` type.
- **NumPy [NEP 54](https://numpy.org/neps/nep-0054-simd-cpp-highway.html)** moved to Highway with
  the stated motivation *"the need to add support for sizeless SIMD instructions like ARM's SVE
  and RISC-V's RVV."* The world's largest whole-array numerics library concluded its fixed-width
  intrinsic layer could not reach them.
- **Go split the same way**: `archsimd` (fixed, `Float64x8`) + portable `simd` (scalable,
  `Float64s`, runtime `.Len()`).

In assembly this is moot anyway — a public vector type costs a call boundary per operation and
would lose to scalar Go. **Whole-slice kernels, with fused combinators to mitigate the
memory-pass weakness.**

## Sources

[vek](https://github.com/viterin/vek) · [vek asm README](https://github.com/viterin/vek/blob/master/asm/README.md) · [vek#6](https://github.com/viterin/vek/issues/6) · [vek#11](https://github.com/viterin/vek/issues/11) · [vek#12](https://github.com/viterin/vek/issues/12) · [gonum](https://github.com/gonum/gonum) · [gonum internal/asm/f64](https://github.com/gonum/gonum/tree/master/internal/asm/f64) · [go-highway](https://github.com/ajroetker/go-highway) · [kelindar/simd](https://github.com/kelindar/simd) · [avo#156](https://github.com/mmcloughlin/avo/issues/156) · [pehringer/simd](https://github.com/pehringer/simd) · [segmentio/asm](https://github.com/segmentio/asm) · [tphakala/simd](https://github.com/tphakala/simd) · [alivanz/go-simd](https://github.com/alivanz/go-simd) · [gorgonia/vecf64](https://github.com/gorgonia/vecf64) · [coregx/coregex#120](https://github.com/coregx/coregex/issues/120) · [golang/go#73787](https://github.com/golang/go/issues/73787) · [#78902](https://github.com/golang/go/issues/78902) · [#78979](https://github.com/golang/go/issues/78979) · [#76175](https://github.com/golang/go/issues/76175) · [Go 1.27 release notes](https://go.dev/doc/go1.27) · [Highway std::simd comparison](https://github.com/google/highway/blob/master/g3doc/std_simd_comparison.md) · [NumPy NEP 54](https://numpy.org/neps/nep-0054-simd-cpp-highway.html) · [C++26 SIMD critique](https://lucisqr.substack.com/p/c26-shipped-a-simd-library-nobody)
