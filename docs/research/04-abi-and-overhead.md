# Go's ABI, assembly call overhead, and the dispatch idiom

*The mechanics every generated kernel must respect, and the measurements that set the
per-operation thresholds.*

## 1. Two ABIs

From `$GOROOT/src/cmd/internal/obj/link.go:848-873` and
`$GOROOT/src/cmd/compile/abi-internal.md` (the authority — `doc/asm.html` never mentions ABI0 or
ABIInternal, which is a documentation gap):

- **ABI0** — the stable, **stack-based** convention. All arguments and results live in the
  caller's frame at `FP` offsets. A plain `TEXT ·Foo(SB), NOSPLIT, $0-40` declaration gets this.
- **ABIInternal** — the **register** convention, **unstable across Go versions**.
  `link.go:852`: *"All Go functions use the internal ABI and the compiler generates wrappers for
  calls to and from other ABIs."*

## 2. Special-purpose registers, per architecture

From `abi-internal.md` §"Architecture specifics":

| Arch | Int args | Float args | Reserved |
|---|---|---|---|
| **amd64** (:385) | `RAX RBX RCX RDI RSI R8 R9 R10 R11` | `X0–X14` | `R14`=goroutine, `X15`=zero, `RDX`=closure ctx, `R15`=GOT temp if dynlink, `RBP`=frame ptr |
| **arm64** (:514) | `R0–R15` | `F0–F15` | `R28`=goroutine, `R27`=assembler scratch, `R26`=closure ctx, `R18`=**reserved, never used**, `R16/R17`=linker trampoline scratch, `F16–F31` permanent scratch |
| loong64 (:636) | `R4–R19` | `F0–F15` | `R22`=goroutine, `R2` reserved |
| ppc64 (:686) | `R3–R10, R14–R17` | — | `R30`=goroutine |
| riscv64 (:785) | — | — | `X27`=goroutine |
| s390x (:834) | — | — | `R13`=goroutine, `R14`=link register |

### The amd64 R14 question — settled

Clang **rejects `-ffixed-r14`** on x86 and does emit `r14`. This turns out to be fine.
`abi-internal.md:421-425`:

> *"These register meanings are compatible with Go's stack-based calling convention except for
> R14 and X15, which will have to be restored on transitions from ABI0 code to ABIInternal code.
> **In ABI0, these are undefined, so transitions from ABIInternal to ABI0 can ignore these
> registers.**"*

**ABI0 assembly may clobber `R14` and `X15` freely.** 28 stdlib `*_amd64.s` files already do.

### arm64 — constrain clang

`-ffixed-x28 -ffixed-x27 -ffixed-x18` are all accepted by clang and effective (measured: `x18`
usage drops 2 → 0). `x27`/`x28` are *"reserved by the compiler and linker"* (`doc/asm.html:850`);
`x18` is the OS platform register.

**arm64 FPCR bonus:** `abi-internal.md` states FPCR is fixed at calls (DN=FZ=RC=0, `NEP=0`
*"scalar operations do not affect higher elements in vector registers"*) explicitly *"to allow Go
functions to use floating-point and vector (SIMD) operations without modifying or saving the
FPCR."* Nothing to do here.

### `BP` must be preserved

`RBP` is the frame pointer and is callee-save in the Go ABI. Clobbering it without restore breaks
stack unwinding, tracebacks, profiling, and GC scanning — non-deterministically, far from the
cause. This is [kelindar/simd#5](https://github.com/kelindar/simd/issues/5), root cause
[avo#156](https://github.com/mmcloughlin/avo/issues/156). **The generator must verify this.**

## 3. What a Go → assembly call actually costs

Three components:

1. **The ABI wrapper thunk.** A plain `TEXT ·Foo(SB)` is ABI0, so when Go code calls it the
   compiler synthesizes a wrapper (`cmd/compile/internal/ssagen/abi.go`, flag `ABIWRAPPER = 4096`
   in `$GOROOT/src/runtime/textflag.h`) that spills register-passed arguments to the frame, jumps
   to the ABI0 body, and reloads results. **A real `CALL` plus N stores and M loads that would
   not exist for an inlined Go function or a compiler intrinsic.**
2. **No inlining, ever.** Assembly is opaque to the inliner and to escape analysis, and acts as
   an optimization barrier — the compiler must spill anything live across the call.
3. **`VZEROUPPER`** on AVX paths, 4–35 cycles depending on CPU.

### Measurements

| Source | Measurement |
|---|---|
| [mmcloughlin](https://mmcloughlin.com/posts/geohash-assembly) | `BenchmarkNoopAsm` = **1.40 ns/op**; ~0.92 ns of pure call overhead above that baseline |
| [Lemire](https://lemire.me/blog/2016/12/21/performance-overhead-when-calling-assembly-from-go/) (Skylake @3.4 GHz) | pure-Go de Bruijn `tzcnt` = 3.55 cycles/call; **assembly `TZCNT` (a 3-cycle instruction) = 11.5 cycles/call — ~2× *slower* than pure Go** |
| [coregx/coregex#120](https://github.com/coregx/coregex/issues/120) | **50–65 cycles per boundary crossing** (call + `VZEROUPPER` + register save/restore) in a real SIMD prefilter; 96% of runtime; **their AVX2 build was 4× slower than their SSSE3 build** because VZEROUPPER fired per call |

**Baseline: ~1.4 ns / ~8 cycles minimum per Go→asm call, before the kernel does any work.**

### Why intrinsics beat assembly at small n

[Callista](https://callistaenterprise.se/blogg/teknik/2025/10/20/trying-out-go-simd-support/),
adding two 32-element `uint8` slices:

| Approach | ns/op |
|---|---|
| Plain loop, not inlined | 17.48 |
| Plain loop, inlined | 8.843 |
| **avo-generated assembly** (can never be inlined) | **2.765** |
| `simd` package, not inlined | 1.941 |
| **`simd` package, inlined** | **0.4811** |

At n=32 the inlinable intrinsic is **5.7× faster than the best possible assembly** — because the
boundary *is* the workload at that size.

### Why assembly wins at large n

The boundary is a **fixed** cost, so it amortizes:

| n | Boundary as % of a simple f32 kernel | Winner |
|---|---|---|
| 32 | ~85% | Intrinsics, by 5.7× |
| 128 | ~20% | roughly even |
| 4K | ~1% | Assembly |
| 64K+ | ~0.01% | Assembly, and the gap grows |

Once amortized, the loop body decides, and there LLVM beats Go's backend structurally:

1. **Go's compiler does not unroll loops.** clang unrolled our test kernel 4× with
   `vfmadd213ps` automatically.
2. **Go has no comparable instruction scheduler**, no software pipelining, no cost model.
3. **Go inserts bounds checks** in the vector loop; eliminating them is worth a measured **+14%**
   ([marselester](https://marselester.com/go-archsimd-preview.html)). `__restrict` + raw pointers
   in C have none.
4. `archsimd` is *intrinsics*, not auto-vectorization — you write the vectorization either way,
   but in C, LLVM also does unrolling, scheduling and tail handling.

**Design consequence: the public unit of work is a whole loop over a whole slice, never a vector
operation — plus a benchmarked per-op threshold below which no call happens at all.**

## 4. Directives and flags

### `//go:noescape` — mandatory, not an optimization

From `$GOROOT/src/cmd/compile/doc.go:216-223`. Must precede a bodyless declaration. Asserts that
pointer arguments do not escape to the heap or into results, letting escape analysis keep caller
slices and backing arrays **stack-allocated**.

**Without it, every `[]T` handed to assembly is force-heap-allocated.** Every declaration in
`index_native.go`, `indexbyte_native.go`, `compare_native.go`, `sha256block_*.go`,
`crc32_amd64.go` carries it. It is an **unchecked promise** — get it wrong and you get memory
corruption. (goat emits it automatically.)

### `NOSPLIT`, `NOFRAME`, `RODATA`, `NOPTR`

From `$GOROOT/src/runtime/textflag.h`:

| Flag | Value | Meaning |
|---|---|---|
| `NOSPLIT` | 4 | Omit the stack-growth preamble (`CMP SP, stackguard; CALL morestack`). Saves ~2–3 instructions and a branch per call. **Required** for leaf routines running where growing the stack is illegal. Budget: the frame must fit the nosplit limit (~800 bytes) *counting the whole nosplit call chain*. |
| `NOFRAME` | 512 | No frame setup at all. Only legal with `$0` frame size. |
| `RODATA` | 8 | For `DATA`/`GLOBL` constant tables (shuffle masks etc.). |
| `NOPTR` | 16 | Same — tells the GC there are no pointers in there. |
| `ABIWRAPPER` | 4096 | Set by the compiler on synthesized wrappers. |

Our extracted kernels need no stack adjustment (measured), so they fit
`TEXT ·kernel(SB), NOSPLIT|NOFRAME, $0-N`.

### Skipping the wrapper: `<ABIInternal>`

`TEXT ·IndexByte<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-40` — see
`$GOROOT/src/internal/bytealg/indexbyte_s390x.s:9`. You take arguments in registers and pay zero
thunk cost, but you have hand-coded an **explicitly unstable** ABI: you maintain the register
mapping yourself and it can break on any Go release. **Use only where a benchmark shows the thunk
mattering.** The stdlib does this in exactly the hot paths where it does.

## 5. The dispatch idiom — copy `internal/bytealg`

The stdlib's four-file pattern (`$GOROOT/src/internal/bytealg/`) is the template:

1. **`*_native.go`** — allowlist build tag, bodyless declarations, `//go:noescape` on each.
   `index_native.go`: `//go:build amd64 || arm64 || loong64 || s390x || ppc64le || ppc64`.
2. **`*_generic.go`** — the exactly-negated tag, pure-Go body. Note `indexbyte_generic.go` also
   carries a `(!amd64 || plan9)` carve-out, because SSE is illegal in Plan 9 note handlers.
3. **`*_<arch>.go`** — per-arch tuning constants set in `init()` from `internal/cpu`:
   - `index_amd64.go`: `if cpu.X86.HasAVX2 { MaxLen = 63 } else { MaxLen = 31 }`
   - `index_arm64.go`: unconditional `MaxLen = 32` (NEON assumed — it's mandatory on AArch64)
   - `index_loong64.go`: `if cpu.Loong64.HasLASX || cpu.Loong64.HasLSX { MaxLen = 64 }`
   - `index_s390x.go`: `if cpu.S390X.HasVX { MaxLen = 64 }`
4. **`bytealg.go`** — exports `unsafe.Offsetof` constants so **the assembly itself branches on
   CPU features** without a Go call:

   ```go
   offsetX86HasAVX2     = unsafe.Offsetof(cpu.X86.HasAVX2)
   offsetS390xHasVX     = unsafe.Offsetof(cpu.S390X.HasVX)
   offsetLOONG64HasLASX = unsafe.Offsetof(cpu.Loong64.HasLASX)
   offsetRISCV64HasV    = unsafe.Offsetof(cpu.RISCV64.HasV)
   offsetPPC64HasPOWER9 = unsafe.Offsetof(cpu.PPC64.IsPOWER9)
   ```

   The `.s` files `#include "go_asm.h"` and do `CMPB ·offsetX86HasAVX2(SB), $1`.

`MaxBruteForce` being a `const` lets `const haveFastIndex = bytealg.MaxBruteForce > 0` in
`$GOROOT/src/bytes/bytes.go:140` be dead-code-eliminated on unsupported arches.

**`bytes` and `strings` contain no `.s` files at all** — they call into `bytealg`. That is the
layering to copy.

### Other stdlib patterns worth knowing

- **`crypto/internal/fips140/*`** — a `block` shim per arch with `impl.Register`:
  `var useAVX2 = cpu.X86HasAVX && cpu.X86HasAVX2 && cpu.X86HasBMI2`,
  `impl.Register("sha256","AVX2",&useAVX2)`, then dispatch `blockSHANI → blockAVX2 →
  blockGeneric`. Every build tag carries a `|| purego` escape valve. Where no runtime bit exists
  (`ppc64x`), it uses a **GODEBUG knob** `godebug.Value("#ppc64sha2") != "off"` read once at init.
- **`hash/crc32`** — an arch-interface of three funcs: `archAvailableIEEE() bool`,
  `archInitIEEE()`, `archUpdateIEEE(crc, p) uint32`. `crc32_otherarch.go` returns `false`.
- **`math/big`** — `arithvec_s390x.go`: `var hasVX = cpu.S390X.HasVX` + declarations; the `.s`
  branches on it.
- **Generation** — avo in `_asm/` subdirs with their own `go.mod` (`crypto/md5/_asm`,
  `crypto/sha1/_asm`, `crypto/internal/fips140/{aes,sha256,sha512,sha3,bigmod,nistec}/_asm`), and
  **text/template emitting Plan 9 directly** for arm64 where avo doesn't reach
  (`crypto/internal/fips140/aes/ctr_arm64_gen.go`, `//go:build ignore`).

## 6. Rules this library adopts

1. `//go:noescape` on every assembly declaration. Non-negotiable.
2. `NOSPLIT|NOFRAME, $0-N` for extracted kernels; constant tables `RODATA|NOPTR`.
3. Preserve `BP` on amd64; `-ffixed-x28,x27,x18` on arm64. Verified by the generator, not by hope.
4. `R14`/`X15` are free to clobber on amd64 (per `abi-internal.md:424`).
5. Feature detection via `golang.org/x/sys/cpu` — `internal/cpu` is unavailable to third-party
   modules, and only `x/sys/cpu` exposes `ARM64.HasSVE`/`HasSVE2`.
6. **No package-level `var` may execute a SIMD instruction** — go-highway#69 was exactly that:
   AVX2 broadcast constants in variable initialization, running before the dispatch check.
7. `<ABIInternal>` only where a benchmark shows the thunk mattering.
8. Per-op, per-type element thresholds — benchmarked, not guessed — below which the dispatcher
   runs an inlined scalar Go loop and never crosses the boundary.

## Sources

- `$GOROOT/src/cmd/compile/abi-internal.md`, `cmd/internal/obj/link.go`, `runtime/textflag.h`,
  `cmd/compile/doc.go`, `internal/bytealg/*`, `crypto/internal/fips140/sha256/*`, `hash/crc32/*`
- [mmcloughlin, Geohash in Golang Assembly](https://mmcloughlin.com/posts/geohash-assembly)
- [Lemire, Performance overhead when calling assembly from Go](https://lemire.me/blog/2016/12/21/performance-overhead-when-calling-assembly-from-go/)
- [coregx/coregex#120](https://github.com/coregx/coregex/issues/120) · [golang/go#77647](https://github.com/golang/go/issues/77647)
- [Callista, Trying out Go SIMD support](https://callistaenterprise.se/blogg/teknik/2025/10/20/trying-out-go-simd-support/)
- [marselester, archsimd preview](https://marselester.com/go-archsimd-preview.html)
