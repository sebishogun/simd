# simd

SIMD-accelerated slice operations for Go, on every architecture that has a
vector unit, **without cgo**.

```go
import "github.com/sebishogun/simd"

simd.Add(a, b)                    // a[i] += b[i]     — in place, no allocation
simd.AddScaled(a, b, 0.5)         // a[i] += b[i]*0.5 — one pass over memory
total := simd.Sum(a)
i     := simd.Index(line, ",")    // string or []byte, no copy
n     := simd.Base64Encode(dst, src)
```

Ordinary functions on ordinary slices. No vector type, no lane count, no target
selection, nothing to initialize, and nothing that allocates.

---

## Status

**Early, and honest about it.** There is no tagged release and the API changed
twice in the last week. What is in place:

- **298 exported functions** over ten element types, plus complex, bytes, text
  and the narrow float formats.
- **4,721 generated kernels** across nine targets — amd64 sse2/avx2/avx512,
  arm64 neon/sve2, riscv64 rvv, s390x vx, loong64 lasx, ppc64le vsx.
- Every architecture is **executed**, under emulation, on every change.
- The portable Go implementation is always there. A kernel that could not be
  generated for a target is not a hole; it is a slower path.

**The one thing to know before depending on this:** nothing outside amd64 has
ever run on real hardware. Everything else is qemu, which proves semantics and
proves nothing about timing.

---

## Why another one

The existing Go options leave most machines on the table.

| | ISAs covered | arm64 |
|---|---|---|
| [gonum](https://github.com/gonum/gonum) `internal/asm` | **SSE2 only** — zero `V*` instructions in the whole repo | **none** |
| [viterin/vek](https://github.com/viterin/vek) | AVX2 only, disabled entirely on macOS | pure Go, [and never will be](https://github.com/viterin/vek/issues/12) |

And there is a structural reason nobody covers the rest: **Go's assembler
cannot spell SVE2 or RVV.** It has no `Z` or `P` registers at all, and upstream
has deferred scalable vectors with no design. The route taken here — compile C
per target, lift the encoded bytes into Plan 9 assembly — is the only one that
reaches them, and it is why this is the only Go library with arm64 SVE2 and
riscv64 RVV numeric kernels.

---

## What it does

**Elementwise** `Add` `Sub` `Mul` `Div` `Minimum` `Maximum` `Abs` `Neg` `Sqrt`
`Reciprocal` `Reverse` · **Saturating** `SatAdd` `SatSub` · **Scalar** `Scale`
`AddScalar` `SubScalar` `DivScalar` `Clamp` `Fill` `Zero` `Lerp` `AddScaled` ·
**Rounding** `Floor` `Ceil` `Trunc` `Round` `RoundToEven`

**Reductions** `Sum` `Prod` `Dot` `Min` `Max` `MinMax` `ArgMin` `ArgMax` `Norm`
`L1Norm` `SumSquares` `Mean` `Variance` `StdDev` `CosineSimilarity` ·
**Scans** `CumSum` `CumProd` `CumMin` `CumMax` `Diff`

**Transcendental** `Exp` `Exp2` `Expm1` `Log` `Log2` `Log10` `Log1p` `Cbrt`
`Pow` `Hypot` `Sin` `Cos` `Tan` `Asin` `Acos` `Atan` `Atan2` `Sinh` `Cosh`
`Tanh` `Sigmoid` — each with a `Fast` twin

**Comparisons** to `[]bool` masks, `Select`, `All` `Any` `CountTrue`

**Complex** `AddComplex` … `DotComplexConj` `AbsComplexInto` `FromPartsInto`

**Text and bytes**, taking `string` or `[]byte` without copying: `Index`
`LastIndex` `IndexByte` `LastIndexByte` `IndexAny` `IndexNotAny` `Contains`
`HasPrefix` `HasSuffix` `Count` `CountByte` `CountAny` `IndexAll` `TrimAny`
`TrimSpaceASCII` `IsASCII` `ValidUTF8` `EqualFoldASCII` `ToUpperASCII`
`HexEncode` `Base64Encode` `Base64Decode`

**Narrow floats** `Float16ToFloat32Into` `Float32ToFloat16Into`
`BFloat16ToFloat32Into` `Float32ToBFloat16Into`

Element types: `float32` `float64` `int8` `int16` `int32` `int64` `uint8`
`uint16` `uint32` `uint64` `complex64` `complex128`.

---

## Numbers

Measured on a Ryzen AI MAX+ 395 (Zen 5, AVX-512), `-count 6` or more, compared
with `benchstat`. Regenerate with `make bench-check`.

**Against the portable Go build** — integer and saturating arithmetic, geomean
over the whole set: **−86% time, +593% throughput**.

| int8 `SatAdd` | portable | accelerated | |
|---|---|---|---|
| n=256 | 146 ns | 6.6 ns | −95% |
| n=4096 | 2.42 µs | 22.0 ns | **−99.1%** |

**Against `bytes` and `strings`** — the harder comparison, since `bytealg` is
hand-written assembly on four of the six architectures. Geomean **+186%**.

| | vs stdlib |
|---|---|
| `LastIndex` n=4096 | **+8309%** |
| `IndexAny` n=1 MiB | +1084% |
| `Index` n=1 MiB | +623% |
| `IndexAll` n=1 MiB | +135% |
| `ValidUTF8` n=1 MiB | +54% |
| `TrimSpaceASCII` | +29% |

**Against `encoding/base64`:** −42% to −63%, +74% throughput.

**`Fast` against accurate:** `FastSin` −45%, `FastExp` −43%, `FastSigmoid`
−36%, `FastLog` −25%.

Where the standard library is *already* assembly doing the same work —
`bytes.Equal` is `memequal`, `bytealg.Count` popcounts a compare mask — there
is no margin, and this library defers to it rather than pretending otherwise.

---

## The accuracy contract

**Every operation is bit-identical on every instruction set**, including for
NaN payloads, ±Inf, ±0 and denormals. Reductions use a fixed accumulation order
that a 128-bit and a 512-bit machine both reproduce exactly, so a computation
cannot change answer because it moved to a different server.

This costs throughput and is the point. The alternative is what
[vek does](https://github.com/viterin/vek/issues/11): its vectorized body and
its scalar remainder disagree on NaN, so the answer depends on the length of
the input.

Two documented exceptions, both opt-in by name:

- **Transcendentals** guarantee a ULP bound rather than bit identity, because
  the polynomial correct to 1 ULP in float32 is not the one correct to 1 ULP in
  float64. The bound is measured against the standard library and reported, not
  asserted from theory.
- **`Fast*`** promises 3.5 ULP and gives up agreement *between* architectures,
  because it is compiled with fused multiply-add. It does not give up meaning:
  NaN in gives NaN out, the infinities go where IEEE 754 says, and signed zeros
  survive. `-ffast-math` would buy more and is refused.

---

## Coverage, honestly

Kernel counts are not uniform, and the reasons are ABI walls rather than
effort.

| | kernels | |
|---|---|---|
| amd64 (3 tiers) | 1652 | essentially complete |
| arm64 (2 tiers) | 1112 | essentially complete |
| s390x | 614 | **partial** |
| riscv64 | 556 | essentially complete |
| loong64 | 506 | ~88% |
| ppc64le | 281 | **partial** |

- **s390x** loses kernels because clang uses `r13`, the register Go keeps the
  current goroutine in, and there is no `-ffixed` for SystemZ — the global
  register variable is accepted and silently ignored.
- **ppc64le** loses kernels to the TOC pointer: clang reaches its constants
  through `r2`, which Go does not maintain for these objects, and Power9 has no
  PC-relative data addressing to rewrite it into.

Neither is a correctness hole. A kernel that cannot be generated is not
registered, and the portable implementation stands in.

---

## How it is verified

The verification is the part of this project most worth borrowing.

- **Differential testing** of every generated kernel against the portable
  reference, at every length from 0 to 70 and at the block boundaries beyond,
  with adversarial inputs — NaN, ±Inf, ±0, denormals, the extremes of every
  integer type.
- **Tier against tier**, so the promise that results do not change with vector
  width is checked directly rather than inferred.
- **Fuzzing** over the whole kernel set, millions of executions per run.
- **Gate-versus-emission**: every generated `.s` is disassembled and checked
  against the CPU feature its file is gated on, so an EVEX instruction cannot
  reach an AVX2 path. This mechanically prevents the SIGILL class of bug that
  is live in several comparable projects.
- **ABI checks** on the generated code: no kernel may use a register the Go
  runtime owns, write outside its frame, or leave a reserved register changed.
- **Execution on every architecture** under emulation, with
  `simdinfo -require-accelerated` asserting an accelerated tier was actually
  selected before a green run is believed.

That last check exists because its absence cost two backends. The riscv64 and
loong64 lanes were green for months while executing nothing at all — the
emulator in the image predated the vector extension, so every tier was skipped
as unexecutable and the suite passed having tested none of it. The first run
that actually executed them found a segfault in one and wrong answers from
every constant-reading kernel in the other.

---

## Requirements

Consumers need nothing but `go get`. No cgo, no `GOEXPERIMENT`, no build tags,
no C toolchain — the generated assembly is committed.

Contributors who want to regenerate it need clang and `llvm-objdump`. The
generator lives in a nested module so it never becomes a dependency of anyone
using the library.

```
make verify        # fmt, vet, tests, purego build, every tier this CPU can run
make test-cross    # arm64, s390x, ppc64le under docker + qemu
make test-riscv64  # cross-compile and run under a recent qemu-user
make test-loong64  # likewise; there is no golang image for loong64
make bench-check   # benchmarks against the stored baseline for this GOARCH
make codegen       # regenerate every backend (needs clang)
```

---

## Design notes

`docs/research/` carries the reasoning, including the parts that were wrong
first. `05-decisions.md` is the decision record; `06-numerical-findings.md` is
what measurement contradicted — the reference's own architecture dependence
through fused multiply-add, four places where Go's `math` is the less accurate
side, five loop shapes LLVM will not vectorize and what to write instead, and
the two ABI registers that are reserved by *value* rather than by name and are
therefore invisible to every compiler flag.
