# simd

[![Latest Release](docs/assets/badges/release.svg)](https://github.com/sebishogun/simd/releases/latest)
[![Go Reference](https://pkg.go.dev/badge/github.com/sebishogun/simd.svg)](https://pkg.go.dev/github.com/sebishogun/simd)
[![CI](https://github.com/sebishogun/simd/actions/workflows/ci-local.yml/badge.svg)](https://github.com/sebishogun/simd/actions/workflows/ci-local.yml)

**Fast slice math and text scanning for Go, using the vector unit your CPU
already has — without cgo.**

```
go get github.com/sebishogun/simd
```

```go
import "github.com/sebishogun/simd"

simd.Add(a, b)                     // a[i] += b[i]      — in place, no allocation
simd.AddScaled(a, b, 0.5)          // a[i] += b[i]*0.5  — one pass over memory
total := simd.Sum(a)               // fixed accumulation order, same bits everywhere
i     := simd.Index(line, ",")     // takes string or []byte, no copy
simd.GemvInto(y, matrix, x, m, k)  // matrix times vector
```

That is the whole API surface: **ordinary generic functions on ordinary Go
slices**. There is no vector type, no lane count, no target to select, nothing
to initialize, no build tag, and nothing that allocates. It compiles and runs
correctly on any machine Go supports, and goes fast on the ones with a vector
unit.

## Scope

**What this is for** — the loops that dominate numeric and parsing code, where
the same operation runs over a whole slice:

- **Array math**: elementwise arithmetic, saturating arithmetic, scalar
  operations, rounding, comparisons to masks, prefix scans.
- **Reductions**: sum, dot, min/max, argmin/argmax, norms, mean, variance.
- **Transcendentals**: exp, log, sin, tan, tanh, sigmoid and the rest — with a
  stated ULP bound, and a `Fast*` twin where you would rather have the speed.
- **Text and bytes**: index, count, trim, UTF-8 validation, case folding, hex,
  base64 — taking `string` or `[]byte` without copying.
- **Linear algebra**: dot, `Gemv`, a register-blocked `MatMul`.

**What this is not.** It is not a BLAS, not a tensor library, not an
autodiff framework, and not a place to get a `Vector[T]` type. It has no
opinion about how your data is laid out beyond "it is a slice". Operations are
one-shot over whole slices, because the alternative — exposing a vector value
you combine yourself — costs a non-inlinable call per operation in Go and loses
to a plain loop.

**Where the win is.** The crossover is around 16–64 elements depending on the
operation; below it the library runs a plain Go loop, because crossing into
assembly costs more than it saves. It is worth reaching for when your slices
are thousands of elements, not tens.

More: **[runnable examples](example_test.go)** for every operation below
(checked by `go test`, rendered on pkg.go.dev), and
**[complete programs](docs/examples/)** in `docs/examples/`.

## Which function do I want

Named for what you are trying to do, rather than for what the operation is
called. Every one of these has a runnable example in
[`example_test.go`](example_test.go), checked on every build.

| I want to… | Call |
|---|---|
| add/scale/clamp a slice in place | `Add` `Scale` `AddScalar` `Clamp` |
| …without destroying the input | the same name with `Into` |
| do `y += a*x` in one pass (axpy) | `AddScaled` |
| **sum or multiply many slices at once** | `AddAll` `MulAll` — one pass, not one per slice |
| sort a slice | `Sort` `SortInto` (allocation-free) |
| **sort one slice by another's values** | `Argsort` + `GatherInto` |
| split a slice about a threshold | `PartitionInto` — stable on both sides |
| total / average / spread of a slice | `Sum` `Mean` `StdDev` `Variance` |
| length of a vector, distance between two | `Norm` `Distance` `CosineSimilarity` |
| make a vector unit length | `Normalize` |
| smallest and largest in one pass | `MinMax`, or `ArgMin`/`ArgMax` for positions |
| **keep only the elements that pass a test** | a comparison → `[]bool`, then `CompressInto` |
| apply an arbitrary Go predicate | `FilterInto` (convenient, not fast — see its doc) |
| running totals / differences | `CumSum` `Diff` |
| exp/log/trig over a slice | `Exp` `Log` `Sin` … and the `Fast*` twins |
| pick between two slices per element | `SelectInto` |
| **apply a matrix to a vector** | `GemvInto` |
| multiply two matrices | `MatMulInto` |
| find a byte or substring | `IndexByte` `Index` `LastIndex` |
| **find every occurrence at once** | `IndexAll` — the structural-index step of a parser |
| find any of a set of bytes | `IndexAny` `IndexNotAny` `CountAny` |
| trim, fold case, validate UTF-8 | `TrimAny` `TrimSpaceASCII` `EqualFoldASCII` `ValidUTF8` |
| hex or base64 | `HexEncode` `Base64Encode` `Base64Decode` |
| convert to/from float16 or bfloat16 | `Float16ToFloat32Into` and friends |

## Status

**v0.1.0 is the first tag.** See [CHANGELOG.md](CHANGELOG.md) for what is and
is not covered by compatibility, and [ROADMAP.md](ROADMAP.md) for the gaps.

- **309 exported functions** over ten element types, plus complex, bytes, text
  and the narrow float formats.
- **5,247 generated kernels** across nine targets — amd64 sse2/avx2/avx512,
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

**N-ary** `AddAll` `MulAll` — three or more slices in a single pass over
memory, with the element type enforced by the compiler ·
**Comparisons** to `[]bool` masks, `Select`, `All` `Any` `CountTrue` ·
**Compression** `CompressInto` `ExpandInto` `FilterInto` — a comparison writes
the mask, `CompressInto` packs it, so one function serves every predicate

**Sorting** `Sort` `SortInto` `Argsort` `PartitionInto` `SortedIndex` — a
quicksort around a compress-based partition, 19–27% faster than `slices.Sort`
above ~16K elements ·
**Linear algebra** `MatMulInto` (register-blocked microkernel) `GemvInto`
`AddScaled` `Dot` `Norm` — `Gemv` is bit-identical to `Dot` per row

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

**`MatMulInto`**, register-blocked against the previous naive kernel — and the
right-hand column is the number that matters, since single-core AVX-512 f32
peak on this machine is about 290 GFLOP/s:

| f32, square | naive | blocked | | GFLOP/s |
|---|---|---|---|---|
| n=64 | 9.51 µs | 2.13 µs | −78% | 246 |
| n=128 | 51.7 µs | 16.9 µs | −67% | 249 |
| n=256 | 331 µs | 129 µs | −61% | **260** |
| n=512 | 3.25 ms | 1.31 ms | −60% | 204 |

`GemvInto` reaches 172 GB/s while the matrix is cache-resident and 49 GB/s at
4096×4096, where it is bound by memory rather than by arithmetic.

**`CompressInto` against the scalar filter loop it replaces**, geomean −51%.
The axis that matters is match density, and a single-density benchmark
misreports it in either direction — the scalar loop costs a branch per element,
so it is fastest exactly when that branch is predictable:

| 1 M int32 | scalar loop | `CompressInto` | |
|---|---|---|---|
| 1% match | 12.3 GiB/s | 19.3 GiB/s | −36% |
| 25% match | 2.17 GiB/s | 19.4 GiB/s | −89% |
| 50% match | 1.29 GiB/s | 19.3 GiB/s | **−93%** |
| 90% match | 4.12 GiB/s | 19.1 GiB/s | −78% |

The right column barely moves. That is the point: the vector version costs the
same whatever the data does, and the branch predictor is what collapses.

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
| amd64 (3 tiers) | 1784 | essentially complete |
| arm64 (2 tiers) | 1201 | essentially complete |
| s390x | 650 | **partial** |
| riscv64 | 598 | essentially complete |
| loong64 | 546 | ~88% |
| ppc64le | 468 | was 281; see below |

- **s390x** loses kernels because clang uses `r13`, the register Go keeps the
  current goroutine in, and there is no `-ffixed` for SystemZ — the global
  register variable is accepted and silently ignored.
- **ppc64le** used to lose 184 kernels to the TOC pointer, and no longer does.
  clang reaches its constants through `r2`, which Go does not maintain for
  these objects, and Power9 has no PC-relative data addressing — which looked
  like the obstacle and was not. Go's own assembler materialises a symbol
  address in two instructions with no TOC involvement, so the pool becomes a
  standalone symbol, `R2` is pointed at it, and clang's global-entry prologue
  is replaced in place. 281 kernels became 468.

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

## Where the obvious answer was wrong

**[`docs/wrong.md`](docs/wrong.md)** is the part of this project most worth
borrowing: twenty-two things a competent person would have assumed, that were
false, and what each one cost. Among them —

- A register can be reserved by *value* rather than by name, which makes it
  invisible to every compiler flag. The symptom is Go's allocator dying several
  calls later.
- A compiler builtin that silently compiles to nothing at `-O1` and above, and
  is correct at `-O0`.
- Green test lanes that had been executing no accelerated code for months.
- Four loops that were slower *after* being vectorized, one of them by 1700×.
- `--mattr=+sve2` removing NEON rather than adding SVE2.
- Go's own SIMD intrinsics being 4.4× *slower* than the generated assembly.
- A closure comparator costing a sort 2.5×, and the wrong fix making it worse.
- A scripted edit that silently did not apply, and would have crashed most CPUs.
- A failing test naming a kernel that was not the one with the bug.
- A test lane that was hung, not slow — thirty-two minutes at 0.1% CPU.
- `ENOSPC` with 40 GB free.

`docs/research/` carries the longer reasoning behind the design decisions;
`05-decisions.md` is the decision record.
