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

**New here?** [`docs/tutorial.md`](docs/tutorial.md) is the one to read. This
library cannot vectorize your program — it can only vectorize the loops you
hand it — and how you shape your data decides whether there is anything to
hand over. The tutorial covers struct-of-arrays, buffer reuse, fusing instead
of chaining, and the operations that will never vectorize no matter what you
do. Every snippet in it compiles, and the worked example is a program you can
run.

**Looking for how to do a specific thing?** [`docs/guide/`](docs/guide/) is
task-shaped prose: [arrays and reductions](docs/guide/arrays.md),
[text and bytes](docs/guide/text.md),
[search, sets and bit vectors](docs/guide/search.md),
[encodings](docs/guide/encoding.md), and
[signal and matrices](docs/guide/signal.md). Each page explains the problem, shows the
code, and says where the operation stops paying — including the cases where a
plain loop wins.

## Scope

**What this is for** — anything a vector unit can do faster than a scalar loop.
If an operation has a known SIMD formulation, it belongs here; the only reasons
to refuse one are that it was measured and lost, or that nobody has built it
yet. Both are recorded rather than implied.

That is broader than this section used to claim. It previously listed five
families and called everything else out of scope, and the effect was that the
catalogue grew by completing a list rather than by asking what people reach
for — which is how it reached four hundred operations without `Quantize` in
it. The scope was the bug, so the scope changed.

The families today:

- **Array math**: elementwise arithmetic, saturating arithmetic, scalar
  operations, rounding, comparisons to masks, prefix scans.
- **Reductions**: sum, dot, min/max, argmin/argmax, norms, mean, variance.
- **Transcendentals**: exp, log, sin, tan, tanh, sigmoid and the rest — with a
  stated ULP bound, and a `Fast*` twin where you would rather have the speed.
- **Text and bytes**: index, count, trim, UTF-8 validation, case folding, hex,
  base64 — taking `string` or `[]byte` without copying.
- **Linear algebra**: dot, `Gemv`, a register-blocked `MatMul`, and CSR
  sparse matrix-vector.
- **Search and set operations**: batched binary search, sorted-set intersection
  and difference, rank/select over a bit vector, longest common prefix.
- **Encodings**: quantization to int8 and fp8, zigzag, bit packing, run-length,
  and the varint widths an encoder needs before it writes a byte.

**What this is not.** Not a BLAS, not a tensor library, not an autodiff
framework. It has no opinion about how your data is laid out beyond "it is a
slice", and it does not own your program's control flow.

**Why the default shape is whole-slice, and where that stops.** Most operations
take and return slices, because exposing a vector value you combine yourself
costs a non-inlinable call per operation in Go and loses to a plain loop — a
measured claim, not a stylistic one. But whole-slice is the default shape, not
the boundary. An algorithm with a known SIMD formulation belongs here whether
or not it fits that mould: data-dependent output lengths already appear in
`HexDecode` and `CompressInto`, and a stateful or multi-pass algorithm is a
design problem rather than a disqualification.

For the two cases the catalogue cannot serve — an operation nobody has built
yet, or one small enough that the call boundary dominates — see
[`docs/kernels.md`](docs/kernels.md) to add a kernel that reaches all six
architectures, and the `goexperiment.simd` vector type for writing one inline
on amd64. [`CONTRIBUTING.md`](CONTRIBUTING.md) covers how to verify one.

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
| running totals / differences | `CumSum` `DiffInto` |
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
| median / percentile without sorting | `Median` `Quantile`, `MedianInto` for zero-alloc |
| the k largest or smallest | `TopK` `BottomK` — selects, does not sort |
| histogram / count occurrences | `Histogram` `Bincount` |
| find the NaNs, sum around them | `IsNaNInto` `CountNaN` `NanSum` `NanMean` |
| shifts, rotates, popcount per element | `Shl` `Rotl` `OnesCountInto` `LeadingZerosInto` `ByteSwapInto` |
| a Fourier transform | `FFTInto` with a reusable plan; `RFFT` for real input |
| envelope / analytic signal | `HilbertInto`, then `AbsComplexInto` |
| window a signal | `Hann` `Hamming` `Blackman`, then `ApplyWindowInto` |
| convolve or correlate | `ConvolveFullInto` — picks direct or FFT by a measured crossover |
| interpolate a table | `InterpInto` — numpy's interp, clamping |
| transpose a matrix | `TransposeInto` — blocked, 3.6× the naive loop |
| parse a CSV of integers | `IndexAll` + `ParseInts` — 5× strconv |
| **quantize a tensor to int8** | `QuantizeInt8`, or `QuantizePerChannelInt8` for weights |
| **multiply int8 tensors** | `QMatMulInt8Into` → int32, then `RequantizeInt8Into` |
| normalize a transformer layer | `LayerNorm`, or `LayerNormInto` with gamma and beta |
| **look up many keys in a sorted table** | `LowerBoundInto` — one binary search per query, batched |
| how much two byte slices share | `CommonPrefixLen` — the LCP step of a suffix array |
| rolling minimum or maximum | `RollingMinInto` `RollingMaxInto` — see the window note in its doc |
| **intersect or subtract sorted sets** | `IntersectInto` `DifferenceInto` — posting lists |
| rank/select over a bit vector | `RankTableInto`, then `Rank` and `Select` |
| size a varint stream before writing it | `VarintSize` `VarintLenInto` `AppendVarints` |
| **multiply a sparse matrix by a vector** | `SpMVInto`, or `SparseDot` for one CSR row |
| convert to/from fp8 | `Float32ToFloat8E4M3Into` and the e5m2 pair |
| make negative deltas small | `ZigzagEncodeInt32Into`, before varint |
| pack a column densely | `DiffInto` → `ZigzagEncodeInt32Into` → `BitPackInto` |
| run-length encode a column | `RunLengthEncodeInt32`, or `RunStartsInto` for the mask alone |
| compare two bit vectors | `HammingDistance` — fused, not `Xor` then `PopCount` |
| convert planar RGB | `GrayscaleInto`, `RGBToUVInto` |
| **sum data arriving in chunks** | `Accumulator[T]` — bit-identical to `Sum` of the whole |
| fill a slice with random values | `RandomInto` — reproducible, splittable, same everywhere |

## Status

**v0.1.0 is the first tag.** See [CHANGELOG.md](CHANGELOG.md) for what is and
is not covered by compatibility, and [ROADMAP.md](ROADMAP.md) for the gaps.

- **461 exported functions** over ten element types, plus complex, bytes, text
  and the narrow float formats.
- **6,671 generated kernels** across nine targets — amd64 sse2/avx2/avx512,
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
**Scans** `CumSum` `CumProd` `CumMin` `CumMax` `Diff` `FastCumSum` `FastCumProd`

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

### Measure it here instead

Every number above is about someone else's CPU. To get one about yours:

```
go run ./cmd/site      # http://localhost:8080
```

It runs the `docs/tutorial.md` comparisons live, shows both implementations
side by side, and reports the minimum of several samples with the detected tier
and the load average printed beside it — warning you off the result when the
machine is busy, because a number measured under load looks exactly like a real
one. Datastar is vendored in `cmd/site/assets/`; nothing contacts a CDN.

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

Counts are what the generator emits, from `make check-emission`, and the skip
column is kernels it declined with a stated reason rather than kernels nobody
wrote. Both columns sum over an architecture's tiers, so amd64's three each
contribute — a kernel declined on sse2 and emitted on avx2 counts once in
each column.

| | kernels | skipped | |
|---|---|---|---|
| amd64 (sse2/avx2/avx512) | 2493 | 88 | essentially complete |
| arm64 (neon/sve2) | 1614 | 108 | essentially complete |
| riscv64 (rvv) | 849 | 14 | essentially complete |
| loong64 (lasx) | 710 | 149 | see the `.L0` note below |
| ppc64le (vsx) | 592 | 267 | was 281 before the TOC rewrite |
| s390x (vx) | 413 | 446 | **partial**, and the reason is an ABI wall |

6,671 kernels in total. Almost all of the remaining skips are the `Fast*` tier:
it is the newest and the least portable, and where a target declines one the
accurate kernel stands in, which satisfies the bound because the bound is an
upper bound.

amd64's column used to be the interesting one and no longer is. It read 2277
kernels and 199 skips, and the large majority of those skips were sse2 refusing
a legacy SSE instruction that reads a constant pool: such an instruction needs
its operand 16-byte aligned, and nothing promises the alignment of bytes inside
a TEXT symbol. [`docs/wrong.md`](docs/wrong.md) entry 61 argued the reason on
file for not fixing it was answering the wrong question, and entry 67 fixed it
— the pool moves to a separate RODATA symbol, which the linker does align, and
the one instruction that reads it is emitted as a Plan 9 mnemonic of exactly
the same length. sse2 went from 673 kernels to 789 and there are no alignment
refusals left on any tier.

Two earlier versions of this table were wrong in opposite directions, and both
are worth stating rather than quietly restating. One gave s390x as 650: it
added the registrations and the wrapper functions, which are one per kernel,
and so reported double. The other gave the skip column as 13/12/4/9/11/13,
which was accurate before the `Fast*` tier existed and counted none of it
afterwards — a column that stops moving is easier to trust than one that was
never right, and harder to notice.

- **s390x** loses kernels because clang uses `r13`, the register Go keeps the
  current goroutine in, and there is no `-ffixed` for SystemZ — the global
  register variable is accepted and silently ignored.
- **ppc64le** used to lose 184 kernels to the TOC pointer, and no longer does.
  clang reaches its constants through `r2`, which Go does not maintain for
  these objects, and Power9 has no PC-relative data addressing — which looked
  like the obstacle and was not. Go's own assembler materialises a symbol
  address in two instructions with no TOC involvement, so the pool becomes a
  standalone symbol, `R2` is pointed at it, and clang's global-entry prologue
  is replaced in place. 281 kernels became 468, and 588 as of this writing.
- **loong64** declines nine, and the largest group is not an ABI wall but a
  relocation one: clang emits LoongArch branches with a displacement of zero
  and expects the linker to fill them in, so the bodies cannot be copied
  verbatim the way AArch64's and RISC-V's can. Making them work needs the
  generator to compute and patch each branch, which is bounded work nobody has
  done. Treating them as already-resolved, which is true elsewhere, produces
  branches to themselves — entry 46.

None of this is a correctness hole. A kernel that cannot be generated is not
registered, and the portable implementation stands in. The differential suite
compares whatever a tier actually ended up with against that reference, so a
skipped kernel is slower and never wrong.

**Throughput off amd64 is modelled, not measured.** `make perf-model` runs
llvm-mca over each kernel's inner loop for arm64, ppc64le and s390x; nothing
models below 1.2x, and the model agrees with measured amd64 to within 5–12%
on the avx512-versus-avx2 comparison. What no model can give is wall-clock
under a real memory system, which is what dominates a whole-slice kernel at
large n — so every GB/s figure in this file is amd64 and says so.

---

## How it is verified

Start with `make menu`, which lists every verification target and marks the
ones this machine can actually run:

```
$ make menu
gosimd  make targets on darwin/arm64
        arm64 tier=neon available=[scalar neon]
        go clang llvm-mca llvm-objdump docker benchstat qemu-riscv64 …

● test-tiers      Run the suite once per instruction-set tier this CPU has
● perf-model      Model kernel throughput on architectures this machine …
○ test-riscv64    Full suite on riscv64 RVV under qemu
○ test-loong64    Full suite on loong64 LASX under qemu
```

A third of these targets cannot run on any given machine — the qemu lanes need
a Linux host, the cross lane needs docker with binfmt, codegen needs clang and
llvm-objdump — and the preview says which, why, and what to install, rather
than leaving you to find out by reading a failure. `make targets` prints the
same list plainly. Neither needs fzf; it is used when present.

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
borrowing: twenty-three things a competent person would have assumed, that were
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
