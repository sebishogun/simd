# simd.go

[![Latest Release](docs/assets/badges/release.svg)](https://github.com/sebishogun/simd/releases/latest)
[![Go Reference](https://pkg.go.dev/badge/github.com/sebishogun/simd.svg)](https://pkg.go.dev/github.com/sebishogun/simd)
[![CI](https://github.com/sebishogun/simd/actions/workflows/ci-local.yml/badge.svg)](https://github.com/sebishogun/simd/actions/workflows/ci-local.yml)

**Fast slice math and text scanning for Go, using the vector unit your CPU
already has — without cgo.**

The project is *simd.go*. The import path is `github.com/sebishogun/simd` and
the package is `simd`.

## Install

```
go get github.com/sebishogun/simd
```

Go 1.25 or later. No cgo, no C toolchain, no build tags, no `GOEXPERIMENT`. The
kernels are compiled ahead of time and committed as assembly, so this is an
ordinary Go dependency with one transitive import (`golang.org/x/sys`, for CPU
feature detection). Dispatch is one static table per operation, so the linker
keeps only the operations a program actually calls: a binary using three
functions carries three functions' kernels, not the library.

## Quick start

```go
import "github.com/sebishogun/simd"

simd.Add(a, b)                     // a[i] += b[i]      — in place, no allocation
simd.AddScaled(a, b, 0.5)          // a[i] += b[i]*0.5  — one pass over memory
total := simd.Sum(a)               // fixed accumulation order, same bits everywhere
i     := simd.Index(line, ",")     // takes string or []byte, no copy
simd.GemvInto(y, matrix, x, m, k)  // matrix times vector
```

Generic functions over ordinary Go slices. There is no vector type, no lane
count, no target to select, and nothing to initialize. The best available
instruction set is detected once at startup.

**Two conventions cover the whole API:**

```go
simd.Add(a, b)          // plain name: in place, a[i] += b[i]
simd.AddInto(dst, a, b) // Into suffix: dst[i] = a[i] + b[i], inputs untouched
```

Nothing allocates. `Into` functions take the destination from the caller
precisely so they do not have to allocate one.

## Documentation

- **[docs/tutorial.md](docs/tutorial.md)** — start here. Covers
  struct-of-arrays layout, buffer reuse, fusing instead of chaining, and the
  operations that will not vectorize at all. Data layout decides whether there
  is anything to vectorize, and it matters more than which function you call.
- **[docs/guide/](docs/guide/)** — task-oriented pages:
  [arrays and reductions](docs/guide/arrays.md),
  [text and bytes](docs/guide/text.md),
  [search, sets and bit vectors](docs/guide/search.md),
  [encodings](docs/guide/encoding.md),
  [signal and matrices](docs/guide/signal.md).
- **[Runnable examples](https://pkg.go.dev/github.com/sebishogun/simd#pkg-examples)**
  for every operation in the index below, in `example_*_test.go`, compiled and
  checked by `go test`.
- **[docs/examples/](docs/examples/)** — complete programs.
- **[docs/kernels.md](docs/kernels.md)** — how to add a kernel.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — how to verify one.

## Function index

Organised by task rather than by operation name.

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
| **filter a column by an Arrow validity bitmap** | `CompressBitsInto` — packed bits, LSB-first; take is `GatherInto` |
| **null-aware sum and non-null count** | `SumValid` `CountValid` — never read a null slot, bit-identical to `Sum` |
| apply an arbitrary Go predicate | `FilterInto` (convenient, not fast — see its doc) |
| running totals / differences | `CumSum` `DiffInto` |
| exp/log/trig over a slice | `Exp` `Log` `Sin` … and the `Fast*` twins |
| pick between two slices per element | `SelectInto` |
| **apply a matrix to a vector** | `GemvInto`, or `GemvParallelInto` for a large one |
| multiply two matrices | `MatMulInto`, or `MatMulParallelInto` when the multiply is the whole job |
| **add an outer product to a matrix** | `RankOneInto` — BLAS ger, the inner loop of an LU |
| apply a Givens rotation | `Rotate` — what QR and the SVD are made of |
| exchange two slices | `Swap` — pivoting a decomposition |
| find a byte or substring | `IndexByte` `Index` `LastIndex` |
| **find every occurrence at once** | `IndexAll`, or `IndexAllAny` for a set of delimiters in one pass |
| **the same, as a bitmask** | `MaskBits` `MaskBitsAny` `MaskBitsLess` — a bit per byte, 22–37× the offset list when matches are dense |
| find any of a set of bytes | `IndexAny` `IndexNotAny` `CountAny` |
| **the next byte that is not plain text** | `IndexAnyOrLess` — a set and a threshold in one pass, the inner loop of an escape routine |
| **copy text up to the byte that needs escaping** | `JSONCopyRun` — the scan and the copy in one pass, for a JSON encoder |
| **classify a JSON document five ways at once** | `JSONMasks` `MaskWords` — quotes, backslashes, brackets, control bytes and whitespace from one pass; 2x five separate calls |
| **validate a JSON document in one kernel call** | `JSONValid` — classification, quote parity, escape checks and the grammar walk fused, no mask buffers; `JSONStage1` + `JSONValidTokens` are its staged halves for callers that also need the masks |
| **escape a string into a JSON encoder's buffer** | `JSONQuote` (assumes valid UTF-8) or `JSONCopyValid` (replaces invalid bytes), both with optional HTML-safe escaping |
| trim, fold case, validate UTF-8 | `TrimAny` `TrimSpaceASCII` `EqualFoldASCII` `ValidUTF8` |
| hex or base64 | `HexEncode` `Base64Encode` `Base64Decode` |
| checksum a buffer | `Adler32` — vectorized where stdlib is scalar; `CRC32C` matches hash/crc32 Castagnoli |
| format a float shortest-form | `FormatFloat64` — Schubfach with the render attached, 1.65× strconv, encoding/json's format rule |
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

## What is included

Any operation a vector unit can do faster than a scalar loop. An operation is
absent only because it was measured and lost, or because nobody has built it
yet; both are recorded rather than implied.

- **Array math** — elementwise and saturating arithmetic, scalar operations,
  rounding, comparisons to masks, prefix scans.
- **Reductions** — sum, dot, min/max, argmin/argmax, norms, mean, variance.
- **Transcendentals** — exp, log, sin, tan, tanh, sigmoid and the rest, each
  with a stated ULP bound and a `Fast*` twin.
- **Text and bytes** — index, count, trim, UTF-8 validation, case folding, hex,
  base64, accepting `string` or `[]byte` without copying.
- **Linear algebra** — dot, `Gemv`, a register-blocked `MatMul` with an
  opt-in parallel variant, CSR sparse matrix-vector, and the pieces a
  decomposition is made of: rank-1 update, Givens rotation, swap.
- **Search and sets** — batched binary search, sorted-set intersection and
  difference, rank/select over a bit vector, longest common prefix.
- **Encodings** — int8 and fp8 quantization, zigzag, bit packing, run-length,
  varint widths.

Element types: `float32` `float64` `int8` `int16` `int32` `int64` `uint8`
`uint16` `uint32` `uint64` `complex64` `complex128`, plus bytes, text and the
narrow float formats.

**Not** a BLAS, a tensor library, or an autodiff framework. It has no opinion
about your data layout beyond "it is a slice" and does not own your control
flow.

### Where the win is

The crossover is roughly 16 to 64 elements depending on the operation. Below it
the library runs a plain Go loop, because crossing into assembly costs about
1.4 ns and cannot be inlined. Reach for this when slices are thousands of
elements, not tens. Calling a vector operation once per element in a loop is
slower than not using the library at all — see the tutorial.

Most operations take and return whole slices. Exposing a vector value that
callers combine themselves costs a non-inlinable call per operation and loses
to a plain loop; that is measured, not stylistic. Whole-slice is the default
shape rather than a hard boundary — `HexDecode` and `CompressInto` already have
data-dependent output lengths. For operations too small for the call boundary,
there is a `goexperiment.simd` vector type on amd64 and arm64.

## Accuracy

**Every operation is bit-identical across instruction sets**, including NaN
payloads, ±Inf, ±0 and denormals. Reductions use a fixed sixteen-accumulator
tree that a 128-bit and a 512-bit machine both reproduce exactly, so a result
cannot change because the computation moved to a different server. `Dot` never
contracts into FMA on any architecture, including those where Go's compiler
would.

This costs throughput deliberately. The alternative is what
[vek does](https://github.com/viterin/vek/issues/11): its vectorized body and
scalar remainder disagree on NaN, so the answer depends on input length.

Two exceptions, both opt-in by name:

- **Transcendentals** guarantee a ULP bound rather than bit identity, because
  the polynomial correct to 1 ULP in float32 is not the one correct to 1 ULP in
  float64. The bound is measured against the standard library, not asserted
  from theory.
- **`Fast*`** promises 3.5 ULP and gives up agreement between architectures,
  because it compiles with fused multiply-add. It keeps IEEE 754 semantics: NaN
  in gives NaN out, infinities go where the standard says, signed zeros
  survive. `-ffast-math` is not used.

## Performance

Measured on a Ryzen AI MAX+ 395 (Zen 5, AVX-512), `-count 6` or more, compared
with `benchstat`. Regenerate with `make bench-check`. **Every number here is
amd64** — see [platform support](#platform-support) for why there are no others.

**Against the portable Go build.** Integer and saturating arithmetic, geomean
over the whole set: −86% time, +593% throughput.

| int8 `SatAdd` | portable | accelerated | |
|---|---|---|---|
| n=256 | 146 ns | 6.6 ns | −95% |
| n=4096 | 2.42 µs | 22.0 ns | **−99.1%** |

**Against `bytes` and `strings`.** The harder comparison, since `bytealg` is
already hand-written assembly on four of the six architectures. Geomean +186%.

| | vs stdlib |
|---|---|
| `LastIndex` n=4096 | **+8309%** |
| `IndexAny` n=1 MiB | +1084% |
| `Index` n=1 MiB | +623% |
| `IndexAll` n=1 MiB | +135% |
| `ValidUTF8` n=1 MiB, non-ASCII | +332% |
| `ValidUTF8` n=1 MiB, ASCII | +58% |
| `TrimSpaceASCII` | +29% |

**Against `encoding/base64`:** −42% to −63%, +74% throughput.

**`MatMulInto`**, register-blocked, against the previous naive kernel.
Single-core AVX-512 float32 peak on this machine is about 290 GFLOP/s, so the
right-hand column is the one that matters:

| f32, square | naive | blocked | | GFLOP/s |
|---|---|---|---|---|
| n=64 | 9.51 µs | 2.13 µs | −78% | 246 |
| n=128 | 51.7 µs | 16.9 µs | −67% | 249 |
| n=256 | 331 µs | 127 µs | −62% | **264** |
| n=512 | 3.25 ms | 1.31 ms | −60% | 204 |

`GemvInto` reaches 172 GB/s while the matrix is cache-resident and 49 GB/s at
4096×4096, where it is bound by memory rather than arithmetic.

**`CompressInto` against the scalar filter loop**, geomean −51%. Match density
is the axis that matters, and a single-density benchmark misleads in either
direction: the scalar loop costs a branch per element, so it is fastest exactly
when that branch is predictable.

| 1 M int32 | scalar loop | `CompressInto` | |
|---|---|---|---|
| 1% match | 12.3 GiB/s | 19.3 GiB/s | −36% |
| 25% match | 2.17 GiB/s | 19.4 GiB/s | −89% |
| 50% match | 1.29 GiB/s | 19.3 GiB/s | **−93%** |
| 90% match | 4.12 GiB/s | 19.1 GiB/s | −78% |

The vector column barely moves across densities. The scalar column collapses
with branch prediction.

**`Fast` against accurate:** `FastSin` −45%, `FastExp` −43%, `FastSigmoid`
−36%, `FastLog` −25%.

Where the standard library is already assembly doing the same work —
`bytes.Equal` is `memequal`, `bytealg.Count` popcounts a compare mask — there
is no margin, and this library calls it instead.

### Measuring on your own machine

```
go run ./cmd/site      # http://localhost:8080
```

Runs the `docs/tutorial.md` comparisons live, shows both implementations side
by side, and reports the minimum of several samples with the detected tier and
load average printed alongside. It warns when the machine is busy. Datastar is
vendored in `cmd/site/assets/`; nothing contacts a CDN.

## Platform support

| architecture | tiers | correctness | wall-clock |
|---|---|---|---|
| amd64 | sse2, avx2, avx512 | **real hardware** | **real hardware** |
| arm64 | neon | **real hardware** | unmeasured |
| arm64 | sve2 | emulation | unmeasured |
| riscv64 | rvv | emulation | unmeasured |
| ppc64le | vsx | emulation | unmeasured |
| s390x | vx | emulation | unmeasured |
| loong64 | lasx | emulation | unmeasured |

Every kernel is differential-tested against the portable implementation on
every architecture on every change, which is what catches a wrong answer.
Emulation proves nothing about timing, because qemu does not model a pipeline.
Where the table says *unmeasured*, no performance claim in this repository
applies.

Throughput off amd64 is modelled rather than measured: `make perf-model` runs
llvm-mca over each kernel's inner loop for arm64, ppc64le and s390x. Nothing
models below 1.2×, and the model agrees with measured amd64 to within 5–12% on
the avx512-versus-avx2 comparison. No model gives wall-clock under a real
memory system, which is what dominates a whole-slice kernel at large n.

**Have one of these machines?** A single real run is worth more than any amount
of emulation, and it takes two commands. Failures are as useful as passes. See
[Reporting a hardware run](CONTRIBUTING.md#reporting-a-hardware-run). No
knowledge of the internals is needed, and the table above moves as reports
arrive.

The library has no OS-specific source. It builds and vets clean for
darwin/amd64, darwin/arm64, windows/amd64, windows/arm64 and freebsd/amd64; the
entire OS-dependent surface is `x/sys/cpu` feature detection.

## Kernel coverage

483 exported functions and 6,858 generated kernels across nine targets. The
function count is for an ordinary build; the `goexperiment.simd` vector type
adds four more. Kernel counts come from `make check-emission`. The skip column is kernels the generator
declined with a stated reason, not kernels nobody wrote. Both columns sum over
an architecture's tiers, so a kernel declined on sse2 and emitted on avx2
counts once in each.

| | kernels | skipped | dominant reason for skips |
|---|---|---|---|
| amd64 (sse2/avx2/avx512) | 2574 | 100 | LLVM declined to vectorize |
| arm64 (neon/sve2) | 1661 | 123 | LLVM declined to vectorize |
| riscv64 (rvv) | 879 | 15 | LLVM declined to vectorize |
| loong64 (lasx) | 723 | 167 | `$fp`, then `.L0` references |
| ppc64le (vsx) | 602 | 288 | `r30`, then LLVM refusals |
| s390x (vx) | 419 | 471 | `r13` — an ABI wall, see below |

Most remaining skips are in the `Fast*` tier, which is the newest and least
portable. Where a target declines a `Fast*` kernel the accurate kernel stands
in, which still satisfies the promise because 3.5 ULP is an upper bound.

The ABI walls, in order of how much they cost:

- **s390x** loses 405 of its 446 skips to `r13`. clang allocates it; Go keeps
  the current goroutine there. SystemZ has no `-ffixed` equivalent — the global
  register variable is accepted and silently ignored.
- **ppc64le** loses most of its skips to `r30`, which Go uses for `g`, plus a
  smaller group where clang writes a nonzero value to `r0`. The ppc64le ABI
  defines `r0` as constant zero, so a signal arriving before the epilogue would
  run the runtime with a poisoned zero register. The TOC pointer (`r2`) used to
  cost 184 kernels and now costs 8: the constant pool became a standalone
  symbol addressed in two instructions, and clang's global-entry prologue is
  replaced in place. That took ppc64le from 281 kernels to 592.
- **loong64** loses 92 to `$fp` and 28 to `.L0` references, where clang points
  at a label that is not a constant pool and so cannot be lifted the way a pool
  can. A further group emits branches with a displacement of zero for the
  linker to fill in, which the generator would have to compute and patch —
  bounded work nobody has done. Copying them verbatim, which is correct on
  AArch64 and RISC-V, produces branches to themselves.

None of this is a correctness hole. A kernel that cannot be generated is not
registered and the portable implementation runs instead. The differential suite
compares whatever a tier actually ended up with against that reference, so a
skipped kernel is slower and never wrong.

## Testing

`make menu` lists every verification target and marks the ones the current
machine can run:

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

Roughly a third of the targets cannot run on any given machine — the qemu lanes
need a Linux host, the cross lane needs docker with binfmt, codegen needs clang
and llvm-objdump. The preview says which, why, and what to install. `make
targets` prints the same list plainly. Neither requires fzf; it is used when
present.

What runs:

- **Differential testing** of every generated kernel against the portable
  reference, at every length from 0 to 70 and at block boundaries beyond, with
  adversarial inputs: NaN, ±Inf, ±0, denormals, the extremes of every integer
  type.
- **Tier against tier**, so the promise that results do not vary with vector
  width is checked directly rather than inferred.
- **Fuzzing** across the kernel set, millions of executions per run.
- **Gate-versus-emission**: every generated `.s` is disassembled and checked
  against the CPU feature its file is gated on, so an EVEX instruction cannot
  reach an AVX2 path. This prevents the SIGILL class of bug mechanically.
- **ABI checks**: no kernel may use a register the Go runtime owns, write
  outside its frame, or leave a reserved register modified.
- **Execution on every architecture** under emulation, with `simdinfo
  -require-accelerated` asserting that an accelerated tier was actually
  selected before a green run is believed.

That last check exists because its absence cost two backends. The riscv64 and
loong64 lanes were green for months while executing nothing: the emulator in
the image predated the vector extension, so every tier was skipped as
unexecutable and the suite passed having tested none of it. The first run that
actually executed them found a segfault in one and wrong answers from every
constant-reading kernel in the other.

## Repository layout

```
.                    the simd package — the public API, one file per topic
csrc/                the kernels, in C. The source of truth for every fast path
internal/
  amd64/ arm64/ …    generated Plan 9 assembly, one directory per architecture
  ref/               the portable Go implementation everything is tested against
  kernel/            dispatch table and the numerical contract
  cpu/               feature detection and tier selection
  conformance/       the differential suite: every tier against ref, and each other
  asmcheck/          static assertions on the committed assembly
  benchmarks/        every benchmark
  tests/             the public-API test suite, by topic:
                       arrays reduce text search encode dsp matrix docs
cmd/simdinfo/        prints the tier actually selected on this machine
cmd/site/            local benchmark site
tools/               the code generator — a separate module, never your dependency
docs/                tutorial, guide, kernels.md, wrong.md, examples
testdata/bench/      recorded benchmark baseline, per GOARCH
testdata/hardware/   one report per machine that has run on real silicon
```

**The C is the source; the assembly is the output.** A kernel is written once
in [`csrc/`](csrc), compiled per instruction set by [`tools/`](tools), and
committed under [`internal/`](internal) so that using this library needs no C
toolchain. Every generated `.s` names the C file it came from and the target it
was built for, and none of them should be edited by hand — `make codegen`
regenerates them.

**Tests live next to what they can reach.** Anything that only calls the public
API is in `internal/tests/<topic>/`; what stays in the root directory is what
Go requires to be there — tests of unexported behaviour, the `export_test.go`
hooks and the files using them, and the runnable examples, which have to sit
beside the package to appear on pkg.go.dev.

Every directory above has a README explaining what is in it.

## Development

Consumers need nothing beyond `go get`. Regenerating the assembly needs clang
and `llvm-objdump`. The generator lives in a nested module so it never becomes
a dependency of anyone using the library.

```
make verify        # fmt, vet, tests, purego build, every tier this CPU can run
make test-cross    # arm64, s390x, ppc64le under docker + qemu
make test-riscv64  # cross-compile and run under a recent qemu-user
make test-loong64  # likewise; there is no golang image for loong64
make bench-check   # benchmarks against the stored baseline for this GOARCH
make codegen       # regenerate every backend (needs clang)
```

## Built on this

Four libraries use these kernels. Each is a case where the vector unit changes
what the code can do, rather than a wrapper around the API.

| | | |
|---|---|---|
| [**simdblas**](https://github.com/sebishogun/simdblas) | A BLAS backend for [gonum](https://github.com/gonum/gonum) | One `blas64.Use` call and `mat`, `stat` and `optimize` run on it. Covariance plus Cholesky 4.4×, QR 2.0×, `mat.Mul` 4.3×. |
| [**simdjson**](https://github.com/sebishogun/simdjson) | Structural-index JSON parsing | 6.6× `encoding/json` for field extraction, and 1.4–1.7× [minio/simdjson-go](https://github.com/minio/simdjson-go) — which is amd64-only. Its structural pass is built on `MaskBits`. |
| [**simdcsv**](https://github.com/sebishogun/simdcsv) | CSV reading | 1.7–2.1× `encoding/csv` on unquoted data, with fields as subslices rather than copies. |
| [**simdvec**](https://github.com/sebishogun/simdvec) | Embedding search | 18–38× a hand-written loop; the whole index scan is one matrix-vector product. |

simdjson is the one worth reading if you are deciding whether any of this is
real. It was 1.3–1.8× *slower* than minio until benchmarking against it named
the missing primitive — `IndexAll` took a single byte, so six JSON delimiters
meant six reads of the document. `IndexAllAny` closed it in one release, and
profiling the result named the next one: on input where matches are common, a
list of offsets is larger than the input and the store is most of the work. The
`MaskBits` family in v1.5.0 came from that, and took simdjson's structural pass
from 1.76 ms to 370 µs.

## Why this exists

The existing Go options leave most machines unserved.

| | instruction sets | operations | same answer on every one |
|---|---|---|---|
| [gonum](https://github.com/gonum/gonum) `internal/asm` | **SSE2 only** — zero `V*` instructions in the whole repo | linear algebra | — |
| [viterin/vek](https://github.com/viterin/vek) | AVX2 only, disabled entirely on macOS, arm64 is pure Go [and never will not be](https://github.com/viterin/vek/issues/12) | broad | [no](https://github.com/viterin/vek/issues/11) |
| [kelindar/simd](https://github.com/kelindar/simd) | AVX2, NEON | 7 | no |
| **this** | sse2, avx2, avx512, neon, sve2, rvv, vx, lasx, vsx | 457 | yes |

[kelindar/simd](https://github.com/kelindar/simd) deserves singling out,
because it is the closest relative: it reaches its two instruction sets the
same way this does, by auto-vectorizing C with clang and translating the result
into Plan 9 assembly, and it dispatches on a runtime CPU check rather than a
build tag. If AVX2 and NEON are all you need and `Add`, `Sub`, `Mul`, `Div`,
`Sum`, `Min` and `Max` are all you want, it is a smaller dependency with a
lower Go floor, and there is nothing wrong with it.

The difference beyond scope is what the answer is allowed to do. Summing a
float32 slice of one large value and a thousand small ones:

```
                        accelerated          portable
kelindar/simd           0x4cbebc9c    ≠      0x4cbebc20
this library            0x4cbebc98    =      0x4cbebc98
```

Reproduce it with [`docs/comparison`](docs/comparison), a separate module so
that nothing you import pulls in a second SIMD library.

That is not a defect in kelindar — it never promises otherwise, and an
auto-vectorized reduction naturally accumulates in whatever order the vector
width implies. It does mean a result computed there can change when the binary
moves to a machine with different SIMD support. This library fixes the
accumulation order instead, which is what the [accuracy contract](#accuracy)
above costs throughput to buy.

There is also a structural reason nobody covers the rest: **Go's assembler
cannot spell SVE2 or RVV.** It has no `Z` or `P` registers at all, and upstream
has deferred scalable vectors with no design. Compiling C per target and
lifting the encoded bytes into Plan 9 assembly is the only route that reaches
them, and it is why this is the only Go library with arm64 SVE2 and riscv64 RVV
numeric kernels.

## Where the obvious answer was wrong

[**docs/wrong.md**](docs/wrong.md) records 77 things a competent person would
have assumed that turned out false, and what each cost. Among them:

- A register can be reserved by *value* rather than by name, which makes it
  invisible to every compiler flag. The symptom is Go's allocator dying several
  calls later.
- A compiler builtin that silently compiles to nothing at `-O1` and above, and
  is correct at `-O0`.
- Green test lanes that had been executing no accelerated code for months.
- Four loops that were slower *after* being vectorized, one by 1700×.
- `--mattr=+sve2` removing NEON rather than adding SVE2.
- Go's own SIMD intrinsics being 4.4× *slower* than the generated assembly.
- A closure comparator costing a sort 2.5×, and the wrong fix making it worse.
- A scripted edit that silently did not apply, and would have crashed most CPUs.
- A failing test naming a kernel that was not the one with the bug.
- A test lane that was hung, not slow — thirty-two minutes at 0.1% CPU.
- `ENOSPC` with 40 GB free.

`docs/research/` carries the longer reasoning behind design decisions;
`05-decisions.md` is the decision record.

## Status

**v1.17.0.** The API is stable: every exported function keeps its name,
signature and meaning for the life of v1, and so does the numerical contract
above. [CHANGELOG.md](CHANGELOG.md) states exactly what compatibility covers
and what it excludes. [ROADMAP.md](ROADMAP.md) lists what is still open.

## License

MIT — see [LICENSE](LICENSE). Use it in anything, including commercially; keep
the copyright notice.

The only dependency is `golang.org/x/sys` (BSD-3-Clause), for CPU feature
detection. The benchmark site vendors the Datastar browser bundle, which is MIT
and carries its own notice in [`cmd/site/assets/`](cmd/site/assets/); it is not
part of the library and nothing you import pulls it in.
