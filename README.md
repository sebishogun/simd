# simd.go

[![Latest Release](docs/assets/badges/release.svg)](https://github.com/sebishogun/simd/releases/latest)
[![Go Reference](https://pkg.go.dev/badge/github.com/sebishogun/simd.svg)](https://pkg.go.dev/github.com/sebishogun/simd)
[![CI](https://github.com/sebishogun/simd/actions/workflows/ci-local.yml/badge.svg)](https://github.com/sebishogun/simd/actions/workflows/ci-local.yml)

**Fast slice math and text scanning for Go, using the vector unit your CPU
already has — without cgo.**

`simd` applies runtime-selected SIMD kernels to ordinary Go slices. The v1.20.0
tree contains 493 exported functions and 6,931 generated kernels for SSE2,
AVX2, AVX-512, NEON, SVE2, RVV, VSX, VX, and LASX. A portable Go path covers
every operation and every unsupported target.

## Install

```sh
go get github.com/sebishogun/simd
```

Go 1.25 or later. No cgo, no C toolchain, no build tags, no `GOEXPERIMENT`. The
kernels are compiled ahead of time and committed as assembly, so this is an
ordinary Go dependency with one transitive import (`golang.org/x/sys`, for CPU
feature detection). Dispatch is one static table per operation, so the linker
keeps only the operations a program actually calls: a binary using three
operation families does not retain all 6,931 kernels.

## Quick start

```go
package main

import (
    "fmt"

    "github.com/sebishogun/simd"
)

func main() {
    a := []float32{1, 2, 3, 4}
    b := []float32{10, 20, 30, 40}

    simd.Add(a, b) // a[i] += b[i]

    fmt.Println(a)
    fmt.Println(simd.Sum(a))
    fmt.Println(simd.Index("alpha,beta", ","))
}
```

There is no vector width to select and nothing to initialize. CPU features are
detected once during package initialization.

## API model

Arithmetic calls commonly come in an in-place form and an `Into` form:

```go
simd.Add(a, b)          // a[i] += b[i]
simd.AddInto(dst, a, b) // dst[i] = a[i] + b[i]
```

This is the dominant convention, not a rule for every function. Reductions
return a scalar. Decoders return progress counts. `SortInto` takes workspace
while sorting its first argument. Matrix, tile, and packed-codec operations
have explicit sizing rules.

Generated kernels do not allocate. Most `Into` forms let a hot path provide and
reuse output or workspace. Convenience functions that return a slice, build a
plan, or grow an append destination can allocate; examples include `Sort`,
`TopK`, `Histogram`, `FFT`, and plan constructors. See
[allocation and workspace](docs/api.md#allocation-and-workspace).

Elementwise operations generally stop at the shortest slice. That rule does not
apply to every decoder, matrix, compaction, or overlapping-slice operation; the
function comment is the contract. The [API guide](docs/api.md) documents the
cross-cutting rules and lists operations by task.

## Documentation

| Workload | Guide |
|---|---|
| data layout, batching, buffer reuse, and fusion | [tutorial](docs/tutorial.md) |
| arithmetic, reductions, sorting, filtering, nullable columns | [arrays and reductions](docs/guide/arrays.md) |
| text search, parsing, UTF-8, JSON, hex, base64 | [text and bytes](docs/guide/text.md) |
| batched search, sorted sets, sparse data, bit vectors | [search, sets and bit vectors](docs/guide/search.md) |
| quantization, packed columns, varints, shuffles | [encodings](docs/guide/encoding.md) |
| matrices, FFTs, windows, convolution | [signal and matrices](docs/guide/signal.md) |
| every operation by task | [API guide](docs/api.md) |
| architecture tiers and generated coverage | [platforms](docs/platforms.md) |
| complete programs | [examples](docs/examples/) |
| adding and verifying a kernel | [kernel guide](docs/kernels.md) |

## What is included

| Area | Representative work |
|---|---|
| arrays and reductions | arithmetic, masks, scans, sums, norms, statistics, sorting |
| text and structured data | byte search, UTF-8, JSON stages, CSV conversion, hex, base64 |
| linear algebra and DSP | dot products, GEMV/GEMM, sparse rows, FFT, windows, convolution |
| search and succinct data | batched lower bounds, sorted sets, rank/select, common prefixes |
| columnar and ML codecs | validity bitmaps, quantization, narrow floats, bit packing, varints, RLE, bitshuffle |

Element types: `float32` `float64` `int8` `int16` `int32` `int64` `uint8`
`uint16` `uint32` `uint64` `complex64` `complex128`, plus bytes, text and the
narrow float formats.

This package is not a BLAS, tensor library, dataframe, or autodiff framework.
It supplies whole-slice primitives and leaves ownership, data models, and
control flow to the caller. Higher-level projects built on it are listed below.

## When it pays

The crossover is roughly 16 to 64 elements depending on the operation. Below it
the library runs a plain Go loop, because crossing into assembly costs about
1.4 ns and cannot be inlined. Batches of hundreds or thousands of elements are
the intended shape. Calling a whole-slice operation once per element is slower
than writing the loop; [hand over batches](docs/tutorial.md#1-hand-over-batches-not-elements).

At large sizes, memory traffic matters more than arithmetic. Prefer one fused
call such as `AddScaled` or `AddAll` to multiple passes. Data layout matters
more still: a struct-of-arrays gives the vector unit contiguous lanes; an
array-of-structs often does not.

An optional vector-type escape hatch exists under Go's SIMD experiment on
amd64 only. It is for expressions absent from the whole-slice catalog, not a
replacement for the generated kernels. See
[the platform reference](docs/platforms.md#optional-go-vector-type).

## Accuracy

Core elementwise operations, integer codecs, text operations, checksums, and
fixed-order reductions return the same bits on every generated tier and on the
portable path. Floating-point reductions use sixteen accumulators and the same
combine tree at every vector width. `Dot` does not contract into FMA.

The exceptions are explicit:

- Transcendentals state an error bound against the standard library rather than
  promising identical bits.
- `Fast*` transcendental functions permit up to 3.5 ULP and may vary by
  architecture because they use fused multiply-add. They do not use
  `-ffast-math` and retain documented IEEE behavior.
- Sort and order-statistic functions treat `-0` and `+0` as equal; their order
  within that tie can differ while the sorted values remain equal under Go's
  comparison.

Function comments define any narrower exception. The conformance suite checks
generated kernels against the portable reference at every tier rather than
assuming the compiler preserved the contract.

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

Every generated tier is differential-tested against the portable
implementation. Emulation proves execution and semantics, not timing; where the
table says *unmeasured*, no wall-clock claim in this repository applies.

The **[platform reference](docs/platforms.md)** gives the source-backed
per-architecture inventory, fallback behavior, ABI limits, OS support, and the
amd64-only experimental Go vector type.

Have one of the unverified machines? See
[Reporting a hardware run](CONTRIBUTING.md#reporting-a-hardware-run). One real
run is more useful than any amount of emulation, and failures are useful data.

## Testing

`make verify` runs formatting, vet, the public tests, the pure-Go build, and
every native tier the current CPU can execute. `make menu` lists the remaining
verification targets and says which tools or host capabilities they need.

The deeper gates differentially test generated kernels against the portable
reference, compare tiers directly, fuzz adversarial values, inspect stack and
reserved-register use, and disassemble every kernel to ensure its instructions
match the CPU feature that gates it. Emulated lanes run `simdinfo
-require-accelerated` before accepting a pass, so a lane cannot report green
after silently selecting scalar code.

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
docs/                tutorial, API/platform references, guides, examples, records
testdata/bench/      recorded benchmark baseline, per GOARCH
testdata/hardware/   one report per machine that has run on real silicon
```

**The C is the source; the assembly is the output.** A kernel is written once
in [`csrc/`](csrc), compiled per instruction set by [`tools/`](tools), and
committed under [`internal/`](internal) so that using this library needs no C
toolchain. Every generated `.s` names the C file it came from and the target it
was built for, and none of them should be edited by hand — `make codegen`
regenerates them.

Public-API suites live under `internal/tests/<topic>/`. Tests of unexported
behavior, `export_test.go` hooks, and runnable pkg.go.dev examples remain beside
the package because Go requires that placement.

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

These projects use the kernels as components of a larger algorithm rather than
re-exporting the slice API. Performance figures are from each project's own
amd64 benchmark record; follow the link for corpus, configuration, and losing
cases.

### Released libraries

| Project | Workload | How `simd` is used | Measured scope |
|---|---|---|---|
| [**simdblas**](https://github.com/sebishogun/simdblas) | BLAS backend for [gonum](https://github.com/gonum/gonum) | Whole-slice reductions and matrix kernels under `blas32.Use` / `blas64.Use` | Dense routines and end-to-end gonum decompositions; covariance plus Cholesky 4.36× and QR up to 1.98× in its recorded workloads |
| [**simdjson**](https://github.com/sebishogun/simdjson) | Indexed JSON parsing and an `encoding/json`-compatible surface | JSON classification, validation, mask, string, and number kernels | Corpus results vary by shape; it publishes wins, ties, and cases where goccy, sonic, gjson, or the standard library is the better choice |
| [**simdcsv**](https://github.com/sebishogun/simdcsv) | CSV reading with byte-slice fields | One delimiter scan per unquoted record | 1.49–1.92× `encoding/csv` on recorded unquoted shapes; all-quoted four-column input is 0.81× |
| [**simdvec**](https://github.com/sebishogun/simdvec) | Exact embedding search | The entire index scan is one `GemvParallelInto`, followed by selection | 18.0–38.4× the recorded `[][]float32` dot-and-sort loop; intentionally not an approximate index |

### Active projects

These repositories are public and use `simd` v1.20.0, but do not yet have a
GitHub release.

| Project | Workload | Kernel pipeline | Recorded result |
|---|---|---|---|
| [**simdhttp**](https://github.com/sebishogun/simdhttp) | HTTP/1.1 request-head parsing | One structural scan, then boundary validation and zero-copy fields | Near level with `net/http` on a typical nine-header request; 4.7× on the 100-header shape |
| [**simdcbor**](https://github.com/sebishogun/simdcbor) | RFC 8949 CBOR decode, skip, and canonical encode | Two-stage item indexing plus UTF-8 and bulk-copy kernels | 1.35–1.84× fxamacker decode on its four recorded shapes |
| [**simdparquet**](https://github.com/sebishogun/simdparquet) | Parquet RLE/bit-packed hybrid decode | `BitUnpackInto`, `RunLengthDecodeInt32`, and `VarintDecode` behind format-aware thresholds | 1.11–1.18× its byte-at-a-time reference on recorded level/index pages |
| [**simdimage**](https://github.com/sebishogun/simdimage) | Planar image grayscale and separable box blur | `GrayscaleInto`; row-wise `Add`/`Sub` for the vertical blur | 19.4× scalar grayscale and 1.45× scalar vertical blur at 1920×1080 |
| [**simdlogs**](https://github.com/sebishogun/simdlogs) | Columnar log storage and query execution | Bit packing, RLE, varints, hashing, bitshuffle, JSON ingest, and vector predicate scans | Its 3-million-row VictoriaLogs comparison reports wins on every measured query class; exact ratios and engine/wire separation are maintained in that repository |

`simdjson` also feeds requirements back into this package. Multi-delimiter JSON
scanning produced `IndexAllAny`; dense structural matches produced the
`MaskBits` family; staged validation produced `JSONStage1`, `JSONValidTokens`,
and finally the fused `JSONValid`. The dependent was measured first, and the
missing primitive was added here only after the profile named it.

## Why this exists

The existing Go options leave most machines unserved.

| | instruction sets | operations | same answer on every one |
|---|---|---|---|
| [gonum](https://github.com/gonum/gonum) `internal/asm` | **SSE2 only** — zero `V*` instructions in the whole repo | linear algebra | — |
| [viterin/vek](https://github.com/viterin/vek) | AVX2 only, disabled entirely on macOS, arm64 is pure Go [and never will not be](https://github.com/viterin/vek/issues/12) | broad | [no](https://github.com/viterin/vek/issues/11) |
| [kelindar/simd](https://github.com/kelindar/simd) | AVX2, NEON | 7 | no |
| **this** | sse2, avx2, avx512, neon, sve2, rvv, vx, lasx, vsx | 493 | yes |

[kelindar/simd](https://github.com/kelindar/simd) is the closest relative: it
also auto-vectorizes C with clang, translates the result into Plan 9 assembly,
and dispatches at runtime. If AVX2 and NEON cover your deployment and its seven
operations cover your workload, it is the smaller dependency and has a lower Go
floor.

The difference beyond scope is what the answer is allowed to do. Summing a
float32 slice of one large value and a thousand small ones:

```
                        accelerated          portable
kelindar/simd           0x4cbebc9c    ≠      0x4cbebc20
this library            0x4cbebc98    =      0x4cbebc98
```

Reproduce it with [`docs/comparison`](docs/comparison), a separate module so
that nothing you import pulls in a second SIMD library.

kelindar does not promise tier-independent reduction bits. An auto-vectorized
reduction naturally accumulates in the order implied by its vector width. This
library fixes that order instead, which is the throughput cost behind the
[accuracy contract](#accuracy).

Go's assembler cannot spell SVE2 or RVV vector registers. This project compiles
one C source per target and lifts the encoded instruction bytes into Plan 9
assembly, which is how the ordinary Go dependency reaches those tiers without
cgo.

## Engineering record

[**docs/wrong.md**](docs/wrong.md) records 79 things that measurement disproved,
including changes that were deleted rather than shipped. Examples:

- Green test lanes had executed no accelerated code for months.
- Four loops became slower after vectorization; one regressed by 1700×.
- A closure comparator cost a sort 2.5×, and the first attempted fix was slower.
- Go's experimental SIMD intrinsics lost to generated assembly in the public
  call path they were intended to replace.
- Reserved registers, caller-frame writes, and constant-pool rewrites produced
  failures that only appeared on one backend.

`docs/research/` carries the longer reasoning behind design decisions;
`05-decisions.md` is the decision record.

## Status

**v1.20.0.** The API is stable: every exported function keeps its name,
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
