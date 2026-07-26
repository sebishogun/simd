# simd

SIMD-accelerated slice operations for Go, on every architecture that has a
vector unit, **without cgo**.

```go
import "github.com/sebishogun/simd"

simd.Add(a, b)              // a[i] += b[i]      — in place, no allocation
simd.AddScaled(a, b, 0.5)   // a[i] += b[i]*0.5  — one pass over memory
total := simd.Sum(a)
sim   := simd.CosineSimilarity(x, y)
```

Ordinary functions on ordinary slices. No vector type, no lane count, no
target selection, nothing to initialize.

> **Status: early but real.** The API, the portable reference and the
> code-generation pipeline are in place and tested. Sixteen kernels are
> generated for **nine targets** — amd64 sse2/avx2/avx512, arm64 neon/sve2,
> riscv64 rvv, s390x vx, loong64 lasx, ppc64le vsx — and every architecture
> builds clean and passes `go vet` asmdecl. Everything outside those sixteen
> still runs the portable path.
>
> On this machine (Zen 5), public API, generated versus portable Go:
>
> | n | 8 | 32 | 128 | 1024 | 16384 |
> |---|---|---|---|---|---|
> | Sum | 2.39x | 4.48x | 8.68x | **13.78x** | 7.96x |
> | Dot | 2.42x | 4.55x | 8.23x | **12.13x** | 5.18x |
> | Add | 0.95x | 1.91x | 3.54x | **6.33x** | 2.61x |

## Why another one

The existing Go options leave most machines on the table:

| | ISAs covered | arm64 |
|---|---|---|
| [gonum](https://github.com/gonum/gonum) `internal/asm` | **SSE2 only** — zero `V*` instructions in the whole repo | **none** |
| [viterin/vek](https://github.com/viterin/vek) | AVX2 only, and disabled entirely on macOS | pure Go, [and never will be](https://github.com/viterin/vek/issues/12) |
| [kelindar/simd](https://github.com/kelindar/simd) | AVX2, NEON — 7 operations | yes |
| this | SSE2 → AVX-512, NEON, **SVE2**, RVV, VX, LSX/LASX, VSX | yes |

Two things make the difference. Kernels are written once in plain C and
vectorized by LLVM per target, so adding an architecture is a build-matrix row
rather than a new hand-written implementation. And because the pipeline emits
raw instruction encodings, it is not limited by what Go's assembler can spell —
which matters more than it sounds, because **Go's arm64 assembler has no
floating-point vector arithmetic at all** and cannot encode a single SVE
instruction. See [`docs/research/`](docs/research/).

## Design

**In place by default.** The plain name mutates its first argument and
allocates nothing. The `Into` suffix takes a destination:

```go
simd.Mul(a, b)              // a[i] *= b[i]
simd.MulInto(dst, a, b)     // dst[i] = a[i] * b[i]
```

**This package never allocates.** Every function writes only into memory you
supplied. There is no variant that returns a new slice, and
[a test enforces it](alloc_test.go) across the whole API.

**Generic over the element type** — `float32`, `float64`, `int32`, `int64` —
so there are no `AddFloat32` / `AddFloat64` name suffixes to remember.

**Results do not depend on the hardware.** Every operation is bit-identical on
every instruction set, including for NaN, ±Inf, ±0 and denormals. Reductions
use a fixed accumulation order that a 128-bit and a 512-bit machine both
reproduce exactly, so a computation cannot change answer because it moved to a
different server.

This costs some throughput and is deliberate. The alternative is what
[vek does](https://github.com/viterin/vek/issues/11): its vectorized body and
its scalar remainder loop disagree on NaN, so the answer depends on the length
of the input. Operations that trade reproducibility for speed here are named
`Fast*` and document their error bound.

**The right instructions are chosen at startup** from the CPU actually running
the program. A binary built on a machine with AVX-512 runs correctly on one
without it. Nothing to configure, nothing to build twice.

## Operations

209 exported functions. The plain name works in place; the `Into` suffix takes
a destination.

**Elementwise** — `Add` `Sub` `Mul` `Div` `Minimum` `Maximum` `Abs` `Neg`
`Sqrt` `Reciprocal` `Reverse`

**With a scalar** — `Scale` `AddScalar` `SubScalar` `DivScalar` `Clamp` `Fill`
`Zero` `Ramp` `Tile`

**Rounding** — `Floor` `Ceil` `Trunc` `Round` `RoundToEven`

**Transcendental** — `Exp` `Exp2` `Expm1` `Log` `Log2` `Log10` `Log1p` `Cbrt`
`Pow` `Hypot` `Sin` `Cos` `Tan` `Asin` `Acos` `Atan` `Atan2` `Sinh` `Cosh`
`Tanh` `Sigmoid`

**Fused** — `AddScaled` (AXPY: `a += b*s` in one pass, not two) · `Lerp`

**Scans** — `CumSum` `CumProd` `CumMin` `CumMax` `DiffInto`

**Reductions** — `Sum` `Prod` `Dot` `SumSquares` `L1Norm` `Norm` `Min` `Max`
`MinMax` `ArgMin` `ArgMax` `Median` `Quantile`

**Comparisons → `[]bool`** — `EqualInto` `NotEqualInto` `LessInto`
`LessEqualInto` `GreaterInto` `GreaterEqualInto`, each with a `Scalar` variant

**Boolean vectors** — `All` `Any` `CountTrue` `AndMask` `OrMask` `XorMask`
`NotMask` `SelectInto`

**Data movement** — `GatherInto` `ScatterInto` `ConvertInto`

**Statistics** — `Mean` `Variance` `SampleVariance` `StdDev` `SampleStdDev`
`Covariance` `Correlation` `LinearRegression`

**Distance** — `Distance` `SquaredDistance` `ManhattanDistance`
`CosineSimilarity` · **Rescaling** — `Normalize` `Standardize` `Rescale`

**Signal** — `PolyEvalInto` (Horner) `ConvolveInto` `CorrelateInto`
`MovingAverageInto` `EMAInto` · **Linear algebra** — `MatMulInto`

**Quadrature and ODEs** — `Trapezoid` `Simpson` · `EulerStep` `RK4Step`
`VerletStep` with a reusable workspace so stepping never allocates

**Machine learning** — `Softmax` `LogSumExp` `ReLU` `LeakyReLU` `Softplus`
`SiLU` `GELU` `LayerNorm` `RMSNorm`

**Bytes and bits** — `IndexByte` `LastIndexByte` `Count` `Equal` `Compare`
`PopCount` `And` `Or` `Xor` `AndNot`, drop-in compatible with `bytes`

**Text scanning** — `IndexAll` (the structural-index primitive a parser is
built from) `IndexAny` `CountAny` `Index` `IsASCII` `ValidUTF8` `ToUpperASCII`
`ToLowerASCII` `EqualFoldASCII` `ReplaceByte` `HexEncode` `HexDecode`

### Compared with vek

Every operation `viterin/vek` offers is here, plus a good deal more.

| vek has | here |
|---|---|
| arithmetic, scalar arithmetic, `Abs` `Neg` `Inv` `Sqrt` | ✅ |
| `Round` `Floor` `Ceil` `Pow` `Exp` `Log` `Log2` `Log10` `Sin` `Cos` | ✅ |
| `Sum` `CumSum` `Prod` `CumProd` `Mean` `Median` `Quantile` `Min` `Max` `ArgMin` `ArgMax` | ✅ |
| `Dot` `Norm` `CosineSimilarity` `ManhattanDistance` `MatMul` | ✅ |
| `Eq` `Neq` `Gt` `Gte` `Lt` `Lte` and the `[]bool` operations | ✅ |
| `Zeros` `Ones` `Range` `Repeat` `Gather` `Scatter`, casts | ✅ |

Not in vek: the fused `AddScaled` and `Lerp`, `MinMax`, `Clamp`, `SumSquares`,
`L1Norm`, `Reverse`, the hyperbolic and inverse trigonometric functions,
variance and standard deviation, `Covariance` `Correlation` `LinearRegression`,
`Standardize` `Rescale`, the whole signal group, quadrature and ODE
integrators, every machine-learning function, and the entire bytes and text
domain.

## Diagnostics

```console
$ go run ./cmd/simdinfo
amd64 tier=avx512 available=[scalar sse2 avx2 avx512]

$ GOSIMD=sse2 go run ./cmd/simdinfo
amd64 tier=sse2 available=[scalar sse2 avx2 avx512] forced

$ SIMD_DISABLE=avx512 go run ./cmd/simdinfo
amd64 tier=avx2 available=[scalar sse2 avx2] disabled=[avx512]
```

`GOSIMD` pins a tier, for benchmarking and for bisecting a numerical
difference. It can only select *down*: naming a tier the CPU lacks falls back
to portable Go and says so, rather than executing instructions it cannot
decode.

`SIMD_DISABLE` masks tiers out, which is useful on CPUs where wide vectors
cause frequency throttling.

`-tags purego` builds with no assembly at all.

## Contributing

Consumers only need `go get`; the generated assembly is committed and the
toolchain lives in a [separate module](tools/go.mod) so it is never a
dependency of your build. Contributors need clang.

```console
make verify        # fmt, vet, tests, purego, and every tier this CPU supports
make test-tiers    # the suite once per instruction set
make test-cross    # arm64, riscv64, s390x, ppc64le under emulation
make benchcmp      # accelerated vs portable Go, via benchstat
```

The research behind the design decisions is in
[`docs/research/`](docs/research/) — what Go's assembler can and cannot encode
per architecture, how the clang-to-Plan 9 pipeline works and what it costs,
what every comparable library got wrong, and the ABI rules generated code has
to respect.

## License

TBD
