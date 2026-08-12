# API guide

The package exposes ordinary Go functions over ordinary slices. Runtime
dispatch chooses an instruction-set tier; callers do not select vector widths
or initialize a backend.

Use the [tutorial](tutorial.md) first if you are deciding how to lay out data.
This page is the reference for choosing an operation once the data is already
in contiguous slices.

## Conventions

Arithmetic operations commonly use two forms:

```go
simd.Add(a, b)          // a[i] += b[i]
simd.AddInto(dst, a, b) // dst[i] = a[i] + b[i]
```

The plain form usually modifies its first slice. An `Into` form writes into
caller-owned output or uses caller-owned workspace. Reductions return values
instead of writing a slice. These are common shapes, not universal rules:
`SortInto`, for example, takes a scratch slice while sorting its first argument
in place. Read the function documentation when output length, overlap, or
malformed input matters.

## Allocation and workspace

Kernel calls and most `Into` forms are designed for hot paths: supply output
and workspace once, then reuse them. Convenience APIs can allocate when
returning a new slice, constructing a plan or workspace, or growing an append
destination.
Representative allocating calls include `Sort`, `Median`, `Quantile`, `TopK`,
`BottomK`, `Histogram`, `Bincount`, `FFT`, `RFFT`, and plan or workspace
constructors. Append-style functions reuse capacity when it is sufficient and
grow otherwise.

Caller-owned workspace is normally explicit in the signature:

```go
scratch := make([]float64, len(batch))
for _, values := range batches {
    simd.SortInto(values, scratch)
}
```

`SortInto` avoids `Sort`'s unconditional element-scratch allocation. Its
duplicate-recovery path can still allocate temporary boolean masks when a
partition is badly skewed by many copies of its pivot.

## Lengths, overlap, and malformed input

Elementwise operations generally process the shortest participating slice.
Output-shaped operations have contracts of their own:

- compaction, decoding, and parsing functions may return a count or consumed
  length;
- matrix, tile, and packed encodings may require exact dimensions or capacity;
- malformed encoded input may return a negative result or stop before the
  incomplete value;
- some destinations may alias an input, while partial overlap is unsupported
  for others.

No package-wide aliasing or truncation rule replaces the function comment.
Size destinations from the corresponding length helper where one exists.

## Find an operation by task

Every operation named in this table has a runnable example in the package's
`example_*_test.go` files. Those examples compile and run under `go test` and
render beside the API on [pkg.go.dev](https://pkg.go.dev/github.com/sebishogun/simd).

| I want to… | Call |
|---|---|
| add/scale/clamp a slice in place | `Add` `Scale` `AddScalar` `Clamp` |
| …without destroying the input | the same name with `Into` |
| do `y += a*x` in one pass (axpy) | `AddScaled` |
| **sum or multiply many slices at once** | `AddAll` `MulAll` — one pass, not one per slice |
| sort a slice | `Sort` `SortInto` (caller-supplied element scratch) |
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
| **the same, as a bitmask** | `MaskBits` `MaskBitsAny` `MaskBitsLess` — a bit per byte |
| find any of a set of bytes | `IndexAny` `IndexNotAny` `CountAny` |
| **the next byte that is not plain text** | `IndexAnyOrLess` — a set and a threshold in one pass |
| **copy text up to the byte that needs escaping** | `JSONCopyRun` — scan and copy in one pass |
| **classify a JSON document five ways at once** | `JSONMasks` `MaskWords` — quotes, backslashes, brackets, control bytes, and whitespace |
| **validate a JSON document in one kernel call** | `JSONValid`; `JSONStage1` + `JSONValidTokens` expose the staged halves |
| **escape a string into a JSON encoder's buffer** | `JSONQuote` (valid UTF-8) or `JSONCopyValid` (rejects invalid bytes), both with optional HTML-safe escaping |
| trim, fold case, validate UTF-8 | `TrimAny` `TrimSpaceASCII` `EqualFoldASCII` `ValidUTF8` |
| hex or base64 | `HexEncode` `Base64Encode` `Base64Decode` |
| checksum a buffer | `Adler32`; `CRC32C` matches `hash/crc32` Castagnoli |
| format a float shortest-form | `FormatFloat64` — encoding/json's formatting rule |
| decode an LZ4 block | `LZ4BlockDecode` — returns -1 on malformed input |
| expand run-length pairs | `RunLengthDecodeInt32` |
| **fill a slice with random values** | `RandFillU64` — xoshiro256++, deterministic per seed, not cryptographic |
| merge two sorted arrays | `MergeSortedUint32` |
| weave or unweave two byte planes | `Interleave2` `Deinterleave2`; `Transpose8x8Bytes` for 64-byte tiles |
| decode a varint stream | `VarintDecode` — returns values written and bytes consumed |
| **hash a column of keys** | `HashUint64` — seeded splitmix64 for bulk numeric keys |
| bit-transpose for compression | `Bitshuffle` `Unbitshuffle` — 64-byte tiles |
| convert to/from float16 or bfloat16 | `Float16ToFloat32Into` and friends |
| median / percentile without sorting | `Median` `Quantile`, `MedianInto` for caller-owned scratch |
| the k largest or smallest | `TopK` `BottomK` — selects, does not sort the whole input |
| histogram / count occurrences | `Histogram` `Bincount` |
| find the NaNs, sum around them | `IsNaNInto` `CountNaN` `NanSum` `NanMean` |
| shifts, rotates, popcount per element | `Shl` `Rotl` `OnesCountInto` `LeadingZerosInto` `ByteSwapInto` |
| a Fourier transform | `FFTInto` with a reusable plan; `RFFT` for real input |
| envelope / analytic signal | `HilbertInto`, then `AbsComplexInto` |
| window a signal | `Hann` `Hamming` `Blackman`, then `ApplyWindowInto` |
| convolve or correlate | `ConvolveFullInto` — picks direct or FFT by a measured crossover |
| interpolate a table | `InterpInto` — numpy's interp, clamping |
| transpose a matrix | `TransposeInto` — blocked |
| parse a CSV of integers | `IndexAll` + `ParseInts` |
| **quantize a tensor to int8** | `QuantizeInt8`, or `QuantizePerChannelInt8` for weights |
| **multiply int8 tensors** | `QMatMulInt8Into` → int32, then `RequantizeInt8Into` |
| normalize a transformer layer | `LayerNorm`, or `LayerNormInto` with gamma and beta |
| **look up many keys in a sorted table** | `LowerBoundInto` — one binary search per query, batched |
| how much two byte slices share | `CommonPrefixLen` — the LCP step of a suffix array |
| rolling minimum or maximum | `RollingMinInto` `RollingMaxInto` — see the window limit in its doc |
| **intersect or subtract sorted sets** | `IntersectInto` `DifferenceInto` — posting lists |
| rank/select over a bit vector | `RankTableInto`, then `Rank` and `Select` |
| size a varint stream before writing it | `VarintSize` `VarintLenInto` `AppendVarints` |
| **multiply a sparse matrix by a vector** | `SpMVInto`, or `SparseDot` for one CSR row |
| convert to/from fp8 | `Float32ToFloat8E4M3Into` and the e5m2 pair |
| make negative deltas small | `ZigzagEncodeInt32Into`, before varint or bit packing |
| pack a column densely | `DiffInto` → `ZigzagEncodeInt32Into` → `BitPackInto` |
| run-length encode a column | `RunLengthEncodeInt32`, or `RunStartsInto` for the mask alone |
| compare two bit vectors | `HammingDistance` — fused, not `Xor` then `PopCount` |
| convert planar RGB | `GrayscaleInto`, `RGBToUVInto` |
| **sum data arriving in chunks** | `Accumulator[T]` — bit-identical to `Sum` of the whole |
| fill a slice with portable random values | `RandomInto` — reproducible and splittable across machines |

## Supported element types

The broad numeric families cover `float32`, `float64`, signed and unsigned
integers from 8 to 64 bits, and selected operations over `complex64` and
`complex128`. Text functions accept strings or byte slices where their
signatures permit it. Narrow formats use their storage representations:
float16 and bfloat16 as `uint16`, and fp8 as `byte`.

Some hardware operations support a narrower set. Columnar validity operations,
for example, cover `float32`, `float64`, `int32`, and `int64`, matching the
full-width compress instructions available at the library's AVX-512 tier.
Generic constraints and function signatures are the authoritative type list.

## v1.19 and v1.20 additions

### Data movement

v1.19 added the pieces that move structured data without converting it:
`RunLengthDecodeInt32` expands value/count pairs into caller-owned output;
`MergeSortedUint32` merges two sorted streams while preserving duplicates;
`Interleave2` and `Deinterleave2` convert between two byte planes and
alternating bytes; and `Transpose8x8Bytes` handles independent 64-byte tiles.
`RandFillU64` fills bulk random data deterministically from a seed and is not a
cryptographic generator.

See [encodings](guide/encoding.md) for column layouts and [search and sets](guide/search.md)
for sorted-stream operations.

### Columnar codecs

v1.20 made `BitUnpackInto` select width-specialized kernels internally for bit
widths 1 through 31; callers keep the same API. `VarintDecode` decodes complete
LEB128 values until either input or output ends and returns both progress
counts, so a caller can resume at `src[consumed:]`. `HashUint64` hashes a batch
of numeric keys with a caller-supplied seed; use `hash/maphash` for one string.
`Bitshuffle` and `Unbitshuffle` transpose bits into compression-friendly planes
over complete 64-byte tiles.

The [encoding guide](guide/encoding.md) shows these operations in complete
pipelines.
