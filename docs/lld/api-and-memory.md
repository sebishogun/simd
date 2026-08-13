# LLD: API and memory

Implementation-level contracts for the public API: operation families,
naming, lengths, allocation, ownership, overlap, generic constraints, and
error behavior. The user-facing reference is
[`docs/api.md`](../api.md); this page states the rules the implementation
actually follows, with the source that backs each one.

## Operation families

| Family | Shape | Examples |
|---|---|---|
| elementwise arithmetic | mutate first slice, or `Into` form | `Add`, `AddInto`, `Scale`, `Clamp` |
| n-ary | one pass over several slices, left to right | `AddAll`, `MulAll` (arity 3–4 kernels; longer calls fold in groups of four) |
| reductions | return a scalar; fixed 16-accumulator tree for floats | `Sum`, `Dot`, `MinMax`, `ArgMin` |
| comparisons/masks | write `[]bool` per element | `LessMask`, `EqualScalarMask` |
| decoders/parsers | return counts or consumed lengths | `VarintDecode`, `LZ4BlockDecode`, `RunLengthDecodeInt32` |
| matrix/DSP | explicit dimensions, caller-sized output | `GemvInto`, `MatMulInto`, `ConvolveFullInto` |
| sorted sets | inputs sorted; return count written | `IntersectInto`, `DifferenceInto` |
| columnar codecs | exact capacity or tile-multiple rules | `BitPackInto`, `Bitshuffle`, `JSONMasks` |
| random | counter-based, seed-dependent, reproducible | `RandomInto`, `RandFillU64` |

The kernel-side shapes are the `Set`, `Ops`, `Bytes`, `Convert`, `Mask`,
`Complex`, and `ComplexParts` structs in
[`internal/kernel/kernel.go`](../../internal/kernel/kernel.go); each field's
comment states its exact contract.

## In-place / Into / reducer / decode conventions

- The plain form commonly mutates its first slice (`Add(a, b)` is
  `a[i] += b[i]`); the `Into` form writes caller-owned output
  (`AddInto(dst, a, b)` is `dst[i] = a[i] + b[i]`). This is the dominant
  convention, not a universal rule — `SortInto` takes a scratch slice while
  sorting its first argument in place.
- Reductions return values rather than writing a slice.
- Decoders report progress instead of or in addition to writing:
  `VarintDecode` returns `(n, consumed)` so a caller can resume at
  `src[consumed:]`; `LZ4BlockDecode` returns `-1` on malformed input.
- Matrix, tile, and packed-codec operations take their dimensions as
  arguments and size the destination from the corresponding length helper
  where one exists.

The function comment is the contract wherever a convention does not apply.

## Lengths

- Elementwise operations generally process the **shortest slice**; the
  generated guard clamps to `min(len(dst), len(a), ...)`.
- Operations whose output is not the same length as the input declare two
  lengths in the manifest (`CArgs`), turning the clamp off: `Diff` writes one
  fewer element, `RollingMin`/`RollingMax` write `len(a)-window+1`,
  `BitPackInto` writes fewer words than it reads, reductions read an
  arbitrarily long input.
- Output-shaped operations may require exact capacity (`JSONQuote` needs
  `6*len(b)`; `MaskBits` needs `(len(b)+7)/8`; `DtoaF64` needs 25 bytes),
  may do nothing on invalid dimensions, or may stop before an incomplete
  value. The manifest's `RefWhen` carries size relations between
  differently-shaped parameters into the guard.
- A length mismatch that would silently process the wrong number of elements
  is the failure mode the two-length rule exists for; see
  [`docs/kernels.md`](../kernels.md) (the `sum_lanes` and `bitpack` cases).

## Allocation, ownership, and scratch

- Kernel calls do not allocate. Most `Into` forms are caller-owned fast
  paths: supply output and workspace once and reuse them.
- Convenience functions can allocate when returning a slice, constructing a
  plan or workspace, or growing an append destination: `Sort`, `Median`,
  `Quantile`, `TopK`, `BottomK`, `Histogram`, `Bincount`, `FFT`, `RFFT`, and
  the plan/workspace constructors. Append-style functions reuse capacity when
  sufficient and grow otherwise.
- `SortInto` avoids `Sort`'s element-scratch allocation; its
  duplicate-recovery path can still allocate temporary boolean masks when a
  partition is badly skewed by many copies of its pivot.
- Workspace passed in belongs to the caller for the duration of the call;
  the library never retains it.

## Overlap and aliasing

There is no package-wide aliasing rule. Some destinations may alias an input;
partial overlap is unsupported for others. The function comment is the
contract. Kernel C sources mark every pointer `__restrict`, so the
vectorizer's assumption of no aliasing matches the wrapper-level promise.

## Generic constraints

- The numeric families cover `float32`, `float64`, and the signed and
  unsigned integers 8–64 bits, parameterised by the `Number` constraint and
  dispatched through the single type switch in `dispatch.go`.
- Float-only operations (transcendentals, `Div`, `Sqrt`, `Norm`, `Dot`)
  leave their fields nil on non-float instantiations; the exported API
  constrains callers, so a nil field is never reached.
- Complex numbers are their own groups (`Complex`, `ComplexParts`) because
  ordering and most of `Ops` do not apply; the split by whether a signature
  mentions the real component type is what keeps the generic API inferable.
- `Convert` handles float16/bfloat16 (as `uint16`) and fp8 (as `byte`);
  narrow formats use their storage representations.
- Go's semantics, not C's, where they differ: shifts at or above the element
  width give zero (or −1 for an arithmetic shift of a negative value),
  `LeadingZeros`/`TrailingZeros` of zero are the element width, integer
  division truncates toward zero, and `Rotl`/`Rotr` follow `math/bits`.

## Errors and panics

The package does not return error values; contracts are expressed with
counts, negative results, and documented panics:

- malformed input → negative or partial results: `LZ4BlockDecode` returns
  `-1` on malformed input; `Base64Decode` returns `-1` when the destination
  is too short or the input is not valid base64; `VarintDecode` stops at
  input or output end and reports both counts.
- documented panics: `DivScalar` states that integer division truncates
  toward zero and panics on a zero divisor; the same rule follows from Go
  semantics for the other division forms. No exported function panics
  without its comment saying so.
- nil slices are valid inputs with length zero; some wrappers check a nil
  dispatch slot before calling it (`PartitionInto` is an example) rather
  than relying on the kernel.
- aliasing that is allowed is stated per function: `DivScalarInto`
  documents "dst may alias a", and the same class of comments appears where
  the implementation permits it.
- The numeric contract (rules 1–6 in `internal/kernel/kernel.go`) governs
  bit identity, reduction shape, `Dot`'s no-FMA rounding, `Fast*` bounds,
  and the transcendental ULP bounds; the README's Accuracy section and
  `docs/api.md` are the user-facing statements of the same rules.
