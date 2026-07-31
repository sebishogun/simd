# internal/tests — the public-API suite

Everything here calls only the exported API, which is why it can sit behind a
package boundary. Splitting it by topic keeps the repository root down to the
library itself plus what Go insists lives beside it.

| package | covers |
|---|---|
| [`arrays/`](arrays) | elementwise and scalar arithmetic, scans, predicates, thresholds, allocation checks |
| [`reduce/`](reduce) | sums, argmin/argmax, accumulators, histograms, floating-point accuracy |
| [`text/`](text) | index and count, trimming, case folding, base64, UTF-8 and UTF-16, integer and float parsing |
| [`search/`](search) | sorted-set operations, bit vectors, rank/select, rolling windows, compression |
| [`encode/`](encode) | quantization, fp8, bit packing, run-length, zigzag, varint, colour conversion |
| [`dsp/`](dsp) | Fourier transforms and windows |
| [`matrix/`](matrix) | Gemv, MatMul, packed and quantized variants, LayerNorm, sparse |
| [`docs/`](docs) | the repository's own documentation, checked against the tree — counts, identifiers, examples |

## What did not move, and why

Some tests have to stay in the root directory:

- **Tests of unexported behaviour** — `package simd`, so they must sit in the
  package's own directory.
- **`export_test.go` and its users.** That file exposes internals to tests
  beside it; `ConvolveForTest` and `SelectMinLenForTest` do not exist outside
  that directory, so the tests calling them cannot leave either.
- **The runnable examples.** `ExampleAdd` only appears on pkg.go.dev as an
  example *of package simd* if it lives beside package simd. Two of them had
  been written inside ordinary test files and were moved here by accident; the
  check in [`docs/`](docs) failed until they were put back.

## How few files can the root be

Sixty-seven, and it is there. Forty-six of those are the library — 7,236 lines
across a dozen topics, largest file 483 — and Go requires a package to live in
one directory, so the only way to fewer is fewer or larger files, not folders.

For scale, in the standard library: `math` is 67 files in one directory,
`net/http` is 71, `time` is 38. A flat package is the idiomatic Go shape for
this kind of library, and nobody splits `strings` into `strings/search`.

## Helpers

Each package carries its own. They are a few lines each — a random-slice
generator, a comparison that treats NaN as equal to itself, a sink to stop the
compiler deleting the call under test — and duplicating them costs less than a
shared package that every test directory has to import.
