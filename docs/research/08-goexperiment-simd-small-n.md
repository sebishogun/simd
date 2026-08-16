# `GOEXPERIMENT=simd` for the small-n band

**Status: measured, favourable, not implemented.** This record exists so the
implementation step has a number behind it rather than a table copied from a
plan. Nothing in it is shipped; the open item lives in ROADMAP.md's tiers
section.

## The question

Every generated guard falls back to `internal/ref` below its threshold:

```go
func addF32AVX512Guarded(dst, a, b []float32) {
    if len(dst) < threshold { ref.AddInto(dst, a, b); return }
    addF32AVX512(dst, a, b)
}
```

`ref` is a plain Go loop, and below the threshold that loop is the whole answer.
The proposal is to route the guard's fallback through Go's own `simd/archsimd`
instead — inlinable intrinsics rather than a non-inlinable assembly call, which
is what makes them usable at n = 8 where a call costs more than the work.

Two things had to be true before any of it is worth writing: the intrinsics have
to be **bit-identical** to the portable loop, and they have to actually **win**
at the sizes where the fallback runs. Both are measurable today — Go 1.26.5 ships
`simd/archsimd` behind `GOEXPERIMENT=simd` — so they were measured before
anything was built.

## Bit-identity

Float addition is not reassociated by either path (elementwise, one operation
per lane), so identity is expected rather than hoped for. Checked anyway, over
n = 0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33 with values chosen to be
cancellation-prone (`i*0.1 + 1e-8` against `i*-0.3 + 1e30`), including every
tail length the partial-load helpers have to handle:

**Identical at every index, every n.** `LoadFloat32x8SlicePart` /
`StoreSlicePart` handle the tail without reading or writing past the slice, which
is the part a hand-written version gets wrong.

## The measurement

`f32 dst[i] = a[i] + b[i]`, the scalar loop against an `archsimd` 8-lane loop
plus a partial-load tail. Instructions retired and cycles, `perf stat -e
instructions:u,cycles:u`, three interleaved rounds in one session, 2,000,000
iterations each, minimum reported. Instructions rather than wall clock because
the machine's load average was above 1 and this repository's wall-clock noise
floor is 8.3%.

| n | scalar instr | archsimd instr | | scalar cycles | archsimd cycles | |
|---|---|---|---|---|---|---|
| 8 | 227,775,443 | 183,973,765 | **−19.2%** | 38,225,341 | 33,269,977 | −13.0% |
| 16 | 387,771,780 | 284,004,004 | **−26.8%** | 59,050,260 | 46,597,720 | −21.1% |
| 32 | 708,189,014 | 483,774,777 | **−31.7%** | 103,791,526 | 64,829,258 | −37.5% |

Every range is disjoint: the slowest archsimd round beats the fastest scalar
round at all three sizes. The gain grows with n across the band, which is what a
vector width of 8 predicts — at n = 8 one vector operation replaces eight scalar
ones but the loop scaffolding is a larger share; by n = 32 it is four vectors
against thirty-two.

## What this does NOT measure

Stated because the number above is the tempting one to quote:

- **It is a direct call, not the guarded one.** The library's fallback is reached
  through a dispatch guard, and the guard's own compare and branch are not in
  either column. The gain should carry — the guard is identical on both sides —
  but the end-to-end figure will be smaller than 19–32%.
- **One operation.** `AddInto` is the friendliest possible case: elementwise,
  one instruction per lane, no reassociation question. The reductions are the
  interesting ones and they are the ones where bit-identity is genuinely at
  risk, because `kernel.CombineTree` uses a fixed sixteen-accumulator shape that
  an intrinsics loop would have to reproduce exactly rather than merely
  correctly.
- **amd64 only, here.** `archsimd` builds for amd64 and arm64; riscv64, s390x,
  loong64 and ppc64le keep the `ref` fallback and gain nothing. Any claim made
  from this table has to say so.
- **The build tag is viral.** Behind `//go:build goexperiment.simd`, every lane
  has to run twice — tagged and untagged — or the tagged path becomes the
  vacuously-green lane `docs/wrong.md` entry 41 warned about. That cost is
  ongoing and is not in any of these numbers.

## The decision this supports

Build it, elementwise operations first, with bit-identity as a gate rather than
an expectation: an operation whose intrinsics path is not bit-identical to `ref`
does not ship for that operation. Reductions are a separate decision needing
their own measurement.

The reproducer is thirty lines and lives in this document's history rather than
in the tree, because a benchmark for a package that does not exist yet is a
maintenance cost with no reader.
