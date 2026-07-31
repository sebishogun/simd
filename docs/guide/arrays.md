# Arrays and reductions

The bread and butter: arithmetic over a whole slice, and turning a whole slice
into one number. If you only read one guide, read this one — the conventions
here apply to all four hundred and fifty operations.

## In place, or into somewhere else

Every operation comes in two spellings. The plain name modifies its first
argument:

```go
a := []float64{1, 2, 3}
simd.Add(a, []float64{10, 20, 30})
// a is now [11 22 33]
```

The `Into` name writes to a destination you supply and leaves the inputs alone:

```go
dst := make([]float64, 3)
simd.AddInto(dst, []float64{1, 2, 3}, []float64{10, 20, 30})
// dst is [11 22 33], both inputs unchanged
```

That is the entire convention. There is no options struct, no builder, and no
third form.

The reason `Into` takes a destination rather than returning one is that it must
not allocate. A function that returns a fresh slice allocates once per call,
and in a loop over a million rows that is a million allocations and the garbage
collector's problem afterwards. Handing in the destination lets you allocate
once outside the loop and reuse it, which is what makes the zero-allocation
promise possible at all.

Lengths are handled by taking the minimum: if `dst` is shorter than the inputs,
you get a short answer rather than a panic. Nothing here grows a slice for you.

## Fuse, do not chain

This is the mistake that costs the most, and it is invisible if you only count
arithmetic.

```go
// Three passes over memory.
simd.AddInto(dst, a, b)
simd.Add(dst, c)
simd.Add(dst, d)

// One pass.
simd.AddAll(dst, a, b, c, d)
```

Both do the same additions. The first reads and writes `dst` three times over,
and once your slices are larger than the last-level cache the cost is entirely
in that traffic — the adds are free by comparison, because the CPU is waiting
on memory either way. `AddAll` and `MulAll` take any number of slices and make
one pass.

The accumulation is strictly left to right, `((a+b)+c)+d`, so the answer is
bit-identical to writing the chained calls out by hand. Floating-point addition
is not associative, so that is a real promise rather than a formality.

The same idea is why `AddScaled` exists:

```go
simd.AddScaled(y, x, 0.5) // y[i] += x[i] * 0.5
```

That is axpy. Doing it as a multiply pass and then an add pass reads `x` twice
and `y` twice; this reads each once.

## Reductions

```go
total := simd.Sum(a)
mean  := simd.Mean(a)
sd    := simd.StdDev(a)
lo, hi := simd.MinMax(a)
```

`MinMax` returns both extremes from a single pass, which is the point of it
existing separately from `Min` and `Max`. On a slice that does not fit in cache
two passes cost twice, however cheap the comparison is.

### The accumulation order is fixed, and that is the interesting part

A vectorized sum does not add left to right. It keeps several running totals in
parallel and combines them at the end, because a single accumulator makes every
add wait for the one before it. Different vector widths naturally give
different groupings — and therefore different rounding — so the same code would
produce different answers on different machines.

This library does not allow that. Every floating-point reduction uses exactly
sixteen accumulators and combines them with a fixed 16→8→4→2→1 tree, on every
architecture, whatever the hardware width. Element *i* always contributes to
accumulator *i mod 16*. An AVX-512 machine and a 128-bit NEON machine return
the same bits.

The cost is that a very wide machine cannot use all its lanes on a reduction.
That is deliberate. A result that changes when you deploy to a different
instance type is a debugging problem that lasts weeks, and the throughput was
not worth it.

Integer reductions have no such constraint, because integer addition is
associative and the order cannot be observed.

## Statistics and NaN

The NaN-aware functions take their scratch space as arguments, for the same
reason `Into` does:

```go
a := []float64{1, math.NaN(), 3}
mask := make([]bool, len(a))
scratch := make([]float64, len(a))

n := simd.CountNaN(a, mask)
mean, count := simd.NanMean(a, scratch, mask)
// mean is 2, count is 2
```

`NanMean` returns how many values it averaged over as well as the average,
because "the mean of the non-NaN values" is not meaningful without knowing how
many there were.

## Sorting, and sorting by something else

```go
simd.Sort(a) // in place, allocates a scratch slice internally
```

In a loop, hand the scratch in so it happens once:

```go
scratch := make([]float64, len(batch))
for _, batch := range batches {
    simd.SortInto(batch, scratch) // sorts batch in place, scratch is workspace
}
```

Note that `SortInto` does *not* mean "sort from src into dst" — it is `Sort`
with the workspace supplied. The contents of `scratch` afterwards are
unspecified.

When several columns share one ordering, sort the keys and permute the rest:

```go
order := make([]int32, len(score))
simd.Argsort(order, score)

sorted := make([]string, len(names))
for i, j := range order {
    sorted[i] = names[j]
}
```

For numeric columns the permutation step is itself an operation —
`simd.GatherInto(dst, column, order)` — which is a good deal faster than the
loop above.

## Keeping only what passes a test

The idiomatic shape is a comparison into a `[]bool`, then a compaction:

```go
mask := make([]bool, len(a))
simd.GreaterScalarInto(mask, a, 0)

dst := make([]float64, len(a))
n := simd.CompressInto(dst, a, mask)
// dst[:n] holds the positive values
```

There is also `FilterInto`, which takes an ordinary Go predicate:

```go
n := simd.FilterInto(dst, a, func(v float64) bool { return v > 0 })
```

That is convenient and it is not fast — a function call per element cannot be
vectorized, and the whole point of the compare-and-compress form is that
neither half has a branch in it. Use `FilterInto` when the predicate is
genuinely arbitrary and the slice is small; use the two-step form on anything
hot.

The measured difference is worth knowing: `CompressInto` runs at about 19 GiB/s
regardless of how many elements match, while the equivalent scalar filter loop
ranges from 12 GiB/s down to 1.3 GiB/s depending on how predictable the branch
is. The vector version does not care what the data looks like. That is usually
the more valuable property.
