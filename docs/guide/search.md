# Search, sets and bit vectors

The operations whose output is not one element per input. These are the ones
where "can this be vectorized at all?" has an interesting answer, so each
section says what the trick was.

## Binary search, for many queries at once

One binary search does not vectorize. Every probe's address comes from the
previous comparison, which is a dependency chain nothing in a vector unit
helps with — and a branchless scalar search is already close to optimal.

A *batch* is a different problem entirely:

```go
table := []float64{0, 10, 20, 30}
queries := []float64{5, 25, 10, 99}

pos := make([]int32, len(queries))
simd.LowerBoundInto(pos, table, queries)
// pos is [1 3 1 4]
```

Every query walks the same number of steps over the same table, so the loop
nest turns inside out: step on the outside, query on the inside. The inner loop
is then elementwise over the batch — one probe index per lane, one comparison,
one masked update.

Each answer is the number of elements strictly less than the query, which is
what `std::lower_bound` and `sort.SearchInts` return. A query that equals a
table entry lands *on* it, not after it: `10` above gives 1, not 2.

Two things make this fast, and it is worth separating them because only one is
about lanes. Measured against the bisection you would otherwise write, on a
65536-entry table with 16384 queries:

| tier | time | vs bisection |
|---|---|---|
| portable | 1.08 ms | — |
| SSE2 | 0.161 ms | 6.7× |
| AVX2 | 0.151 ms | 7.2× |
| AVX-512 | 0.104 ms | 10.4× |

SSE2 has no gather instruction at all, so its 6.7× is almost entirely
*branchlessness* — sixteen unpredictable branches per query become sixteen
conditional moves. The lanes are the further 1.55× from SSE2 to AVX-512. If you
take one thing from this table, it is that the branch predictor was the enemy,
not the arithmetic.

## Sorted sets

Both inputs must be sorted and free of duplicates. That is the shape a posting
list, an inverted index or a column of sorted keys is already in, and it is not
checked — verifying it costs a pass over both slices, which is what the
operation itself costs.

```go
a := []int32{1, 3, 5, 7, 9}
b := []int32{3, 4, 5, 6, 9}

dst := make([]int32, min(len(a), len(b)))
n := simd.IntersectInto(dst, a, b)
// dst[:n] is [3 5 9]
```

`DifferenceInto` keeps the elements of `a` that are not in `b`, and needs a
destination of `len(a)`.

Both panic if the destination is too small rather than truncating. That is
deliberate: the kernel uses all six argument registers the amd64 ABI provides,
so it cannot also be told the destination's length, and a silent short write
would corrupt your data rather than fail.

### Why this is not a merge

The obvious implementation walks both slices with two cursors. It is optimal in
comparisons and vectorizes on nothing — each step's decision depends on the
last one's, through both cursors.

What vectorizes is asking a different question. Take a block of eight from each
side and ask, for every element of `a`'s block at once, whether it appears
anywhere in `b`'s block: eight broadcast comparisons, no reduction and no
branch. Then retire whichever block has the smaller maximum, which is sound
because both sides are sorted, so a retired block can never match anything
further along.

The gain depends on how the two sets interleave, so here is the honest spread
rather than one number: it is ahead at 1% and 10% overlap, ahead again at 100%,
and a tie at 50% where no block ever gets to skip.

## Merging sorted streams

`MergeSortedUint32` merges two ascending `[]uint32` streams into one ascending
destination:

```go
a := []uint32{1, 3, 3, 8}
b := []uint32{2, 3, 9}

dst := make([]uint32, len(a)+len(b))
n := simd.MergeSortedUint32(dst, a, b)
// dst[:n] is [1 2 3 3 3 8 9]
```

`dst` must hold `len(a)+len(b)` elements. Equal keys from `a` are emitted
first, and duplicates are preserved. The kernel replaces the two-pointer
walk's data-dependent branch with a fixed min/max exchange ladder. It is the
merge half of merge sort and the ordered-input step in compaction and joins.

### There is still no set Union

Intersection and difference assume duplicate-free set inputs and emit a subset.
A set union must merge **and deduplicate**. `MergeSortedUint32` supplies the
ordered merge for uint32, but deliberately preserves every duplicate, including
equal values repeated within one input. If you need set semantics, merge into
scratch and compact adjacent duplicates in a second pass.

## Rank and select

The two primitives every succinct data structure is built on — wavelet trees,
FM-indexes, compressed tries — and the pair people most often get subtly wrong,
because the useful definition of rank is exclusive and the obvious
implementation is inclusive.

```go
v := []uint64{0b1011, 0b1100}

table := make([]uint64, len(v)+1)
simd.RankTableInto(table, v)

simd.Rank(v, table, 3)   // 2 — set bits strictly below position 3
simd.Select(v, table, 2) // 3 — position of the third set bit
table[len(v)]            // 5 — the total
```

Build the table once; every query afterwards is O(1) for rank and O(log n) for
select.

The table has `len(v)+1` entries and holds an *exclusive* prefix: `table[i]` is
the number of set bits in `v[:i]`. That off-by-one is the whole reason this is
a function rather than something you write inline — it is what makes `Rank` a
single addition with no special case at a word boundary, and what makes
`Select` its exact inverse.

This is a composition rather than a kernel, and says so: the table is a
population count per word through the accelerated `OnesCountInto`, then a
prefix sum through `CumSumInto`. The prefix sum runs serially, because this
library measured integer scans and found them slower than the serial loop — a
one-cycle dependency is not one worth breaking. That is a measured choice, not
a gap.

Neither `Rank` nor `Select` touches the vector unit, and neither should: a
query reads two words. All the vector work is in building the table.

## Sliding-window extremes

```go
a := []float64{5, 1, 4, 2, 8, 3}

dst := make([]float64, len(a)-3+1)
simd.RollingMinInto(dst, a, 3)
// dst is [1 1 2 2]
```

There are `len(a)-window+1` outputs. The extreme is IEEE 754-2019 minimum, the
same one `Min` uses: a window containing a NaN yields NaN, and -0 is smaller
than +0.

### This one has a limit, and you should know where

The textbook sliding-window minimum is a monotonic deque: two amortized
comparisons per element, independent of the window. This does `window-1`
elementwise passes — more arithmetic, but each pass is contiguous and
vectorized, and they are tiled so the block being accumulated stays in L1.

So it does sixteen windows at a time where the deque does one, and which wins
turns on the *window*, not on `n`. Measured on a Zen 5 at a million float64:

| window | this | hand-written deque | |
|---|---|---|---|
| 4 | 0.65 ms | 8.35 ms | **12.8×** |
| 8 | 1.35 | 8.90 | 6.6× |
| 16 | 2.79 | 8.62 | 3.1× |
| 32 | 5.65 | 8.44 | 1.5× |
| 64 | 11.2 | 8.33 | **0.75×** |
| 256 | 44.7 | 8.21 | **0.18×** |

The deque's column barely moves — that is the point of it. Above a window of
about 48, write the deque; it is fifteen lines.

This function does not switch to one itself. A deque needs an index ring
proportional to the window, which would add hidden workspace to an otherwise
caller-owned-output operation, and getting IEEE
minimum out of one is subtle in a way that would not show up in testing:
"pop the back while it is worse" does nothing when neither operand orders, so a
plain deque holds a NaN without ever reporting it. Three implementations of one
operation is a liability; a documented limit is not.

## Sparse matrices

```go
//  [ 1 0 2 ]       [1]     [7]
//  [ 0 3 0 ]   *   [2]  =  [6]
//                  [3]
values := []float64{1, 2, 3}
colIdx := []int32{0, 2, 1}
rowPtr := []int32{0, 2, 3}

dst := make([]float64, len(rowPtr)-1)
simd.SpMVInto(dst, values, colIdx, rowPtr, x)
```

Compressed sparse row, the layout every sparse library stores matrices in.
`SparseDot` is one row on its own if you want to drive the loop yourself.

The row is the kernel and the matrix is not, because a whole SpMV needs five
pointers and their lengths — past the six integer registers the amd64 ABI
passes arguments in. So the row loop stays in Go, which costs one call per row.
That trade is fine for the rows sparse matrices actually have, tens to hundreds
of nonzeros, and wrong for a matrix whose rows are two elements long.

Be clear-eyed about the gain: it is about **1.1×**, reaching 1.17× on a long
row, and never a loss. It is gather-bound, and a hardware gather is barely
faster than the scalar loads it replaces. No arrangement of the multiply-add
changes that. What you get for free is the accumulation contract — sixteen
accumulators and a fixed tree, so an iterative solver's residual does not
depend on which machine ran it.
