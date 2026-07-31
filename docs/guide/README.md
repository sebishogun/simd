# Guide

Task-shaped documentation. Each page starts with a problem you actually have,
shows the code, and then explains what the machine is doing and where it stops
paying.

If you are new, read [the tutorial](../tutorial.md) first. It is about how to
shape your data so there is something to vectorize at all, which matters more
than which function you call. These pages assume you have done that and want to
know what is available.

## The pages

**[Arrays and reductions](arrays.md)** — elementwise arithmetic, sums and
statistics, the `Into` convention, and the three habits that decide whether any
of this is faster than a plain loop.

**[Text and bytes](text.md)** — searching, splitting, trimming, validating
UTF-8, and parsing a CSV of integers about five times faster than `strconv`.

**[Search, sets and bit vectors](search.md)** — batched binary search, sorted
set intersection, rank and select, sliding-window extremes. The operations whose
output is not one element per input.

**[Encodings](encoding.md)** — int8 and fp8 quantization, bit packing,
run-length, zigzag, varint widths. What a column store and an inference runtime
spend their time in.

**[Signal and matrices](signal.md)** — FFT, windows, convolution, Gemv, matrix
multiply, sparse matrix-vector.

## Two things that apply to every page

**The plain name works in place. The `Into` name writes somewhere else.**

```go
simd.Add(a, b)          // a[i] += b[i]
simd.AddInto(dst, a, b) // dst[i] = a[i] + b[i], a and b untouched
```

That is the whole convention. There is no third form and no options struct.
Nothing in this library allocates unless its name says so — `Into` functions
take the destination from you precisely so they do not have to make one.

**Below roughly 16 to 64 elements you get a plain Go loop, on purpose.**

Crossing from Go into assembly costs about 1.4 nanoseconds and cannot be
inlined. Under a few dozen elements that call is more than the arithmetic it
saves, so every operation checks its length and runs the portable path instead.
You do not have to think about this — but it does mean calling a vector
operation on a four-element slice buys you nothing, and calling it once per
element in a loop is slower than not using the library at all.

The fix for that is the first section of [the tutorial](../tutorial.md), and it
is the single most common reason people measure no speedup.
