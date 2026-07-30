# Writing code a vector unit can help with

This library has no vector type. You call ordinary functions on ordinary Go
slices, and the instruction set is chosen for you. That is deliberate — a
public `Vec4` costs a function call per operation and would lose to plain Go —
but it moves the interesting decisions somewhere else.

**The library cannot vectorize your program. It can only vectorize the loops
you hand it.** How you shape your data decides whether there is anything to
hand over.

Everything below is measured on this machine (Zen 5, AVX-512) and every
snippet compiles. The complete programs are in [`examples/`](examples/).

---

## 1. Hand over batches, not elements

There is a fixed cost to calling into assembly — about 1.4 ns at the floor,
more once the register save and `VZEROUPPER` are counted — and it cannot be
inlined away. That cost is per *call*, not per element, so it disappears into
a large slice and dominates a small one.

```go
// Wrong. One call per element: the boundary cost is the whole runtime.
for i := range dst {
    simd.AddInto(dst[i:i+1], a[i:i+1], b[i:i+1])
}

// Right. One call, n elements.
simd.AddInto(dst, a, b)
```

The first form is not a little slower, it is about two orders of magnitude
slower. You do not need to check for the small case yourself — below a
per-kernel threshold the dispatcher runs an inlined scalar loop instead — but
you do need to *give* it the batch.

**Rule of thumb:** below ~16 elements write plain Go, 16–64 is a wash, and from
a few hundred up the assembly wins by multiples.

---

## 2. Struct of arrays, not array of structs

This is the decision that matters most, and it is made long before you call
this library.

```go
// AoS — the natural Go shape, and the one a vector unit cannot use.
type Particle struct{ X, Y, Z, Mass float64 }
ps := make([]Particle, 1_000_000)

// SoA — the same data, laid out so each field is a contiguous run.
type Particles struct{ X, Y, Z, Mass []float64 }
```

With `[]Particle`, the X values are 32 bytes apart. A vector register holding
eight float64 would need eight separate loads and a gather; there is no
contiguous run of X to load. With `Particles`, every field is exactly the
contiguous slice the API takes:

```go
// Scale every X by 2, in one pass, no allocation.
simd.Scale(p.X, 2)

// Kinetic energy per particle: 0.5*m*v² over three components.
simd.MulInto(e, p.X, p.X)          // e  = x*x
simd.MulAll(tmp, p.Y, p.Y)         // tmp = y*y
simd.Add(e, tmp)                   // e += y*y
simd.MulAll(tmp, p.Z, p.Z)
simd.Add(e, tmp)
simd.Mul(e, p.Mass)
simd.Scale(e, 0.5)
```

If you already have `[]Particle` and cannot change it, transposing once is
often still worth it — but measure, because the transpose is a full pass and
you need to amortise it over several operations.

**Not everything needs SoA.** If you touch every field of one struct at a
time, AoS has better locality. SoA pays when you sweep one field across many
elements, which is exactly when this library is useful.

---

## 3. Reuse buffers: the `Into` convention

Every operation comes in two forms. The plain name works in place on its first
argument; the `Into` suffix takes a destination first.

```go
simd.Add(a, b)          // a[i] += b[i]
simd.AddInto(dst, a, b) // dst[i] = a[i] + b[i]
```

**This package never allocates.** No function returns a freshly made slice. So
a loop over batches allocates once, outside it:

```go
dst := make([]float64, batchSize)   // once
for _, batch := range batches {
    simd.AddInto(dst, batch.a, batch.b)
    use(dst)
}
```

Lengths need not match — every operation processes the minimum length of its
slice arguments — so slicing is how you bound work:

```go
simd.AddInto(dst[:n], a, b)
```

---

## 4. Fuse, don't chain

A slice library's structural weakness is one memory pass per operation. At
sizes past L2 you are bandwidth-bound, and three chained calls read and write
memory three times over.

```go
// Three passes over memory.
simd.MulInto(dst, a, b)
simd.Add(dst, c)
simd.Add(dst, d)

// One pass.
simd.AddAll(dst, ab, c, d)   // where ab is a*b
```

The fused forms are the reason the catalogue has `AddAll`, `MulAll`,
`AddScaled` and the arity-3 and arity-4 kernels. Measured on `dst = a*b + c`
at n = 1024: the fused kernel is **13.9×** a handwritten Go loop, and
**1.4×** the same work done as two separate calls.

There is deliberately **no** general `ZipInto(dst, f, srcs...)` taking a
closure. It was built and measured: a closure call per element cannot be
vectorized, and it came out 1.24–3.0× *slower* than the plain Go loop you
would have written without it. Where your expression is not in the catalogue,
write the loop.

---

## 5. Give the text functions scratch

The text side follows the same discipline, but some operations need working
space. Pass it in and they allocate nothing:

```go
// Case-insensitive search over one page for many words.
scratch := make([]byte, len(page)+64)   // once
for _, w := range words {
    if simd.IndexFoldASCII(page, w, scratch) >= 0 {
        ...
    }
}
```

Give it `len(haystack)+len(needle)` and the call is allocation-free — the
needle's folded copy is carved from the same buffer.

The append-style functions follow Go's own convention, so reusing the buffer
is the same idiom you already know:

```go
buf = simd.AppendUTF16(buf[:0], line)
out = simd.AppendEscapeJSON(out[:0], s)
```

---

## 6. Two phases: find structure, then compute

Real work usually splits into a byte scan and some arithmetic, and they want
different things from you.

```go
// Phase 1 — structure. One pass, offsets written into a buffer you own.
n := simd.IndexAll(idx, line, ',')
idx[n] = int32(len(line))

// Phase 2 — conversion and arithmetic over the fields.
count, ok := simd.ParseFloats(vals, line, idx[:n+1])
mean := simd.Mean(vals[:count])
```

Splitting them is not incidental. The scan is already fast — `IndexAll` runs a
CSV corpus at 4.06 GB/s — and separating the two lets a caller reuse one scan
for several conversions. [`examples/csvscan`](examples/csvscan/) is this
shape end to end.

---

## 7. Know what will not vectorize

Some operations are serial through their own output and no amount of data
shaping changes that. The library says so at each one rather than pretending:

- **`CumSum` and `CumProd` on floats.** Every partial result is written, so the
  grouping is observable, and a vector form would not match the loop you would
  have written. Opt in with `FastCumSum`/`FastCumProd` if you want the vector
  grouping — 3.65× on float32 products — and read what they trade first.
- **`EMA`.** `y[i] = a*x[i] + (1-a)*y[i-1]` is a dependency chain.
- **`ExpandInto`.** Compression's serial half is the store, which an
  instruction fixes; expansion's is the load, which none does.

Integer `CumProd` *is* accelerated, because two's-complement multiplication is
associative and the regrouping is not observable. The distinction is the point:
what blocks vectorization is the observability of the reassociation, not the
operation.

---

## 8. Results do not change with the machine

Every operation returns bit-identical results on every instruction set,
including for NaN, ±Inf, ±0 and denormals. Reductions use a fixed
sixteen-accumulator order that a 128-bit and a 512-bit machine both reproduce
exactly.

You can rely on this: a checksum computed on your laptop matches the one your
server computes. Operations that trade it away are named `Fast*` and document
what they trade.

```go
// Same bits on every tier, and on -tags purego.
total := simd.Sum(xs)
```

`GOSIMD=sse2` forces a tier, which is how you check that claim yourself.

---

## 9. A worked example

Normalising a batch of feature vectors — the shape a lot of numeric code has.

```go
package main

import (
    "fmt"

    "github.com/sebishogun/simd"
)

// Features is struct-of-arrays: one contiguous slice per dimension, which is
// what lets every operation below be a single vector pass.
type Features struct {
    Dims [][]float64 // Dims[d][i] is dimension d of sample i
}

// Standardise rewrites each dimension to zero mean and unit variance, in
// place, allocating nothing.
func Standardise(f Features) {
    for _, col := range f.Dims {
        mean := simd.Mean(col)          // one pass
        simd.SubScalar(col, mean)       // one pass, in place
        sd := simd.StdDev(col)          // one pass
        if sd != 0 {
            simd.DivScalar(col, sd)     // one pass
        }
    }
}

func main() {
    f := Features{Dims: make([][]float64, 4)}
    for d := range f.Dims {
        col := make([]float64, 1<<20)
        for i := range col {
            col[i] = float64((i*7+d)%1000) - 500
        }
        f.Dims[d] = col
    }
    Standardise(f)
    fmt.Printf("dim 0: mean %.3g, stddev %.3f\n",
        simd.Mean(f.Dims[0]), simd.StdDev(f.Dims[0]))
}
```

Four passes per column and no allocation. The same code in AoS form would need
a gather per element and would not vectorize at all.

---

## 10. Check what you got

```go
fmt.Println(simd.Describe())
// amd64 tier=avx512 available=[scalar sse2 avx2 avx512]
```

`simdinfo` reports the same from the command line, and `GOSIMD=scalar` forces
the portable path so you can measure what the acceleration is actually worth
on your data:

```
go test -bench . | tee fast.txt
GOSIMD=scalar go test -bench . | tee slow.txt
benchstat slow.txt fast.txt
```

That last comparison is the only one that answers "is this worth it for me",
because it uses your data and your machine rather than this file's.
