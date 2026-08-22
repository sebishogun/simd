# A general n-ary combinator taking a Go closure

**Status: measured, design fixed, not implemented.** This record exists so
the implementation step has a decision behind it rather than a preference.
Nothing in it is shipped; ROADMAP.md's n-ary section is where the open item
lives.

## The question

`AddAll` and `MulAll` ship, backed by arity-3 and arity-4 kernels with
folding beyond. The open half of that roadmap item is a general combinator:

```go
simd.CombineInto(dst, f, a, b, c)   // dst[i] = f(a[i], b[i], c[i])
```

One function that takes an arbitrary Go closure and applies it elementwise.
The roadmap already states the constraint — a closure call per element
defeats vectorization — so the only design question worth measuring is how
much it defeats it, and whether the answer leaves room for a useful API.

## The measurement

A `//go:noinline` loop calling a closure per element, against the identical
loop with the body written in. `dst[i] = a[i]*2 + b[i]`, float64, on the
Zen 5. Instructions retired, slope between `-benchtime 2000x` and `10000x`
at n=65536, so the per-call setup cancels:

| loop | instructions/element |
|---|---|
| closure per element | 17.998 |
| body written in | 7.032 |

**11.0 extra instructions per element, a 2.56x ratio.** Instructions
retired are layout- and load-independent, which matters here: the machine
was not quiet.

Wall clock on the same runs, minimum of three at `-benchtime 2000x` — a
weaker number, quoted only because it is the one a caller feels:

| n | closure | written in | ratio |
|---|---|---|---|
| 64 | 76.06 ns | 15.03 ns | 5.06x |
| 1024 | 1027 ns | 210.4 ns | 4.88x |
| 65536 | 67391 ns | 14581 ns | 4.62x |

The wall-clock ratio is worse than the instruction ratio because the
closure loop also serialises: each call is a dependency the scheduler
cannot hoist, so the extra instructions do not overlap.

The two ratios bracket the real cost. Against a *kernel* rather than
against a written-in scalar loop the gap is larger still, because the
kernel is the thing the closure makes impossible.

## What that rules out

A combinator whose contract is "give me any closure and I will make it
fast" cannot be built. At 11 extra instructions per element it is slower
than the scalar loop the caller would have written, and far slower than
chaining the binary calls that already exist. Shipping it would be an API
whose name promises the opposite of what it measures.

## The design that survives

The same shape `FilterInto` already uses in this library: **dispatch the
shapes that have kernels, and be explicit that the rest is a scalar loop.**

1. **A closed set of named operations dispatches to real kernels.** Not a
   closure — a value from an enumerated type, so the compiler and the
   reader both know the set is finite:

   ```go
   simd.CombineInto(dst, simd.OpAdd, a, b, c)
   ```

   `OpAdd` and `OpMul` reach `Add3`/`Add4`/`Mul3`/`Mul4` and the existing
   fold. This is `AddAll`/`MulAll` with the operation as a parameter, and
   it is worth having only if a caller genuinely needs to choose at run
   time; otherwise `AddAll` is clearer at the call site.

2. **The closure form is a separate function with the cost in its name and
   its doc.** Something like `CombineFuncInto`, documented as a scalar
   loop, with the measured 11 instructions per element in the comment so
   nobody has to rediscover it. It exists for the case where the caller has
   an operation this library does not have and wants one pass over the data
   rather than materialising intermediates — which is a real want, and is
   about memory traffic, not about vector width.

3. **No heuristic that silently switches between them.** A combinator that
   inspected the closure and sometimes used a kernel would be a guess about
   what the closure does. The two functions do different things and say so.

The measurement gate for step 2: `CombineFuncInto` ships only if it beats
the binary chain it replaces at a size where memory traffic dominates — the
same gate `AddAll` passed. If it does not beat it anywhere, it does not
ship, and the number goes in `docs/wrong.md`.

## What is not decided here

The exact names and signatures. Those follow the API conventions in
`docs/api.md` at implementation time, and the enumerated-operation form may
turn out not to be worth its own function at all — in which case only the
closure form ships, or neither does.
