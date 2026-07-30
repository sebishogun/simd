# Adding a kernel

This is the guide for adding an operation that reaches all six architectures.
It exists because the answer used to be "read eight files and infer it", and
because most of the traps below cost someone a day each.

Everything here is a real example: `Zigzag` was added exactly this way, and the
file paths and snippets are the ones in the tree.

---

## What actually happens

You write **C**. The generator compiles it once per target with clang, checks
the result against a list of things Go's runtime will not tolerate, and rewrites
the machine code into Plan 9 assembly that is committed to the repository.

```
csrc/convert.c
      │  clang --target=… -O3 (nine times, one per tier)
      ▼
  an object file
      │  tools/simdgen: disassemble, verify, rewrite
      ▼
internal/amd64/convert_avx2_amd64.s      ← committed
internal/amd64/convert_register_avx2_amd64.go
```

Two consequences that shape everything else:

- **There is no libc.** A call to anything the object does not define cannot be
  resolved. `sqrtf` is a call; `__builtin_elementwise_sqrt` is an instruction.
- **A kernel that fails verification is not an error.** It is dropped, the
  portable Go implementation stands in, and `make check-emission` says why.
  Shipping a kernel on five of six targets is normal.

---

## The eight files

| | file | what goes in it |
|---|---|---|
| 1 | `csrc/*.c` | the kernel |
| 2 | `tools/simdgen/kernels/kernels.go` | the manifest entry |
| 3 | `internal/kernel/kernel.go` | the dispatch field **and its numerical contract** |
| 4 | `internal/ref/*.go` | the portable implementation, its ops-table entry, its exported entry point |
| 5 | the public file (`bytes.go`, `convert_fp.go`, `numeric.go`, …) | the exported API |
| 6 | `*_test.go` | tests |
| 7 | `internal/conformance/*_test.go` | the differential against `ref` |
| 8 | `bench_*_test.go` | a benchmark against what it replaces |

Skipping 4 gives a `nil` dispatch entry and a panic under `-tags purego`, which
is how `TestDispatchTableComplete` came to exist. Skipping 7 is how a wrong
constant shipped on ppc64le; see [`wrong.md`](wrong.md) entry 53.

---

## 1. The C

Pick the file by **group**, not by taste: the generator looks for the function
in the source that the manifest's `Group` maps to. A `Group: "Bytes"` kernel
must be in `csrc/bytes.c`. Putting it elsewhere gives

```
- simd_hamming_u8: no such function in the C source (manifest says hammingU8)
```

```c
#define ZIGZAG(T, UT, SUF)                                                 \
  void simd_zigzag_encode_##SUF(UT *__restrict d, const T *__restrict a,   \
                                isize n) {                                 \
    for (isize i = 0; i < n; i++) {                                        \
      UT u = (UT)a[i];                                                     \
      UT m = (UT)(a[i] < 0 ? ~(UT)0 : (UT)0);                              \
      d[i] = (UT)((UT)(u << 1) ^ m);                                       \
    }                                                                      \
  }

ZIGZAG(int, unsigned int, i32)
```

The rules, all of which the vectorizer or the verifier will otherwise enforce
the hard way:

- `__restrict` on every pointer. Without it clang must assume the output
  aliases the input and will not vectorize.
- A **signed** loop counter (`isize`). Unsigned induction variables have
  defined wraparound, which blocks the trip-count reasoning the vectorizer
  needs.
- No calls. No `printf`, no libm, no `memcpy` — the last is why the build
  passes `-fno-builtin-memcpy`.
- No early exit that depends on the data, unless you are writing a search
  kernel and have read how `OR_ESCAPE` in `csrc/fold.h` does it.

## 2. The manifest

```go
conv("simd_zigzag_encode_i32", "zigzagEncodeI32", "ZigzagEncodeI32",
	"ZigzagEncodeI32", spec.SliceU32, spec.SliceI32),
```

expanding to:

```go
spec.Kernel{
	CName: "simd_zigzag_encode_i32",  // the C symbol
	GoName: "zigzagEncodeI32",        // the generated Go stub
	Group: "Convert",                 // which kernel.Set field, and which .c file
	Field: "ZigzagEncodeI32",         // the field inside that group
	RefFunc: "ZigzagEncodeI32",       // the exported ref.* the guard falls back to
	Params: []spec.Param{sl("dst", spec.SliceU32), sl("a", spec.SliceI32)},
	CArgs: []spec.CArg{base("dst"), base("a"), lenOf("dst")},
	Threshold: thElementwise,
}
```

`Params` is the **Go** signature. `CArgs` is the **C** one, and they differ:
`base` passes a slice's data pointer, `lenOf` its length, `val` a scalar, and
`out` a pointer to a single result for a reduction. The generated guard clamps
to the minimum length of every slice parameter, so `lenOf` may name any of
them.

`Threshold` is the element count below which the guard calls `ref` instead.
It is not a style choice — a Go-to-assembly call costs ~1.4 ns and cannot be
inlined, so below the threshold the call is the whole runtime. Use the existing
`thElementwise`, `thBytes`, `thScan` or `thReduction` unless you have measured
something else.

**A kernel whose output is not the same length as its input must declare two
lengths.** The guard's default is `n := min(len(dst), len(a), ...)` over every
slice, which is right for elementwise work and wrong for anything else — and it
is wrong *silently*, by processing the wrong number of elements rather than by
failing.

Two kernels hit this in one week, from opposite directions. `sum_lanes` writes
sixteen accumulators from an arbitrarily long input, so the input was clamped
to sixteen and a 4096-element chunk summed its first sixteen. `bitpack` writes
*fewer* words than it reads, which is the entire point of packing, so the input
was clamped to the output and everything past `len(dst)` was dropped. Adding a
second `lenOf` to `CArgs` turns the clamping off, and the kernel checks the
sizes itself:

```go
CArgs: []spec.CArg{base("dst"), base("a"), val("bits"),
	lenOf("a"), lenOf("dst")},
```

`Diff` — whose output is one element shorter than its input — has done this
since it was written, and the comment on `countLens` in the emitter explains
why. If your output length is a function of anything other than the input
length, you are in this case.

**Watch the argument count.** SysV amd64 passes six integer arguments in
registers, and the generator declines anything longer:

```
simd_rgb_to_yuv_u8 (needs more argument registers than amd64 has)
```

Three in, three out and a length is seven. Splitting that kernel in two is what
took it from zero amd64 tiers to three.

## 3. The dispatch field

```go
// Zigzag maps a signed integer onto an unsigned one so that a small
// magnitude of either sign becomes a small unsigned value …
//
// Exact and total in both directions under rule 1: these are shifts and
// exclusive ors, so every tier gives the same bits, and every value round
// trips including the most negative one.
ZigzagEncodeI32 func(dst []uint32, a []int32)
```

**This file is not generated, deliberately.** The doc comment is where the
numerical contract lives — which of the six rules the operation falls under,
what it promises for NaN and ±0, whether it is exact or carries a bound. That
is the most valuable prose in the repository and a generator would flatten it.

## 4. The portable implementation

Three edits in `internal/ref`: the function, its entry in the ops table, and an
exported wrapper the generated guard calls by name.

```go
func zigzagEncode[S ~int8 | ~int16 | ~int32 | ~int64, U ~uint8 | …](dst []U, a []S, shift int) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = U(a[i]<<1) ^ U(a[i]>>shift)
	}
}

func ZigzagEncodeI32(dst []uint32, a []int32) { zigzagEncode(dst, a, 31) }
```

This is the **specification**, not a fallback. Every kernel is checked against
it, so where the two disagree it is the kernel that is wrong — unless the bug
is here, which has happened.

## 5–8. API, tests, conformance, benchmark

The public function is a one-liner through `active`:

```go
func ZigzagEncodeInt32Into(dst []uint32, a []int32) { active.Convert.ZigzagEncodeI32(dst, a) }
```

The benchmark should compare against **what a caller would otherwise write** —
the plain Go loop, or the sequence of existing calls this replaces. Benchmarking
a fused kernel against a scalar loop rather than against the chain it replaces
is how a 1.76× regression shipped for an afternoon; see entry 55.

---

## Verifying

```
make codegen        # regenerate; then check git diff is only what you expect
make check-emission # per-target counts and the reason for every skip
make verify         # fmt, vet, test, purego, every amd64 tier
make test-cross     # arm64, s390x, ppc64le under docker + qemu
make test-riscv64 make test-loong64 make test-gates
```

Two things about running these that are easy to get wrong and both cost real
time here:

- **Do not edit the tree while `make test-cross` runs.** It mounts the working
  directory into the container. A result from a half-written tree is worse than
  no result.
- **Do not pipe it through `tail`.** `make test-cross | tail -20` reports
  `tail`'s exit status, which is 0 whatever make did, and truncates the log to
  the last architecture. Redirect to a file and check `$?`.

`make test-gates` is the one that is not obvious. Every other emulated lane
runs a CPU with all features enabled, so a kernel gated on a feature it does
not actually use is selected and works. The gates lane runs the same binaries
on a CPU *without* the vector extension, which is where that shows up — as
SIGILL, at the first call.

---

## The traps, already paid for

Each of these is a day someone already spent. The long version of every one is
in [`wrong.md`](wrong.md).

**A cast to a vector type is not a broadcast.** `(u8x16)x` puts `x` in lane 0
and zeroes the rest, and it is target-dependent whether you notice. Use the
macro:

```c
#define SPLAT(VT, X) ((X) - ((VT){0}))
```

Subtraction, not addition. `((VT){0}) + (-0.0)` is `+0.0`, because IEEE says
so, and a scan carrying a signed zero loses the sign. Entry 52.

**Do not write a multiply-add in `internal/ref`.** Go may fuse it into an FMA
and does on riscv64, loong64, arm64 and ppc64 — but not amd64. The kernels
compile `-ffp-contract=off` and never fuse, so the portable path ends up
disagreeing with its own kernels on four of six targets. Force the rounding:

```go
t := (a[i] + shift) / denom
dst[i] = T(t*gamma[i]) + beta[i]   // the T() is load-bearing
```

`internal/ref/fusion_test.go` checks this. Add your operation to it. Entry 56.

**Do not multiply by a reciprocal to save a divide.** `x * (1/d)` is not `x / d`,
and `DivScalar` promises the latter. Fusing two passes is allowed to change the
number of passes and nothing else.

**Reductions are the dangerous ones.** Float accumulation must use exactly
`kernel.SumLanes` accumulators combined by `kernel.CombineTree`, on every tier,
whatever the hardware width. Integer reductions are free — addition is
associative, so the lane grouping is not observable.

**Use the existing fold macros.** `COUNT_FOLD` and `COUNT_BYTES` in
`csrc/fold.h` exist because a naive `s += f(a[i])` with a 64-bit accumulator is
a serial dependence with a widening per element. That is not a small effect: it
made a fused kernel 1.76× *slower* than the two-pass chain it replaced. Look at
what the neighbouring kernel does before writing the loop. Entry 55.

**Every `DATA` byte must be covered.** If you touch the constant-pool emitter,
note that `GLOBL` declares a size and the assembler zero-fills anything no
`DATA` directive wrote — silently. 46 pools shipped four bytes short that way.
Entry 53.

**Fixed point beats float for an 8-bit answer**, but pick enough of it. Q8
weights for BT.601 luma sum to 256 and preserve grey exactly, which looks like
proof they are right; they still put full green one level off. libjpeg uses Q16.

---

## When it does not vectorize

`make check-emission` prints a reason for every declined kernel. The common
ones and what they mean:

| reason | what to do |
|---|---|
| `LLVM did not vectorize it for <tier>` | check `-Rpass-missed=loop-vectorize`; often a missing instruction on that target |
| `uses r13 / r30 / $fp / r2` | a register the Go runtime owns. Usually unfixable from the C |
| `writes a nonzero value to r0` (ppc64le) | Go's ppc64le ABI defines r0 as constant zero |
| `needs more argument registers` | split the kernel |
| `reads a constant pool with <legacy SSE insn>` | 16-byte alignment the Go linker does not guarantee |
| `spills N bytes, over the 512-byte budget` | the NOSPLIT frame limit; simplify or split |

A declined kernel is a recorded tradeoff, not a failure. What is *not*
acceptable is a kernel that is accepted and wrong, which is what everything in
the previous section is about.
