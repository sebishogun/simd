# Roadmap

What is not built yet, in the order it is worth building, with the reason each
one is where it is.

Nothing here is a promise of a date. It is a statement of what the gaps are, so
that a reader can tell the difference between a thing that was decided against
and a thing that has not been reached.

---

## Kernels

### ~~compress / expand / filter~~ — done in v0.1.0

`CompressInto`, `ExpandInto` and `FilterInto` ship, and `IndexAll` is built on
the first of them. Accelerated on AVX-512 (`vpcompressd`) and SVE2 (`compact`);
portable on the other seven, permanently, because they have no compress
instruction and the operation cannot be vectorized without one.

`ExpandInto` is portable everywhere and will stay that way. Compression's
serial half is the store, which a compress instruction fixes; expansion's is
the load, which it does not — clang emits the same scalar loop for it on both
of the targets that have the instruction.

### ~~n-ary and variadic operations~~ — done after v0.1.0

`AddAll` and `MulAll` take any number of slices and make a single pass. Arity-3
and arity-4 kernels on every architecture; longer calls fold in groups of four,
so a sixteen-way sum is four passes rather than fifteen. Arity stops at four
because a five-source kernel needs seven pointer arguments and System V passes
six integers in registers.

The element type is enforced by the compiler — the parameter is `...[]T`, so
mixing types will not build. Lengths remain a run-time property, because a Go
slice carries its length in its header rather than in its type; the work is
bounded by the shortest slice, as everywhere else here.

Still open: the **other half of what was asked for**, a general n-ary
combinator taking a Go closure. That is easy to write and hard to make fast —
a closure call per element defeats vectorization entirely — so it needs a
design where the common shapes dispatch to real kernels and the rest is honest
about being a scalar loop. `FilterInto` is the same problem already solved once
that way.

### ~~sort / argsort~~ — done after v0.1.0

`Sort`, `SortInto`, `Argsort`, `PartitionInto` and `SortedIndex` ship. A
quicksort in Go around a compress-based partition kernel, accelerated on
AVX-512 and SVE2 and portable elsewhere. Against `slices.Sort` on float64 it is
19-27% faster at every size from 16K elements up, and even at 1024.

The one case that loses is few distinct values at 16384, by 34%: the pivot
equals much of the range, the split is lopsided, and the skew guard hands the
range to pdqsort after paying for one partition, which at that size does not
earn its cost back. **The fix is a three-way partition** — splitting into
below / equal / above so duplicates are consumed at each level instead of being
pushed to one side. That needs a second kernel and is the obvious next step
here.

---

## Backends

### ppc64le: repoint the TOC prologue at an appended pool

**The design is settled and the mechanism is verified on hardware. What remains
is the generator change.**

157 kernels are not generated for ppc64le because clang reaches its constants
through `r2`, the TOC pointer, which Go does not maintain for these objects.
This is the largest single coverage gap in the library: ppc64le has 281 kernels
where amd64 has 1664.

The shape is uniform and that is what makes it tractable. Every one of those
kernels opens with clang's ELFv2 global-entry prologue —

    addis r2, r12, .TOC.@ha     R_PPC64_REL16_HA
    addi  r2, r2,  .TOC.@l      R_PPC64_REL16_LO

— and only 10 of 541 functions surveyed write `r2` after it.

Power9 has no PC-relative data addressing, which for a long time looked like
the obstacle. It is not, because none is needed. **Go's own assembler
materialises a RODATA address in two instructions with no TOC at all:**

    MOVD $pool<>(SB), R12   ->   lis  r12, 15
                                 addi r12, r12, -560

Go builds non-PIE, so the address is a link-time constant. A probe doing this
was built and *run* under emulated ppc64le and read back the right value. So
there is no need to clobber the link register with `bcl`/`mflr`, no need for a
protected-zone slot to save it in, and no dependency on whatever `r2` held on
entry.

The rewrite is therefore:

1. Emit the pool as a Go `DATA`/`GLOBL` symbol, as the amd64 path already does.
2. Put `MOVD $pool<>(SB), R2` in the generator's prologue. Its length is free —
   the prologue precedes the body, and the body's branches are all relative to
   the body.
3. Overwrite the two global-entry instructions with NOPs. Length-preserving,
   which `constpool.go` explains is not optional.
4. Recompute each `R_PPC64_TOC16_HA/LO` immediate as an offset from the pool
   base instead of from `.TOC.`, minding the HA/LO sign adjustment — the same
   arithmetic `resolvePool` already does for four other architectures.

A further 64 kernels use `r30`, which Go's linker owns; that is separate.

### s390x

614 kernels, and the missing ones are missing because clang uses `r13`, which
is where Go keeps the current goroutine. There is no `-ffixed-r13` for SystemZ;
the global register variable is accepted and silently ignored. No fix is known,
which is why this has no entry above — it is recorded here so that its absence
is not mistaken for an oversight.

---

## Tiers

### A `GOEXPERIMENT=simd` tier — measured, and deliberately not shipped

**This has been built and benchmarked. The measurement says do not wire it up
yet, and that is the decision.**

Go 1.26's intrinsics are real and usable — `simd/archsimd` exists, the
experiment is accepted, `LoadFloat32x8Slice` and friends compile to the
instructions you would expect. So the question was never whether they work, it
was whether they beat what is already here. On a Zen 5, float32 `AddInto`:

| n | Go scalar loop | Go intrinsics | this library (assembly) |
|---|---|---|---|
| 4 | 3.87 ns | **3.81 ns** | 4.46 ns |
| 8 | 7.15 ns | **2.82 ns** | 5.27 ns |
| 16 | 14.23 ns | **3.74 ns** | 5.61 ns |
| 32 | 26.91 ns | 6.13 ns | **5.83 ns** |
| 64 | 53.81 ns | 11.07 ns | **5.83 ns** |
| 128 | 77.22 ns | 20.09 ns | **6.91 ns** |
| 256 | 158.75 ns | 38.70 ns | **8.81 ns** |

Two things fall out, and the second was not expected.

**The opportunity is real but tiny.** Intrinsics win only at n ≤ 16, by about
two nanoseconds, in a band where the absolute cost is already about five.

**Above n = 32 the assembly is not merely ahead, it is several times ahead** —
4.4× at n = 256. The common assumption that intrinsics are equivalent to
hand-written assembly is wrong here, and the reason is that they are equivalent
to *what you write with them*: an idiomatic 8-lane Go loop against a
clang-generated kernel that uses 512-bit vectors and unrolls. Matching it means
writing that unrolled 512-bit loop by hand in Go, per type, per width — which
is the work this library's generator exists to avoid.

So shipping it now would mean asking consumers to set a `GOEXPERIMENT` — which
a library cannot do — for two nanoseconds on amd64 only. The decision is no.

**Revisit when it needs no flag** (targeted at Go 1.27). At that point it costs
a consumer nothing, and the same measurement decides per operation: below the
threshold where the dispatcher currently runs a scalar loop, an inlined
intrinsic is worth having; above it, the assembly stays. The bit-identity
contract binds it either way.

The reasoning behind why it can only ever win at small n follows.

### Why the small-n band is the only opening

Go 1.26 shipped SIMD intrinsics behind `GOEXPERIMENT=simd`, and they are
targeted at the language proper in 1.27.

They cannot replace this library — they do not reach SVE2 or RVV, they cover
amd64 and arm64 only, and requiring a consumer to set a `GOEXPERIMENT` is not
something a library can do. But there is one band where they beat assembly and
always will: **below the call boundary**.

An assembly kernel cannot be inlined. It pays ~1.4 ns at the floor and 50–65
cycles once `VZEROUPPER` and register save/restore are counted, and that cost
is fixed rather than proportional to `n`. At n = 4096 it is a rounding error.
At n = 32 it is most of the runtime — which is why the dispatcher currently
gives up below its threshold and runs a scalar Go loop instead. An inlined
intrinsic pays none of it.

So the plan is additive and measured:

- Build the tier now, behind the build tag, against the same `kernel.Ops`
  contract every other backend implements, so dispatch selects it the same way.
- Benchmark it against **both** the assembly tier and the scalar fallback at
  n = 4, 8, 16, 32, 64, 128, 256. Only the sizes where it wins get wired up.
  The same rule as every other tier in this library: a measurement decides.
- The bit-identity contract still binds. Go's intrinsics have their own
  rounding and NaN behaviour, and they get differentially tested against the
  portable reference like everything else.

When the intrinsics are in the language and need no flag, re-run that
benchmark and move every operation where the intrinsic wins — keeping assembly
everywhere it does not, which on current evidence is everywhere above n ≈ 64
and every architecture Go's intrinsics do not cover.

---

## Verification

### Real hardware

Everything outside amd64 is verified under emulation. That proves semantics and
proves nothing about timing, so every per-architecture threshold outside amd64
is currently a guess carried over from a machine that does not resemble it.

The thresholds are the part that is actually wrong without this — a kernel that
is correct under qemu is correct on the metal, but a crossover measured on
nothing is a number with no evidence behind it.
