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

### sort / argsort

The partition step is compress, which now exists.

A vectorized sort is not a vectorized version of a scalar sort; it is a
different algorithm — a sorting network on register-sized blocks, then a
bitonic merge. Worth doing because the gap over `slices.Sort` on primitives is
large, and because `argsort` has no good answer in Go at all today.

### ~~A blocked GEMM microkernel, and `Gemv`~~ — done in v0.1.0

`MatMul` is register-blocked: a tile of the output is accumulated in registers
across the whole shared dimension and stored once, so the traffic falls from
2·m·k·n element accesses to about m·n·k·(1/MR + 1/NR) + m·n. The tile
dimensions are chosen per target by `#if`, because it is the register file
rather than the instruction set that constrains them.

Still not done, and the next thing to do here: **packed panels and
cache-blocked outer loops**. Register blocking fixes the innermost level only.
Above the L2 working set, B is re-read from memory for every row block, and the
strided read of A costs a TLB miss per row on a large matrix. Packing both into
contiguous scratch is the standard fix and is worth another large factor — but
it needs scratch memory, and this library promises zero allocations, so it
needs a design decision first rather than just an implementation.

`Gemv` ships and is bit-identical to `Dot` per row.

---

## Backends

### ppc64le: rewrite TOC-relative constant loads to PC-relative

182 kernels are not generated for ppc64le because clang reaches its constants
through `r2`, the TOC pointer, which Go does not maintain for these objects.

Power9 has no PC-relative data addressing, so the fix is a rewriter: recognise
the `addis`/`ld` TOC pair, resolve what it was pointing at, and re-materialise
the address the way the other backends do. A further 64 kernels use `r30`,
which Go's linker owns, and that is a separate investigation.

This is the largest single coverage gap in the library — ppc64le has 281
kernels where amd64 has 1652.

### s390x

614 kernels, and the missing ones are missing because clang uses `r13`, which
is where Go keeps the current goroutine. There is no `-ffixed-r13` for SystemZ;
the global register variable is accepted and silently ignored. No fix is known,
which is why this has no entry above — it is recorded here so that its absence
is not mistaken for an oversight.

---

## Tiers

### A `GOEXPERIMENT=simd` tier, and the language feature after it

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
