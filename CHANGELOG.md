# Changelog

## v1.15.0

**The columnar family** -- Arrow-style validity bitmaps driving typed value
buffers, one bit per row, LSB-first, with no arrow-go dependency: raw
slices and a []byte bitmap, so arrow users hand over an Array's buffers
and everyone else gets the same toolkit on plain Go data.
**`CompressBitsInto`** is the columnar filter: compressK's contract with
the mask arriving as packed bits, which is the shape the compress
instructions' predicate registers want anyway. **`SumValid`** is the
null-aware sum, a select at each lane inside Sum's exact sixteen-lane
accumulation tree -- bit-identical to `Sum` over a masked copy (which is
literally the reference), and it never reads the value under a clear bit,
so Arrow's undefined null slots may hold NaN. **`CountValid`** is the
non-null count through the PopCount kernel; take is [GatherInto],
unchanged. Four full-width element types (float32/64, int32/64), the same
compress-instruction boundary the Compress family documents; 43 new
kernels across the targets. First consumers: the columnar/arrow compute
layer this release opens, and any Go analytics on nullable columns.

## v1.14.0

**Per-operation dispatch, and binaries a quarter the size.** The registry --
one `Set` of function values per tier, assembled from init -- made every
kernel reachable in every binary that imported the package: a program
calling one function carried all 2,553 amd64 kernels, 6.35 MB of assembly
plus the runtime tables to describe it. Dispatch is now one static table
per operation, a composite literal of exported guards indexed by the tier
selected at startup, and the linker drops every operation a program never
calls -- assembly included. A JSON-validation binary drops from 15.9 MB to
4.4 MB (13.4 to 2.9 stripped) and links exactly the nine operations it
uses. The numeric groups arrive the same way via per-tier partial `Ops`
structs overlaid lazily per element type, so their granularity is the
element type the caller instantiates.

Nothing else moves: the API is unchanged, every one of the 6,807 committed
kernels is byte-identical, thresholds and `GOSIMD` and `Tier()` behave as
before, and the whole-set machinery survives for the conformance suite
behind test-only constructors. Measured neutral end to end: the 15-row
downstream gate held, a 33-row interleaved sweep read inside the noise
floor on 32 rows, and the flagged wall-clock readings all showed equal or
fewer instructions retired at fixed iteration counts.

## v1.13.0

**`JSONValid`** -- the whole of a JSON `Valid` in one pass, no mask buffers:
per 64-byte block the five predicates become mask words that never leave
registers (`__builtin_convertvector` to a lane mask, bit-cast to a word --
the movemask idiom, minus the store), escapes resolve, quote parity runs
(carry-less multiply where the tier has it), control bytes inside strings
reject, the block's escape targets validate at a cost proportional to their
count, and the significant bits feed the `JSONValidTokens` grammar machine,
whose number fast-forward now skips whole blocks -- provably carry-safe,
since a block inside a number holds no quote, backslash, control byte or
whitespace. The staged pipeline writes and re-reads eight mask regions;
this writes none. On amd64, arm64 and riscv64 tiers; s390x keeps the
reference path (little-endian movemask and literal reads), and the one
partial block is classified a byte at a time because a padded vector load
would need a variable-length copy, which a freestanding kernel cannot make.
Downstream in simdjson, `Valid` on gsoc-2018 runs 32% faster and every
string- and structure-heavy shape gains 14-47%; a density router keeps the
number-wall corpora on their descent walk.

The qemu lanes now run `internal/tests/text` (the new kernel's differential
home) and carry the guest's exit status instead of laundering it through a
pipe.

## v1.12.0

**`JSONValidTokens`** -- the grammar half of a JSON `Valid`, fused into one
scalar kernel: the state machine over every significant byte, reading the
`JSONStage1` masks, with the number scanner and literal checks inlined and
the container bit-stack spilling into a caller-provided buffer. Scalar on
purpose: the new **`AllowScalar`** manifest field exempts kernels whose
value is fused control flow -- a grammar walk, a state machine -- from the
vector-instruction check, making them first-class citizens of the
generator. Downstream in simdjson, twitter `Valid` runs 28% faster and
holds 1.75x over sonic on that shape. s390x keeps the reference path (the
masks are read as little-endian words).

## v1.11.0

**`JSONStage1`** -- the JSON indexer's first word pass, fused into one
kernel: escape resolution, quote parity, the in-string mask, the structural
and whitespace counts, the control-in-string check, and the
escape-verification worklist, consumed straight from the five `JSONMasks`
regions. Quote parity is carry-less multiply (`PCLMULQDQ`) on the amd64
vector tiers -- `-mpclmul` joined the v3/v4 tier flags, which no
AVX2-capable CPU lacks -- and a two-word-wide vector shift chain elsewhere.
s390x keeps the reference path: the mask words are read little-endian.
Downstream in simdjson, gsoc-2018 `Valid` runs 1.42× faster end to end.

The cross lane is hermetic now: `--network=none`, `GOPROXY=off`,
`GOTOOLCHAIN=local`, the host module cache mounted read-only. Two release
rituals' worth of silent 0%-CPU hangs under emulated net/http become
one-second failures that name the missing thing.

## v1.10.2

**The portable `jsonMasks` goes word-at-a-time.** Eight bytes per load, all
five masks from SWAR containment tests collapsed with the movemask multiply.
On architectures without generated kernels this is the whole of stage one:
purego twitter `Scan` runs 1.93× faster and `Parse` 1.73×, and a 50-byte
document's `Parse` through simdjson drops from 170 ns to 129 (204 before this
release series). Bit-identical, held by the conformance differential against
every assembly tier and a three-million-word brute-force of the two SWAR
helpers — one of which exists because Mycroft's zero test is a detector, not a
per-byte mask.

## v1.10.1

**The portable `jsonMasks` classifies through a table and flushes whole
words.** The reference is the production path below the kernel threshold and
everywhere `purego` runs, and it was written as an oracle — per input byte, an
index division, five flag tests and five read-modify-writes. Parsing a 50-byte
document in simdjson spent 49% of its time in it; the rewrite takes that
document's `Parse` from 204 ns to 170 through the public API, with `Scan` and
`Valid` down about 10%. Inputs at the kernel threshold and above are untouched.
Output is bit-identical, held by the conformance differential against every
assembly tier.

## v1.10.0

**`KernelThreshold`** — the input length below which a named operation's guard
takes the portable reference implementation instead of a vector kernel, on the
running architecture: `KernelThreshold("IndexByte")` is 1024 on amd64 and 64 on
riscv64, because the portable loop it competes with differs that much. The
number was previously knowable only by reading generated guard code, and a
caller in another repository deriving its own constant from memory paid 35% on
an encode for a call that looked like a kernel and ran a byte loop. Generated
per architecture from the same manifest as the guards, and a test now parses
all 6,789 generated guards across the six architectures and holds each one's
number to the manifest.

**`JSONQuote`** — copies a string into a JSON encoder's output with the
escapes written in place, instead of stopping at each byte that needs one. The
destination must hold six bytes per input byte, the worst case of every byte
becoming `\u00XX`; that reservation is what removes the per-byte bounds check,
and it makes the kernel one for callers that size their destination up front.
An append-based encoder that grows as it writes measured slower using it than
not, both before and after fixing its buffer sizing — recorded in simdjson's
docs/wrong.md — so the kernel ships for the callers whose allocation matches
its precondition.

**`JSONCopyValid`** — `JSONCopyRun` with UTF-8 validation folded into the same
pass: copies the bytes an encoder may write verbatim, proves them valid UTF-8
as it goes, and returns the count or a negative value on invalid input. Needs
no extra output space — it still stops at the first byte needing an escape.

**`JSONCopyRun` finishes its tail with a block, not a byte loop**, and its
guard threshold drops from 64 bytes to 32. The tail was a byte loop and it
showed: 32 bytes took 4.1 ns and 48 took 9.3, one block plus sixteen
single-byte steps. An overlapping block ending at the input's end re-reads
bytes already known clean, which is cheaper than stepping over half of them:

	bytes     32     40     48     64     96
	kernel  4.26   4.89   4.96   4.64   5.48

**Committed benchmark baselines no longer name the recording machine's CPU
model.**


## v1.5.0

**`MaskBitsAny` gets a four-wide kernel**, chosen automatically when the set has
four bytes or fewer.

A vector unit compares every byte of the set whether or not the caller supplied
them, so a set of four padded out to eight did twice the work for the same
answer — and four is the size real sets tend to be. A JSON parser wants `{}[]`
and `" \t\n\r"`; both are four.

Measured on 1 MiB with ~40% of bytes matching:

	{}[]           25775 ns -> 12512 ns   2.06x   84 GB/s
	" \t\n\r"       25762 ns -> 12576 ns   2.05x   83 GB/s
	{}[]:,         unchanged, 41 GB/s — six bytes still needs the eight-wide form

The API does not change. `MaskBitsAny` dispatches on `len(chars)`, so callers
get it for free; five to eight bytes take the existing kernel and more than
eight still takes the portable path.

This came out of profiling simdjson, where the two mask passes over the document
were 15% of a parse and both had four-byte sets.

## v1.4.0

**`MaskBits`, `MaskBitsAny`, `MaskBitsLess`** — one bit per input byte instead
of a list of offsets.

`IndexAll` answers "where does this byte occur" with four bytes per match. When
matches are rare that is the small answer. When they are common it is not: a
JSON document is around 40% structural characters, so the offset list comes out
four times the size of the document it describes, and the store is most of the
kernel's work. These write a bit per byte — an eighth of the input, whatever the
density.

Measured on 1 MiB of input with ~40% matching, against the same question asked
as offsets:

	one byte   249987 ns ->   6700 ns   37.3x   156 GB/s
	a set      570806 ns ->  25573 ns   22.3x    41 GB/s
	below c            —     6770 ns            155 GB/s

The speed is not cleverness, it is the absence of a store. On a target with mask
registers the whole loop is `vpcmpeqb` into a predicate and `kmovq` out of it —
two instructions per sixty-four bytes, no compress, and no count. Without the
compress the lane count is no longer held down to the width of an int32 index
vector, so these run sixty-four bytes at a time where `IndexAll` runs sixteen.

The point is what the caller does next. A bitmask is the right form when the
next question is itself bitwise, and for a parser it usually is: "which quotes
are escaped" is an add and two shifts over the backslash mask, "which bytes are
inside a string" is a prefix XOR over the quote mask, and "which structural
characters survive" is an and-not. All of it sixty-four bytes at a time with no
per-match work. In simdjson, replacing offset lists with these took stage one
from 1.76 ms to 370 µs on a 1.17 MB document.

`MaskBitsLess` exists for the range tests a set of eight cannot express —
finding the control characters a JSON string may not contain is thirty-two
values as a set and one call as a comparison.

Reaches amd64 (sse2/avx2/avx512), arm64 (neon/sve2) and riscv64 (rvv). ppc64le,
s390x and loong64 decline them for the usual ABI reasons and run the portable
path. 6,751 kernels in total.

Also in this release: `spec.Kernel.UnclampedDst`, for kernels whose destination
is not the same length as their source. The generated threshold guard clamps
every slice operand to the shortest, which is right for an elementwise kernel
and silently processes an eighth of the input when the destination holds one bit
per source byte. It did exactly that until the differential test against
`IndexAllAny` caught it.

## v1.3.0

**`IndexAllAny`** — the positions of every byte matching any of up to eight
values, in one pass.

`IndexAll` takes a single byte, so a parser wanting several delimiters at once
had to ask several times and read its input once per delimiter. A JSON
structural index needs six. Measured against six calls to `IndexAll`:

	n=4096      5811 ns -> 2213 ns   2.63x
	n=65536    94820 ns -> 32947 ns  2.88x
	n=1 MiB     1520 us ->  561 us   2.71x

The set arrives packed into a uint64, one byte per slot, because six integer
arguments is the SysV limit and the single-byte form already spends all of
them. A caller with fewer than eight repeats one — a duplicate compare is free
and an extra pass is not. More than eight takes the portable path.

Reaches the tiers with a compress instruction, avx512 and sve2, exactly as
`IndexAll` does; elsewhere the portable path runs. 6,733 kernels in total.

This came out of measuring simdjson against minio/simdjson-go, which is 1.3-1.8x
faster on amd64 because it computes structural positions in one fused pass while
a six-call structural index reads the document six times.

## v1.2.0

**Rank-1 update, Givens rotation and swap.** `RankOneInto`, `Rotate` and
`Swap` — BLAS calls them ger, rot and swap.

These were added expecting them to speed up a matrix decomposition. They do
not, and entry 72 of docs/wrong.md records why. LAPACK works on submatrix
windows with a scaling factor throughout — `ger` gets a strided column, and the
blocked `gemm` gets an alpha of -1 and a leading dimension that is the parent's
width — so every fast path in simdblas rejects it, `ger` merely being the one
looked at first. `trsm` is not accelerated at all.

They are worth having on their own terms: a contiguous rank-1 update is 8.2x
the portable path, a rotation 13.3x and a swap 4.4x. That is what they claim.

Against the portable path, float64, Zen 5:

	RankOneInto   n=64     8.18x
	              n=512    5.05x
	              n=2048   2.40x
	Rotate        n=1000  13.30x
	              n=100k   6.25x
	Swap          n=1000   4.36x
	              n=100k   2.63x

59 new kernels, taking the total to 6,730. `ger` reaches amd64, arm64, riscv64
and loong64; ppc64le and s390x decline it for the usual ABI reasons. arm64
declines `swap`, where LLVM will not vectorize the two-store loop; the portable
path covers it there.

`RankOneInto` has no leading-dimension parameter, so rows must be exactly n
apart. Six C arguments is the SysV limit and the choice was between dropping
lda or passing a stride that cannot be bounds-checked.

An abs-max reduction for BLAS `idamax` was written and dropped before release.
The existing `ArgMax` is hand-vectorized with explicit vector types and a
NaN-ordering contract, and an autovectorized version of the same shape was
refused by every target. Reproducing that machinery was not worth it for a
routine LU calls once per column, against `ger`, which it calls once per column
over the whole trailing matrix.

## v1.1.0

**Opt-in parallel matrix routines.** `MatMulParallelInto` and
`GemvParallelInto` divide the work across goroutines and are otherwise their
serial counterparts — the result is bit-identical, because the split is by
output row and no element's summation over k changes order. There is a test
asserting that rather than a comment claiming it.

Measured on a 32-thread Zen 5, square float64, serial against parallel:

	MatMul   n=128    1.00x   below the threshold, runs serial
	         n=256    3.86x
	         n=512   13.18x
	         n=1024  16.02x

	Gemv     n=512    1.00x   below the threshold
	         n=1024   2.62x
	         n=2048   5.79x
	         n=4096   3.34x

Everything else in the library still runs on one core, and that stays the
default. A function that spawns goroutines takes them from the caller, and a
batch of a thousand small multiplies wants one goroutine per multiply rather
than a thousand fan-outs — so parallelism is opt-in and lives in its own name.

Only the matrix routines are offered, and the reason is the numerical contract.
Splitting a reduction means combining partial sums, and the number of partials
would be GOMAXPROCS, so `Sum` would return different bits on a machine with a
different core count. That is the one thing this library promises never
happens.

`MatMulParallelIntoScratch` was written, measured and deleted before release: it
was slower than the unpacked parallel path at every size tried, including well
past the cache cliff where packing pays in the serial kernel. Entry 71 of
docs/wrong.md has the numbers and the reason — row-splitting and packing both
exist to keep the working set in cache, so the second one to run only pays.

## v1.0.5

Structure only, and the last of it. **No API change.**

`vec_test.go` was the one file left in the repository root that was
`package simd_test`, was not an example, and did not call an `export_test.go`
hook. It moved to `internal/tests/arrays`, which puts the root at **67 files**
— down from 114 two releases ago.

That is the floor without changing what the library is. Forty-six of the 67 are
`package simd`: 7,236 lines across a dozen topics, largest file 483. Go
requires a package to occupy one directory, so fewer files would mean larger
ones rather than folders. The remaining 21 are tests Go pins there — seven
reading unexported state, eight runnable examples that stop being examples of
this package if they move, five calling `export_test.go` hooks that do not
exist outside that directory, and the helpers those five share.

For scale, in the standard library `math` is 67 files in one directory,
`net/http` is 71 and `time` is 38. A flat package is the idiomatic shape for a
library like this one; the folder structure belongs in `internal/`, which is
where it is.

## v1.0.4

Structure only. **No API change** — every exported name, signature and result
is exactly what v1.0.3 had, so there is nothing to update when you upgrade.

The repository root went from 114 Go files to 68: the 46 that are `package
simd`, plus the 22 test files Go requires to sit beside them. Everything that
only calls the public API moved to `internal/tests/`, split by topic — arrays,
reduce, text, search, encode, dsp, matrix, and a docs package that checks the
repository's own documentation against the tree.

`internal/tests/README.md` records what could not move and why: tests of
unexported behaviour, the `export_test.go` hooks and their callers, and the
runnable examples, which only render on pkg.go.dev while they sit beside the
package they document.

## v1.0.3

**arm64 NEON is verified on real hardware.** An Apple M1 Pro, macOS 25.5.0,
go1.26.5: the accelerated suite and the portable suite both pass, including the
69-second differential run in `internal/conformance` that checks every tier
against the reference and against the other tiers. The report is in
`testdata/hardware/darwin-arm64-neon.md`.

This is the first time any of these kernels has executed on an arm64 chip
rather than under qemu, and it moves the second row of the README's
verification table. M1 has NEON and no SVE2, so `arm64 sve2` stays emulated and
the arm64 line in that table is now two rows rather than one. Wall-clock is
still unmeasured everywhere except amd64.

Five tiers remain unverified: arm64 sve2, riscv64 rvv, ppc64le vsx, s390x vx
and loong64 lasx.

### Everything else in this release is structure

`make hardware-report` runs the suite and writes the report file, so sending
one is a command rather than a transcription exercise. It does not abort when
the suite fails, because that is the report worth having. `make hardware-bench`
adds timing separately, since benchmarks need a quiet machine and correctness
does not.

The benchmarks moved to `internal/benchmarks`, taking the root from 132 Go
files to 114. They only ever called the public API. The recorded baseline is
unaffected — `benchcheck` matches on benchmark name rather than package path,
and all 674 names still resolve.

Fourteen directories gained a README saying what is in them, and the root
README gained a layout tree. The C in `csrc/` is the source; the assembly under
`internal/<arch>/` is the output; `make codegen` turns one into the other.

Also names [kelindar/simd](https://github.com/kelindar/simd) in the comparison
— the closest relative, since it reaches its two instruction sets the same way
— with a measured demonstration of the difference the accuracy contract makes,
reproducible from `docs/comparison`.

## v1.0.2

Adds the MIT license.

v1.0.0 and v1.0.1 shipped without a `LICENSE` file, which under default
copyright means all rights reserved: legally unimportable by anyone who checks,
which is every company with a policy scanner. Those two versions are cached
immutably by the module proxy and cannot be fixed in place, so **use v1.0.2 or
later** — it is the first version that is actually licensed. Nothing else
changed.

Datastar's own copyright notice now sits beside the bundle in
`cmd/site/assets/`, which MIT requires of anyone redistributing a copy, along
with a README explaining why the file is committed rather than fetched. Both
land inside the `embed.FS`, so the running binary serves the attribution too.

## v1.0.1

Lowers the required Go version from 1.26.2 to 1.25.0, and says so in the README,
which had never stated one.

The `go` directive was pinned to the patch release of whatever toolchain
happened to run `go mod tidy`. Nothing needed it: the real floor is
`golang.org/x/sys`, which asks for 1.25.0, and the whole suite passes there —
`make verify`, the `goexperiment.simd` build, `test-gates`, `test-riscv64`,
`test-loong64` and `test-cross`. Go would have downloaded the newer toolchain
automatically for most people, but not for anyone with `GOTOOLCHAIN=local`, a
distribution-pinned Go, or an air-gapped build, and none of them were getting
anything for the trouble.

## v1.0.0

**simd.go 1.0.** The API is stable and the numerical contract is one you can
build on. The import path is unchanged: `github.com/sebishogun/simd`.

### What 1.0 commits to

Every exported function keeps its name, signature and meaning for the life of
v1. The numerical contract in package `kernel` — elementwise operations
bit-identical on every tier, floating-point reductions using a fixed
sixteen-accumulator tree so a 128-bit and a 512-bit machine agree exactly, `Dot`
never contracting into FMA, `Fast*` the only operations permitted to trade any
of that away — is part of the promise, not an implementation detail. A future
release may make an operation faster. It may not change what it returns.

What is *not* covered: anything under `internal/`, the generated assembly, the
`tools/` module, and the `goexperiment.simd` vector type in `vec.go`, which
tracks an experiment Go itself has not settled. `v2.0.0` is reserved for the day
Go's intrinsics need no build flag.

### What 1.0 does not claim

Speed, on six of the seven tiers. Correctness is verified on real hardware for
amd64 and under emulation everywhere else; wall-clock is verified only on
amd64. The README carries a per-architecture table saying so, and where it says
*unmeasured*, no speed claim in this repository applies — every published number
is amd64.

Emulation proves semantics and nothing about timing — qemu does not model a
pipeline. It also cannot catch a chip that implements an instruction
differently from the emulator, an errata, or a scalable vector length nobody
configured. `testdata/hardware/` takes a report per machine and CONTRIBUTING
explains how to send one in two commands; the table moves as they arrive.


### The baseline x86-64 tier: 673 kernels to 789

A legacy SSE instruction reading a constant pool needs its operand 16-byte
aligned and has no unaligned form, and nothing promises the alignment of bytes
inside a TEXT symbol. That cost the sse2 tier 170 kernels — the largest
addressable gap in the repository.

The pool now goes into a separate RODATA symbol, which the linker does align,
and the one instruction that reads it is emitted as a Plan 9 mnemonic. It is
safe because the mnemonic assembles to *exactly* the encoding it replaces, so
every branch displacement stays correct — and that is now
`TestRespelledEncodingsMatchClang`, run for all forty mnemonics against both
register ranges, rather than an argument. **No alignment refusals remain on any
tier.** Entries 61 and 67.

### The transcendentals now vectorize everywhere

Thirteen accurate transcendentals refused on RVV and five float64 ones on
NEON. Two earlier investigations concluded this needed hand-written intrinsics.
It did not.

NEON was a cost model declining a good trade, which a pragma overrules: +20
kernels. RVV was `llvm.is.fpclass`, which that target cannot price, emitted
from a denormal test written as a float comparison — asking the same question
of the bits is an ordinary unsigned comparison. riscv64 went 801 to 849, and
**no accurate transcendental refuses on any target**. Entries 68, 69 and 70.

### Documentation

`docs/guide/` is five prose pages — arrays, text, search, encodings, signal —
each explaining the problem, showing the code, and saying where the operation
stops paying.

The README's function table claimed every entry had a runnable example. It was
false for 100 of 109 rows. All 109 have one now, and
`TestReadmeTableHasExamples` fails the build if a row appears without one.
`TestGuidesNameRealFunctions` does the same for the prose.

Writing them found three documentation bugs that only executing catches: the
table named three functions that do not exist, a `ParseInts` snippet silently
dropped its last field, and a `BitPack` snippet produced zeros because
unpacking needs one word more than packing.

### Fixed

- `make test-cross` hung forever building `cmd/site` under emulation. It now
  runs the library packages, which is what the lane is for.
- Five exported functions had no doc comment, three of them sharing one above
  the group — legal Go that renders as nothing on pkg.go.dev.
- 90 public wrappers were never called by any test. The conformance suite tests
  kernels; a wrapper reaching for the wrong kernel field is invisible to it.
  443 of 461 are now exercised through the public API.


### The qemu lanes were never reading their flags

The emulators here follow the binfmt_misc preserve-argv[0] convention, so run
by hand they consume the first argument as the guest's `argv[0]`. Every
`-test.run`, `-test.count` and `-test.short` was silently discarded, and a Go
test binary with no flags runs its full default suite and prints `PASS` — so
the lanes were green while testing something other than what they said.

The casualty that matters is `-require-accelerated`, the guard against a
vacuously green lane, which **had never once been evaluated**. Fixed by
passing the binary path twice; `simdinfo -argv0-probe` now asserts flags
arrive before any lane is trusted. Entry 41.

### New operations

- **UTF-16.** `AppendUTF16`, `AppendUTF8`, `UTF16Len`, `UTF8Len`. The
  conversion is a dependent scan, so the design finds ASCII *runs* and widens
  those. Against what a `[]byte` caller writes today at 64 KiB: 83x encode and
  108x decode on ASCII, 5.4x and 12.5x at 10% non-ASCII, 2.0x and 2.4x on
  all-non-ASCII, which is the honest floor.
- **Multi-pattern search.** `MultiSearcher`, anchored on each needle's *rarest*
  byte rather than its first. The obvious first-byte filter is 252x **slower**
  when those bytes are common; this is 32x faster than the k-loop at k=64 in
  both the rare and common cases, and still 3.3–12.5x ahead on an alphabet
  where no byte is rare.
- **Case-insensitive search.** `IndexFoldASCII`, `ContainsFoldASCII`,
  `CountFoldASCII`, composed from `ToLowerASCII` and `Index`.
- **JSON escaping.** `AppendEscapeJSON`, `NeedsEscapeJSON`.
- **`ParseUints`**, 4.5x `strconv` across 64 to 200,000 fields. Not a wrapper
  around `ParseInts`: the signed limit of 2^63 would reject half the domain.
- **`ParseFloats`**, 2.3x on two-decimal CSV and 2.0x on scientific, and
  **bit-identical to `strconv`** because Clinger's fast path is exactly
  rounded. Deliberately **no kernel** — every target refused it and both
  refusals were right.
- **`FastCumSum` / `FastCumProd`**, and an exact accelerated integer
  `CumProd`. See below.
- **Packed-B GEMM.** `MatMulIntoPacked`, `PackBInto`, `MatMulIntoScratch`.

- **Sorted sets.** `IntersectInto`, `DifferenceInto` over sorted,
  duplicate-free integer slices — the shape a posting list is already in. The
  two-cursor merge vectorizes on nothing; this compares a tile of eight against
  eight with no reduction and no branch, then retires whichever block has the
  smaller maximum. Ahead at sparse and total overlap, a tie at 50% where no
  block skips. **No `Union`**: it emits everything in order, which is a merge,
  and merging two sorted vectors needs a bitonic network this library does not
  have.
- **Batched binary search.** `LowerBoundInto`. One search is a chain of
  dependent probes; a batch turns the loop nest inside out and becomes
  elementwise over the queries. 6.7x on sse2, 10.4x on avx512 — and the
  decomposition is worth knowing, because sse2 has no gather at all: most of
  the win is *branchlessness*, and lanes are the 1.55x on top.
- **Rank and select.** `RankTableInto`, `Rank`, `Select`. A composition over
  `OnesCountInto` and `CumSumInto` rather than a kernel, and it says so. What
  it adds is the exclusive-prefix off-by-one that makes `Rank` a single
  addition and `Select` its exact inverse.
- **Sliding-window extremes.** `RollingMinInto`, `RollingMaxInto`, with IEEE
  754-2019 minimum so NaN propagates. **The doc states where it stops paying**:
  12.8x against a hand-written deque at a window of 4, 1.5x at 32, and 0.75x at
  64. Above about 48, write the deque.
- **Sparse matrix-vector.** `SparseDot` per CSR row and `SpMVInto` for the
  loop. About **1.1x**, gather-bound, and documented as such — the accumulation
  contract is the reason to use it, not the speed.
- **Longest common prefix.** `CommonPrefixLen`, 21x a byte-at-a-time loop.
- **Varint widths.** `VarintSize`, `VarintLenInto`, `AppendVarints`. The
  emission is serial and always will be; what vectorizes is asking how wide
  each value is, which is what lets an encoder allocate once. 27x on the size
  pass, 1.7x end to end.
- **UTF-32 encode.** `AppendUTF8FromRunes`, closing a pair that had a decode
  direction and no encode — and whose kernel had been generated with no caller.
- **`FastAsinh`, `FastAcosh`, `FastAtanh`, `FastErf`.** These had kernels on
  every architecture, wired into the dispatch table and conformance-tested,
  with no public wrapper. The work was done and the door was missing.

### Prefix scans: what Fast means, and what it does not

`FastCumProd` is 3.65x on float32 and 1.76x on float64; integer `CumProd` is
2.17x and needs **no** `Fast` prefix, because two's-complement multiplication
is associative and the grouping is not observable.

Four variants were measured and **not** shipped: float64 `FastCumSum` (0.91x,
slower than the plain one on the tier most machines select), both integer
sums, and int64 product. The rule is the latency of the serial combine, not
associativity — every one of these is associative.

`Fast` here drops **agreement with a naive loop, not accuracy**. Blocked
summation has O(log n) error growth against a serial accumulator's O(n), so it
is *closer* to the truth: 7143x lower total error on the case that breaks
serial accumulation, asserted by a test. Entries 44 and 45.

### ppc64le: the intermittent crash is solved

Not one kernel — the r0-by-value violation, and **fifteen** kernels were doing
it, which is why two bisections over three candidates read coin flips.
Confirmed by interleaved experiment: violators registered crashed 5 of 20,
rule active 0 of 20. Entry 42.

### perf-model: throughput where this machine cannot measure

`make perf-model` runs llvm-mca over each kernel's inner loop against the same
kernel compiled without vectorization, for amd64, arm64, ppc64le and s390x.
Nothing models below 1.2x. Validated against measured amd64 to within 5–12% on
the avx512-versus-avx2 comparison. riscv64 and loong64 are excluded with
stated reasons. Entry 49.

### make menu

An interactive picker over every make target, marked for the machine it is
shown on — a third of them cannot run on any given machine. Uses fzf when
present, falls back to a numbered list. Fixed three things that were broken
off Linux: `BENCH_PIN` defaulted to `taskset`, the qemu hint recommended an
install that cannot help on macOS, and `benchcheck` died with a bare
file-not-found on any architecture without a baseline.

### Corrected: the coverage table

The README gave s390x as 650 kernels. That double-counted registrations and
their wrappers; the real figure was 325 then and is 395 now, and it has risen
monotonically throughout. The other five entries were accurate when written
and had merely stopped being current. Current counts, from the generator:
amd64 2170, arm64 1455, riscv64 731, loong64 658, ppc64le 568, s390x 395.

### New operations

- **FFT.** Radix-2 Cooley-Tukey with a reusable plan, plus a real-input
  transform and the Hilbert transform built on it. 6.20µs at n=1024 against
  15.2ms for the naive DFT. `RFFT` returns the n/2+1 non-redundant bins and is
  54% faster than transforming as complex at 65536.
- **DSP.** Hann, HannPeriodic, Hamming, Blackman, Bartlett windows; full-mode
  convolution and correlation with a **measured** direct/FFT crossover at
  ~1234 taps, twenty times the 64 that folklore quotes.
- **Shifts, rotates and bit manipulation.** `Shl`, `Shr`, `Rotl`, `Rotr`,
  `OnesCount`, `LeadingZeros`, `TrailingZeros`, `ReverseBits`, `ByteSwap` for
  all eight integer types, following Go's semantics rather than C's undefined
  behaviour at and above the element width.
- **`ParseInts`**, five times `strconv.ParseInt` on delimited integers.
- **Predicates.** `IsNaN`, `IsInf`, `IsFinite`, `Sign`, `CountNaN`, `AnyNaN`,
  `NanSum`, `NanMean` — all composed from existing kernels, 1.3× to 12× ahead
  of the loop they replace.
- **Selection and shaping.** `TopK`, `BottomK`, `Interp`, and a blocked
  `Transpose` at 3.6× the naive loop.
- **Statistics.** `Histogram` and `Bincount`.
- **Transcendentals.** `Asinh`, `Acosh`, `Atanh` at ~2 ULP, and `Erf` at
  1.4e-7 absolute. `Erfc` deliberately has no kernel: the same approximation is
  1.2e-2 *relative* at x=6, which is the only regime anyone uses erfc for.

### riscv64: 19 kernels recovered, and a two-session mystery closed

`CompressInto`, `IndexAll` and `PartitionInto` now work on riscv64 — and
through `PartitionInto`, so do `Sort`, `Median` and `Quantile`. Coverage goes
from 698 to 717 slots.

The compress family had been excluded since the architecture was added, first
for a reason that was false (that RVV has no compress instruction; it has
`vcompress.vm`) and then for memory corruption nobody could explain. It was a
stack overflow: every kernel in the family spilled 640 to 2032 bytes against
the 512-byte budget a NOSPLIT function has.

The reason it took so long is the more useful part. The check that names this
had been dead on riscv64 the whole time, because `stackAdjust` had no case for
the architecture and measured every frame as zero. Underneath *that*, the
disassembly parser was deleting every arm64 immediate by cutting operands at
the first `#` — which on arm64 is the immediate prefix. Fixing the parser made
the checks work, which explained the corruption, which pointed at the per-type
lane counts that fixed it.

The same check immediately found ten more kernels — `ArgMin`, `ArgMax`,
`MinMax` on riscv64 — that were shipping over budget.

### Verification

- The frame-write and stack-budget checks now run on **every** architecture.
  They previously covered one of six.
- **`make test-gates`** runs the suite on a CPU that *lacks* the vector
  extension. Every other lane runs `-cpu max`, which is the one configuration
  where a mis-gated kernel cannot fail.
- **A threshold meta-test** fails when an operation dispatches above the
  suite's default input length without a test covering it. It found four text
  kernels being exercised only by benchmarks.

### Versioning

The release plan is now written down in ROADMAP.md rather than assumed. `v1.0.0`
is gated on five things. `v2.0.0` is reserved for when Go's intrinsics leave
`GOEXPERIMENT=simd` and the tier measurements get re-run against them.

### Complex reductions

`SumComplex`, `DotComplex` and `DotComplexConj` had no kernel on any
architecture. They now have one on every tier but s390x.

| n = 65536 | accelerated | portable | naive loop |
|---|---|---|---|
| `DotConj` complex64 | 6.89us | 80.5us | |
| `DotConj` complex128 | 15.6us | 55.7us | 155us |
| `Sum` complex128 | 6.08us | 29.4us | |

Ten times the loop a caller writes by hand, which is the comparison that
matters: `sum(a[i] * conj(b[i]))` is the inner product of signal processing and
Go had no fast way to spell it.

The sixteen accumulators the numerical contract requires are held as two
vectors of eight rather than one of sixteen. For `complex128` the obvious
spelling needs six 128-byte vectors live at once, which measured 1120 bytes of
spill against a 512-byte budget and lost the kernel on every amd64 tier.
Splitting each accumulator into halves changes no arithmetic — lane `k%16`
still receives element `k`, and `CombineTree`'s first step *is* adding the two
halves — and it fits everywhere, including SSE2.

### NaN and predicate helpers

`IsNaNInto`, `IsInfInto`, `IsFiniteInto`, `SignInto`, `CountNaN`, `AnyNaN`,
`NanSum` and `NanMean`.

None of them adds a kernel and none needs one: every question reduces to a
comparison the library already vectorizes, so all eight are accelerated on
every architecture at once. `IsNaN` is `NotEqualInto` of a slice against itself
— the IEEE definition read literally, since NaN is the only value not equal to
itself, with no bit-masking and no special case for the payload.

`IsFinite` is deliberately not "not infinite": a NaN fails the comparison
against `+Inf` as unordered rather than as less-than, so it needs a strict `<`,
which excludes both. `sign(NaN)` is NaN rather than zero, and `NanMean` returns
the surviving count alongside the mean, because an average over three points
out of a thousand is not an average.

### Fixed: the regression gate could not gate anything

`benchcheck` compared the *median* of six samples against a flat 25% threshold,
on bodies running at 6 to 15 ns/op. Two consecutive runs on an idle machine,
same commit and same binary, reported sixteen regressions and then five, with
**zero overlap** — every one a transient that reached the median.

It now compares *minimums*, because benchmark interference is one-sided: a
frequency drop or a migration can only make a run slower, so the samples are
the true cost plus a non-negative contaminant and the minimum is the
maximum-likelihood estimate of the true cost. It cannot hide a real regression,
since a real regression raises every sample including the best one. Sixteen of
the twenty-one false positives disappear.

### Four operations leave the portable path on x86

`HexDecode`, `Base64Decode`, `Median` and `Quantile` were four of the eight
operations still running plain Go on amd64. Each was blocked on something
different, and only one of the four was blocked on what the roadmap said.

**HexDecode** genuinely needed the two-value return. It reports a count and a
validity flag, and the generator's result slot held one value, so it was
portable on every architecture for a reason that had nothing to do with the
hardware. The kernel validates a block without branching and only decodes one
that is wholly valid, dropping to a scalar tail to find the exact offset of a
bad character. Against `encoding/hex` at 1 MiB: **44.2us versus 660us**, 15x,
at 22 GB/s.

**Base64Decode** returns one value and was never blocked on that at all. It was
excluded for spilling 576 bytes on AVX2 and 704 on AVX-512, past the 512-byte
budget a NOSPLIT function has. Four bytes in and three out makes LLVM build a
shuffle tree whose cost grows faster than the vector width, and left alone it
picked a width that spilled. Pinning that width — 64 where `__AVX512F__` is
defined, 32 elsewhere, because 64 everywhere would cost the AVX2 and ppc64le
tiers — gives **43.7us at 1 MiB against 894us portable and 399us for
`encoding/base64`**: 20.5x and 9.1x.

**Median and Quantile** now run a quickselect around the accelerated partition.
float64, taking the median of nine runs:

| n | accelerated | portable | via `slices.Sort` |
|---|---|---|---|
| 4096 | 10.4us | 15.0us | 119us |
| 65536 | 157us | 762us | 3.80ms |
| 1048576 | 3.12ms | 13.4ms | 76.7ms |

New `MedianInto` and `QuantileInto` take the scratch buffer from the caller and
allocate nothing, in the same relationship `SortInto` has to `Sort`.

That leaves `EMA`, `CumSum` and `CumProd` as the only operations still portable
on amd64, and all three are permanently so: each is serial through its own
output, and the contract forbids the reassociation that would break the
dependency.

### Fixed

- **Signed zero in `Sort` is now documented.** The accelerated and portable
  paths can place `-0` and `+0` differently — 848 of 4096 positions on a slice
  containing both — because `-0 < +0` is false and the two therefore tie under
  the `<` that defines the order. Every such output is a correct sort and every
  differing pair is `==`. Making them agree needs a comparator closure, already
  measured at 2.5x slower. `Median` and `Quantile` inherit the caveat.

## v0.2.0 — 2026-07-28

### ppc64le: 281 kernels become 468

The largest coverage gain in the library, and it came from discarding a
constraint that was never real.

clang reaches its constants on ppc64le through `r2`, the TOC pointer, which Go
does not maintain for these objects. Power9 has no PC-relative data addressing,
so reaching an appended pool appeared to require either `bcl`/`mflr` — which
clobbers the link register and wants a save slot in a protected zone the kernels
already use down to −256 bytes — or a dependency on whatever `r2` held on entry,
which is unsafe under `-shared`.

Neither is necessary. **Go's own assembler materialises a symbol address in two
instructions with no TOC involvement**, because Go builds non-PIE and the
address is a link-time constant. That was settled by building and running a
probe under emulation rather than by reasoning about it.

So the pool becomes a standalone `GLOBL`, `R2` is pointed at it with one `MOVD`
in the prologue, clang's two global-entry instructions are replaced with nops in
place, and every TOC16 immediate is rewritten as an offset from the pool base.
Two other checks were in the way, and both were wrong rather than conservative:
`.TOC.` was counted as an undefined *call* when it is a linker-defined data
anchor, and `r2` was rejected outright when its only uses here are clang's own
prologue and reads.

One kernel of 469 corrupts memory with the rewrite enabled and is not
registered, so `CountAny` keeps the portable path it already had on this
architecture. It was bisected to `countAnyVSX` in fourteen runs and its
addressing then verified correct by hand, so the fault is something else about
that kernel — see the note at its skip.

Verified on emulated ppc64le with 3.5 million clean fuzz executions.

### Also

- **Sorting**: `Sort`, `SortInto`, `Argsort`, `PartitionInto`, `SortedIndex`.
- **N-ary arithmetic**: `AddAll`, `MulAll`.

## v0.1.1 — 2026-07-28

**Fixes a crash.** `PartitionInto` dereferenced a nil function pointer on every
architecture without a hardware compress instruction — s390x, ppc64le, riscv64,
loong64, and amd64 below AVX-512 or arm64 without SVE2. That is most machines,
and it was a panic on the first call rather than a wrong answer. Anyone on
v0.1.0 who touches the sorting API should take this.

The cause was two things at once: the portable implementation was never
registered in the reference, because a scripted edit stopped matching after
gofmt reflowed the surrounding lines and failed silently; and `PartitionInto`
called through the slot unguarded, although `CompressInto` — the same
arrangement — already had the guard. Both are fixed, and the emulated s390x
lane is what found it. See entry 12 of [docs/wrong.md](docs/wrong.md).

### Added since v0.1.0

- **Sorting**: `Sort`, `SortInto` (allocation-free), `Argsort`, `PartitionInto`
  and `SortedIndex`. A quicksort around a compress-based partition, 19–27%
  faster than `slices.Sort` above 16K elements on float64. NaN sorts last, as
  in `Median` and `Quantile`, which differs from `slices.Sort`.
- **N-ary arithmetic**: `AddAll` and `MulAll` over any number of slices in a
  single pass, with the element type enforced by the compiler.

## v0.1.0 — 2026-07-28

The first tagged version. Everything below is new; there was nothing before it.

### What this is

SIMD-accelerated slice operations for Go on every architecture with a vector
unit, without cgo. 309 exported functions, 5,247 generated kernels across nine
targets: amd64 sse2/avx2/avx512 (1664), arm64 neon/sve2 (1121), s390x vx (614),
riscv64 rvv (558), loong64 lasx (506), ppc64le vsx (281).

### What is covered by compatibility

The **exported API of the root package** — names, signatures and documented
semantics — from this tag onward.

Explicitly **not** covered, and it matters here more than it usually would:

- **Which kernels exist on which architecture.** Coverage is uneven and will
  move as ABI problems are solved. A function whose kernel is missing on a
  target runs the portable implementation, so this is a performance property,
  never a correctness one.
- **The measured numbers.** They are from one machine and will differ on
  yours.
- **`internal/`, `tools/` and `csrc/`.** The generator, the reference and the
  kernel sources are implementation.

### The accuracy contract

Every operation is bit-identical on every instruction set, including for NaN
payloads, ±Inf, ±0 and denormals. Two exceptions, both opt-in by name: the
transcendentals guarantee a stated ULP bound rather than bit identity, and
`Fast*` promises 3.5 ULP and gives up agreement between architectures.

### Known limits

- **Nothing outside amd64 has run on real hardware.** Every other architecture
  is verified under emulation, which proves semantics and proves nothing about
  timing.
- **ppc64le (281 kernels) and s390x (614) are partial**, both because clang
  uses a register the Go runtime owns and no compiler flag stops it. ppc64le
  additionally reaches its constants through the TOC pointer, which Go does not
  maintain for these objects.
- **`Compress` and `IndexAll` are accelerated on AVX-512 and SVE2 only**, and
  permanently so. Those are the two instruction sets with a compress
  instruction, and the operation cannot be vectorized without one: where each
  element lands depends on how many earlier ones matched, which is a real
  loop-carried dependency rather than a compiler shortcoming. The other seven
  targets run the portable loop, which is what their compilers would emit
  anyway.
- **`HexDecode` is portable everywhere**, because it returns two values where
  the generator's result slot holds one.
- **Not yet built:** sort/argsort, and packed-panel
  cache blocking above the GEMM microkernel. See [ROADMAP.md](ROADMAP.md).

### Where to start

[`example_test.go`](example_test.go) has a runnable example for every operation
— checked by `go test`, so none of them can drift — and
[`docs/examples/`](docs/examples/) has complete programs. The README opens with
a table indexed by what you are trying to do rather than by what the operation
is called.

[`docs/wrong.md`](docs/wrong.md) is the twenty-two things that turned out not to
be true, which is the part of this project most worth borrowing.

### One deliberate semantic choice worth stating

`MatMul` does not skip zeros in `a`. An earlier draft did, and it is not the
free optimization it looks like: under IEEE 754 a zero times an infinity is a
NaN, and skipping suppresses it. BLAS does not skip, numpy does not skip, and
the standard says what the answer is — so neither does this. It is also what
makes the register-blocked microkernel possible, since in a tile that test
would guard a single fused multiply-add rather than a whole row.

### On Go's own SIMD intrinsics

Go 1.26 shipped them behind `GOEXPERIMENT=simd`, targeted at the language in
1.27. They do not replace this library and are not a competitor to it: they
cover amd64 and arm64 only, they do not reach SVE2 or RVV, and a library cannot
require its consumers to set a `GOEXPERIMENT`.

They do win in one band, and will keep winning there. An assembly kernel cannot
be inlined, so it pays a fixed call boundary — ~1.4 ns at the floor, 50–65
cycles once `VZEROUPPER` and register save/restore are counted. Above n ≈ 128
that is a rounding error. Below n ≈ 64 it is most of the runtime, which is why
the dispatcher gives up and runs a scalar loop there. An inlined intrinsic pays
none of it.

The plan is therefore additive: build the tier behind the experiment flag,
benchmark it against both the assembly tier and the scalar fallback, and wire
up only the sizes where it wins. When it needs no flag, re-measure and adopt it
wherever the number says so. The bit-identity contract binds it like every
other tier.

### Measured on a Ryzen AI MAX+ 395 (Zen 5, AVX-512)

Against the portable Go build, integer and saturating arithmetic, geomean over
the set: **−86% time, +593% throughput**. `SatAdd` on int8 at n=4096 is −99.1%.

Against `bytes` and `strings` — the harder comparison, since `bytealg` is
assembly on four of the six architectures — geomean **+186%**. `LastIndex` at
n=4096 is +8309%; `IndexAny` at 1 MiB is +1084%.

Against `encoding/base64`: −42% to −63%.

`MatMulInto` against the naive kernel it replaced: **−60% to −86%** depending
on size, reaching 260 GFLOP/s on f32 — about 90% of this core's single-thread
AVX-512 peak. `GemvInto` is new, and is bit-identical to `Dot` per row.

`CompressInto` against the scalar filter loop, geomean **−51%**; at 1 M
elements and 50% match density, **−93%** (1.29 GiB/s → 19.3 GiB/s).

`Fast` against accurate: `FastSin` −45%, `FastExp` −43%.
