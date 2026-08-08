# Kernel backlog campaign (goal-locked)

Standing goal: complete tasks #207-#212 (+#205 project backlog after),
kernels maximally performant, and ALWAYS re-run downstream benchmarks
(simdjson bench-check + affected shape sweeps) when a change touches an
operation a consumer uses. simdjson uses: JSON* family, MaskBits*,
CountAny, IndexByte/Any/AnyOrLess/All, ValidUTF8, PopCount — and will use
dtoa (#208) + itoa (#211) when they land. Order: 207, 208, 209, 210, 211,
212, then #205 projects (separate repos: simdhttp, simdcbor, ...).

## #207 CRC32C + Adler-32 (csrc/checksum.c)
- Adler-32: THE win — stdlib hash/adler32 is scalar Go everywhere. u8→u16
  lane sums, two accumulators (a=sum bytes, b=sum of prefix sums via
  b += n*a_incr trick), NMAX=5552 block before mod 65521. Vectorizes on
  every tier. Expect 10-20x.
- CRC32C: PCLMUL 4-lane folding under __PCLMUL__ (+ arm64: __crc32cd
  builtins under __ARM_FEATURE_CRC32 — check tier MArch has it; armv8-a
  default includes CRC? armv8.1+ yes; verify target.go flags). stdlib
  amd64/arm64 already have strong asm (SSE4.2 crc32 3-way ~15GB/s) —
  MEASURE HONESTLY vs hash/crc32 Castagnoli; ship only the arms that win,
  wrong.md the rest. Ref: small nibble-table Go crc32c + RFC1950 adler.
- Manifest: Group "Bytes", Fields Adler32 / CRC32C, args (data []byte,
  seed u32) -> u32. Result u32 (spec has U32 scalar? check spec.Type for
  scalar u32 — Result used Int/scalars; may need spec.U32 support!).
  CArgs out(), base(data), val(seed)?, lenOf(data). RefWhen len==0.
- Tests: differential vs hash/crc32.Checksum(Castagnoli) + hash/adler32
  across sizes 0..8K + seeds/rolling updates. Bench vs both stdlib.

## #208 dtoa (Schubfach) — the simdjson marshal gap [IN PROGRESS]

RECON DONE (post-v1.16.0): simdjson has the full Go Schubfach --
schubfach.go (235 lines: decimal{sig u64, exp i32}, schubfach() on raw
bits, needs bits.Mul64 = __int128 in C) + pow10_table.go (~350 entries as
pow10Hi/pow10Lo split u64 arrays, ~5.6 KB total; pow10Sig(i) indexes at
i - pow10Min) + a render layer (schubfach_render_test.go names it) that
turns decimal into bytes, checked against strconv everywhere.

PORT PLAN:
1. csrc/dtoa.c: port schubfach() verbatim (u128 = unsigned __int128;
   table as static const u64 arrays -- 5.6 KB constant pool: amd64 fine,
   arm64 likely, SkipOn the rest initially) PLUS the digit render to
   bytes (port simdjson's renderer; the kernel returns the final string).
   Signature: simd_dtoa_f64(i64 *out_n, u8 *dst, f64 v) -> writes
   shortest repr incl sign/exponent form IDENTICAL to simdjson's
   renderer (which matches strconv 'g'-shortest for JSON: check its
   exact format rules -- JSON never emits +, uses e-notation per
   encoding/json rules). KEEP THE CONTRACT = simdjson's existing output.
2. simd manifest: Group Bytes, Field DtoaF64, Params (dst SliceU8, v
   F64), Result Int; RefWhen len(dst) < 32; Threshold 0 (single value!).
   Ref = PORT the Go schubfach+render INTO internal/ref (copy from
   simdjson with attribution comment; simdjson later switches to calling
   simd, single source of truth ends up in simd).
3. Public: simd.AppendFloat64Shortest(dst []byte, v float64) []byte or
   Dtoa(dst, v) int style matching JSONQuote conventions.
4. simdjson wiring: its float render path (find call site of
   schubfach/render in marshal float encoders) switches to
   simd.Dtoa when tier has the kernel (fusedValid-style gate or just
   always -- ref is same speed as its own Go). Then MANDATED: simdjson
   bench-check x2 + MarshalFloats + canada-struct Marshal + SML rows +
   flat-float sweep vs sonic. Target: close the ~1.2x flat-float gap.
5. Single-value call overhead ~1.4ns vs ~20ns/value work: kernel pays
   off ~7%+ per value; batch variant (vals []f64, idx out) only if the
   single-value bench says the call boundary matters.

- C port of Schubfach (u128 mults via __int128 ✓ freestanding ok), one
  f64 -> shortest digits + exp. simdjson has Go Schubfach to differential
  against + its render layer. Kernel shape: batch API dtoa_f64(out
  []byte, vals []f64, idx []i32?) or single-value? sonic's win is
  per-value; call overhead 1.4ns vs ~20ns/value scalar — single-value
  kernel viable. Decide after profiling simdjson MarshalFloats row.
- AFTER landing: simdjson swap + bench-check + MarshalFloats/canada
  marshal rows + flat-float sweep. This is the one with the mandated
  downstream benches.

## #209 prefix-sum scan: u64/i64 Blelloch in-block shift-add + carry.
   Read wrong.md 44/45 FIRST (integer sum stays portable — different
   shape here, but verify with perf-model before shipping).
## #210 LZ4 block decode: vector overlap-copy; fuzz vs pierrec/lz4.
## #211 itoa SWAR batch emission (FormatInts upgrade + single-value).
## #212 tier-2: interleave/deinterleave + 8x8 byte transpose; RLE decode;
   sorted merge; vectorized xoshiro. + audit: ToUpper/ToLowerASCII,
   B64* already exist — bench vs named competitors, close gaps only.

## State
- #207 NEARLY DONE: csrc/checksum.c written+vector-verified (adler ACHUNK=32
  weighted form; crc32c PCLMUL fold-by-4 + crc32di drain, portable #else body
  for astcheck; known vectors + 300B fold-cross pass in standalone C).
  Manifest Checksum() source registered; kernel.Bytes fields Adler32/CRC32C;
  ref/checksum.go (nibble CRC FIXED: crc^=c first); public checksum.go
  wrappers (honest doc: stdlib asm strong on amd64, measure before switching);
  TestChecksumsMatchStdlib GREEN on avx512/scalar/avx2 incl rolling.
  REMAINING for #207: README index rows + ExampleAdler32/ExampleCRC32C (gate
  demands), bench vs hash/adler32 + hash/crc32-Castagnoli (SHIP-VERDICT for
  crc32c: if below stdlib asm everywhere, say so in doc + wrong.md; adler
  expect 10-20x), full six-arch regen + thresholds regen + make verify +
  qemu lanes + coverage-table refresh + CHANGELOG v1.16.0 + release ritual.
  No downstream simdjson use of checksums -> no simdjson bench needed for
  #207 (goal's downstream-bench clause applies from #208 dtoa on).
- Task list: #207 (this), #208 dtoa (wire INTO simdjson Marshal + bench-check
  + MarshalFloats sweep after), #209 prefix-sum, #210 LZ4, #211 itoa (also
  wire into simdjson), #212 tier-2 four + audits, #213 varint-decode/
  bitpacked-decode-audit/simd-hash/bitshuffle, then #205 projects.
- Goal hook active: all tasks, performant kernels, downstream benches when
  a consumer library can use a new kernel -- and WIRE the consumer when it
  makes it better, then bench it (user's addendum).
