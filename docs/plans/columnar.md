# Columnar compute (the arrow surface) — campaign plan

User greenlit (2026-08-08): build the Tier-1 "simd-arrow / columnar compute"
item inside the simd repo. Performance-focused columnar operations over
Arrow-style memory: typed value buffers plus validity bitmaps (one bit per
row, LSB-first within a byte, Arrow's layout). No dependency on arrow-go —
raw slices and []byte bitmaps, so arrow users pass their buffers directly
and everyone else gets a columnar toolkit for free.

Ships on the per-operation dispatch (v1.14.0): consumers link only the
columnar ops they call.

## Surface, v1 (target: v1.15.0)

1. `CompressBitsInto[T Number](dst, src []T, validity []byte) int` — filter
   by bitmap, the columnar filter primitive. New kernels
   `simd_compress_bits_{f32,f64,i32,i64,u32,u64,u8,u16}` in csrc/columnar.c:
   the compress.c idiom with MASK_FROM replaced by a direct bit-load from
   the bitmap (AVX-512 k-registers come straight from the bitmap word —
   cheaper than the byte-mask form). COMPRESS_LANES per block,
   `__builtin_masked_compress_store`, popcount of the bit window advances k.
2. `SumValid[T](a []T, validity []byte) T` — null-aware sum. MUST be
   bit-identical to ref across tiers: ref mirrors ref.Sum's fixed
   16-accumulator tree with zeros substituted at invalid lanes;
   the kernel uses the same tree with masked lanes (zero-substitution
   preserves the tree shape, so the bits match by construction).
3. `CountValid(validity []byte, off, n int) int` — popcount over a bit
   range. Pure Go over the existing PopCount kernel + edge masking; no new
   kernel.
4. `Take` = existing `GatherInto[T](dst, src, idx []int32)` — document the
   mapping in the guide; no new code.
5. Bitmap algebra (and/or/andnot over []byte) — check csrc/sets.c first
   (grep came back empty for simd_and/or; likely NOT present → small new
   kernels or defer to v2; Go's per-word loop may be enough — measure
   before writing kernels, entry-48 discipline).
6. `MinMaxValid` — defer to v2 unless v1 lands fast.

## Wiring checklist (per kernels.md, now 7 steps)

csrc/columnar.c → manifest entries in tools/simdgen/kernels/kernels.go
(Group "Bytes"? NO — typed: numeric groups per element type for
compress_bits/sum_valid; UnclampedDst: true everywhere, validity is in
different units than values, the recorded shear lesson) → ref
implementations in internal/ref (mirror Sum's tree for SumValid!) → public
wrappers columnar.go → tests (differential vs ref across tiers +
bit-identity for SumValid + Arrow-layout fixtures incl. offset edge cases)
→ benchmarks (vs the caller's alternative: scalar loop with bit test; and
vs CompressInto+bool-expansion) → guide section + README index rows (the
README examples gate will demand runnable examples).

## State at last save

- v1.14.0 (per-op dispatch) + simdjson swap: released, pushed, done.
- csrc/columnar.c WRITTEN and syntax-clean on x86-64-v4: compress_bits_
  {i32,i64,f32,f64} (CB_LANES=16, rvv 8; bit-window mask via lanebit
  compare; masked_compress_store; scalar tail) and sum_valid_
  {f32,f64,i32,i64} (SUM_LANES=16 matching reduce.c's tree EXACTLY; select
  not multiply -- Arrow nulls hold garbage, garbage*0 is NaN when the
  garbage is; tail per REDUCE_SUM shape; halving tree).
- Guards need RefWhen "len(bm) < (len(src)+7)/8"-style (bitmap shorter than
  values => ref) and UnclampedDst: true (bitmap in different units!).
- DONE since: manifest constructors compressBitsK/sumValidK added,
  instantiated for F32/F64/I32/I64 in a NEW Columnar() source registered as
  {Path: "csrc/columnar.c"} in kernels.All (NOT inside Compress -- astcheck
  checks per-source C definitions). tools build green.
- NEXT (in order): (1) kernel.Ops[T] gains fields `CompressBits func(dst,
  src []T, bm []byte) int` and `SumValid func(a []T, bm []byte) T` in
  internal/kernel/kernel.go with doc comments. (2) ref: exported
  CompressBits{F32,F64,I32,I64} (plain bit-test loop) and SumValid{...}
  (= ref Sum over a masked COPY -- bit-identity by construction; allocation
  fine in ref) + wire into ref.go's floatOps/intOps constructors or as
  direct Set fields -- NOTE Ops fields for numerics are built by generic
  constructors floatOps/satOps/intOps in ref; add per-type assignment after
  construction in ref.Set() for these four types only. (3) regen amd64,
  chase compile: dispatch tables emit the new ops automatically (numeric
  partial Ops vars). (4) public wrappers in new columnar.go:
  CompressBitsInto[T]/SumValid[T] via ops[T]() -- numerics go through the
  Ops overlay, NOT flat tables, so NO wrapper-table naming needed; plus
  CountValid pure Go. (5) README index rows + ExampleCompressBitsInto/
  ExampleSumValid/ExampleCountValid (gate demands examples). (6) tests:
  differential tiers × NaN-under-null × n%8/n%16 edges × empty/all/none;
  conformance picks up via Sets automatically. (7) full regen six arches,
  make verify, lanes, size probe unchanged-for-nonusers check. (8) guide
  section + CHANGELOG v1.15.0; release ritual; then bench vs scalar loop
  and vs arrow-go in simdjson-style bench later.
- Old "Manifest entries NOT yet written" plan superseded by the above. Next: kernels.go entries (numeric
  groups F32/F64/I32/I64, Fields CompressBits / SumValid; CArgs
  out(),base(dst),base(src),base(bm),lenOf(src) for compress;
  out(),base(a),base(bm),lenOf(a) for sum), SkipOn for compress on targets
  without compress instruction (mirror compress.c's SkipOn list -- check
  the existing compress manifest entry and COPY its SkipOn + Threshold).
- Then: ref impls (internal/ref) -- SumValid MUST be ref.Sum over a masked
  copy (bit-identity by construction); CompressBits a plain bit-test loop.
  Wrappers in new columnar.go (CompressBitsInto[T], SumValid[T],
  CountValid in pure Go via PopCount + edge masks). Tests: differential
  across tiers incl. NaN-under-null fixtures, offset/tail edges (n%8,
  n%16), empty, all-null, none-null. Bench vs scalar bit-test loop.
  README index rows + runnable examples (the gate demands them), guide
  section. Release as v1.15.0 after full verify + lanes.

## State update (post-wrappers)
DONE: kernel.Ops fields; ref columnar.go (clamped, maskedCopy+sumFloat/sumInt);
ref.Set() wires the four types via setBase(); manifest SliceU8 fix; amd64 regen
green; public columnar.go (CompressBitsInto/SumValid/CountValid, ColumnarElem
constraint); tests+examples GREEN incl NaN-under-null + bit-identity; README
rows + 480 exported count. Tasks #207-#212 hold the user's new kernel list.
REMAINING: full six-arch regen (running/next) -> coverage-table numbers from
check-emission (kernel total changes!) -> make verify -> qemu lanes -> CHANGELOG
v1.15.0 + release ritual -> bench columnar vs scalar loop (bench file not yet
written; do before release or note as follow-up) -> guide section.
