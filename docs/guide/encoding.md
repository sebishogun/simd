# Encodings

Quantization, narrow float formats, and the column-store compression pipeline.
What an inference runtime and a Parquet reader spend their time in.

## Quantizing to int8

The operation an inference runtime applies to every weight and every
activation. Values become 8-bit integers with a scale and a zero point, and
`q = clamp(round(x/scale) + zeroPoint, -128, 127)`.

```go
dst := make([]int8, len(weights))
simd.QuantizeInt8(dst, weights, scale, zeroPoint)
```

Rounding is half **to even**, not half away from zero. That is not a detail:
ONNX, PyTorch and TFLite all specify round-half-to-even, and a symmetric scale
produces exact `.5` values in quantity rather than rarely. Getting it wrong
gives you a model that is subtly, consistently off and agrees with nobody.

The clamp happens before the integer conversion, in wider arithmetic, because
converting an out-of-range float to an integer type in Go is
implementation-defined rather than saturating.

### Per channel

One scale for a whole tensor loses accuracy when channels have very different
magnitudes, which they usually do. Per-channel quantization gives each output
channel its own scale:

```go
// Two channels of two values each.
a := []float32{1, 2, 30, 40}
scale := []float32{0.5, 10}
zero := []int32{0, 0}

dst := make([]int8, len(a))
simd.QuantizePerChannelInt8(dst, a, scale, zero, 2, 2)
// dst is [2 4 3 4]
```

The last two arguments are the channel count and the inner length — how many
values share each scale. This is the layout weights are stored in, so no
transposition is needed.

## Multiplying quantized tensors

```go
acc := make([]int32, m*n)
simd.QMatMulInt8Into(acc, a, b, m, k, n)

out := make([]int8, m*n)
simd.RequantizeInt8Into(out, acc, scale, zeroPoint)
```

The accumulator is int32 because the product of two int8 values overflows int8
immediately, and a sum of them overflows int16 not long after.

Requantizing is a separate call rather than fused, for two reasons that are
about real models rather than about convenience: a real layer adds a bias to
the int32 accumulator before scaling, and the scale is usually per output
channel. Fusing would make both of those impossible.

`RequantizeInt8Into` computes `round(acc*scale) + zeroPoint`, half to even
again, saturating rather than wrapping:

```go
simd.RequantizeInt8Into(dst, []int32{100, 200, 300, 100000}, 0.5, 0)
// [50 100 127 127]  — the last one saturates
```

One implementation note worth knowing if you compare against other libraries:
the accumulator here is a plain int32 multiply-add, not a widening
multiply-add such as x86's `VPMADDUBSW`. That instruction pairs adjacent
products and saturates its intermediate, which changes the answer for inputs a
caller may legitimately pass. This library takes the slower instruction to keep
the arithmetic exact.

## Narrow float formats

```go
dst := make([]float32, len(src))
simd.Float16ToFloat32Into(dst, src)
simd.BFloat16ToFloat32Into(dst, src)
```

These are storage formats — half the bytes, converted on the way in and out.
Rounding to bfloat16 is to nearest even rather than truncating, which is one
instruction more expensive and unbiased; a bias applied to every weight in a
network is a drift, not a rounding error.

FP8 comes in two shapes and the difference matters:

```go
dst := make([]byte, len(src))
simd.Float32ToFloat8E4M3Into(dst, src) // more mantissa, range to 448
simd.Float32ToFloat8E5M2Into(dst, src) // more exponent, range to 57344
```

`e4m3` trades exponent range for mantissa precision, which is the trade
inference weights want, and it has **no infinity** — the encoding that would be
infinity is used for a finite value instead, so the maximum is 448. `e5m2` is
IEEE-shaped and does have infinities. Round-tripping a value that overflows
`e4m3` gives you NaN, not `+Inf`.

## The column-store pipeline

Three operations that are meant to be used together, in this order.

**Delta first.** Successive differences turn a slowly-varying column into small
numbers:

```go
deltas := make([]int32, len(col)-1)
simd.DiffInto(deltas, col)
```

**Zigzag second**, if the deltas can be negative. It maps small negatives to
small unsigned values — `-1` becomes 1, `1` becomes 2 — so a two's-complement
`-1` stops being `0xFFFFFFFF`:

```go
zz := make([]uint32, len(deltas))
simd.ZigzagEncodeInt32Into(zz, deltas)
```

**Pack last**, to whatever width now actually fits:

```go
// Eight 3-bit values need 24 bits. Size for the READER, which needs one word
// more: a value whose bits straddle a word boundary reads the next word, and
// the last one straddles whenever the total is not a multiple of 32.
packed := make([]uint32, (len(zz)*3+31)/32 + 1)
simd.BitPackInto(packed, zz, 3)

back := make([]uint32, len(zz))
simd.BitUnpackInto(back, packed, 3)
```

That extra word is the thing to get right. `BitPackInto` will happily write
into a tightly-sized slice, and `BitUnpackInto` will then silently do nothing
because its guard refuses — no panic, just zeros. Size for the reader.

In v1.20, complete 32-value blocks at widths 1 through 31 automatically use a
width-specialized unpack kernel. The width switch happens once outside the
block loop, so each generated case contains literal shifts instead of the
runtime shift count that prevented vectorization. The API and packed format did
not change; width 32 and the tail keep the general path.

This is the representation Parquet, Arrow and Lucene use for an integer column,
and the three steps are separate because real formats mix and match them.

## Run-length

```go
a := []int32{7, 7, 7, 9, 9, 4}

values := make([]int32, len(a))
lengths := make([]int32, len(a))
n := simd.RunLengthEncodeInt32(values, lengths, a, make([]bool, len(a)))
// values[:n] is [7 9 4], lengths[:n] is [3 2 1]
```

The scratch `[]bool` is where the run boundaries go, and handing it in is what
keeps this allocation-free. `RunStartsInto` gives you just that mask if you
want to drive the compaction yourself — it is the vectorizable half; walking
the boundaries to emit pairs is not.

Decode takes the values and lengths directly and stops when the destination is
full:

```go
dst := make([]int32, 6)
n = simd.RunLengthDecodeInt32(dst, values[:n], lengths[:n])
// dst[:n] is [7 7 7 9 9 4]
```

Expansion is kernel-backed: each run is one broadcast followed by wide stores.
The dependency is one output-position update per run, not one per element.

## Varints

This one is honest about doing less than you might expect, and the reason is
worth understanding because it generalises.

Writing a varint stream is **serial**. Where value *i* lands depends on the
width of every value before it, which is a genuine loop-carried dependency
through the output cursor. No rewriting removes it.

What vectorizes is the question you ask *first*: how wide is each value.

```go
a := []uint64{1, 300, 70000}

simd.VarintSize(a) // 6 — total encoded bytes, one vectorized pass

lens := make([]int32, len(a))
simd.VarintLenInto(lens, a) // [1 2 3] — per-value widths
```

That is worth having on its own, twice over. Knowing the exact encoded size
before writing a byte lets an encoder allocate once instead of growing — and
growing a byte slice by append means copying everything written so far,
repeatedly, which on a large column costs more than the encoding does. And the
widths, prefix-summed, give every value its offset, at which point the writes
are independent and can be split across goroutines.

`AppendVarints` is the encoder built on that:

```go
buf := simd.AppendVarints(nil, []uint64{1, 300})
// [0x01 0xac 0x02]
```

It sizes the buffer exactly, grows at most once, and then runs the serial emit
loop it has to. About 1.7× an append-and-grow encoder — all of it from the
allocation, none from the emit.

Decode reports both output and input progress:

```go
encoded := []byte{0x01, 0xac, 0x02, 0x80} // final value is incomplete
decoded := make([]uint64, 4)

n, consumed := simd.VarintDecode(decoded, encoded)
// decoded[:n] is [1 300], consumed is 3

encoded = encoded[consumed:] // retain the incomplete suffix for the next chunk
```

It stops before a truncated value or one that does not terminate within ten
bytes. Complete values remain usable, and `consumed` is the safe resume point.
The fast path loads one eight-byte word per value where a byte loop branches on
every encoded byte; the stream remains serial through each value's width.

The widths are computed as a sum of four unsigned comparisons rather than from
a leading-zero count. Both are correct; only one vectorizes everywhere.
`bits.Len` lowers to an intrinsic that has no vector form on SSE2 or AVX2,
where LLVM scalarizes it lane by lane and hands back something slower than the
loop it replaced.

## Bulk hashing for numeric keys

`HashUint64` applies a seeded splitmix64 finalizer to a whole key column:

```go
keys := []uint64{42, 99, 1234}
hashes := make([]uint64, len(keys))
simd.HashUint64(hashes, keys, 7)
```

This is the shape bloom filters, hash partitioning, and dictionary probes need:
many independent integer keys and one seed. It is not a replacement for
`hash/maphash` when hashing one string; maphash's AES path is built for that
case. `dst` and `keys` should have the same length.

## Byte planes and bitshuffle

Two-plane layouts convert without building structs:

```go
lo := []byte{1, 2, 3}
hi := []byte{10, 20, 30}
interleaved := make([]byte, 2*len(lo))
simd.Interleave2(interleaved, lo, hi)

backLo := make([]byte, len(lo))
backHi := make([]byte, len(hi))
simd.Deinterleave2(backLo, backHi, interleaved)
```

`Transpose8x8Bytes` transposes independent 64-byte byte matrices. `Bitshuffle`
goes one level lower: for each 64-byte tile it gathers equal-significance bits
into planes. Mostly small values then produce long zero runs for the compressor
that follows:

```go
src := make([]byte, 64) // one complete tile
shuffled := make([]byte, len(src))
simd.Bitshuffle(shuffled, src)

// Compress shuffled here, then decompress before the inverse.
back := make([]byte, len(src))
simd.Unbitshuffle(back, shuffled)
```

For `Transpose8x8Bytes`, `Bitshuffle`, and `Unbitshuffle`, source and
destination lengths must match and be a multiple of 64. They do not compress by
themselves; they reshape bytes so a general-purpose compressor sees simpler
planes.

## Colour

```go
gray := make([]byte, len(r))
simd.GrayscaleInto(gray, r, g, b)

u := make([]byte, len(r))
v := make([]byte, len(r))
simd.RGBToUVInto(u, v, r, g, b)
```

The weights are libjpeg's BT.601 constants in Q16 — 19595, 38470, 7471 — so
the output agrees with every other implementation of the same conversion to the
bit. An earlier version used Q8 weights and put full green at 149 instead of
150, which is the kind of off-by-one that survives a long time in an image
pipeline.

Luma and chroma are separate calls because the fused version needed seven
arguments and the amd64 ABI passes six in registers. Splitting it was cheaper
than spilling.
