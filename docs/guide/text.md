# Text and bytes

Searching, splitting, trimming and parsing. This is the half of the library
that competes with the standard library rather than with a loop you wrote, and
that is a harder comparison — `bytes` and `strings` are already hand-written
assembly on four of the six architectures here.

## String or []byte, without copying

Every function in this section is generic over both:

```go
simd.Index("the quick brown fox", "brown") // 10
simd.Index([]byte("a,b,c"), []byte(","))   // 1
```

No `[]byte(s)` conversion, no allocation, no separate `...String` variants.
Mixing the two in one call works too — the haystack and the needle are
independent type parameters.

## Finding things

```go
simd.IndexByte("a,b,c", ',')      // 1
simd.LastIndex("a,b,c", ",")      // 3
simd.IndexAny("key=value;x", "=;") // 3 — first byte from a set
simd.IndexNotAny("   text", " ")   // 3 — first byte NOT from a set
simd.CountAny("a,b;c,d", ",;")     // 3
```

`IndexAny` is the one worth knowing about. A tokenizer that looks for "the next
delimiter, whichever it is" would otherwise call `IndexByte` once per delimiter
and take the minimum, reading the input once per delimiter. This reads it once.

### Finding every occurrence at once

```go
idx := make([]int32, len(line))
n := simd.IndexAll(idx, line, ',')
// idx[:n] holds every comma position
```

This is the structural-index step that fast JSON and CSV parsers are built on.
Instead of scanning, stopping, processing, and resuming — which defeats the
vector unit every time it stops — you find all the boundaries in one pass and
then work from the index.

It runs at about 4 GB/s. The scan is not where the time goes in a parser, which
is exactly why separating it from the conversion is worth doing.

## Parsing a CSV of integers

The two-step form is around five times faster than `strconv.Atoi` per field:

```go
line := "10,20,30,40"

idx := make([]int32, len(line)+1)
n := simd.IndexAll(idx, line, ',')

// The last field is not separator-terminated, so it needs a sentinel at the
// end of the input. Without it the final value is silently dropped.
idx[n] = int32(len(line))

dst := make([]int64, n+1)
count, ok := simd.ParseInts(dst, line, idx[:n+1])
// dst[:count] is [10 20 30 40], ok is true
```

That sentinel is the one thing people get wrong, so it is worth stating plainly:
`ParseInts` converts the field ending at each offset you give it. Three commas
describe three fields; the fourth field ends at the end of the line, and you
have to say so.

`ok` is false and `count` is the index of the offending field if a field is not
a valid integer — empty, containing a non-digit after an optional sign, or
naming a value outside `int64`. An over-long field is rejected rather than
wrapped, so you can report where the input went wrong rather than silently
storing nonsense.

The split between scanning and converting is deliberate. On 200,000 short
fields, `IndexAll` alone runs at 4.06 GB/s and the same scan followed by
`strconv.Atoi` at 0.83 — so the scan is a fifth of the work and the conversion
is the other four fifths. Keeping them apart lets you reuse a scan, and keeps
the accelerated part to the half that was actually slow.

## Trimming, folding, validating

```go
simd.TrimSpaceASCII("  hello\t\n")           // "hello"
simd.TrimAny("xxhelloxx", "x")               // "hello"
simd.EqualFoldASCII("Content-Type", "content-type") // true
simd.ValidUTF8("héllo")                      // true
```

`EqualFoldASCII` folds only ASCII letters, which is what makes it safe to run
over UTF-8 rather than merely over ASCII: every continuation byte is 0x80 or
above and falls outside both the upper and lower ranges, so it passes through
untouched. It is not a Unicode-aware fold and does not claim to be — if you
need one, `strings.EqualFold` is correct and slower.

`ValidUTF8` is about 54% faster than `utf8.Valid` on a megabyte.

## Hex and base64

```go
dst := make([]byte, 8)
n := simd.HexEncode(dst, []byte{0xde, 0xad, 0xbe, 0xef})
// string(dst[:n]) is "deadbeef"

out := make([]byte, simd.Base64DecodedLen(len(src)))
n = simd.Base64Decode(out, src)
```

Base64 is 42% to 63% faster than `encoding/base64`. The `...Len` helpers exist
so you can size the destination exactly rather than guessing.

## Comparing and sharing prefixes

`Compare` matches `bytes.Compare` — the ordering. `CommonPrefixLen` answers the
other question, which the standard library has no function for:

```go
simd.CommonPrefixLen("/usr/local/bin", "/usr/local/lib") // 11
```

This is the inner loop of suffix-array construction, trie descent and the LCP
array, where the strings being compared are usually nearly identical. A
byte-at-a-time loop pays a compare and a branch for every byte the two share;
this reduces sixty-four bytes at a time to a single "did anything differ", and
only walks the block that did. About 21× the naive loop on a megabyte with a
long shared prefix.

It counts bytes, not runes, so the answer can land in the middle of a UTF-8
sequence. Backing up to a boundary is a constant-time step from any byte index
if you need one.

## Counting bits

```go
simd.HammingDistance(a, b) // number of differing bits
simd.PopCount(a)           // number of set bits
```

`HammingDistance` is the fused `popcount(a xor b)` — one pass over both slices
rather than an xor pass producing a temporary and then a popcount pass over it.
For similarity search over binary fingerprints that is the whole workload.

Note the name. `Hamming` on its own is the *window function*, from signal
processing, and the two are unrelated. The collision is why one of them is
spelled out.

## Where this does not win

Where the standard library is already assembly doing the same work, there is no
margin and this library says so rather than pretending otherwise. `bytes.Equal`
is `memequal`; `bytealg.Count` already popcounts a compare mask. `Equal` and
`Count` here are competitive, not faster.

The wins are concentrated in the operations `bytealg` does not have: `IndexAll`,
`IndexAny` over a set, `CommonPrefixLen`, `ParseInts`, and `LastIndex` on long
inputs — where the measured figure is +8309% at 4 KiB, because the standard
library's backward search is not vectorized at all.
