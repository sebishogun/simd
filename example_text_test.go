package simd_test

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// Text and byte scanning. Every one of these takes a string or a []byte
// without copying, which is what the Text constraint is for.

func ExampleIndex() {
	fmt.Println(simd.Index("the quick brown fox", "brown"))
	// Output: 10
}

func ExampleIndexByte() {
	fmt.Println(simd.IndexByte("a,b,c", ','))
	// Output: 1
}

func ExampleLastIndex() {
	fmt.Println(simd.LastIndex("a,b,c", ","))
	// Output: 3
}

// IndexAny finds the first byte belonging to a set, which is the shape a
// tokenizer wants.
func ExampleIndexAny() {
	fmt.Println(simd.IndexAny("key=value;next", "=;"))
	// Output: 3
}

// IndexAnyOrLess adds a threshold to that set, which is what an encoder wants:
// the next byte that has to be rewritten is either punctuation it cares about
// or any control character at all.
func ExampleIndexAnyOrLess() {
	fmt.Println(simd.IndexAnyOrLess("plain text\tthen a tab", `"\\`, 0x20))
	// Output: 10
}

// JSONCopyRun copies and scans in the same pass, which is what an encoder
// wants: the bytes it can write verbatim, written, and the position of the
// first one it cannot.
func ExampleJSONCopyRun() {
	src := `plain then "a quote`
	dst := make([]byte, len(src))
	n := simd.JSONCopyRun(dst, src, true)
	fmt.Printf("%d %q\n", n, dst[:n])
	// Output: 11 "plain then "
}

func ExampleIndexNotAny() {
	fmt.Println(simd.IndexNotAny("   indented", " "))
	// Output: 3
}

func ExampleCountAny() {
	fmt.Println(simd.CountAny("a,b;c,d", ",;"))
	// Output: 3
}

func ExampleTrimAny() {
	fmt.Printf("%q\n", simd.TrimAny("xxhelloxx", "x"))
	// Output: "hello"
}

func ExampleTrimSpaceASCII() {
	fmt.Printf("%q\n", simd.TrimSpaceASCII("  hello\t\n"))
	// Output: "hello"
}

// Only ASCII letters are folded, which is what makes it safe to run over
// UTF-8: every continuation byte is 0x80 or above and outside both ranges.
func ExampleEqualFoldASCII() {
	fmt.Println(simd.EqualFoldASCII("Content-Type", "content-type"))
	// Output: true
}

func ExampleValidUTF8() {
	fmt.Println(simd.ValidUTF8("héllo"), simd.ValidUTF8([]byte{0xff, 0xfe}))
	// Output: true false
}

func ExampleHexEncode() {
	dst := make([]byte, 8)
	n := simd.HexEncode(dst, []byte{0xde, 0xad, 0xbe, 0xef})
	fmt.Println(string(dst[:n]))
	// Output: deadbeef
}

func ExampleBase64Decode() {
	dst := make([]byte, simd.Base64DecodedLen(len("aGVsbG8=")))
	n := simd.Base64Decode(dst, "aGVsbG8=")
	fmt.Println(string(dst[:n]))
	// Output: hello
}

// The two-step CSV integer parse: IndexAll finds every separator in one pass,
// then ParseInts converts the fields between them.
func ExampleParseInts() {
	line := "10,20,30,40"

	idx := make([]int32, len(line)+1)
	n := simd.IndexAll(idx, line, ',')
	// The last field is not separator-terminated, so it needs a sentinel at
	// the end of the input. Without it the final value is silently dropped.
	idx[n] = int32(len(line))

	dst := make([]int64, n+1)
	got, ok := simd.ParseInts(dst, line, idx[:n+1])
	fmt.Println(dst[:got], ok)
	// Output: [10 20 30 40] true
}

// HammingDistance is the number of differing bits, in one pass over both
// slices rather than an xor pass and a popcount pass.
func ExampleHammingDistance() {
	fmt.Println(simd.HammingDistance([]byte{0b1111_0000}, []byte{0b1010_0000}))
	// Output: 2
}

// CommonPrefixLen is Compare without the ordering: how far two slices agree.
// It is the inner loop of suffix-array construction and trie descent.
func ExampleCommonPrefixLen() {
	fmt.Println(simd.CommonPrefixLen("/usr/local/bin", "/usr/local/lib"))
	// Output: 11
}

// IndexAllAny finds several delimiters in one pass. A parser wanting six of
// them at once would otherwise read the input six times.
func ExampleIndexAllAny() {
	doc := []byte(`{"a":[1,2]}`)

	pos := make([]int32, len(doc))
	n := simd.IndexAllAny(pos, doc, "{}[]:,")

	fmt.Println(pos[:n])
	// Output: [0 4 5 7 9 10]
}

// MaskBits answers the same question as IndexAll in the form a bitwise parser
// wants: a bit per byte rather than a list of offsets.
func ExampleMaskBits() {
	doc := []byte(`{"a":1}`)

	mask := make([]byte, simd.MaskLen(len(doc)))
	simd.MaskBits(mask, doc, '"')

	// Bit i of mask[i/8] describes doc[i], least-significant bit first.
	// Quotes at 1 and 3, so bits 1 and 3, printed most-significant first.
	fmt.Printf("%08b\n", mask[0])
	// Output: 00001010
}

// MaskBitsAny is MaskBits for a set of up to eight bytes.
//
// The mask is an eighth the size of the input however many bytes match, which
// is what makes it the cheaper form on input where matches are common — and it
// is the form to want when the next question is itself bitwise.
func ExampleMaskBitsAny() {
	doc := []byte(`{"a":[1,2]}`)

	mask := make([]byte, simd.MaskLen(len(doc)))
	simd.MaskBitsAny(mask, doc, "{}[]:,")

	var at []int
	for i := range doc {
		if mask[i/8]&(1<<(i%8)) != 0 {
			at = append(at, i)
		}
	}
	fmt.Println(at)
	// Output: [0 4 5 7 9 10]
}

// MaskBitsLess covers the range tests a set of eight cannot express.
func ExampleMaskBitsLess() {
	// A JSON string may not contain a raw control character.
	s := []byte("ok\tno")

	mask := make([]byte, simd.MaskLen(len(s)))
	simd.MaskBitsLess(mask, s, 0x20)

	fmt.Println(mask[0] != 0)
	// Output: true
}

// JSONMasks classifies a document five ways in one pass, which is the first
// stage of a two-stage JSON parser: find every structural byte before looking
// at any of them individually.
func ExampleJSONMasks() {
	doc := []byte(`{"a": "b\"c"}`)

	// Five regions of one byte per eight input bytes, in a fixed order.
	stride := simd.MaskWords(len(doc))
	masks := make([]byte, 5*stride)
	simd.JSONMasks(masks, doc, simd.JSONMaskAll)

	names := []string{"quote", "escape", "structural", "control", "space"}
	for i, name := range names {
		region := masks[i*stride : (i+1)*stride]
		// Which byte positions the mask marks.
		var at []int
		for p := 0; p < len(doc); p++ {
			if region[p/8]>>(p%8)&1 != 0 {
				at = append(at, p)
			}
		}
		fmt.Printf("%-10s %v\n", name, at)
	}
	// Output:
	// quote      [1 3 6 9 11]
	// escape     [8]
	// structural [0 12]
	// control    []
	// space      [5]
}

// MaskWords sizes a JSONMasks region: enough whole 64-bit words to hold a bit
// per input byte. The masks are read as words, so a region that stopped at the
// last byte would let the final word of one run into the next.
func ExampleMaskWords() {
	for _, n := range []int{0, 1, 64, 65, 100, 1000} {
		fmt.Printf("%d bytes -> %d per region, %d for all five\n",
			n, simd.MaskWords(n), 5*simd.MaskWords(n))
	}
	// Output:
	// 0 bytes -> 0 per region, 0 for all five
	// 1 bytes -> 8 per region, 40 for all five
	// 64 bytes -> 8 per region, 40 for all five
	// 65 bytes -> 16 per region, 80 for all five
	// 100 bytes -> 16 per region, 80 for all five
	// 1000 bytes -> 128 per region, 640 for all five
}
