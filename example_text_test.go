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
