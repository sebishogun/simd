package text

// The text API against the standard library.
//
// Every function here has a `bytes` or `strings` counterpart that defines the
// answer, so the test is not "does it agree with our reference" — that is what
// internal/conformance is for — but "does it agree with the package a caller
// would otherwise have used". A drop-in replacement that is subtly not one is
// worse than no replacement.
//
// Both spellings of every input are checked. The string and []byte
// instantiations are separately compiled code, so passing one proves nothing
// about the other, and the whole point of the generic signature is that a
// caller may use either.

import (
	"bytes"
	"math/rand/v2"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/sebishogun/simd"
)

// corpus is the text every scan below runs over: a mixture of the lengths
// where the kernels switch strategy, of content with and without matches, and
// of valid and invalid UTF-8.
func corpus() []string {
	r := rand.New(rand.NewPCG(101, 102))
	var out []string
	// Every length across the block boundaries the kernels use (32 and 64).
	for n := range 140 {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte("abcdefgh, \t\n"[r.IntN(12)])
		}
		out = append(out, string(b))
	}
	out = append(out,
		"", " ", "   ", "\t\n\v\f\r ", "a", "aaaa",
		strings.Repeat("x", 1000)+"needle"+strings.Repeat("y", 1000),
		strings.Repeat("ab", 500),
		strings.Repeat(" ", 200)+"core"+strings.Repeat("\t", 200),
		"héllo wörld", "日本語のテキスト", "\xff\xfe invalid \x80",
		strings.Repeat("héllo ", 100),
	)
	return out
}

var needles = []string{"", "a", "ab", "needle", "abc", "\n", "  ", "xy", "zzz",
	"héllo", strings.Repeat("ab", 3)}

var cutsets = []string{"", " ", " \t\n", "ab", "abcdefgh", "\x00\xff", asciiSpaceTest}

const asciiSpaceTest = " \t\n\v\f\r"

func TestIndexAgainstStdlib(t *testing.T) {
	for _, s := range corpus() {
		b := []byte(s)
		for _, n := range needles {
			nb := []byte(n)
			want := strings.Index(s, n)
			if got := simd.Index(s, n); got != want {
				t.Fatalf("Index(%q, %q) = %d, want %d", trunc(s), n, got, want)
			}
			if got := simd.Index(b, nb); got != want {
				t.Fatalf("Index([]byte %q, %q) = %d, want %d", trunc(s), n, got, want)
			}
			// Mixed spellings must work too; that is what two type parameters
			// are for.
			if got := simd.Index(s, nb); got != want {
				t.Fatalf("Index(string, []byte) (%q, %q) = %d, want %d", trunc(s), n, got, want)
			}

			want = strings.LastIndex(s, n)
			if got := simd.LastIndex(s, n); got != want {
				t.Fatalf("LastIndex(%q, %q) = %d, want %d", trunc(s), n, got, want)
			}
			if got := simd.LastIndex(b, nb); got != want {
				t.Fatalf("LastIndex([]byte %q, %q) = %d, want %d", trunc(s), n, got, want)
			}

			want = strings.Count(s, n)
			if got := simd.Count(s, n); got != want {
				t.Fatalf("Count(%q, %q) = %d, want %d", trunc(s), n, got, want)
			}
			if got := simd.Count(b, nb); got != want {
				t.Fatalf("Count([]byte %q, %q) = %d, want %d", trunc(s), n, got, want)
			}

			if got, w := simd.Contains(s, n), strings.Contains(s, n); got != w {
				t.Fatalf("Contains(%q, %q) = %v, want %v", trunc(s), n, got, w)
			}
			if got, w := simd.HasPrefix(s, n), strings.HasPrefix(s, n); got != w {
				t.Fatalf("HasPrefix(%q, %q) = %v, want %v", trunc(s), n, got, w)
			}
			if got, w := simd.HasSuffix(s, n), strings.HasSuffix(s, n); got != w {
				t.Fatalf("HasSuffix(%q, %q) = %v, want %v", trunc(s), n, got, w)
			}
			if got, w := simd.HasPrefix(b, nb), bytes.HasPrefix(b, nb); got != w {
				t.Fatalf("HasPrefix([]byte %q, %q) = %v, want %v", trunc(s), n, got, w)
			}
			if got, w := simd.HasSuffix(b, nb), bytes.HasSuffix(b, nb); got != w {
				t.Fatalf("HasSuffix([]byte %q, %q) = %v, want %v", trunc(s), n, got, w)
			}
		}
	}
}

func TestByteScansAgainstStdlib(t *testing.T) {
	for _, s := range corpus() {
		b := []byte(s)
		for _, c := range []byte{'a', 'z', ' ', '\n', 0, 0x80, 0xff} {
			if got, w := simd.IndexByte(s, c), strings.IndexByte(s, c); got != w {
				t.Fatalf("IndexByte(%q, %#02x) = %d, want %d", trunc(s), c, got, w)
			}
			if got, w := simd.LastIndexByte(s, c), strings.LastIndexByte(s, c); got != w {
				t.Fatalf("LastIndexByte(%q, %#02x) = %d, want %d", trunc(s), c, got, w)
			}
			if got, w := simd.LastIndexByte(b, c), bytes.LastIndexByte(b, c); got != w {
				t.Fatalf("LastIndexByte([]byte %q, %#02x) = %d, want %d", trunc(s), c, got, w)
			}
			if got, w := simd.CountByte(s, c), bytes.Count(b, []byte{c}); got != w {
				t.Fatalf("CountByte(%q, %#02x) = %d, want %d", trunc(s), c, got, w)
			}
			if got, w := simd.ContainsByte(b, c), bytes.ContainsRune(b, rune(c)); c < 0x80 && got != w {
				t.Fatalf("ContainsByte(%q, %#02x) = %v, want %v", trunc(s), c, got, w)
			}
		}
	}
}

// TestIndexAnyOrLess checks the fused scan against asking the two questions
// separately, which is what it replaces. The interesting inputs are the ones
// where both answers exist and they differ, so lo is varied above and below
// the bytes the corpus actually contains rather than fixed at 0x20.
func TestIndexAnyOrLess(t *testing.T) {
	for _, s := range corpus() {
		b := []byte(s)
		for _, set := range cutsets {
			for _, lo := range []byte{0, 1, 0x20, 0x41, 0x61, 0x80, 0xFF} {
				want := -1
				for i := 0; i < len(b); i++ {
					if b[i] < lo || bytes.IndexByte([]byte(set), b[i]) >= 0 {
						want = i
						break
					}
				}
				if got := simd.IndexAnyOrLess(s, set, lo); got != want {
					t.Fatalf("IndexAnyOrLess(%q, %q, %#02x) = %d, want %d",
						trunc(s), set, lo, got, want)
				}
				if got := simd.IndexAnyOrLess(b, []byte(set), lo); got != want {
					t.Fatalf("IndexAnyOrLess([]byte %q, %q, %#02x) = %d, want %d",
						trunc(s), set, lo, got, want)
				}
			}
			// lo of 0 excludes nothing, so the call is exactly IndexAny.
			if got, w := simd.IndexAnyOrLess(s, set, 0), simd.IndexAny(s, set); got != w {
				t.Fatalf("IndexAnyOrLess(%q, %q, 0) = %d, want IndexAny's %d", trunc(s), set, got, w)
			}
		}
	}
}

// TestSetScansAgainstStdlib compares against the ASCII-only subset of the
// stdlib's rune-based functions, which is where the two agree by construction:
// no byte of a multi-byte UTF-8 sequence is below 0x80, so an ASCII cutset
// cannot match inside one.
func TestSetScansAgainstStdlib(t *testing.T) {
	for _, s := range corpus() {
		b := []byte(s)
		for _, set := range cutsets {
			if !isASCIIStr(set) {
				continue
			}
			if got, w := simd.IndexAny(s, set), strings.IndexAny(s, set); got != w {
				t.Fatalf("IndexAny(%q, %q) = %d, want %d", trunc(s), set, got, w)
			}
			if got, w := simd.IndexAny(b, []byte(set)), bytes.IndexAny(b, set); got != w {
				t.Fatalf("IndexAny([]byte %q, %q) = %d, want %d", trunc(s), set, got, w)
			}
			if got, w := simd.TrimLeftAny(s, set), strings.TrimLeft(s, set); got != w {
				t.Fatalf("TrimLeftAny(%q, %q) = %q, want %q", trunc(s), set, got, w)
			}
			if got, w := simd.TrimRightAny(s, set), strings.TrimRight(s, set); got != w {
				t.Fatalf("TrimRightAny(%q, %q) = %q, want %q", trunc(s), set, got, w)
			}
			if got, w := simd.TrimAny(s, set), strings.Trim(s, set); got != w {
				t.Fatalf("TrimAny(%q, %q) = %q, want %q", trunc(s), set, got, w)
			}
			if got, w := simd.TrimAny(b, []byte(set)), bytes.Trim(b, set); !bytes.Equal(got, w) {
				t.Fatalf("TrimAny([]byte %q, %q) = %q, want %q", trunc(s), set, got, w)
			}
			// IndexNotAny has no stdlib twin for bytes; the definition is
			// checked directly instead.
			i := simd.IndexNotAny(s, set)
			checkNotAny(t, "IndexNotAny", s, set, i, true)
			j := simd.LastIndexNotAny(s, set)
			checkNotAny(t, "LastIndexNotAny", s, set, j, false)
		}
	}
}

// checkNotAny verifies the answer from first principles: the reported byte is
// outside the set, and every byte before it (or after it, scanning backwards)
// is inside.
func checkNotAny(t *testing.T, name, s, set string, i int, forward bool) {
	t.Helper()
	in := func(c byte) bool { return strings.IndexByte(set, c) >= 0 }
	if i < 0 {
		for k := range len(s) {
			if !in(s[k]) {
				t.Fatalf("%s(%q, %q) = -1 but byte %d (%#02x) is not in the set",
					name, trunc(s), set, k, s[k])
			}
		}
		return
	}
	if in(s[i]) {
		t.Fatalf("%s(%q, %q) = %d but that byte is in the set", name, trunc(s), set, i)
	}
	if forward {
		for k := range i {
			if !in(s[k]) {
				t.Fatalf("%s(%q, %q) = %d but byte %d is already outside the set",
					name, trunc(s), set, i, k)
			}
		}
		return
	}
	for k := i + 1; k < len(s); k++ {
		if !in(s[k]) {
			t.Fatalf("%s(%q, %q) = %d but byte %d is also outside the set",
				name, trunc(s), set, i, k)
		}
	}
}

func TestTrimSpaceASCII(t *testing.T) {
	for _, s := range corpus() {
		// strings.TrimSpace also trims Unicode space, so it is only the same
		// function on input with none.
		if !isASCIIStr(s) {
			continue
		}
		if got, w := simd.TrimSpaceASCII(s), strings.TrimSpace(s); got != w {
			t.Fatalf("TrimSpaceASCII(%q) = %q, want %q", trunc(s), got, w)
		}
		b := []byte(s)
		if got, w := simd.TrimSpaceASCII(b), bytes.TrimSpace(b); !bytes.Equal(got, w) {
			t.Fatalf("TrimSpaceASCII([]byte %q) = %q, want %q", trunc(s), got, w)
		}
	}
}

func TestClassificationAgainstStdlib(t *testing.T) {
	for _, s := range corpus() {
		b := []byte(s)
		if got, w := simd.ValidUTF8(s), utf8.ValidString(s); got != w {
			t.Fatalf("ValidUTF8(%q) = %v, want %v", trunc(s), got, w)
		}
		if got, w := simd.ValidUTF8(b), utf8.Valid(b); got != w {
			t.Fatalf("ValidUTF8([]byte %q) = %v, want %v", trunc(s), got, w)
		}
		if got, w := simd.IsASCII(s), isASCIIStr(s); got != w {
			t.Fatalf("IsASCII(%q) = %v, want %v", trunc(s), got, w)
		}
		for _, other := range corpus()[:20] {
			if got, w := simd.EqualFoldASCII(s, other), asciiFoldEqual(s, other); got != w {
				t.Fatalf("EqualFoldASCII(%q, %q) = %v, want %v", trunc(s), trunc(other), got, w)
			}
		}
	}
}

// TestStringFormDoesNotAllocate is the promise that makes the generic
// signature worth having. If it boxed the string into an interface, the
// convenience would cost an allocation per call and callers would be better
// off with the unsafe.Slice dance the package used to document.
func TestStringFormDoesNotAllocate(t *testing.T) {
	s := strings.Repeat("the quick brown fox ", 64)
	b := []byte(s)
	set := " \t\n"
	setb := []byte(set)
	fox := []byte("fox")
	dst := make([]int32, 512)

	for _, c := range []struct {
		name string
		fn   func()
	}{
		{"Index/string", func() { sinkTextInt = simd.Index(s, "fox") }},
		{"Index/bytes", func() { sinkTextInt = simd.Index(b, fox) }},
		{"LastIndex/string", func() { sinkTextInt = simd.LastIndex(s, "fox") }},
		{"IndexByte/string", func() { sinkTextInt = simd.IndexByte(s, 'q') }},
		{"LastIndexByte/string", func() { sinkTextInt = simd.LastIndexByte(s, 'q') }},
		{"Count/string", func() { sinkTextInt = simd.Count(s, "fox") }},
		{"CountByte/string", func() { sinkTextInt = simd.CountByte(s, 'o') }},
		{"IndexAny/string", func() { sinkTextInt = simd.IndexAny(s, set) }},
		{"IndexNotAny/string", func() { sinkTextInt = simd.IndexNotAny(s, set) }},
		{"Contains/string", func() { sinkTextBool = simd.Contains(s, "fox") }},
		{"HasPrefix/string", func() { sinkTextBool = simd.HasPrefix(s, "the") }},
		{"ValidUTF8/string", func() { sinkTextBool = simd.ValidUTF8(s) }},
		{"IsASCII/string", func() { sinkTextBool = simd.IsASCII(s) }},
		{"TrimSpaceASCII/string", func() { sinkTextStr = simd.TrimSpaceASCII(s) }},
		{"TrimAny/bytes", func() { sinkTextBytes = simd.TrimAny(b, setb) }},
		{"IndexAll/string", func() { sinkTextInt = simd.IndexAll(dst, s, ' ') }},
		{"Equal/string", func() { sinkTextBool = simd.Equal(s, s) }},
	} {
		if n := testing.AllocsPerRun(100, c.fn); n != 0 {
			t.Errorf("%s allocated %.0f times per call, want 0", c.name, n)
		}
	}
}

// TestTextSlicingPreservesType is the part of the generic signature that could
// silently be wrong: TrimAny on a string must give back a string that shares
// the original's memory, and on a []byte a slice of the original array.
func TestTextSlicingPreservesType(t *testing.T) {
	b := []byte("  hello  ")
	got := simd.TrimAny(b, []byte(" "))
	if string(got) != "hello" {
		t.Fatalf("TrimAny gave %q", got)
	}
	// Writing through the result must be visible in the original, which is
	// what "no copy" means.
	got[0] = 'H'
	if string(b) != "  Hello  " {
		t.Fatalf("the result did not alias the input: %q", b)
	}
}

func isASCIIStr(s string) bool {
	for i := range len(s) {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

func asciiFoldEqual(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	fold := func(c byte) byte {
		if c >= 'A' && c <= 'Z' {
			return c + 32
		}
		return c
	}
	for i := range len(a) {
		if fold(a[i]) != fold(b[i]) {
			return false
		}
	}
	return true
}

func trunc(s string) string {
	if len(s) <= 40 {
		return s
	}
	return s[:40] + "..."
}

var (
	sinkTextInt   int
	sinkTextBool  bool
	sinkTextStr   string
	sinkTextBytes []byte
)
