package text

// JSONQuote against a plain loop written from encoding/json's rules rather than
// from the kernel, so agreement means something.
//
// The kernel writes escapes rather than stopping at them, which makes it the
// only text kernel here whose output length differs from its input length. Both
// the bytes and the count have to be checked: a kernel returning the right
// count while writing the wrong bytes, or writing correct bytes and a wrong
// count, both pass a test that looks at one of them.

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/sebishogun/simd"
)

// wantQuote is encoding/json's escaping rules, written out.
func wantQuote(dst, b []byte, html bool) int {
	const hex = "0123456789abcdef"
	o := 0
	put := func(bs ...byte) {
		for _, c := range bs {
			dst[o] = c
			o++
		}
	}
	for i := 0; i < len(b); {
		c := b[i]
		switch {
		case c == '"':
			put('\\', '"')
			i++
		case c == '\\':
			put('\\', '\\')
			i++
		case c == '\b':
			put('\\', 'b')
			i++
		case c == '\f':
			put('\\', 'f')
			i++
		case c == '\n':
			put('\\', 'n')
			i++
		case c == '\r':
			put('\\', 'r')
			i++
		case c == '\t':
			put('\\', 't')
			i++
		case c < 0x20:
			put('\\', 'u', '0', '0', hex[c>>4], hex[c&0xF])
			i++
		case html && (c == '<' || c == '>' || c == '&'):
			put('\\', 'u', '0', '0', hex[c>>4], hex[c&0xF])
			i++
		case html && c == 0xE2 && i+2 < len(b) && b[i+1] == 0x80 &&
			(b[i+2] == 0xA8 || b[i+2] == 0xA9):
			last := byte('8')
			if b[i+2] == 0xA9 {
				last = '9'
			}
			put('\\', 'u', '2', '0', '2', last)
			i += 3
		default:
			put(c)
			i++
		}
	}
	return o
}

func checkQuote(t *testing.T, name string, in []byte) {
	t.Helper()
	for _, html := range []bool{true, false} {
		want := make([]byte, 6*len(in))
		wn := wantQuote(want, in, html)
		got := make([]byte, 6*len(in))
		gn := simd.JSONQuote(got, in, html)
		if gn != wn {
			t.Errorf("%s html=%v: wrote %d bytes, want %d\n got %q\nwant %q",
				name, html, gn, wn, got[:min(gn, len(got))], want[:wn])
			return
		}
		if !bytes.Equal(got[:gn], want[:wn]) {
			t.Errorf("%s html=%v:\n got %q\nwant %q", name, html, got[:gn], want[:wn])
			return
		}
	}
}

func TestJSONQuote(t *testing.T) {
	// Every byte value on its own, so nothing above ASCII is treated as a
	// stopping point and every control character gets its own escape.
	for c := 0; c < 256; c++ {
		checkQuote(t, "single", []byte{byte(c)})
	}
	// Every byte value inside a run, at a length that crosses a vector block.
	for c := 0; c < 256; c++ {
		b := bytes.Repeat([]byte("a"), 100)
		b[50] = byte(c)
		checkQuote(t, "embedded", b)
	}
	cases := map[string]string{
		"empty":         "",
		"plain":         "hello world",
		"quote":         `he said "hi"`,
		"backslash":     `a\b`,
		"newline":       "line\nbreak",
		"tab":           "a\tb",
		"all shorthand": "\"\\\b\f\n\r\t",
		"controls":      "\x00\x01\x1f",
		"html":          "<script>&</script>",
		"japanese":      "名前前田あゆみ",
		"emoji":         "🙂🙃",
		// The two that must become \u2028 and \u2029, and the neighbours that
		// must not: same lead byte, different followers.
		"U+2028":       "a\u2028b",
		"U+2029":       "a\u2029b",
		"U+2027":       "a\u2027b",
		"U+202A":       "a\u202ab",
		"E2 alone":     "\xe2",
		"E2 80 alone":  "\xe2\x80",
		"E2 at end":    "abc\xe2",
		"E2 80 at end": "abc\xe2\x80",
		// Long enough to cross several vector blocks with escapes between.
		"long mixed": strings.Repeat("abcdefgh\"ijklmnop\n", 40),
		"long clean": strings.Repeat("abcdefgh", 100),
		"long jp":    strings.Repeat("名前前田あゆみ", 40),
	}
	for name, s := range cases {
		checkQuote(t, name, []byte(s))
	}
	// Every length across the vector block boundary, with an escape at every
	// position in it -- where a block-at-a-time kernel gets its offsets wrong.
	for n := 1; n <= 80; n++ {
		for pos := 0; pos < n; pos++ {
			b := bytes.Repeat([]byte("x"), n)
			b[pos] = '\n'
			checkQuote(t, "boundary", b)
		}
	}
}

// Random bytes, including invalid UTF-8. The kernel is documented as copying
// bytes above ASCII through, so it must not behave differently on a sequence
// that happens to be malformed -- the caller is responsible for validity.
func TestJSONQuoteRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 3000; trial++ {
		b := make([]byte, rng.Intn(200))
		for i := range b {
			switch rng.Intn(4) {
			case 0:
				b[i] = byte(rng.Intn(0x20)) // controls
			case 1:
				b[i] = byte(0x20 + rng.Intn(0x60)) // printable ASCII
			case 2:
				b[i] = []byte{'"', '\\', '<', '>', '&', 0xE2, 0x80, 0xA8, 0xA9}[rng.Intn(9)]
			default:
				b[i] = byte(rng.Intn(256))
			}
		}
		checkQuote(t, "random", b)
	}
}

// And the output has to be what encoding/json would have written, for input
// that is valid UTF-8 -- which is the contract the caller depends on.
func TestJSONQuoteMatchesStdlib(t *testing.T) {
	inputs := []string{
		"", "plain", `with "quotes"`, "with\nnewline", "<html>&</html>",
		"日本語のテキスト", "🙂 emoji", "a\u2028b", "a\u2029b",
		strings.Repeat("mixed \"text\" with\nescapes ", 20),
	}
	for _, s := range inputs {
		if !utf8.ValidString(s) {
			t.Fatalf("test input is not valid UTF-8: %q", s)
		}
		want, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		dst := make([]byte, 6*len(s))
		n := simd.JSONQuote(dst, []byte(s), true)
		got := `"` + string(dst[:n]) + `"`
		if got != string(want) {
			t.Errorf("%q:\n ours %s\n  std %s", s, got, want)
		}
	}
}
