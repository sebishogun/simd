package text

// JSONCopyValid against the pair it replaces: a plain copy-run loop and
// utf8.Valid over exactly the bytes it copied.
//
// It answers two questions in one pass, so both have to be checked
// independently. A kernel that copies correctly and misjudges validity, or
// judges validity correctly and copies the wrong count, each pass a test that
// looks at one of them -- and the second is the dangerous one, because a false
// "valid" ships malformed UTF-8 into a document.

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/sebishogun/simd"
)

func wantCopyValid(dst, b []byte, html bool) int {
	n := 0
	for ; n < len(b); n++ {
		c := b[n]
		if c < 0x20 || c == '"' || c == '\\' {
			break
		}
		if html && (c == '<' || c == '>' || c == '&' || c == 0xE2) {
			break
		}
		dst[n] = c
	}
	if !utf8.Valid(b[:n]) {
		return -1
	}
	return n
}

func checkCopyValid(t *testing.T, name string, in []byte) {
	t.Helper()
	for _, html := range []bool{true, false} {
		want := make([]byte, len(in))
		wn := wantCopyValid(want, in, html)
		got := make([]byte, len(in))
		gn := simd.JSONCopyValid(got, in, html)

		// The kernel validates whole blocks, so it may reject on bytes past
		// its stopping point. That is allowed in one direction only: it may
		// say invalid when the reference says valid, never the reverse.
		if wn < 0 && gn >= 0 {
			t.Errorf("%s html=%v: kernel accepted %d bytes the reference calls invalid UTF-8: % x",
				name, html, gn, in)
			return
		}
		if gn < 0 {
			continue
		}
		if gn != wn {
			t.Errorf("%s html=%v: copied %d, want %d\n in % x", name, html, gn, wn, in)
			return
		}
		if !bytes.Equal(got[:gn], want[:wn]) {
			t.Errorf("%s html=%v:\n got % x\nwant % x", name, html, got[:gn], want[:wn])
			return
		}
	}
}

func TestJSONCopyValid(t *testing.T) {
	for c := 0; c < 256; c++ {
		checkCopyValid(t, "single", []byte{byte(c)})
	}
	for c := 0; c < 256; c++ {
		b := bytes.Repeat([]byte("a"), 100)
		b[50] = byte(c)
		checkCopyValid(t, "embedded", b)
	}
	cases := map[string]string{
		"empty": "", "ascii": "hello world",
		"quote": `a"b`, "backslash": `a\b`, "control": "a\x01b",
		"html": "<a>&</a>", "japanese": "名前前田あゆみ", "emoji": "🙂🙃",
		"long clean":     strings.Repeat("abcdefgh", 100),
		"long jp":        strings.Repeat("名前前田あゆみ", 40),
		"jp then escape": strings.Repeat("名前", 40) + "\"tail",
	}
	for name, s := range cases {
		checkCopyValid(t, name, []byte(s))
	}
	// Malformed UTF-8 in every shape, at every length across a block boundary.
	bad := [][]byte{
		{0x80}, {0xC2}, {0xE0, 0xA0}, {0xF0, 0x9F, 0x92},
		{0xC0, 0xAF}, {0xE0, 0x80, 0xAF}, {0xF0, 0x80, 0x80, 0xAF},
		{0xED, 0xA0, 0x80}, {0xF4, 0x90, 0x80, 0x80}, {0xFE}, {0xFF},
	}
	for _, seq := range bad {
		for n := 0; n <= 80; n++ {
			b := append(bytes.Repeat([]byte("x"), n), seq...)
			checkCopyValid(t, "bad tail", b)
			b2 := append(append([]byte{}, seq...), bytes.Repeat([]byte("x"), n)...)
			checkCopyValid(t, "bad head", b2)
		}
	}
	// A multi-byte character straddling the block boundary, which is what the
	// carried-over lookback exists for.
	for n := 20; n <= 70; n++ {
		b := append(bytes.Repeat([]byte("x"), n), []byte("名前前田")...)
		checkCopyValid(t, "straddle", b)
	}
}

func TestJSONCopyValidRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 5000; trial++ {
		b := make([]byte, rng.Intn(200))
		for i := range b {
			switch rng.Intn(5) {
			case 0:
				b[i] = byte(rng.Intn(0x20))
			case 1:
				b[i] = byte(0x20 + rng.Intn(0x60))
			case 2:
				b[i] = byte(0x80 + rng.Intn(0x40)) // continuations
			case 3:
				b[i] = byte(0xC0 + rng.Intn(0x40)) // leaders
			default:
				b[i] = byte(rng.Intn(256))
			}
		}
		checkCopyValid(t, "random", b)
	}
}

// Valid input must never be rejected. That is the direction the one-sided
// allowance above does not cover, and the one that would lose data.
func TestJSONCopyValidAcceptsValid(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	runes := []rune("aA0 名前あ🙂é\u00ff\u07ff\uffff")
	for trial := 0; trial < 5000; trial++ {
		var sb strings.Builder
		for i := 0; i < rng.Intn(80); i++ {
			sb.WriteRune(runes[rng.Intn(len(runes))])
		}
		s := []byte(sb.String())
		if !utf8.Valid(s) {
			t.Fatal("test input is not valid UTF-8")
		}
		dst := make([]byte, len(s))
		if n := simd.JSONCopyValid(dst, s, true); n < 0 {
			t.Fatalf("rejected valid UTF-8: % x", s)
		} else if !bytes.Equal(dst[:n], s[:n]) {
			t.Fatalf("copied wrong bytes:\n got % x\nwant % x", dst[:n], s[:n])
		}
	}
}
