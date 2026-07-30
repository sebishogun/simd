package simd_test

import (
	"strings"
	"testing"
	"unicode/utf8"

	simd "github.com/sebishogun/simd"
)

// truncCase shortens a case for an error message. text_test.go already has a
// trunc, so this one is named for what it is used on.
func truncCase(s string) string {
	if len(s) > 30 {
		return s[:30] + "..."
	}
	return s
}

func TestAppendRunes(t *testing.T) {
	cases := []string{
		"", "a", "hello world",
		strings.Repeat("ascii text that is long enough to hit the vector path ", 40),
		"héllo wörld",
		"日本語のテキスト",
		"mixed ascii 日本語 and back to a long ascii run " + strings.Repeat("x", 200),
		"\xff\xfe invalid bytes \x80",
		strings.Repeat("é", 100),
		"a\xffb",
	}
	for _, s := range cases {
		want := []rune(s)
		got := simd.AppendRunes(nil, s)
		if len(got) != len(want) {
			t.Fatalf("%q: got %d runes, want %d", truncCase(s), len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%q: rune %d = %q, want %q", truncCase(s), i, got[i], want[i])
			}
		}
		if n := simd.RuneCount(s); n != utf8.RuneCountInString(s) {
			t.Errorf("%q: RuneCount = %d, want %d", truncCase(s), n, utf8.RuneCountInString(s))
		}
	}
}

// The append convention: it must extend rather than replace, and reuse the
// buffer's capacity.
func TestAppendRunesExtends(t *testing.T) {
	dst := []rune("abc")
	dst = simd.AppendRunes(dst, "def")
	if string(dst) != "abcdef" {
		t.Errorf("got %q, want abcdef", string(dst))
	}

	buf := make([]rune, 0, 1024)
	for i := 0; i < 10; i++ {
		buf = simd.AppendRunes(buf[:0], "reused buffer, no allocation")
	}
	if cap(buf) != 1024 {
		t.Errorf("capacity changed to %d; the buffer was not reused", cap(buf))
	}
}

func TestAppendRunesNoAlloc(t *testing.T) {
	s := strings.Repeat("some ascii text ", 200)
	buf := make([]rune, 0, len(s))
	if n := testing.AllocsPerRun(50, func() { buf = simd.AppendRunes(buf[:0], s) }); n != 0 {
		t.Errorf("AppendRunes allocated %v times per run into a large enough buffer", n)
	}
}
