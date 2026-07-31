package text

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

// AppendUTF8FromRunes must agree with string(runes) on everything, including
// the inputs that are not valid runes at all.
func TestAppendUTF8FromRunes(t *testing.T) {
	cases := [][]rune{
		nil,
		[]rune(""),
		[]rune("hello"),
		[]rune("héllo wörld"),
		[]rune("日本語テキスト"),
		[]rune("𝄞 music"),
		// Long enough to clear asciiRunFloor and use the kernel.
		[]rune(strings.Repeat("the quick brown fox ", 40)),
		// ASCII run, then not, then ASCII again — the path that alternates.
		[]rune(strings.Repeat("a", 200) + "日本" + strings.Repeat("b", 200)),
		// Not valid runes: a surrogate half and an out-of-range value. Both
		// must become RuneError, as string(runes) does.
		{0xD800, 'a', 0x110000, 'b'},
		{-1, 'x'},
	}
	for i, in := range cases {
		got := simd.AppendUTF8FromRunes(nil, in)
		want := string(in)
		if string(got) != want {
			t.Errorf("case %d: got %q want %q", i, got, want)
		}
	}

	// It appends rather than replacing, and round-trips against AppendRunes.
	prefix := []byte("keep:")
	got := simd.AppendUTF8FromRunes(append([]byte(nil), prefix...), []rune("héllo"))
	if string(got) != "keep:héllo" {
		t.Errorf("append onto a prefix gave %q", got)
	}

	text := strings.Repeat("mixed ascii and 日本語 ", 30)
	runes := simd.AppendRunes(nil, text)
	back := simd.AppendUTF8FromRunes(nil, runes)
	if string(back) != text {
		t.Error("AppendRunes then AppendUTF8FromRunes did not round-trip")
	}
}

func TestAppendUTF8FromRunesReusesBuffer(t *testing.T) {
	in := []rune(strings.Repeat("x", 4096))
	buf := make([]byte, 0, len(in))
	n := testing.AllocsPerRun(20, func() {
		buf = simd.AppendUTF8FromRunes(buf[:0], in)
	})
	if n != 0 {
		t.Errorf("appending into a buffer with capacity allocated %v times", n)
	}
}
