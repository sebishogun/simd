package text

// The oracle is unicode/utf16 driven exactly as a caller would drive it:
// utf16.Encode([]rune(s)) one way, string(utf16.Decode(u)) the other. Equality
// is the contract, not a tolerance, including for the malformed inputs where
// both sides substitute U+FFFD.

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/sebishogun/simd"
)

func TestAppendUTF16(t *testing.T) {
	cases := []string{
		"",
		"a",
		"plain ascii",
		"é",                                 // 2-byte, 1 unit
		"世界",                                // 3-byte, 1 unit each
		"𝄞",                                 // 4-byte, SURROGATE PAIR
		"a𝄞b",                               // pair between ASCII
		"héllo wörld",                       // sparse non-ASCII
		"日本語のテキストです",                        // dense non-ASCII
		strings.Repeat("x", 1000),           // long pure-ASCII, past the run floor
		strings.Repeat("é", 500),            // long pure non-ASCII
		strings.Repeat("abcdefghij é", 200), // alternating runs
		// Runs that straddle the 64-byte floor in both directions.
		strings.Repeat("a", 63) + "é" + strings.Repeat("b", 65),
		strings.Repeat("a", 64) + "é",
		"\x80 lone continuation",      // invalid UTF-8
		"\xed\xa0\x80 cesu surrogate", // encoded surrogate, invalid in UTF-8
		"trailing truncation \xf0\x9f",
	}
	for i, in := range cases {
		t.Run(fmt.Sprintf("case=%d", i), func(t *testing.T) {
			want := utf16.Encode([]rune(in))
			got := simd.AppendUTF16(nil, in)
			if len(got) != len(want) {
				t.Fatalf("len = %d, want %d (in %q)", len(got), len(want), in)
			}
			for j := range want {
				if got[j] != want[j] {
					t.Fatalf("unit[%d] = %#04x, want %#04x (in %q)", j, got[j], want[j], in)
				}
			}
			if n := simd.UTF16Len(in); n != len(want) {
				t.Fatalf("UTF16Len = %d, want %d", n, len(want))
			}

			// And back. The round trip is only the identity when the input was
			// valid UTF-8, so compare against the standard library's own round
			// trip rather than against the input.
			wantBack := string(utf16.Decode(want))
			gotBack := string(simd.AppendUTF8(nil, got))
			if gotBack != wantBack {
				t.Fatalf("round trip:\ngot  %q\nwant %q", gotBack, wantBack)
			}
			if n := simd.UTF8Len(got); n != len(wantBack) {
				t.Fatalf("UTF8Len = %d, want %d", n, len(wantBack))
			}
		})
	}

	// Unpaired surrogates on the UTF-16 side, which cannot be produced by
	// encoding valid UTF-8 and so need their own case.
	t.Run("unpairedSurrogate", func(t *testing.T) {
		for _, u := range [][]uint16{
			{0xD800},                 // high surrogate, nothing after
			{0xDC00},                 // low surrogate alone
			{0xD800, 'a'},            // high then ASCII
			{'a', 0xDC00, 'b'},       // low in the middle
			{0xD800, 0xD800, 0xDC00}, // high, then a valid pair
		} {
			want := string(utf16.Decode(u))
			got := string(simd.AppendUTF8(nil, u))
			if got != want {
				t.Fatalf("Decode(%#v):\ngot  %q\nwant %q", u, got, want)
			}
			if n := simd.UTF8Len(u); n != len(want) {
				t.Fatalf("UTF8Len(%#v) = %d, want %d", u, n, len(want))
			}
		}
	})

	// Random text against the oracle, sizes straddling the run floor, with the
	// non-ASCII density swept so both the widen path and the scalar path are
	// the majority in some case.
	r := rand.New(rand.NewPCG(227, 229))
	for _, density := range []int{0, 1, 10, 50, 100} {
		for _, n := range []int{63, 64, 65, 300, 5000} {
			var sb strings.Builder
			for sb.Len() < n {
				if r.IntN(100) < density {
					sb.WriteRune([]rune{'é', '世', '𝄞', 0x7FF, 0xFFFF}[r.IntN(5)])
				} else {
					sb.WriteByte(byte(' ' + r.IntN(95)))
				}
			}
			in := sb.String()
			want := utf16.Encode([]rune(in))
			got := simd.AppendUTF16(nil, in)
			if len(got) != len(want) {
				t.Fatalf("density=%d n=%d: len %d, want %d", density, n, len(got), len(want))
			}
			for j := range want {
				if got[j] != want[j] {
					t.Fatalf("density=%d n=%d unit[%d]: %#04x != %#04x",
						density, n, j, got[j], want[j])
				}
			}
			if back := string(simd.AppendUTF8(nil, got)); back != in {
				t.Fatalf("density=%d n=%d: round trip differs", density, n)
			}
		}
	}

	// The append convention: an adequately sized buffer allocates nothing.
	t.Run("allocFree", func(t *testing.T) {
		in := strings.Repeat("some mostly ascii text with é in it, ", 100)
		buf := make([]uint16, 0, utf8.RuneCountInString(in)*2)
		out := make([]byte, 0, len(in)*2)
		if got := testing.AllocsPerRun(20, func() {
			buf = simd.AppendUTF16(buf[:0], in)
			out = simd.AppendUTF8(out[:0], buf)
		}); got != 0 {
			t.Errorf("allocated %.1f times with capacity in hand, want 0", got)
		}
	})
}
