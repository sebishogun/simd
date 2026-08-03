package text

// JSONCopyRun against a plain loop. It both copies and reports, so both have to
// be checked: a kernel that returns the right count and writes the wrong bytes
// past it would pass a test that only looked at the number.

import (
	"bytes"
	"testing"

	"github.com/sebishogun/simd"
)

func wantCopyRun(dst, b []byte, html bool) int {
	for i := 0; i < len(b); i++ {
		c := b[i]
		if c < 0x20 || c == '"' || c == '\\' {
			return i
		}
		if html && (c == '<' || c == '>' || c == '&' || c == 0xE2) {
			return i
		}
		dst[i] = c
	}
	return len(b)
}

func TestJSONCopyRun(t *testing.T) {
	// Every byte value, so the ones above ASCII are covered: they must be
	// copied, not treated as stopping points.
	all := make([]byte, 256)
	for i := range all {
		all[i] = byte(i)
	}
	corpus := [][]byte{
		nil, []byte(""), []byte("plain text"),
		[]byte("with \"a quote"), []byte(`back\slash`),
		[]byte("html <tag> & more"), []byte("nul\x00here"),
		[]byte("日本語のテキストです"), []byte("dash — and ellipsis …"),
		[]byte("sep   here"), all,
		bytes.Repeat([]byte("clean"), 200),
		append(bytes.Repeat([]byte("clean"), 60), '"'),
		append(bytes.Repeat([]byte("日本語"), 60), '<'),
	}
	for _, base := range corpus {
		for n := 0; n <= len(base); n++ {
			in := base[:n]
			for _, html := range []bool{true, false} {
				got := make([]byte, len(in)+8)
				want := make([]byte, len(in)+8)
				for i := range got {
					got[i], want[i] = 0xAA, 0xAA
				}
				gn := simd.JSONCopyRun(got, in, html)
				wn := wantCopyRun(want, in, html)
				if gn != wn {
					t.Fatalf("html=%v %q: returned %d, want %d", html, in, gn, wn)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("html=%v %q: copied %x, want %x", html, in, got, want)
				}
			}
		}
	}
}
