package text

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

// The oracle is encoding/json with HTML escaping off, which is exactly the
// dialect AppendEscapeJSON documents. Round-tripping through json.Unmarshal
// additionally proves the output is a valid JSON string body, not merely equal
// to another implementation's.
func jsonOracle(t *testing.T, s string) string {
	t.Helper()
	var sb strings.Builder
	enc := json.NewEncoder(&sb)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s); err != nil {
		t.Fatal(err)
	}
	q := strings.TrimSuffix(sb.String(), "\n")
	return q[1 : len(q)-1] // strip the quotes; we escape the body only
}

func TestAppendEscapeJSON(t *testing.T) {
	cases := []string{
		"",
		"plain ascii, nothing to do",
		`quote " and backslash \`,
		"newline\nand\ttab\rand\x08\x0c",
		"\x00\x01\x1f control soup \x7f", // 0x7f is NOT escaped, as in json
		"unicode passes: héllo — 世界",
		strings.Repeat("x", 5000),    // long clean run, one copy
		strings.Repeat("a\"b", 2000), // escape every third byte
		"trailing escape\n",
		"\"leading",
	}
	for i, in := range cases {
		t.Run(fmt.Sprintf("case=%d", i), func(t *testing.T) {
			got := string(simd.AppendEscapeJSON(nil, in))
			want := jsonOracle(t, in)
			if got != want {
				t.Fatalf("in %q:\ngot  %q\nwant %q", in, got, want)
			}
			// And the output must decode back to the input.
			var back string
			if err := json.Unmarshal([]byte(`"`+got+`"`), &back); err != nil {
				t.Fatalf("output does not parse as JSON: %v", err)
			}
			if back != in {
				t.Fatalf("round trip: got %q, want %q", back, in)
			}
			if needs, want := simd.NeedsEscapeJSON(in), got != in; needs != want {
				t.Fatalf("NeedsEscapeJSON = %v, want %v", needs, want)
			}
		})
	}

	// Random bytes, all values 0..255, against the oracle — this is where an
	// off-by-one in the escape set shows up. Sizes straddle IndexAny's
	// dispatch threshold so the accelerated scan is what runs.
	r := rand.New(rand.NewPCG(193, 197))
	for _, n := range []int{63, 64, 65, 300, 4096} {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte(r.IntN(256))
		}
		// Valid UTF-8 not required by the escaper, but the oracle
		// (encoding/json) replaces invalid sequences — so restrict to ASCII
		// here, where the two must agree byte for byte.
		for i := range b {
			b[i] &= 0x7f
		}
		in := string(b)
		got := string(simd.AppendEscapeJSON(nil, in))
		want := jsonOracle(t, in)
		if got != want {
			t.Fatalf("n=%d: escaped form differs from encoding/json", n)
		}
	}

	// The append convention: capacity is reused, nothing is allocated when the
	// buffer is big enough.
	t.Run("allocFree", func(t *testing.T) {
		in := strings.Repeat("clean text with one \n escape ", 100)
		buf := make([]byte, 0, 2*len(in))
		if got := testing.AllocsPerRun(20, func() {
			buf = simd.AppendEscapeJSON(buf[:0], in)
		}); got != 0 {
			t.Errorf("allocated %.1f times with sufficient capacity, want 0", got)
		}
	})
}
