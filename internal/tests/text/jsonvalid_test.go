package text

import (
	"encoding/json"
	"math/rand"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

// JSONValid's contract has two halves. Against encoding/json.Valid it must
// agree on every document that is not deep enough to spill (stdlib's own
// depth limit is far above the spill used here). Against the reference it
// must agree bit for bit everywhere, including which of 0 and -1 a document
// that is both too deep and malformed reports -- the per-block order of
// checks is part of the contract.

func checkValidAgree(t *testing.T, b []byte) {
	t.Helper()
	stk := make([]uint64, 256)
	got := simd.JSONValid(b, stk)
	refGot := ref.JSONValid(b, stk)
	if got != refGot {
		t.Fatalf("kernel %d, reference %d on %.120q (len %d)", got, refGot, b, len(b))
	}
	if got == -1 {
		return
	}
	want := 0
	if json.Valid(b) {
		want = 1
	}
	if got != want {
		t.Fatalf("JSONValid %d, encoding/json %v on %.120q (len %d)", got, want == 1, b, len(b))
	}
}

func TestJSONValidHandcrafted(t *testing.T) {
	pad := strings.Repeat(" ", 61)
	cases := []string{
		// The plain shapes.
		`{}`, `[]`, `""`, `0`, `-0.5e+10`, `true`, `false`, `null`,
		`{"a":1}`, `[1,2,3]`, `{"a":{"b":[1,{"c":null}]}}`,
		` [ 1 , "two" , { "three" : false } ] `,
		// Malformed.
		``, ` `, `{`, `}`, `[`, `]`, `{]`, `[}`, `{"a"}`, `{"a":}`,
		`{"a":1,}`, `[1,]`, `[,1]`, `{,}`, `01`, `1.`, `1e`, `-`, `+1`,
		`tru`, `trux`, `nul`, `falsy`, `"a`, `"a" "b"`, `1 2`, `{} []`,
		`true"`, `[1"]"`, `{"a":1}}`, `[[]]]`,
		// Strings: escapes, good and bad.
		`"\n\t\r\b\f\/\\\""`, `"\q"`, `"A"`, `"뻯"", `,
		`"\u00"`, `"\uZZZZ"`, `"\u12g4"`, `"ab\`, `"\`,
		// Control bytes inside and outside strings.
		"\"a\x01b\"", "\"a\tb\"", "[1,\n2]", "\x1f", "[\x00]",
		// Whitespace-only tails and heads crossing the block edge.
		pad + `1`, `1` + pad, pad + `[1]` + pad,
		// A number spanning blocks: the fast-forward's skip path.
		`[` + strings.Repeat("1", 200) + `]`,
		`[` + strings.Repeat("1", 200) + `.5e2]`,
		strings.Repeat("9", 130),
		strings.Repeat("9", 130) + `x`,
		// A long string spanning blocks, with escapes near the edges.
		`"` + strings.Repeat("a", 60) + `\n` + strings.Repeat("b", 60) + `"`,
		`"` + strings.Repeat("a", 62) + `A` + strings.Repeat("b", 60) + `"`,
		`"` + strings.Repeat("a", 63) + `\u00` + `"`,
		// Literals straddling a block edge.
		`[` + pad + `true]`, `[` + pad + `false]`, `[` + pad + `null]`,
		// An unterminated string whose tail is only quiet bytes.
		`"` + strings.Repeat("a", 100),
		// Deep-but-fine nesting (under one spill word).
		strings.Repeat("[", 60) + strings.Repeat("]", 60),
		strings.Repeat("[", 60) + `1` + strings.Repeat("]", 60),
	}
	for _, c := range cases {
		checkValidAgree(t, []byte(c))
	}
	// Escapes at every offset relative to the block edge, so the leader and
	// target land on each side of it at least once.
	for off := 50; off < 80; off++ {
		doc := `"` + strings.Repeat("a", off) + `B` + `"`
		checkValidAgree(t, []byte(doc))
		bad := `"` + strings.Repeat("a", off) + `\u004G` + `"`
		checkValidAgree(t, []byte(bad))
	}
	// Numbers ending at every offset around the block edge exercise both
	// arms of the fast-forward.
	for l := 56; l < 72; l++ {
		checkValidAgree(t, []byte(`[`+strings.Repeat("7", l)+`,1]`))
		checkValidAgree(t, []byte(strings.Repeat("7", l)))
	}
}

func TestJSONValidSpill(t *testing.T) {
	deep := func(depth int) []byte {
		return []byte(strings.Repeat("[", depth) + "1" + strings.Repeat("]", depth))
	}
	for _, tc := range []struct {
		depth, stk int
		want       int
	}{
		{64, 0, 1},    // fits without a spill word
		{65, 0, -1},   // the first push finds no room
		{128, 1, 1},   // one word is enough
		{129, 1, -1},  // and then it is not
		{1000, 64, 1}, // plenty
	} {
		b := deep(tc.depth)
		var stk []uint64
		if tc.stk > 0 {
			stk = make([]uint64, tc.stk)
		}
		got := simd.JSONValid(b, stk)
		refGot := ref.JSONValid(b, stk)
		if got != refGot {
			t.Fatalf("depth %d stk %d: kernel %d, reference %d", tc.depth, tc.stk, got, refGot)
		}
		if got != tc.want {
			t.Fatalf("depth %d stk %d: got %d, want %d", tc.depth, tc.stk, got, tc.want)
		}
	}
	// Deep and malformed: the reference's check order decides 0 vs -1 and
	// the kernel must match it exactly.
	for _, doc := range []string{
		strings.Repeat("[", 200) + "\x01",
		strings.Repeat("[", 200) + `"\q"`,
		"\"\x01\"" + strings.Repeat("[", 200),
	} {
		b := []byte(doc)
		got := simd.JSONValid(b, nil)
		refGot := ref.JSONValid(b, nil)
		if got != refGot {
			t.Fatalf("kernel %d, reference %d on %.60q", got, refGot, doc)
		}
	}
}

func TestJSONValidRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	// Random valid documents via the stdlib encoder, then mutations.
	var build func(depth int) any
	build = func(depth int) any {
		switch k := rng.Intn(7); {
		case k == 0 && depth < 4:
			n := rng.Intn(4)
			m := map[string]any{}
			for i := 0; i < n; i++ {
				m[randKey(rng)] = build(depth + 1)
			}
			return m
		case k == 1 && depth < 4:
			n := rng.Intn(5)
			s := make([]any, n)
			for i := range s {
				s[i] = build(depth + 1)
			}
			return s
		case k == 2:
			return randKey(rng)
		case k == 3:
			return rng.NormFloat64() * 1e6
		case k == 4:
			return rng.Intn(2) == 0
		default:
			return nil
		}
	}
	for i := 0; i < 400; i++ {
		enc, err := json.Marshal(build(0))
		if err != nil {
			t.Fatal(err)
		}
		checkValidAgree(t, enc)
		// A handful of single-byte corruptions of each.
		for j := 0; j < 4 && len(enc) > 0; j++ {
			mut := append([]byte(nil), enc...)
			mut[rng.Intn(len(mut))] = byte(rng.Intn(256))
			checkValidAgree(t, mut)
		}
	}
	// Random garbage outright.
	for i := 0; i < 200; i++ {
		b := make([]byte, rng.Intn(300))
		rng.Read(b)
		checkValidAgree(t, b)
	}
}

func randKey(rng *rand.Rand) string {
	const alpha = "abcdefghij \\\"\té世"
	n := rng.Intn(20)
	var sb strings.Builder
	for i := 0; i < n; i++ {
		r := []rune(alpha)
		sb.WriteRune(r[rng.Intn(len(r))])
	}
	return sb.String()
}
