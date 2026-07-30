package conformance

// Conformance for the three groups that are not plain arithmetic: comparisons
// writing []bool, the boolean-vector logic over those masks, and the byte and
// text kernels.
//
// They need their own length sweep. The numeric kernels work in elements, so
// sweeping 0..70 crosses several vector widths; a byte kernel works in bytes,
// where a 512-bit register holds 64 of them and the scanning kernels step in
// blocks of 64. A sweep that stops at 70 would never run a second block, so
// the interesting lengths are listed explicitly and go past 256.

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// byteLens are the lengths the byte and mask kernels are checked at: every
// length up to 70, then each boundary either side of one block, two blocks and
// four.
var byteLens = func() []int {
	var out []int
	for n := range maxLen + 1 {
		out = append(out, n)
	}
	return append(out, 127, 128, 129, 191, 192, 193, 255, 256, 257, 383, 512)
}()

// genBool produces a mask with runs in it. Independent coin flips per element
// would make All almost always false and Any almost always true past a handful
// of elements, so neither kernel's interesting case would ever be reached;
// runs make both outcomes common at every length.
func genBool(n int, r *rand.Rand) []bool {
	out := make([]bool, n)
	v := r.IntN(2) == 0
	for i := range out {
		if r.IntN(8) == 0 {
			v = !v
		}
		out[i] = v
	}
	return out
}

// genBytes draws from a small alphabet. Uniform bytes would match a given
// needle about once in 256, so at these lengths IndexByte would almost always
// be answering "not present" and the found path would go untested.
func genBytes(n int, r *rand.Rand) []byte {
	const alphabet = "aAbB{}\x00\x7f\x80\xff"
	out := make([]byte, n)
	for i := range out {
		out[i] = alphabet[r.IntN(len(alphabet))]
	}
	return out
}

// ---------- comparisons ----------

func checkCmp[T comparable](t *testing.T, tier, op string,
	got, want func(dst []bool, a, b []T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(11, 12))
	for n := range maxLen + 1 {
		a, b := gen(n, r), gen(n, r)
		// Half the elements are copied across so equality and ordering both
		// occur; two independent draws would make Equal false nearly always.
		for i := 0; i < n; i += 2 {
			b[i] = a[i]
		}
		g, w := make([]bool, n), make([]bool, n)
		got(g, a, b)
		want(w, a, b)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d: got %v want %v (a=%v b=%v)",
				tier, op, n, i, g[i], w[i], a[i], b[i])
		}
	}
}

func checkCmpScalar[T comparable](t *testing.T, tier, op string,
	got, want func(dst []bool, a []T, v T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(13, 14))
	for n := range maxLen + 1 {
		a := gen(n, r)
		v := gen(1, r)[0]
		if n > 0 {
			v = a[n/2] // guarantees at least one equal element
		}
		g, w := make([]bool, n), make([]bool, n)
		got(g, a, v)
		want(w, a, v)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d v=%v: got %v want %v (a=%v)",
				tier, op, n, i, v, g[i], w[i], a[i])
		}
	}
}

func checkSelect[T comparable](t *testing.T, tier, op string,
	got, want func(dst []T, mask []bool, yes, no []T), gen func(int, *rand.Rand) []T) {
	t.Helper()
	if got == nil || want == nil {
		return
	}
	r := rand.New(rand.NewPCG(15, 16))
	for n := range maxLen + 1 {
		m := genBool(n, r)
		yes, no := gen(n, r), gen(n, r)
		g, w := make([]T, n), make([]T, n)
		got(g, m, yes, no)
		want(w, m, yes, no)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s n=%d i=%d m=%v: got %v want %v (yes=%v no=%v)",
				tier, op, n, i, m[i], g[i], w[i], yes[i], no[i])
		}
	}
}

// ---------- boolean vectors ----------

func checkMask(t *testing.T, tier string, got, want kernel.Mask) {
	t.Helper()
	r := rand.New(rand.NewPCG(17, 18))

	reduce := func(op string, g, w func([]bool) bool) {
		if g == nil || w == nil {
			return
		}
		for _, n := range byteLens {
			m := genBool(n, r)
			if a, b := g(m), w(m); a != b {
				t.Fatalf("%s/Mask.%s n=%d: got %v want %v (m=%v)", tier, op, n, a, b, m)
			}
			// An all-true and an all-false mask are the cases All and Any exist
			// to distinguish, and a single flipped element at the very end is
			// where a kernel that drops its remainder gets caught.
			for _, fill := range []bool{true, false} {
				u := make([]bool, n)
				for i := range u {
					u[i] = fill
				}
				if a, b := g(u), w(u); a != b {
					t.Fatalf("%s/Mask.%s n=%d all-%v: got %v want %v", tier, op, n, fill, a, b)
				}
				if n > 0 {
					u[n-1] = !fill
					if a, b := g(u), w(u); a != b {
						t.Fatalf("%s/Mask.%s n=%d all-%v but last: got %v want %v",
							tier, op, n, fill, a, b)
					}
				}
			}
		}
	}
	reduce("All", got.All, want.All)
	reduce("Any", got.Any, want.Any)

	if got.Count != nil && want.Count != nil {
		for _, n := range byteLens {
			m := genBool(n, r)
			if a, b := got.Count(m), want.Count(m); a != b {
				t.Fatalf("%s/Mask.Count n=%d: got %d want %d", tier, n, a, b)
			}
		}
	}

	binary := func(op string, g, w func(dst, a, b []bool)) {
		if g == nil || w == nil {
			return
		}
		for _, n := range byteLens {
			a, b := genBool(n, r), genBool(n, r)
			gr, wr := make([]bool, n), make([]bool, n)
			g(gr, a, b)
			w(wr, a, b)
			if i, ok := same(gr, wr); !ok {
				t.Fatalf("%s/Mask.%s n=%d i=%d: got %v want %v", tier, op, n, i, gr[i], wr[i])
			}
		}
	}
	binary("And", got.And, want.And)
	binary("Or", got.Or, want.Or)
	binary("Xor", got.Xor, want.Xor)

	if got.Not != nil && want.Not != nil {
		for _, n := range byteLens {
			a := genBool(n, r)
			gr, wr := make([]bool, n), make([]bool, n)
			got.Not(gr, a)
			want.Not(wr, a)
			if i, ok := same(gr, wr); !ok {
				t.Fatalf("%s/Mask.Not n=%d i=%d: got %v want %v", tier, n, i, gr[i], wr[i])
			}
		}
	}
}

// ---------- bytes, bits and ASCII text ----------

func checkBytes(t *testing.T, tier string, got, want kernel.Bytes) {
	t.Helper()
	r := rand.New(rand.NewPCG(19, 20))

	// The needles cover a byte that occurs often, one that occurs never, and
	// the two ends of the range — a kernel comparing as signed rather than
	// unsigned gets 0x80 and 0xff wrong and nothing else.
	needles := []byte{'a', 'z', 0x00, 0x7f, 0x80, 0xff}

	query := func(op string, g, w func([]byte, byte) int) {
		if g == nil || w == nil {
			return
		}
		for _, n := range byteLens {
			b := genBytes(n, r)
			for _, c := range needles {
				if a, e := g(b, c), w(b, c); a != e {
					t.Fatalf("%s/Bytes.%s n=%d c=%#02x: got %d want %d", tier, op, n, c, a, e)
				}
			}
			// A match planted at each end catches a kernel that scans the
			// blocked body but drops the remainder, or reports any match in a
			// block rather than the first.
			if n > 0 {
				for _, at := range []int{0, n / 2, n - 1} {
					u := append([]byte(nil), b...)
					for i := range u {
						u[i] = 'q'
					}
					u[at] = 'a'
					if a, e := g(u, 'a'), w(u, 'a'); a != e {
						t.Fatalf("%s/Bytes.%s n=%d planted at %d: got %d want %d",
							tier, op, n, at, a, e)
					}
				}
			}
		}
	}
	query("IndexByte", got.IndexByte, want.IndexByte)
	query("LastIndexByte", got.LastIndexByte, want.LastIndexByte)
	query("Count", got.Count, want.Count)

	if got.PopCount != nil && want.PopCount != nil {
		for _, n := range byteLens {
			b := genBytes(n, r)
			if a, e := got.PopCount(b), want.PopCount(b); a != e {
				t.Fatalf("%s/Bytes.PopCount n=%d: got %d want %d", tier, n, a, e)
			}
		}
	}

	if got.IsASCII != nil && want.IsASCII != nil {
		for _, n := range byteLens {
			for _, b := range asciiCases(n, r) {
				if a, e := got.IsASCII(b), want.IsASCII(b); a != e {
					t.Fatalf("%s/Bytes.IsASCII n=%d: got %v want %v", tier, n, a, e)
				}
			}
		}
	}

	if got.ValidUTF8 != nil && want.ValidUTF8 != nil {
		for _, n := range byteLens {
			for _, b := range utf8Cases(n, r) {
				if g, w := got.ValidUTF8(b), want.ValidUTF8(b); g != w {
					t.Fatalf("%s/Bytes.ValidUTF8 n=%d %x: got %v want %v",
						tier, n, b, g, w)
				}
			}
		}
	}

	if got.Equal != nil && want.Equal != nil {
		for _, n := range byteLens {
			a := genBytes(n, r)
			b := append([]byte(nil), a...)
			cases := [][2][]byte{{a, b}}
			if n > 0 {
				// Differing in one byte, at each end and in the middle; and
				// differing only in length, which is the case a kernel handed
				// one length cannot see.
				for _, at := range []int{0, n / 2, n - 1} {
					c := append([]byte(nil), a...)
					c[at] ^= 0xff
					cases = append(cases, [2][]byte{a, c})
				}
				cases = append(cases, [2][]byte{a, b[:n-1]}, [2][]byte{a[:n-1], b})
			}
			for _, c := range cases {
				if x, e := got.Equal(c[0], c[1]), want.Equal(c[0], c[1]); x != e {
					t.Fatalf("%s/Bytes.Equal n=%d len=%d,%d: got %v want %v",
						tier, n, len(c[0]), len(c[1]), x, e)
				}
			}
		}
	}

	binary := func(op string, g, w func(dst, a, b []byte)) {
		if g == nil || w == nil {
			return
		}
		for _, n := range byteLens {
			a, b := genBytes(n, r), genBytes(n, r)
			gr, wr := make([]byte, n), make([]byte, n)
			g(gr, a, b)
			w(wr, a, b)
			if i, ok := same(gr, wr); !ok {
				t.Fatalf("%s/Bytes.%s n=%d i=%d: got %#02x want %#02x", tier, op, n, i, gr[i], wr[i])
			}
		}
	}
	binary("And", got.And, want.And)
	binary("Or", got.Or, want.Or)
	binary("Xor", got.Xor, want.Xor)
	binary("AndNot", got.AndNot, want.AndNot)

	unary := func(op string, g, w func(dst, b []byte)) {
		if g == nil || w == nil {
			return
		}
		for _, n := range byteLens {
			// Every byte value is covered, not just the alphabet: ASCII case
			// folding is a range test and its boundaries are '@'/'A'/'Z'/'['
			// and the same for lowercase, all of which a small alphabet misses.
			b := make([]byte, n)
			for i := range b {
				b[i] = byte(i)
			}
			gr, wr := make([]byte, n), make([]byte, n)
			g(gr, b)
			w(wr, b)
			if i, ok := same(gr, wr); !ok {
				t.Fatalf("%s/Bytes.%s n=%d i=%d in=%#02x: got %#02x want %#02x",
					tier, op, n, i, b[i], gr[i], wr[i])
			}
		}
	}
	unary("Not", got.Not, want.Not)
	unary("ToUpperASCII", got.ToUpperASCII, want.ToUpperASCII)
	unary("ToLowerASCII", got.ToLowerASCII, want.ToLowerASCII)

	if got.Fill != nil && want.Fill != nil {
		for _, n := range byteLens {
			for _, v := range needles {
				gr, wr := make([]byte, n), make([]byte, n)
				got.Fill(gr, v)
				want.Fill(wr, v)
				if i, ok := same(gr, wr); !ok {
					t.Fatalf("%s/Bytes.Fill n=%d i=%d: got %#02x want %#02x", tier, n, i, gr[i], wr[i])
				}
			}
		}
	}

	if got.Compare != nil && want.Compare != nil {
		for _, n := range byteLens {
			a := genBytes(n, r)
			// Equal, differing at each end and in the middle, and differing
			// only in length — which is the case a kernel handed one length
			// cannot see, and the one Compare's contract turns on.
			cases := [][2][]byte{{a, append([]byte(nil), a...)}}
			if n > 0 {
				for _, at := range []int{0, n / 2, n - 1} {
					for _, delta := range []int{-1, +1} {
						c := append([]byte(nil), a...)
						c[at] = byte(int(c[at]) + delta)
						cases = append(cases, [2][]byte{a, c})
					}
				}
				cases = append(cases, [2][]byte{a, a[:n-1]}, [2][]byte{a[:n-1], a})
			}
			for _, c := range cases {
				g, w := got.Compare(c[0], c[1]), want.Compare(c[0], c[1])
				if g != w {
					t.Fatalf("%s/Bytes.Compare len=%d,%d: got %d want %d",
						tier, len(c[0]), len(c[1]), g, w)
				}
			}
		}
	}

	if got.CommonPrefix != nil && want.CommonPrefix != nil {
		for _, n := range byteLens {
			a := genBytes(n, r)
			// The same cases Compare uses, for the same reason: the answer
			// turns on where the first difference is, including the case where
			// there is none and one slice simply ends.
			cases := [][2][]byte{{a, append([]byte(nil), a...)}}
			if n > 0 {
				for _, at := range []int{0, n / 2, n - 1} {
					c := append([]byte(nil), a...)
					c[at] ^= 0xff
					cases = append(cases, [2][]byte{a, c})
				}
				cases = append(cases, [2][]byte{a, a[:n-1]}, [2][]byte{a[:n-1], a})
			}
			for _, c := range cases {
				g, w := got.CommonPrefix(c[0], c[1]), want.CommonPrefix(c[0], c[1])
				if g != w {
					t.Fatalf("%s/Bytes.CommonPrefix len=%d,%d: got %d want %d",
						tier, len(c[0]), len(c[1]), g, w)
				}
			}
		}
	}

	if got.EqualFoldASCII != nil && want.EqualFoldASCII != nil {
		for _, n := range byteLens {
			a := make([]byte, n)
			for i := range a {
				a[i] = byte(i)
			}
			// The same bytes with ASCII letters case-flipped must still
			// compare equal, and a flipped byte outside those ranges must not.
			flipped := append([]byte(nil), a...)
			for i, c := range flipped {
				switch {
				case c >= 'a' && c <= 'z':
					flipped[i] = c - 32
				case c >= 'A' && c <= 'Z':
					flipped[i] = c + 32
				}
			}
			cases := [][2][]byte{{a, a}, {a, flipped}}
			if n > 0 {
				// 0x80 and 0xa0 differ by the same bit as 'A' and 'a', which
				// is exactly what a fold that forgot to range-check would get
				// wrong.
				hi := append([]byte(nil), a...)
				hi[n-1] = 0x80
				hi2 := append([]byte(nil), a...)
				hi2[n-1] = 0xa0
				cases = append(cases, [2][]byte{hi, hi2}, [2][]byte{a, a[:n-1]})
			}
			for _, c := range cases {
				g, w := got.EqualFoldASCII(c[0], c[1]), want.EqualFoldASCII(c[0], c[1])
				if g != w {
					t.Fatalf("%s/Bytes.EqualFoldASCII len=%d,%d: got %v want %v",
						tier, len(c[0]), len(c[1]), g, w)
				}
			}
		}
	}

	sets := [][]byte{{}, {'a'}, {'a', 'z'}, {0x00, 0x7f, 0x80, 0xff}, []byte("aeiou{}")}
	if got.IndexAny != nil && want.IndexAny != nil {
		for _, n := range byteLens {
			b := genBytes(n, r)
			for _, set := range sets {
				if g, w := got.IndexAny(b, set), want.IndexAny(b, set); g != w {
					t.Fatalf("%s/Bytes.IndexAny n=%d set=%q: got %d want %d",
						tier, n, set, g, w)
				}
			}
		}
	}
	if got.CountAny != nil && want.CountAny != nil {
		for _, n := range byteLens {
			b := genBytes(n, r)
			for _, set := range sets {
				if g, w := got.CountAny(b, set), want.CountAny(b, set); g != w {
					t.Fatalf("%s/Bytes.CountAny n=%d set=%q: got %d want %d",
						tier, n, set, g, w)
				}
			}
		}
	}

	// LastIndex and CountSeq answer the same questions as Index over the same
	// needles, so they share its table rather than getting one that would drift
	// away from it.
	for _, c := range []struct {
		op        string
		got, want func(h, n []byte) int
	}{
		{"LastIndex", got.LastIndex, want.LastIndex},
		{"CountSeq", got.CountSeq, want.CountSeq},
	} {
		if c.got == nil || c.want == nil {
			continue
		}
		for _, n := range byteLens {
			h := genBytes(n, r)
			needles := [][]byte{{}, {'a'}, {'z'}, {'a', 'a'}, {'a', 'b', 'a'}}
			if n > 0 {
				for _, at := range []int{0, n / 2, n - 1} {
					for _, l := range []int{1, 2, 3, 9, 33} {
						if at+l <= n {
							needles = append(needles, h[at:at+l])
						}
					}
				}
				needles = append(needles, h)
			}
			flat := make([]byte, n)
			for i := range flat {
				flat[i] = 'a'
			}
			for _, hay := range [][]byte{h, flat} {
				for _, ndl := range needles {
					if g, w := c.got(hay, ndl), c.want(hay, ndl); g != w {
						t.Fatalf("%s/Bytes.%s n=%d needle=%q: got %d want %d",
							tier, c.op, n, ndl, g, w)
					}
				}
			}
		}
	}

	// The set scans and their complements. The sets are chosen so that the
	// answer is at every position in turn: one that matches nothing, one that
	// matches everything the generator produces, and the boundary bytes a
	// signed comparison would get wrong.
	for _, c := range []struct {
		op        string
		got, want func(b, chars []byte) int
	}{
		{"IndexNotAny", got.IndexNotAny, want.IndexNotAny},
		{"LastIndexNotAny", got.LastIndexNotAny, want.LastIndexNotAny},
	} {
		if c.got == nil || c.want == nil {
			continue
		}
		for _, n := range byteLens {
			b := genBytes(n, r)
			for _, set := range [][]byte{
				{}, {'q'}, {0x00}, {0x80}, {0xff}, {0x7f, 0x80},
				[]byte("abcdefghijklmnopqrstuvwxyz"),
			} {
				if g, w := c.got(b, set), c.want(b, set); g != w {
					t.Fatalf("%s/Bytes.%s n=%d set=%q: got %d want %d",
						tier, c.op, n, set, g, w)
				}
			}
			// A slice made entirely of one byte, with the set holding exactly
			// that byte, is the case where the scan runs to the end and must
			// report -1 rather than the last index it looked at.
			flat := make([]byte, n)
			for i := range flat {
				flat[i] = 'a'
			}
			if g, w := c.got(flat, []byte("a")), c.want(flat, []byte("a")); g != w {
				t.Fatalf("%s/Bytes.%s all-'a' n=%d: got %d want %d", tier, c.op, n, g, w)
			}
		}
	}

	// A set with a byte in it twice is still one member. index_any folds with
	// OR and does not care; count_any sums and did, counting every position
	// once per duplicate. Found by the fuzzer, kept here by name.
	if got.CountAny != nil && want.CountAny != nil {
		for _, n := range byteLens {
			b := genBytes(n, r)
			for _, set := range [][]byte{
				{'a', 'a'}, {'a', 'a', 'a', 'b', 'b'},
				bytes.Repeat([]byte{0}, 64), bytes.Repeat([]byte("ab"), 40),
			} {
				if g, w := got.CountAny(b, set), want.CountAny(b, set); g != w {
					t.Fatalf("%s/Bytes.CountAny n=%d duplicated set (%d bytes): got %d want %d",
						tier, n, len(set), g, w)
				}
			}
		}
	}

	// Base64. Every length rather than a sample: the encoder's tail is one of
	// three shapes depending on len%3 and the decoder's is one of three
	// depending on the padding, and each has its own arithmetic.
	if got.B64Encode != nil && want.B64Encode != nil {
		for _, n := range byteLens {
			b := genBytes(n, r)
			enc := (n + 2) / 3 * 4
			for _, dn := range []int{0, enc - 1, enc, enc + 7} {
				if dn < 0 {
					continue
				}
				gd, wd := make([]byte, dn), make([]byte, dn)
				g, w := got.B64Encode(gd, b), want.B64Encode(wd, b)
				if g != w {
					t.Fatalf("%s/Bytes.B64Encode n=%d dst=%d: wrote %d want %d",
						tier, n, dn, g, w)
				}
				if i, ok := same(gd, wd); !ok {
					t.Fatalf("%s/Bytes.B64Encode n=%d dst=%d i=%d: got %#02x want %#02x",
						tier, n, dn, i, gd[i], wd[i])
				}
			}
			// And the decode of a well-formed encoding, plus a corrupted one
			// so the rejection path is exercised at every length too.
			buf := make([]byte, enc)
			want.B64Encode(buf, b)
			for _, mutate := range []bool{false, true} {
				in := append([]byte(nil), buf...)
				if mutate && len(in) > 0 {
					in[len(in)/2] = '!'
				}
				dn := n + 3
				gd, wd := make([]byte, dn), make([]byte, dn)
				g, w := got.B64Decode(gd, in), want.B64Decode(wd, in)
				if g != w {
					t.Fatalf("%s/Bytes.B64Decode n=%d mutated=%v: wrote %d want %d",
						tier, n, mutate, g, w)
				}
				if g > 0 {
					if i, ok := same(gd[:g], wd[:g]); !ok {
						t.Fatalf("%s/Bytes.B64Decode n=%d i=%d: got %#02x want %#02x",
							tier, n, i, gd[i], wd[i])
					}
				}
			}
		}
	}

	if got.Index != nil && want.Index != nil {
		for _, n := range byteLens {
			h := genBytes(n, r)
			needles := [][]byte{{}, {'a'}, {'z'}, {'a', 'a'}, {'a', 'b', 'a'}}
			if n > 0 {
				// Needles taken out of the haystack itself, at each end and
				// across the middle, so the found path is exercised at every
				// alignment. A search that filters on the first and last byte
				// gets a repeated needle wrong if it skips verification.
				for _, at := range []int{0, n / 2, n - 1} {
					for _, l := range []int{1, 2, 3, 9, 33} {
						if at+l <= n {
							needles = append(needles, h[at:at+l])
						}
					}
				}
				needles = append(needles, h, append(append([]byte(nil), h...), 'Q'))
			}
			for _, ndl := range needles {
				if g, w := got.Index(h, ndl), want.Index(h, ndl); g != w {
					t.Fatalf("%s/Bytes.Index n=%d needle=%q: got %d want %d",
						tier, n, ndl, g, w)
				}
			}
			// A haystack of one repeated byte is where a first/last filter
			// accepts every position and the verification does all the work.
			flat := make([]byte, n)
			for i := range flat {
				flat[i] = 'a'
			}
			for _, ndl := range [][]byte{{'a'}, {'a', 'a'}, {'a', 'b'}, {'b', 'a'}} {
				if g, w := got.Index(flat, ndl), want.Index(flat, ndl); g != w {
					t.Fatalf("%s/Bytes.Index flat n=%d needle=%q: got %d want %d",
						tier, n, ndl, g, w)
				}
			}
		}
	}

	if got.HexEncode != nil && want.HexEncode != nil {
		for _, n := range byteLens {
			b := make([]byte, n)
			for i := range b {
				b[i] = byte(i)
			}
			// The destination is deliberately sized short, exact and long, so
			// the count the kernel reports is checked rather than assumed.
			for _, dn := range []int{0, n, 2*n - 1, 2 * n, 2*n + 3} {
				if dn < 0 {
					continue
				}
				gd, wd := make([]byte, dn), make([]byte, dn)
				g, w := got.HexEncode(gd, b), want.HexEncode(wd, b)
				if g != w {
					t.Fatalf("%s/Bytes.HexEncode n=%d dst=%d: wrote %d want %d",
						tier, n, dn, g, w)
				}
				if i, ok := same(gd, wd); !ok {
					t.Fatalf("%s/Bytes.HexEncode n=%d dst=%d i=%d: got %#02x want %#02x",
						tier, n, dn, i, gd[i], wd[i])
				}
			}
		}
	}

	if got.ReplaceByte != nil && want.ReplaceByte != nil {
		for _, n := range byteLens {
			b := genBytes(n, r)
			gr, wr := make([]byte, n), make([]byte, n)
			got.ReplaceByte(gr, b, 'a', 'Z')
			want.ReplaceByte(wr, b, 'a', 'Z')
			if i, ok := same(gr, wr); !ok {
				t.Fatalf("%s/Bytes.ReplaceByte n=%d i=%d: got %#02x want %#02x",
					tier, n, i, gr[i], wr[i])
			}
		}
	}
}

// utf8Cases returns inputs that exercise both sides of the hand-off between
// the ASCII fast path and the byte-wise scan, and every rule that makes a
// naive validator wrong: overlong forms, surrogates, out-of-range starts, and
// sequences truncated by the end of the input.
func utf8Cases(n int, r *rand.Rand) [][]byte {
	out := [][]byte{genBytes(n, r)}
	ascii := make([]byte, n)
	for i := range ascii {
		ascii[i] = byte(' ' + i%64)
	}
	out = append(out, ascii)
	// Multi-byte sequences repeated to length, valid and invalid.
	seqs := [][]byte{
		{0xc3, 0xa9},             // é
		{0xe6, 0x97, 0xa5},       // 日
		{0xf0, 0x9f, 0x8e, 0x89}, // 🎉
		{0xc0, 0x80},             // overlong
		{0xed, 0xa0, 0x80},       // surrogate
		{0xf4, 0x90, 0x80, 0x80}, // past U+10FFFF
		{0xf0, 0x8f, 0xbf, 0xbf}, // overlong four-byte
		{0xe0, 0x9f, 0xbf},       // overlong three-byte
	}
	for _, s := range seqs {
		b := make([]byte, n)
		for i := range b {
			b[i] = s[i%len(s)]
		}
		out = append(out, b)
		// The same sequence at the very end, where it may be truncated —
		// the case a validator that reads ahead gets wrong.
		if n >= len(s) {
			c := append([]byte(nil), ascii...)
			copy(c[n-len(s):], s)
			out = append(out, c)
			for k := 1; k < len(s); k++ {
				d := append([]byte(nil), ascii[:n-len(s)+k]...)
				out = append(out, append(d, s[:k]...))
			}
		}
	}
	return out
}

// asciiCases returns inputs that are all-ASCII, all-ASCII but for the last
// byte, and mixed. The middle one is where a kernel that drops its remainder
// gives the wrong answer.
func asciiCases(n int, r *rand.Rand) [][]byte {
	pure := make([]byte, n)
	for i := range pure {
		pure[i] = byte(r.IntN(0x80))
	}
	out := [][]byte{pure, genBytes(n, r)}
	if n > 0 {
		last := append([]byte(nil), pure...)
		last[n-1] = 0x80
		first := append([]byte(nil), pure...)
		first[0] = 0xff
		out = append(out, last, first)
	}
	return out
}

// ---------- the suite ----------

// checkCompareOps runs the comparison and selection shapes of one Ops group.
func checkCompareOps[T comparable](t *testing.T, tier, typeName string,
	got, want kernel.Ops[T], gen func(int, *rand.Rand) []T) {

	p := func(op string) string { return typeName + "." + op }

	checkCmp(t, tier, p("EqualMask"), got.EqualMask, want.EqualMask, gen)
	checkCmp(t, tier, p("NotEqualMask"), got.NotEqualMask, want.NotEqualMask, gen)
	checkCmp(t, tier, p("LessMask"), got.LessMask, want.LessMask, gen)
	checkCmp(t, tier, p("LessEqualMask"), got.LessEqualMask, want.LessEqualMask, gen)
	checkCmp(t, tier, p("GreaterMask"), got.GreaterMask, want.GreaterMask, gen)
	checkCmp(t, tier, p("GreaterEqualMask"), got.GreaterEqualMask, want.GreaterEqualMask, gen)

	checkCmpScalar(t, tier, p("EqualScalarMask"), got.EqualScalarMask, want.EqualScalarMask, gen)
	checkCmpScalar(t, tier, p("NotEqualScalarMask"), got.NotEqualScalarMask, want.NotEqualScalarMask, gen)
	checkCmpScalar(t, tier, p("LessScalarMask"), got.LessScalarMask, want.LessScalarMask, gen)
	checkCmpScalar(t, tier, p("LessEqualScalarMask"), got.LessEqualScalarMask, want.LessEqualScalarMask, gen)
	checkCmpScalar(t, tier, p("GreaterScalarMask"), got.GreaterScalarMask, want.GreaterScalarMask, gen)
	checkCmpScalar(t, tier, p("GreaterEqualScalarMask"), got.GreaterEqualScalarMask, want.GreaterEqualScalarMask, gen)

	checkSelect(t, tier, p("Select"), got.Select, want.Select, gen)
}

// TestComparisonsMasksAndBytes extends the headline conformance check to the
// three groups that are not plain arithmetic.
func TestComparisonsMasksAndBytes(t *testing.T) {
	want := ref.Set()
	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			checkCompareOps(t, tier, "F32", got.F32, want.F32, genF32)
			checkCompareOps(t, tier, "F64", got.F64, want.F64, genF64)
			checkCompareOps(t, tier, "I32", got.I32, want.I32, genI32)
			checkCompareOps(t, tier, "I64", got.I64, want.I64, genI64)
			checkMask(t, tier, got.Mask, want.Mask)
			checkBytes(t, tier, got.Bytes, want.Bytes)
		})
	}
}
