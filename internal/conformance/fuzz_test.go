package conformance

// Differential fuzzing over the whole kernel set.
//
// The tables in the other files here choose their own inputs, which means
// they test the cases someone thought of. This one lets the fuzzer choose the
// bits, which is how the cases nobody thought of turn up — and for
// floating-point kernels that matters more than usual, because the
// interesting inputs are not "large" or "small" but specific bit patterns:
// signalling NaNs, denormals a hair above zero, the exact boundary where an
// exponent overflows.
//
// Every generated kernel is compared against the portable implementation on
// the same input. That is the same property the table tests check, so a
// failure here is a real disagreement rather than a fuzzer artefact, and the
// crasher the fuzzer writes out is a complete reproducer.
//
// Transcendentals are compared to their documented ULP bound rather than bit
// for bit, for the reason rule 6 gives.

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/sebishogun/simd/internal/backend"
	"github.com/sebishogun/simd/internal/cpu"
	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// slicesFrom reinterprets the fuzzer's bytes as each element type.
//
// Reinterpreting rather than converting is the point: it lets the fuzzer
// reach every bit pattern a float can hold, including the ones no arithmetic
// would produce, which is where the disagreements live.
func slicesFrom(b []byte) (f32 []float32, f64 []float64, i32 []int32, i64 []int64) {
	n8 := len(b) / 8
	n4 := len(b) / 4
	f64 = make([]float64, n8)
	i64 = make([]int64, n8)
	for i := range f64 {
		u := binary.LittleEndian.Uint64(b[i*8:])
		f64[i] = math.Float64frombits(u)
		i64[i] = int64(u)
	}
	f32 = make([]float32, n4)
	i32 = make([]int32, n4)
	for i := range f32 {
		u := binary.LittleEndian.Uint32(b[i*4:])
		f32[i] = math.Float32frombits(u)
		i32[i] = int32(u)
	}
	return
}

// narrowFrom reinterprets the fuzzer's bytes as the six element types added
// after the original four.
//
// Reinterpreting rather than generating is the whole point: whatever the
// fuzzer produced is a bit pattern, and for the integer types every bit
// pattern is a legal value, so nothing is wasted and the extremes of each
// range come up as often as the fuzzer decides to emit 0x7f or 0x80.
func narrowFrom(b []byte) (i8 []int8, u8 []uint8, i16 []int16, u16 []uint16,
	u32 []uint32, u64 []uint64) {

	i8 = make([]int8, len(b))
	u8 = make([]uint8, len(b))
	for i, v := range b {
		i8[i], u8[i] = int8(v), v
	}
	n2, n4, n8 := len(b)/2, len(b)/4, len(b)/8
	i16, u16 = make([]int16, n2), make([]uint16, n2)
	for i := range i16 {
		v := binary.LittleEndian.Uint16(b[i*2:])
		i16[i], u16[i] = int16(v), v
	}
	u32 = make([]uint32, n4)
	for i := range u32 {
		u32[i] = binary.LittleEndian.Uint32(b[i*4:])
	}
	u64 = make([]uint64, n8)
	for i := range u64 {
		u64[i] = binary.LittleEndian.Uint64(b[i*8:])
	}
	return
}

// fuzzOps runs every elementwise and reduction kernel of one group against the
// reference. It is generic so the same body covers every element type.
func fuzzOps[T comparable](t *testing.T, tier, name string,
	got, want kernel.Ops[T], a, b []T) {

	n := len(a)
	g, w := make([]T, n), make([]T, n)

	binaryOps := []struct {
		op        string
		got, want func(dst, x, y []T)
	}{
		{"Add", got.Add, want.Add},
		{"Sub", got.Sub, want.Sub},
		{"Mul", got.Mul, want.Mul},
		{"Minimum", got.Minimum, want.Minimum},
		{"Maximum", got.Maximum, want.Maximum},
		{"SatAdd", got.SatAdd, want.SatAdd},
		{"SatSub", got.SatSub, want.SatSub},
	}
	for _, c := range binaryOps {
		if c.got == nil || c.want == nil {
			continue
		}
		c.got(g, a, b)
		c.want(w, a, b)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s.%s at %d: got %v want %v (a=%v b=%v)",
				tier, name, c.op, i, g[i], w[i], a[i], b[i])
		}
	}

	unary := []struct {
		op        string
		got, want func(dst, x []T)
	}{
		{"Abs", got.Abs, want.Abs},
		{"Neg", got.Neg, want.Neg},
		{"CumSum", got.CumSum, want.CumSum},
		{"CumMin", got.CumMin, want.CumMin},
		{"CumMax", got.CumMax, want.CumMax},
	}
	for _, c := range unary {
		if c.got == nil || c.want == nil {
			continue
		}
		c.got(g, a)
		c.want(w, a)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s.%s at %d: got %v want %v (a=%v)",
				tier, name, c.op, i, g[i], w[i], a[i])
		}
	}

	reduce := []struct {
		op        string
		got, want func(x []T) T
	}{
		{"Sum", got.Sum, want.Sum},
		{"SumSquares", got.SumSquares, want.SumSquares},
		{"Min", got.Min, want.Min},
		{"Max", got.Max, want.Max},
		{"L1Norm", got.L1Norm, want.L1Norm},
	}
	for _, c := range reduce {
		if c.got == nil || c.want == nil {
			continue
		}
		if gv, wv := c.got(a), c.want(a); !sameScalar(gv, wv) {
			t.Fatalf("%s/%s.%s: got %v want %v", tier, name, c.op, gv, wv)
		}
	}

	if got.Dot != nil && want.Dot != nil {
		if gv, wv := got.Dot(a, b), want.Dot(a, b); !sameScalar(gv, wv) {
			t.Fatalf("%s/%s.Dot: got %v want %v", tier, name, gv, wv)
		}
	}

	// Comparisons write one bool per element, and the float ones are where
	// NaN makes NotEqual not the negation of Equal.
	gm, wm := make([]bool, n), make([]bool, n)
	cmps := []struct {
		op        string
		got, want func(dst []bool, x, y []T)
	}{
		{"EqualMask", got.EqualMask, want.EqualMask},
		{"NotEqualMask", got.NotEqualMask, want.NotEqualMask},
		{"LessMask", got.LessMask, want.LessMask},
		{"GreaterEqualMask", got.GreaterEqualMask, want.GreaterEqualMask},
	}
	for _, c := range cmps {
		if c.got == nil || c.want == nil {
			continue
		}
		c.got(gm, a, b)
		c.want(wm, a, b)
		if i, ok := same(gm, wm); !ok {
			t.Fatalf("%s/%s.%s at %d: got %v want %v (a=%v b=%v)",
				tier, name, c.op, i, gm[i], wm[i], a[i], b[i])
		}
	}

	if got.Select != nil && want.Select != nil {
		got.Select(g, gm, a, b)
		want.Select(w, gm, a, b)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/%s.Select at %d: got %v want %v", tier, name, i, g[i], w[i])
		}
	}
}

func FuzzKernelsAgainstReference(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, []byte{8, 7, 6, 5, 4, 3, 2, 1})
	// Seeds chosen for the bit patterns rather than the values: quiet and
	// signalling NaNs, both infinities, the smallest denormal, and both zeros.
	f.Add(
		[]byte{0, 0, 0, 0, 0, 0, 0xf8, 0x7f, 1, 0, 0, 0, 0, 0, 0xf0, 0x7f},
		[]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80},
	)
	f.Add(make([]byte, 256), make([]byte, 256))

	want := ref.Set()
	// The same intersection tiers() computes: compiled in *and* executable
	// here. A tier the CPU cannot run is a SIGILL, not a finding.
	runnable := map[string]bool{}
	for _, tr := range cpu.Detail().Available {
		runnable[tr.String()] = true
	}
	all := map[string]kernel.Set{}
	for _, n := range backend.Tiers() {
		if s, ok := backend.Lookup(n); ok && runnable[n] {
			all[n] = s
		}
	}

	f.Fuzz(func(t *testing.T, ab, bb []byte) {
		n := min(len(ab), len(bb))
		if n < 8 {
			return
		}
		// Cap the length: the fuzzer gains nothing from a longer slice once
		// every code path — whole blocks, the remainder, the threshold — has
		// been crossed, and a short body keeps its throughput up.
		if n > 512 {
			n = 512
		}
		a32, a64, ai32, ai64 := slicesFrom(ab[:n])
		b32, b64, bi32, bi64 := slicesFrom(bb[:n])

		for tier, got := range all {
			fuzzOps(t, tier, "F32", got.F32, want.F32, a32, b32)
			fuzzOps(t, tier, "F64", got.F64, want.F64, a64, b64)
			fuzzOps(t, tier, "I32", got.I32, want.I32, ai32, bi32)
			fuzzOps(t, tier, "I64", got.I64, want.I64, ai64, bi64)

			aI8, aU8, aI16, aU16, aU32, aU64 := narrowFrom(ab[:n])
			bI8, bU8, bI16, bU16, bU32, bU64 := narrowFrom(bb[:n])
			fuzzOps(t, tier, "I8", got.I8, want.I8, aI8, bI8)
			fuzzOps(t, tier, "U8", got.U8, want.U8, aU8, bU8)
			fuzzOps(t, tier, "I16", got.I16, want.I16, aI16, bI16)
			fuzzOps(t, tier, "U16", got.U16, want.U16, aU16, bU16)
			fuzzOps(t, tier, "U32", got.U32, want.U32, aU32, bU32)
			fuzzOps(t, tier, "U64", got.U64, want.U64, aU64, bU64)

			fuzzBytes(t, tier, got.Bytes, want.Bytes, ab[:n], bb[:n])
			fuzzTextPairs(t, tier, got.Bytes, want.Bytes, ab[:n], bb[:n])
		}
	})
}

// fuzzTextPairs compares the kernels that take two byte slices, with the
// fuzzer choosing both. Substring search and set scanning are where an
// off-by-one at a block boundary hides, and the fuzzer reaches lengths and
// contents the tables do not.
func fuzzTextPairs(t *testing.T, tier string, got, want kernel.Bytes, a, b []byte) {
	// The needle is taken from the haystack as well as used as given, so the
	// found path is exercised rather than only the rejecting one.
	needles := [][]byte{b}
	if len(a) > 3 {
		needles = append(needles, a[:3], a[len(a)-3:], a[len(a)/2:len(a)/2+2])
	}
	for _, ndl := range needles {
		for _, c := range []struct {
			op        string
			got, want func(x, y []byte) int
		}{
			{"Index", got.Index, want.Index},
			{"LastIndex", got.LastIndex, want.LastIndex},
			{"CountSeq", got.CountSeq, want.CountSeq},
			{"IndexAny", got.IndexAny, want.IndexAny},
			{"CountAny", got.CountAny, want.CountAny},
			{"IndexNotAny", got.IndexNotAny, want.IndexNotAny},
			{"LastIndexNotAny", got.LastIndexNotAny, want.LastIndexNotAny},
		} {
			if c.got == nil || c.want == nil {
				continue
			}
			if g, w := c.got(a, ndl), c.want(a, ndl); g != w {
				t.Fatalf("%s/Bytes.%s(%x, %x) = %d, want %d", tier, c.op, a, ndl, g, w)
			}
		}
	}
}

// fuzzBytes compares the byte kernels, which have the sharpest contract of
// any of them because the standard library defines the answer exactly.
func fuzzBytes(t *testing.T, tier string, got, want kernel.Bytes, a, b []byte) {
	if got.IndexByte != nil {
		for _, c := range []byte{0, 'a', 0x7f, 0x80, 0xff} {
			if g, w := got.IndexByte(a, c), want.IndexByte(a, c); g != w {
				t.Fatalf("%s/Bytes.IndexByte(%#02x) = %d, want %d", tier, c, g, w)
			}
			if g, w := got.LastIndexByte(a, c), want.LastIndexByte(a, c); g != w {
				t.Fatalf("%s/Bytes.LastIndexByte(%#02x) = %d, want %d", tier, c, g, w)
			}
			if g, w := got.Count(a, c), want.Count(a, c); g != w {
				t.Fatalf("%s/Bytes.Count(%#02x) = %d, want %d", tier, c, g, w)
			}
		}
	}
	if got.Compare != nil {
		if g, w := got.Compare(a, b), want.Compare(a, b); g != w {
			t.Fatalf("%s/Bytes.Compare = %d, want %d", tier, g, w)
		}
	}
	if got.Equal != nil {
		if g, w := got.Equal(a, b), want.Equal(a, b); g != w {
			t.Fatalf("%s/Bytes.Equal = %v, want %v", tier, g, w)
		}
	}
	if got.ValidUTF8 != nil {
		if g, w := got.ValidUTF8(a), want.ValidUTF8(a); g != w {
			t.Fatalf("%s/Bytes.ValidUTF8 = %v, want %v", tier, g, w)
		}
	}
	if got.IsASCII != nil {
		if g, w := got.IsASCII(a), want.IsASCII(a); g != w {
			t.Fatalf("%s/Bytes.IsASCII = %v, want %v", tier, g, w)
		}
	}
	if got.PopCount != nil {
		if g, w := got.PopCount(a), want.PopCount(a); g != w {
			t.Fatalf("%s/Bytes.PopCount = %d, want %d", tier, g, w)
		}
	}
	if got.Index != nil && len(b) >= 3 {
		for _, ndl := range [][]byte{b[:1], b[:2], b[:3], a[:1]} {
			if g, w := got.Index(a, ndl), want.Index(a, ndl); g != w {
				t.Fatalf("%s/Bytes.Index(%x) = %d, want %d", tier, ndl, g, w)
			}
		}
	}
	if got.Xor != nil {
		g, w := make([]byte, len(a)), make([]byte, len(a))
		got.Xor(g, a, b)
		want.Xor(w, a, b)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/Bytes.Xor at %d: got %#02x want %#02x", tier, i, g[i], w[i])
		}
	}
	if got.ToUpperASCII != nil {
		g, w := make([]byte, len(a)), make([]byte, len(a))
		got.ToUpperASCII(g, a)
		want.ToUpperASCII(w, a)
		if i, ok := same(g, w); !ok {
			t.Fatalf("%s/Bytes.ToUpperASCII at %d: got %#02x want %#02x", tier, i, g[i], w[i])
		}
	}
}
