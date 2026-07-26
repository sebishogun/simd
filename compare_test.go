package simd_test

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/sebishogun/simd"
)

func TestComparisonsMatchGoOperators(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	for n := range maxLen + 1 {
		a, b := randF64(n, r), randF64(n, r)
		// Force some equalities, so the equal branches are actually exercised
		// rather than being astronomically unlikely.
		for i := 0; i < n; i += 3 {
			b[i] = a[i]
		}
		mask := make([]bool, n)

		for _, c := range []struct {
			name string
			fn   func(dst []bool, a, b []float64)
			want func(x, y float64) bool
		}{
			{"Equal", simd.EqualInto[float64], func(x, y float64) bool { return x == y }},
			{"NotEqual", simd.NotEqualInto[float64], func(x, y float64) bool { return x != y }},
			{"Less", simd.LessInto[float64], func(x, y float64) bool { return x < y }},
			{"LessEqual", simd.LessEqualInto[float64], func(x, y float64) bool { return x <= y }},
			{"Greater", simd.GreaterInto[float64], func(x, y float64) bool { return x > y }},
			{"GreaterEqual", simd.GreaterEqualInto[float64], func(x, y float64) bool { return x >= y }},
		} {
			c.fn(mask, a, b)
			for i := range mask {
				if want := c.want(a[i], b[i]); mask[i] != want {
					t.Fatalf("%s n=%d i=%d: got %v want %v", c.name, n, i, mask[i], want)
				}
			}
		}
	}
}

// IEEE 754 says every comparison involving NaN is false, so NotEqual is not
// the negation of Equal. Backends that implement one as the inverse of the
// other will fail here, which is the point.
func TestNaNComparisonSemantics(t *testing.T) {
	nan := math.NaN()
	a := []float64{nan, nan, 1}
	b := []float64{nan, 1, 1}
	mask := make([]bool, 3)

	simd.EqualInto(mask, a, b)
	if mask[0] || mask[1] || !mask[2] {
		t.Errorf("Equal with NaN = %v, want [false false true]", mask)
	}
	simd.NotEqualInto(mask, a, b)
	if !mask[0] || !mask[1] || mask[2] {
		t.Errorf("NotEqual with NaN = %v, want [true true false]", mask)
	}
	simd.LessInto(mask, a, b)
	if mask[0] || mask[1] || mask[2] {
		t.Errorf("Less with NaN = %v, want all false", mask)
	}
	simd.GreaterEqualInto(mask, a, b)
	if mask[0] || mask[1] || !mask[2] {
		t.Errorf("GreaterEqual with NaN = %v, want [false false true]", mask)
	}
}

func TestScalarComparisons(t *testing.T) {
	a := []float64{-2, -1, 0, 1, 2}
	mask := make([]bool, len(a))

	simd.GreaterScalarInto(mask, a, 0)
	if simd.CountTrue(mask) != 2 {
		t.Errorf("GreaterScalar(>0) matched %d, want 2", simd.CountTrue(mask))
	}
	simd.LessEqualScalarInto(mask, a, 0)
	if simd.CountTrue(mask) != 3 {
		t.Errorf("LessEqualScalar(<=0) matched %d, want 3", simd.CountTrue(mask))
	}
	simd.EqualScalarInto(mask, a, 1)
	if !simd.Any(mask) || simd.CountTrue(mask) != 1 {
		t.Errorf("EqualScalar(==1) = %v", mask)
	}
}

func TestMaskLogic(t *testing.T) {
	a := []bool{true, true, false, false}
	b := []bool{true, false, true, false}

	if !simd.All([]bool{true, true}) || simd.All(a) {
		t.Error("All is wrong")
	}
	if !simd.Any(a) || simd.Any([]bool{false, false}) {
		t.Error("Any is wrong")
	}
	// An empty mask is vacuously all-true and definitely not any-true.
	if !simd.All(nil) || simd.Any(nil) {
		t.Error("empty mask: want All=true, Any=false")
	}
	if got := simd.CountTrue(a); got != 2 {
		t.Errorf("CountTrue = %d, want 2", got)
	}

	dst := make([]bool, 4)
	simd.AndMaskInto(dst, a, b)
	if dst[0] != true || dst[1] || dst[2] || dst[3] {
		t.Errorf("AndMask = %v", dst)
	}
	simd.OrMaskInto(dst, a, b)
	if !dst[0] || !dst[1] || !dst[2] || dst[3] {
		t.Errorf("OrMask = %v", dst)
	}
	simd.XorMaskInto(dst, a, b)
	if dst[0] || !dst[1] || !dst[2] || dst[3] {
		t.Errorf("XorMask = %v", dst)
	}
	simd.NotMaskInto(dst, a)
	if dst[0] || dst[1] || !dst[2] || !dst[3] {
		t.Errorf("NotMask = %v", dst)
	}
}

func TestSelect(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	zero := make([]float64, 4)
	mask := make([]bool, 4)
	out := make([]float64, 4)

	// Keep the elements above 2, zero the rest — the canonical use.
	simd.GreaterScalarInto(mask, a, 2)
	simd.SelectInto(out, mask, a, zero)
	if !equalF64(out, []float64{0, 0, 3, 4}) {
		t.Errorf("Select = %v, want [0 0 3 4]", out)
	}
}

func TestGatherAndScatter(t *testing.T) {
	src := []float64{10, 20, 30, 40, 50}
	idx := []int32{4, 0, 2}
	dst := make([]float64, 3)

	simd.GatherInto(dst, src, idx)
	if !equalF64(dst, []float64{50, 10, 30}) {
		t.Errorf("Gather = %v, want [50 10 30]", dst)
	}

	out := make([]float64, 5)
	simd.ScatterInto(out, idx, []float64{1, 2, 3})
	if !equalF64(out, []float64{2, 0, 3, 0, 1}) {
		t.Errorf("Scatter = %v, want [2 0 3 0 1]", out)
	}

	// Out-of-range indices are skipped rather than panicking.
	bad := []int32{-1, 99, 1}
	dst2 := []float64{-1, -1, -1}
	simd.GatherInto(dst2, src, bad)
	if dst2[0] != -1 || dst2[1] != -1 || dst2[2] != 20 {
		t.Errorf("Gather with bad indices = %v, want the valid one applied only", dst2)
	}
	out2 := make([]float64, 5)
	simd.ScatterInto(out2, bad, []float64{7, 8, 9}) // must not panic
	if out2[1] != 9 {
		t.Errorf("Scatter with bad indices = %v", out2)
	}
}

func TestRampAndTile(t *testing.T) {
	a := make([]float64, 5)
	simd.Ramp(a, 10, 0.5)
	if !equalF64(a, []float64{10, 10.5, 11, 11.5, 12}) {
		t.Errorf("Ramp = %v", a)
	}

	b := make([]int32, 7)
	simd.Tile(b, []int32{1, 2, 3})
	want := []int32{1, 2, 3, 1, 2, 3, 1}
	for i := range want {
		if b[i] != want[i] {
			t.Errorf("Tile = %v, want %v", b, want)
			break
		}
	}
	// An empty pattern leaves the slice alone rather than dividing by zero.
	c := []int32{9, 9}
	simd.Tile(c, nil)
	if c[0] != 9 {
		t.Errorf("Tile with an empty pattern changed the slice: %v", c)
	}
}

func TestMedian(t *testing.T) {
	odd := []float64{5, 1, 3}
	if got := simd.Median(odd); got != 3 {
		t.Errorf("Median odd = %v, want 3", got)
	}
	even := []float64{4, 1, 3, 2}
	if got := simd.Median(even); got != 2.5 {
		t.Errorf("Median even = %v, want 2.5", got)
	}
	// Integers average without overflowing.
	big := []int64{math.MaxInt64, math.MaxInt64 - 2}
	if got := simd.Median(big); got != math.MaxInt64-1 {
		t.Errorf("Median of two huge ints = %d, want %d", got, int64(math.MaxInt64-1))
	}
	// NaN sorts to the end and must not corrupt the middle.
	withNaN := []float64{2, math.NaN(), 1, 3}
	if got := simd.Median(withNaN); math.IsNaN(got) {
		t.Errorf("Median with one NaN = %v; NaN should sort to the end", got)
	}
}

func TestMatMul(t *testing.T) {
	// [1 2 3]   [7  8]     [58  64]
	// [4 5 6] x [9 10]  =  [139 154]
	//           [11 12]
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{7, 8, 9, 10, 11, 12}
	dst := make([]float64, 4)
	simd.MatMulInto(dst, a, b, 2, 3, 2)
	if !equalF64(dst, []float64{58, 64, 139, 154}) {
		t.Errorf("MatMul = %v, want [58 64 139 154]", dst)
	}

	// Multiplying by the identity must be a no-op.
	id := []float64{1, 0, 0, 1}
	m := []float64{3, 5, 7, 9}
	out := make([]float64, 4)
	simd.MatMulInto(out, m, id, 2, 2, 2)
	if !equalF64(out, m) {
		t.Errorf("M x I = %v, want %v", out, m)
	}

	// Undersized slices do nothing rather than reading out of bounds.
	short := make([]float64, 1)
	simd.MatMulInto(short, a, b, 2, 3, 2)
}

func TestConvert(t *testing.T) {
	f64 := []float64{1.9, -1.9, 3.0}
	i32 := make([]int32, 3)
	simd.ConvertInto(i32, f64)
	// Float to integer truncates toward zero, matching Go and the hardware.
	if i32[0] != 1 || i32[1] != -1 || i32[2] != 3 {
		t.Errorf("float64 to int32 = %v, want [1 -1 3]", i32)
	}

	f32 := make([]float32, 3)
	simd.ConvertInto(f32, f64)
	if f32[0] != 1.9 {
		t.Errorf("float64 to float32 = %v", f32)
	}

	back := make([]float64, 3)
	simd.ConvertInto(back, i32)
	if !equalF64(back, []float64{1, -1, 3}) {
		t.Errorf("int32 to float64 = %v", back)
	}

	// The shorter of the two bounds the work.
	small := make([]int64, 1)
	simd.ConvertInto(small, f64)
	if small[0] != 1 {
		t.Errorf("bounded convert = %v", small)
	}
}

func TestNewSurfaceDoesNotAllocate(t *testing.T) {
	const n = 256
	a, b := make([]float64, n), make([]float64, n)
	mask := make([]bool, n)
	mask2 := make([]bool, n)
	idx := make([]int32, n)
	i32 := make([]int32, n)
	mm := make([]float64, 64)

	for _, c := range []struct {
		name string
		fn   func()
	}{
		{"EqualInto", func() { simd.EqualInto(mask, a, b) }},
		{"GreaterScalarInto", func() { simd.GreaterScalarInto(mask, a, 0) }},
		{"All", func() { sinkB = simd.All(mask) }},
		{"CountTrue", func() { sinkN = simd.CountTrue(mask) }},
		{"AndMaskInto", func() { simd.AndMaskInto(mask, mask, mask2) }},
		{"SelectInto", func() { simd.SelectInto(a, mask, a, b) }},
		{"GatherInto", func() { simd.GatherInto(a, b, idx) }},
		{"ScatterInto", func() { simd.ScatterInto(a, idx, b) }},
		{"Ramp", func() { simd.Ramp(a, 0, 1) }},
		{"Tile", func() { simd.Tile(a, b[:4]) }},
		{"Median", func() { sinkF = simd.Median(a) }},
		{"MatMulInto", func() { simd.MatMulInto(mm, mm, mm, 8, 8, 8) }},
		{"ConvertInto", func() { simd.ConvertInto(i32, a) }},
		{"PolyEvalInto", func() { simd.PolyEvalInto(a, b, b[:4]) }},
		{"CorrelateInto", func() { simd.CorrelateInto(a, b, b[:8]) }},
		{"Prod", func() { sinkF = simd.Prod(a) }},
		{"CumProd", func() { simd.CumProdInto(a, b) }},
		{"DiffInto", func() { simd.DiffInto(a, b) }},
		{"Trapezoid", func() { sinkF = simd.Trapezoid(a, 0.1) }},
		{"Simpson", func() { sinkF = simd.Simpson(a[:255], 0.1) }},
		{"Softmax", func() { simd.Softmax(a) }},
		{"LogSumExp", func() { sinkF = simd.LogSumExp(a) }},
		{"ReLU", func() { simd.ReLU(a) }},
		{"GELU", func() { simd.GELU(a) }},
		{"LayerNorm", func() { simd.LayerNorm(a, 1e-5) }},
		{"Correlation", func() { sinkF = simd.Correlation(a, b) }},
		{"IndexAll", func() { sinkN = simd.IndexAll(idx, make([]byte, 0), 0) }},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := testing.AllocsPerRun(50, c.fn); got != 0 {
				t.Errorf("%s allocated %.1f times per call, want 0", c.name, got)
			}
		})
	}
}

// Quickselect is easy to get subtly wrong — the first implementation here
// corrupted two-element partitions and only the even-length case caught it.
// This checks it against a fully sorted copy across every length and several
// adversarial shapes, which is the cheap way to be sure.
func TestMedianAgainstSortedOracle(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 32))
	oracle := func(s []float64) float64 {
		c := clone(s)
		slices.Sort(c)
		n := len(c)
		if n%2 == 1 {
			return c[n/2]
		}
		return (c[n/2-1] + c[n/2]) / 2
	}

	shapes := map[string]func(n int) []float64{
		"random":   func(n int) []float64 { return randF64(n, r) },
		"sorted":   func(n int) []float64 { s := randF64(n, r); slices.Sort(s); return s },
		"reversed": func(n int) []float64 { s := randF64(n, r); slices.Sort(s); slices.Reverse(s); return s },
		"constant": func(n int) []float64 { s := make([]float64, n); simd.Fill(s, 7); return s },
		"twoValued": func(n int) []float64 {
			s := make([]float64, n)
			for i := range s {
				s[i] = float64(i % 2)
			}
			return s
		},
	}

	for name, gen := range shapes {
		t.Run(name, func(t *testing.T) {
			for n := 1; n <= maxLen; n++ {
				in := gen(n)
				want := oracle(in)
				got := simd.Median(clone(in))
				if got != want {
					t.Fatalf("n=%d: got %v want %v (input %v)", n, got, want, in)
				}
			}
		})
	}

	// Integers, including the even-length averaging that must not overflow.
	for n := 1; n <= maxLen; n++ {
		in := randI64(n, r)
		c := clone(in)
		slices.Sort(c)
		var want int64
		if n%2 == 1 {
			want = c[n/2]
		} else {
			lo, hi := c[n/2-1], c[n/2]
			want = lo + (hi-lo)/2
		}
		if got := simd.Median(clone(in)); got != want {
			t.Fatalf("int n=%d: got %d want %d", n, got, want)
		}
	}
}

func TestQuantileAgainstSortedOracle(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	// R type 7 / numpy "linear": interpolate between the bracketing order
	// statistics of the sorted data.
	oracle := func(s []float64, q float64) float64 {
		c := clone(s)
		slices.Sort(c)
		pos := q * float64(len(c)-1)
		lo := int(pos)
		frac := pos - float64(lo)
		if lo >= len(c)-1 {
			return c[len(c)-1]
		}
		return c[lo] + frac*(c[lo+1]-c[lo])
	}
	for n := 1; n <= maxLen; n++ {
		in := randF64(n, r)
		for _, q := range []float64{0, 0.1, 0.25, 0.5, 0.75, 0.9, 1} {
			want := oracle(in, q)
			got := simd.Quantile(clone(in), q)
			if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
				t.Fatalf("Quantile n=%d q=%v: got %v want %v", n, q, got, want)
			}
		}
	}
	// Endpoints are the extremes, and q is clamped rather than rejected.
	a := []float64{5, 1, 3}
	if got := simd.Quantile(clone(a), 0); got != 1 {
		t.Errorf("Quantile(0) = %v, want the minimum 1", got)
	}
	if got := simd.Quantile(clone(a), 1); got != 5 {
		t.Errorf("Quantile(1) = %v, want the maximum 5", got)
	}
	if got := simd.Quantile(clone(a), -3); got != 1 {
		t.Errorf("Quantile(-3) = %v, want it clamped to the minimum", got)
	}
	if got := simd.Quantile(clone(a), 99); got != 5 {
		t.Errorf("Quantile(99) = %v, want it clamped to the maximum", got)
	}
	if got := testing.AllocsPerRun(50, func() { sinkF = simd.Quantile(a, 0.5) }); got != 0 {
		t.Errorf("Quantile allocated %.1f times, want 0", got)
	}
}
