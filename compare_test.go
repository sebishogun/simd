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

// TestSelectAcceleratedMatchesReference pins the accelerated Median and
// Quantile against the portable ones at sizes above selectCutoff.
//
// Every other test in this file runs at maxLen = 70, which is far below the
// threshold, so none of them execute the quickselect built on the partition
// kernel at all. This one does, and it compares against the reference rather
// than against a sorted oracle: the contract is bit-identical results between
// the accelerated and portable paths, which is stricter than agreeing with an
// oracle to within a tolerance.
//
// Passing a nil scratch is what forces the reference path — MedianInto falls
// back when the scratch is too short rather than panicking — so both sides of
// the comparison go through exactly the same entry point.
func TestSelectAcceleratedMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(97, 98))

	shapes := map[string]func(n int) []float64{
		"random":   func(n int) []float64 { return randF64(n, r) },
		"sorted":   func(n int) []float64 { s := randF64(n, r); slices.Sort(s); return s },
		"reversed": func(n int) []float64 { s := randF64(n, r); slices.Sort(s); slices.Reverse(s); return s },
		"constant": func(n int) []float64 { s := make([]float64, n); simd.Fill(s, 7); return s },
		// Few distinct values is the shape that makes a median-of-three pivot
		// equal to much of the window, which is where a select stops making
		// progress and has to notice.
		"fewDistinct": func(n int) []float64 {
			s := make([]float64, n)
			for i := range s {
				s[i] = float64(i % 3)
			}
			return s
		},
		"withNaN": func(n int) []float64 {
			s := randF64(n, r)
			for i := 0; i < n; i += 17 {
				s[i] = math.NaN()
			}
			return s
		},
		"signedZeroAndInf": func(n int) []float64 {
			s := randF64(n, r)
			for i := 0; i < n; i += 11 {
				switch i % 44 {
				case 0:
					s[i] = 0
				case 11:
					s[i] = math.Copysign(0, -1)
				case 22:
					s[i] = math.Inf(1)
				default:
					s[i] = math.Inf(-1)
				}
			}
			return s
		},
	}

	// Straddling the cutoff on both sides and on both parities, because the
	// even case takes a different route through the lower middle.
	sizes := []int{simd.SelectMinLenForTest - 1, simd.SelectMinLenForTest, simd.SelectMinLenForTest + 1,
		2047, 2048, 4097, 10000}

	// Bit equality, with one carve-out: -0 and +0. The order these functions
	// use is defined by `<`, exactly as slices.Sort and cmp.Less define it,
	// and under `<` the two zeros compare equal. Which of two equal-comparing
	// values lands in a given position is therefore a property of the
	// algorithm, and the accelerated path is a different algorithm — a stable
	// out-of-place partition feeding pdqsort, against an in-place Hoare
	// quickselect. Sort has always had this; see the note there. Everything
	// else, NaN included, must match bit for bit.
	eq := func(x, y float64) bool {
		if x == 0 && y == 0 {
			return true
		}
		return math.Float64bits(x) == math.Float64bits(y) ||
			(math.IsNaN(x) && math.IsNaN(y))
	}

	for name, gen := range shapes {
		t.Run(name, func(t *testing.T) {
			for _, n := range sizes {
				in := gen(n)
				scratch := make([]float64, n)

				want := simd.MedianInto(clone(in), nil)
				got := simd.MedianInto(clone(in), scratch)
				if !eq(got, want) {
					t.Fatalf("Median n=%d: accelerated %v, reference %v", n, got, want)
				}

				for _, q := range []float64{0, 0.001, 0.25, 0.5, 0.75, 0.999, 1} {
					want := simd.QuantileInto(clone(in), nil, q)
					got := simd.QuantileInto(clone(in), scratch, q)
					if !eq(got, want) {
						t.Fatalf("Quantile n=%d q=%v: accelerated %v, reference %v",
							n, q, got, want)
					}
				}
			}
		})
	}

	// Integers too, where the even-length average halves the gap rather than
	// the sum and must not overflow.
	t.Run("int64", func(t *testing.T) {
		for _, n := range sizes {
			in := randI64(n, r)
			scratch := make([]int64, n)
			if got, want := simd.MedianInto(clone(in), scratch), simd.MedianInto(clone(in), nil); got != want {
				t.Fatalf("Median int64 n=%d: accelerated %d, reference %d", n, got, want)
			}
			for _, q := range []float64{0, 0.25, 0.5, 0.75, 1} {
				got := simd.QuantileInto(clone(in), scratch, q)
				want := simd.QuantileInto(clone(in), nil, q)
				if got != want {
					t.Fatalf("Quantile int64 n=%d q=%v: accelerated %d, reference %d",
						n, q, got, want)
				}
			}
		}
	})

	// The reordering contract: a[k] is the k-th order statistic and nothing
	// left of it is larger. Median promises this implicitly by returning the
	// middle, but a caller that reuses the slice depends on it directly.
	t.Run("leavesSlicePartitioned", func(t *testing.T) {
		a := randF64(4096, r)
		scratch := make([]float64, len(a))
		m := simd.MedianInto(a, scratch)
		mid := a[len(a)/2]
		for i, v := range a[:len(a)/2] {
			if v > mid {
				t.Fatalf("a[%d]=%v is greater than the middle %v (median %v)", i, v, mid, m)
			}
		}
	})

	// The whole point of the Into forms.
	t.Run("allocFree", func(t *testing.T) {
		a := randF64(4096, r)
		scratch := make([]float64, len(a))
		if got := testing.AllocsPerRun(20, func() { sinkF = simd.MedianInto(a, scratch) }); got != 0 {
			t.Errorf("MedianInto allocated %.1f times, want 0", got)
		}
		if got := testing.AllocsPerRun(20, func() { sinkF = simd.QuantileInto(a, scratch, 0.9) }); got != 0 {
			t.Errorf("QuantileInto allocated %.1f times, want 0", got)
		}
	})
}

// CommonPrefixLen is checked against the definition rather than against
// bytes.Compare, which cannot express it: Compare says which way the first
// difference went, and this says where it was.
func TestCommonPrefixLen(t *testing.T) {
	naive := func(a, b []byte) int {
		n := min(len(a), len(b))
		for i := range n {
			if a[i] != b[i] {
				return i
			}
		}
		return n
	}

	r := rand.New(rand.NewPCG(31, 32))
	// Shared prefixes on both sides of the 64-byte block and of the dispatch
	// threshold: the blocked scan, the block containing the difference and the
	// byte-wise tail all have to be exercised.
	for _, shared := range []int{0, 1, 63, 64, 65, 127, 128, 129, 1000, 5000} {
		for _, extra := range []int{0, 1, 7, 64, 500} {
			a := make([]byte, shared+extra)
			for i := range a {
				a[i] = byte(r.UintN(256))
			}
			b := append([]byte(nil), a...)
			if extra > 0 {
				b[shared] ^= 0xff // first difference exactly at `shared`
			}
			if got, want := simd.CommonPrefixLen(a, b), naive(a, b); got != want {
				t.Fatalf("shared=%d extra=%d: got %d want %d",
					shared, extra, got, want)
			}
			// And with one side truncated, so the answer is bounded by the
			// shorter rather than by a difference.
			for _, cut := range []int{0, 1, shared / 2} {
				if cut > len(b) {
					continue
				}
				sb := b[:cut]
				if got, want := simd.CommonPrefixLen(a, sb), naive(a, sb); got != want {
					t.Fatalf("shared=%d cut=%d: got %d want %d", shared, cut, got, want)
				}
			}
		}
	}

	// Strings and byte slices are interchangeable, as everywhere else here.
	if got := simd.CommonPrefixLen("prefixAAA", "prefixBBB"); got != 6 {
		t.Errorf("CommonPrefixLen on strings = %d, want 6", got)
	}
	if got := simd.CommonPrefixLen[[]byte, []byte](nil, nil); got != 0 {
		t.Errorf("CommonPrefixLen(nil, nil) = %d, want 0", got)
	}
}

func TestCommonPrefixLenNoAlloc(t *testing.T) {
	a := make([]byte, 8192)
	b := make([]byte, 8192)
	if n := testing.AllocsPerRun(50, func() { _ = simd.CommonPrefixLen(a, b) }); n != 0 {
		t.Errorf("CommonPrefixLen allocated %v times per run", n)
	}
}

// LowerBoundInto against sort.SearchInts, which is what callers would
// otherwise reach for and is the definition it has to match.
func TestLowerBound(t *testing.T) {
	r := rand.New(rand.NewPCG(61, 62))
	// Table lengths spanning powers of two, since the kernel steps down through
	// them and an off-by-one at the top step would only show up when the length
	// is not one itself.
	for _, ntab := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1000} {
		table := make([]int32, ntab)
		for i := range table {
			table[i] = int32(i) * 2 // even values, so odd queries fall between
		}
		// Every table value, every gap, and both ends past the range.
		var q []int32
		for i := range ntab {
			q = append(q, table[i], table[i]-1, table[i]+1)
		}
		q = append(q, -1000, 1000000, 0)
		for range 50 {
			q = append(q, int32(r.IntN(2*max(ntab, 1)+4))-2)
		}

		dst := make([]int32, len(q))
		simd.LowerBoundInto(dst, table, q)
		for i, v := range q {
			w := 0
			for w < len(table) && table[w] < v {
				w++
			}
			if int(dst[i]) != w {
				t.Fatalf("ntab=%d q=%d: got %d want %d", ntab, v, dst[i], w)
			}
		}
	}
}

// Duplicates are the case lower_bound is defined by: the answer is the index
// of the FIRST equal element, not any of them.
func TestLowerBoundDuplicates(t *testing.T) {
	table := []int32{1, 1, 1, 2, 2, 5, 5, 5, 5, 9}
	q := []int32{0, 1, 2, 3, 5, 9, 10}
	want := []int32{0, 0, 3, 5, 5, 9, 10}
	dst := make([]int32, len(q))
	simd.LowerBoundInto(dst, table, q)
	for i := range q {
		if dst[i] != want[i] {
			t.Errorf("q=%d: got %d want %d", q[i], dst[i], want[i])
		}
	}
}

func TestLowerBoundFloats(t *testing.T) {
	table := []float64{-3.5, -1, 0, 0.25, 2, 7.75}
	q := []float64{-4, -3.5, -2, 0, 0.1, 8}
	want := []int32{0, 0, 1, 2, 3, 6}
	dst := make([]int32, len(q))
	simd.LowerBoundInto(dst, table, q)
	for i := range q {
		if dst[i] != want[i] {
			t.Errorf("q=%v: got %d want %d", q[i], dst[i], want[i])
		}
	}
}

func TestLowerBoundNoAlloc(t *testing.T) {
	table := make([]int32, 4096)
	for i := range table {
		table[i] = int32(i)
	}
	q := make([]int32, 1024)
	dst := make([]int32, len(q))
	if n := testing.AllocsPerRun(20, func() { simd.LowerBoundInto(dst, table, q) }); n != 0 {
		t.Errorf("LowerBoundInto allocated %v times per run", n)
	}
}
