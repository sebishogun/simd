package arrays

import (
	"testing"

	"github.com/sebishogun/simd"
)

// TestNoAllocations backs the claim in the package documentation that nothing
// here allocates. It is a real guarantee people build on, so it is enforced
// rather than asserted: a stray allocation in a kernel or in the dispatch path
// fails the build.
//
// The generic dispatch is the specific thing at risk. Every exported function
// funnels through a type switch on an interface value, and if the compiler
// ever stops keeping that boxed value on the stack it will show up here.
func TestNoAllocations(t *testing.T) {
	const n = 1024

	f32, f32b := make([]float32, n), make([]float32, n)
	f64, f64b := make([]float64, n), make([]float64, n)
	i32, i32b := make([]int32, n), make([]int32, n)
	i64, i64b := make([]int64, n), make([]int64, n)
	by, byb := make([]byte, n), make([]byte, n)
	dst64 := make([]float64, n)

	// Give the float slices non-degenerate values so Normalize and
	// Standardize take their real paths rather than their zero shortcuts.
	for i := range f64 {
		f64[i] = float64(i) + 1
		f64b[i] = float64(n - i)
	}

	cases := []struct {
		name string
		fn   func()
	}{
		// elementwise, in place
		{"Add/f32", func() { simd.Add(f32, f32b) }},
		{"Add/f64", func() { simd.Add(f64, f64b) }},
		{"Add/i32", func() { simd.Add(i32, i32b) }},
		{"Add/i64", func() { simd.Add(i64, i64b) }},
		{"Sub/f64", func() { simd.Sub(f64, f64b) }},
		{"Mul/f64", func() { simd.Mul(f64, f64b) }},
		{"Div/f64", func() { simd.Div(f64, f64b) }},
		{"Minimum/f64", func() { simd.Minimum(f64, f64b) }},
		{"Maximum/f64", func() { simd.Maximum(f64, f64b) }},

		// elementwise, with a destination
		{"AddInto/f64", func() { simd.AddInto(dst64, f64, f64b) }},
		{"AddScaledInto/f64", func() { simd.AddScaledInto(dst64, f64, f64b, 2) }},

		// unary
		{"Abs/f64", func() { simd.Abs(f64) }},
		{"Neg/f64", func() { simd.Neg(f64) }},
		{"Sqrt/f64", func() { simd.Sqrt(f64) }},
		{"Reverse/f64", func() { simd.Reverse(f64) }},

		// scalar operand
		{"Scale/f64", func() { simd.Scale(f64, 1) }},
		{"AddScalar/f64", func() { simd.AddScalar(f64, 0) }},
		{"DivScalar/f64", func() { simd.DivScalar(f64, 1) }},
		{"Clamp/f64", func() { simd.Clamp(f64, -1e30, 1e30) }},
		{"Fill/f64", func() { simd.Fill(dst64, 1) }},
		{"CumSum/f64", func() { simd.CumSumInto(dst64, f64) }},

		// reductions
		{"Sum/f64", func() { sinkF = simd.Sum(f64) }},
		{"Sum/i64", func() { sinkI = simd.Sum(i64) }},
		{"Dot/f64", func() { sinkF = simd.Dot(f64, f64b) }},
		{"Norm/f64", func() { sinkF = simd.Norm(f64) }},
		{"L1Norm/f64", func() { sinkF = simd.L1Norm(f64) }},
		{"Min/f64", func() { sinkF = simd.Min(f64) }},
		{"Max/f64", func() { sinkF = simd.Max(f64) }},
		{"MinMax/f64", func() { sinkF, _ = simd.MinMax(f64) }},
		{"ArgMin/f64", func() { sinkN = simd.ArgMin(f64) }},
		{"ArgMax/f64", func() { sinkN = simd.ArgMax(f64) }},

		// composite scenarios
		{"Mean/f64", func() { sinkF = simd.Mean(f64) }},
		{"Variance/f64", func() { sinkF = simd.Variance(f64) }},
		{"StdDev/f64", func() { sinkF = simd.StdDev(f64) }},
		{"Distance/f64", func() { sinkF = simd.Distance(f64, f64b) }},
		{"SquaredDistance/f64", func() { sinkF = simd.SquaredDistance(f64, f64b) }},
		{"ManhattanDistance/f64", func() { sinkF = simd.ManhattanDistance(f64, f64b) }},
		{"CosineSimilarity/f64", func() { sinkF = simd.CosineSimilarity(f64, f64b) }},
		{"Normalize/f64", func() { simd.Normalize(f64) }},
		{"Standardize/f64", func() { simd.Standardize(f64) }},
		{"Rescale/f64", func() { simd.Rescale(f64, 0, 1) }},

		// bytes and bits
		{"IndexByte", func() { sinkN = simd.IndexByte(by, 0xff) }},
		{"CountByte", func() { sinkN = simd.CountByte(by, 0) }},
		{"Equal", func() { sinkB = simd.Equal(by, byb) }},
		{"Compare", func() { sinkN = simd.Compare(by, byb) }},
		{"PopCount", func() { sinkN = simd.PopCount(by) }},
		{"Xor", func() { simd.Xor(by, byb) }},
		{"AndNot", func() { simd.AndNot(by, byb) }},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := testing.AllocsPerRun(100, c.fn); got != 0 {
				t.Errorf("%s allocated %.1f times per call, want 0", c.name, got)
			}
		})
	}
}

// Sinks, to stop the compiler deleting the calls whose results are unused.
var (
	sinkF float64
	sinkI int64
	sinkN int
	sinkB bool
)
