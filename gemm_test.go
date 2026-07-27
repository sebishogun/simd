package simd_test

// Matrix multiply and matrix-vector multiply.
//
// The register-blocked matmul computes a tile of the output in registers and
// stores it once, which means its failure modes are geometric rather than
// arithmetic: a tile written to the wrong rows, an edge column left as the zero
// it was initialised to, a row tail that never runs. None of those show up on a
// square matrix whose size happens to be a multiple of the tile — which is
// exactly the shape everyone benchmarks with. So the sizes below are chosen to
// straddle the tile in both dimensions, and the values are chosen so that every
// output element is distinguishable from every other.

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

// naiveMatMul is the textbook triple loop, in the order the contract specifies:
// each output element accumulated over p ascending.
//
// The conversion around the product is load-bearing and is not a style choice.
// Go's spec permits an implementation to fuse a multiply and an add into a
// single operation with one rounding, and on arm64 it does exactly that — so
// written the obvious way this reference computes different bits from the
// kernel it is checking, and the test fails on arm64 while passing on amd64.
// An explicit conversion rounds to the target type and forbids the fusion.
// internal/ref does the same thing for the same reason.
func naiveMatMul[T float32 | float64](dst, a, b []T, m, k, n int) {
	for i := range m {
		for j := range n {
			var acc T
			for p := range k {
				acc += T(a[i*k+p] * b[p*n+j])
			}
			dst[i*n+j] = acc
		}
	}
}

// gemmSizes straddles every tile boundary the kernel can have. GEMM_MR is 8 on
// AVX-512, 6 on AVX2 and 4 elsewhere, and the column tile is the vector width,
// so this covers one below, exactly, and one above each of them.
var gemmSizes = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 32, 33, 64, 65}

func TestMatMulShapes(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, m := range gemmSizes {
		for _, k := range []int{1, 3, 8, 17, 64} {
			for _, n := range gemmSizes {
				a := make([]float64, m*k)
				b := make([]float64, k*n)
				for i := range a {
					a[i] = float64(rng.Intn(17) - 8)
				}
				for i := range b {
					b[i] = float64(rng.Intn(17) - 8)
				}
				got := make([]float64, m*n)
				want := make([]float64, m*n)
				simd.MatMulInto(got, a, b, m, k, n)
				naiveMatMul(want, a, b, m, k, n)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("m=%d k=%d n=%d: dst[%d] (row %d, col %d) = %v, want %v",
							m, k, n, i, i/n, i%n, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestMatMulExactBits uses values whose products and sums are not exactly
// representable, so that any difference in the order of accumulation shows up
// as a different answer rather than being hidden by exact arithmetic. Integers
// would pass any accumulation order at all, which is why the shape test above
// is not enough on its own.
func TestMatMulExactBits(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, dim := range [][3]int{{7, 13, 19}, {16, 16, 16}, {33, 5, 65}, {1, 64, 1}} {
		m, k, n := dim[0], dim[1], dim[2]
		a := make([]float32, m*k)
		b := make([]float32, k*n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := range b {
			b[i] = float32(rng.NormFloat64())
		}
		got := make([]float32, m*n)
		want := make([]float32, m*n)
		simd.MatMulInto(got, a, b, m, k, n)
		naiveMatMul(want, a, b, m, k, n)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%dx%dx%d: dst[%d] = %#08x, want %#08x (%v vs %v)",
					m, k, n, i, math.Float32bits(got[i]), math.Float32bits(want[i]),
					got[i], want[i])
			}
		}
	}
}

// TestMatMulZeroTimesInfinity pins the semantic change made when the blocked
// kernel landed. A zero in a multiplied by an infinity in b is a NaN, which is
// what IEEE 754 says and what BLAS and numpy produce. The previous
// implementation skipped a zero scalar and so returned a finite number here.
func TestMatMulZeroTimesInfinity(t *testing.T) {
	// One row, two columns of a; a[0] is zero, so the infinity in b's first
	// row is what it multiplies.
	a := []float64{0, 1}
	b := []float64{math.Inf(1), 2}
	dst := make([]float64, 1)
	simd.MatMulInto(dst, a, b, 1, 2, 1)
	if !math.IsNaN(dst[0]) {
		t.Errorf("0*Inf + 1*2 = %v, want NaN", dst[0])
	}

	// And large enough to go through the tiled path rather than an edge.
	const m, k, n = 64, 64, 64
	la := make([]float64, m*k)
	lb := make([]float64, k*n)
	for i := range lb {
		lb[i] = 1
	}
	lb[0] = math.Inf(1) // b[0][0]
	ld := make([]float64, m*n)
	simd.MatMulInto(ld, la, lb, m, k, n) // la is all zeros
	if !math.IsNaN(ld[0]) {
		t.Errorf("tiled path: dst[0] = %v, want NaN from 0*Inf", ld[0])
	}
}

// TestMatMulRejectsBadSizes checks that an undersized slice is a no-op rather
// than a panic or a partial write, which is the documented contract and the
// condition the generated guard routes to the portable path.
func TestMatMulRejectsBadSizes(t *testing.T) {
	for _, c := range []struct {
		name       string
		dl, al, bl int
		m, k, n    int
	}{
		{"short dst", 3, 4, 4, 2, 2, 2},
		{"short a", 4, 3, 4, 2, 2, 2},
		{"short b", 4, 4, 3, 2, 2, 2},
		{"zero m", 4, 4, 4, 0, 2, 2},
		{"negative k", 4, 4, 4, 2, -1, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			dst := make([]float64, c.dl)
			for i := range dst {
				dst[i] = -7
			}
			simd.MatMulInto(dst, make([]float64, c.al), make([]float64, c.bl), c.m, c.k, c.n)
			for i, v := range dst {
				if v != -7 {
					t.Fatalf("dst[%d] was written (%v); the call should have done nothing", i, v)
				}
			}
		})
	}
}

// TestGemvIsDotPerRow is the property Gemv is documented to have, checked
// directly rather than inferred: row i of Gemv is bit-identical to Dot of that
// row against x. If the kernel ever picks its own summation order this fails,
// and it should.
func TestGemvIsDotPerRow(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, m := range []int{1, 2, 7, 8, 9, 33, 64} {
		for _, k := range []int{1, 2, 15, 16, 17, 31, 64, 257} {
			a := make([]float64, m*k)
			x := make([]float64, k)
			for i := range a {
				a[i] = rng.NormFloat64()
			}
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			dst := make([]float64, m)
			simd.GemvInto(dst, a, x, m, k)
			for i := range m {
				want := simd.Dot(a[i*k:(i+1)*k], x)
				if math.Float64bits(dst[i]) != math.Float64bits(want) {
					t.Fatalf("m=%d k=%d row %d: Gemv gave %v, Dot gave %v",
						m, k, i, dst[i], want)
				}
			}
		}
	}
}

func TestGemvFloat32(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	for _, k := range []int{1, 16, 17, 100} {
		const m = 9
		a := make([]float32, m*k)
		x := make([]float32, k)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, m)
		simd.GemvInto(dst, a, x, m, k)
		for i := range m {
			want := simd.Dot(a[i*k:(i+1)*k], x)
			if math.Float32bits(dst[i]) != math.Float32bits(want) {
				t.Fatalf("k=%d row %d: %v vs %v", k, i, dst[i], want)
			}
		}
	}
}

func TestGemvSpecialValues(t *testing.T) {
	a := []float64{
		1, math.Inf(1), // row 0
		0, math.Inf(1), // row 1: 0*Inf
		math.NaN(), 1, // row 2
	}
	x := []float64{2, 3}
	dst := make([]float64, 3)
	simd.GemvInto(dst, a, x, 3, 2)

	if !math.IsInf(dst[0], 1) {
		t.Errorf("row 0 = %v, want +Inf", dst[0])
	}
	if !math.IsInf(dst[1], 1) {
		t.Errorf("row 1 = %v, want +Inf (0*2 + Inf*3)", dst[1])
	}
	if !math.IsNaN(dst[2]) {
		t.Errorf("row 2 = %v, want NaN", dst[2])
	}
}

func TestGemvRejectsBadSizes(t *testing.T) {
	dst := []float64{-7, -7}
	simd.GemvInto(dst, make([]float64, 3), make([]float64, 2), 2, 2)
	for i, v := range dst {
		if v != -7 {
			t.Fatalf("dst[%d] = %v; a short matrix should have been a no-op", i, v)
		}
	}
}

// TestGemvMatchesMatMul checks the two against each other on the same data,
// which is worth doing because they take different paths through the library
// and have deliberately different accumulation orders — so this asserts they
// agree to within reduction slack, not bit for bit.
func TestGemvMatchesMatMul(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const m, k = 40, 40
	a := make([]float64, m*k)
	x := make([]float64, k)
	for i := range a {
		a[i] = rng.NormFloat64()
	}
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	viaGemv := make([]float64, m)
	viaMatMul := make([]float64, m)
	simd.GemvInto(viaGemv, a, x, m, k)
	simd.MatMulInto(viaMatMul, a, x, m, k, 1)
	for i := range m {
		if d := math.Abs(viaGemv[i] - viaMatMul[i]); d > 1e-12*math.Abs(viaMatMul[i])+1e-12 {
			t.Fatalf("row %d: Gemv %v, MatMul %v (differ by %g)", i, viaGemv[i], viaMatMul[i], d)
		}
	}
}

func TestMatMulIdentity(t *testing.T) {
	const n = 20
	id := make([]float64, n*n)
	for i := range n {
		id[i*n+i] = 1
	}
	rng := rand.New(rand.NewSource(8))
	a := make([]float64, n*n)
	for i := range a {
		a[i] = rng.NormFloat64()
	}
	dst := make([]float64, n*n)

	simd.MatMulInto(dst, a, id, n, n, n)
	for i := range a {
		if dst[i] != a[i] {
			t.Fatalf("a*I: dst[%d] = %v, want %v", i, dst[i], a[i])
		}
	}
	simd.MatMulInto(dst, id, a, n, n, n)
	for i := range a {
		if dst[i] != a[i] {
			t.Fatalf("I*a: dst[%d] = %v, want %v", i, dst[i], a[i])
		}
	}
}

func BenchmarkMatMul(b *testing.B) {
	for _, n := range []int{16, 64, 128, 256, 512} {
		a := make([]float32, n*n)
		bb := make([]float32, n*n)
		dst := make([]float32, n*n)
		for i := range a {
			a[i] = float32(i%13) - 6
			bb[i] = float32(i%7) - 3
		}
		b.Run(fmt.Sprintf("f32/n=%d", n), func(b *testing.B) {
			// Two flops per multiply-add, n^3 of them.
			b.SetBytes(int64(2 * n * n * n))
			for b.Loop() {
				simd.MatMulInto(dst, a, bb, n, n, n)
			}
		})
	}
}

func BenchmarkGemv(b *testing.B) {
	for _, dim := range [][2]int{{256, 256}, {1024, 256}, {256, 1024}, {4096, 4096}} {
		m, k := dim[0], dim[1]
		a := make([]float32, m*k)
		x := make([]float32, k)
		dst := make([]float32, m)
		for i := range a {
			a[i] = float32(i%13) - 6
		}
		for i := range x {
			x[i] = float32(i%7) - 3
		}
		b.Run(fmt.Sprintf("f32/m=%d/k=%d", m, k), func(b *testing.B) {
			b.SetBytes(int64(m) * int64(k) * 4)
			for b.Loop() {
				simd.GemvInto(dst, a, x, m, k)
			}
		})
	}
}
