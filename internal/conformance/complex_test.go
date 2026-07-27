package conformance

// Conformance for the complex kernels.
//
// The interesting inputs here are not the ordinary ones. A complex multiply
// is four real products and two sums, so a zero component times an infinite
// one is reachable from perfectly ordinary operands, and a divide by the
// textbook formula overflows for values whose quotient is representable. Both
// are in the tables below.
//
// Divide is compared to a tolerance rather than bit for bit. It is the one
// operation here where the kernel and the reference are separately derived —
// both use Smith's method, but the reference works through float64 for the
// float32 case, which is a double rounding — so demanding identical bits
// would be demanding something the shapes cannot give.

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// awkwardC128 are the component pairs a complex kernel is most likely to get
// wrong: the zeros against the infinities, values whose squares overflow but
// whose quotient does not, and NaN in one component only.
var awkwardC128 = []complex128{
	0, complex(0, 0), complex(math.Copysign(0, -1), 0),
	1, 1i, -1, -1i,
	complex(3, 4), complex(-3, 4), complex(3, -4),
	complex(1e300, 1e300), complex(1e-300, 1e-300),
	complex(1e300, 1e-300), complex(1e-300, 1e300),
	complex(math.Inf(1), 0), complex(0, math.Inf(1)),
	complex(math.Inf(-1), 1), complex(1, math.Inf(-1)),
	complex(math.NaN(), 0), complex(0, math.NaN()),
	complex(math.MaxFloat64, math.MaxFloat64),
	complex(math.SmallestNonzeroFloat64, 1),
}

func genC128(n int, r *rand.Rand) []complex128 {
	out := make([]complex128, n)
	for i := range out {
		if i%3 == 0 {
			out[i] = awkwardC128[(i/3+int(r.Uint32()))%len(awkwardC128)]
			continue
		}
		out[i] = complex(r.NormFloat64()*10, r.NormFloat64()*10)
	}
	return out
}

func genC64(n int, r *rand.Rand) []complex64 {
	c := genC128(n, r)
	out := make([]complex64, n)
	for i := range out {
		out[i] = complex64(c[i])
	}
	return out
}

func sameC[C comparable](got, want []C) (int, bool) {
	for i := range got {
		g, w := any(got[i]), any(want[i])
		var gr, gi, wr, wi float64
		switch v := g.(type) {
		case complex64:
			gr, gi = float64(real(v)), float64(imag(v))
			u := w.(complex64)
			wr, wi = float64(real(u)), float64(imag(u))
		case complex128:
			gr, gi = real(v), imag(v)
			u := w.(complex128)
			wr, wi = real(u), imag(u)
		}
		if !sameFloatBits(gr, wr) || !sameFloatBits(gi, wi) {
			return i, false
		}
	}
	return 0, true
}

func sameFloatBits(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

// closeC compares to a relative tolerance, for the one operation whose two
// implementations round differently by construction.
func closeC(gr, gi, wr, wi, tol float64) bool {
	if math.IsNaN(gr) != math.IsNaN(wr) || math.IsNaN(gi) != math.IsNaN(wi) {
		return false
	}
	if math.IsNaN(gr) || math.IsNaN(gi) {
		return true
	}
	if math.IsInf(gr, 0) || math.IsInf(wr, 0) || math.IsInf(gi, 0) || math.IsInf(wi, 0) {
		return math.IsInf(gr, 0) == math.IsInf(wr, 0) &&
			math.IsInf(gi, 0) == math.IsInf(wi, 0)
	}
	scale := math.Abs(wr) + math.Abs(wi) + 1e-300
	return math.Abs(gr-wr)/scale <= tol && math.Abs(gi-wi)/scale <= tol
}

func checkComplex[C comparable](t *testing.T, tier, name string,
	got, want kernel.Complex[C], gen func(int, *rand.Rand) []C,
	toC128 func(C) complex128, divTol float64) {

	r := rand.New(rand.NewPCG(61, 62))
	for _, n := range apiLensC {
		a, b := gen(n, r), gen(n, r)
		g, w := make([]C, n), make([]C, n)

		bin := []struct {
			op        string
			got, want func(dst, x, y []C)
		}{
			{"Add", got.Add, want.Add},
			{"Sub", got.Sub, want.Sub},
			{"Mul", got.Mul, want.Mul},
		}
		for _, c := range bin {
			if c.got == nil || c.want == nil {
				continue
			}
			c.got(g, a, b)
			c.want(w, a, b)
			if i, ok := sameC(g, w); !ok {
				t.Fatalf("%s/%s.%s n=%d i=%d: got %v want %v (a=%v b=%v)",
					tier, name, c.op, n, i, g[i], w[i], a[i], b[i])
			}
		}

		// Divide is the one operation whose two implementations round
		// differently by construction — both use Smith's method, but the
		// reference works through float64 even for the float32 case, which is
		// a double rounding — so it is compared to a tolerance.
		if got.Div != nil && want.Div != nil {
			got.Div(g, a, b)
			want.Div(w, a, b)
			for i := range g {
				gv, wv := toC128(g[i]), toC128(w[i])
				if !closeC(real(gv), imag(gv), real(wv), imag(wv), divTol) {
					t.Fatalf("%s/%s.Div n=%d i=%d: got %v want %v (a=%v b=%v)",
						tier, name, n, i, g[i], w[i], a[i], b[i])
				}
			}
		}

		un := []struct {
			op        string
			got, want func(dst, x []C)
		}{
			{"Neg", got.Neg, want.Neg},
			{"Conj", got.Conj, want.Conj},
		}
		for _, c := range un {
			if c.got == nil || c.want == nil {
				continue
			}
			c.got(g, a)
			c.want(w, a)
			if i, ok := sameC(g, w); !ok {
				t.Fatalf("%s/%s.%s n=%d i=%d: got %v want %v (a=%v)",
					tier, name, c.op, n, i, g[i], w[i], a[i])
			}
		}

		red := []struct {
			op        string
			got, want func(x []C) C
		}{{"Sum", got.Sum, want.Sum}}
		for _, c := range red {
			if c.got == nil || c.want == nil {
				continue
			}
			gv, wv := c.got(a), c.want(a)
			if i, ok := sameC([]C{gv}, []C{wv}); !ok {
				_ = i
				t.Fatalf("%s/%s.%s n=%d: got %v want %v", tier, name, c.op, n, gv, wv)
			}
		}

		for _, c := range []struct {
			op        string
			got, want func(x, y []C) C
		}{
			{"Dot", got.Dot, want.Dot},
			{"DotConj", got.DotConj, want.DotConj},
		} {
			if c.got == nil || c.want == nil {
				continue
			}
			gv, wv := c.got(a, b), c.want(a, b)
			if _, ok := sameC([]C{gv}, []C{wv}); !ok {
				t.Fatalf("%s/%s.%s n=%d: got %v want %v", tier, name, c.op, n, gv, wv)
			}
		}
	}
}

var apiLensC = []int{0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 65, 127}

func TestComplexKernels(t *testing.T) {
	want := ref.Set()
	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			checkComplex(t, tier, "C64", got.C64, want.C64, genC64,
				func(c complex64) complex128 { return complex128(c) }, 1e-5)
			checkComplex(t, tier, "C128", got.C128, want.C128, genC128,
				func(c complex128) complex128 { return c }, 1e-12)
			checkComplexParts64(t, tier, got, want)
			checkComplexParts128(t, tier, got, want)
		})
	}
}

func checkComplexParts64(t *testing.T, tier string, got, want kernel.Set) {
	r := rand.New(rand.NewPCG(63, 64))
	for _, n := range apiLensC {
		a := genC64(n, r)
		g, w := make([]float32, n), make([]float32, n)
		for _, c := range []struct {
			op        string
			got, want func(dst []float32, x []complex64)
		}{
			{"Abs", got.C64Parts.Abs, want.C64Parts.Abs},
			{"Real", got.C64Parts.Real, want.C64Parts.Real},
			{"Imag", got.C64Parts.Imag, want.C64Parts.Imag},
		} {
			if c.got == nil || c.want == nil {
				continue
			}
			c.got(g, a)
			c.want(w, a)
			if i, ok := same(g, w); !ok {
				t.Fatalf("%s/C64Parts.%s n=%d i=%d: got %v want %v (a=%v)",
					tier, c.op, n, i, g[i], w[i], a[i])
			}
		}
		if got.C64Parts.Scale != nil {
			gs, ws := make([]complex64, n), make([]complex64, n)
			for _, s := range []float32{0, 1, -1, 2.5, float32(math.Inf(1))} {
				got.C64Parts.Scale(gs, a, s)
				want.C64Parts.Scale(ws, a, s)
				if i, ok := sameC(gs, ws); !ok {
					t.Fatalf("%s/C64Parts.Scale n=%d s=%v i=%d: got %v want %v",
						tier, n, s, i, gs[i], ws[i])
				}
			}
		}
		if got.C64Parts.FromParts != nil {
			re, im := make([]float32, n), make([]float32, n)
			for i := range re {
				re[i], im[i] = float32(i)-3, float32(i)*0.5
			}
			gs, ws := make([]complex64, n), make([]complex64, n)
			got.C64Parts.FromParts(gs, re, im)
			want.C64Parts.FromParts(ws, re, im)
			if i, ok := sameC(gs, ws); !ok {
				t.Fatalf("%s/C64Parts.FromParts n=%d i=%d: got %v want %v",
					tier, n, i, gs[i], ws[i])
			}
		}
	}
}

func checkComplexParts128(t *testing.T, tier string, got, want kernel.Set) {
	r := rand.New(rand.NewPCG(65, 66))
	for _, n := range apiLensC {
		a := genC128(n, r)
		g, w := make([]float64, n), make([]float64, n)
		for _, c := range []struct {
			op        string
			got, want func(dst []float64, x []complex128)
		}{
			{"Abs", got.C128Parts.Abs, want.C128Parts.Abs},
			{"Real", got.C128Parts.Real, want.C128Parts.Real},
			{"Imag", got.C128Parts.Imag, want.C128Parts.Imag},
		} {
			if c.got == nil || c.want == nil {
				continue
			}
			c.got(g, a)
			c.want(w, a)
			if i, ok := same(g, w); !ok {
				t.Fatalf("%s/C128Parts.%s n=%d i=%d: got %v want %v (a=%v)",
					tier, c.op, n, i, g[i], w[i], a[i])
			}
		}
		// Abs must also agree with the standard library's magnitude wherever
		// that is finite, which is an independent check on the scaling.
		if got.C128Parts.Abs != nil {
			got.C128Parts.Abs(g, a)
			for i := range g {
				w := cmplx.Abs(a[i])
				if math.IsNaN(w) || math.IsInf(w, 0) {
					continue
				}
				if math.Abs(g[i]-w) > 1e-12*math.Abs(w) {
					t.Fatalf("%s/C128Parts.Abs vs cmplx.Abs at %d: got %v want %v (a=%v)",
						tier, i, g[i], w, a[i])
				}
			}
		}
	}
}
