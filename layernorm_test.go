package simd_test

import (
	"math"
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

// LayerNorm was SubScalar followed by DivScalar and is now one fused kernel.
// Fusing two passes is only allowed to change the number of passes, so this
// checks the result against what the old composition produced, bit for bit.
//
// The trap it guards is specific: the fused form is (a-m)/sd, and writing it
// as (a-m)*(1/sd) would be faster and would disagree in the last bit for most
// inputs. kernel.Ops says the same thing about DivScalar for the same reason.
func TestLayerNormMatchesTheComposition(t *testing.T) {
	for _, n := range []int{1, 2, 15, 16, 17, 63, 65, 1000, 4096} {
		r := rand.New(rand.NewPCG(11, uint64(n)))
		a := make([]float64, n)
		for i := range a {
			a[i] = r.NormFloat64()*17 + 3
		}

		got := append([]float64(nil), a...)
		simd.LayerNorm(got, 1e-5)

		// The old body, written out.
		want := append([]float64(nil), a...)
		m := simd.Mean(want)
		v := simd.Variance(want)
		if n < 2 {
			v = 0 // Variance is defined as zero below two elements.
		}
		simd.SubScalar(want, m)
		simd.DivScalar(want, math.Sqrt(v+1e-5))

		for i := range got {
			if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
				t.Fatalf("n=%d i=%d: fused %v (%#016x) != composed %v (%#016x)",
					n, i, got[i], math.Float64bits(got[i]),
					want[i], math.Float64bits(want[i]))
			}
		}
	}
}

// The affine form has to agree with normalizing and then applying gamma and
// beta by hand, which is what a caller would write without it.
func TestLayerNormInto(t *testing.T) {
	for _, n := range []int{1, 16, 17, 128, 1024} {
		r := rand.New(rand.NewPCG(13, uint64(n)))
		a := make([]float32, n)
		gamma := make([]float32, n)
		beta := make([]float32, n)
		for i := range a {
			a[i] = float32(r.NormFloat64()) * 5
			gamma[i] = float32(r.NormFloat64())
			beta[i] = float32(r.NormFloat64())
		}

		dst := make([]float32, n)
		simd.LayerNormInto(dst, a, gamma, beta, 1e-5)

		want := append([]float32(nil), a...)
		simd.LayerNorm(want, 1e-5)
		for i := range want {
			// float32() around the multiply for the same reason ref does it:
			// Go fuses a multiply-add into an FMA on riscv64, loong64, arm64
			// and ppc64, the kernels never fuse, and without the conversion
			// this expectation would be right on amd64 and wrong elsewhere.
			want[i] = float32(want[i]*gamma[i]) + beta[i]
		}

		for i := range dst {
			if math.Float32bits(dst[i]) != math.Float32bits(want[i]) {
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], want[i])
			}
		}
	}
}

// Normalizing should actually normalize: mean zero, unit variance. This is the
// property a caller depends on, as opposed to the identity above which is
// about not having changed anything.
func TestLayerNormNormalizes(t *testing.T) {
	r := rand.New(rand.NewPCG(17, 19))
	a := make([]float64, 10000)
	for i := range a {
		a[i] = r.NormFloat64()*100 + 50
	}
	simd.LayerNorm(a, 1e-12)

	if m := simd.Mean(a); math.Abs(m) > 1e-9 {
		t.Errorf("mean after LayerNorm = %v, want ~0", m)
	}
	if sd := simd.StdDev(a); math.Abs(sd-1) > 1e-6 {
		t.Errorf("stddev after LayerNorm = %v, want ~1", sd)
	}
}

// gamma=1, beta=0 must reduce to plain LayerNorm — the identity case a model
// starts training from.
func TestLayerNormIntoIdentityParams(t *testing.T) {
	const n = 512
	r := rand.New(rand.NewPCG(23, 29))
	a := make([]float64, n)
	gamma := make([]float64, n)
	beta := make([]float64, n)
	for i := range a {
		a[i] = r.NormFloat64()
		gamma[i] = 1
	}
	dst := make([]float64, n)
	simd.LayerNormInto(dst, a, gamma, beta, 1e-5)

	want := append([]float64(nil), a...)
	simd.LayerNorm(want, 1e-5)
	for i := range dst {
		if math.Float64bits(dst[i]) != math.Float64bits(want[i]) {
			t.Fatalf("i=%d: gamma=1 beta=0 gave %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestLayerNormNoAlloc(t *testing.T) {
	a := make([]float32, 4096)
	g := make([]float32, 4096)
	b := make([]float32, 4096)
	d := make([]float32, 4096)
	if n := testing.AllocsPerRun(50, func() { simd.LayerNormInto(d, a, g, b, 1e-5) }); n != 0 {
		t.Errorf("LayerNormInto allocated %v times per run, want 0", n)
	}
}
