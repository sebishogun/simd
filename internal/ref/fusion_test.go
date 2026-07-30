package ref

// The reference must return the same bits on every architecture.
//
// That is not automatic, and getting it wrong is invisible on the machine most
// development happens on. Go's spec permits an implementation to fuse a
// multiply and a following add into one operation with a single rounding, and
// gc takes that licence on arm64, ppc64, s390x and riscv64 but not on amd64.
// A reference written the obvious way therefore computes a different Lerp on a
// Graviton than on a laptop — before any assembly is involved at all.
//
// The kernels are compiled with -ffp-contract=off and never fuse, so the
// reference must not either. Each multiply feeding an add in this package is
// wrapped in a conversion to the element type, which the spec says rounds and
// which therefore forbids the fusion.
//
// This test compares each such operation against the same arithmetic carried
// out one rounding at a time, through a function the compiler cannot see
// into. On a fusing architecture an unprotected reference fails here; on amd64
// it passes either way, which is exactly why the test exists rather than a
// comment.

import (
	"math"
	"math/rand/v2"
	"testing"
)

// stepwise computes a + b*c with two separate roundings. //go:noinline keeps
// the multiply and the add in different frames, where nothing can fuse them.
//
//go:noinline
func stepwiseMulAdd(a, b, c float32) float32 {
	p := mul32(b, c)
	return add32(a, p)
}

//go:noinline
func mul32(x, y float32) float32 { return x * y }

//go:noinline
func add32(x, y float32) float32 { return x + y }

func TestNoFusedMultiplyAdd(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	const n = 200000

	for i := range n {
		_ = i
		a := float32(r.NormFloat64() * 100)
		b := float32(r.NormFloat64() * 100)
		c := float32(r.NormFloat64() * 100)

		// AddScaled: dst = a + b*s
		var got, want [1]float32
		AddScaled(got[:], []float32{a}, []float32{b}, c)
		want[0] = stepwiseMulAdd(a, b, c)
		if math.Float32bits(got[0]) != math.Float32bits(want[0]) {
			t.Fatalf("AddScaled(%v, %v, %v) = %v, want %v — the multiply was fused "+
				"into the add, so this build does not agree with the kernels or with "+
				"the same code on amd64", a, b, c, got[0], want[0])
		}

		// Lerp: dst = a + (b-a)*t
		Lerp(got[:], []float32{a}, []float32{b}, c)
		want[0] = stepwiseMulAdd(a, sub32(b, a), c)
		if math.Float32bits(got[0]) != math.Float32bits(want[0]) {
			t.Fatalf("Lerp(%v, %v, %v) = %v, want %v — fused multiply-add",
				a, b, c, got[0], want[0])
		}

		// LayerNorm: dst = (x+shift)/denom * gamma + beta. The multiply and
		// the add at the end are the fusable pair, and this one shipped
		// written as a single expression — right on amd64, wrong on riscv64
		// and loong64, where Go fuses and the kernel does not.
		LayerNorm(got[:], []float32{a}, []float32{b}, []float32{c}, 0, 1)
		want[0] = stepwiseMulAdd(c, a, b)
		if math.Float32bits(got[0]) != math.Float32bits(want[0]) {
			t.Fatalf("LayerNorm(%v, %v, %v) = %v, want %v — fused multiply-add",
				a, b, c, got[0], want[0])
		}

		// Dot over a single element, which is the accumulator's multiply.
		d := DotFloat([]float32{a}, []float32{b})
		if math.Float32bits(d) != math.Float32bits(mul32(a, b)) {
			t.Fatalf("Dot(%v, %v) = %v, want %v", a, b, d, mul32(a, b))
		}

		// SparseDot is Dot with a gather in front of one operand, so it has
		// the same fusable multiply and needs the same conversion.
		//
		// Seventeen elements, not two. Element i accumulates into lane
		// i%SumLanes, so two elements land in different lanes and neither ever
		// reaches an add — the test would pass whether or not the reference
		// was protected. The seventeenth element is the first one to meet a
		// non-zero accumulator, which is where the multiply and the add become
		// a fusable pair.
		sv := make([]float32, 17)
		sidx := make([]int32, 17)
		sv[0], sv[16] = a, b
		sp := SparseDotFloat(sv, sidx, []float32{c})
		if w := stepwiseMulAdd(mul32(a, c), b, c); math.Float32bits(sp) != math.Float32bits(w) {
			t.Fatalf("SparseDot(%v, %v, %v) = %v, want %v — fused multiply-add",
				a, b, c, sp, w)
		}
	}
}

//go:noinline
func sub32(x, y float32) float32 { return x - y }
