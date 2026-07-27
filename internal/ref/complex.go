package ref

import (
	"math"

	"github.com/sebishogun/simd/internal/kernel"
)

// Complex kernels, portably.
//
// Go stores a complex as its two components adjacent in memory, so these are
// written against the values rather than against the halves — the kernels
// read the same memory as a pair of reals per element and get the same
// answers, and stating the semantics in terms of Go's own complex arithmetic
// is what makes the two comparable.
//
// One place that is not true, and it is the reason Mul is written out rather
// than left to the language: Go's complex multiply is not simply
// (ac-bd) + (ad+bc)i. gc lowers it to exactly that, but the specification
// permits an implementation to handle infinities the way C99's Annex G does,
// where an infinite operand times a zero one gives an infinite result rather
// than a NaN. Writing the four products explicitly pins the kernels and the
// reference to the same rule.

type complexT interface{ ~complex64 | ~complex128 }

func cAdd[C complexT](dst, a, b []C) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func cSub[C complexT](dst, a, b []C) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

// cMul writes the four products out rather than using the language operator,
// so that the kernel and the reference agree on the infinity cases the Go
// specification leaves open.
func cMul[C complexT, R float](dst, a, b []C) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		ar, ai := R(real(complex128(a[i]))), R(imag(complex128(a[i])))
		br, bi := R(real(complex128(b[i]))), R(imag(complex128(b[i])))
		dst[i] = C(complex(float64(R(ar*br)-R(ai*bi)), float64(R(ar*bi)+R(ai*br))))
	}
}

// cDiv uses Smith's method: dividing through by the larger component keeps
// the intermediate products in range, where the textbook formula overflows
// for operands whose squares do not fit even though the quotient does.
func cDiv[C complexT, R float](dst, a, b []C) {
	n := min(len(dst), len(a), len(b))
	dst, a, b = dst[:n], a[:n], b[:n]
	for i := range dst {
		ar, ai := R(real(complex128(a[i]))), R(imag(complex128(a[i])))
		br, bi := R(real(complex128(b[i]))), R(imag(complex128(b[i])))
		var re, im R
		if absR(br) >= absR(bi) {
			r := bi / br
			d := br + R(r*bi)
			re = (ar + R(ai*r)) / d
			im = (ai - R(ar*r)) / d
		} else {
			r := br / bi
			d := bi + R(r*br)
			re = (R(ar*r) + ai) / d
			im = (R(ai*r) - ar) / d
		}
		dst[i] = C(complex(float64(re), float64(im)))
	}
}

func absR[R float](x R) R {
	if x < 0 {
		return -x
	}
	return x
}

func cNeg[C complexT](dst, a []C) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = -a[i]
	}
}

func cConj[C complexT, R float](dst, a []C) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		re, im := R(real(complex128(a[i]))), R(imag(complex128(a[i])))
		dst[i] = C(complex(float64(re), float64(-im)))
	}
}

func cScale[C complexT, R float](dst, a []C, s R) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		re, im := R(real(complex128(a[i]))), R(imag(complex128(a[i])))
		dst[i] = C(complex(float64(R(re*s)), float64(R(im*s))))
	}
}

// cAbs is the magnitude, computed through the larger component so that an
// intermediate cannot overflow for a representable answer. Same reasoning as
// Hypot, and it matters more here because both components are user data.
//
// Every step is in R rather than in float64, which matters for complex64:
// computing in double and rounding once at the end is a different answer from
// computing in single throughout, by an ULP, and the kernel does the latter.
// The reference has to model the kernel, not the other way round.
//
// sqrtR is the exception and is safe: the square root of a float32 computed in
// float64 and rounded is the correctly rounded float32 square root, because
// double has more than twice the precision needed.
//
// The comparison is "im > re" rather than "re > im" so that a NaN component
// survives. Both are false for NaN, and only this order leaves hi holding it.
func cAbs[C complexT, R float](dst []R, a []C) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		re := absR(R(real(complex128(a[i]))))
		im := absR(R(imag(complex128(a[i]))))
		hi, lo := re, im
		if im > re {
			hi, lo = im, re
		}
		t := lo / hi
		v := hi * sqrtR(1+R(t*t))
		if hi == 0 {
			v = 0
		}
		if math.IsInf(float64(re), 0) || math.IsInf(float64(im), 0) {
			v = R(math.Inf(1))
		}
		dst[i] = v
	}
}

func sqrtR[R float](x R) R { return R(math.Sqrt(float64(x))) }

func cReal[C complexT, R float](dst []R, a []C) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = R(real(complex128(a[i])))
	}
}

func cImag[C complexT, R float](dst []R, a []C) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = R(imag(complex128(a[i])))
	}
}

func cFromParts[C complexT, R float](dst []C, re, im []R) {
	n := min(len(dst), len(re), len(im))
	dst, re, im = dst[:n], re[:n], im[:n]
	for i := range dst {
		dst[i] = C(complex(float64(re[i]), float64(im[i])))
	}
}

// cSum accumulates into kernel.SumLanes independent lanes, component by
// component, for exactly the reason the real sum does: the answer must not
// change when the program moves to a machine with wider vectors.
func cSum[C complexT, R float](a []C) C {
	var accRe, accIm [kernel.SumLanes]R
	i := 0
	for ; i+kernel.SumLanes <= len(a); i += kernel.SumLanes {
		blk := a[i : i+kernel.SumLanes : i+kernel.SumLanes]
		for j := range kernel.SumLanes {
			accRe[j] += R(real(complex128(blk[j])))
			accIm[j] += R(imag(complex128(blk[j])))
		}
	}
	for j := 0; i < len(a); i, j = i+1, j+1 {
		accRe[j] += R(real(complex128(a[i])))
		accIm[j] += R(imag(complex128(a[i])))
	}
	re := kernel.CombineTree(&accRe)
	im := kernel.CombineTree(&accIm)
	return C(complex(float64(re), float64(im)))
}

// cDot is the bilinear product and cDotConj the Hermitian one. Both fold with
// the same lane discipline as cSum.
func cDot[C complexT, R float](a, b []C, conj bool) C {
	var accRe, accIm [kernel.SumLanes]R
	n := min(len(a), len(b))
	i := 0
	step := func(j, k int) {
		ar, ai := R(real(complex128(a[k]))), R(imag(complex128(a[k])))
		br, bi := R(real(complex128(b[k]))), R(imag(complex128(b[k])))
		if conj {
			ai = -ai
		}
		accRe[j] += R(ar*br) - R(ai*bi)
		accIm[j] += R(ar*bi) + R(ai*br)
	}
	for ; i+kernel.SumLanes <= n; i += kernel.SumLanes {
		for j := range kernel.SumLanes {
			step(j, i+j)
		}
	}
	for j := 0; i < n; i, j = i+1, j+1 {
		step(j, i)
	}
	re := kernel.CombineTree(&accRe)
	im := kernel.CombineTree(&accIm)
	return C(complex(float64(re), float64(im)))
}

// complexOps fills one Complex group with the portable implementations.
func complexOps[C complexT, R float](o *kernel.Complex[C]) {
	o.Add = cAdd[C]
	o.Sub = cSub[C]
	o.Mul = cMul[C, R]
	o.Div = cDiv[C, R]
	o.Neg = cNeg[C]
	o.Conj = cConj[C, R]
	o.Sum = cSum[C, R]
	o.Dot = func(a, b []C) C { return cDot[C, R](a, b, false) }
	o.DotConj = func(a, b []C) C { return cDot[C, R](a, b, true) }
}

func complexPartsOps[C complexT, R float](o *kernel.ComplexParts[C, R]) {
	o.Scale = cScale[C, R]
	o.Abs = cAbs[C, R]
	o.Real = cReal[C, R]
	o.Imag = cImag[C, R]
	o.FromParts = cFromParts[C, R]
}

// Exported entry points for generated code.

func CAdd[C complexT](dst, a, b []C) { cAdd(dst, a, b) }
func CSub[C complexT](dst, a, b []C) { cSub(dst, a, b) }
func CNeg[C complexT](dst, a []C)    { cNeg(dst, a) }

func CMul64(dst, a, b []complex64)   { cMul[complex64, float32](dst, a, b) }
func CMul128(dst, a, b []complex128) { cMul[complex128, float64](dst, a, b) }
func CDiv64(dst, a, b []complex64)   { cDiv[complex64, float32](dst, a, b) }
func CDiv128(dst, a, b []complex128) { cDiv[complex128, float64](dst, a, b) }

func CConj64(dst, a []complex64)   { cConj[complex64, float32](dst, a) }
func CConj128(dst, a []complex128) { cConj[complex128, float64](dst, a) }

func CScale64(dst, a []complex64, s float32)   { cScale[complex64, float32](dst, a, s) }
func CScale128(dst, a []complex128, s float64) { cScale[complex128, float64](dst, a, s) }

func CAbs64(dst []float32, a []complex64)   { cAbs[complex64, float32](dst, a) }
func CAbs128(dst []float64, a []complex128) { cAbs[complex128, float64](dst, a) }

func CReal64(dst []float32, a []complex64)   { cReal[complex64, float32](dst, a) }
func CReal128(dst []float64, a []complex128) { cReal[complex128, float64](dst, a) }
func CImag64(dst []float32, a []complex64)   { cImag[complex64, float32](dst, a) }
func CImag128(dst []float64, a []complex128) { cImag[complex128, float64](dst, a) }

func CFromParts64(dst []complex64, re, im []float32) {
	cFromParts[complex64, float32](dst, re, im)
}
func CFromParts128(dst []complex128, re, im []float64) {
	cFromParts[complex128, float64](dst, re, im)
}

func CSum64(a []complex64) complex64    { return cSum[complex64, float32](a) }
func CSum128(a []complex128) complex128 { return cSum[complex128, float64](a) }

func CDot64(a, b []complex64) complex64     { return cDot[complex64, float32](a, b, false) }
func CDot128(a, b []complex128) complex128  { return cDot[complex128, float64](a, b, false) }
func CDotConj64(a, b []complex64) complex64 { return cDot[complex64, float32](a, b, true) }
func CDotConj128(a, b []complex128) complex128 {
	return cDot[complex128, float64](a, b, true)
}
