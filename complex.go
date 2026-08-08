package simd

// Complex arithmetic.
//
// The same conventions as everywhere else: the plain name works in place on
// its first argument, the Into suffix takes a destination, every function
// processes min(len(...)) elements, and nothing allocates.
//
// What is not here is as deliberate as what is. There is no ordering on the
// complex numbers, so there is no Minimum, no Less, no sort and no argmin —
// offering them would mean inventing an order, and every caller who wanted
// one would want a different one. Compare magnitudes with [AbsComplex] and
// then use the real functions.

import "github.com/sebishogun/simd/internal/kernel"

// Complex is the constraint for the complex element types.
type Complex interface{ ~complex64 | ~complex128 }

// complexOps returns the complex kernel group for C, and complexParts the
// half of it that also names the real component type.
//
// Both are one type switch on a zero value, like ops, so the dispatch costs a
// comparison rather than a map lookup or an interface call. Converting a
// pointer to any does not allocate — the pointer fits in the interface's data
// word — which is what keeps these off the heap.
func complexOps[C Complex]() *kernel.Complex[C] {
	var zero C
	switch any(zero).(type) {
	case complex64:
		return any(&refBase.C64).(*kernel.Complex[C])
	case complex128:
		return any(&refBase.C128).(*kernel.Complex[C])
	}
	panic("simd: unsupported complex type")
}

// complexParts needs both type parameters, and both are inferable from the
// arguments of every function that calls it: the destination gives the real
// type and the source the complex one. A mismatched pair — asking for the
// magnitude of a []complex128 into a []float32 — fails the assertion here
// rather than reading the wrong memory.
func complexParts[C Complex, R Float]() *kernel.ComplexParts[C, R] {
	var zero C
	switch any(zero).(type) {
	case complex64:
		if p, ok := any(&refBase.C64Parts).(*kernel.ComplexParts[C, R]); ok {
			return p
		}
	case complex128:
		if p, ok := any(&refBase.C128Parts).(*kernel.ComplexParts[C, R]); ok {
			return p
		}
	}
	panic("simd: complex and real element types do not match")
}

// ---------- in place ----------

// AddComplex adds b into a elementwise: a[i] += b[i].
//
// It processes min(len(a), len(b)) elements and allocates nothing.
// Use [AddComplexInto] to write the result elsewhere.
func AddComplex[C Complex](a, b []C) { AddComplexInto(a, a, b) }

// SubComplex subtracts b from a elementwise: a[i] -= b[i].
func SubComplex[C Complex](a, b []C) { SubComplexInto(a, a, b) }

// MulComplex multiplies a by b elementwise: a[i] *= b[i].
func MulComplex[C Complex](a, b []C) { MulComplexInto(a, a, b) }

// DivComplex divides a by b elementwise: a[i] /= b[i].
//
// The quotient is formed by Smith's method, which divides through by the
// larger component first, so an intermediate cannot overflow for a result
// that is representable.
func DivComplex[C Complex](a, b []C) { DivComplexInto(a, a, b) }

// NegComplex negates a in place.
func NegComplex[C Complex](a []C) { NegComplexInto(a, a) }

// ConjComplex replaces each element with its complex conjugate.
func ConjComplex[C Complex](a []C) { ConjComplexInto(a, a) }

// ---------- with a destination ----------

// AddComplexInto sets dst[i] = a[i] + b[i]. dst may alias a or b.
func AddComplexInto[C Complex](dst, a, b []C) { complexOps[C]().Add(dst, a, b) }

// SubComplexInto sets dst[i] = a[i] - b[i]. dst may alias a or b.
func SubComplexInto[C Complex](dst, a, b []C) { complexOps[C]().Sub(dst, a, b) }

// MulComplexInto sets dst[i] = a[i] * b[i]. dst may alias a or b.
func MulComplexInto[C Complex](dst, a, b []C) { complexOps[C]().Mul(dst, a, b) }

// DivComplexInto sets dst[i] = a[i] / b[i]. dst may alias a or b.
func DivComplexInto[C Complex](dst, a, b []C) { complexOps[C]().Div(dst, a, b) }

// NegComplexInto sets dst[i] = -a[i]. dst may alias a.
func NegComplexInto[C Complex](dst, a []C) { complexOps[C]().Neg(dst, a) }

// ConjComplexInto sets dst[i] to the conjugate of a[i]. dst may alias a.
func ConjComplexInto[C Complex](dst, a []C) { complexOps[C]().Conj(dst, a) }

// ---------- reductions ----------

// SumComplex returns the sum of a.
//
// Both components accumulate into the same fixed number of lanes the real
// [Sum] uses, so the answer does not change with the machine's vector width.
func SumComplex[C Complex](a []C) C { return complexOps[C]().Sum(a) }

// DotComplex returns the bilinear product, the sum of a[i]*b[i].
//
// This is not the inner product of a complex vector space; see
// [DotComplexConj] for that one. Both are offered because both are wanted and
// neither is obviously what "dot" should mean.
func DotComplex[C Complex](a, b []C) C { return complexOps[C]().Dot(a, b) }

// DotComplexConj returns the Hermitian inner product, the sum of
// conj(a[i])*b[i]. This is the one that makes DotComplexConj(a, a) the
// squared norm, and the one linear algebra usually means.
func DotComplexConj[C Complex](a, b []C) C { return complexOps[C]().DotConj(a, b) }

// ---------- crossing between complex and real ----------

// ScaleComplex multiplies a by the real s in place.
func ScaleComplex[C Complex, R Float](a []C, s R) { ScaleComplexInto(a, a, s) }

// ScaleComplexInto sets dst[i] = a[i] * s for a real s. dst may alias a.
//
// Scaling by a real is four multiplies cheaper than a full complex product,
// which is why it is offered separately rather than left to MulComplex with a
// slice of reals.
func ScaleComplexInto[C Complex, R Float](dst, a []C, s R) {
	complexParts[C, R]().Scale(dst, a, s)
}

// AbsComplexInto writes the magnitude of each element of a into dst.
//
// The magnitude is formed through the larger component, so an intermediate
// cannot overflow for a result that is representable — the same reasoning as
// [Hypot], and it matters more here because both components are caller data.
func AbsComplexInto[R Float, C Complex](dst []R, a []C) {
	complexParts[C, R]().Abs(dst, a)
}

// RealInto writes the real component of each element of a into dst.
func RealInto[R Float, C Complex](dst []R, a []C) { complexParts[C, R]().Real(dst, a) }

// ImagInto writes the imaginary component of each element of a into dst.
func ImagInto[R Float, C Complex](dst []R, a []C) { complexParts[C, R]().Imag(dst, a) }

// FromPartsInto assembles dst[i] from re[i] and im[i].
func FromPartsInto[C Complex, R Float](dst []C, re, im []R) {
	complexParts[C, R]().FromParts(dst, re, im)
}
