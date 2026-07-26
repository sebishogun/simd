package simd

// Reductions. Each collapses a slice to a single value and writes nothing.
//
// Floating-point reductions use a fixed accumulation order that every
// instruction set reproduces exactly, so results do not change when the
// program moves to a machine with wider vectors. See the package
// documentation.

// Sum returns the sum of all elements, or zero for an empty slice.
//
// Integer summation wraps on overflow.
func Sum[T Number](a []T) T { return ops[T]().Sum(a) }

// Dot returns the dot product of a and b over min(len(a), len(b)) elements,
// or zero if either is empty.
//
// The multiplication and the addition round separately; they are not fused,
// which matches a plain scalar loop.
func Dot[T Number](a, b []T) T { return ops[T]().Dot(a, b) }

// SumSquares returns the sum of the squares of all elements.
//
// This is Dot(a, a), and is the un-rooted form of [Norm].
func SumSquares[T Number](a []T) T { return ops[T]().SumSquares(a) }

// L1Norm returns the sum of the absolute values of all elements, also called
// the taxicab or Manhattan norm.
func L1Norm[T Number](a []T) T { return ops[T]().L1Norm(a) }

// Norm returns the Euclidean length of a: the square root of the sum of
// squares. Also called the L2 norm.
func Norm[T Float](a []T) T { return ops[T]().Norm(a) }

// Min returns the smallest element.
//
// For floats this is the IEEE 754-2019 minimum: NaN propagates, and -0 is
// smaller than +0. It panics on an empty slice.
//
// For the elementwise minimum of two slices, see [Minimum].
func Min[T Number](a []T) T { return ops[T]().Min(a) }

// Max returns the largest element.
//
// For floats this is the IEEE 754-2019 maximum: NaN propagates, and +0 is
// larger than -0. It panics on an empty slice.
//
// For the elementwise maximum of two slices, see [Maximum].
func Max[T Number](a []T) T { return ops[T]().Max(a) }

// MinMax returns the smallest and largest elements in a single pass, which is
// cheaper than calling [Min] and [Max] separately because the data is read
// once. It panics on an empty slice.
func MinMax[T Number](a []T) (lo, hi T) { return ops[T]().MinMax(a) }

// ArgMin returns the index of the first smallest element.
// It panics on an empty slice.
func ArgMin[T Number](a []T) int { return ops[T]().ArgMin(a) }

// ArgMax returns the index of the first largest element.
// It panics on an empty slice.
func ArgMax[T Number](a []T) int { return ops[T]().ArgMax(a) }
