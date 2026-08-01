package simd

// Operations shaped for linear algebra rather than for slice arithmetic.
//
// These exist because a decomposition is not built out of matrix multiplies.
// LU, QR and Cholesky spend their time in rank-1 updates, Givens rotations and
// row swaps, so a library with a fast Gemm and nothing else leaves them at
// scalar speed. BLAS calls them ger, rot and swap.

// RankOneInto adds the outer product of two vectors to a matrix:
//
//	a[i*n+j] += alpha * x[i] * y[j]
//
// a is m×n row-major, x has m elements and y has n. This is the trailing-matrix
// update inside an LU decomposition, and BLAS calls it ger.
//
// Rows of a must be exactly n apart. There is no leading-dimension parameter,
// so a view into a wider matrix has to be copied out first — the six-argument
// limit on a kernel is the reason, and passing a stride nobody can prove is
// in-bounds is the alternative that was not taken.
//
// The row scale is computed once per row, so the result is (alpha*x[i])*y[j]
// rather than alpha*(x[i]*y[j]). Those differ in the last place, and the
// portable path does it the same way.
func RankOneInto[T Float](a, x, y []T, alpha T, m, n int) {
	ops[T]().RankOne(a, x, y, alpha, m, n)
}

// Rotate applies a Givens rotation to a pair of vectors:
//
//	x[i] = c*x[i] + s*y[i]
//	y[i] = c*y[i] - s*x[i]
//
// Both assignments use the original x[i]. This is BLAS rot, and it is what a QR
// factorisation and the bidiagonal phase of an SVD are made of.
//
// Both slices are written, over min(len(x), len(y)) elements. They must not
// overlap.
func Rotate[T Float](x, y []T, c, s T) { ops[T]().Rotate(x, y, c, s) }

// Swap exchanges the contents of two slices, over min(len(x), len(y)) elements.
//
// Trivial and worth having: pivoting a decomposition swaps rows, which makes
// this one of the most-called routines in gonum's linear algebra. The slices
// must not overlap.
func Swap[T Number](x, y []T) { ops[T]().Swap(x, y) }
