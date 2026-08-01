package ref

// The portable forms of the BLAS-shaped kernels. Every generated kernel is
// differential-tested against these, and they run wherever a kernel could not
// be generated or the input is below the dispatch threshold.

// RankOneFloat is a[i*n+j] += alpha*x[i]*y[j].
//
// The row scale is hoisted exactly as the kernel hoists it. Writing
// alpha*x[i]*y[j] in the inner loop would associate the multiplications
// differently and disagree with the kernel in the last place, which the
// differential suite would then report as a kernel bug.
func RankOneFloat[T Float](a, x, y []T, alpha T, m, n int) {
	if m < 0 || n < 0 || len(a) < m*n || len(x) < m || len(y) < n {
		return
	}
	for i := 0; i < m; i++ {
		s := alpha * x[i]
		row := a[i*n : i*n+n]
		for j := range row {
			// The conversion is load-bearing. Go fuses a multiply into an
			// adjacent add on arm64, riscv64, loong64 and ppc64 but not on
			// amd64, so without it this reference computes fma(s, y[j], row[j])
			// on four of the six architectures while the kernel — compiled with
			// -ffp-contract=off — always rounds the product first. They then
			// disagree in the last place, on those targets only. Third time
			// this trap has been paid for in this repository.
			row[j] += T(s * y[j])
		}
	}
}

// RotateFloat applies a Givens rotation to a pair of vectors, using the
// original x[i] in both assignments.
func RotateFloat[T Float](x, y []T, c, s T) {
	n := min(len(x), len(y))
	for i := 0; i < n; i++ {
		xi, yi := x[i], y[i]
		// Each product rounds before the add, for the same reason as RankOne.
		x[i] = T(c*xi) + T(s*yi)
		y[i] = T(c*yi) - T(s*xi)
	}
}

// SwapFloat exchanges two vectors.
func SwapFloat[T Float](x, y []T) {
	n := min(len(x), len(y))
	for i := 0; i < n; i++ {
		x[i], y[i] = y[i], x[i]
	}
}

// SwapInt is SwapFloat for the integer types.
func SwapInt[T Integer](x, y []T) {
	n := min(len(x), len(y))
	for i := 0; i < n; i++ {
		x[i], y[i] = y[i], x[i]
	}
}
