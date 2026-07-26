package simd

import "math"

// Common whole-task operations, built from the primitives in this package and
// therefore accelerated by the same backends. They are here so the everyday
// jobs — normalize this vector, how far apart are these two, standardize this
// column — are one call rather than an assembled sequence, and so that they
// are done in as few passes over memory as possible.
//
// Like everything else here, none of them allocate.

// ---------- descriptive statistics ----------

// Mean returns the arithmetic mean, or zero for an empty slice.
func Mean[T Float](a []T) T {
	if len(a) == 0 {
		return 0
	}
	return Sum(a) / T(len(a))
}

// Variance returns the population variance: the mean of the squared deviations
// from the mean. It returns zero for a slice shorter than two elements.
//
// It uses two passes rather than the single-pass sum-of-squares identity,
// because that identity loses most of its significant digits when the variance
// is small relative to the mean.
func Variance[T Float](a []T) T {
	if len(a) < 2 {
		return 0
	}
	return ops[T]().SumSqDev(a, Mean(a)) / T(len(a))
}

// SampleVariance returns the sample variance, dividing by n-1 rather than n.
// It returns zero for a slice shorter than two elements.
func SampleVariance[T Float](a []T) T {
	if len(a) < 2 {
		return 0
	}
	return ops[T]().SumSqDev(a, Mean(a)) / T(len(a)-1)
}

// StdDev returns the population standard deviation, the square root of
// [Variance].
func StdDev[T Float](a []T) T { return T(math.Sqrt(float64(Variance(a)))) }

// SampleStdDev returns the sample standard deviation, the square root of
// [SampleVariance].
func SampleStdDev[T Float](a []T) T { return T(math.Sqrt(float64(SampleVariance(a)))) }

// ---------- distance and similarity ----------

// Distance returns the Euclidean distance between a and b over
// min(len(a), len(b)) elements.
func Distance[T Float](a, b []T) T {
	return T(math.Sqrt(float64(ops[T]().SumSqDiff(a, b))))
}

// SquaredDistance returns the squared Euclidean distance between a and b.
//
// Prefer it over [Distance] when comparing distances to each other, since the
// square root changes nothing about the ordering and costs time.
func SquaredDistance[T Number](a, b []T) T { return ops[T]().SumSqDiff(a, b) }

// ManhattanDistance returns the sum of the absolute differences between a
// and b, also called the L1 or taxicab distance.
func ManhattanDistance[T Number](a, b []T) T { return ops[T]().L1Diff(a, b) }

// CosineSimilarity returns the cosine of the angle between a and b: their dot
// product divided by the product of their lengths.
//
// The result lies in [-1, 1], where 1 means the vectors point the same way.
// If either vector is all zeros the angle is undefined and the result is 0.
func CosineSimilarity[T Float](a, b []T) T {
	na, nb := Norm(a), Norm(b)
	if na == 0 || nb == 0 {
		return 0
	}
	return Dot(a, b) / (na * nb)
}

// ---------- rescaling, in place ----------

// Normalize scales a to unit Euclidean length. A vector of all zeros is left
// unchanged, since it has no direction to preserve.
func Normalize[T Float](a []T) {
	if n := Norm(a); n != 0 {
		DivScalar(a, n)
	}
}

// Standardize rescales a to zero mean and unit standard deviation, the
// transform usually called a z-score. If every element is identical the
// standard deviation is zero and a is set to all zeros.
func Standardize[T Float](a []T) {
	m := Mean(a)
	sd := T(math.Sqrt(float64(ops[T]().SumSqDev(a, m) / T(len(a)))))
	if sd == 0 {
		Zero(a)
		return
	}
	SubScalar(a, m)
	DivScalar(a, sd)
}

// Rescale maps a linearly onto the range [lo, hi], so its smallest element
// becomes lo and its largest becomes hi. If every element is identical there
// is no range to map and a is set to lo.
//
// Rescale(a, 0, 1) is the common min-max normalization.
func Rescale[T Float](a []T, lo, hi T) {
	if len(a) == 0 {
		return
	}
	curLo, curHi := MinMax(a)
	span := curHi - curLo
	if span == 0 {
		Fill(a, lo)
		return
	}
	AddScalar(a, -curLo)
	Scale(a, (hi-lo)/span)
	AddScalar(a, lo)
}
