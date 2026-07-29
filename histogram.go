package simd

// Histogram and bincount.
//
// # Why these have no kernel
//
// Every other operation in this package is a map or a reduction: each output
// depends on one input lane, or all lanes fold into one accumulator. A
// histogram is neither. It is a scatter with an increment, and two lanes in
// the same vector may target the same bin — so the vector unit cannot add one
// to eight counters, because "eight counters" might be the same counter eight
// times.
//
// AVX-512 has an instruction for exactly this, vpconflictd, which reports
// which earlier lanes collide with each lane so the increments serialise only
// where they must. Nothing else has it: not AVX2, not NEON, not SVE2, not RVV,
// not VSX. A kernel would be one tier of one architecture and portable Go on
// the other eight, and it would still lose whenever the data has few distinct
// values, because then every lane conflicts and the instruction degenerates
// into the scalar loop it replaced.
//
// # And why the obvious trick does not help either
//
// The scalar loop looks like it should be limited by the store-to-load
// dependency on counts[k]: the increment cannot start until the previous
// increment to the same address retires, and with skewed data that is the same
// address over and over. The standard remedy is several private tables summed
// at the end, so consecutive increments go to different addresses.
//
// That was implemented, measured, and removed. int32 on Zen 5, four private
// tables against the plain loop:
//
//	                          four tables   plain loop
//	n=4096    16 bins uniform     2.40µs       2.14µs
//	n=65536  256 bins uniform     37.7µs       23.5µs
//	n=1M      16 bins skewed       621µs        485µs
//	n=1M     256 bins uniform      587µs        377µs
//
// Slower everywhere, by 22% to 39%, including on the skewed data it was meant
// to rescue. The dependency it targets is not the bottleneck — the core covers
// that latency out of order — and four times the table traffic, the clearing,
// and the combining pass are all real. So the simple loop is what ships, and
// the measurement is recorded here so nobody reinvents the trick.
//
// The index computation for a uniform-bin histogram is vectorisable, and for
// float input it is most of the arithmetic. That remains open: it needs a
// kernel writing indices to a scratch slice, with the accumulation still
// scalar afterwards.

// BincountInto counts occurrences of each value in a, adding the count of
// value v to counts[v]. Values outside [0, len(counts)) are skipped.
//
// counts is not zeroed first, so repeated calls accumulate. Clear it yourself
// if that is not what you want:
//
//	clear(counts)
//	simd.BincountInto(counts, a)
func BincountInto[T Integer](counts []int32, a []T) {
	nb := int64(len(counts))
	if nb == 0 {
		return
	}
	for _, v := range a {
		if k := int64(v); k >= 0 && k < nb {
			counts[k]++
		}
	}
}

// Bincount is [BincountInto] allocating the counts. n is the number of bins.
func Bincount[T Integer](a []T, n int) []int32 {
	counts := make([]int32, n)
	BincountInto(counts, a)
	return counts
}

// HistogramInto counts the elements of a falling into len(counts) equal-width
// bins spanning [lo, hi), adding to counts[i] the number that land in bin i.
//
// Values below lo or at or above hi are skipped, and so is a NaN, which
// compares false against both bounds. The range is half-open at the top, which
// is numpy's default and the usual convention, so a value exactly equal to hi
// is not counted.
//
// counts is not zeroed first, so repeated calls accumulate. It does nothing if
// hi is not greater than lo.
func HistogramInto[T Number](counts []int32, a []T, lo, hi T) {
	nb := len(counts)
	if nb == 0 || !(hi > lo) {
		return
	}
	// The scale is computed once, in float64, so the division does not happen
	// per element and so an integer element type does not truncate it to zero.
	scale := float64(nb) / (float64(hi) - float64(lo))
	flo := float64(lo)
	for _, v := range a {
		if !(v >= lo) || !(v < hi) {
			continue
		}
		k := int((float64(v) - flo) * scale)
		if k >= nb {
			// Guards the rounding at the top edge: a value just under hi can
			// scale to exactly nb.
			k = nb - 1
		}
		counts[k]++
	}
}

// Histogram is [HistogramInto] allocating the counts.
func Histogram[T Number](a []T, n int, lo, hi T) []int32 {
	counts := make([]int32, n)
	HistogramInto(counts, a, lo, hi)
	return counts
}
