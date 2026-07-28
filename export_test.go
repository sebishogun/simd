package simd

// Constants the external test package needs in order to generate inputs that
// straddle a dispatch threshold. A test that never crosses one silently tests
// only the fallback: every Median and Quantile test in this package ran at 70
// elements, well under SelectCutoffForTest, so none of them reached the
// quickselect built on the partition kernel until one was written that did.

// SelectMinLenForTest is the length at or above which Median and Quantile use
// the accelerated partition.
const SelectMinLenForTest = selectMinLen

// SortCutoffForTest is the equivalent threshold for Sort.
const SortCutoffForTest = sortCutoff
