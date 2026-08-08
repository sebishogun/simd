package simd

// Run-length encoding, split the way the hardware wants it.
//
// RLE is two operations with opposite characters. Finding where the runs
// start is a comparison against the previous element — elementwise, no
// dependence between iterations, and it vectorizes completely. Writing one
// (value, length) pair per run has a data-dependent output position, which is
// a serial prefix that no amount of shuffling fixes.
//
// So the split is not an implementation detail, it is the design: the first
// half is a kernel, the second is a Go loop over the far smaller set of
// positions the first half found. That is the same two-phase shape
// docs/tutorial.md argues for with IndexAll and ParseFloats, and on a run-heavy
// column it touches each element once in vector code and then once per run.

// RunStartsInto marks every element of a that begins a run of equal values:
// dst[0] is true, and dst[i] is true when a[i] differs from a[i-1].
//
// This is the vector half of [RunLengthEncodeInt32] and it is exported on its
// own because the mask is useful by itself — feed it to [CompressInto] to keep
// one representative per run, or count it to find how many distinct runs a
// column has before deciding whether encoding is worth it.
//
// It writes min(len(dst), len(a)) entries and allocates nothing.
func RunStartsInto(dst []bool, a []int32) { tblBytesRunStartsI32[tierIdx](dst, a) }

// RunStartsInt64Into is [RunStartsInto] for int64.
func RunStartsInt64Into(dst []bool, a []int64) { tblBytesRunStartsI64[tierIdx](dst, a) }

// RunStartsBytesInto is [RunStartsInto] for bytes.
func RunStartsBytesInto(dst []bool, a []byte) { tblBytesRunStartsU8[tierIdx](dst, a) }

// RunLengthEncodeInt32 writes the runs of a into values and lengths, returning
// how many runs there were.
//
// It needs a scratch []bool at least as long as a — passed in rather than
// allocated, like every other operation here that needs working space. The
// run-start mask is computed into it by the kernel and then walked once.
//
//	scratch := make([]bool, len(col))
//	vals := make([]int32, len(col))
//	lens := make([]int32, len(col))
//	n := simd.RunLengthEncodeInt32(vals, lens, col, scratch)
//	vals, lens = vals[:n], lens[:n]
//
// values and lengths must each have room for the number of runs, which in the
// worst case — no two adjacent elements equal — is len(a). It stops early if
// they are shorter, returning the number of runs written, so a caller who
// knows the data is run-heavy can size them optimistically and check.
//
// It allocates nothing.
func RunLengthEncodeInt32(values, lengths, a []int32, scratch []bool) int {
	n := min(len(a), len(scratch))
	if n == 0 {
		return 0
	}
	tblBytesRunStartsI32[tierIdx](scratch[:n], a[:n])

	runs := 0
	start := 0
	for i := 1; i <= n; i++ {
		// A run ends at i when i is past the end or begins a new run.
		if i < n && !scratch[i] {
			continue
		}
		if runs >= len(values) || runs >= len(lengths) {
			return runs
		}
		values[runs] = a[start]
		lengths[runs] = int32(i - start)
		runs++
		start = i
	}
	return runs
}

// RunLengthDecodeInt32 expands runs back into dst, returning how many elements
// it wrote.
//
// The earlier form of this comment argued expansion is a serial prefix
// like ExpandInto and left it a plain Go loop. That analogy holds for
// per-element masks and not for runs: a run needs one broadcast and then
// wide stores to its own end, so the serial part is one addition per run
// rather than one per element. Kernel-backed since it was measured.
//
// It stops when dst is full and allocates nothing.
func RunLengthDecodeInt32(dst, values, lengths []int32) int {
	return tblBytesRLEDecodeInt32[tierIdx](dst, values, lengths)
}
