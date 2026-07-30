package simd

// Sparse matrix-vector multiply, in the CSR layout every sparse library stores
// matrices in.
//
// # Why the row is the kernel and the matrix is not
//
// A whole SpMV needs five pointers — values, column indices, row pointers, the
// dense vector and the destination — plus their lengths. That is well past the
// six integer registers the SysV amd64 ABI passes arguments in, and a kernel
// that spills into the caller's frame is one this library's generator declines
// to emit. So the row loop stays in Go and [SparseDot] is the row.
//
// It is also where the arithmetic is. A row is a gather and a dot product; the
// loop around it is bookkeeping, and bookkeeping in Go costs one call per row.
// That trade is fine for the rows sparse matrices actually have — a finite
// element or graph adjacency row is tens to hundreds of nonzeros — and it is
// the wrong trade for a matrix whose rows are two elements long, where the
// call dominates and a plain Go loop wins.
//
// # The accumulation is the library's, not the hardware's
//
// [SparseDot] is a floating-point reduction and obeys the same rule every other
// one does: exactly kernel.SumLanes accumulators combined by a fixed tree, so a
// 128-bit and a 512-bit machine return the same bits. Sparse solvers are
// iterative and a residual that depends on which machine ran it is a debugging
// problem that lasts for weeks.
//
// # How much this actually buys, which is not much
//
// It is gather-bound, and a hardware gather is barely faster than the scalar
// loads it replaces on the machine this was measured on. Against a
// *well-written* scalar loop — sixteen accumulators, the same tree, no call
// overhead — on a Zen 5, minimum of three runs, nanoseconds per row:
//
//	nonzeros    scalar    sse2     avx2    avx512
//	      16     14.1     13.1     12.8     14.1
//	      64     34.2     33.3     30.4     31.6
//	     256    120.3    121.6    110.7    109.7
//	    4096     1935     1824     1711     1656
//
// So roughly 1.1x, reaching 1.17x on a long row, and never a loss. That is
// worth having and it is not a headline. The reason it is small is worth
// stating too: the gather is the bottleneck, not the arithmetic, and no
// arrangement of the multiply-add changes that.
//
// Against a *naive* scalar loop — one accumulator, so every multiply-add waits
// on the last — the gap is much larger, but that is a comparison against a
// dependency chain rather than against the gather, and quoting it would be
// measuring the wrong thing.

// SparseDot returns the sum of v[i] * x[idx[i]] — one row of a sparse
// matrix-vector product.
//
// v and idx are the row's values and column indices and must be the same
// length; the shorter bounds the work. x is the dense vector.
//
// An index outside x contributes nothing rather than panicking, the same
// contract [GatherInto] has: these indices are usually computed, and a stray
// one should not take the process down. Check them yourself if you need
// strictness.
func SparseDot[T Float](v []T, idx []int32, x []T) T {
	return ops[T]().SparseDot(v, idx, x)
}

// SpMVInto computes dst = A*x for a matrix A in compressed sparse row form.
//
// values and colIdx hold the nonzeros in row-major order; rowPtr has one entry
// per row plus a final total, so row r occupies values[rowPtr[r]:rowPtr[r+1]].
// dst must have one entry per row — len(rowPtr)-1 of them.
//
// This is the loop [SparseDot] is meant for, written out so callers do not have
// to get the row slicing right. It allocates nothing and it is not parallel:
// the rows are independent, so a caller who wants goroutines should split
// rowPtr and call this on each piece.
func SpMVInto[T Float](dst []T, values []T, colIdx []int32, rowPtr []int32, x []T) {
	if len(rowPtr) < 2 {
		return
	}
	n := min(len(dst), len(rowPtr)-1)
	for r := range n {
		lo, hi := int(rowPtr[r]), int(rowPtr[r+1])
		// Clamped rather than trusted: a malformed rowPtr is a data error, and
		// the alternative is an out-of-range slice expression deep inside a
		// loop with no useful message.
		lo = max(lo, 0)
		hi = min(hi, len(values), len(colIdx))
		if hi <= lo {
			dst[r] = 0
			continue
		}
		dst[r] = SparseDot(values[lo:hi], colIdx[lo:hi], x)
	}
}
