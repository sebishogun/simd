package simd

import "unsafe"

// Packed-B matrix multiplication.
//
// [MatMulInto] holds ~125 GFLOP/s while its operands fit L2 and fell to 34
// once they did not — measured at 512x512 float64, where each operand is 2
// MiB. The microkernel reads a column strip of B with stride n, touching k
// scattered cache lines per tile. Packing that strip contiguous is the fix,
// it is the arrangement OpenBLAS and BLIS use, and it is data *movement*, not
// reassociation: the multiply consumes the same values in the same p-ascending
// order, so results are bit-identical to [MatMulInto] and the package's
// contract holds. Measured through the cliff: 36.6 → 85.6 GF/s at 512,
// 37.3 → 83.0 at 1024.
//
// The packed layout's tile width is fixed per element type — sixteen floats,
// eight doubles — on every tier and in the portable reference, so any mix of
// accelerated and portable pack/multiply agrees. bp[t*(k*W) + p*W + v] holds
// b[p*n + t*W + v], zero-padded past n.

// gemmPackW is the packed tile width for T: 16 for float32, 8 for float64.
func gemmPackW[T Float]() int {
	var z T
	if unsafe.Sizeof(z) == 4 {
		return 16
	}
	return 8
}

// GemmPackLen returns the scratch length [PackBInto] needs for a k-by-n B.
func GemmPackLen[T Float](k, n int) int {
	if k <= 0 || n <= 0 {
		return 0
	}
	w := gemmPackW[T]()
	return ((n + w - 1) / w) * k * w
}

// PackBInto packs the k-by-n row-major matrix b into bp for [MatMulIntoPacked].
// bp must be at least [GemmPackLen](k, n) long; it does nothing otherwise.
//
// Packing costs one pass over B. A caller multiplying several A matrices
// against one B — the classifier-weights shape — packs once:
//
//	bp := make([]float64, simd.GemmPackLen[float64](k, n))
//	simd.PackBInto(bp, weights, k, n)
//	for _, batch := range batches {
//	    simd.MatMulIntoPacked(out, batch, bp, m, k, n)
//	}
func PackBInto[T Float](bp, b []T, k, n int) { ops[T]().GemmPackB(bp, b, k, n) }

// MatMulIntoPacked multiplies the m-by-k matrix a against a B previously
// packed by [PackBInto], writing the m-by-n result to dst. dst is zeroed
// first. It does nothing if any slice is too short for the stated shape.
//
// Bit-identical to [MatMulInto] of the same operands, by construction.
func MatMulIntoPacked[T Float](dst, a, bp []T, m, k, n int) {
	ops[T]().MatMulPk(dst, a, bp, m, k, n)
}

// gemmPackCliff is the B-operand footprint in bytes past which packing wins.
//
// Measured on float64, Zen 5: at 256x256 (512 KiB per operand, everything in
// L2) packing costs 12%; at 512x512 (2 MiB) it is 2.3x ahead. The crossover
// is where B stops fitting alongside A and the destination — around 1 MiB on
// this cache — and the constant is set there. Below it MatMulIntoScratch
// takes the plain path, so small multiplies never pay for a pack.
const gemmPackCliff = 1 << 20

// MatMulIntoScratch is [MatMulInto] with caller-supplied working space,
// switching to the packed algorithm when B is large enough that packing wins.
//
// scratch needs [GemmPackLen](k, n) elements for the packed path; a shorter
// one — including nil — falls back to the plain kernel, costing speed and
// never correctness. Results are bit-identical on either path.
func MatMulIntoScratch[T Float](dst, a, b, scratch []T, m, k, n int) {
	var z T
	if k > 0 && n > 0 && int(unsafe.Sizeof(z))*k*n >= gemmPackCliff &&
		len(scratch) >= GemmPackLen[T](k, n) {
		bp := scratch[:GemmPackLen[T](k, n)]
		PackBInto(bp, b, k, n)
		MatMulIntoPacked(dst, a, bp, m, k, n)
		return
	}
	MatMulInto(dst, a, b, m, k, n)
}
