package ref

import "unsafe"

// The blocked log-shift scan, in Go.
//
// This is the reference for FastCumSum and FastCumProd, and it has to
// reproduce the kernel's grouping exactly rather than merely closely. The Fast
// prefix relaxes agreement with a naive scalar loop; it does NOT relax
// agreement between tiers, and the portable path is one of the tiers. A
// reference that summed serially here would make every accelerated tier
// disagree with `-tags purego`, which is the defect this library exists to
// avoid.
//
// So the shape is copied from csrc/scan.c and must stay in step with it: a
// block of L elements, four (or three) combine steps against the block shifted
// right by 1, 2, 4 and 8 lanes with an identity shifted in, then the running
// carry from previous blocks, then a serial tail.

// scanLanes is SCAN_LANES from csrc/scan.c: sixteen for a four-byte element,
// eight for an eight-byte one, on every target regardless of vector width.
func scanLanes[T any]() int {
	var z T
	if unsafe.Sizeof(z) == 4 {
		return 16
	}
	return 8
}

// scanBlocked runs the log-shift scan with the given combine and identity.
func scanBlocked[T number](dst, a []T, id T, comb func(x, y T) T) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	l := scanLanes[T]()

	var buf [16]T
	run := id
	i := 0
	for ; i+l <= n; i += l {
		copy(buf[:l], a[i:i+l])
		// The shift steps. Lanes with nothing that far to their left take the
		// identity, which is what makes the combine correct for addition —
		// duplicating the neighbour, as the min and max scans do, would add a
		// value to itself.
		for s := 1; s < l; s *= 2 {
			var sh [16]T
			for j := range l {
				if j >= s {
					sh[j] = buf[j-s]
				} else {
					sh[j] = id
				}
			}
			for j := range l {
				buf[j] = comb(buf[j], sh[j])
			}
		}
		for j := range l {
			buf[j] = comb(buf[j], run)
			dst[i+j] = buf[j]
		}
		run = buf[l-1]
	}
	for ; i < n; i++ {
		run = comb(run, a[i])
		dst[i] = run
	}
}

// FastCumSum and FastCumProd are the exported references the generated
// backends fall back to and the differential tests compare against.
//
// FastCumSumFloat is the blocked scan for float32 and the SERIAL loop for
// float64, and that asymmetry is measured rather than arbitrary. Eight doubles
// fill one AVX-512 register, so the shift steps become cross-lane permutes,
// and against a serial chain of four-cycle adds the blocked form measured
// 0.91x — slower. float32 has sixteen lanes to hide the same latency and wins.
// Since no kernel is generated for float64, this branch is what every tier
// runs there, so they still agree with each other.
func FastCumSumFloat[T Float](dst, a []T) {
	var z T
	if unsafe.Sizeof(z) == 8 {
		n := min(len(dst), len(a))
		var run T
		for i := range dst[:n] {
			run += a[i]
			dst[i] = run
		}
		return
	}
	scanBlocked(dst, a, 0, func(x, y T) T { return x + y })
}

func FastCumProdFloat[T Float](dst, a []T) {
	scanBlocked(dst, a, 1, func(x, y T) T { return x * y })
}

// CumProdInt is the EXACT integer product scan. Two's-complement
// multiplication is associative, so the blocked grouping is bit-identical to
// the serial loop for every input including ones that overflow — verified over
// four million deliberately overflowing values. It therefore needs no Fast
// prefix, and this is the reference the int32 kernel falls back to.
//
// It is written as the blocked scan rather than as the obvious serial loop so
// that reference and kernel have the same shape, and a change to one is
// visibly a change to the other.
func CumProdInt[T Integer](dst, a []T) {
	scanBlocked(dst, a, 1, func(x, y T) T { return x * y })
}
