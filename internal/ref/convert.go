package ref

// Narrow floating-point storage formats, portably.
//
// These define the semantics the kernels are checked against, and unlike the
// byte scanners they cannot delegate: the standard library has no float16 or
// bfloat16. math.Float32bits is the only help available, and the rest is the
// same arithmetic the kernels do, written so that the two can only agree by
// being right rather than by sharing a mistake.

import (
	"math"

	"github.com/sebishogun/simd/internal/kernel"
)

func bf16ToF32(dst []float32, a []uint16) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = math.Float32frombits(uint32(a[i]) << 16)
	}
}

// f32ToBF16 rounds to nearest even. Truncating is one instruction cheaper and
// biased, and a bias applied to every weight in a network is a drift rather
// than a rounding error.
//
// A NaN is passed through quieted rather than rounded, because rounding can
// carry into the exponent and turn a NaN with a low mantissa into an infinity.
func f32ToBF16(dst []uint16, a []float32) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		u := math.Float32bits(a[i])
		if u&0x7f800000 == 0x7f800000 && u&0x007fffff != 0 {
			dst[i] = uint16(u>>16) | 0x0040
			continue
		}
		dst[i] = uint16((u + 0x7fff + ((u >> 16) & 1)) >> 16)
	}
}

func f16ToF32(dst []float32, a []uint16) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		h := a[i]
		sign := uint32(h&0x8000) << 16
		exp := uint32(h>>10) & 0x1f
		man := uint32(h & 0x3ff)
		switch {
		case exp == 0x1f:
			// Infinity or NaN: the exponent is all ones in both formats.
			dst[i] = math.Float32frombits(sign | 0x7f800000 | man<<13)
		case exp == 0 && man == 0:
			dst[i] = math.Float32frombits(sign)
		case exp == 0:
			// Denormal. Normalize by hand: shift the mantissa up until the
			// implied bit appears, and take the shifts out of the exponent.
			e := int32(-14)
			for man&0x400 == 0 {
				man <<= 1
				e--
			}
			man &= 0x3ff
			dst[i] = math.Float32frombits(sign | uint32(e+127)<<23 | man<<13)
		default:
			dst[i] = math.Float32frombits(sign | (exp+127-15)<<23 | man<<13)
		}
	}
}

// f32ToF16 rounds to nearest even, saturates above the largest float16, and
// produces denormals rather than flushing them.
//
// Written as explicit cases rather than as the kernel's branch-free
// arithmetic, on purpose: the two arriving at the same answer by different
// routes is what makes the differential test worth running.
func f32ToF16(dst []uint16, a []float32) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		u := math.Float32bits(a[i])
		sign := uint16(u>>16) & 0x8000
		mag := u & 0x7fffffff
		switch {
		case mag > 0x7f800000:
			dst[i] = sign | 0x7e00 // NaN, quieted
		case mag >= 0x47800000:
			dst[i] = sign | 0x7c00 // overflows to infinity
		case mag < 0x33000000:
			dst[i] = sign // rounds to zero
		case mag < 0x38800000:
			// Denormal. Shift the mantissa, with its implied bit, down to
			// where the exponent would have been -14, and round to nearest
			// even on what falls off the end.
			e := int32(mag>>23) - 127
			m := (mag & 0x7fffff) | 0x800000
			shift := uint32(-e - 14 + 13)
			half := uint32(1) << (shift - 1)
			r := m + half - 1 + ((m >> shift) & 1)
			dst[i] = sign | uint16(r>>shift)
		default:
			e := int32(mag>>23) - 127 + 15
			m := mag & 0x7fffff
			r := (uint32(e) << 10) | (m >> 13)
			// Round to nearest even on the thirteen bits dropped.
			rest := m & 0x1fff
			if rest > 0x1000 || (rest == 0x1000 && r&1 == 1) {
				r++
			}
			dst[i] = sign | uint16(r)
		}
	}
}

func convertOps() kernel.Convert {
	return kernel.Convert{
		BF16ToF32: bf16ToF32, F32ToBF16: f32ToBF16,
		F16ToF32: f16ToF32, F32ToF16: f32ToF16,
	}
}

// Exported entry points for the generated guards.

func BF16ToF32(dst []float32, a []uint16) { bf16ToF32(dst, a) }
func F32ToBF16(dst []uint16, a []float32) { f32ToBF16(dst, a) }
func F16ToF32(dst []float32, a []uint16)  { f16ToF32(dst, a) }
func F32ToF16(dst []uint16, a []float32)  { f32ToF16(dst, a) }
