package simd

// Narrow floating-point storage formats: float16 and bfloat16.
//
// Neither is an element type this package computes in, and that is deliberate.
// A vector unit that has half-precision arithmetic has it for reasons — power,
// throughput on a specific accelerator — that do not carry over to a general
// library, and anything computed in sixteen bits accumulates error fast enough
// to make a reproducibility guarantee meaningless. What a caller doing
// inference or graphics actually wants is to widen a buffer, work in float32,
// and narrow it again, which is exactly what is here.
//
// Both formats are carried as uint16, because Go has neither type. That is not
// a workaround: a []uint16 is the same sixteen bits per element, nothing is
// reinterpreted, and a caller who stores them in a struct or reads them off a
// wire already has them in that shape.
//
// The difference between the two is where the bits go. A bfloat16 is the top
// half of a float32 — same exponent range, eight fewer mantissa bits — so
// widening is exact and free, and it is the format that survives being used
// for weights. A float16 has a five-bit exponent, so it holds more precision
// over a much narrower range, and overflows to infinity above 65520.

// BFloat16ToFloat32Into widens each bfloat16 in a into a float32 in dst.
//
// Exact: every bfloat16 is a float32 with its low mantissa bits zero, so
// nothing is approximated, including NaN payloads and both infinities.
func BFloat16ToFloat32Into(dst []float32, a []uint16) { active.Convert.BF16ToF32(dst, a) }

// Float32ToBFloat16Into narrows each float32 in a into a bfloat16 in dst,
// rounding to nearest even.
//
// Rounding rather than truncating, which costs one add and is the difference
// between a rounding error and a drift: truncation is biased, and a bias
// applied to every weight in a network accumulates. A NaN is passed through
// quieted rather than rounded, because rounding can carry into the exponent
// and turn a NaN with a low mantissa into an infinity.
func Float32ToBFloat16Into(dst []uint16, a []float32) { active.Convert.F32ToBF16(dst, a) }

// Float16ToFloat32Into widens each float16 in a into a float32 in dst.
//
// Exact, including denormals, which are renormalized rather than flushed.
func Float16ToFloat32Into(dst []float32, a []uint16) { active.Convert.F16ToF32(dst, a) }

// Float32ToFloat16Into narrows each float32 in a into a float16 in dst,
// rounding to nearest even.
//
// Values above 65520 in magnitude become infinities and values below the
// smallest float16 denormal become zeros, both with the sign preserved.
// Denormals are produced rather than flushed.
func Float32ToFloat16Into(dst []uint16, a []float32) { active.Convert.F32ToF16(dst, a) }

// ---------- affine quantization ----------

// QuantizeInt8 converts float32 to int8 with an affine scale and zero point,
// which is the quantization every inference runtime uses:
//
//	q = clamp(round(x/scale) + zeroPoint, -128, 127)
//
// Rounding is half to even, matching ONNX, PyTorch and TFLite. That is worth
// stating because the naive form, int8(x/scale + 0.5), rounds half away from
// zero and disagrees on every exact .5 — which a symmetric scale produces in
// quantity rather than rarely, so the two differ in the middle of a typical
// distribution and not only at its edges.
//
// Values outside the representable range saturate rather than wrapping. It
// writes min(len(dst), len(a)) elements and allocates nothing.
func QuantizeInt8(dst []int8, a []float32, scale float32, zeroPoint int32) {
	active.Convert.QuantizeI8(dst, a, scale, zeroPoint)
}

// DequantizeInt8 is the inverse: x = (q - zeroPoint) * scale.
func DequantizeInt8(dst []float32, a []int8, scale float32, zeroPoint int32) {
	active.Convert.DequantizeI8(dst, a, scale, zeroPoint)
}

// QuantizeUint8 is [QuantizeInt8] into the unsigned range, clamping to
// [0, 255]. This is the form TFLite and most mobile runtimes use.
func QuantizeUint8(dst []uint8, a []float32, scale float32, zeroPoint int32) {
	active.Convert.QuantizeU8(dst, a, scale, zeroPoint)
}

// DequantizeUint8 is the inverse of [QuantizeUint8].
func DequantizeUint8(dst []float32, a []uint8, scale float32, zeroPoint int32) {
	active.Convert.DequantizeU8(dst, a, scale, zeroPoint)
}

// ---------- zigzag ----------

// ZigzagEncodeInt32Into maps signed integers onto unsigned ones so that a
// small magnitude of either sign becomes a small unsigned value:
//
//	0, -1, 1, -2, 2  ->  0, 1, 2, 3, 4
//
// This is the transform that makes a varint of a negative number short, and it
// is what protobuf, Avro and delta-encoded column stores apply before varint
// encoding. Without it, -1 as a two's complement 32-bit value has every high
// bit set and costs the full five bytes.
//
// The mapping is a bijection: every value round-trips through
// [ZigzagDecodeInt32Into], including math.MinInt32, which is the case a naive
// negate-and-double formulation overflows on.
//
// It writes min(len(dst), len(a)) elements and allocates nothing.
func ZigzagEncodeInt32Into(dst []uint32, a []int32) { active.Convert.ZigzagEncodeI32(dst, a) }

// ZigzagDecodeInt32Into is the inverse of [ZigzagEncodeInt32Into].
func ZigzagDecodeInt32Into(dst []int32, a []uint32) { active.Convert.ZigzagDecodeI32(dst, a) }

// ZigzagEncodeInt64Into is [ZigzagEncodeInt32Into] for 64-bit values, which is
// the width protobuf's sint64 uses.
func ZigzagEncodeInt64Into(dst []uint64, a []int64) { active.Convert.ZigzagEncodeI64(dst, a) }

// ZigzagDecodeInt64Into is the inverse of [ZigzagEncodeInt64Into].
func ZigzagDecodeInt64Into(dst []int64, a []uint64) { active.Convert.ZigzagDecodeI64(dst, a) }

// ZigzagEncodeInt16Into is [ZigzagEncodeInt32Into] for 16-bit values.
func ZigzagEncodeInt16Into(dst []uint16, a []int16) { active.Convert.ZigzagEncodeI16(dst, a) }

// ZigzagDecodeInt16Into is the inverse of [ZigzagEncodeInt16Into].
func ZigzagDecodeInt16Into(dst []int16, a []uint16) { active.Convert.ZigzagDecodeI16(dst, a) }

// ZigzagEncodeInt8Into is [ZigzagEncodeInt32Into] for 8-bit values.
func ZigzagEncodeInt8Into(dst []byte, a []int8) { active.Convert.ZigzagEncodeI8(dst, a) }

// ZigzagDecodeInt8Into is the inverse of [ZigzagEncodeInt8Into].
func ZigzagDecodeInt8Into(dst []int8, a []byte) { active.Convert.ZigzagDecodeI8(dst, a) }
