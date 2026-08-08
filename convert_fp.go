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
func BFloat16ToFloat32Into(dst []float32, a []uint16) { tblConvertBF16ToF32[tierIdx](dst, a) }

// Float32ToBFloat16Into narrows each float32 in a into a bfloat16 in dst,
// rounding to nearest even.
//
// Rounding rather than truncating, which costs one add and is the difference
// between a rounding error and a drift: truncation is biased, and a bias
// applied to every weight in a network accumulates. A NaN is passed through
// quieted rather than rounded, because rounding can carry into the exponent
// and turn a NaN with a low mantissa into an infinity.
func Float32ToBFloat16Into(dst []uint16, a []float32) { tblConvertF32ToBF16[tierIdx](dst, a) }

// Float16ToFloat32Into widens each float16 in a into a float32 in dst.
//
// Exact, including denormals, which are renormalized rather than flushed.
func Float16ToFloat32Into(dst []float32, a []uint16) { tblConvertF16ToF32[tierIdx](dst, a) }

// Float32ToFloat16Into narrows each float32 in a into a float16 in dst,
// rounding to nearest even.
//
// Values above 65520 in magnitude become infinities and values below the
// smallest float16 denormal become zeros, both with the sign preserved.
// Denormals are produced rather than flushed.
func Float32ToFloat16Into(dst []uint16, a []float32) { tblConvertF32ToF16[tierIdx](dst, a) }

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
	tblConvertQuantizeI8[tierIdx](dst, a, scale, zeroPoint)
}

// DequantizeInt8 is the inverse: x = (q - zeroPoint) * scale.
func DequantizeInt8(dst []float32, a []int8, scale float32, zeroPoint int32) {
	tblConvertDequantizeI8[tierIdx](dst, a, scale, zeroPoint)
}

// QuantizeUint8 is [QuantizeInt8] into the unsigned range, clamping to
// [0, 255]. This is the form TFLite and most mobile runtimes use.
func QuantizeUint8(dst []uint8, a []float32, scale float32, zeroPoint int32) {
	tblConvertQuantizeU8[tierIdx](dst, a, scale, zeroPoint)
}

// DequantizeUint8 is the inverse of [QuantizeUint8].
func DequantizeUint8(dst []float32, a []uint8, scale float32, zeroPoint int32) {
	tblConvertDequantizeU8[tierIdx](dst, a, scale, zeroPoint)
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
func ZigzagEncodeInt32Into(dst []uint32, a []int32) { tblConvertZigzagEncodeI32[tierIdx](dst, a) }

// ZigzagDecodeInt32Into is the inverse of [ZigzagEncodeInt32Into].
func ZigzagDecodeInt32Into(dst []int32, a []uint32) { tblConvertZigzagDecodeI32[tierIdx](dst, a) }

// ZigzagEncodeInt64Into is [ZigzagEncodeInt32Into] for 64-bit values, which is
// the width protobuf's sint64 uses.
func ZigzagEncodeInt64Into(dst []uint64, a []int64) { tblConvertZigzagEncodeI64[tierIdx](dst, a) }

// ZigzagDecodeInt64Into is the inverse of [ZigzagEncodeInt64Into].
func ZigzagDecodeInt64Into(dst []int64, a []uint64) { tblConvertZigzagDecodeI64[tierIdx](dst, a) }

// ZigzagEncodeInt16Into is [ZigzagEncodeInt32Into] for 16-bit values.
func ZigzagEncodeInt16Into(dst []uint16, a []int16) { tblConvertZigzagEncodeI16[tierIdx](dst, a) }

// ZigzagDecodeInt16Into is the inverse of [ZigzagEncodeInt16Into].
func ZigzagDecodeInt16Into(dst []int16, a []uint16) { tblConvertZigzagDecodeI16[tierIdx](dst, a) }

// ZigzagEncodeInt8Into is [ZigzagEncodeInt32Into] for 8-bit values.
func ZigzagEncodeInt8Into(dst []byte, a []int8) { tblConvertZigzagEncodeI8[tierIdx](dst, a) }

// ZigzagDecodeInt8Into is the inverse of [ZigzagEncodeInt8Into].
func ZigzagDecodeInt8Into(dst []int8, a []byte) { tblConvertZigzagDecodeI8[tierIdx](dst, a) }

// ---------- quantized matrix multiply ----------

// QMatMulInt8Into multiplies two row-major int8 matrices into an int32
// destination: an m*k matrix by a k*n matrix, giving m*n.
//
// The accumulator is int32 and that is the point of a separate function.
// [MatMulInto] is generic over one element type, so instantiating it at int8
// would accumulate in int8 and overflow after two or three terms — two
// full-scale int8 values already multiply to 16129. Here the worst case is
// k*127*128, which stays inside int32 up to k = 132097, past any layer anyone
// runs.
//
// This is what [QuantizeInt8] produces tensors for. Follow it with
// [RequantizeInt8Into] to get back to int8, or keep the int32 result if a bias
// or a residual connection is added next.
//
// The result is exact and identical on every instruction set: integer addition
// is associative, so unlike the float matmul there is no accumulation order to
// preserve. It does nothing if the slices are too short for the stated
// dimensions, and it allocates nothing.
func QMatMulInt8Into(dst []int32, a, b []int8, m, k, n int) {
	tblConvertQMatMulI8[tierIdx](dst, a, b, m, k, n)
}

// RequantizeInt8Into takes an int32 accumulator back down to int8 with a scale
// and zero point:
//
//	q = clamp(round(acc*scale) + zeroPoint, -128, 127)
//
// Rounding is half to even, matching [QuantizeInt8] and the runtimes this
// interoperates with. Values outside the range saturate rather than wrapping.
//
// It is separate from [QMatMulInt8Into] rather than fused because a real layer
// adds a bias to the int32 accumulator first, and because the scale is usually
// per output channel — see the per-channel note on [QuantizeInt8].
//
// It writes min(len(dst), len(a)) elements and allocates nothing.
func RequantizeInt8Into(dst []int8, a []int32, scale float32, zeroPoint int32) {
	tblConvertRequantizeI8[tierIdx](dst, a, scale, zeroPoint)
}

// ---------- per-channel quantization ----------

// QuantizePerChannelInt8 is [QuantizeInt8] with one scale and zero point per
// output channel rather than one for the whole tensor.
//
// This is what inference actually uses for weights. Output channels are
// trained independently and their ranges differ by an order of magnitude or
// more, so a single tensor-wide scale is set by the widest channel and wastes
// most of the int8 range on every other one — typically one to two bits of
// effective precision, for no cost beyond storing a scale per channel.
//
// The layout is channels groups of inner consecutive elements: element
// c*inner+i belongs to channel c. That is how a weight tensor shaped
// [outChannels][inChannels*kh*kw] already sits in memory, so no rearrangement
// is needed.
//
// Rounding, saturation and the zero point behave exactly as in [QuantizeInt8].
// It does nothing if the slices are too short for the stated shape, and it
// allocates nothing.
func QuantizePerChannelInt8(dst []int8, a []float32, scale []float32, zeroPoint []int32, channels, inner int) {
	tblConvertQuantizePerChannelI8[tierIdx](dst, a, scale, zeroPoint, channels, inner)
}

// QuantizePerChannelUint8 is [QuantizePerChannelInt8] into the unsigned range,
// clamping to [0, 255].
func QuantizePerChannelUint8(dst []uint8, a []float32, scale []float32, zeroPoint []int32, channels, inner int) {
	tblConvertQuantizePerChannelU8[tierIdx](dst, a, scale, zeroPoint, channels, inner)
}

// DequantizePerChannelInt8 is the inverse of [QuantizePerChannelInt8].
func DequantizePerChannelInt8(dst []float32, a []int8, scale []float32, zeroPoint []int32, channels, inner int) {
	tblConvertDequantizePerChannelI8[tierIdx](dst, a, scale, zeroPoint, channels, inner)
}

// DequantizePerChannelUint8 is the inverse of [QuantizePerChannelUint8].
func DequantizePerChannelUint8(dst []float32, a []uint8, scale []float32, zeroPoint []int32, channels, inner int) {
	tblConvertDequantizePerChannelU8[tierIdx](dst, a, scale, zeroPoint, channels, inner)
}

// ---------- fp8 ----------

// Float8E4M3ToFloat32Into widens OCP e4m3 to float32: 1 sign bit, 4 exponent
// bits, 3 mantissa bits, bias 7. This is the format weights and activations
// use.
//
// **e4m3 has no infinity.** Exponent 1111 with mantissa 111 is the only NaN
// encoding and every other value at that exponent is finite, which is what
// gives the format its 448 maximum rather than 240. That is the OCP OFP8
// definition and NVIDIA's e4m3fn — the "fn" is finite-not-nan — and it is what
// every shipping implementation does. Compare [Float8E5M2ToFloat32Into], which
// is IEEE-shaped and does have infinities.
//
// It writes min(len(dst), len(a)) elements and allocates nothing.
func Float8E4M3ToFloat32Into(dst []float32, a []byte) { tblConvertF8E4M3ToF32[tierIdx](dst, a) }

// Float32ToFloat8E4M3Into narrows to OCP e4m3, rounding to nearest even.
//
// Because the format has no infinity, values above 448 in magnitude saturate
// to ±448 rather than becoming one, and an input infinity saturates too. A NaN
// stays a NaN. Denormals are produced rather than flushed.
func Float32ToFloat8E4M3Into(dst []byte, a []float32) { tblConvertF32ToF8E4M3[tierIdx](dst, a) }

// Float8E5M2ToFloat32Into widens e5m2 to float32: 1 sign bit, 5 exponent bits,
// 2 mantissa bits, bias 15. This is the format gradients use, and it trades
// e4m3's extra mantissa bit for float16's exponent range.
//
// Unlike e4m3 this one is IEEE-shaped: it has infinities and NaNs where a
// float16 has them.
func Float8E5M2ToFloat32Into(dst []float32, a []byte) { tblConvertF8E5M2ToF32[tierIdx](dst, a) }

// Float32ToFloat8E5M2Into narrows to e5m2, rounding to nearest even.
//
// Values above 57344 in magnitude become infinities, as in float16. Denormals
// are produced rather than flushed.
func Float32ToFloat8E5M2Into(dst []byte, a []float32) { tblConvertF32ToF8E5M2[tierIdx](dst, a) }

// ---------- bit packing ----------

// BitPackInto packs the low `bits` bits of each value in a into a dense
// bitstream in dst, least significant bit first and with no padding.
//
// This is the representation Parquet, Arrow and Lucene use for an integer
// column after delta encoding: once the deltas are small, storing each in a
// full 32 bits is mostly zeroes. Pair it with [DiffInto] to produce the deltas
// and [ZigzagEncodeInt32Into] if they can be negative.
//
// dst must have room for ceil(len(a)*bits/32) words. bits must be 1 to 32.
// It does nothing if either is not so, and it allocates nothing.
func BitPackInto(dst, a []uint32, bits int32) { tblConvertBitPackU32[tierIdx](dst, a, bits) }

// BitUnpackInto is the inverse: it reads len(dst) values of `bits` bits each
// from the bitstream in a.
//
// The number of values is taken from len(dst), because a bitstream does not
// record how many values it holds — the caller knows, and a packed block in
// any real format carries the count beside it.
//
// a must hold ceil(len(dst)*bits/32)+1 words. The extra word is not slack: a
// value whose bits straddle a word boundary reads the next word, and the last
// value straddles whenever len(dst)*bits is not a multiple of 32. Requiring it
// in the guard is what lets the kernel read unconditionally rather than
// branching on every element.
func BitUnpackInto(dst, a []uint32, bits int32) {
	// Whole 32-value blocks go through the width-specialized kernel -- the
	// general form's runtime shift count defeats every vectorizer, so the
	// width is switched once and each case unpacks with literal shifts.
	// The tail, and any degenerate width, keeps the general path.
	if bits >= 1 && bits <= 32 {
		blocks := len(dst) / 32
		for blocks > 0 && len(a)*32 < blocks*32*int(bits) {
			blocks--
		}
		if blocks > 0 {
			tblBytesBitUnpackFastU32[tierIdx](dst, a, blocks, uint32(bits))
			dst = dst[32*blocks:]
			a = a[int(bits)*blocks:]
		}
	}
	if len(dst) > 0 {
		tblConvertBitUnpackU32[tierIdx](dst, a, bits)
	}
}
