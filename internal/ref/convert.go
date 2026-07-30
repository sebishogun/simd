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
		BitPackU32: bitPackU32, BitUnpackU32: bitUnpackU32,
		F8E4M3ToF32: F8E4M3ToF32, F32ToF8E4M3: F32ToF8E4M3,
		F8E5M2ToF32: F8E5M2ToF32, F32ToF8E5M2: F32ToF8E5M2,
		F16ToF32: f16ToF32, F32ToF16: f32ToF16,
		QuantizeI8: quantizeI8, DequantizeI8: dequantizeI8,
		QuantizeU8: quantizeU8, DequantizeU8: dequantizeU8,
		QMatMulI8: qMatMulI8, RequantizeI8: requantizeI8,
		QuantizePerChannelI8: QuantizePerChannelI8, QuantizePerChannelU8: QuantizePerChannelU8,
		DequantizePerChannelI8: DequantizePerChannelI8, DequantizePerChannelU8: DequantizePerChannelU8,
		ZigzagEncodeI8: ZigzagEncodeI8, ZigzagDecodeI8: ZigzagDecodeI8,
		ZigzagEncodeI16: ZigzagEncodeI16, ZigzagDecodeI16: ZigzagDecodeI16,
		ZigzagEncodeI32: ZigzagEncodeI32, ZigzagDecodeI32: ZigzagDecodeI32,
		ZigzagEncodeI64: ZigzagEncodeI64, ZigzagDecodeI64: ZigzagDecodeI64,
		VarintLenU32: VarintLenU32, VarintLenU64: VarintLenU64,
		VarintSizeU32: VarintSizeU32, VarintSizeU64: VarintSizeU64,
	}
}

// Exported entry points for the generated guards.

func BF16ToF32(dst []float32, a []uint16) { bf16ToF32(dst, a) }
func F32ToBF16(dst []uint16, a []float32) { f32ToBF16(dst, a) }
func F16ToF32(dst []float32, a []uint16)  { f16ToF32(dst, a) }
func F32ToF16(dst []uint16, a []float32)  { f32ToF16(dst, a) }

// ---------- affine quantization ----------
//
// These define the semantics the kernels are checked against, so the rounding
// mode is the specification and not an implementation detail.
//
// math.RoundToEven, not math.Round: the runtimes this interoperates with —
// ONNX, PyTorch, TFLite — all specify round-half-to-even, and C's rintf under
// the default rounding mode does the same. math.Round is half-away-from-zero
// and would disagree on every exact .5, which a symmetric scale produces in
// quantity rather than rarely.
//
// The clamp happens in float64 and before the integer conversion, because a
// Go conversion of an out-of-range float to an integer type is
// implementation-defined, not saturating.

func quantize[T int8 | uint8](dst []T, a []float32, scale float32, zeroPoint int32, lo, hi float64) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		// float64 throughout: the division and the rounding are specified on
		// the float32 value, but the clamp needs a range wider than the
		// destination type to detect that it was exceeded at all.
		q := math.RoundToEven(float64(a[i]/scale)) + float64(zeroPoint)
		if q < lo {
			q = lo
		}
		if q > hi {
			q = hi
		}
		dst[i] = T(q)
	}
}

func dequantize[T int8 | uint8](dst []float32, a []T, scale float32, zeroPoint int32) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = float32(int32(a[i])-zeroPoint) * scale
	}
}

func quantizeI8(dst []int8, a []float32, scale float32, zeroPoint int32) {
	quantize(dst, a, scale, zeroPoint, -128, 127)
}

func quantizeU8(dst []uint8, a []float32, scale float32, zeroPoint int32) {
	quantize(dst, a, scale, zeroPoint, 0, 255)
}

func dequantizeI8(dst []float32, a []int8, scale float32, zeroPoint int32) {
	dequantize(dst, a, scale, zeroPoint)
}

func dequantizeU8(dst []float32, a []uint8, scale float32, zeroPoint int32) {
	dequantize(dst, a, scale, zeroPoint)
}

// Exported entry points for the generated guards.

func QuantizeI8(dst []int8, a []float32, scale float32, zeroPoint int32) {
	quantizeI8(dst, a, scale, zeroPoint)
}

func QuantizeU8(dst []uint8, a []float32, scale float32, zeroPoint int32) {
	quantizeU8(dst, a, scale, zeroPoint)
}

func DequantizeI8(dst []float32, a []int8, scale float32, zeroPoint int32) {
	dequantizeI8(dst, a, scale, zeroPoint)
}

func DequantizeU8(dst []float32, a []uint8, scale float32, zeroPoint int32) {
	dequantizeU8(dst, a, scale, zeroPoint)
}

// zigzagEncode maps a signed integer onto an unsigned one so that a small
// magnitude of either sign becomes a small unsigned value. The generic form is
// written over both the signed and unsigned type because Go, unlike C, has no
// implicit relationship between them.
//
// Go's << and >> on a signed value are defined — left shift wraps and right
// shift is arithmetic — so this reads as the textbook identity, where the C
// kernel has to go through the unsigned domain to say the same thing.
func zigzagEncode[S ~int8 | ~int16 | ~int32 | ~int64, U ~uint8 | ~uint16 | ~uint32 | ~uint64](dst []U, a []S, shift int) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = U(a[i]<<1) ^ U(a[i]>>shift)
	}
}

func zigzagDecode[S ~int8 | ~int16 | ~int32 | ~int64, U ~uint8 | ~uint16 | ~uint32 | ~uint64](dst []S, a []U) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		dst[i] = S(a[i]>>1) ^ -S(a[i]&1)
	}
}

func ZigzagEncodeI8(dst []byte, a []int8)     { zigzagEncode(dst, a, 7) }
func ZigzagDecodeI8(dst []int8, a []byte)     { zigzagDecode(dst, a) }
func ZigzagEncodeI16(dst []uint16, a []int16) { zigzagEncode(dst, a, 15) }
func ZigzagDecodeI16(dst []int16, a []uint16) { zigzagDecode(dst, a) }
func ZigzagEncodeI32(dst []uint32, a []int32) { zigzagEncode(dst, a, 31) }
func ZigzagDecodeI32(dst []int32, a []uint32) { zigzagDecode(dst, a) }
func ZigzagEncodeI64(dst []uint64, a []int64) { zigzagEncode(dst, a, 63) }
func ZigzagDecodeI64(dst []int64, a []uint64) { zigzagDecode(dst, a) }

// ---------- varint widths ----------

// varintLen is the LEB128 width of x: seven payload bits per byte, so
// ceil(bits(x)/7), and 1 for zero.
//
// The kernel spells this as a sum of comparisons because bits.Len does not
// vectorize on SSE2 or AVX2. Here the clear form is used, and the differential
// test is what says the two agree.
func varintLen(x uint64) int {
	n := 1
	for x >= 0x80 {
		x >>= 7
		n++
	}
	return n
}

func VarintLenU32(dst []int32, a []uint32) {
	n := min(len(dst), len(a))
	for i, v := range a[:n] {
		dst[i] = int32(varintLen(uint64(v)))
	}
}

func VarintLenU64(dst []int32, a []uint64) {
	n := min(len(dst), len(a))
	for i, v := range a[:n] {
		dst[i] = int32(varintLen(v))
	}
}

// The totals need no fixed accumulator tree: integer addition is associative,
// so the kernel's eight-lane fold and this running sum agree exactly.

func VarintSizeU32(a []uint32) int {
	t := 0
	for _, v := range a {
		t += varintLen(uint64(v))
	}
	return t
}

func VarintSizeU64(a []uint64) int {
	t := 0
	for _, v := range a {
		t += varintLen(v)
	}
	return t
}

// ---------- quantized matrix multiply ----------

// qMatMulI8 is the specification the kernel is checked against: int8 inputs,
// int32 accumulator, row-major, in the same p-ascending order the tile uses so
// the two agree exactly.
//
// Integer addition is associative, so unlike the float matmul this needs no
// fixed accumulation shape to be reproducible — every tier reaches the same
// total whatever order its lanes reduce in.
func qMatMulI8(dst []int32, a, b []int8, m, k, n int) {
	if m <= 0 || k <= 0 || n <= 0 || len(dst) < m*n || len(a) < m*k || len(b) < k*n {
		return
	}
	for i := range dst[:m*n] {
		dst[i] = 0
	}
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			s := int32(a[i*k+p])
			br := b[p*n : p*n+n]
			dr := dst[i*n : i*n+n]
			for j := range dr {
				dr[j] += s * int32(br[j])
			}
		}
	}
}

// requantizeI8 takes an int32 accumulator back down to int8 with a scale and
// zero point. Rounding is half to even, matching quantize and the runtimes
// this interoperates with.
func requantizeI8(dst []int8, a []int32, scale float32, zeroPoint int32) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	for i := range dst {
		q := math.RoundToEven(float64(float32(a[i])*scale)) + float64(zeroPoint)
		switch {
		case q < -128:
			dst[i] = -128
		case q > 127:
			dst[i] = 127
		default:
			dst[i] = int8(q)
		}
	}
}

func QMatMulI8(dst []int32, a, b []int8, m, k, n int) { qMatMulI8(dst, a, b, m, k, n) }

func RequantizeI8(dst []int8, a []int32, scale float32, zeroPoint int32) {
	requantizeI8(dst, a, scale, zeroPoint)
}

// ---------- per-channel quantization ----------
//
// The specification for the per-channel kernels: identical arithmetic to the
// per-tensor form, with the scale and zero point looked up per channel.

func quantizePerChannel[T int8 | uint8](dst []T, a []float32, scale []float32,
	zeroPoint []int32, channels, inner int, lo, hi float64) {

	if channels <= 0 || inner <= 0 || len(scale) < channels ||
		len(zeroPoint) < channels || len(dst) < channels*inner ||
		len(a) < channels*inner {
		return
	}
	for c := 0; c < channels; c++ {
		s, z := scale[c], zeroPoint[c]
		ac := a[c*inner : (c+1)*inner]
		dc := dst[c*inner : (c+1)*inner]
		for i := range dc {
			q := math.RoundToEven(float64(ac[i]/s)) + float64(z)
			if q < lo {
				q = lo
			}
			if q > hi {
				q = hi
			}
			dc[i] = T(q)
		}
	}
}

func dequantizePerChannel[T int8 | uint8](dst []float32, a []T, scale []float32,
	zeroPoint []int32, channels, inner int) {

	if channels <= 0 || inner <= 0 || len(scale) < channels ||
		len(zeroPoint) < channels || len(dst) < channels*inner ||
		len(a) < channels*inner {
		return
	}
	for c := 0; c < channels; c++ {
		s, z := scale[c], zeroPoint[c]
		ac := a[c*inner : (c+1)*inner]
		dc := dst[c*inner : (c+1)*inner]
		for i := range dc {
			dc[i] = float32(int32(ac[i])-z) * s
		}
	}
}

func QuantizePerChannelI8(dst []int8, a []float32, scale []float32, zeroPoint []int32, channels, inner int) {
	quantizePerChannel(dst, a, scale, zeroPoint, channels, inner, -128, 127)
}

func QuantizePerChannelU8(dst []uint8, a []float32, scale []float32, zeroPoint []int32, channels, inner int) {
	quantizePerChannel(dst, a, scale, zeroPoint, channels, inner, 0, 255)
}

func DequantizePerChannelI8(dst []float32, a []int8, scale []float32, zeroPoint []int32, channels, inner int) {
	dequantizePerChannel(dst, a, scale, zeroPoint, channels, inner)
}

func DequantizePerChannelU8(dst []float32, a []uint8, scale []float32, zeroPoint []int32, channels, inner int) {
	dequantizePerChannel(dst, a, scale, zeroPoint, channels, inner)
}

// ---------- fp8 ----------
//
// Both OCP OFP8 formats. e4m3 has no infinity — exponent 1111 with mantissa
// 111 is the only NaN and everything else at that exponent is finite, which is
// what gives it a 448 maximum. e5m2 is IEEE-shaped and has infinities where
// float16 does.
//
// These go through float64 and math.Ldexp rather than repeating the kernel's
// bit arithmetic. That is deliberate: two implementations that share a
// derivation can share a mistake, and the differential test is only worth
// running if they arrive at the same answer by different routes.

type f8Format struct {
	manBits uint // mantissa bits
	bias    int  // exponent bias
	maxExp  uint // largest exponent field
	hasInf  bool // whether the top exponent means infinity

	// overflowAt is the magnitude at or above which the result leaves the
	// finite range — infinity for e5m2, saturation for e4m3.
	//
	// For e5m2 this is NOT the largest finite value. Round to nearest sends
	// everything below the midpoint between 57344 and the next power of two
	// back down to 57344, so the threshold is 61440. Using 57344 made 57345
	// an infinity while the kernel correctly gave 57344, which is what the
	// conformance differential reported.
	overflowAt float64
}

var (
	e4m3 = f8Format{manBits: 3, bias: 7, maxExp: 15, hasInf: false, overflowAt: 448}
	e5m2 = f8Format{manBits: 2, bias: 15, maxExp: 31, hasInf: true, overflowAt: 61440}
)

func f8ToF32(dst []float32, a []byte, f f8Format) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	manMask := uint(1)<<f.manBits - 1
	for i := range dst {
		u := uint(a[i])
		neg := u&0x80 != 0
		exp := (u >> f.manBits) & f.maxExp
		man := u & manMask

		var v float64
		switch {
		case exp == f.maxExp && f.hasInf:
			if man != 0 {
				v = math.NaN()
			} else {
				v = math.Inf(1)
			}
		case exp == f.maxExp && man == manMask:
			v = math.NaN() // e4m3's single NaN encoding
		case exp == 0:
			// Denormal: no implied leading bit, exponent fixed at 1-bias.
			v = math.Ldexp(float64(man)/float64(uint(1)<<f.manBits), 1-f.bias)
		default:
			frac := 1 + float64(man)/float64(uint(1)<<f.manBits)
			v = math.Ldexp(frac, int(exp)-f.bias)
		}
		if neg {
			v = -v
		}
		dst[i] = float32(v)
	}
}

func f32ToF8(dst []byte, a []float32, f f8Format) {
	n := min(len(dst), len(a))
	dst, a = dst[:n], a[:n]
	manMask := uint(1)<<f.manBits - 1
	for i := range dst {
		x := float64(a[i])
		var sign byte
		if math.Signbit(x) {
			sign = 0x80
		}
		mag := math.Abs(x)

		switch {
		case mag == 0:
			// math.Frexp(0) returns (0, 0), so without this the code below
			// synthesises an exponent field of bias-1 and turns zero into 0.5.
			// The conformance differential caught it: the kernel said 0x00 and
			// this said 0x30. Zero is the one input every format agrees on and
			// the one most easily lost in a general path.
			dst[i] = sign
			continue
		case math.IsNaN(x):
			// All-ones mantissa in both formats. Any non-zero mantissa at the
			// top exponent is a NaN in e5m2 and rule 1 does not promise a
			// payload, but the kernel and this have to agree or the
			// differential fails — and picking the same encoding for both
			// formats is one fewer thing to remember.
			dst[i] = sign | byte(f.maxExp<<f.manBits|manMask)
			continue
		case math.IsInf(x, 0) || mag >= f.overflowAt:
			if f.hasInf {
				dst[i] = sign | byte(f.maxExp<<f.manBits)
			} else {
				// No infinity: saturate at the largest finite value.
				dst[i] = sign | byte(f.maxExp<<f.manBits|manMask-1)
			}
			continue
		}

		// Round the mantissa to nearest even at the target precision, by
		// scaling so the units digit is the last kept bit.
		frac, exp := math.Frexp(mag) // mag == frac * 2^exp, frac in [0.5,1)
		e := exp - 1                 // unbiased exponent of the leading 1
		var q float64
		var field uint
		if e < 1-f.bias {
			// Denormal in the target: the exponent field is zero and the
			// mantissa is scaled by the fixed minimum exponent.
			q = math.RoundToEven(math.Ldexp(mag, int(f.manBits)+f.bias-1))
			if q >= float64(uint(1)<<f.manBits) {
				// Rounded up into the smallest normal.
				dst[i] = sign | byte(1<<f.manBits)
				continue
			}
			dst[i] = sign | byte(uint(q))
			continue
		}
		q = math.RoundToEven(frac * 2 * float64(uint(1)<<f.manBits))
		if q >= float64(uint(2)<<f.manBits) {
			q /= 2
			e++
		}
		field = uint(e + f.bias)
		man := uint(q) & manMask
		if f.hasInf && field >= f.maxExp {
			dst[i] = sign | byte(f.maxExp<<f.manBits)
			continue
		}
		if !f.hasInf && field == f.maxExp && man >= manMask {
			man = manMask - 1 // never produce the NaN encoding by rounding
		}
		if field > f.maxExp {
			field = f.maxExp
			man = manMask - 1
		}
		dst[i] = sign | byte(field<<f.manBits|man)
	}
}

func F8E4M3ToF32(dst []float32, a []byte) { f8ToF32(dst, a, e4m3) }
func F32ToF8E4M3(dst []byte, a []float32) { f32ToF8(dst, a, e4m3) }
func F8E5M2ToF32(dst []float32, a []byte) { f8ToF32(dst, a, e5m2) }
func F32ToF8E5M2(dst []byte, a []float32) { f32ToF8(dst, a, e5m2) }

// ---------- bit packing ----------
//
// The specification: values are packed little-endian, least significant bit
// first, with no padding between them and no alignment to word boundaries.
// That is the layout Parquet and Arrow use.

func bitPackU32(dst, a []uint32, bits int32) {
	if bits <= 0 || bits > 32 || len(dst) < (len(a)*int(bits)+31)/32 {
		return
	}
	mask := uint32(0xffffffff)
	if bits < 32 {
		mask = 1<<uint(bits) - 1
	}
	words := (len(a)*int(bits) + 31) / 32
	// Written as a bit cursor, which is the readable form and the one that
	// does not vectorize. The kernel inverts the loop; the two agreeing is
	// what makes the differential worth running.
	for w := range dst[:words] {
		dst[w] = 0
	}
	pos := 0
	for _, v := range a {
		v &= mask
		w, sh := pos/32, uint(pos%32)
		dst[w] |= v << sh
		if sh+uint(bits) > 32 {
			dst[w+1] |= v >> (32 - sh)
		}
		pos += int(bits)
	}
}

func bitUnpackU32(dst, a []uint32, bits int32) {
	if bits <= 0 || bits > 32 || len(a) < (len(dst)*int(bits)+31)/32+1 {
		return
	}
	mask := uint32(0xffffffff)
	if bits < 32 {
		mask = 1<<uint(bits) - 1
	}
	for i := range dst {
		pos := i * int(bits)
		w, sh := pos/32, uint(pos%32)
		v := a[w] >> sh
		if sh+uint(bits) > 32 {
			v |= a[w+1] << (32 - sh)
		}
		dst[i] = v & mask
	}
}

func BitPackU32(dst, a []uint32, bits int32)   { bitPackU32(dst, a, bits) }
func BitUnpackU32(dst, a []uint32, bits int32) { bitUnpackU32(dst, a, bits) }
