package simd

// ConvertInto converts each element of src to the destination element type and
// writes it to dst, processing min(len(dst), len(src)) elements.
//
// It covers every pair among float32, float64, int32 and int64:
//
//	simd.ConvertInto(f64, f32)   // widen
//	simd.ConvertInto(f32, f64)   // narrow, rounding to nearest even
//	simd.ConvertInto(i32, f64)   // truncate toward zero
//	simd.ConvertInto(f64, i64)   // widen, rounding if beyond 2^53
//
// The conversions follow Go's own rules, which is to say the hardware's:
// float to integer truncates toward zero, and the result is
// implementation-defined if the value does not fit — so range-check first with
// [Clamp] if the input is untrusted.
//
// This is a single generic loop rather than a per-pair kernel. Widening and
// narowing conversions are single vector instructions (VCVTPS2PD and friends)
// and will get their own kernels; the signature will not change when they do.
func ConvertInto[D, S Number](dst []D, src []S) {
	n := min(len(dst), len(src))
	dst, src = dst[:n], src[:n]
	for i := range dst {
		dst[i] = D(src[i])
	}
}
