package simd

// FormatFloat64 renders a finite float64 into dst shortest-form and
// returns the byte count. The format rule is encoding/json's, not
// strconv's 'g': decimal while the magnitude is in [1e-6, 1e21),
// scientific outside it, single-digit exponents unpadded, negative zero
// as "-0". The digits are Schubfach's and agree with strconv everywhere;
// only the format choice differs, and it differs exactly as
// encoding/json's does.
//
// dst must have room for 25 bytes. Infinities and NaN are the caller's
// to reject, as they are in JSON.
func FormatFloat64(dst []byte, v float64) int {
	return tblBytesDtoaF64[tierIdx](dst, v)
}
