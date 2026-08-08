package ref

// The columnar family's portable reference: the filter and the null-aware
// sum over Arrow-style validity bitmaps, one bit per row, LSB-first within
// a byte.

// compressBits packs the elements whose validity bit is set.
func compressBits[T any](dst, src []T, bm []byte) int {
	n := min(len(src), 8*len(bm))
	src = src[:n]
	k := 0
	for i := range src {
		if bm[i>>3]>>(i&7)&1 != 0 {
			dst[k] = src[i]
			k++
		}
	}
	return k
}

// maskedCopy zeroes the lanes whose validity bit is clear. The copy is what
// guarantees the null-aware sum's accumulation tree is Sum's own, so the
// kernel -- which selects instead of copying -- has a bit-exact target to
// match. The reference is allowed the allocation; the kernel is the fast
// path.
func maskedCopy[T Number](a []T, bm []byte) []T {
	masked := make([]T, len(a))
	a = a[:min(len(a), 8*len(bm))]
	for i := range a {
		if bm[i>>3]>>(i&7)&1 != 0 {
			masked[i] = a[i]
		}
	}
	return masked
}

func CompressBitsFloat32(dst, src []float32, bm []byte) int { return compressBits(dst, src, bm) }
func CompressBitsFloat64(dst, src []float64, bm []byte) int { return compressBits(dst, src, bm) }
func CompressBitsInt32(dst, src []int32, bm []byte) int     { return compressBits(dst, src, bm) }
func CompressBitsInt64(dst, src []int64, bm []byte) int     { return compressBits(dst, src, bm) }

func SumValidFloat32(a []float32, bm []byte) float32 { return sumFloat(maskedCopy(a, bm)) }
func SumValidFloat64(a []float64, bm []byte) float64 { return sumFloat(maskedCopy(a, bm)) }
func SumValidInt32(a []int32, bm []byte) int32       { return sumInt(maskedCopy(a, bm)) }
func SumValidInt64(a []int64, bm []byte) int64       { return sumInt(maskedCopy(a, bm)) }
