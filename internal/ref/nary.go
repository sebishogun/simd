package ref

// Elementwise operations over three or four slices at once.
//
// These exist for memory traffic, not for arithmetic. As repeated binary calls,
// dst = a+b+c+d makes three passes over memory for what is one instruction per
// element; done in one loop it makes one. Above the last level of cache that is
// the whole cost of the operation.
//
// The accumulation is left to right — ((a+b)+c)+d — and that is a promise
// rather than an implementation detail. Floating-point addition is not
// associative, so a caller must be able to rely on AddAll giving exactly the
// bits they would have got from writing the binary calls out by hand. The
// conversions are what stop Go fusing the operations and quietly breaking that.

func add3[T number](dst, a, b, c []T) {
	n := min(len(dst), len(a), len(b), len(c))
	dst, a, b, c = dst[:n], a[:n], b[:n], c[:n]
	for i := range dst {
		dst[i] = T(T(a[i]+b[i]) + c[i])
	}
}

func add4[T number](dst, a, b, c, d []T) {
	n := min(len(dst), len(a), len(b), len(c), len(d))
	dst, a, b, c, d = dst[:n], a[:n], b[:n], c[:n], d[:n]
	for i := range dst {
		dst[i] = T(T(T(a[i]+b[i])+c[i]) + d[i])
	}
}

func mul3[T number](dst, a, b, c []T) {
	n := min(len(dst), len(a), len(b), len(c))
	dst, a, b, c = dst[:n], a[:n], b[:n], c[:n]
	for i := range dst {
		dst[i] = T(T(a[i]*b[i]) * c[i])
	}
}

func mul4[T number](dst, a, b, c, d []T) {
	n := min(len(dst), len(a), len(b), len(c), len(d))
	dst, a, b, c, d = dst[:n], a[:n], b[:n], c[:n], d[:n]
	for i := range dst {
		dst[i] = T(T(T(a[i]*b[i])*c[i]) * d[i])
	}
}

// Exported entry points for generated code; the threshold guards call these
// directly rather than through the kernel set.

func Add3[T Number](dst, a, b, c []T)    { add3(dst, a, b, c) }
func Add4[T Number](dst, a, b, c, d []T) { add4(dst, a, b, c, d) }
func Mul3[T Number](dst, a, b, c []T)    { mul3(dst, a, b, c) }
func Mul4[T Number](dst, a, b, c, d []T) { mul4(dst, a, b, c, d) }
