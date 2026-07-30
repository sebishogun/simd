//go:build goexperiment.simd && amd64

package simd

// The escape hatch: a vector type, for the operations this package does not
// have.
//
// # When to reach for this, measured
//
// Not for operations the catalogue already has. float32 addition, this CPU,
// ns per call:
//
//	n     plain Go   slice API   this file
//	4       3.07       4.82        4.95
//	8       5.48       5.57        4.11
//	16     11.07       5.89        6.15
//	32     19.41       6.13        8.89
//	64     33.44       6.11       15.73
//	256   162.9        9.22       54.99
//
// The slice API wins from 16 elements up and the margin grows: at 256 it is
// six times faster than this file. Two reasons, both measured. The closure is
// the larger one — a hand-written loop with the same eight-wide blocking and
// no func value runs 256 elements in 31.7 ns against this file's 55.0, so
// about 23 ns is the call through the function parameter, which Go does not
// inline. The rest is width: the same loop at sixteen wide is 26.4 ns. Even
// that loses to 9.22, because the generated kernels are unrolled and these are
// not.
//
// So the niche is narrow and worth stating exactly: **an expression the
// catalogue does not have**, at any size, or **eight to sixteen elements**,
// where the call boundary is the whole runtime. For anything else the slice
// API is both faster and shorter. This file is an escape hatch, not a
// replacement, and the helpers below are here for tail-safety rather than for
// throughput.
//
// # Why this is not the main API
//
// The rest of this library takes slices, and that is a measured choice rather
// than a stylistic one. A vector type built on *this package's assembly* would
// need one non-inlinable call per operation — about 1.4 ns, floor — where Go's
// own intrinsics do a whole eight-element add in 2.82 ns. It would lose to a
// plain Go loop, which is not a hypothetical: it was built and measured.
//
// What changed is that Go grew its own intrinsics. simd/archsimd compiles to
// the instruction with no call at all, so a vector type built on *that* is
// fast. This file is that, and almost nothing else.
//
// # Why it is so small
//
// archsimd already has the operations. Re-wrapping them would be duplication
// with a maintenance cost and no benefit, so the types below are ALIASES, not
// defined types: every archsimd method remains available on them, and nothing
// is redeclared. What this file adds is the three things archsimd does not:
// one import instead of two, a truthful answer to "how wide should my blocks
// be", and the loop-plus-tail shape that hand-written SIMD gets wrong.
//
// # Availability
//
// Go 1.26 ships simd/archsimd behind GOEXPERIMENT=simd, and it is amd64 only —
// all ten of its implementation files are _amd64.go, and building for arm64
// fails with `undefined: archsimd.LoadFloat32x8Slice`. Its own documentation
// says "It currently supports AMD64".
//
// So this file exists on amd64 with the experiment enabled and nowhere else.
// The other five architectures, and every build without the experiment, get
// [Lanes] reporting zero and no vector type. Nothing else in this library
// changes: the slice API is fully accelerated on all six architectures either
// way, and none of it needs the experiment.
//
// # The Go 1.27 path
//
// Everything here is behind a build tag and no other file is touched, so when
// archsimd becomes flagless this becomes v2 by deleting the tags and bumping
// the module path. ROADMAP.md reserves v2.0.0 for exactly that.

import "simd/archsimd"

// Vector types, aliased from archsimd so that every method it defines is
// available here and nothing is redeclared.
//
// The names are shortened — F32x8 rather than Float32x8 — because the point of
// a vector type is code that reads like the arithmetic it performs, and the
// element type is already in the name twice.
type (
	F32x4  = archsimd.Float32x4
	F32x8  = archsimd.Float32x8
	F32x16 = archsimd.Float32x16

	F64x2 = archsimd.Float64x2
	F64x4 = archsimd.Float64x4
	F64x8 = archsimd.Float64x8

	I32x4  = archsimd.Int32x4
	I32x8  = archsimd.Int32x8
	I32x16 = archsimd.Int32x16

	I64x2 = archsimd.Int64x2
	I64x4 = archsimd.Int64x4
	I64x8 = archsimd.Int64x8

	U8x16 = archsimd.Uint8x16
	U8x32 = archsimd.Uint8x32
	U8x64 = archsimd.Uint8x64
)

// Mask types, aliased for the same reason. A comparison produces one of these
// and Select consumes it.
type (
	Mask32x4  = archsimd.Mask32x4
	Mask32x8  = archsimd.Mask32x8
	Mask32x16 = archsimd.Mask32x16
	Mask64x2  = archsimd.Mask64x2
	Mask64x4  = archsimd.Mask64x4
	Mask64x8  = archsimd.Mask64x8
)

// Lanes reports how many elements of type T the widest vector this CPU can
// actually execute holds — 16 float32 on AVX-512, 8 on AVX2, 4 otherwise.
//
// It exists because the alternative is guessing. Writing against F32x16 on a
// machine without AVX-512 does not fail to compile; it runs, slowly, through
// whatever the compiler emits to fake the width.
//
// It returns 0 on any build where this package has no vector type — every
// architecture other than amd64, and every build without GOEXPERIMENT=simd.
// Check for zero rather than dividing by it.
func Lanes[T float32 | float64 | int32 | int64 | uint8]() int {
	var z T
	width := 16 // bytes, the SSE baseline every amd64 CPU has
	switch {
	case archsimd.X86.AVX512():
		width = 64
	case archsimd.X86.AVX2():
		width = 32
	}
	switch any(z).(type) {
	case float32, int32:
		return width / 4
	case float64, int64:
		return width / 8
	case uint8:
		return width
	}
	return 0
}

// MapFloat32x8 applies f to every element of a, eight at a time, writing to
// dst. It handles the tail with a partial load and store, so a length that is
// not a multiple of eight is neither a special case for the caller nor an
// out-of-bounds read.
//
// That tail is the whole reason this function exists. It is the part of
// hand-written SIMD that is got wrong most often, and the two failure modes
// are both quiet: reading past the end of a slice usually lands in the same
// allocation and returns plausible garbage, and skipping the remainder leaves
// the last few elements of the output untouched rather than obviously wrong.
//
//	simd.MapFloat32x8(dst, src, func(v simd.F32x8) simd.F32x8 {
//		return v.Mul(v).Add(v)
//	})
//
// f is called with the loaded block and must return the block to store. It is
// NOT inlined: Go does not inline through a function-value parameter, and the
// call costs about 23 ns per 256 elements — most of the gap between this and
// the same loop written out by hand. If you are in the loop that matters,
// write the loop; this is for getting the tail right while you find out
// whether it matters.
//
// It writes min(len(dst), len(a)) elements and allocates nothing.
func MapFloat32x8(dst, a []float32, f func(F32x8) F32x8) {
	n := min(len(dst), len(a))
	i := 0
	for ; i+8 <= n; i += 8 {
		f(archsimd.LoadFloat32x8Slice(a[i:])).StoreSlice(dst[i:])
	}
	if i < n {
		f(archsimd.LoadFloat32x8SlicePart(a[i:n])).StoreSlicePart(dst[i:n])
	}
}

// MapFloat64x4 is [MapFloat32x8] for float64, four at a time.
func MapFloat64x4(dst, a []float64, f func(F64x4) F64x4) {
	n := min(len(dst), len(a))
	i := 0
	for ; i+4 <= n; i += 4 {
		f(archsimd.LoadFloat64x4Slice(a[i:])).StoreSlice(dst[i:])
	}
	if i < n {
		f(archsimd.LoadFloat64x4SlicePart(a[i:n])).StoreSlicePart(dst[i:n])
	}
}

// ZipFloat32x8 is [MapFloat32x8] over two inputs: f receives corresponding
// blocks of a and b and returns the block to store.
//
//	simd.ZipFloat32x8(dst, a, b, func(x, y simd.F32x8) simd.F32x8 {
//		return x.Mul(y)
//	})
//
// It writes min(len(dst), len(a), len(b)) elements and allocates nothing.
func ZipFloat32x8(dst, a, b []float32, f func(x, y F32x8) F32x8) {
	n := min(len(dst), len(a), len(b))
	i := 0
	for ; i+8 <= n; i += 8 {
		f(archsimd.LoadFloat32x8Slice(a[i:]), archsimd.LoadFloat32x8Slice(b[i:])).
			StoreSlice(dst[i:])
	}
	if i < n {
		f(archsimd.LoadFloat32x8SlicePart(a[i:n]), archsimd.LoadFloat32x8SlicePart(b[i:n])).
			StoreSlicePart(dst[i:n])
	}
}

// ZipFloat64x4 is [ZipFloat32x8] for float64.
func ZipFloat64x4(dst, a, b []float64, f func(x, y F64x4) F64x4) {
	n := min(len(dst), len(a), len(b))
	i := 0
	for ; i+4 <= n; i += 4 {
		f(archsimd.LoadFloat64x4Slice(a[i:]), archsimd.LoadFloat64x4Slice(b[i:])).
			StoreSlice(dst[i:])
	}
	if i < n {
		f(archsimd.LoadFloat64x4SlicePart(a[i:n]), archsimd.LoadFloat64x4SlicePart(b[i:n])).
			StoreSlicePart(dst[i:n])
	}
}

// HasVectorType reports whether this build has the vector type — that is,
// whether it is amd64 and GOEXPERIMENT=simd is set.
//
// It is a constant, so a caller can branch on it and have the dead side
// compiled away.
const HasVectorType = true
