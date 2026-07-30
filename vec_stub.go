//go:build !(goexperiment.simd && amd64)

package simd

// The build where there is no vector type.
//
// simd/archsimd is amd64-only in Go 1.26 and exists only under
// GOEXPERIMENT=simd — without the experiment the package has no files at all,
// so importing it is a build error rather than an empty API. This file is what
// every other configuration compiles instead.
//
// Only the names that can be answered truthfully here are declared. There is
// deliberately no F32x8 aliased to something else and no MapFloat32x8 running
// a scalar loop: a vector type that is not a vector is worse than no vector
// type, because code written against it compiles everywhere and is fast in one
// place, and nothing says which.
//
// Nothing else in this library is affected. The slice API — every operation in
// the catalogue — is fully accelerated on all six architectures with no build
// tag and no experiment. What is missing here is only the escape hatch for
// writing your own kernel inline, and the fallback for that is the same as it
// has always been: write the loop, or add a kernel with docs/kernels.md.

// Lanes reports how many elements of type T the widest usable vector holds,
// and returns 0 here because this build has no vector type. See the amd64
// version for what it reports when there is one.
//
// Check for zero rather than dividing by it.
func Lanes[T float32 | float64 | int32 | int64 | uint8]() int { return 0 }

// HasVectorType reports whether this build has the vector type — that is,
// whether it is amd64 and GOEXPERIMENT=simd is set. It is false here.
//
// It is a constant, so a caller can branch on it and have the dead side
// compiled away:
//
//	if simd.HasVectorType {
//		// ... vector path
//	} else {
//		// ... slice API, which is accelerated either way
//	}
const HasVectorType = false
