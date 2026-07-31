package simd_test

import (
	"math"
	"math/rand/v2"
)

// Helpers for the tests that had to stay in this directory — the ones using a
// hook from export_test.go, which only exists for the package it sits beside.
// The rest of the suite moved to internal/tests and carries its own copies;
// these are small enough that sharing them across a package boundary would
// cost more than repeating them.

const maxLen = 70

func randF64(n int, r *rand.Rand) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64() * 10
	}
	return s
}

func equalF64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] && !(math.IsNaN(a[i]) && math.IsNaN(b[i])) {
			return false
		}
	}
	return true
}

func randI64(n int, r *rand.Rand) []int64 {
	s := make([]int64, n)
	for i := range s {
		s[i] = int64(r.Int64N(2001) - 1000)
	}
	return s
}

func clone[T any](s []T) []T {
	out := make([]T, len(s))
	copy(out, s)
	return out
}

// Sinks, so the compiler cannot delete the call a test is measuring.
var (
	sinkB bool
	sinkF float64
	sinkN int
)
