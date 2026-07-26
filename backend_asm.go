//go:build !purego

package simd

// puregoOnly is false in a normal build, so generated backends are eligible.
// See backend_purego.go.
const puregoOnly = false
