// Separate from the library's module on purpose: it depends on another SIMD
// package, and no consumer of github.com/sebishogun/simd should inherit that.
module comparison

go 1.25.0

require (
	github.com/kelindar/simd v1.2.0
	github.com/sebishogun/simd v1.0.2
)

require (
	github.com/klauspost/cpuid/v2 v2.0.12 // indirect
	golang.org/x/sys v0.47.0 // indirect
)
