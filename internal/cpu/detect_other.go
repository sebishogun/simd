//go:build !amd64 && !arm64 && !riscv64 && !s390x && !loong64 && !ppc64 && !ppc64le

package cpu

// detect reports the tiers this CPU supports, weakest first.
//
// Architectures without a generated backend get the portable Go path. wasm
// lands here permanently: its assembler has zero SIMD128 opcodes and no
// assembler path will ever exist, so the only route would be Go 1.27
// intrinsics, which this library deliberately does not depend on.
func detect() []Tier { return []Tier{Scalar} }
