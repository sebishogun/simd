//go:build purego

package simd

// puregoOnly forces every tier to resolve to the portable Go reference,
// regardless of what the CPU supports.
//
// Build with -tags purego to get a binary containing no hand-written or
// generated assembly at all. That matters for three audiences: people
// auditing what actually executes, people on a toolchain or platform where
// the assembly cannot be trusted, and the differential tests, which use it to
// confirm the reference is self-consistent before anything is compared
// against it.
const puregoOnly = true
