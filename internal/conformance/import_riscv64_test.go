//go:build riscv64 && !purego

package conformance

// Importing the architecture package for its side effect: its init functions
// register the generated backends, which is what this suite tests.
//
// One file per architecture, each carrying the same build constraint as the
// package it imports. A single unconditional import would not compile, because
// the constraints exclude every architecture but the current one and Go treats
// "no Go files" as an error rather than an empty package.
import _ "github.com/sebishogun/simd/internal/riscv64"
