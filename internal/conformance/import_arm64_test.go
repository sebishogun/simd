//go:build arm64 && !purego

package conformance

// The architecture package no longer registers anything from init -- that
// registration made every kernel reachable in every consumer's binary, which
// the per-operation dispatch tables exist to prevent. The suite asks for
// whole sets explicitly instead; only test binaries pay for them.
import "github.com/sebishogun/simd/internal/arm64"

func init() { archSetsFn = arm64.Sets }
