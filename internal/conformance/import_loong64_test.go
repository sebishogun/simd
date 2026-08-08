//go:build loong64 && !purego

package conformance

// The architecture package no longer registers anything from init -- that
// registration made every kernel reachable in every consumer's binary, which
// the per-operation dispatch tables exist to prevent. The suite asks for
// whole sets explicitly instead; only test binaries pay for them.
import "github.com/sebishogun/simd/internal/loong64"

func init() { archSetsFn = loong64.Sets }
