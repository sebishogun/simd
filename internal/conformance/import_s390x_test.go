//go:build s390x && !purego

package conformance

// The architecture package no longer registers anything from init -- that
// registration made every kernel reachable in every consumer's binary, which
// the per-operation dispatch tables exist to prevent. The suite asks for
// whole sets explicitly instead; only test binaries pay for them.
import "github.com/sebishogun/simd/internal/s390x"

func init() { archSetsFn = s390x.Sets }
