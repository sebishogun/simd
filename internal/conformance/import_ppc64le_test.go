//go:build ppc64le && !purego

package conformance

// The generated backends register themselves from init, so the test binary has
// to import the architecture package to have anything to check.
import _ "github.com/sebishogun/simd/internal/ppc64le"
