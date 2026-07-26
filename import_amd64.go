//go:build amd64 && !purego

package simd

// Importing the architecture package for its side effect: its init function
// registers the generated backends with internal/backend, which is where the
// dispatcher looks them up.
//
// The blank import carries the same build constraint as the generated files,
// so a build for another architecture never pulls in assembly it cannot use,
// and a purego build pulls in none at all.
import _ "github.com/sebishogun/simd/internal/amd64"
