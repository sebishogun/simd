package benchmarks

// What compression is worth, against the loop it replaces.
//
// The comparison is against a plain Go filter loop rather than against the
// portable tier, because that loop is what a caller writes when this function
// does not exist — and because it is not a strawman: it is the same loop, and
// on seven of the nine targets it is also what this library runs.
//
// Density is the axis that matters and is the one a naive benchmark leaves
// out. The scalar loop's cost is a branch per element, so it is fastest when
// the branch is predictable — all-true or all-false — and worst at 50%, where
// the predictor is guessing. The vector version costs the same at every
// density, so the ratio between them varies by a factor of several across this
// sweep and a single density would misreport it in either direction.
//
//	go test -run '^$' -bench Compress -count 8 | benchstat -col /impl -

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

// goFilter is the loop being replaced, written the way it is normally written.
func goFilter(dst, src []int32, mask []bool) int {
	k := 0
	for i := range src {
		if mask[i] {
			dst[k] = src[i]
			k++
		}
	}
	return k
}

func compressInputs(n int, density float64, seed int64) (src []int32, mask []bool) {
	rng := rand.New(rand.NewSource(seed))
	src, mask = make([]int32, n), make([]bool, n)
	for i := range src {
		src[i] = int32(i)
		mask[i] = rng.Float64() < density
	}
	return
}

func BenchmarkCompress(b *testing.B) {
	densities := []struct {
		name string
		p    float64
	}{
		{"p=0.01", 0.01},
		{"p=0.25", 0.25},
		{"p=0.50", 0.50},
		{"p=0.90", 0.90},
	}
	for _, n := range []int{64, 256, 4096, 65536, 1 << 20} {
		for _, d := range densities {
			src, mask := compressInputs(n, d.p, 7)
			dst := make([]int32, n)
			for _, c := range []struct {
				impl string
				fn   func() int
			}{
				{"go", func() int { return goFilter(dst, src, mask) }},
				{"simd", func() int { return simd.CompressInto(dst, src, mask) }},
			} {
				b.Run(fmt.Sprintf("n=%d/%s/impl=%s", n, d.name, c.impl), func(b *testing.B) {
					b.SetBytes(int64(n) * 4)
					for b.Loop() {
						sink = c.fn()
					}
				})
			}
		}
	}
}

var sink int
