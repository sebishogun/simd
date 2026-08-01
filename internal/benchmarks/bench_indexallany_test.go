package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// The question this kernel exists to answer: is one pass over the input
// comparing against six bytes faster than six passes comparing against one?
func BenchmarkIndexAllAny(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	const set = "{}[]:,"
	alphabet := `abcdefgh{}[]:,"\ ` + "\n"
	for _, n := range []int{4096, 65536, 1 << 20} {
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = alphabet[r.IntN(len(alphabet))]
		}
		dst := make([]int32, n+1)
		b.Run(fmt.Sprintf("n=%d/six-passes", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				total := 0
				for i := 0; i < len(set); i++ {
					total += simd.IndexAll(dst, buf, set[i])
				}
				sinkN = total
			}
		})
		b.Run(fmt.Sprintf("n=%d/one-pass", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkN = simd.IndexAllAny(dst, buf, set)
			}
		})
	}
}

var sinkN int
