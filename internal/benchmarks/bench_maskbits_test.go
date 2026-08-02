package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// The question these kernels exist to answer: when the caller's next step is
// itself bitwise, is a bit per byte cheaper than a list of offsets?
//
// The two produce the same information. IndexAll writes four bytes per match,
// so on input where matches are common its output is larger than its input and
// the store is most of the work; MaskBits writes one bit per byte whatever the
// density, and on a target with mask registers the whole loop is a compare into
// a predicate and a store of that predicate.
//
// The alphabet here is deliberately dense — around 40% of bytes match, which is
// what a JSON document looks like. On sparse input the offset list is the
// smaller answer and this comparison would run the other way.
func BenchmarkMaskBits(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	const set = "{}[]:,"
	alphabet := `abcdefgh{}[]:,"\ ` + "\n"
	for _, n := range []int{4096, 65536, 1 << 20} {
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = alphabet[r.IntN(len(alphabet))]
		}
		idx := make([]int32, n+1)
		mask := make([]byte, simd.MaskLen(n))

		b.Run(fmt.Sprintf("n=%d/one-byte/offsets", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkN = simd.IndexAll(idx, buf, '"')
			}
		})
		b.Run(fmt.Sprintf("n=%d/one-byte/bits", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				simd.MaskBits(mask, buf, '"')
			}
		})
		b.Run(fmt.Sprintf("n=%d/set/offsets", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkN = simd.IndexAllAny(idx, buf, set)
			}
		})
		b.Run(fmt.Sprintf("n=%d/set/bits", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				simd.MaskBitsAny(mask, buf, set)
			}
		})
		b.Run(fmt.Sprintf("n=%d/less/bits", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				simd.MaskBitsLess(mask, buf, 0x20)
			}
		})
	}
}
