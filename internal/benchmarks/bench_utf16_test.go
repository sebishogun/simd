package benchmarks

// The baseline is what a []byte caller writes today: utf16.Encode over a
// []rune, which means []byte -> []rune -> []uint16, two passes and two
// allocations for one conversion. The question this measures is whether the
// intermediate rune slice is the cost, or whether the decoding is.
//
// Corpora are chosen by ASCII fraction, because that is the variable: an
// all-ASCII input is a pure widening and vectorizes, while dense CJK is a
// dependent scan that cannot.

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/sebishogun/simd"
)

var (
	sinkU16      []uint16
	sinkU16Bytes []byte
	sinkRune     []rune
)

// utf16Corpus returns n bytes of text with roughly the given fraction of
// non-ASCII runes, mixing 2-byte, 3-byte and 4-byte forms so the surrogate
// path is exercised too.
func utf16Corpus(n int, nonASCII float64) []byte {
	var sb strings.Builder
	i := 0
	for sb.Len() < n {
		if nonASCII > 0 && float64(i%100) < nonASCII*100 {
			switch i % 3 {
			case 0:
				sb.WriteRune('é') // 2 bytes, 1 unit
			case 1:
				sb.WriteRune('世') // 3 bytes, 1 unit
			default:
				sb.WriteRune('𝄞') // 4 bytes, SURROGATE PAIR
			}
		} else {
			sb.WriteByte(byte('a' + i%26))
		}
		i++
	}
	return []byte(sb.String())
}

func BenchmarkUTF16Baseline(b *testing.B) {
	for _, frac := range []float64{0, 0.1, 1.0} {
		for _, n := range []int{1024, 65536} {
			src := utf16Corpus(n, frac)
			name := fmt.Sprintf("nonascii=%.0f%%/n=%d", frac*100, n)

			// What a caller writes today, end to end.
			b.Run(name+"/impl=stdlib", func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for b.Loop() {
					sinkU16 = utf16.Encode([]rune(string(src)))
				}
			})

			// The same thing with the destination preallocated, which is the
			// fairest baseline for an Append-style API: it removes the growth
			// but keeps the []rune round trip.
			runes := []rune(string(src))
			dst := make([]uint16, 0, len(src)*2)
			b.Run(name+"/impl=stdlib_prealloc", func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for b.Loop() {
					sinkU16 = utf16.AppendRune(dst[:0], 0)[:0]
					for _, r := range runes {
						sinkU16 = utf16.AppendRune(sinkU16, r)
					}
				}
			})

			// And the decode direction, for reference.
			enc := utf16.Encode(runes)
			b.Run(name+"/impl=stdlib_decode", func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for b.Loop() {
					sinkRune = utf16.Decode(enc)
					sinkU16Bytes = []byte(string(sinkRune))
				}
			})

			// The floor: how fast can the input even be validated? If the
			// conversion lands near this, there is nothing left to win.
			b.Run(name+"/impl=utf8_scan_only", func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for b.Loop() {
					sinkTextInt = utf8.RuneCount(src)
				}
			})

			// This package, both directions, with the destination reused —
			// which is the point of the append convention and the only fair
			// comparison against stdlib_prealloc.
			u16 := make([]uint16, 0, len(src))
			b.Run(name+"/impl=simd", func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for b.Loop() {
					sinkU16 = simd.AppendUTF16(u16[:0], src)
				}
			})
			encoded := simd.AppendUTF16(nil, src)
			u8 := make([]byte, 0, len(src)+8)
			b.Run(name+"/impl=simd_decode", func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for b.Loop() {
					sinkU16Bytes = simd.AppendUTF8(u8[:0], encoded)
				}
			})
		}
	}
}
