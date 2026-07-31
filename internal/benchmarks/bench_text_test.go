package benchmarks

// Benchmarks for the text scanners, against the standard library rather than
// against the portable build.
//
// That is the right comparison here and it is a harder one. `bytes` and
// `strings` are not naive Go: IndexByte is already assembly on amd64 and
// arm64, Index is a tuned Rabin-Karp with an assembly inner loop, and Equal is
// memequal. A kernel that beats a plain Go loop proves nothing about whether a
// caller should switch. The question this file answers is whether they should.
//
//	go test -run '^$' -bench Text -count 10 | benchstat -col /impl -
//
// The sizes are a short line, a long line, and a buffer that does not fit in
// L2 — the three shapes a log or CSV processor actually sees.

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/sebishogun/simd"
)

// textCorpus is prose-shaped rather than random, because the branch predictor
// and the candidate filter both behave differently on data with structure.
func textCorpus(n int) string {
	const unit = "the quick brown fox jumps over the lazy dog, and then it does so again; "
	var b strings.Builder
	for b.Len() < n {
		b.WriteString(unit)
	}
	return b.String()[:n]
}

var textSizes = []int{64, 4096, 1 << 20}

func benchText(b *testing.B, name string, simdFn, stdFn func() int) {
	b.Run(name+"/impl=simd", func(b *testing.B) {
		for b.Loop() {
			sinkTextInt = simdFn()
		}
	})
	b.Run(name+"/impl=std", func(b *testing.B) {
		for b.Loop() {
			sinkTextInt = stdFn()
		}
	})
}

func BenchmarkTextIndexByte(b *testing.B) {
	for _, n := range textSizes {
		s := textCorpus(n)
		// A byte that does not occur, so the whole input is scanned. Stopping
		// early would measure the corpus rather than the kernel.
		benchText(b, fmt.Sprintf("n=%d", n),
			func() int { return simd.IndexByte(s, '~') },
			func() int { return strings.IndexByte(s, '~') })
	}
}

func BenchmarkTextIndex(b *testing.B) {
	for _, n := range textSizes {
		s := textCorpus(n)
		needle := "zebra crossing"
		benchText(b, fmt.Sprintf("n=%d", n),
			func() int { return simd.Index(s, needle) },
			func() int { return strings.Index(s, needle) })
	}
}

func BenchmarkTextLastIndex(b *testing.B) {
	for _, n := range textSizes {
		s := textCorpus(n)
		needle := "zebra crossing"
		benchText(b, fmt.Sprintf("n=%d", n),
			func() int { return simd.LastIndex(s, needle) },
			func() int { return strings.LastIndex(s, needle) })
	}
}

// BenchmarkTextCount is measured for two needles, because the answer differs
// and averaging them would hide both. "the" occurs roughly once per ten bytes
// of this corpus, so the candidate filter fires in every block and the work is
// verification; "zebra" never occurs, so the filter rejects whole registers at
// a time and the work is the scan.
func BenchmarkTextCount(b *testing.B) {
	for _, n := range textSizes {
		s := textCorpus(n)
		for _, needle := range []string{"the", "zebra"} {
			benchText(b, fmt.Sprintf("needle=%s/n=%d", needle, n),
				func() int { return simd.Count(s, needle) },
				func() int { return strings.Count(s, needle) })
		}
	}
}

func BenchmarkTextIndexAny(b *testing.B) {
	for _, n := range textSizes {
		s := textCorpus(n)
		const set = "~^`|"
		benchText(b, fmt.Sprintf("n=%d", n),
			func() int { return simd.IndexAny(s, set) },
			func() int { return strings.IndexAny(s, set) })
	}
}

// BenchmarkTextCountAny has no strings counterpart at all — the closest is a
// loop over IndexAny — so this one is measured against the portable build
// instead, like the numeric kernels.
func BenchmarkTextCountAny(b *testing.B) {
	for _, n := range textSizes {
		s := textCorpus(n)
		const set = "aeiou"
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextInt = simd.CountAny(s, set)
			}
		})
	}
}

func BenchmarkTextTrimSpace(b *testing.B) {
	for _, n := range textSizes {
		// Whitespace at both ends, which is the shape trimming is for, with
		// enough of it that the scan is not entirely prologue.
		s := strings.Repeat(" ", 64) + textCorpus(n) + strings.Repeat(" \t", 32)
		b.Run(fmt.Sprintf("n=%d/impl=simd", n), func(b *testing.B) {
			for b.Loop() {
				sinkTextStr = simd.TrimSpaceASCII(s)
			}
		})
		b.Run(fmt.Sprintf("n=%d/impl=std", n), func(b *testing.B) {
			for b.Loop() {
				sinkTextStr = strings.TrimSpace(s)
			}
		})
	}
}

func BenchmarkTextValidUTF8(b *testing.B) {
	for _, n := range textSizes {
		s := textCorpus(n)
		b.Run(fmt.Sprintf("n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextBool = simd.ValidUTF8(s)
			}
		})
		b.Run(fmt.Sprintf("n=%d/impl=std", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextBool = utf8.ValidString(s)
			}
		})
	}
}

// BenchmarkTextIndexAll is the structural-index step, which has no standard
// library counterpart because nothing there produces one. The comparison is
// against the obvious loop over IndexByte, which is what a caller writes
// today.
func BenchmarkTextIndexAll(b *testing.B) {
	for _, n := range textSizes {
		s := textCorpus(n)
		bs := []byte(s)
		dst := make([]int32, n)
		b.Run(fmt.Sprintf("n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextInt = simd.IndexAll(dst, s, ' ')
			}
		})
		b.Run(fmt.Sprintf("n=%d/impl=std", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				k, off := 0, 0
				for {
					i := bytes.IndexByte(bs[off:], ' ')
					if i < 0 || k == len(dst) {
						break
					}
					dst[k] = int32(off + i)
					k++
					off += i + 1
				}
				sinkTextInt = k
			}
		})
	}
}

// BenchmarkTextBase64 is against encoding/base64, which is a byte-at-a-time
// state machine — the comparison a caller actually faces, and the reason
// base64 is one of the standard examples of what a vector unit is for.
func BenchmarkTextBase64(b *testing.B) {
	for _, n := range textSizes {
		src := []byte(textCorpus(n))
		dst := make([]byte, simd.Base64EncodedLen(n))
		enc := base64.StdEncoding.EncodeToString(src)
		out := make([]byte, simd.Base64DecodedLen(len(enc)))
		b.Run(fmt.Sprintf("Encode/n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextInt = simd.Base64Encode(dst, src)
			}
		})
		b.Run(fmt.Sprintf("Encode/n=%d/impl=std", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				base64.StdEncoding.Encode(dst, src)
				sinkTextInt = len(dst)
			}
		})
		b.Run(fmt.Sprintf("Decode/n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextInt = simd.Base64Decode(out, enc)
			}
		})
		b.Run(fmt.Sprintf("Decode/n=%d/impl=std", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				k, _ := base64.StdEncoding.Decode(out, []byte(enc))
				sinkTextInt = k
			}
		})
	}
}

// BenchmarkTextHex is against encoding/hex, which decodes a nibble at a time
// through a 256-entry table and validates per character. Decode is the
// interesting half: it returns a count and a validity flag, so until the
// generator learned two-value returns it was portable on every architecture
// for a reason that had nothing to do with the hardware.
func BenchmarkTextHex(b *testing.B) {
	for _, n := range textSizes {
		src := []byte(textCorpus(n))
		enc := make([]byte, hex.EncodedLen(n))
		hex.Encode(enc, src)
		out := make([]byte, n)
		b.Run(fmt.Sprintf("Encode/n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextInt = simd.HexEncode(enc, src)
			}
		})
		b.Run(fmt.Sprintf("Encode/n=%d/impl=std", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextInt = hex.Encode(enc, src)
			}
		})
		b.Run(fmt.Sprintf("Decode/n=%d/impl=simd", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkTextInt, _ = simd.HexDecode(out, enc)
			}
		})
		b.Run(fmt.Sprintf("Decode/n=%d/impl=std", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				k, _ := hex.Decode(out, enc)
				sinkTextInt = k
			}
		})
	}
}
