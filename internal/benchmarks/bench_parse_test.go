package benchmarks

// Against IndexAll + strconv.Atoi, which is what a caller writes today and is
// already using this package for the scan. If ParseInts is not clearly ahead
// of that, it does not earn its place.

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkParse int64

func parseCorpus(fields int) ([]byte, []int32) {
	p := make([]string, fields)
	for i := range p {
		p[i] = strconv.Itoa(i*7919 - 500000)
	}
	line := []byte(strings.Join(p, ","))
	idx := make([]int32, 0, fields)
	for i, c := range line {
		if c == ',' {
			idx = append(idx, int32(i))
		}
	}
	return line, append(idx, int32(len(line)))
}

func BenchmarkParseInts(b *testing.B) {
	for _, fields := range []int{64, 4096, 200000} {
		line, idx := parseCorpus(fields)
		dst := make([]int64, fields)
		b.Run(fmt.Sprintf("fields=%d/impl=simd", fields), func(b *testing.B) {
			b.SetBytes(int64(len(line)))
			for b.Loop() {
				n, _ := simd.ParseInts(dst, line, idx)
				sinkParse = int64(n)
			}
		})
		b.Run(fmt.Sprintf("fields=%d/impl=strconv", fields), func(b *testing.B) {
			b.SetBytes(int64(len(line)))
			for b.Loop() {
				var t int64
				prev := 0
				for _, e := range idx {
					v, _ := strconv.ParseInt(string(line[prev:e]), 10, 64)
					t += v
					prev = int(e) + 1
				}
				sinkParse = t
			}
		})
	}
}

func BenchmarkFormatInts(b *testing.B) {
	vals := make([]int64, 200000)
	for i := range vals {
		vals[i] = int64(i*7919 - 500000)
	}
	dst := make([]byte, 21*len(vals))
	b.Run("impl=simd", func(b *testing.B) {
		var n int
		for b.Loop() {
			n = simd.FormatInts(dst, vals, ',')
		}
		b.SetBytes(int64(n))
		sinkParse = int64(n)
	})
	b.Run("impl=strconv", func(b *testing.B) {
		var out []byte
		for b.Loop() {
			out = out[:0]
			for i, v := range vals {
				out = strconv.AppendInt(out, v, 10)
				if i != len(vals)-1 {
					out = append(out, ',')
				}
			}
		}
		b.SetBytes(int64(len(out)))
		sinkParse = int64(len(out))
	})
}

// parseUintCorpus mirrors parseCorpus but spreads values across the whole
// uint64 range, so the 20-digit path — the one the signed kernel cannot reach
// at all — is a large share of the fields rather than a curiosity.
func parseUintCorpus(fields int) ([]byte, []int32) {
	p := make([]string, fields)
	for i := range p {
		p[i] = strconv.FormatUint(uint64(i)*3689348814741910323, 10)
	}
	line := []byte(strings.Join(p, ","))
	idx := make([]int32, 0, fields)
	for i, c := range line {
		if c == ',' {
			idx = append(idx, int32(i))
		}
	}
	return line, append(idx, int32(len(line)))
}

func BenchmarkParseUints(b *testing.B) {
	for _, fields := range []int{64, 4096, 200000} {
		line, idx := parseUintCorpus(fields)
		dst := make([]uint64, fields)
		b.Run(fmt.Sprintf("fields=%d/impl=simd", fields), func(b *testing.B) {
			b.SetBytes(int64(len(line)))
			for b.Loop() {
				n, _ := simd.ParseUints(dst, line, idx)
				sinkParse = int64(n)
			}
		})
		b.Run(fmt.Sprintf("fields=%d/impl=strconv", fields), func(b *testing.B) {
			b.SetBytes(int64(len(line)))
			for b.Loop() {
				var t uint64
				prev := 0
				for _, e := range idx {
					v, _ := strconv.ParseUint(string(line[prev:e]), 10, 64)
					t += v
					prev = int(e) + 1
				}
				sinkParse = int64(t)
			}
		})
	}
}
