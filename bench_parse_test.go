package simd_test

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
