package simd_test

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/sebishogun/simd"
)

func BenchmarkFormatFloat64(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	vals := make([]float64, 4096)
	for i := range vals {
		vals[i] = rng.NormFloat64() * 1e6
	}
	var dst [32]byte
	b.Run("simd", func(b *testing.B) {
		i := 0
		for b.Loop() {
			sinkInt = simd.FormatFloat64(dst[:], vals[i&4095])
			i++
		}
	})
	b.Run("strconv", func(b *testing.B) {
		i := 0
		for b.Loop() {
			sinkInt = len(strconv.AppendFloat(dst[:0], vals[i&4095], 'f', -1, 64))
			i++
		}
	})
	whole := make([]float64, 4096)
	for i := range whole {
		whole[i] = math.Trunc(rng.NormFloat64() * 1e6)
	}
	b.Run("simd-whole", func(b *testing.B) {
		i := 0
		for b.Loop() {
			sinkInt = simd.FormatFloat64(dst[:], whole[i&4095])
			i++
		}
	})
}

var sinkInt int
