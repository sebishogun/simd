package simd_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

func BenchmarkLZ4BlockDecode(b *testing.B) {
	rng := rand.New(rand.NewSource(6))
	long := make([]byte, 1<<20)
	for i := range long {
		long[i] = byte(rng.Intn(4))
	}
	jsonish := bytes.Repeat([]byte(`{"key":"value","n":12345},`), 40000)
	for _, tc := range []struct {
		name string
		data []byte
	}{{"compressible-1M", long}, {"jsonish-1M", jsonish[:1000000]}} {
		comp := compressLZ4(tc.data)
		dst := make([]byte, len(tc.data))
		b.Run(tc.name+"/kernel", func(b *testing.B) {
			b.SetBytes(int64(len(tc.data)))
			for b.Loop() {
				sinkInt = simd.LZ4BlockDecode(dst, comp)
			}
		})
		b.Run(tc.name+"/ref", func(b *testing.B) {
			b.SetBytes(int64(len(tc.data)))
			for b.Loop() {
				sinkInt = ref.LZ4BlockDecode(dst, comp)
			}
		})
	}
}
