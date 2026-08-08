package simd_test

import (
	"hash/adler32"
	"hash/crc32"
	"testing"

	"github.com/sebishogun/simd"
)

// The ship-verdict benchmark: stdlib is the incumbent, and for CRC32C its
// amd64 assembly is strong; the doc comment sends callers there unless a
// number says otherwise.
func BenchmarkChecksum(b *testing.B) {
	castagnoli := crc32.MakeTable(crc32.Castagnoli)
	for _, n := range []int{64, 1024, 65536, 1 << 20} {
		p := make([]byte, n)
		for i := range p {
			p[i] = byte(i * 131)
		}
		name := map[int]string{64: "64", 1024: "1K", 65536: "64K", 1 << 20: "1M"}[n]
		b.Run("Adler32/"+name, func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkU32 = simd.Adler32(p, 1)
			}
		})
		b.Run("Adler32-stdlib/"+name, func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkU32 = adler32.Checksum(p)
			}
		})
		b.Run("CRC32C/"+name, func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkU32 = simd.CRC32C(p, 0)
			}
		})
		b.Run("CRC32C-stdlib/"+name, func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				sinkU32 = crc32.Checksum(p, castagnoli)
			}
		})
	}
}

var sinkU32 uint32
