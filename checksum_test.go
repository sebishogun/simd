package simd_test

import (
	"hash/adler32"
	"hash/crc32"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

func TestChecksumsMatchStdlib(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	castagnoli := crc32.MakeTable(crc32.Castagnoli)
	for _, n := range []int{0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 300, 4096, 100000} {
		p := make([]byte, n)
		rng.Read(p)
		if got, want := simd.Adler32(p, 1), adler32.Checksum(p); got != want {
			t.Fatalf("Adler32 n=%d got %08x want %08x", n, got, want)
		}
		if got, want := simd.CRC32C(p, 0), crc32.Checksum(p, castagnoli); got != want {
			t.Fatalf("CRC32C n=%d got %08x want %08x", n, got, want)
		}
		// Rolling: two halves must equal the whole.
		h := n / 2
		a := simd.Adler32(p[h:], simd.Adler32(p[:h], 1))
		if want := adler32.Checksum(p); a != want {
			t.Fatalf("rolling Adler32 n=%d got %08x want %08x", n, a, want)
		}
		c := simd.CRC32C(p[h:], simd.CRC32C(p[:h], 0))
		if want := crc32.Checksum(p, castagnoli); c != want {
			t.Fatalf("rolling CRC32C n=%d got %08x want %08x", n, c, want)
		}
	}
}
