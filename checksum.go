package simd

// Adler32 returns RFC 1950's checksum of p, continued from seed; pass 1
// to start a new sum, exactly as hash/adler32 does. Measured 7.2x the
// standard library from 1 KB up (36.4 against 5.1 GB/s on amd64/avx512,
// minimum of three) and 2x at 64 bytes: two vector adds per sixteen
// bytes, the weighted structure recovered once per window edge, on every
// tier where hash/adler32 runs one byte at a time.
func Adler32[S Text](p S, seed uint32) uint32 {
	return tblBytesAdler32[tierIdx](textBytes(p), seed)
}

// CRC32C returns the Castagnoli CRC of p, continued from seed; pass 0 to
// start. It matches hash/crc32 with the Castagnoli table bit for bit.
//
// Measured: the standard library's three-stream amd64 assembly wins from
// 1 KB up -- 37 against this kernel's 20 GB/s -- and this kernel wins
// only below ~128 bytes (16.8 against 13.2 GB/s at 64). Use hash/crc32
// for bulk hashing on amd64; this exists for small inputs, for rolling
// seeds alongside other kernels, and as the portable specification.
// docs/wrong.md holds the measurement.
func CRC32C[S Text](p S, seed uint32) uint32 {
	return tblBytesCRC32C[tierIdx](textBytes(p), seed)
}
