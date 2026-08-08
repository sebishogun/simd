package ref

// Adler32 is RFC 1950's checksum, seedable for rolling use; seed 1 is the
// standard start. The specification for simd_adler32.
func Adler32(p []byte, seed uint32) uint32 {
	const mod = 65521
	a, b := seed&0xffff, seed>>16
	for len(p) > 0 {
		n := len(p)
		if n > 5552 {
			n = 5552
		}
		for _, c := range p[:n] {
			a += uint32(c)
			b += a
		}
		a %= mod
		b %= mod
		p = p[n:]
	}
	return b<<16 | a
}

// crc32cNibble is the reflected Castagnoli polynomial, four bits at a
// time: sixteen entries beat a 1 KB table for a reference whose job is to
// be obviously right.
var crc32cNibble = func() [16]uint32 {
	var t [16]uint32
	for i := range t {
		c := uint32(i)
		for k := 0; k < 4; k++ {
			c = c>>1 ^ (0x82F63B78 & (0 - (c & 1)))
		}
		t[i] = c
	}
	return t
}()

// CRC32C is the Castagnoli CRC, matching hash/crc32's Castagnoli table
// and the crc32c instruction family. The specification for simd_crc32c.
func CRC32C(p []byte, seed uint32) uint32 {
	crc := ^seed
	for _, c := range p {
		crc ^= uint32(c)
		crc = crc>>4 ^ crc32cNibble[crc&0xf]
		crc = crc>>4 ^ crc32cNibble[crc&0xf]
	}
	return ^crc
}
