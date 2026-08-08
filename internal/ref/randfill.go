package ref

// RandFillU64 fills dst with the eight-stream xoshiro256++ sequence
// seeded from seed by splitmix64, lane-interleaved. The specification for
// simd_rand_fill_u64; kernel and reference emit the identical stream.
// Not cryptographic.
func RandFillU64(dst []uint64, seed uint64) {
	var s [4][8]uint64
	sm := seed
	for lane := 0; lane < 8; lane++ {
		for w := 0; w < 4; w++ {
			sm += 0x9E3779B97f4A7C15
			z := sm
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
			z = (z ^ (z >> 27)) * 0x94D049BB133111EB
			s[w][lane] = z ^ (z >> 31)
		}
	}
	rotl := func(x uint64, k uint) uint64 { return x<<k | x>>(64-k) }
	i := 0
	for ; i+8 <= len(dst); i += 8 {
		for lane := 0; lane < 8; lane++ {
			dst[i+lane] = rotl(s[0][lane]+s[3][lane], 23) + s[0][lane]
		}
		for lane := 0; lane < 8; lane++ {
			t := s[1][lane] << 17
			s[2][lane] ^= s[0][lane]
			s[3][lane] ^= s[1][lane]
			s[1][lane] ^= s[2][lane]
			s[0][lane] ^= s[3][lane]
			s[2][lane] ^= t
			s[3][lane] = rotl(s[3][lane], 45)
		}
	}
	for lane := 0; i < len(dst); i, lane = i+1, lane+1 {
		dst[i] = rotl(s[0][lane]+s[3][lane], 23) + s[0][lane]
	}
}
