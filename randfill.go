package simd

// RandFillU64 fills dst with a deterministic pseudo-random sequence:
// eight xoshiro256++ streams seeded from seed by splitmix64, lane-
// interleaved, identical on every tier. Bulk generation for simulations,
// test data and sampling -- NOT cryptographic, and the same seed always
// gives the same slice.
func RandFillU64(dst []uint64, seed uint64) {
	tblBytesRandFillU64[tierIdx](dst, seed)
}
