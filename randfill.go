package simd

// RandFillU64 fills dst with a deterministic pseudo-random sequence:
// eight xoshiro256++ streams seeded from seed by splitmix64, lane-
// interleaved, identical on every tier. Bulk generation for simulations,
// test data and sampling -- NOT cryptographic, and the same seed always
// gives the same slice.
func RandFillU64(dst []uint64, seed uint64) {
	tblBytesRandFillU64[tierIdx](dst, seed)
}

// HashUint64 mixes each key through the seeded splitmix64 finalizer:
// bulk hashing for bloom filters, partitioning and dictionary probes
// over columns of integer keys. dst and keys must be the same length.
// Hashing one string belongs to hash/maphash, whose AES path this does
// not try to beat; this exists for the shape maphash cannot batch.
func HashUint64(dst, keys []uint64, seed uint64) {
	tblBytesHashU64[tierIdx](dst, keys, seed)
}
