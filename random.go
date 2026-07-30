package simd

// RandomInto fills dst with uniformly distributed values from a counter-based
// generator seeded by seed.
//
// float32 and float64 get values in [0, 1); uint64 gets the full range. It
// allocates nothing.
//
// # What "counter-based" buys, and why it is not the usual design
//
// Element i depends on the seed and on i, and on nothing else. A conventional
// generator — xorshift, PCG, Mersenne Twister — threads a state through the
// loop, so element i+1 cannot begin until element i has finished. That is a
// serial dependence and it cannot be vectorized at all.
//
// Computing each element from its index instead makes the loop elementwise,
// and three useful properties fall out of the same change:
//
//		simd.RandomInto(buf, 42)          // the same bytes on every architecture
//		simd.RandomInto(buf[1000:], 42)   // the same bytes as the tail of the above
//
//	  - **The same stream everywhere.** This is integer arithmetic with no
//	    accumulation order, so rule 2 applies and there is nothing to negotiate.
//	    A simulation seeded the same way gives the same answer on your laptop and
//	    on an ARM server.
//	  - **No state to carry.** Filling a window gives the same bytes whether or
//	    not the preceding window was filled, which is what makes a checkpointed
//	    run resumable without serialising a generator.
//	  - **Splittable.** Two goroutines filling disjoint halves produce exactly
//	    what one goroutine filling the whole would.
//
// # What it is not
//
// Not cryptographic. The mixing is splitmix64's finalizer, which passes
// BigCrush and is the standard choice for simulation and initialisation, but
// an attacker who sees output can recover the seed. Use crypto/rand where that
// matters.
//
// Not a drop-in for math/rand: the stream is different, and deliberately so —
// math/rand's is neither reproducible across Go versions nor vectorizable.
func RandomInto[T float32 | float64 | uint64](dst []T, seed uint64) {
	ops[T]().Random(dst, seed)
}
