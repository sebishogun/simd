//go:build amd64 && !purego

package amd64

import (
	"fmt"
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
)

// Where the time actually goes.
//
// Two questions this answers:
//
//  1. At n=32 the public API is only ~1.6x faster than portable Go, while the
//     raw kernel should be several times that. The call chain is three deep —
//     generic dispatch, threshold guard, assembly — and each layer is an
//     un-inlinable call. This measures each layer separately so the cost is
//     attributable rather than guessed at.
//
//  2. Add gets *worse* as n grows past 1024 while Sum keeps winning. The
//     suspicion is memory bandwidth: Add touches three arrays (read a, read b,
//     write dst) for 24 bytes of traffic per element, while Sum touches one
//     for 8. Reporting achieved bandwidth rather than ns/op makes that visible
//     directly — if the number flattens out, the kernel is waiting on memory
//     and no amount of vector width will help.

// callDepth measures each layer of the call chain on the same work.
func BenchmarkCallDepth(b *testing.B) {
	for _, n := range []int{8, 16, 32, 64, 128, 512} {
		x, y, dst := benchInputs(n)

		// Layer 0: the assembly kernel, called directly. No dispatch, no
		// guard, no interface — the floor.
		b.Run(fmt.Sprintf("n=%d/0-raw-asm", n), func(b *testing.B) {
			for b.Loop() {
				addFloat64AVX512(dst, x, y)
			}
		})

		// Layer 1: through the generated threshold guard, which is what the
		// dispatch table actually holds.
		b.Run(fmt.Sprintf("n=%d/1-guarded", n), func(b *testing.B) {
			for b.Loop() {
				AddFloat64AVX512(dst, x, y)
			}
		})

		// Layer 2: through the kernel.Set function pointer, which is the
		// indirect call the generic API makes.
		var set kernel.Set
		set.F64.Add = AddFloat64AVX512
		b.Run(fmt.Sprintf("n=%d/2-indirect", n), func(b *testing.B) {
			for b.Loop() {
				set.F64.Add(dst, x, y)
			}
		})

		// Layer 3: the portable Go loop, for reference.
		b.Run(fmt.Sprintf("n=%d/3-portable", n), func(b *testing.B) {
			for b.Loop() {
				goAdd(dst, x, y)
			}
		})
	}
}

// BenchmarkBandwidth reports achieved memory bandwidth rather than time, at
// sizes chosen to step through this machine's cache hierarchy.
//
// Working-set sizes for float64, per level of the hierarchy:
//
//	Add: 3 arrays x 8 bytes = 24 bytes per element (read a, read b, write dst)
//	Sum: 1 array  x 8 bytes =  8 bytes per element
//
// So Add leaves a 48 KiB L1 at about n=2048 and a 1 MiB L2 at about n=43000,
// while Sum stays in L1 three times longer. If the bandwidth figure stops
// rising with vector width, the kernel is memory-bound and the vector unit is
// idle waiting on loads.
func BenchmarkBandwidth(b *testing.B) {
	sizes := []int{512, 2048, 8192, 43000, 262144, 1048576}
	for _, n := range sizes {
		x, y, dst := benchInputs(n)

		b.Run(fmt.Sprintf("Add/n=%d/avx512", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 24) // two reads and a write
			for b.Loop() {
				addFloat64AVX512(dst, x, y)
			}
		})
		b.Run(fmt.Sprintf("Add/n=%d/sse2", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 24)
			for b.Loop() {
				addFloat64SSE2(dst, x, y)
			}
		})
		b.Run(fmt.Sprintf("Sum/n=%d/avx512", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 8) // one read
			for b.Loop() {
				sinkF = sumFloat64AVX512(x)
			}
		})
		b.Run(fmt.Sprintf("Sum/n=%d/sse2", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 8)
			for b.Loop() {
				sinkF = sumFloat64SSE2(x)
			}
		})
	}
}
