package benchmarks

import (
	"encoding/binary"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// Against the loop each one replaces, at a size where the call overhead has
// long since disappeared.

func BenchmarkCommonPrefixLen(b *testing.B) {
	// A long shared prefix is the case worth measuring: it is what suffix
	// arrays and trie descent produce, and it is the case the blocked scan is
	// built for. A pair that differs in the first byte is a threshold check,
	// not a throughput one.
	const n = 1 << 20
	x := make([]byte, n)
	r := rand.New(rand.NewPCG(1, 2))
	for i := range x {
		x[i] = byte(r.UintN(256))
	}
	y := append([]byte(nil), x...)
	y[n-1] ^= 0xff

	b.Run("scalar", func(b *testing.B) {
		b.SetBytes(n)
		for b.Loop() {
			k := 0
			for k < len(x) && x[k] == y[k] {
				k++
			}
			_ = k
		}
	})
	b.Run("simd", func(b *testing.B) {
		b.SetBytes(n)
		for b.Loop() {
			_ = simd.CommonPrefixLen(x, y)
		}
	})
}

func BenchmarkRollingMin(b *testing.B) {
	const n = 1 << 20
	a := make([]float64, n)
	r := rand.New(rand.NewPCG(3, 4))
	for i := range a {
		a[i] = r.NormFloat64()
	}

	// The window is the axis that matters, not n: the kernel does window-1
	// vectorized passes where the deque does two comparisons per element
	// regardless. Measuring one window would misreport it in either direction.
	for _, w := range []int{4, 8, 16, 32, 64, 256} {
		dst := make([]float64, n-w+1)
		b.Run("deque/w="+itoa(w), func(b *testing.B) {
			idx := make([]int, 0, w)
			for b.Loop() {
				idx = idx[:0]
				for i := range a {
					for len(idx) > 0 && a[idx[len(idx)-1]] >= a[i] {
						idx = idx[:len(idx)-1]
					}
					idx = append(idx, i)
					if idx[0] <= i-w {
						idx = idx[1:]
					}
					if i >= w-1 {
						dst[i-w+1] = a[idx[0]]
					}
				}
			}
		})
		b.Run("simd/w="+itoa(w), func(b *testing.B) {
			for b.Loop() {
				simd.RollingMinInto(dst, a, w)
			}
		})
	}
}

func BenchmarkVarintSize(b *testing.B) {
	const n = 1 << 20
	a := make([]uint64, n)
	r := rand.New(rand.NewPCG(5, 6))
	for i := range a {
		a[i] = r.Uint64() >> (r.UintN(64))
	}

	b.Run("scalar", func(b *testing.B) {
		b.SetBytes(n * 8)
		for b.Loop() {
			t := 0
			for _, v := range a {
				k := 1
				for v >= 0x80 {
					v >>= 7
					k++
				}
				t += k
			}
			_ = t
		}
	})
	b.Run("simd", func(b *testing.B) {
		b.SetBytes(n * 8)
		for b.Loop() {
			_ = simd.VarintSize(a)
		}
	})
}

// The encoder comparison is the one that answers "is this worth using": the
// baseline is what an ordinary Go encoder does, appending as it goes, and the
// difference is that the vectorized size pass lets the buffer be allocated
// once at exactly the right size.
func BenchmarkAppendVarints(b *testing.B) {
	const n = 1 << 18
	a := make([]uint64, n)
	r := rand.New(rand.NewPCG(7, 8))
	for i := range a {
		a[i] = r.Uint64() >> (r.UintN(64))
	}

	b.Run("append-and-grow", func(b *testing.B) {
		buf := make([]byte, binary.MaxVarintLen64)
		for b.Loop() {
			var out []byte
			for _, v := range a {
				out = append(out, buf[:binary.PutUvarint(buf, v)]...)
			}
			_ = out
		}
	})
	b.Run("simd", func(b *testing.B) {
		for b.Loop() {
			_ = simd.AppendVarints(nil, a)
		}
	})
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d [8]byte
	i := len(d)
	for n > 0 {
		i--
		d[i] = byte('0' + n%10)
		n /= 10
	}
	return string(d[i:])
}

// The set operations against the two-cursor merge they replace.
//
// Selectivity is the axis that matters. Block skipping retires a whole block of
// eight per tile when the sets are sparse relative to each other, and neither
// side when they interleave densely, so a single-density benchmark misreports
// this in either direction — the same reason CompressInto is benchmarked across
// match densities rather than at one.
func BenchmarkIntersect(b *testing.B) {
	const n = 1 << 18
	r := rand.New(rand.NewPCG(9, 10))
	a := make([]int32, n)
	for i := range a {
		a[i] = int32(i) * 2 // sorted, distinct, no duplicates
	}

	for _, share := range []int{1, 10, 50, 100} {
		other := make([]int32, n)
		for i := range other {
			if r.IntN(100) < share {
				other[i] = a[i] // a match
			} else {
				other[i] = a[i] + 1 // a miss, still sorted and distinct
			}
		}
		dst := make([]int32, n)

		b.Run("merge/"+itoa(share)+"pct", func(b *testing.B) {
			for b.Loop() {
				k := 0
				for i, j := 0, 0; i < len(a) && j < len(other); {
					switch {
					case a[i] < other[j]:
						i++
					case other[j] < a[i]:
						j++
					default:
						dst[k] = a[i]
						k++
						i++
						j++
					}
				}
				_ = k
			}
		})
		b.Run("simd/"+itoa(share)+"pct", func(b *testing.B) {
			for b.Loop() {
				_ = simd.IntersectInto(dst, a, other)
			}
		})
	}
}

// The batch lower bound against the scalar bisection, per tier.
//
// Measured per tier on purpose. The inner loop needs a gather, and a target
// without one can still have LLVM "vectorize" the loop by scalarizing the
// gather into a load and an insert per lane — vector-shaped code that is
// slower than the scalar bisection it replaced, which the repository's
// has-vector-instructions gate cannot tell apart from a real win. That is
// docs/wrong.md entry 59, and this benchmark is how the SSE2 case gets an
// answer rather than an assumption.
func BenchmarkLowerBound(b *testing.B) {
	const ntab = 1 << 16
	const nq = 1 << 14
	table := make([]int32, ntab)
	for i := range table {
		table[i] = int32(i) * 3
	}
	r := rand.New(rand.NewPCG(13, 14))
	q := make([]int32, nq)
	for i := range q {
		q[i] = int32(r.IntN(ntab * 3))
	}
	dst := make([]int32, nq)

	b.Run("bisect", func(b *testing.B) {
		for b.Loop() {
			for i, v := range q {
				lo, hi := 0, len(table)
				for lo < hi {
					mid := int(uint(lo+hi) >> 1)
					if table[mid] < v {
						lo = mid + 1
					} else {
						hi = mid
					}
				}
				dst[i] = int32(lo)
			}
		}
	})
	b.Run("simd", func(b *testing.B) {
		for b.Loop() {
			simd.LowerBoundInto(dst, table, q)
		}
	})
}

// SparseDot per tier, against the scalar loop it replaces.
//
// Gather-bound, so this is the shape entry 59 warns about: a target without a
// gather instruction can still have LLVM build the vector lane by lane, which
// passes the repository's has-vector-instructions gate while being slower than
// the loop it replaced. Measuring per tier is the only way to tell, and it is
// why this benchmark exists rather than an assumption.
//
// Row length is the axis. A finite-element or graph-adjacency row is tens to
// hundreds of nonzeros; below the dispatch threshold the call cost dominates
// and the portable path is what runs.
func BenchmarkSparseDot(b *testing.B) {
	r := rand.New(rand.NewPCG(15, 16))
	x := make([]float64, 1<<16)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	for _, nnz := range []int{16, 64, 256, 4096} {
		v := make([]float64, nnz)
		idx := make([]int32, nnz)
		for i := range v {
			v[i] = r.NormFloat64()
			idx[i] = int32(r.IntN(len(x)))
		}
		b.Run("scalar/nnz="+itoa(nnz), func(b *testing.B) {
			for b.Loop() {
				var acc [16]float64
				for i := range v {
					acc[i%16] += v[i] * x[idx[i]]
				}
				for w := 8; w >= 1; w /= 2 {
					for j := range w {
						acc[j] += acc[j+w]
					}
				}
				_ = acc[0]
			}
		})
		b.Run("simd/nnz="+itoa(nnz), func(b *testing.B) {
			for b.Loop() {
				_ = simd.SparseDot(v, idx, x)
			}
		})
	}
}
