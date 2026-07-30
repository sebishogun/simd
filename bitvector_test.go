package simd_test

import (
	"math/bits"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// naiveRank and naiveSelect are the definitions, walked bit by bit. They are
// what the table-driven versions have to reproduce, including at every word
// boundary — which is where an inclusive-versus-exclusive mistake hides, since
// the two agree everywhere else.
func naiveRank(v []uint64, p int) int {
	n := 0
	for i := 0; i < p && i < len(v)*64; i++ {
		if v[i/64]>>(uint(i)%64)&1 == 1 {
			n++
		}
	}
	return n
}

func naiveSelect(v []uint64, k int) int {
	for i := range len(v) * 64 {
		if v[i/64]>>(uint(i)%64)&1 == 1 {
			if k == 0 {
				return i
			}
			k--
		}
	}
	return -1
}

func TestRankAndSelect(t *testing.T) {
	r := rand.New(rand.NewPCG(51, 52))
	for _, nw := range []int{1, 2, 7, 8, 17, 64, 100} {
		// Three densities: nearly empty, half, nearly full. An all-ones vector
		// is the case where rank equals the position and a mistake cancels.
		for _, density := range []int{1, 50, 99, 100, 0} {
			v := make([]uint64, nw)
			for i := range v {
				var w uint64
				for b := range 64 {
					if r.IntN(100) < density {
						w |= 1 << uint(b)
					}
				}
				v[i] = w
			}

			table := make([]uint64, nw+1)
			simd.RankTableInto(table, v)

			total := 0
			for _, w := range v {
				total += bits.OnesCount64(w)
			}
			if int(table[nw]) != total {
				t.Fatalf("nw=%d d=%d: table total %d, want %d",
					nw, density, table[nw], total)
			}

			// Every word boundary and a scattering of interior positions.
			var ps []int
			for w := 0; w <= nw; w++ {
				ps = append(ps, w*64)
				if w < nw {
					ps = append(ps, w*64+1, w*64+31, w*64+63)
				}
			}
			ps = append(ps, -1, 0, nw*64+1, nw*64+1000)
			for _, p := range ps {
				if got, want := simd.Rank(v, table, p), naiveRank(v, p); got != want {
					t.Fatalf("nw=%d d=%d: Rank(%d) = %d, want %d",
						nw, density, p, got, want)
				}
			}

			for _, k := range []int{-1, 0, 1, total / 2, total - 1, total, total + 5} {
				if got, want := simd.Select(v, table, k), naiveSelect(v, k); got != want {
					t.Fatalf("nw=%d d=%d: Select(%d) = %d, want %d",
						nw, density, k, got, want)
				}
			}
		}
	}
}

// Rank and Select must invert each other, which is the property everything
// built on them relies on and the one a table off by one breaks.
func TestRankSelectInvert(t *testing.T) {
	r := rand.New(rand.NewPCG(53, 54))
	v := make([]uint64, 200)
	for i := range v {
		v[i] = r.Uint64()
	}
	table := make([]uint64, len(v)+1)
	simd.RankTableInto(table, v)

	total := int(table[len(v)])
	for k := range total {
		p := simd.Select(v, table, k)
		if p < 0 {
			t.Fatalf("Select(%d) = -1 with %d bits set", k, total)
		}
		if v[p/64]>>(uint(p)%64)&1 != 1 {
			t.Fatalf("Select(%d) = %d, which is not a set bit", k, p)
		}
		if got := simd.Rank(v, table, p); got != k {
			t.Fatalf("Rank(Select(%d)) = %d", k, got)
		}
	}
}

func TestRankTablePanicsOnWrongLength(t *testing.T) {
	v := make([]uint64, 4)
	for _, n := range []int{0, 4, 6} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("len(dst)=%d did not panic", n)
				}
			}()
			simd.RankTableInto(make([]uint64, n), v)
		}()
	}
}

func TestRankTableNoAlloc(t *testing.T) {
	v := make([]uint64, 4096)
	table := make([]uint64, len(v)+1)
	if n := testing.AllocsPerRun(20, func() { simd.RankTableInto(table, v) }); n != 0 {
		t.Errorf("RankTableInto allocated %v times per run", n)
	}
}
