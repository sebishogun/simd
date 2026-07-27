package conformance

// Conformance for the kernels in csrc/numeric.c, whose shapes none of the
// existing checkers cover: a polynomial applied elementwise, two windowed
// products, a repeating copy and an indexed load.
//
// Each of them takes more than one length and reconciles the lengths itself
// rather than leaving it to the dispatcher's guard, so the interesting cases
// are not only "how long" but "which one is shorter". A kernel that used the
// wrong length reads past the end of something, and on a slice that came from
// a caller that is not a test, silently.

import (
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// lenPairs are the relative lengths worth trying wherever two slices meet.
// The equal case is the one that works by accident; the others are where an
// off-by-one lives.
var lenPairs = [][2]int{
	{0, 0}, {0, 8}, {8, 0}, {1, 1}, {1, 8}, {8, 1},
	{7, 8}, {8, 7}, {8, 8}, {15, 16}, {16, 15}, {16, 16},
	{17, 16}, {31, 33}, {64, 64}, {65, 64}, {64, 65}, {129, 128},
}

func checkNumericOps[T comparable](t *testing.T, tier, typeName string,
	got, want kernel.Ops[T], gen func(int, *rand.Rand) []T) {

	p := func(op string) string { return typeName + "." + op }
	r := rand.New(rand.NewPCG(51, 52))

	// PolyEval: dst and x are clamped together, coeffs is independent.
	if got.PolyEval != nil && want.PolyEval != nil {
		for _, lp := range lenPairs {
			for _, nc := range []int{0, 1, 2, 5, 13} {
				a, c := gen(lp[1], r), gen(nc, r)
				g, w := make([]T, lp[0]), make([]T, lp[0])
				got.PolyEval(g, a, c)
				want.PolyEval(w, a, c)
				if i, ok := same(g, w); !ok {
					t.Fatalf("%s dst=%d x=%d coeffs=%d i=%d: got %v want %v",
						p("PolyEval"), lp[0], lp[1], nc, i, g[i], w[i])
				}
			}
		}
	}

	window := func(op string, g, w func(dst, sig, ker []T)) {
		if g == nil || w == nil {
			return
		}
		for _, lp := range lenPairs {
			// A kernel longer than the signal produces nothing, which is its
			// own case; so is a kernel of exactly one element.
			for _, nk := range []int{0, 1, 2, 3, 8, 17} {
				sig, ker := gen(lp[1], r), gen(nk, r)
				gd, wd := make([]T, lp[0]), make([]T, lp[0])
				g(gd, sig, ker)
				w(wd, sig, ker)
				if i, ok := same(gd, wd); !ok {
					t.Fatalf("%s dst=%d sig=%d ker=%d i=%d: got %v want %v",
						p(op), lp[0], lp[1], nk, i, gd[i], wd[i])
				}
			}
		}
	}
	window("Convolve", got.Convolve, want.Convolve)
	window("Correlate", got.Correlate, want.Correlate)

	if got.Tile != nil && want.Tile != nil {
		for _, lp := range lenPairs {
			pat := gen(lp[1], r)
			g, w := make([]T, lp[0]), make([]T, lp[0])
			// Pre-fill, because an empty pattern must leave the destination
			// alone rather than zero it.
			fill := gen(lp[0], r)
			copy(g, fill)
			copy(w, fill)
			got.Tile(g, pat)
			want.Tile(w, pat)
			if i, ok := same(g, w); !ok {
				t.Fatalf("%s dst=%d pattern=%d i=%d: got %v want %v",
					p("Tile"), lp[0], lp[1], i, g[i], w[i])
			}
		}
	}

	if got.Gather != nil && want.Gather != nil {
		for _, lp := range lenPairs {
			src := gen(lp[1], r)
			for _, ni := range []int{0, 1, lp[0], lp[0] + 3} {
				if ni < 0 {
					continue
				}
				// Indices deliberately include negatives and values past the
				// end: those elements must be left untouched, not zeroed, and
				// must certainly not be read.
				idx := make([]int32, ni)
				for i := range idx {
					switch i % 4 {
					case 0:
						idx[i] = int32(i % max(1, lp[1]))
					case 1:
						idx[i] = -1
					case 2:
						idx[i] = int32(lp[1] + 5)
					default:
						idx[i] = int32(max(0, lp[1]-1))
					}
				}
				fill := gen(lp[0], r)
				g, w := make([]T, lp[0]), make([]T, lp[0])
				copy(g, fill)
				copy(w, fill)
				got.Gather(g, src, idx)
				want.Gather(w, src, idx)
				if i, ok := same(g, w); !ok {
					t.Fatalf("%s dst=%d src=%d idx=%d i=%d: got %v want %v",
						p("Gather"), lp[0], lp[1], ni, i, g[i], w[i])
				}
			}
		}
	}
}

// checkMatrixOps covers the three kernels whose arguments are sizes rather
// than only slices, where a wrong bound writes outside the destination.
func checkMatrixOps[T comparable](t *testing.T, tier, typeName string,
	got, want kernel.Ops[T], gen func(int, *rand.Rand) []T) {

	p := func(op string) string { return typeName + "." + op }
	r := rand.New(rand.NewPCG(53, 54))

	if got.Scatter != nil && want.Scatter != nil {
		for _, lp := range lenPairs {
			src := gen(lp[1], r)
			idx := make([]int32, lp[1])
			for i := range idx {
				// Duplicates on purpose: where two indices collide the later
				// one must win, which is what a sequential loop gives and
				// what a hardware scatter would not promise.
				switch i % 5 {
				case 0:
					idx[i] = int32(i % max(1, lp[0]))
				case 1:
					idx[i] = -3
				case 2:
					idx[i] = int32(lp[0] + 2)
				case 3:
					idx[i] = 0
				default:
					idx[i] = int32(max(0, lp[0]-1))
				}
			}
			fill := gen(lp[0], r)
			g, w := make([]T, lp[0]), make([]T, lp[0])
			copy(g, fill)
			copy(w, fill)
			got.Scatter(g, idx, src)
			want.Scatter(w, idx, src)
			if i, ok := same(g, w); !ok {
				t.Fatalf("%s dst=%d src=%d i=%d: got %v want %v",
					p("Scatter"), lp[0], lp[1], i, g[i], w[i])
			}
		}
	}

	if got.MovingAverage != nil && want.MovingAverage != nil {
		for _, lp := range lenPairs {
			a := gen(lp[1], r)
			for _, width := range []int{0, -1, 1, 2, 3, 8, 17, lp[1], lp[1] + 1} {
				fill := gen(lp[0], r)
				g, w := make([]T, lp[0]), make([]T, lp[0])
				copy(g, fill)
				copy(w, fill)
				got.MovingAverage(g, a, width)
				want.MovingAverage(w, a, width)
				if i, ok := same(g, w); !ok {
					t.Fatalf("%s dst=%d a=%d width=%d i=%d: got %v want %v",
						p("MovingAverage"), lp[0], lp[1], width, i, g[i], w[i])
				}
			}
		}
	}

	if got.MatMul != nil && want.MatMul != nil {
		type dims struct{ m, k, n int }
		for _, dm := range []dims{
			{0, 0, 0}, {1, 1, 1}, {1, 4, 1}, {4, 1, 4}, {2, 3, 4}, {3, 3, 3},
			{8, 8, 8}, {5, 7, 9}, {16, 16, 16}, {17, 3, 19}, {-1, 2, 2}, {2, -1, 2},
		} {
			m, k, n := dm.m, dm.k, dm.n
			sz := func(x int) int {
				if x < 0 {
					return 0
				}
				return x
			}
			a, b := gen(sz(m*k), r), gen(sz(k*n), r)
			// Undersized destinations too: the contract is that a call that
			// does not fit writes nothing at all.
			for _, dn := range []int{sz(m * n), max(0, sz(m*n)-1), sz(m*n) + 4} {
				fill := gen(dn, r)
				g, w := make([]T, dn), make([]T, dn)
				copy(g, fill)
				copy(w, fill)
				got.MatMul(g, a, b, m, k, n)
				want.MatMul(w, a, b, m, k, n)
				if i, ok := same(g, w); !ok {
					t.Fatalf("%s m=%d k=%d n=%d dst=%d i=%d: got %v want %v",
						p("MatMul"), m, k, n, dn, i, g[i], w[i])
				}
			}
		}
	}
}

func TestNumericKernels(t *testing.T) {
	want := ref.Set()
	for tier, got := range tiers(t) {
		t.Run(tier, func(t *testing.T) {
			checkNumericOps(t, tier, "F32", got.F32, want.F32, genF32)
			checkNumericOps(t, tier, "F64", got.F64, want.F64, genF64)
			checkNumericOps(t, tier, "I32", got.I32, want.I32, genI32)
			checkNumericOps(t, tier, "I64", got.I64, want.I64, genI64)
			checkMatrixOps(t, tier, "F32", got.F32, want.F32, genF32)
			checkMatrixOps(t, tier, "F64", got.F64, want.F64, genF64)
			checkMatrixOps(t, tier, "I32", got.I32, want.I32, genI32)
			checkMatrixOps(t, tier, "I64", got.I64, want.I64, genI64)
		})
	}
}
