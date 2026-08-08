package simd_test

import (
	"math/rand"
	"sort"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

func TestMergeSortedUint32(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for trial := 0; trial < 150; trial++ {
		na, nb := rng.Intn(3000), rng.Intn(3000)
		if trial < 5 {
			na, nb = trial, 5-trial
		}
		a := make([]uint32, na)
		b := make([]uint32, nb)
		for i := range a {
			a[i] = rng.Uint32() >> uint(rng.Intn(20))
		}
		for i := range b {
			b[i] = rng.Uint32() >> uint(rng.Intn(20))
		}
		sort.Slice(a, func(i, j int) bool { return a[i] < a[j] })
		sort.Slice(b, func(i, j int) bool { return b[i] < b[j] })
		d1 := make([]uint32, na+nb)
		d2 := make([]uint32, na+nb)
		g1 := ref.MergeSortedU32(d1, a, b)
		g2 := simd.MergeSortedUint32(d2, a, b)
		if g1 != g2 || g1 != na+nb {
			t.Fatalf("trial %d: ref %d kernel %d want %d", trial, g1, g2, na+nb)
		}
		for i := range d1 {
			if d1[i] != d2[i] {
				t.Fatalf("trial %d: differs at %d: %d vs %d", trial, i, d1[i], d2[i])
			}
		}
	}
}
