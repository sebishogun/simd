package main

import (
	"math/rand/v2"

	"github.com/sebishogun/simd"
)

// A scenario is one claim from docs/tutorial.md, with both implementations it
// compares and the code for each.
//
// The pairs are taken from the tutorial rather than invented here, so the site
// and the tutorial cannot drift: if a claim there stops being true, the
// scenario measuring it fails visibly rather than the two disagreeing quietly.
//
// Every baseline is written the way someone would actually write it. A rigged
// baseline measures nothing, and this page exists to answer "is it worth it",
// which is a question a dishonest comparison cannot answer.
type Scenario struct {
	ID       string
	Title    string
	Claim    string // what the tutorial says, in one line
	Section  string // where in the tutorial
	BaseName string
	FastName string
	BaseCode string
	FastCode string

	// setup allocates the working data once, outside timing. run is measured.
	setup func(n int) any
	base  func(state any)
	fast  func(state any)
	N     int
}

type soa struct{ x, y, z, m, e, tmp []float64 }

type aos struct {
	ps []struct{ X, Y, Z, M float64 }
	e  []float64
}

func randFloats(n int, r *rand.Rand) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64()
	}
	return s
}

func scenarios() []Scenario {
	return []Scenario{
		{
			ID:      "standardise",
			Title:   "Standardise a column",
			Claim:   "Four passes per column, no allocation — the worked example.",
			Section: "9. A worked example",
			N:       1 << 20,
			setup: func(n int) any {
				r := rand.New(rand.NewPCG(1, 2))
				return &soa{x: randFloats(n, r)}
			},
			BaseName: "hand-written loop",
			FastName: "this library",
			BaseCode: `var sum float64
for _, v := range col {
	sum += v
}
mean := sum / float64(len(col))
var ss float64
for i := range col {
	col[i] -= mean
	ss += col[i] * col[i]
}
sd := math.Sqrt(ss / float64(len(col)))
for i := range col {
	col[i] /= sd
}`,
			FastCode: `mean := simd.Mean(col)
simd.SubScalar(col, mean)
sd := simd.StdDev(col)
simd.DivScalar(col, sd)`,
			base: func(s any) {
				col := s.(*soa).x
				var sum float64
				for _, v := range col {
					sum += v
				}
				mean := sum / float64(len(col))
				var ss float64
				for i := range col {
					col[i] -= mean
					ss += col[i] * col[i]
				}
				sd := sqrt(ss / float64(len(col)))
				if sd != 0 {
					for i := range col {
						col[i] /= sd
					}
				}
			},
			fast: func(s any) {
				col := s.(*soa).x
				mean := simd.Mean(col)
				simd.SubScalar(col, mean)
				sd := simd.StdDev(col)
				if sd != 0 {
					simd.DivScalar(col, sd)
				}
			},
		},
		{
			ID:      "fused",
			Title:   "Fuse, don't chain",
			Claim:   "Three chained calls read and write memory three times over; one fused call does it once.",
			Section: "4. Fuse, don't chain",
			N:       1 << 20,
			setup: func(n int) any {
				r := rand.New(rand.NewPCG(3, 4))
				return &soa{
					x: randFloats(n, r), y: randFloats(n, r),
					z: randFloats(n, r), e: make([]float64, n),
				}
			},
			BaseName: "three calls",
			FastName: "one fused call",
			BaseCode: `simd.AddInto(dst, a, b)
simd.Add(dst, c)
simd.Add(dst, d)`,
			FastCode: `simd.AddAll(dst, a, b, c, d)`,
			base: func(s any) {
				v := s.(*soa)
				simd.AddInto(v.e, v.x, v.y)
				simd.Add(v.e, v.z)
				simd.Add(v.e, v.x)
			},
			fast: func(s any) {
				v := s.(*soa)
				simd.AddAll(v.e, v.x, v.y, v.z, v.x)
			},
		},
		{
			ID:      "batched",
			Title:   "Hand over batches, not elements",
			Claim:   "The call boundary is per call, not per element — it disappears into a large slice and dominates a small one.",
			Section: "1. Hand over batches, not elements",
			N:       1 << 16,
			setup: func(n int) any {
				r := rand.New(rand.NewPCG(5, 6))
				return &soa{x: randFloats(n, r), y: randFloats(n, r), e: make([]float64, n)}
			},
			BaseName: "one call per element",
			FastName: "one call, n elements",
			BaseCode: `for i := range dst {
	simd.AddInto(dst[i:i+1], a[i:i+1], b[i:i+1])
}`,
			FastCode: `simd.AddInto(dst, a, b)`,
			base: func(s any) {
				v := s.(*soa)
				for i := range v.e {
					simd.AddInto(v.e[i:i+1], v.x[i:i+1], v.y[i:i+1])
				}
			},
			fast: func(s any) {
				v := s.(*soa)
				simd.AddInto(v.e, v.x, v.y)
			},
		},
		{
			ID:      "soa",
			Title:   "Struct of arrays, not array of structs",
			Claim:   "With []Particle the X values are 32 bytes apart and cannot be loaded into a vector register at all.",
			Section: "2. Struct of arrays, not array of structs",
			N:       1 << 20,
			setup: func(n int) any {
				r := rand.New(rand.NewPCG(7, 8))
				a := &aos{ps: make([]struct{ X, Y, Z, M float64 }, n), e: make([]float64, n)}
				for i := range a.ps {
					a.ps[i].X = r.NormFloat64()
				}
				s := &soa{x: randFloats(n, r), e: make([]float64, n)}
				return [2]any{a, s}
			},
			BaseName: "array of structs",
			FastName: "struct of arrays",
			BaseCode: `type Particle struct{ X, Y, Z, Mass float64 }
ps := make([]Particle, n)

for i := range ps {
	e[i] = ps[i].X * 2   // X values are 32 bytes apart
}`,
			FastCode: `type Particles struct{ X, Y, Z, Mass []float64 }

simd.ScaleInto(e, p.X, 2)   // one contiguous run`,
			base: func(s any) {
				a := s.([2]any)[0].(*aos)
				for i := range a.ps {
					a.e[i] = a.ps[i].X * 2
				}
			},
			fast: func(s any) {
				v := s.([2]any)[1].(*soa)
				simd.ScaleInto(v.e, v.x, 2)
			},
		},
	}
}

// sqrt avoids importing math into the hot loop's file scope for one call; the
// baseline is meant to read like ordinary Go and this keeps it that way.
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton, seeded well enough that this is not what the baseline measures.
	z := x
	for range 12 {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
