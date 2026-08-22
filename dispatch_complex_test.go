package simd

import (
	"reflect"
	"testing"
)

// The complex surface must reach the generated kernels, the same way the
// numeric surface does.
//
// It did not, from v1.14.0 to v1.20.0. complexOps returned &refBase.C64 /
// &refBase.C128 directly, with no per-tier overlay, so every complex
// operation ran the portable Go reference on every machine -- while the
// repository generated, tested and shipped nine tiers of complex assembly
// for them. The existing coverage did not catch it because
// TestDeclaredKernelsAreWired checks the runtime tables for exactly three
// groups (Bytes, Convert, Mask) and routes everything else -- numeric and
// complex alike -- through archSets(), the test-only whole-set aggregator
// whose own generated header says consumers never call it. Nothing walked
// the tables the runtime actually indexes.
//
// So this asks the question a caller asks: is the function DotComplex ends
// up calling the one the tier's table supplies?

// tierPartial returns the per-tier partial for a group at the running tier,
// or nil when this build has no table for it (the purego and no-backend
// builds, and slot 0, the reference).
func tierPartial[G any](byTier []*G) *G {
	if tierIdx < 0 || tierIdx >= len(byTier) {
		return nil
	}
	return byTier[tierIdx]
}

// checkGroup is the whole contract, field by field: where the tier's partial
// supplies a kernel the dispatch must reach exactly that function, and where
// it does not the dispatch must keep the reference.
//
// The second half matters as much as the first. A missing kernel on a tier
// is documented as slower and never a correctness gap
// (docs/lld/kernels-and-dispatch.md), and three architectures rely on it
// today -- s390x has no complex64 Dot or DotConj, ppc64le adds Sum to that
// list, loong64 is missing Mul, Div and Conj. An earlier version of this
// test asserted "not the reference" for one hardcoded field and turned that
// documented fallback into a red cross-compile gate on two architectures.
func checkGroup[G any](t *testing.T, name string, got, ref, partial *G) {
	t.Helper()
	g := reflect.ValueOf(got).Elem()
	r := reflect.ValueOf(ref).Elem()
	var p reflect.Value
	if partial != nil {
		p = reflect.ValueOf(partial).Elem()
	}
	typ := g.Type()

	supplied := 0
	for i := 0; i < g.NumField(); i++ {
		if typ.Field(i).Type.Kind() != reflect.Func {
			continue
		}
		field := name + "." + typ.Field(i).Name
		gp := g.Field(i).Pointer()
		rp := r.Field(i).Pointer()

		if partial != nil && !p.Field(i).IsNil() {
			supplied++
			want := p.Field(i).Pointer()
			if gp != want {
				t.Errorf("%s on tier %q: dispatch reaches %#x, the tier's table holds %#x",
					field, Tier(), gp, want)
			}
			if gp == rp && rp != want {
				t.Errorf("%s on tier %q: dispatch is still the portable reference", field, Tier())
			}
			continue
		}
		// No kernel on this tier: the reference stands, unchanged.
		if gp != rp {
			t.Errorf("%s on tier %q: no kernel is generated, but dispatch is not the reference",
				field, Tier())
		}
	}

	// A table that supplies nothing on an accelerated tier is a table that
	// was emitted empty -- which is what the defect looked like.
	if partial != nil && supplied == 0 && Tier() != "scalar" {
		t.Errorf("%s on tier %q: the per-tier table supplies no kernel at all", name, Tier())
	}
}

func TestComplexDispatchReachesGeneratedKernels(t *testing.T) {
	checkGroup(t, "C64", complexOps[complex64](), &refBase.C64, tierPartial(cplxC64ByTier[:]))
	checkGroup(t, "C128", complexOps[complex128](), &refBase.C128, tierPartial(cplxC128ByTier[:]))
}

// The real-scalar half of the split -- Abs, Real, Imag, Scale -- goes through
// ComplexParts and had the same hole.
func TestComplexPartsDispatchReachesGeneratedKernels(t *testing.T) {
	checkGroup(t, "C64Parts", complexParts[complex64, float32](),
		&refBase.C64Parts, tierPartial(partsC64PartsByTier[:]))
	checkGroup(t, "C128Parts", complexParts[complex128, float64](),
		&refBase.C128Parts, tierPartial(partsC128PartsByTier[:]))
}

// groupCache indexes its partial slice by tierIdx and guards with
// `i < len(partial)`, so a table shorter than dispatchTiers silently leaves
// the top tiers on the reference and a longer one silently drops its tail.
// Both are generator bugs that no other test would notice.
func TestComplexTierTablesMatchTheTierList(t *testing.T) {
	for _, c := range []struct {
		name string
		n    int
	}{
		{"cplxC64ByTier", len(cplxC64ByTier)},
		{"cplxC128ByTier", len(cplxC128ByTier)},
		{"partsC64PartsByTier", len(partsC64PartsByTier)},
		{"partsC128PartsByTier", len(partsC128PartsByTier)},
	} {
		if c.n != len(dispatchTiers) {
			t.Errorf("%s has %d slots, dispatchTiers has %d", c.name, c.n, len(dispatchTiers))
		}
	}
}

// Whatever the dispatch routes to, the answer is the reference's answer.
// The numerical contract is the one every other tier keeps: a fixed
// accumulation order, identical on every architecture.
func TestComplexDispatchMatchesTheReference(t *testing.T) {
	n := 1000
	a := make([]complex128, n)
	b := make([]complex128, n)
	for i := range a {
		a[i] = complex(float64(i%17)-8, float64(i%23)-11)
		b[i] = complex(float64(i%13)-6, float64(i%29)-14)
	}
	if got, want := DotComplex(a, b), refBase.C128.Dot(a, b); got != want {
		t.Errorf("DotComplex = %v, reference = %v", got, want)
	}
	if got, want := DotComplexConj(a, b), refBase.C128.DotConj(a, b); got != want {
		t.Errorf("DotComplexConj = %v, reference = %v", got, want)
	}
	if got, want := SumComplex(a), refBase.C128.Sum(a); got != want {
		t.Errorf("SumComplex = %v, reference = %v", got, want)
	}

	gotS := append([]complex128(nil), a...)
	wantS := append([]complex128(nil), a...)
	ScaleComplex(gotS, 1.5)
	refBase.C128Parts.Scale(wantS, wantS, 1.5)
	for i := range gotS {
		if gotS[i] != wantS[i] {
			t.Fatalf("ScaleComplex[%d] = %v, reference = %v", i, gotS[i], wantS[i])
		}
	}
}
