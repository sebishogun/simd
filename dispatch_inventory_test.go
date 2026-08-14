package simd

// Every kernel the manifest declares must actually be reachable.
//
// TestDispatchTableComplete next door checks that no field is nil, but it can
// only do that for the groups where every field applies to every caller —
// Bytes, Convert, Mask, C64, C128. The number groups are one struct
// instantiated per element type, so U8.Sqrt and I32.MatMulPk are nil in the
// reference too and always will be. A blanket check over them would report
// five hundred holes that are not holes, and the real one would be lost in it.
//
// The manifest is the only thing that can tell those apart. If it declares a
// kernel with Group "F32" and Field "ShiftDiv", then that slot is not
// optional, whatever U8 does with the same field name. backend.Inventory is
// that declaration as data, emitted by the generator, so this test is exactly
// as complete as the manifest is.
//
// The subject is `active` — the table dispatch actually assembled — and not
// ref.Set(). That is not a shortcut, it is the only correct choice: ref.Set()
// leaves every Fast slot nil on purpose, because until a backend has finished
// registering, "no Fast kernel here" and "the accurate kernel" cannot be told
// apart. dispatch.go fills them with FillFastFallbacks afterwards. Asserting
// against the raw reference would report 92 Fast slots as holes when the
// finished table has none.
//
// The portable path is covered by running this same test under -tags purego,
// which `make verify` does, because there `active` IS the reference table. And
// that is the configuration the bug this whole family of tests exists for
// showed up in: IndexNonASCII was registered for every generated backend and
// not for the portable one, so every accelerated tier passed and the one build
// guaranteed to work everywhere panicked at the first call.

import (
	"reflect"
	"testing"

	"github.com/sebishogun/simd/internal/backend"
	"github.com/sebishogun/simd/internal/kernel"
)

func TestDeclaredKernelsAreWired(t *testing.T) {
	if len(backend.Inventory) == 0 {
		t.Fatal("backend.Inventory is empty; run make codegen")
	}

	// The runtime tables carry the flat groups; the numeric groups reach the
	// dispatcher through per-tier Sets the arch package assembles. Both must
	// have every declared kernel wired on every tier.
	var sets map[string]kernel.Set
	if archSets != nil {
		sets = archSets()
	}
	flat := map[string]bool{"Bytes": true, "Convert": true, "Mask": true}
	for _, d := range backend.Inventory {
		if flat[d.Group] {
			slots, ok := allFlatTables[d.Group+"."+d.Field]
			if !ok {
				t.Errorf("%s: no dispatch table for %s.%s", d.CName, d.Group, d.Field)
				continue
			}
			for i, fn := range slots {
				if fn == nil || reflect.ValueOf(fn).IsNil() {
					t.Errorf("%s: table %s.%s tier %q is nil", d.CName, d.Group, d.Field, dispatchTiers[i])
				}
			}
			continue
		}
		// The struct groups now have real per-tier tables, so walk those --
		// the runtime indexes them, and archSets() does not. Four whole
		// complex groups were missing from the runtime tables for seven
		// releases with this test green, because everything that was not a
		// flat group went through the test-only aggregator instead.
		if slots, ok := allGroupTables[d.Group]; ok {
			for i, partial := range slots {
				if i == 0 || partial == nil {
					continue // slot 0 is the reference; no overlay
				}
				f := reflect.ValueOf(partial).FieldByName(d.Field)
				if !f.IsValid() {
					t.Errorf("%s: per-tier table for %s has no field %q",
						d.CName, d.Group, d.Field)
					break
				}
				if f.Kind() == reflect.Func && f.IsNil() && present(sets, dispatchTiers[i], d) {
					t.Errorf("tier %q: %s.%s is nil in the dispatch table, but the "+
						"kernel is generated for that tier -- it will never run",
						dispatchTiers[i], d.Group, d.Field)
				}
			}
			continue
		}
		for tier, set := range sets {
			g := reflect.ValueOf(set).FieldByName(d.Group)
			if !g.IsValid() {
				t.Errorf("%s: kernel.Set has no group %q", d.CName, d.Group)
				continue
			}
			f := g.FieldByName(d.Field)
			if !f.IsValid() {
				t.Errorf("%s: kernel.Set.%s has no field %q", d.CName, d.Group, d.Field)
				continue
			}
			if f.Kind() == reflect.Func && f.IsNil() {
				t.Errorf("tier %q: %s.%s is nil, but the manifest declares %s",
					tier, d.Group, d.Field, d.CName)
			}
		}
	}
}

// TestInventoryCoversEveryGroup is the other direction: the inventory is only
// worth trusting if it actually reaches every group of kernel.Set that has
// generated kernels. A manifest change that dropped a whole source file would
// otherwise leave this suite quietly checking less than it did before.
func TestInventoryCoversEveryGroup(t *testing.T) {
	got := map[string]int{}
	for _, d := range backend.Inventory {
		got[d.Group]++
	}
	// Every group that kernel.Set exposes and the generator emits into. The
	// two ComplexParts groups are here because they carry kernels too.
	for _, g := range []string{
		"F32", "F64", "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64",
		"Bytes", "Convert", "Mask", "C64", "C128", "C64Parts", "C128Parts",
	} {
		if got[g] == 0 {
			t.Errorf("no kernel declared for group %s; either the manifest lost a "+
				"source file or the inventory is stale — run make codegen", g)
		}
	}
}

// present reports whether the generator emitted a kernel for d on tier.
//
// Comparing against the REFERENCE, not against nil: archSets() builds each
// tier from ref.Set() and overlays what was generated, so every field is
// non-nil there and a nil test would call every operation "generated" on
// every tier. A tier the generator declined keeps the reference by design
// (docs/lld/kernels-and-dispatch.md), and only a kernel that exists and is
// missing from the dispatch table is a defect.
func present(sets map[string]kernel.Set, tier string, d backend.Declared) bool {
	set, ok := sets[tier]
	if !ok {
		return false
	}
	g := reflect.ValueOf(set).FieldByName(d.Group)
	r := reflect.ValueOf(refBase).FieldByName(d.Group)
	if !g.IsValid() || !r.IsValid() {
		return false
	}
	f, rf := g.FieldByName(d.Field), r.FieldByName(d.Field)
	if !f.IsValid() || !rf.IsValid() || f.Kind() != reflect.Func || f.IsNil() {
		return false
	}
	return rf.IsNil() || f.Pointer() != rf.Pointer()
}
