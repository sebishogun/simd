package simd

// A nil entry in the dispatch table is a panic at the first call, on whichever
// tier forgot to fill it, in whichever program reaches it first. It is not a
// wrong answer that a differential test would catch, and it is not a build
// error: the table is a struct of function values, so a missing one is a valid
// zero.
//
// This found itself immediately. IndexNonASCII was registered for every
// generated backend and not for the portable one, so every accelerated tier
// passed and `-tags purego` panicked inside AppendUTF16 — the one
// configuration guaranteed to build everywhere, failing on the one path that
// is supposed to always work.
//
// The subject is `active`, the table dispatch actually assembled, not the raw
// backend sets: a Fast* slot is legitimately empty in a generated file and is
// filled from the reference afterwards by FillFastFallbacks, so checking the
// pieces would report holes that the finished table does not have. One
// process sees one tier, and `make test-tiers` runs this once per tier the
// machine supports, which is what makes the sweep total.
//
// Only the groups whose every field applies to every caller are checked. The
// number groups are one struct instantiated per element type, so U8.Sqrt and
// I32.MatMulPk are nil in the reference too and always will be — 530 of them,
// which would drown the real signal. Their coverage comes from the threshold
// meta-test and the differential suite instead, both of which are driven by
// the element type. What is checked here is exactly the shape of the bug this
// exists for: a group that is the same for everyone, with a hole in it.

import (
	"reflect"
	"testing"
)

func TestDispatchTableComplete(t *testing.T) {
	// The runtime surface is per-operation tables plus the reference base.
	// Every table must be fully populated -- one entry per tier, none nil --
	// and every field of the reference base must be callable, because it is
	// both slot zero and the overlay's starting point.
	for name, slots := range allFlatTables {
		if len(slots) != len(dispatchTiers) {
			t.Errorf("%s: %d slots for %d tiers", name, len(slots), len(dispatchTiers))
		}
		for i, fn := range slots {
			if fn == nil || reflect.ValueOf(fn).IsNil() {
				t.Errorf("%s: tier %q slot is nil — a call through it panics",
					name, dispatchTiers[i])
			}
		}
	}

	var check func(v reflect.Value, path string)
	check = func(v reflect.Value, path string) {
		switch v.Kind() {
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				f := v.Type().Field(i)
				if !f.IsExported() {
					continue
				}
				check(v.Field(i), path+"."+f.Name)
			}
		case reflect.Func:
			if v.IsNil() {
				t.Errorf("refBase: %s is nil — a call through it panics", path)
			}
		}
	}
	for _, g := range []struct {
		name string
		v    any
	}{
		{"Bytes", refBase.Bytes},
		{"Convert", refBase.Convert},
		{"Mask", refBase.Mask},
		{"C64", refBase.C64},
		{"C128", refBase.C128},
	} {
		check(reflect.ValueOf(g.v), g.name)
	}
}
