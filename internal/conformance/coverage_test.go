package conformance

// How many kernels each tier actually replaced.
//
// This is the guard against a vacuous conformance run. Every tier starts as a
// copy of the portable reference and overrides only what was generated for it,
// which is what makes a partial backend safe — but it also means a group that
// failed to generate entirely would still be compared, field by field, against
// itself, and would pass. Counting the fields that differ turns that from
// invisible into a number on the test log.

import (
	"reflect"
	"sort"
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// generatedFields counts the function fields of a struct that differ from the
// reference, and names those that do not.
//
// Go forbids comparing func values, so the comparison is on code pointers,
// which is exactly the question being asked: is this slot still pointing at
// the portable implementation?
func generatedFields(got, want any) (n int, missing []string) {
	g, w := reflect.ValueOf(got), reflect.ValueOf(want)
	for i := range g.NumField() {
		f := g.Type().Field(i)
		if f.Type.Kind() != reflect.Func {
			continue
		}
		gf, wf := g.Field(i), w.Field(i)
		switch {
		case gf.IsNil() && wf.IsNil():
			// Deliberately empty. Ops[T] is one struct shared by all four
			// element types, so the integer instantiations leave the
			// float-only slots — Div, Sqrt, Norm, the transcendentals — nil.
			// Nothing can reach them: the public functions that use them are
			// constrained to [T Float], which is checked at compile time.
			continue
		case gf.IsNil():
			missing = append(missing, f.Name+" (nil!)")
		case gf.Pointer() == wf.Pointer():
			missing = append(missing, f.Name)
		default:
			n++
		}
	}
	return n, missing
}

func TestGeneratedKernelCoverage(t *testing.T) {
	want := ref.Set()
	names := []string{}
	for n := range tiers(t) {
		names = append(names, n)
	}
	sort.Strings(names)

	for _, tier := range names {
		got := tiers(t)[tier]
		t.Run(tier, func(t *testing.T) {
			groups := []struct {
				name      string
				got, want any
			}{
				{"F32", got.F32, want.F32},
				{"F64", got.F64, want.F64},
				{"I32", got.I32, want.I32},
				{"I64", got.I64, want.I64},
				{"Bytes", got.Bytes, want.Bytes},
				{"Mask", got.Mask, want.Mask},
			}
			total := 0
			for _, g := range groups {
				n, missing := generatedFields(g.got, g.want)
				total += n
				t.Logf("%-6s %3d generated, %2d on the portable path: %v",
					g.name, n, len(missing), missing)
				for _, m := range missing {
					if len(m) > 6 && m[len(m)-6:] == "(nil!)" {
						t.Errorf("%s.%s is nil; a tier must be a complete set", g.name, m)
					}
				}
			}
			t.Logf("%s: %d generated kernels", tier, total)
			if total == 0 {
				t.Errorf("tier %s replaced nothing; the conformance suite would be "+
					"comparing the reference against itself", tier)
			}
		})
	}
}

// TestEveryGroupIsExercised fails if a whole kernel group is untouched on every
// tier, which would mean the C source for it never compiled and nobody noticed.
func TestEveryGroupIsExercised(t *testing.T) {
	want := ref.Set()
	best := map[string]int{}
	for _, got := range tiers(t) {
		for _, g := range []struct {
			name      string
			got, want any
		}{
			{"F32", got.F32, want.F32},
			{"F64", got.F64, want.F64},
			{"I32", got.I32, want.I32},
			{"I64", got.I64, want.I64},
			{"Bytes", got.Bytes, want.Bytes},
			{"Mask", got.Mask, want.Mask},
		} {
			if n, _ := generatedFields(g.got, g.want); n > best[g.name] {
				best[g.name] = n
			}
		}
	}
	for name, n := range best {
		if n == 0 {
			t.Errorf("no tier generated a single %s kernel", name)
		}
	}
}

var _ = kernel.Set{}
