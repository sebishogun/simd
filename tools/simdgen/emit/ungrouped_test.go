package emit

import (
	"testing"

	"github.com/sebishogun/simd/tools/simdgen/spec"
)

// A group with no dispatch table falls back to the reference on every
// machine with every test green. The generator must refuse to emit rather
// than produce a working-looking file -- this is the check that would have
// caught the complex groups.
func TestUngroupedGroupIsRefused(t *testing.T) {
	if got := ungrouped(nil); len(got) != 0 {
		t.Fatalf("an empty manifest reported %v", got)
	}
	for _, g := range []string{"F32", "C128", "C64Parts", "Bytes"} {
		if got := ungrouped([]spec.Kernel{{Group: g}}); len(got) != 0 {
			t.Errorf("known group %s reported as ungrouped: %v", g, got)
		}
	}
	got := ungrouped([]spec.Kernel{{Group: "C32"}, {Group: "F32"}})
	if len(got) != 1 || got[0] != "C32" {
		t.Fatalf("ungrouped = %v, want [C32]", got)
	}
}
