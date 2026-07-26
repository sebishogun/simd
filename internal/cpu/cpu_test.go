package cpu

import (
	"runtime"
	"strings"
	"testing"
)

func TestDetectAlwaysOffersScalar(t *testing.T) {
	avail := detect()
	if len(avail) == 0 {
		t.Fatal("detect returned no tiers; Scalar must always be available")
	}
	if avail[0] != Scalar {
		t.Errorf("detect()[0] = %v, want Scalar first", avail[0])
	}
	// Tiers must be ordered weakest to strongest so that selection can take
	// the last element.
	for i := 1; i < len(avail); i++ {
		if avail[i] <= avail[i-1] {
			t.Errorf("tiers out of order: %v then %v", avail[i-1], avail[i])
		}
	}
	t.Logf("%s: %v", runtime.GOARCH, avail)
}

func TestTierNamesRoundTrip(t *testing.T) {
	for i := Tier(0); i < numTiers; i++ {
		name := i.String()
		got, ok := ParseTier(name)
		if !ok {
			t.Errorf("ParseTier(%q) failed for tier %d", name, i)
			continue
		}
		if got != i {
			t.Errorf("ParseTier(%q) = %v, want %v", name, got, i)
		}
	}
}

func TestParseTierRejectsJunk(t *testing.T) {
	for _, s := range []string{"", "avx3", "SSE", "neon2", "  "} {
		if _, ok := ParseTier(s); ok && s != "" {
			t.Errorf("ParseTier(%q) unexpectedly succeeded", s)
		}
	}
	// Names are case- and space-insensitive.
	if got, ok := ParseTier("  AVX2  "); !ok || got != AVX2 {
		t.Errorf("ParseTier(%q) = %v, %v; want AVX2, true", "  AVX2  ", got, ok)
	}
}

// GOSIMD must only ever select down. Naming a tier the CPU does not have
// would mean executing instructions it cannot decode, which is how the
// surveyed libraries produced SIGILL in production; here it falls back to
// Scalar and records why.
func TestForcingAnUnavailableTierFallsBackNotCrashes(t *testing.T) {
	sel := selectTier("avx512", "")
	if contains(detect(), AVX512) {
		t.Skip("host has AVX-512; cannot test the unavailable path here")
	}
	if sel.Tier != Scalar {
		t.Errorf("forcing an unavailable tier gave %v, want Scalar", sel.Tier)
	}
	if sel.Reason == "" {
		t.Error("fallback recorded no reason")
	}
}

func TestForcingAnUnknownTierFallsBack(t *testing.T) {
	sel := selectTier("nonsense", "")
	if sel.Tier != Scalar {
		t.Errorf("Tier = %v, want Scalar", sel.Tier)
	}
	if !strings.Contains(sel.Reason, "unknown tier") {
		t.Errorf("Reason = %q, want it to mention an unknown tier", sel.Reason)
	}
}

func TestForcingScalarAlwaysWorks(t *testing.T) {
	sel := selectTier("scalar", "")
	if sel.Tier != Scalar {
		t.Errorf("Tier = %v, want Scalar", sel.Tier)
	}
	if !sel.Forced {
		t.Error("Forced = false, want true")
	}
	if sel.Reason != "" {
		t.Errorf("Reason = %q, want empty: scalar is always available", sel.Reason)
	}
}

func TestDisableMasksATier(t *testing.T) {
	avail := detect()
	if len(avail) < 2 {
		t.Skip("host has only the scalar tier")
	}
	top := avail[len(avail)-1]
	sel := selectTier("", top.String())
	if sel.Tier == top {
		t.Errorf("Tier = %v, but %v was disabled", sel.Tier, top)
	}
	if len(sel.Disabled) != 1 || sel.Disabled[0] != top {
		t.Errorf("Disabled = %v, want [%v]", sel.Disabled, top)
	}
	if sel.Tier != avail[len(avail)-2] {
		t.Errorf("Tier = %v, want the next tier down %v", sel.Tier, avail[len(avail)-2])
	}
}

// Scalar is the floor: masking it would leave nothing to run.
func TestDisableCannotMaskScalar(t *testing.T) {
	sel := selectTier("", "scalar")
	if !contains(sel.Available, Scalar) {
		t.Error("Scalar was masked out; it must always remain available")
	}
}

func TestDescribeMentionsTheSelectedTier(t *testing.T) {
	got := Describe()
	if !strings.Contains(got, runtime.GOARCH) {
		t.Errorf("Describe() = %q, want it to mention %s", got, runtime.GOARCH)
	}
	if !strings.Contains(got, Selected().String()) {
		t.Errorf("Describe() = %q, want it to mention tier %s", got, Selected())
	}
}
