package simd_test

// A meta-test over the dispatch thresholds.
//
// Every accelerated operation has a length below which the dispatcher calls
// the reference instead. A test that never exceeds that length exercises the
// fallback and reports success — which is how the accelerated Median and
// Quantile went untested until TestSelectAcceleratedMatchesReference was
// written, and how the n-ary family and CompressInto went untested until
// threshold_test.go was.
//
// Both of those were found by enumerating thresholds by hand. That works once.
// This test makes it mechanical: it reads the thresholds out of the generated
// guards and fails when one appears that is above the suite's default input
// length and is not named below as having dedicated coverage.
//
// The failure is deliberately annoying. Adding a kernel with a threshold of
// 512 should not be possible without either adding a test above 512 or saying
// in one line why none is needed.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"testing"
)

// coveredAboveThreshold names the operations whose accelerated path is
// exercised above its own threshold, and by which test. Anything with a
// threshold over defaultTestLen must appear here.
var coveredAboveThreshold = map[string]string{
	"compress": "TestCompressAboveThreshold, densities 0..100% at n up to 4099",
	"add3":     "TestNaryAboveThreshold, k=2..5 at n up to 4099",
	"add4":     "TestNaryAboveThreshold",
	"mul3":     "TestNaryAboveThreshold",
	"mul4":     "TestNaryAboveThreshold",

	// Deliberately never accelerated: the manifest gives these a threshold of
	// 1<<30, which no caller reaches, because the standard library wins at
	// every measured length. See the note on `never` in the manifest. There is
	// no accelerated path to test.
	"countByte":  "not accelerated by design; threshold is `never`",
	"equalBytes": "not accelerated by design; threshold is `never`",

	// UNCOVERED, and honestly recorded as such rather than silently allowed.
	// Each has a real accelerated path above the length any test reaches, so
	// the kernel is currently exercised only by benchmarks. Fixing this means
	// adding a differential test at a length above the threshold, in the same
	// shape as TestCompressAboveThreshold.
	"index":        "UNCOVERED at 256 — text tests reach only 1000 for a few cases",
	"countSeq":     "UNCOVERED at 256",
	"indexByte":    "UNCOVERED at 1024",
	"compareBytes": "UNCOVERED at 2048",
	"transpose":    "TestTranspose, dimensions to 200x7 and 128x128",
}

// defaultTestLen is the longest input the general suite uses. Anything with a
// threshold at or below this is exercised on both sides of its crossover by
// the ordinary tests.
const defaultTestLen = 70

var reGuardThreshold = regexp.MustCompile(`func ([a-zA-Z0-9]+?)(?:Float32|Float64|Int8|Int16|Int32|Int64|Uint8|Uint16|Uint32|Uint64|Complex64|Complex128)?(?:SSE2|AVX2|AVX512|NEON|SVE2|RVV|VX|LASX|VSX)Guarded\([^)]*\) [^{]*\{\s*(?:[^\n]*\n\s*)?if (?:len\([a-z]+\)|n) < (\d+)`)

func TestEveryThresholdHasCoverage(t *testing.T) {
	_, thisFile, _, _ := runtime.Caller(0)
	root := filepath.Dir(thisFile)
	dir := filepath.Join(root, "internal", "amd64")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Skipf("no generated amd64 backend to inspect: %v", err)
	}

	worst := map[string]int{}
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".go" {
			b, err := os.ReadFile(filepath.Join(dir, e.Name()))
			if err != nil {
				t.Fatal(err)
			}
			for _, m := range reGuardThreshold.FindAllStringSubmatch(string(b), -1) {
				n, err := strconv.Atoi(m[2])
				if err != nil {
					continue
				}
				if n > worst[m[1]] {
					worst[m[1]] = n
				}
			}
		}
	}
	if len(worst) == 0 {
		t.Skip("no guards found; the generated layout must have changed")
	}

	var uncovered []string
	for op, n := range worst {
		if n <= defaultTestLen {
			continue
		}
		if _, ok := coveredAboveThreshold[op]; !ok {
			uncovered = append(uncovered, fmt.Sprintf("%s (threshold %d)", op, n))
		}
	}
	sort.Strings(uncovered)
	if len(uncovered) > 0 {
		t.Errorf("these operations dispatch above the suite's default length of %d "+
			"and have no test recorded above their own threshold:\n  %v\n\n"+
			"Below the threshold the dispatcher calls the reference, so an ordinary "+
			"test proves nothing about the kernel. Either add a test with inputs "+
			"longer than the threshold, or add the operation to "+
			"coveredAboveThreshold naming the test that covers it.",
			defaultTestLen, uncovered)
	}

	// The allowlist must not rot: an entry naming an operation that no longer
	// has a high threshold is a claim about a test nobody is running.
	for op := range coveredAboveThreshold {
		if worst[op] <= defaultTestLen {
			t.Errorf("coveredAboveThreshold names %q, but its threshold is now %d, "+
				"at or below the default test length — remove the entry",
				op, worst[op])
		}
	}
}
