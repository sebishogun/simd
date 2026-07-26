package conformance

// What the comparison, mask and byte kernels are worth.
//
// This is a repetition test, not an average: each measurement is the minimum
// over many runs, because the minimum is the only statistic that estimates
// what the hardware can do. An average measures the machine's interruptions as
// much as the code. See internal/perf.
//
// It lives here rather than under internal/amd64 because it goes through the
// kernel.Set interface, so it measures whichever tier the host actually
// selected — the same test reports NEON on a Graviton and RVV on a SiFive.

import (
	"fmt"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/sebishogun/simd/internal/cpu"
	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/perf"
	"github.com/sebishogun/simd/internal/ref"
)

var (
	sinkInt  int
	sinkBool bool
)

// perfLens span the cache hierarchy: a length that fits in a register file, one
// in L1, one in L2, and one that has to come from L3 or memory. A kernel that
// looks like a large win at 4 KiB often is not at 1 MiB, because past L2 the
// bottleneck stops being instructions and becomes bandwidth — and no amount of
// vector width helps with that.
var perfLens = []int{64, 1024, 16384, 262144}

func TestPerfCompareMaskBytes(t *testing.T) {
	if testing.Short() {
		t.Skip("repetition testing needs time")
	}
	opt := perf.DefaultOptions()
	opt.Patience = 60 * time.Millisecond

	want := ref.Set()
	for tier, got := range tiers(t) {
		t.Logf("\n=== %s (host: %s) ===", tier, cpu.Describe())
		for _, n := range perfLens {
			t.Logf("\n  n=%d", n)
			for _, row := range perfCases(n, got, want) {
				a := perf.Measure("", row.bytes, row.got, opt)
				b := perf.Measure("", row.bytes, row.want, opt)
				t.Logf("    %-22s asm %9.1f ns (%6.2f GB/s)  portable %9.1f ns  %5.2fx",
					row.name, a.Min, a.GBPerSec(), b.Min, b.Min/a.Min)
			}
		}
	}
}

type perfCase struct {
	name      string
	got, want func()
	bytes     int64
}

// perfCases builds one measurement per kernel worth reporting: the shapes that
// differ in how much work they do per byte moved.
func perfCases(n int, got, want kernel.Set) []perfCase {
	r := rand.New(rand.NewPCG(21, 22))
	f32a, f32b := genF32(n, r), genF32(n, r)
	f64a, f64b := genF64(n, r), genF64(n, r)
	mask := genBool(n, r)
	mdst := make([]bool, n)
	fdst := make([]float32, n)
	f64dst := make([]float64, n)
	ba, bb := genBytes(n, r), genBytes(n, r)
	bdst := make([]byte, n)

	// Inputs that settle the answer at the far end, and inputs that settle it
	// at byte zero, for the kernels that can exit early.
	allTrue, allFalse := make([]bool, n), make([]bool, n)
	for i := range allTrue {
		allTrue[i] = true
	}
	falseAt0 := append([]bool(nil), allTrue...)
	trueAt0 := append([]bool(nil), allFalse...)
	if n > 0 {
		falseAt0[0] = false
		trueAt0[0] = true
	}
	sameA := append([]byte(nil), ba...)
	sameB := append([]byte(nil), ba...)
	diffAt0 := append([]byte(nil), ba...)
	if n > 0 {
		diffAt0[0] ^= 0xff
	}
	pureASCII := make([]byte, n)
	for i := range pureASCII {
		pureASCII[i] = byte(i % 0x80)
	}
	hiAt0 := append([]byte(nil), pureASCII...)
	if n > 0 {
		hiAt0[0] = 0x80
	}

	pair := func(name string, bytes int64, g, w func()) perfCase {
		return perfCase{name: name, bytes: bytes, got: g, want: w}
	}
	// The byte count is what the kernel actually touches — inputs plus output —
	// which is what makes GB/s comparable between a kernel reading two slices
	// and one reading one.
	e32, e64, e8 := int64(n)*4, int64(n)*8, int64(n)

	return []perfCase{
		pair("F32.LessMask", e32*2+e8,
			func() { got.F32.LessMask(mdst, f32a, f32b) },
			func() { want.F32.LessMask(mdst, f32a, f32b) }),
		pair("F64.EqualMask", e64*2+e8,
			func() { got.F64.EqualMask(mdst, f64a, f64b) },
			func() { want.F64.EqualMask(mdst, f64a, f64b) }),
		pair("F32.Select", e32*3+e8,
			func() { got.F32.Select(fdst, mask, f32a, f32b) },
			func() { want.F32.Select(fdst, mask, f32a, f32b) }),
		pair("F64.Select", e64*3+e8,
			func() { got.F64.Select(f64dst, mask, f64a, f64b) },
			func() { want.F64.Select(f64dst, mask, f64a, f64b) }),

		// The searches are measured on both sides of their tradeoff. A kernel
		// that accumulates over the whole input beats a scalar loop when the
		// answer is at the end, and loses catastrophically when it is at the
		// start — the first version of these was 1700x slower than portable Go
		// on a 256 KiB slice whose first byte settled it. Reporting only one
		// case would hide half of that.
		pair("Mask.All (scans all)", e8,
			func() { sinkBool = got.Mask.All(allTrue) },
			func() { sinkBool = want.Mask.All(allTrue) }),
		pair("Mask.All (exits at 0)", e8,
			func() { sinkBool = got.Mask.All(falseAt0) },
			func() { sinkBool = want.Mask.All(falseAt0) }),
		pair("Mask.Any (scans all)", e8,
			func() { sinkBool = got.Mask.Any(allFalse) },
			func() { sinkBool = want.Mask.Any(allFalse) }),
		pair("Mask.Any (exits at 0)", e8,
			func() { sinkBool = got.Mask.Any(trueAt0) },
			func() { sinkBool = want.Mask.Any(trueAt0) }),
		pair("Mask.Count", e8,
			func() { sinkInt = got.Mask.Count(mask) },
			func() { sinkInt = want.Mask.Count(mask) }),
		pair("Mask.And", e8*3,
			func() { got.Mask.And(mdst, mask, mask) },
			func() { want.Mask.And(mdst, mask, mask) }),

		pair("Bytes.IndexByte", e8,
			func() { sinkInt = got.Bytes.IndexByte(ba, 0xfe) },
			func() { sinkInt = want.Bytes.IndexByte(ba, 0xfe) }),
		pair("Bytes.Count", e8,
			func() { sinkInt = got.Bytes.Count(ba, 'a') },
			func() { sinkInt = want.Bytes.Count(ba, 'a') }),
		pair("Bytes.Equal (scans all)", e8*2,
			func() { sinkBool = got.Bytes.Equal(sameA, sameB) },
			func() { sinkBool = want.Bytes.Equal(sameA, sameB) }),
		pair("Bytes.Equal (exits at 0)", e8*2,
			func() { sinkBool = got.Bytes.Equal(sameA, diffAt0) },
			func() { sinkBool = want.Bytes.Equal(sameA, diffAt0) }),
		pair("Bytes.PopCount", e8,
			func() { sinkInt = got.Bytes.PopCount(ba) },
			func() { sinkInt = want.Bytes.PopCount(ba) }),
		pair("Bytes.IsASCII (scans all)", e8,
			func() { sinkBool = got.Bytes.IsASCII(pureASCII) },
			func() { sinkBool = want.Bytes.IsASCII(pureASCII) }),
		pair("Bytes.IsASCII (exits at 0)", e8,
			func() { sinkBool = got.Bytes.IsASCII(hiAt0) },
			func() { sinkBool = want.Bytes.IsASCII(hiAt0) }),
		pair("Bytes.Xor", e8*3,
			func() { got.Bytes.Xor(bdst, ba, bb) },
			func() { want.Bytes.Xor(bdst, ba, bb) }),
		pair("Bytes.ToUpperASCII", e8*2,
			func() { got.Bytes.ToUpperASCII(bdst, ba) },
			func() { want.Bytes.ToUpperASCII(bdst, ba) }),
	}
}

var _ = fmt.Sprintf
