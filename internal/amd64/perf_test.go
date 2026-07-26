//go:build amd64 && !purego

package amd64

import (
	"fmt"
	"testing"
	"time"

	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/perf"
)

// Performance investigation, run with:
//
//	go test -run TestPerf -v ./internal/amd64/
//
// These are not pass/fail tests — they print tables. They exist because two
// numbers in the headline benchmarks needed explaining rather than accepting:
// why the public API is only ~1.6x at n=32 when the raw kernel is much more
// than that, and why Add gets worse as the input grows while Sum does not.

func alloc(n int) (a, b, dst []float64) {
	a, b, dst = make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range a {
		a[i], b[i] = float64(i)*0.5, float64(i)*0.25
	}
	return
}

// TestPerfCallChain attributes the cost of each layer between the caller and
// the assembly.
//
// The chain is: generic API -> type switch -> function pointer in the kernel
// set -> threshold guard -> assembly. None of those calls can be inlined,
// because assembly never is and a call through a struct field is indirect. At
// large n that is noise; at small n it is most of the time.
func TestPerfCallChain(t *testing.T) {
	opt := perf.DefaultOptions()
	for _, n := range []int{4, 8, 16, 32, 64, 128, 256} {
		x, y, dst := alloc(n)
		var set kernel.Set
		set.F64.Add = addFloat64AVX512Guarded

		rs := []perf.Result{
			perf.Measure("raw assembly", 0, func() { addFloat64AVX512(dst, x, y) }, opt),
			perf.Measure("+ threshold guard", 0, func() { addFloat64AVX512Guarded(dst, x, y) }, opt),
			perf.Measure("+ indirect call", 0, func() { set.F64.Add(dst, x, y) }, opt),
			perf.Measure("portable Go", 0, func() { goAdd(dst, x, y) }, opt),
		}
		t.Logf("\nAdd float64, n=%d\n%s", n, perf.Table(rs))
		raw, guard, indirect := rs[0].Min, rs[1].Min, rs[2].Min
		t.Logf("  guard %+.2f ns, indirect call %+.2f ns; dispatch is %.0f%% of "+
			"the total at this size", guard-raw, indirect-guard,
			100*(indirect-raw)/indirect)
	}
}

// TestPerfBandwidth walks the cache hierarchy.
//
// Add moves 24 bytes per element (read a, read b, write dst); Sum moves 8.
// This machine has 48 KiB of L1d and 1 MiB of L2 per core, and 32 MiB of L3
// per die, so Add leaves L1 near n=2048 and L2 near n=43000 while Sum stays
// resident three times longer.
//
// The number to watch is GB/s. While it keeps climbing with vector width the
// kernel is compute-bound and a wider unit helps. Once it flattens the kernel
// is waiting on memory, both tiers converge, and no amount of AVX-512 changes
// anything.
func TestPerfBandwidth(t *testing.T) {
	opt := perf.DefaultOptions()
	sizes := []int{256, 1024, 2048, 8192, 43000, 262144, 1048576}

	t.Log("\nAdd float64 — 24 bytes of traffic per element")
	var rows []perf.Result
	for _, n := range sizes {
		x, y, dst := alloc(n)
		bytes := int64(n) * 24
		rows = append(rows,
			perf.Measure(fmt.Sprintf("n=%-8d sse2", n), bytes,
				func() { addFloat64SSE2(dst, x, y) }, opt),
			perf.Measure(fmt.Sprintf("n=%-8d avx512", n), bytes,
				func() { addFloat64AVX512(dst, x, y) }, opt),
			perf.Measure(fmt.Sprintf("n=%-8d portable", n), bytes,
				func() { goAdd(dst, x, y) }, opt),
		)
	}
	t.Logf("\n%s", perf.Table(rows))

	t.Log("\nSum float64 — 8 bytes of traffic per element")
	rows = nil
	for _, n := range sizes {
		x, _, _ := alloc(n)
		bytes := int64(n) * 8
		rows = append(rows,
			perf.Measure(fmt.Sprintf("n=%-8d sse2", n), bytes,
				func() { sinkF = sumFloat64SSE2(x) }, opt),
			perf.Measure(fmt.Sprintf("n=%-8d avx512", n), bytes,
				func() { sinkF = sumFloat64AVX512(x) }, opt),
			perf.Measure(fmt.Sprintf("n=%-8d portable", n), bytes,
				func() { sinkF = goSum(x) }, opt),
		)
	}
	t.Logf("\n%s", perf.Table(rows))
}

// TestPerfCrossover finds, per kernel, the length at which calling into
// assembly starts paying for itself.
//
// This is what the manifest's Threshold should be set from. It is measured
// rather than guessed because it depends on both the operation and the element
// type: a reduction does more arithmetic per byte than an elementwise add, so
// it crosses over sooner.
func TestPerfCrossover(t *testing.T) {
	opt := perf.DefaultOptions()
	opt.Patience = 120 * time.Millisecond // this sweep is wide

	type kernel struct {
		name string
		asm  func(n int, x, y, dst []float64)
		ref  func(n int, x, y, dst []float64)
	}
	kernels := []kernel{
		{"Add",
			func(n int, x, y, dst []float64) { addFloat64AVX512(dst, x, y) },
			func(n int, x, y, dst []float64) { goAdd(dst, x, y) }},
		{"Sum",
			func(n int, x, y, dst []float64) { sinkF = sumFloat64AVX512(x) },
			func(n int, x, y, dst []float64) { sinkF = goSum(x) }},
		{"Dot",
			func(n int, x, y, dst []float64) { sinkF = dotFloat64AVX512(x, y) },
			func(n int, x, y, dst []float64) { sinkF = goDot(x, y) }},
	}

	for _, k := range kernels {
		var b []string
		crossover := -1
		for _, n := range []int{1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128} {
			x, y, dst := alloc(n)
			asm := perf.Measure("", 0, func() { k.asm(n, x, y, dst) }, opt)
			ref := perf.Measure("", 0, func() { k.ref(n, x, y, dst) }, opt)
			ratio := ref.Min / asm.Min
			if ratio >= 1.0 && crossover < 0 {
				crossover = n
			}
			b = append(b, fmt.Sprintf("  n=%-4d asm %7.3f ns  ref %7.3f ns  %5.2fx",
				n, asm.Min, ref.Min, ratio))
		}
		t.Logf("\n%s float64 — assembly overtakes portable Go at n=%d\n%s",
			k.name, crossover, joinLines(b))
	}
}

func joinLines(s []string) string {
	out := ""
	for _, l := range s {
		out += l + "\n"
	}
	return out
}
