package simd

import (
	"testing"

	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/perf"
)

// What the generic layer costs.
//
// Every exported function reaches its kernel through ops[T](), which does:
//
//	var zero T
//	switch any(zero).(type) {
//	case float64:
//	    return any(opsF64).(*kernel.Ops[T])
//	...
//
// That is two interface conversions and a runtime type assertion per call.
// Boxing a zero value is free — the compiler has a static zero for it — but
// asserting *kernel.Ops[float64] back to *kernel.Ops[T] is a real runtime type
// check that cannot be folded away, because T is only known to the
// instantiation and the assertion is on a concrete-to-concrete pair the
// compiler does not prove equal.
//
// The alternative is switching on the argument slice instead of a zero value,
// which is a single interface-type comparison and no assertion at all. This
// measures both against a direct call so the difference is attributable.
//
// Run with: go test -run TestGenericDispatchCost -v .

// viaOps is the current shape.
func viaOps[T Number](dst, a, b []T) { ops[T]().Add(dst, a, b) }

// viaSwitch is the alternative: switch on the slice, call the concrete field.
func viaSwitch[T Number](dst, a, b []T) {
	switch d := any(dst).(type) {
	case []float32:
		ops[float32]().Add(d, any(a).([]float32), any(b).([]float32))
	case []float64:
		ops[float64]().Add(d, any(a).([]float64), any(b).([]float64))
	case []int32:
		ops[int32]().Add(d, any(a).([]int32), any(b).([]int32))
	case []int64:
		ops[int64]().Add(d, any(a).([]int64), any(b).([]int64))
	}
}

func TestGenericDispatchCost(t *testing.T) {
	opt := perf.DefaultOptions()
	for _, n := range []int{8, 32, 128, 1024} {
		a := make([]float64, n)
		b := make([]float64, n)
		dst := make([]float64, n)

		// The kernel the dispatch eventually reaches, called directly, so the
		// layers above it can be priced against something.
		direct := ops[float64]().Add

		rs := []perf.Result{
			perf.Measure("direct field call", 0, func() { direct(dst, a, b) }, opt),
			perf.Measure("generic via ops[T]()", 0, func() { viaOps(dst, a, b) }, opt),
			perf.Measure("generic via type switch", 0, func() { viaSwitch(dst, a, b) }, opt),
			perf.Measure("exported AddInto", 0, func() { AddInto(dst, a, b) }, opt),
		}
		t.Logf("\nfloat64 n=%d\n%s", n, perf.Table(rs))
		t.Logf("  ops[T]() costs %+.2f ns over a direct call; a type switch costs %+.2f ns",
			rs[1].Min-rs[0].Min, rs[2].Min-rs[0].Min)
	}
}

// A sanity check that the two dispatch shapes agree, so the comparison above
// is between two correct implementations rather than one that skips work.
func TestDispatchShapesAgree(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{10, 20, 30, 40, 50}
	x, y := make([]float64, 5), make([]float64, 5)
	viaOps(x, a, b)
	viaSwitch(y, a, b)
	for i := range x {
		if x[i] != y[i] || x[i] != a[i]+b[i] {
			t.Fatalf("i=%d: ops %v, switch %v, want %v", i, x[i], y[i], a[i]+b[i])
		}
	}
}

var _ kernel.Set // the import is used by the type of active
