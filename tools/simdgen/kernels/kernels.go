// Package kernels is the manifest: every kernel the generator builds, with
// the mapping between its Go signature and its C signature.
//
// Declaring these rather than parsing the C removes the failure mode that
// makes gorse-io/goat unreliable, and it is also the only place the two
// signatures can be reconciled — Go passes a slice as three words while C
// wants a pointer and a length, so something has to say which Go parameter
// supplies the length. Package astcheck verifies the two agree.
//
// The entries are built by loops over an element-type table rather than
// written out one at a time. There are close to ninety and they differ only in
// the type, so spelling each out invites exactly the sort of copy-and-paste
// slip — a float64 kernel wired to the F32 group — that nothing downstream
// would catch.
package kernels

import "github.com/sebishogun/simd/tools/simdgen/spec"

// elem describes one element type in every spelling the generator needs.
type elem struct {
	c      string    // suffix in the C symbol name: f32
	goName string    // suffix in the Go function name: Float32
	slice  spec.Type // []float32
	scalar spec.Type // float32
	group  string    // the kernel.Ops group: F32
	float  bool
}

var elems = []elem{
	{"f32", "Float32", spec.SliceF32, spec.F32, "F32", true},
	{"f64", "Float64", spec.SliceF64, spec.F64, "F64", true},
	{"i32", "Int32", spec.SliceI32, spec.I32, "I32", false},
	{"i64", "Int64", spec.SliceI64, spec.I64, "I64", false},
}

func floats() []elem {
	var out []elem
	for _, e := range elems {
		if e.float {
			out = append(out, e)
		}
	}
	return out
}

// ref picks the reference function for an element type. Some operations need
// different implementations for floats and integers: IEEE minimum is not the
// same operation as integer minimum, and integer Abs wraps where float Abs
// clears a sign bit.
func (e elem) ref(base string) string {
	if e.float {
		return base + "Float"
	}
	return base + "Int"
}

// Thresholds, in elements, below which the dispatcher runs the portable path.
//
// Measured, not guessed — see TestPerfCrossover in internal/amd64. The
// elementwise figure is not the raw crossover of 8 but the point where the
// assembly's margin exceeds the threshold guard's own call cost, which is 16.
// Reductions are worth calling at any length at all.
const (
	thElementwise = 16
	thReduction   = 0
)

func base(n string) spec.CArg  { return spec.CArg{From: n, Part: spec.Base} }
func lenOf(n string) spec.CArg { return spec.CArg{From: n, Part: spec.Len} }
func val(n string) spec.CArg   { return spec.CArg{From: n, Part: spec.Value} }
func out() spec.CArg           { return spec.CArg{Part: spec.ResultAddr} }

func sl(name string, t spec.Type) spec.Param { return spec.Param{Name: name, Type: t} }

// ---------- kernel shapes ----------

func binary(op, field, refFunc string, e elem, skip ...string) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op2go(op) + e.goName,
		Group: e.group, Field: field, RefFunc: refFunc,
		Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice), sl("b", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), lenOf("dst")},
		Threshold: thElementwise, SkipOn: skip,
	}
}

func unary(op, field, refFunc string, e elem, skip ...string) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op2go(op) + e.goName,
		Group: e.group, Field: field, RefFunc: refFunc,
		Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst")},
		Threshold: thElementwise, SkipOn: skip,
	}
}

func scalarOp(op, field, refFunc string, e elem, skip ...string) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op2go(op) + e.goName,
		Group: e.group, Field: field, RefFunc: refFunc,
		Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice), sl("s", e.scalar)},
		CArgs:     []spec.CArg{base("dst"), base("a"), val("s"), lenOf("dst")},
		Threshold: thElementwise, SkipOn: skip,
	}
}

func clampK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_clamp_" + e.c, GoName: "clamp" + e.goName,
		Group: e.group, Field: "Clamp", RefFunc: e.ref("Clamp"),
		Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
			sl("lo", e.scalar), sl("hi", e.scalar)},
		CArgs:     []spec.CArg{base("dst"), base("a"), val("lo"), val("hi"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

func fillK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_fill_" + e.c, GoName: "fill" + e.goName,
		Group: e.group, Field: "Fill", RefFunc: "Fill",
		Params:    []spec.Param{sl("dst", e.slice), sl("v", e.scalar)},
		CArgs:     []spec.CArg{base("dst"), val("v"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

func rampK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_ramp_" + e.c, GoName: "ramp" + e.goName,
		Group: e.group, Field: "Ramp", RefFunc: "Ramp",
		Params:    []spec.Param{sl("dst", e.slice), sl("start", e.scalar), sl("step", e.scalar)},
		CArgs:     []spec.CArg{base("dst"), val("start"), val("step"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

func lerpK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_lerp_" + e.c, GoName: "lerp" + e.goName,
		Group: e.group, Field: "Lerp", RefFunc: "Lerp",
		Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
			sl("b", e.slice), sl("t", e.scalar)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), val("t"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

func axpyK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_addscaled_" + e.c, GoName: "addScaled" + e.goName,
		Group: e.group, Field: "AddScaled", RefFunc: "AddScaled",
		Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
			sl("b", e.slice), sl("s", e.scalar)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), val("s"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

func reduce1(op, field, refFunc string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op2go(op) + e.goName,
		Group: e.group, Field: field, RefFunc: refFunc,
		Params:    []spec.Param{sl("a", e.slice)},
		Result:    &spec.Param{Name: "ret", Type: e.scalar},
		CArgs:     []spec.CArg{out(), base("a"), lenOf("a")},
		Threshold: thReduction,
	}
}

func reduce2(op, field, refFunc string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op2go(op) + e.goName,
		Group: e.group, Field: field, RefFunc: refFunc,
		Params:    []spec.Param{sl("a", e.slice), sl("b", e.slice)},
		Result:    &spec.Param{Name: "ret", Type: e.scalar},
		CArgs:     []spec.CArg{out(), base("a"), base("b"), lenOf("a")},
		Threshold: thReduction,
	}
}

// op2go turns a C operation name into the Go identifier prefix, where the two
// spellings differ.
func op2go(op string) string {
	switch op {
	case "recip":
		return "reciprocal"
	case "roundeven":
		return "roundToEven"
	case "addscalar":
		return "addScalar"
	case "subscalar":
		return "subScalar"
	case "divscalar":
		return "divScalar"
	case "addscaled":
		return "addScaled"
	}
	return op
}

// ---------- the manifest ----------

// Arith is everything in csrc/arith.c.
func Arith() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range elems {
		ks = append(ks,
			binary("add", "Add", "Add", e),
			binary("sub", "Sub", "Sub", e),
			binary("mul", "Mul", "Mul", e),
			binary("minimum", "Minimum", e.ref("Minimum"), e),
			binary("maximum", "Maximum", e.ref("Maximum"), e),
			unary("abs", "Abs", e.ref("Abs"), e),
			unary("neg", "Neg", e.ref("Neg"), e),
			scalarOp("scale", "Scale", "Scale", e),
			scalarOp("addscalar", "AddScalar", "AddScalar", e),
			scalarOp("subscalar", "SubScalar", "SubScalar", e),
			clampK(e), fillK(e), lerpK(e), axpyK(e),
			// Ramp and Reverse are deliberately absent. Ramp needs an index
			// vector [0,1,2,...] as a constant, which on every architecture but
			// amd64 is reached through a high/low instruction pair this
			// generator does not rewrite. Reverse is a permutation that LLVM
			// will not vectorize from a plain loop; it needs target-specific
			// shuffles. Both stay portable, and neither is usually hot.
		)
	}
	for _, e := range floats() {
		ks = append(ks,
			binary("div", "Div", "Div", e),
			scalarOp("divscalar", "DivScalar", "DivScalar", e),
			unary("sqrt", "Sqrt", "Sqrt", e),
			unary("recip", "Reciprocal", "Reciprocal", e),
			// Rounding arrived with SSE4.1, so the baseline x86-64 tier has no
			// instruction for any of these and clang emits a libm call — which
			// cannot be linked in Plan 9 assembly. ppc64le has no roundeven
			// and loong64 has no round, for the same reason.
			unary("floor", "Floor", "Floor", e, "amd64/sse2"),
			unary("ceil", "Ceil", "Ceil", e, "amd64/sse2"),
			unary("trunc", "Trunc", "Trunc", e, "amd64/sse2"),
			unary("round", "Round", "Round", e, "amd64/sse2", "loong64"),
			unary("roundeven", "RoundToEven", "RoundToEven", e, "amd64/sse2", "ppc64le"),
		)
	}
	return ks
}

// reduceScalar is a reduction with one extra scalar operand, such as the sum
// of squared deviations from a mean.
func reduceScalar(op, field, refFunc string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op2go(op) + e.goName,
		Group: e.group, Field: field, RefFunc: refFunc,
		Params:    []spec.Param{sl("a", e.slice), sl("c", e.scalar)},
		Result:    &spec.Param{Name: "ret", Type: e.scalar},
		CArgs:     []spec.CArg{out(), base("a"), val("c"), lenOf("a")},
		Threshold: thReduction,
	}
}

// diffK writes successive differences, and so needs both lengths: the output
// is one element shorter than the input.
func diffK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_diff_" + e.c, GoName: "diff" + e.goName,
		Group: e.group, Field: "Diff", RefFunc: "Diff",
		Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst"), lenOf("a")},
		Threshold: thElementwise,
	}
}

// minMaxK is a horizontal minimum or maximum.
//
// Its threshold is 1, not 0: the kernel reads a[0] before looping, so an empty
// slice must reach the portable implementation, which panics as documented
// rather than reading out of bounds.
func minMaxK(op, field, refFunc string, e elem) spec.Kernel {
	k := reduce1(op, field, refFunc, e)
	k.Threshold = 1
	return k
}

// Reduce is everything in csrc/reduce.c.
func Reduce() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range elems {
		ks = append(ks,
			minMaxK("minr", "Min", e.ref("MinReduce"), e),
			minMaxK("maxr", "Max", e.ref("MaxReduce"), e),
			reduce1("sumsq", "SumSquares", e.ref("SumSquares"), e),
			reduceScalar("sumsqdev", "SumSqDev", e.ref("SumSqDev"), e),
			reduce2("sumsqdiff", "SumSqDiff", e.ref("SumSqDiff"), e),
			diffK(e),
		)
	}
	for _, e := range floats() {
		ks = append(ks,
			reduce1("sum", "Sum", "SumFloat", e),
			reduce2("dot", "Dot", "DotFloat", e),
			reduce1("l1norm", "L1Norm", "L1NormFloat", e),
			reduce2("l1diff", "L1Diff", "L1DiffFloat", e),
		)
	}
	// Integer sums and products need no lane discipline, because integer
	// arithmetic is associative and no accumulation order is observable.
	// Floating-point Prod is deliberately absent: products overflow and
	// underflow far more readily than sums, so reassociating one changes which
	// intermediate blows up rather than merely the rounding.
	for _, e := range elems {
		if e.float {
			continue
		}
		ks = append(ks,
			reduce1("sum", "Sum", "SumInt", e),
			reduce1("prod", "Prod", "ProdInt", e),
			reduce2("dot", "Dot", "DotInt", e),
		)
	}
	return ks
}

// Source is a C file and the kernels compiled from it.
type Source struct {
	// Path is relative to the repository root.
	Path string
	// Kernels are the functions to extract from it.
	Kernels []spec.Kernel
}

// All is every source file the generator processes.
var All = []Source{
	{Path: "csrc/arith.c", Kernels: Arith()},
	{Path: "csrc/reduce.c", Kernels: Reduce()},
}
