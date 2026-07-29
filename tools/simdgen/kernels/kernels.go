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

import (
	"fmt"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/spec"
)

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
	{"i8", "Int8", spec.SliceI8, spec.I8, "I8", false},
	{"i16", "Int16", spec.SliceI16, spec.I16, "I16", false},
	{"u8", "Uint8", spec.SliceU8, spec.U8, "U8", false},
	{"u16", "Uint16", spec.SliceU16, spec.U16, "U16", false},
	{"u32", "Uint32", spec.SliceU32, spec.U32, "U32", false},
	{"u64", "Uint64", spec.SliceU64, spec.U64, "U64", false},
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

// saturating returns the element types that have saturating add and subtract.
//
// The 64-bit ones do not, and could not be written the way the others are:
// the kernel widens, clamps and narrows, and there is no integer wider than
// 64 bits to widen into. Writing them as an overflow test instead is possible
// but does not vectorize into the single instruction that is the whole reason
// for having them, so those stay portable.
func saturating() []elem {
	var out []elem
	for _, e := range elems {
		switch e.c {
		case "i8", "i16", "i32", "u8", "u16", "u32":
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
	// thNary is higher than the binary threshold on purpose. What an n-ary
	// kernel saves is memory traffic, and while the data fits in cache there
	// is no traffic to save — the repeated binary calls hit cache on their
	// later passes and cost little more. Measured crossover, not a guess.
	thNary = 256
)

// withThresholds attaches per-architecture thresholds to a kernel built by one
// of the shape helpers.
func withThresholds(k spec.Kernel, on map[string]int) spec.Kernel {
	k.ThresholdOn = on
	return k
}

func base(n string) spec.CArg  { return spec.CArg{From: n, Part: spec.Base} }
func lenOf(n string) spec.CArg { return spec.CArg{From: n, Part: spec.Len} }
func val(n string) spec.CArg   { return spec.CArg{From: n, Part: spec.Value} }
func out() spec.CArg           { return spec.CArg{Part: spec.ResultAddr} }
func out2() spec.CArg          { return spec.CArg{Part: spec.ResultAddr2} }

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

// shiftK is a shift or rotate: elementwise, with an unsigned count that is not
// the element type.
//
// The count travels as uint64 whatever the element width, because the contract
// is defined for counts above the width — Go says a shift by 32 or more of a
// uint32 is zero — and a count in the element type could not express one.
func shiftK(op, field string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op + e.goName,
		Group: e.group, Field: field, RefFunc: field,
		Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
			{Name: "s", Type: spec.U64}},
		CArgs:     []spec.CArg{base("dst"), base("a"), val("s"), lenOf("dst")},
		Threshold: thElementwise,
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
	case "popcount":
		return "onesCount"
	case "leadingzeros":
		return "leadingZeros"
	case "trailingzeros":
		return "trailingZeros"
	case "reversebits":
		return "reverseBits"
	case "byteswap":
		return "byteSwap"
	case "subscalar":
		return "subScalar"
	case "divscalar":
		return "divScalar"
	case "addscaled":
		return "addScaled"
	case "addsat":
		return "satAdd"
	case "subsat":
		return "satSub"
	}
	return op
}

// ---------- the manifest ----------

// Arith is everything in csrc/arith.c.
// reverseK is the in-place-capable reverse. Its C takes no __restrict — d and
// a are the same slice in the common case — and it exists only for the four
// wide types, because the narrow ones would want a byte shuffle rather than the
// element swap the macro generates.
func reverseK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_reverse_" + e.c, GoName: "reverse" + e.goName,
		Group: e.group, Field: "Reverse", RefFunc: "Reverse",
		Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

// argK is a reduction that returns a position rather than a value.
//
// Only the four wide types: the index vector has to have the same lane count as
// the value vector, and for an 8-bit element that would mean four index vectors
// per value vector, which costs more than it saves.
func argK(op, field string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_arg" + op + "_" + e.c, GoName: "arg" + field + e.goName,
		Group: e.group, Field: "Arg" + field, RefFunc: e.ref("Arg" + field),
		Params: []spec.Param{sl("a", e.slice)},
		Result: &spec.Param{Name: "ret", Type: spec.Int},
		CArgs:  []spec.CArg{out(), base("a"), lenOf("a")},
		// One, not zero. ArgMin and ArgMax panic on an empty slice — there is
		// no index to return — and a kernel cannot panic, so the empty case has
		// to reach the reference. A threshold of 1 is what routes it there.
		// With thReduction the kernel answered 0 for an empty slice instead,
		// which the existing contract test caught.
		Threshold: 1,
	}
}

// scanK is a prefix scan. Only min and max: sum and product are serial under
// the bit-identity contract, because every partial result of a scan is written
// to dst and the log-shift form regroups the additions. See csrc/scan.c.
func scanK(op, field string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_cum" + op + "_" + e.c, GoName: "cum" + field + e.goName,
		Group: e.group, Field: "Cum" + field, RefFunc: e.ref("Cum" + field),
		Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

// Scan is the associative prefix-scan family, and it is integers only.
//
// The float versions were built, measured and dropped. A prefix scan costs
// log2(lanes) combines per block instead of one per element, which is only
// worth it if the combine is cheap. Integer minimum is one instruction, so it
// is: CumMin on int32 over a million elements is 266µs against the scalar
// loop's 626µs, 2.4x faster. IEEE-754-2019 minimum on floats is a five-operation
// select chain — NaN propagation and -0 ordering are not what the hardware
// instruction does — so the scan pays five times that, and float64 measured
// 877µs against the scalar loop's 601µs, 46% *slower*.
//
// So the rule that makes a scan vectorizable at all (associativity) is not
// enough on its own; the combine has to be cheap too.
func Scan() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range elems {
		switch e.group {
		case "I32", "I64":
			ks = append(ks, scanK("min", "Min", e), scanK("max", "Max", e))
		}
	}
	return ks
}

// minMaxPairK returns both extremes from one pass, which is the point of having
// it at all rather than calling Min and Max: the array is read once.
//
// It is the first kernel with two results. Each is written through a pointer to
// its frame slot, so a pair costs one extra integer argument register; see
// spec.Result2.
func minMaxPairK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_minmax_" + e.c, GoName: "minMax" + e.goName,
		Group: e.group, Field: "MinMax", RefFunc: e.ref("MinMax"),
		Params:  []spec.Param{sl("a", e.slice)},
		Result:  &spec.Param{Name: "lo", Type: e.scalar},
		Result2: &spec.Param{Name: "hi", Type: e.scalar},
		CArgs: []spec.CArg{{Part: spec.ResultAddr}, {Part: spec.ResultAddr2},
			base("a"), lenOf("a")},
		// One rather than zero: MinMax of an empty slice panics, and a kernel
		// cannot, so the empty case has to reach the reference. Same reasoning
		// as argK.
		Threshold: 1,
	}
}

// ArgReduce is the position-returning reduction family, plus the pair-returning
// MinMax which shares its file.
func ArgReduce() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range elems {
		switch e.group {
		case "F32", "F64", "I32", "I64":
			ks = append(ks, argK("min", "Min", e), argK("max", "Max", e),
				minMaxPairK(e))
		}
	}
	return ks
}

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
			clampK(e), fillK(e), lerpK(e), axpyK(e), rampK(e),
			// Ramp and Reverse are deliberately absent. Ramp needs an index
			// vector [0,1,2,...] as a constant, which on every architecture but
			// amd64 is reached through a high/low instruction pair this
			// generator does not rewrite. Reverse is a permutation that LLVM
			// will not vectorize from a plain loop; it needs target-specific
			// shuffles. Both stay portable, and neither is usually hot.
		)
	}
	// Shifts and rotates, integers only. See the note above the SHIFT_LEFT
	// macro in csrc/arith.c for why the count is clamped rather than passed
	// through: a shift at or above the element width is undefined in C and the
	// hardware disagrees about it, so the kernels implement Go's rule
	// explicitly.
	for _, e := range elems {
		if e.float {
			continue
		}
		ks = append(ks,
			shiftK("shl", "Shl", e), shiftK("shr", "Shr", e),
			shiftK("rotl", "Rotl", e), shiftK("rotr", "Rotr", e),
			unary("popcount", "OnesCount", "OnesCount", e),
			unary("leadingzeros", "LeadingZeros", "LeadingZeros", e),
			unary("trailingzeros", "TrailingZeros", "TrailingZeros", e),
			unary("reversebits", "ReverseBits", "ReverseBits", e),
		)
		// A byte is its own reversal, so there is no eight-bit swap.
		if e.c != "i8" && e.c != "u8" {
			ks = append(ks, unary("byteswap", "ByteSwap", "ByteSwap", e))
		}
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
	// Reverse only exists for the four wide types; the C macro is not
	// instantiated for the narrow ones, which would want a byte shuffle rather
	// than the element swap it generates.
	for _, e := range elems {
		switch e.group {
		case "F32", "F64", "I32", "I64":
			ks = append(ks, reverseK(e))
		}
	}
	for _, e := range saturating() {
		ks = append(ks,
			binary("addsat", "SatAdd", "SatAdd", e),
			binary("subsat", "SatSub", "SatSub", e),
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
// Its threshold is never 0: the kernel reads a[0] before looping, so an empty
// slice must reach the portable implementation, which panics as documented
// rather than reading out of bounds.
//
// For everything four bytes wide and up, 1 is also the right answer — the
// assembly wins at every length that is not empty. The narrow types are the
// exception and it is the only place in the manifest where the element width
// changes a threshold. A byte-wide reduction over sixteen elements is sixteen
// bytes of work, half a register, and the call is not paid for: measured on
// this AVX-512 host, int8 Min is 12% *slower* than the portable loop at n=16
// and 36% faster at n=24; uint16 loses by 17% at n=8 and wins by 25% at n=16.
// So the crossover is where those measurements put it.
//
// The reductions that are shaped like sums do not need this. They accumulate
// into lanes rather than folding pairwise, so their portable form is the one
// that costs more per element, and int8 Sum already wins by 18% at n=16.
func minMaxK(op, field, refFunc string, e elem) spec.Kernel {
	k := reduce1(op, field, refFunc, e)
	switch e.c {
	case "i8", "u8":
		k.Threshold = 24
	case "i16", "u16":
		k.Threshold = 16
	default:
		k.Threshold = 1
	}
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
			reduce1("l1norm", "L1Norm", "L1NormInt", e),
			reduce2("l1diff", "L1Diff", "L1DiffInt", e),
		)
	}
	return ks
}

// ---------- comparisons, selection and masks ----------

// Byte-domain thresholds are higher than the numeric ones because an element
// is one byte: sixteen of them is a single register's worth, and the call is
// not yet paid for. The scanning kernels get a higher one again — they
// accumulate over the whole input where the portable versions can exit at the
// first match, so they need enough length for the vector width to repay that.
const (
	thBytes = 32
	thScan  = 64
)

// stdlibAsm is where the standard library already has a hand-written assembly
// implementation of the identical function, which is the only place in this
// package where the portable path is not a plain Go loop.
//
// bytealg carries assembly for amd64, arm64, ppc64x and s390x, and generic Go
// for riscv64 and loong64. So on the first four the kernel has to beat tuned
// assembly before it is worth a call it cannot inline, and on the last two it
// only has to beat a loop. One threshold cannot express both.
//
// The numbers are measured on this machine and are the length at which the
// kernel starts winning; see the crossover sweeps in BenchmarkText*. Where a
// kernel does not reliably win at any length the threshold is `never`, which
// leaves the standard library in place on those architectures and the kernel
// in place on the two where it is the only vectorized implementation there is.
var stdlibAsm = []string{"amd64", "arm64", "ppc64le", "s390x"}

// onStdlibAsm builds a ThresholdOn map setting n for every architecture whose
// standard library is assembly.
func onStdlibAsm(n int) map[string]int {
	m := make(map[string]int, len(stdlibAsm))
	for _, a := range stdlibAsm {
		m[a] = n
	}
	return m
}

// never is a threshold no caller reaches, which keeps the standard library's
// implementation in place. Used only where it wins at every measured length.
const never = 1 << 30

// cmpK is a comparison writing one bool per element.
func cmpK(op, field string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op + e.goName + "Mask",
		Group: e.group, Field: field, RefFunc: field,
		Params:    []spec.Param{sl("dst", spec.SliceB), sl("a", e.slice), sl("b", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

func cmpScalarK(op, field string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "s_" + e.c, GoName: op + "Scalar" + e.goName + "Mask",
		Group: e.group, Field: field, RefFunc: field,
		Params:    []spec.Param{sl("dst", spec.SliceB), sl("a", e.slice), sl("v", e.scalar)},
		CArgs:     []spec.CArg{base("dst"), base("a"), val("v"), lenOf("dst")},
		Threshold: thElementwise,
	}
}

func selectK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_select_" + e.c, GoName: "select" + e.goName,
		Group: e.group, Field: "Select", RefFunc: "Select",
		Params: []spec.Param{sl("dst", e.slice), sl("mask", spec.SliceB),
			sl("yes", e.slice), sl("no", e.slice)},
		CArgs: []spec.CArg{base("dst"), base("mask"), base("yes"), base("no"),
			lenOf("dst")},
		Threshold: thElementwise,
	}
}

// maskReduce is All, Any or Count over a []bool.
func maskReduce(op, field string, res spec.Type) spec.Kernel {
	return spec.Kernel{
		CName: "simd_mask_" + op, GoName: "mask" + field,
		Group: "Mask", Field: field, RefFunc: "Mask" + field,
		Params:    []spec.Param{sl("m", spec.SliceB)},
		Result:    &spec.Param{Name: "ret", Type: res},
		CArgs:     []spec.CArg{out(), base("m"), lenOf("m")},
		Threshold: thBytes,
	}
}

func maskBinary(op, field string) spec.Kernel {
	return spec.Kernel{
		CName: "simd_mask_" + op, GoName: "mask" + field,
		Group: "Mask", Field: field, RefFunc: "Mask" + field,
		Params: []spec.Param{sl("dst", spec.SliceB), sl("a", spec.SliceB),
			sl("b", spec.SliceB)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), lenOf("dst")},
		Threshold: thBytes,
	}
}

// Compare is everything in csrc/compare.c.
func Compare() []spec.Kernel {
	cmps := []struct{ op, field string }{
		{"eq", "Equal"}, {"ne", "NotEqual"},
		{"lt", "Less"}, {"le", "LessEqual"},
		{"gt", "Greater"}, {"ge", "GreaterEqual"},
	}
	var ks []spec.Kernel
	for _, e := range elems {
		for _, c := range cmps {
			ks = append(ks,
				cmpK(c.op, c.field+"Mask", e),
				cmpScalarK(c.op, c.field+"ScalarMask", e))
		}
		ks = append(ks, selectK(e))
	}
	ks = append(ks,
		maskReduce("all", "All", spec.B),
		maskReduce("any", "Any", spec.B),
		maskReduce("count", "Count", spec.Int),
		maskBinary("and", "And"),
		maskBinary("or", "Or"),
		maskBinary("xor", "Xor"),
		spec.Kernel{
			CName: "simd_mask_not", GoName: "maskNot",
			Group: "Mask", Field: "Not", RefFunc: "MaskNot",
			Params:    []spec.Param{sl("dst", spec.SliceB), sl("a", spec.SliceB)},
			CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst")},
			Threshold: thBytes,
		})
	return ks
}

// ---------- bytes, bits and ASCII text ----------

// byteScan is a whole-slice query returning a scalar.
func byteScan(cname, goName, field, refFunc string, res spec.Type, withByte bool) spec.Kernel {
	k := spec.Kernel{
		CName: cname, GoName: goName,
		Group: "Bytes", Field: field, RefFunc: refFunc,
		Params:    []spec.Param{sl("b", spec.SliceU8)},
		Result:    &spec.Param{Name: "ret", Type: res},
		CArgs:     []spec.CArg{out(), base("b"), lenOf("b")},
		Threshold: thScan,
	}
	if withByte {
		k.Params = append(k.Params, sl("c", spec.U8))
		k.CArgs = []spec.CArg{out(), base("b"), val("c"), lenOf("b")}
	}
	return k
}

func byteBinary(op, field string) spec.Kernel {
	return spec.Kernel{
		CName: "simd_bit_" + op, GoName: "bit" + field,
		Group: "Bytes", Field: field, RefFunc: "Bit" + field,
		Params: []spec.Param{sl("dst", spec.SliceU8), sl("a", spec.SliceU8),
			sl("b", spec.SliceU8)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), lenOf("dst")},
		Threshold: thBytes,
	}
}

func byteMap(cname, goName, field, refFunc string) spec.Kernel {
	return spec.Kernel{
		CName: cname, GoName: goName,
		Group: "Bytes", Field: field, RefFunc: refFunc,
		Params:    []spec.Param{sl("dst", spec.SliceU8), sl("b", spec.SliceU8)},
		CArgs:     []spec.CArg{base("dst"), base("b"), lenOf("dst")},
		Threshold: thBytes,
	}
}

// Bytes is everything in csrc/bytes.c.
func Bytes() []spec.Kernel {
	return []spec.Kernel{
		// bytealg.Count on amd64 compares a register and popcounts the mask,
		// which this kernel matches to within a few percent and does not
		// reliably beat at any length. On riscv64 and loong64 it is the only
		// vector implementation there is.
		withThresholds(
			byteScan("simd_count_byte", "countByte", "Count", "CountByte", spec.Int, true),
			onStdlibAsm(never)),
		// Measured crossover 1024 on amd64: bytealg.IndexByte is assembly and
		// inlinable, so a call cannot pay for itself until there is a
		// kilobyte of work behind it.
		withThresholds(
			byteScan("simd_index_byte", "indexByte", "IndexByte", "IndexByte", spec.Int, true),
			onStdlibAsm(1024)),
		byteScan("simd_last_index_byte", "lastIndexByte", "LastIndexByte",
			"LastIndexByte", spec.Int, true),
		byteScan("simd_popcount", "popCount", "PopCount", "PopCount", spec.Int, false),
		byteScan("simd_is_ascii", "isASCII", "IsASCII", "IsASCII", spec.B, false),
		byteScan("simd_valid_utf8", "validUTF8", "ValidUTF8", "ValidUTF8", spec.B, false),
		{
			// Equal is length-sensitive in a way the kernel is not: it reports
			// whether the two slices hold the same bytes *and* are the same
			// length, so a mismatch is false without reading anything.
			CName: "simd_equal_bytes", GoName: "equalBytes",
			Group: "Bytes", Field: "Equal", RefFunc: "EqualBytes",
			Params:    []spec.Param{sl("a", spec.SliceU8), sl("b", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.B},
			CArgs:     []spec.CArg{out(), base("a"), base("b"), lenOf("a")},
			RefWhen:   "len(a) != len(b)",
			Threshold: thScan,
			// bytes.Equal is memequal, which is already bandwidth-bound: the
			// kernel measured within 7% of it either way from 1 KiB to 1 MiB.
			// There is nothing for a vector to add to a memory compare.
			ThresholdOn: onStdlibAsm(never),
		},

		byteBinary("and", "And"),
		byteBinary("or", "Or"),
		byteBinary("xor", "Xor"),
		byteBinary("andnot", "AndNot"),
		byteMap("simd_bit_not", "bitNot", "Not", "BitNot"),
		{
			CName: "simd_fill_bytes", GoName: "fillBytes",
			Group: "Bytes", Field: "Fill", RefFunc: "FillBytes",
			Params:    []spec.Param{sl("dst", spec.SliceU8), sl("v", spec.U8)},
			CArgs:     []spec.CArg{base("dst"), val("v"), lenOf("dst")},
			Threshold: thBytes,
		},

		{
			// Two lengths, because Compare's answer depends on both: content
			// decides it where the slices differ, and length where one is a
			// prefix of the other. The guard leaves a two-length kernel to
			// reconcile them itself.
			CName: "simd_compare_bytes", GoName: "compareBytes",
			Group: "Bytes", Field: "Compare", RefFunc: "CompareBytes",
			Params:    []spec.Param{sl("a", spec.SliceU8), sl("b", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("a"), base("b"), lenOf("a"), lenOf("b")},
			Threshold: thScan,
			// Measured crossover 2048 on amd64; +20% from 16 KiB.
			ThresholdOn: onStdlibAsm(2048),
		},
		{
			CName: "simd_equal_fold_ascii", GoName: "equalFoldASCII",
			Group: "Bytes", Field: "EqualFoldASCII", RefFunc: "EqualFoldASCII",
			Params:    []spec.Param{sl("a", spec.SliceU8), sl("b", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.B},
			CArgs:     []spec.CArg{out(), base("a"), base("b"), lenOf("a")},
			RefWhen:   "len(a) != len(b)",
			Threshold: thScan,
		},
		{
			// The set length is a second length, so the guard does not try to
			// clamp the haystack against it.
			CName: "simd_index_any", GoName: "indexAny",
			Group: "Bytes", Field: "IndexAny", RefFunc: "IndexAny",
			Params:    []spec.Param{sl("b", spec.SliceU8), sl("chars", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("b"), base("chars"), lenOf("b"), lenOf("chars")},
			Threshold: thScan,
		},
		{
			// The complement of IndexAny, and the primitive under trimming.
			CName: "simd_index_not_any", GoName: "indexNotAny",
			Group: "Bytes", Field: "IndexNotAny", RefFunc: "IndexNotAny",
			Params:    []spec.Param{sl("b", spec.SliceU8), sl("chars", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("b"), base("chars"), lenOf("b"), lenOf("chars")},
			Threshold: thScan,
		},
		{
			CName: "simd_last_index_not_any", GoName: "lastIndexNotAny",
			Group: "Bytes", Field: "LastIndexNotAny", RefFunc: "LastIndexNotAny",
			Params:    []spec.Param{sl("b", spec.SliceU8), sl("chars", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("b"), base("chars"), lenOf("b"), lenOf("chars")},
			Threshold: thScan,
		},
		{
			CName: "simd_count_any", GoName: "countAny",
			Group: "Bytes", Field: "CountAny", RefFunc: "CountAny",
			// Skipped on ppc64le. With the TOC rewrite enabled this is the one
			// kernel of 469 that corrupts memory: bisected to it in fourteen
			// runs, and its constant addressing verified correct by hand, so
			// the fault is something else about it — most likely the 256-byte
			// character-set table it writes at -256(r1), the deepest stack use
			// of any kernel here and close to the 288-byte protected zone.
			//
			// Not registering it is the same arrangement every partial backend
			// in this library uses. The portable implementation stands in, so
			// this is a performance property and never a correctness one.
			SkipOn:    []string{"ppc64le"},
			Params:    []spec.Param{sl("b", spec.SliceU8), sl("chars", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("b"), base("chars"), lenOf("b"), lenOf("chars")},
			Threshold: thScan,
		},
		{
			// Also two lengths: the output holds two bytes per input byte, so
			// clamping them together would give the wrong count for either.
			// Base64. Two lengths for the same reason hex has two: the
			// output is four thirds of the input rounded up, which is not a
			// number the guard can clamp to.
			CName: "simd_b64_encode", GoName: "b64Encode",
			Group: "Bytes", Field: "B64Encode", RefFunc: "B64Encode",
			Params:    []spec.Param{sl("dst", spec.SliceU8), sl("b", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("dst"), base("b"), lenOf("dst"), lenOf("b")},
			Threshold: thBytes,
		},
		{
			CName: "simd_b64_decode", GoName: "b64Decode",
			Group: "Bytes", Field: "B64Decode", RefFunc: "B64Decode",
			Params:    []spec.Param{sl("dst", spec.SliceU8), sl("b", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("dst"), base("b"), lenOf("dst"), lenOf("b")},
			Threshold: thBytes,
		},
		{
			// Two results, the count and whether the input was valid hex,
			// which is the only reason this stayed portable everywhere until
			// the generator learned to return a pair. Six C arguments, which
			// is exactly the System V integer register budget, and the reason
			// the two lengths cannot become three.
			CName: "simd_hex_decode", GoName: "hexDecode",
			Group: "Bytes", Field: "HexDecode", RefFunc: "HexDecode",
			Params:  []spec.Param{sl("dst", spec.SliceU8), sl("src", spec.SliceU8)},
			Result:  &spec.Param{Name: "n", Type: spec.Int},
			Result2: &spec.Param{Name: "ok", Type: spec.B},
			CArgs: []spec.CArg{out(), out2(), base("dst"), base("src"),
				lenOf("dst"), lenOf("src")},
			Threshold: thBytes,
		},
		{
			CName: "simd_hex_encode", GoName: "hexEncode",
			Group: "Bytes", Field: "HexEncode", RefFunc: "HexEncode",
			Params:    []spec.Param{sl("dst", spec.SliceU8), sl("b", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("dst"), base("b"), lenOf("dst"), lenOf("b")},
			Threshold: thBytes,
		},

		{
			// Substring search. Two lengths, and the needle's is genuinely
			// independent of the haystack's, so no clamping.
			CName: "simd_index", GoName: "index",
			Group: "Bytes", Field: "Index", RefFunc: "Index",
			Params: []spec.Param{sl("haystack", spec.SliceU8),
				sl("needle", spec.SliceU8)},
			Result: &spec.Param{Name: "ret", Type: spec.Int},
			CArgs: []spec.CArg{out(), base("haystack"), base("needle"),
				lenOf("haystack"), lenOf("needle")},
			Threshold: thScan,
			// Measured crossover 256 on amd64, against the standard
			// library's Rabin-Karp: +495% at 4 KiB and +655% at 1 MiB.
			ThresholdOn: onStdlibAsm(256),
		},
		{
			// Backward substring search. Same two independent lengths.
			CName: "simd_last_index", GoName: "lastIndex",
			Group: "Bytes", Field: "LastIndex", RefFunc: "LastIndex",
			Params: []spec.Param{sl("haystack", spec.SliceU8),
				sl("needle", spec.SliceU8)},
			Result: &spec.Param{Name: "ret", Type: spec.Int},
			CArgs: []spec.CArg{out(), base("haystack"), base("needle"),
				lenOf("haystack"), lenOf("needle")},
			Threshold: thScan,
		},
		{
			// Non-overlapping occurrence count. RefWhen sends the empty
			// needle to the portable path, because bytes.Count answers that
			// one by counting runes and the kernel has no business knowing
			// about UTF-8.
			CName: "simd_count_seq", GoName: "countSeq",
			Group: "Bytes", Field: "CountSeq", RefFunc: "CountSeq",
			Params: []spec.Param{sl("haystack", spec.SliceU8),
				sl("needle", spec.SliceU8)},
			Result: &spec.Param{Name: "ret", Type: spec.Int},
			CArgs: []spec.CArg{out(), base("haystack"), base("needle"),
				lenOf("haystack"), lenOf("needle")},
			Threshold: thScan,
			// Same filter as Index and the same crossover.
			ThresholdOn: onStdlibAsm(256),
			RefWhen:     "len(needle) == 0",
		},

		byteMap("simd_to_upper_ascii", "toUpperASCII", "ToUpperASCII", "ToUpperASCII"),
		byteMap("simd_to_lower_ascii", "toLowerASCII", "ToLowerASCII", "ToLowerASCII"),
		{
			CName: "simd_replace_byte", GoName: "replaceByte",
			Group: "Bytes", Field: "ReplaceByte", RefFunc: "ReplaceByte",
			Params: []spec.Param{sl("dst", spec.SliceU8), sl("b", spec.SliceU8),
				sl("old", spec.U8), sl("with", spec.U8)},
			CArgs: []spec.CArg{base("dst"), base("b"), val("old"), val("with"),
				lenOf("dst")},
			Threshold: thBytes,
		},
	}
}

// ---------- transcendentals ----------

// Transcendentals do more arithmetic per element than anything else here — a
// double-precision exp is a Cody-Waite reduction and a degree-12 polynomial —
// so the call is repaid at a much shorter length than an add is. The portable
// path is a call into math per element, which is not close.
const thMath = 4

// mathUnary is a transcendental of one argument. It shares the shape of
// `unary` but takes the C name unchanged, because these are already spelled
// the way Go spells them.
func mathUnary(op, field string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op + e.goName,
		Group: e.group, Field: field, RefFunc: field,
		Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst")},
		Threshold: thMath,
	}
}

func mathBinary(op, field string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op + e.goName,
		Group: e.group, Field: field, RefFunc: field,
		Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice), sl("b", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), lenOf("dst")},
		Threshold: thMath,
	}
}

// Math is everything in csrc/math.c.
func Math() []spec.Kernel {
	unaries := []struct{ op, field string }{
		{"exp", "Exp"}, {"exp2", "Exp2"}, {"expm1", "Expm1"},
		{"log", "Log"}, {"log2", "Log2"}, {"log10", "Log10"},
		{"log1p", "Log1p"}, {"cbrt", "Cbrt"}, {"sigmoid", "Sigmoid"},
		{"sin", "Sin"}, {"cos", "Cos"}, {"tan", "Tan"},
		{"asin", "Asin"}, {"acos", "Acos"}, {"atan", "Atan"},
		{"sinh", "Sinh"}, {"cosh", "Cosh"}, {"tanh", "Tanh"},
	}
	binaries := []struct{ op, field string }{
		{"pow", "Pow"}, {"atan2", "Atan2"}, {"hypot", "Hypot"},
	}
	var ks []spec.Kernel
	for _, e := range floats() {
		for _, u := range unaries {
			ks = append(ks, mathUnary(u.op, u.field, e))
		}
		for _, b := range binaries {
			ks = append(ks, mathBinary(b.op, b.field, e))
		}
	}
	return ks
}

// FastMath is csrc/fastmath.c: the same transcendentals at 3.5 ULP instead of
// 1.0, from the same source compiled with shorter polynomials and fused
// multiply-add.
//
// The RefFunc is the *accurate* reference. That is not a shortcut: the Fast
// contract is an upper bound on error, so an answer better than the bound
// satisfies it, and it means the sub-threshold path and any target that does
// not generate these kernels are both correct without a second reference to
// keep in step.
func FastMath() []spec.Kernel {
	var ks []spec.Kernel
	for _, k := range Math() {
		k.CName = "simd_fast_" + strings.TrimPrefix(k.CName, "simd_")
		k.GoName = "fast" + strings.ToUpper(k.GoName[:1]) + k.GoName[1:]
		k.Field = "Fast" + k.Field
		ks = append(ks, k)
	}
	return ks
}

// Convert is csrc/convert.c: the narrow floating-point storage formats.
//
// A float16 or bfloat16 crosses as a uint16, because Go has neither type. The
// C side takes an unsigned short, which is the same sixteen bits, so nothing
// is reinterpreted anywhere — the pointer is the pointer.
func Convert() []spec.Kernel {
	conv := func(cname, goName, field, refFunc string, dst, src spec.Type) spec.Kernel {
		return spec.Kernel{
			CName: cname, GoName: goName,
			Group: "Convert", Field: field, RefFunc: refFunc,
			Params:    []spec.Param{sl("dst", dst), sl("a", src)},
			CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst")},
			Threshold: thElementwise,
		}
	}
	return []spec.Kernel{
		conv("simd_bf16_to_f32", "bf16ToF32", "BF16ToF32", "BF16ToF32",
			spec.SliceF32, spec.SliceU16),
		conv("simd_f32_to_bf16", "f32ToBF16", "F32ToBF16", "F32ToBF16",
			spec.SliceU16, spec.SliceF32),
		conv("simd_f16_to_f32", "f16ToF32", "F16ToF32", "F16ToF32",
			spec.SliceF32, spec.SliceU16),
		conv("simd_f32_to_f16", "f32ToF16", "F32ToF16", "F32ToF16",
			spec.SliceU16, spec.SliceF32),
	}
}

// ---------- numeric kernels with an unusual shape ----------

// normK is the Euclidean length, a reduction like the others.
func normK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_norm_" + e.c, GoName: "norm" + e.goName,
		Group: e.group, Field: "Norm", RefFunc: "NormFloat",
		Params:    []spec.Param{sl("a", e.slice)},
		Result:    &spec.Param{Name: "ret", Type: e.scalar},
		CArgs:     []spec.CArg{out(), base("a"), lenOf("a")},
		Threshold: thReduction,
	}
}

// polyEvalK applies one polynomial to every element. The C takes all three
// lengths so that it clamps them itself; the guard cannot, because a kernel
// with more than one length is one that reconciles them.
func polyEvalK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_polyeval_" + e.c, GoName: "polyEval" + e.goName,
		Group: e.group, Field: "PolyEval", RefFunc: "PolyEval",
		Params: []spec.Param{sl("dst", e.slice), sl("x", e.slice),
			sl("coeffs", e.slice)},
		CArgs: []spec.CArg{base("dst"), base("x"), base("coeffs"),
			lenOf("dst"), lenOf("x"), lenOf("coeffs")},
		Threshold: thElementwise,
	}
}

func windowK(op, field string, e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_" + op + "_" + e.c, GoName: op + e.goName,
		Group: e.group, Field: field, RefFunc: field,
		Params: []spec.Param{sl("dst", e.slice), sl("sig", e.slice),
			sl("ker", e.slice)},
		CArgs: []spec.CArg{base("dst"), base("sig"), base("ker"),
			lenOf("dst"), lenOf("sig"), lenOf("ker")},
		Threshold: thElementwise,
	}
}

func tileK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_tile_" + e.c, GoName: "tile" + e.goName,
		Group: e.group, Field: "Tile", RefFunc: "Tile",
		Params:    []spec.Param{sl("dst", e.slice), sl("pattern", e.slice)},
		CArgs:     []spec.CArg{base("dst"), base("pattern"), lenOf("dst"), lenOf("pattern")},
		Threshold: thElementwise,
	}
}

func gatherK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_gather_" + e.c, GoName: "gather" + e.goName,
		Group: e.group, Field: "Gather", RefFunc: "Gather",
		Params: []spec.Param{sl("dst", e.slice), sl("src", e.slice),
			sl("idx", spec.SliceI32)},
		CArgs: []spec.CArg{base("dst"), base("src"), base("idx"),
			lenOf("dst"), lenOf("idx"), lenOf("src")},
		Threshold: thElementwise,
	}
}

// Numeric is everything in csrc/numeric.c.
func Numeric() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range floats() {
		ks = append(ks,
			normK(e),
			polyEvalK(e),
			windowK("convolve", "Convolve", e),
			windowK("correlate", "Correlate", e),
		)
	}
	for _, e := range elems {
		ks = append(ks, tileK(e), gatherK(e), scatterK(e))
	}
	for _, e := range floats() {
		ks = append(ks, movAvgK(e))
	}
	return ks
}

// scatterK is the indexed store. RefWhen sends the unequal-length case to the
// portable path, which keeps the C signature to five arguments — one under
// what the zSeries ABI passes in registers — while costing nothing in the
// case callers actually use.
func scatterK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_scatter_" + e.c, GoName: "scatter" + e.goName,
		Group: e.group, Field: "Scatter", RefFunc: "Scatter",
		Params: []spec.Param{sl("dst", e.slice), sl("idx", spec.SliceI32),
			sl("src", e.slice)},
		CArgs: []spec.CArg{base("dst"), base("idx"), base("src"),
			lenOf("dst"), lenOf("idx"), lenOf("src")},
		Threshold: thElementwise,
	}
}

func movAvgK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_movavg_" + e.c, GoName: "movingAverage" + e.goName,
		Group: e.group, Field: "MovingAverage", RefFunc: "MovingAverage",
		Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
			{Name: "width", Type: spec.Int}},
		CArgs: []spec.CArg{base("dst"), base("a"), lenOf("dst"), lenOf("a"),
			val("width")},
		Threshold: thElementwise,
	}
}

// Gemm is the matrix family: the blocked matrix multiply and the
// matrix-vector product built beside it.
//
// They are their own source file because the microkernel is the one place here
// where the register file, not the instruction set, sets the shape — the tile
// dimensions are chosen per target by #if, which nothing else in csrc needs.
// transposeK is the blocked matrix transpose. Two dimensions and no result,
// like the GEMM kernels, and the same guard: it does nothing if the slices are
// too short for the stated shape.
func transposeK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_transpose_" + e.c, GoName: "transpose" + e.goName,
		Group: e.group, Field: "Transpose", RefFunc: "Transpose",
		Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
			{Name: "m", Type: spec.Int}, {Name: "n", Type: spec.Int}},
		CArgs: []spec.CArg{base("dst"), base("a"), val("m"), val("n")},
		// Blocking only pays once the matrix is bigger than a block, and the
		// guard has to see both dimensions, so the threshold is on the
		// destination length rather than on either side.
		Threshold: 1024,
	}
}

func Gemm() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range floats() {
		ks = append(ks, matMulK(e), gemvK(e))
	}
	// Transpose is not arithmetic, so it applies to the integer types too.
	for _, e := range elems {
		switch e.c {
		case "f32", "f64", "i32", "i64":
			ks = append(ks, transposeK(e))
		}
	}
	return ks
}

// gemvK multiplies an m*k row-major matrix by a k-vector.
//
// Five arguments, so it stays inside the register-passed set on every ABI
// here, and the size checks live in RefWhen for the same reason matMulK's do.
func gemvK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_gemv_" + e.c, GoName: "gemv" + e.goName,
		Group: e.group, Field: "Gemv", RefFunc: e.ref("Gemv"),
		Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
			sl("x", e.slice), {Name: "m", Type: spec.Int},
			{Name: "k", Type: spec.Int}},
		CArgs: []spec.CArg{base("dst"), base("a"), base("x"),
			val("m"), val("k")},
		RefWhen: "m <= 0 || k <= 0 || len(dst) < m || len(a) < m*k || len(x) < k",
		// Every row is a reduction over k, and reductions are worth calling
		// into assembly at any length; the threshold that matters is on k,
		// which the guard cannot see.
		Threshold: 0,
	}
}

// matMulK multiplies two row-major matrices.
//
// The size checks live in RefWhen rather than in the kernel, which is what
// keeps the signature to six arguments: passing the three slice lengths as
// well would need nine, and System V has six integer argument registers.
// Sending a badly sized call to the portable path is right anyway — the
// reference returns without writing, and that is a caller error worth
// handling in Go rather than in a kernel.
func matMulK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_matmul_" + e.c, GoName: "matMul" + e.goName,
		Group: e.group, Field: "MatMul", RefFunc: "MatMul",
		Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
			sl("b", e.slice), {Name: "m", Type: spec.Int},
			{Name: "k", Type: spec.Int}, {Name: "n", Type: spec.Int}},
		CArgs: []spec.CArg{base("dst"), base("a"), base("b"),
			val("m"), val("k"), val("n")},
		RefWhen: "m <= 0 || k <= 0 || n <= 0 || len(dst) < m*n || " +
			"len(a) < m*k || len(b) < k*n",
		Threshold: 0,
	}
}

// ---------- complex ----------

// celem describes one complex type alongside its real component type.
type celem struct {
	c      string    // suffix in the C symbol: c64
	goName string    // suffix in the Go name: Complex64
	slice  spec.Type // []complex64
	rslice spec.Type // []float32
	rscal  spec.Type // float32
	cscal  spec.Type // complex64, for the reductions that return one
	group  string    // the kernel.Set group: C64
	suf    string    // the ref function suffix: 64
}

var celems = []celem{
	{"c64", "Complex64", spec.SliceC64, spec.SliceF32, spec.F32, spec.C64, "C64", "64"},
	{"c128", "Complex128", spec.SliceC128, spec.SliceF64, spec.F64, spec.C128, "C128", "128"},
}

// Complex is everything in csrc/complex.c.
//
// The length passed to C is the number of complex elements, not of
// components: the kernels index as 2i and 2i+1 and the componentwise ones
// double it themselves, so a slice length is exactly what they want.
func Complex() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range celems {
		bin := func(op, field, ref string) spec.Kernel {
			return spec.Kernel{
				CName: "simd_c" + op + "_" + e.c, GoName: "c" + op + e.goName,
				Group: e.group, Field: field, RefFunc: ref,
				Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
					sl("b", e.slice)},
				CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), lenOf("dst")},
				Threshold: thElementwise,
			}
		}
		un := func(op, field, ref string) spec.Kernel {
			return spec.Kernel{
				CName: "simd_c" + op + "_" + e.c, GoName: "c" + op + e.goName,
				Group: e.group, Field: field, RefFunc: ref,
				Params:    []spec.Param{sl("dst", e.slice), sl("a", e.slice)},
				CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst")},
				Threshold: thElementwise,
			}
		}
		// A complex in, a real out: the destination is the real slice, and
		// its length is the element count either way. These live in the Parts
		// group, which is the half of the split that names the real type.
		toRealParts := func(op, field, ref string) spec.Kernel {
			return spec.Kernel{
				CName: "simd_c" + op + "_" + e.c, GoName: "c" + op + e.goName,
				Group: e.group + "Parts", Field: field, RefFunc: ref,
				Params:    []spec.Param{sl("dst", e.rslice), sl("a", e.slice)},
				CArgs:     []spec.CArg{base("dst"), base("a"), lenOf("dst")},
				Threshold: thElementwise,
			}
		}
		ks = append(ks,
			bin("add", "Add", "CAdd"),
			bin("sub", "Sub", "CSub"),
			bin("mul", "Mul", "CMul"+e.suf),
			bin("div", "Div", "CDiv"+e.suf),
			un("neg", "Neg", "CNeg"),
			un("conj", "Conj", "CConj"+e.suf),
			toRealParts("abs", "Abs", "CAbs"+e.suf),
			toRealParts("real", "Real", "CReal"+e.suf),
			toRealParts("imag", "Imag", "CImag"+e.suf),
			spec.Kernel{
				CName: "simd_cscale_" + e.c, GoName: "cscale" + e.goName,
				Group: e.group + "Parts", Field: "Scale", RefFunc: "CScale" + e.suf,
				Params: []spec.Param{sl("dst", e.slice), sl("a", e.slice),
					{Name: "s", Type: e.rscal}},
				CArgs:     []spec.CArg{base("dst"), base("a"), val("s"), lenOf("dst")},
				Threshold: thElementwise,
			},
			// The three reductions. They return a complex value, which the
			// generator writes through a pointer to the result slot like any
			// other — a Go complex64 is two float32s laid out contiguously,
			// so a float* into that slot fills it correctly.
			spec.Kernel{
				CName: "simd_csum_" + e.c, GoName: "csum" + e.goName,
				Group: e.group, Field: "Sum", RefFunc: "CSum" + e.suf,
				Params:    []spec.Param{sl("a", e.slice)},
				Result:    &spec.Param{Name: "ret", Type: e.cscal},
				CArgs:     []spec.CArg{out(), base("a"), lenOf("a")},
				Threshold: thReduction,
			},
			spec.Kernel{
				CName: "simd_cdot_" + e.c, GoName: "cdot" + e.goName,
				Group: e.group, Field: "Dot", RefFunc: "CDot" + e.suf,
				Params:    []spec.Param{sl("a", e.slice), sl("b", e.slice)},
				Result:    &spec.Param{Name: "ret", Type: e.cscal},
				CArgs:     []spec.CArg{out(), base("a"), base("b"), lenOf("a")},
				Threshold: thReduction,
			},
			spec.Kernel{
				CName: "simd_cdotconj_" + e.c, GoName: "cdotconj" + e.goName,
				Group: e.group, Field: "DotConj", RefFunc: "CDotConj" + e.suf,
				Params:    []spec.Param{sl("a", e.slice), sl("b", e.slice)},
				Result:    &spec.Param{Name: "ret", Type: e.cscal},
				CArgs:     []spec.CArg{out(), base("a"), base("b"), lenOf("a")},
				Threshold: thReduction,
			},
			spec.Kernel{
				CName: "simd_cfromparts_" + e.c, GoName: "cfromParts" + e.goName,
				Group: e.group + "Parts", Field: "FromParts", RefFunc: "CFromParts" + e.suf,
				Params: []spec.Param{sl("dst", e.slice), sl("re", e.rslice),
					sl("im", e.rslice)},
				CArgs:     []spec.CArg{base("dst"), base("re"), base("im"), lenOf("dst")},
				Threshold: thElementwise,
			},
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

	// ExtraFlags are appended to the target's clang invocation for this file
	// alone, after the common ones, so they can override.
	//
	// One file uses it. csrc/fastmath.c is compiled with -ffp-contract=fast
	// where everything else is compiled with it off, and that difference is
	// the whole point of the Fast tier: fusing a multiply into an add halves
	// the instruction count of a Horner evaluation and is *more* accurate, but
	// it gives a different answer on a machine with an FMA than on one
	// without. Every other kernel here promises the same bits everywhere and
	// therefore cannot have it. A function named Fast has already given that
	// promise up, and this is what it buys.
	ExtraFlags []string
}

// naryK is one n-ary elementwise kernel: three or four sources combined into a
// destination in a single pass.
//
// The threshold is higher than the binary one because the thing being saved is
// memory traffic, and below the last level of cache there is no traffic to
// save — the repeated binary calls hit cache on their second and third passes
// and cost little more than this does. Measured, not assumed.
func naryK(op, field string, arity int, e elem) spec.Kernel {
	params := []spec.Param{sl("dst", e.slice), sl("a", e.slice), sl("b", e.slice),
		sl("c", e.slice)}
	cargs := []spec.CArg{base("dst"), base("a"), base("b"), base("c")}
	if arity == 4 {
		params = append(params, sl("d", e.slice))
		cargs = append(cargs, base("d"))
	}
	cargs = append(cargs, lenOf("dst"))
	name := fmt.Sprintf("simd_%s%d_%s", op, arity, e.c)
	return spec.Kernel{
		CName: name, GoName: fmt.Sprintf("%s%d%s", op, arity, e.goName),
		Group: e.group, Field: fmt.Sprintf("%s%d", field, arity),
		RefFunc:   fmt.Sprintf("%s%d", field, arity),
		Params:    params,
		CArgs:     cargs,
		Threshold: thNary,
	}
}

// Nary is the fixed-arity elementwise family.
//
// Arity stops at four because a five-source kernel needs seven pointer
// arguments plus a length and System V passes six integers in registers. The
// Go wrapper folds anything longer in groups of four.
func Nary() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range elems {
		for _, a := range []int{3, 4} {
			ks = append(ks, naryK("add", "Add", a, e), naryK("mul", "Mul", a, e))
		}
	}
	return ks
}

// Sort is the partition step of a quicksort. The recursion and the pivot
// choice live in Go; only this vectorizes, and only where compress exists.
func Sort() []spec.Kernel {
	var ks []spec.Kernel
	for _, e := range elems {
		switch e.group {
		case "F32", "F64", "I32", "I64":
		default:
			continue
		}
		ks = append(ks, spec.Kernel{
			CName: "simd_partition_" + e.c, GoName: "partition" + e.goName,
			Group: e.group, Field: "Partition", RefFunc: "Partition",
			Params: []spec.Param{sl("dst", e.slice), sl("src", e.slice),
				sl("pivot", e.scalar)},
			Result: &spec.Param{Name: "ret", Type: spec.Int},
			CArgs: []spec.CArg{out(), base("dst"), base("src"), val("pivot"),
				lenOf("src")},
			RefWhen: "len(dst) < len(src)",
			// The partition is the inner loop of a quicksort and is only
			// reached on ranges big enough to still be recursing.
			Threshold: 64,
			SkipOn:    noCompress,
		})
	}
	return ks
}

// noCompress is every target with no hardware compress instruction, which is
// all of them but two.
//
// This list is not a to-do. Compression is the one operation in the library
// that autovectorization cannot reach on principle rather than by omission:
// the store address depends on how many earlier lanes matched, which is a
// genuine loop-carried dependency, and the only thing that breaks it is an
// instruction that packs a masked vector in one step. AVX-512 has vpcompressd
// and SVE2 has compact. On everything else clang scalarizes the intrinsic back
// into a per-lane branch, which is slower than the plain C loop it came from,
// so a kernel built there would be a pessimization wearing a kernel's name.
//
// A skipped kernel is not registered and the portable path stands, which is
// the same arrangement every other partial backend uses.
//
// # riscv64 is on this list for a different reason, and it is not the stated one
//
// RVV 1.0 does have vcompress.vm, so the paragraph above is simply wrong about
// riscv64, and removing it from this list does produce kernels: five of them,
// registering nine slots, which would also unblock PartitionInto and with it
// Sort, Median and Quantile on that architecture. LLVM vectorizes csrc/compress.c
// for rv64gcv into thirty-one vector instructions without complaint.
//
// They corrupt memory. Under qemu at vlen=256 the suite dies in
// runtime.scanstack during a garbage collection — a SIGSEGV inside the stack
// unwinder, which means the kernel wrote outside its frame and the collector
// found the damage later, somewhere unrelated to the cause.
//
// That is the same signature as countAnyVSX on ppc64le, which is skipped for
// the same reason and whose root cause is also unknown. Two instances of one
// fault in two unrelated backends points at the generator rather than at
// either kernel, and until that is understood shipping a kernel that passes
// its own tests and then breaks the collector is worse than shipping no kernel
// at all.
var noCompress = []string{
	"amd64/sse2", "amd64/avx2", "arm64/neon",
	"riscv64", "s390x", "loong64", "ppc64le",
}

// compressK is the packing kernel for one element type.
//
// The contract is that dst has room for the whole of src. The kernel cannot
// bound-check per lane — the store is unconditional and it is the *pointer*
// that advances, which is the entire reason the instruction is fast — so a
// caller with a short destination takes the portable path instead, which is
// what RefWhen says. The guard's clamp then reduces to min(len(src),
// len(keep)), because len(dst) is already known not to be the smallest.
func compressK(e elem) spec.Kernel {
	return spec.Kernel{
		CName: "simd_compress_" + e.c, GoName: "compress" + e.goName,
		Group: e.group, Field: "Compress", RefFunc: "Compress" + e.goName,
		Params: []spec.Param{
			sl("dst", e.slice), sl("src", e.slice), sl("keep", spec.SliceB),
		},
		Result:  &spec.Param{Name: "ret", Type: spec.Int},
		CArgs:   []spec.CArg{out(), base("dst"), base("src"), base("keep"), lenOf("src")},
		RefWhen: "len(dst) < len(src)",
		// Measured on avx512, and higher than most thresholds here because
		// what it is being compared against is unusually good. The scalar
		// filter costs a branch per element, and at low match density that
		// branch is perfectly predicted, so the loop this replaces runs at
		// close to load-store speed and the kernel has to make back the call
		// boundary against it. The crossover moves with density: at 90%
		// matches the kernel is already 25% ahead at n=64, at 1% it is still
		// 9% behind at n=128. 192 is where the worst density turns, measured
		// at 1%, 25%, 50% and 90% — and by n=1M the same kernel is 3× to 15×
		// ahead, because the scalar branch stops being predictable long before
		// the vector version notices anything has changed.
		Threshold: 192,
		SkipOn:    noCompress,
	}
}

// Compress is the compress family and the index scan built on it.
func Compress() []spec.Kernel {
	ks := []spec.Kernel{
		{
			// Two lengths, so the guard leaves them alone: dst holds one entry
			// per *match*, which is not a number that can be clamped against
			// the length of the haystack. The kernel reconciles them by
			// stopping when dst fills.
			CName: "simd_index_all", GoName: "indexAll",
			Group: "Bytes", Field: "IndexAll", RefFunc: "IndexAll",
			Params: []spec.Param{
				sl("dst", spec.SliceI32), sl("b", spec.SliceU8), sl("c", spec.U8),
			},
			Result: &spec.Param{Name: "ret", Type: spec.Int},
			CArgs: []spec.CArg{
				out(), base("dst"), base("b"), val("c"), lenOf("dst"), lenOf("b"),
			},
			Threshold: thScan,
			SkipOn:    noCompress,
		},
	}
	for _, e := range elems {
		// The four full-width types only. The narrow integers would need a
		// compress at their own lane width — vpcompressb is AVX512_VBMI2,
		// which is above the tier this library gates avx512 on — and packing
		// them through int32 lanes would cost more than it saves.
		switch e.group {
		case "F32", "F64", "I32", "I64":
			ks = append(ks, compressK(e))
		}
	}
	return ks
}

// All is every source file the generator processes.
var All = []Source{
	{Path: "csrc/compress.c", Kernels: Compress()},
	{Path: "csrc/gemm.c", Kernels: Gemm()},
	{Path: "csrc/nary.c", Kernels: Nary()},
	{Path: "csrc/sort.c", Kernels: Sort()},
	{Path: "csrc/argreduce.c", Kernels: ArgReduce()},
	{Path: "csrc/scan.c", Kernels: Scan()},
	{Path: "csrc/arith.c", Kernels: Arith()},
	{Path: "csrc/reduce.c", Kernels: Reduce()},
	{Path: "csrc/compare.c", Kernels: Compare()},
	{Path: "csrc/bytes.c", Kernels: Bytes()},
	{Path: "csrc/math.c", Kernels: Math()},
	{Path: "csrc/numeric.c", Kernels: Numeric()},
	{Path: "csrc/complex.c", Kernels: Complex()},
	{Path: "csrc/convert.c", Kernels: Convert()},
	{
		Path:    "csrc/fastmath.c",
		Kernels: FastMath(),
		// The one file compiled with contraction on; see the comment on
		// ExtraFlags and the one at the top of csrc/fastmath.c.
		ExtraFlags: []string{"-ffp-contract=fast"},
	},
}
