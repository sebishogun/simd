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
	case "addsat":
		return "satAdd"
	case "subsat":
		return "satSub"
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
		byteScan("simd_count_byte", "countByte", "Count", "CountByte", spec.Int, true),
		byteScan("simd_index_byte", "indexByte", "IndexByte", "IndexByte", spec.Int, true),
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
			CName: "simd_count_any", GoName: "countAny",
			Group: "Bytes", Field: "CountAny", RefFunc: "CountAny",
			Params:    []spec.Param{sl("b", spec.SliceU8), sl("chars", spec.SliceU8)},
			Result:    &spec.Param{Name: "ret", Type: spec.Int},
			CArgs:     []spec.CArg{out(), base("b"), base("chars"), lenOf("b"), lenOf("chars")},
			Threshold: thScan,
		},
		{
			// Also two lengths: the output holds two bytes per input byte, so
			// clamping them together would give the wrong count for either.
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
		ks = append(ks, movAvgK(e), matMulK(e))
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
	group  string    // the kernel.Set group: C64
	suf    string    // the ref function suffix: 64
}

var celems = []celem{
	{"c64", "Complex64", spec.SliceC64, spec.SliceF32, spec.F32, "C64", "64"},
	{"c128", "Complex128", spec.SliceC128, spec.SliceF64, spec.F64, "C128", "128"},
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
}

// All is every source file the generator processes.
var All = []Source{
	{Path: "csrc/arith.c", Kernels: Arith()},
	{Path: "csrc/reduce.c", Kernels: Reduce()},
	{Path: "csrc/compare.c", Kernels: Compare()},
	{Path: "csrc/bytes.c", Kernels: Bytes()},
	{Path: "csrc/math.c", Kernels: Math()},
	{Path: "csrc/numeric.c", Kernels: Numeric()},
	{Path: "csrc/complex.c", Kernels: Complex()},
}
