// Package kernels is the manifest: every kernel the generator builds, with
// the mapping between its Go signature and its C signature.
//
// Declaring these rather than parsing the C removes the failure mode that
// makes gorse-io/goat unreliable, and it is also the only place the two
// signatures can be reconciled — Go passes a slice as three words while C
// wants a pointer and a length, so something has to say which Go parameter
// supplies the length.
package kernels

import "github.com/sebishogun/simd/tools/simdgen/spec"

func sl(name string, t spec.Type) spec.Param { return spec.Param{Name: name, Type: t} }

func base(n string) spec.CArg   { return spec.CArg{From: n, Part: spec.Base} }
func length(n string) spec.CArg { return spec.CArg{From: n, Part: spec.Len} }
func val(n string) spec.CArg    { return spec.CArg{From: n, Part: spec.Value} }

// out passes the address of the Go result slot, so the kernel can write its
// answer there instead of returning it in a register.
func out() spec.CArg { return spec.CArg{Part: spec.ResultAddr} }

// binary builds a three-slice elementwise kernel: dst = a op b.
//
// The length passed to C is dst's, because every operation in this library
// processes the minimum of its slice lengths and the caller-side wrapper has
// already narrowed all three to that minimum.
// group maps a slice element type to the kernel.Ops group that holds it.
func group(t spec.Type) string {
	switch t {
	case spec.SliceF32:
		return "F32"
	case spec.SliceF64:
		return "F64"
	case spec.SliceI32:
		return "I32"
	case spec.SliceI64:
		return "I64"
	}
	panic("kernels: no Ops group for " + t.GoString())
}

func binary(cName, goName, field string, t spec.Type, threshold int) spec.Kernel {
	return spec.Kernel{
		CName: cName, GoName: goName, Group: group(t), Field: field,
		Params:    []spec.Param{sl("dst", t), sl("a", t), sl("b", t)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), length("dst")},
		Threshold: threshold,
	}
}

// scaled builds dst = a * s or dst = a + b*s.
func scaleK(cName, goName, field string, slice, scalar spec.Type, threshold int) spec.Kernel {
	return spec.Kernel{
		CName: cName, GoName: goName, Group: group(slice), Field: field,
		Params:    []spec.Param{sl("dst", slice), sl("a", slice), sl("s", scalar)},
		CArgs:     []spec.CArg{base("dst"), base("a"), val("s"), length("dst")},
		Threshold: threshold,
	}
}

func axpy(cName, goName, field string, slice, scalar spec.Type, threshold int) spec.Kernel {
	return spec.Kernel{
		CName: cName, GoName: goName, Group: group(slice), Field: field,
		Params:    []spec.Param{sl("dst", slice), sl("a", slice), sl("b", slice), sl("s", scalar)},
		CArgs:     []spec.CArg{base("dst"), base("a"), base("b"), val("s"), length("dst")},
		Threshold: threshold,
	}
}

// reduce1 builds a one-slice reduction returning a scalar.
func reduce1(cName, goName, field string, slice, scalar spec.Type, threshold int) spec.Kernel {
	return spec.Kernel{
		CName: cName, GoName: goName, Group: group(slice), Field: field,
		Params:    []spec.Param{sl("a", slice)},
		Result:    &spec.Param{Name: "ret", Type: scalar},
		CArgs:     []spec.CArg{out(), base("a"), length("a")},
		Threshold: threshold,
	}
}

// reduce2 builds a two-slice reduction returning a scalar.
func reduce2(cName, goName, field string, slice, scalar spec.Type, threshold int) spec.Kernel {
	return spec.Kernel{
		CName: cName, GoName: goName, Group: group(slice), Field: field,
		Params:    []spec.Param{sl("a", slice), sl("b", slice)},
		Result:    &spec.Param{Name: "ret", Type: scalar},
		CArgs:     []spec.CArg{out(), base("a"), base("b"), length("a")},
		Threshold: threshold,
	}
}

// Elementwise is the manifest for csrc/elementwise.c.
//
// The thresholds are placeholders until the benchmarks fix them. They are
// deliberately not zero: a Go-to-assembly call costs a fixed ~1.4ns and can
// never be inlined, so a kernel called on four elements loses to a plain Go
// loop — which is exactly what viterin/vek#6 reports. 32 is a conservative
// starting point, roughly where the published crossover measurements sit.
var Elementwise = []spec.Kernel{
	binary("simd_add_f32", "addFloat32", "Add", spec.SliceF32, 32),
	binary("simd_add_f64", "addFloat64", "Add", spec.SliceF64, 32),
	binary("simd_sub_f32", "subFloat32", "Sub", spec.SliceF32, 32),
	binary("simd_sub_f64", "subFloat64", "Sub", spec.SliceF64, 32),
	binary("simd_mul_f32", "mulFloat32", "Mul", spec.SliceF32, 32),
	binary("simd_mul_f64", "mulFloat64", "Mul", spec.SliceF64, 32),
	binary("simd_add_i32", "addInt32", "Add", spec.SliceI32, 32),
	binary("simd_add_i64", "addInt64", "Add", spec.SliceI64, 32),

	scaleK("simd_scale_f32", "scaleFloat32", "Scale", spec.SliceF32, spec.F32, 32),
	scaleK("simd_scale_f64", "scaleFloat64", "Scale", spec.SliceF64, spec.F64, 32),

	axpy("simd_addscaled_f32", "addScaledFloat32", "AddScaled", spec.SliceF32, spec.F32, 32),
	axpy("simd_addscaled_f64", "addScaledFloat64", "AddScaled", spec.SliceF64, spec.F64, 32),

	reduce1("simd_sum_f32", "sumFloat32", "Sum", spec.SliceF32, spec.F32, 32),
	reduce1("simd_sum_f64", "sumFloat64", "Sum", spec.SliceF64, spec.F64, 32),
	reduce2("simd_dot_f32", "dotFloat32", "Dot", spec.SliceF32, spec.F32, 32),
	reduce2("simd_dot_f64", "dotFloat64", "Dot", spec.SliceF64, spec.F64, 32),
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
	{Path: "csrc/elementwise.c", Kernels: Elementwise},
}
