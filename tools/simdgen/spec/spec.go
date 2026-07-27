// Package spec describes the kernels to generate.
//
// Every kernel is declared here explicitly rather than discovered by parsing
// the C source. That is a deliberate departure from the existing tools in this
// space — gorse-io/goat walks clang's JSON AST, and consequently breaks on
// `static inline` helpers, single-line `if (x) { y; }`, union type-punning and
// array initializers with variable elements. Its README calls its own output
// "potentially BUGGY".
//
// A declaration costs three lines and removes an entire class of failure. It
// also lets the Go signature and the C signature differ, which they must: Go
// passes a slice as three words (pointer, length, capacity) while C wants a
// pointer and a separate length, so the mapping between them has to be stated
// somewhere. Stating it here means the generated ABI prologue and the
// generated Go declaration come from one source and cannot disagree.
package spec

import "fmt"

// Type is the type of a Go parameter or result.
type Type int

const (
	Invalid Type = iota

	SliceF32  // []float32
	SliceF64  // []float64
	SliceI32  // []int32
	SliceI64  // []int64
	SliceU8   // []byte
	SliceB    // []bool
	SliceC64  // []complex64
	SliceC128 // []complex128

	// The narrow and unsigned integers. []byte doubles as []uint8 and byte as
	// uint8, since they are the same types in Go and the byte spelling was
	// here first.
	SliceI8  // []int8
	SliceI16 // []int16
	SliceU16 // []uint16
	SliceU32 // []uint32
	SliceU64 // []uint64

	F32  // float32
	F64  // float64
	I32  // int32
	I64  // int64
	U8   // byte
	Int  // int
	B    // bool
	C64  // complex64
	C128 // complex128

	I8  // int8
	I16 // int16
	U16 // uint16
	U32 // uint32
	U64 // uint64
)

var typeNames = map[Type]string{
	SliceF32: "[]float32", SliceF64: "[]float64",
	SliceI32: "[]int32", SliceI64: "[]int64",
	SliceU8: "[]byte", SliceB: "[]bool",
	SliceC64: "[]complex64", SliceC128: "[]complex128",
	SliceI8: "[]int8", SliceI16: "[]int16",
	SliceU16: "[]uint16", SliceU32: "[]uint32", SliceU64: "[]uint64",
	C64: "complex64", C128: "complex128",
	F32: "float32", F64: "float64", I32: "int32", I64: "int64",
	U8: "byte", Int: "int", B: "bool",
	I8: "int8", I16: "int16", U16: "uint16", U32: "uint32", U64: "uint64",
}

// GoString returns the Go spelling of the type.
func (t Type) GoString() string {
	if s, ok := typeNames[t]; ok {
		return s
	}
	return fmt.Sprintf("Type(%d)", int(t))
}

// IsSlice reports whether the type occupies three words in a Go argument
// frame rather than one.
func (t Type) IsSlice() bool {
	switch t {
	case SliceF32, SliceF64, SliceI32, SliceI64, SliceU8, SliceB,
		SliceC64, SliceC128,
		SliceI8, SliceI16, SliceU16, SliceU32, SliceU64:
		return true
	}
	return false
}

// IsFloat reports whether a scalar is passed in a floating-point register
// under the C calling convention. Slices are pointers and so never are.
func (t Type) IsFloat() bool { return t == F32 || t == F64 }

// Size is the width of the type in a Go argument frame, in bytes, on a 64-bit
// platform. Slices are pointer, length and capacity.
func (t Type) Size() int {
	if t.IsSlice() {
		return 24
	}
	switch t {
	case C128:
		// Two float64s. Passed in the frame like any other 16-byte value.
		return 16
	case C64:
		return 8
	case B, U8, I8:
		// A bool or byte occupies one byte but is aligned to its own size;
		// the frame layout rounds up between arguments.
		return 1
	case I16, U16:
		return 2
	case F32, I32, U32:
		return 4
	default:
		return 8
	}
}

// Align is the alignment of the type in a Go argument frame.
func (t Type) Align() int {
	if t.IsSlice() {
		return 8
	}
	switch t {
	case C64:
		// Two float32s, so aligned like a float32 rather than like its own
		// eight-byte size. Getting this wrong shifts every later argument.
		return 4
	case C128:
		return 8
	}
	return t.Size()
}

// Param is one Go parameter or result.
type Param struct {
	Name string
	Type Type
}

// Part selects which piece of a Go parameter becomes a C argument.
//
// A Go slice is three words, but the C kernel wants a bare pointer and,
// usually, a length. Part is how one Go parameter supplies both.
type Part int

const (
	// Base is the data pointer of a slice.
	Base Part = iota
	// Len is the length of a slice, passed as a signed word.
	Len
	// Cap is the capacity of a slice. Rarely useful; included for
	// completeness.
	Cap
	// Value is a scalar parameter passed as-is.
	Value
	// ResultAddr is the address of the Go result slot in the argument frame.
	//
	// A kernel that computes a scalar takes a pointer to write it through
	// rather than returning it in a register. The generator copies a compiled
	// body verbatim and cannot safely append a store after it, because LLVM
	// lays basic blocks out after the return instruction. Handing the kernel
	// somewhere to write removes the need to append anything.
	ResultAddr
)

// CArg is one argument of the C function, in C declaration order.
type CArg struct {
	// From names a Go parameter of the enclosing kernel.
	From string
	// Part selects which piece of that parameter to pass.
	Part Part
}

// Kernel is one function to generate.
type Kernel struct {
	// CName is the symbol name in the compiled object file.
	CName string
	// GoName is the name of the generated Go declaration. It is unexported
	// by convention: the exported API in the root package dispatches to it.
	GoName string
	// Doc is the comment placed on the generated Go declaration.
	Doc string

	// Params are the Go parameters, in declaration order.
	Params []Param
	// Result is the Go result, or nil for a function that returns nothing.
	Result *Param

	// CArgs maps Go parameters onto the C function's arguments, in C order.
	CArgs []CArg

	// SkipOn names targets this kernel cannot be built for, either as a whole
	// GOARCH ("loong64") or as one tier of one ("amd64/sse2").
	//
	// Not every target can express every operation without a libm call, and a
	// libm call is fatal: Plan 9 assembly has no procedure linkage table. The
	// granularity has to be per tier, not per architecture, because the
	// baseline x86-64 has no rounding instruction at all — floor, ceil, trunc
	// and round arrived with SSE4.1 — while the avx2 and avx512 tiers of the
	// same architecture have them.
	//
	// A skipped kernel is simply not registered, and the backend keeps the
	// portable implementation it was built from. That is what makes partial
	// backends safe: there is never a hole, only a slower path.
	SkipOn []string

	// RefFunc is the exported function in internal/ref that implements this
	// kernel portably. The generated threshold guard calls it directly rather
	// than through the kernel set, so the short-slice path costs no indirect
	// call; see the comment on those functions.
	RefFunc string

	// RefWhen is a Go boolean expression over the kernel's parameters that,
	// when true, forces the portable path regardless of length.
	//
	// It exists for the handful of kernels whose Go contract is not simply
	// "process min(len) elements". Equal is the example: it reports whether two
	// slices hold the same bytes *and* are the same length, so on a length
	// mismatch the answer is false without looking at any byte — while the C
	// kernel, handed one length, would compare the common prefix and say true.
	RefWhen string

	// Group and Field name the kernel.Ops slot this kernel fills, so the
	// generator can emit the registration that installs it: Group "F32" and
	// Field "Add" means the generated function is assigned to
	// kernel.Set.F32.Add.
	Group, Field string

	// Threshold is the element count below which the dispatcher should run
	// the portable Go implementation instead of calling into assembly.
	//
	// A Go-to-assembly call costs a fixed ~1.4ns and can never be inlined, so
	// below some size the call is more expensive than the arithmetic it
	// saves; viterin/vek loses to plain Go at n=4 for exactly this reason.
	// The value belongs to the kernel because it depends on both the
	// operation and the element type, and it must be measured rather than
	// guessed — see the benchmarks.
	Threshold int
}

// LenParam returns the name of the parameter whose length bounds the work,
// which is the first slice parameter passed to C as a Len.
func (k Kernel) LenParam() (string, bool) {
	for _, a := range k.CArgs {
		if a.Part == Len {
			return a.From, true
		}
	}
	return "", false
}

// Param looks up a Go parameter by name.
func (k Kernel) Param(name string) (Param, bool) {
	for _, p := range k.Params {
		if p.Name == name {
			return p, true
		}
	}
	return Param{}, false
}

// Validate reports whether the kernel is self-consistent. It is called before
// anything is compiled, so a typo in a manifest is a clear error rather than a
// mysterious mis-generated prologue.
// Skips reports whether this kernel is excluded from a target, given its
// GOARCH and tier.
func (k Kernel) Skips(arch, tier string) bool {
	for _, s := range k.SkipOn {
		if s == arch || s == arch+"/"+tier {
			return true
		}
	}
	return false
}

func (k Kernel) Validate() error {
	if k.CName == "" || k.GoName == "" {
		return fmt.Errorf("kernel %q/%q: both CName and GoName are required", k.CName, k.GoName)
	}
	seen := map[string]bool{}
	for _, p := range k.Params {
		if p.Type == Invalid {
			return fmt.Errorf("kernel %s: parameter %q has no type", k.GoName, p.Name)
		}
		if seen[p.Name] {
			return fmt.Errorf("kernel %s: duplicate parameter %q", k.GoName, p.Name)
		}
		seen[p.Name] = true
	}
	for i, a := range k.CArgs {
		if a.Part == ResultAddr {
			if k.Result == nil {
				return fmt.Errorf("kernel %s: C argument %d wants the result address "+
					"but the kernel has no result", k.GoName, i)
			}
			continue
		}
		p, ok := k.Param(a.From)
		if !ok {
			return fmt.Errorf("kernel %s: C argument %d refers to unknown parameter %q",
				k.GoName, i, a.From)
		}
		if a.Part != Value && !p.Type.IsSlice() {
			return fmt.Errorf("kernel %s: C argument %d takes the %v of %q, which is not a slice",
				k.GoName, i, a.Part, a.From)
		}
		if a.Part == Value && p.Type.IsSlice() {
			return fmt.Errorf("kernel %s: C argument %d passes slice %q by value",
				k.GoName, i, a.From)
		}
	}
	if k.RefFunc == "" {
		return fmt.Errorf("kernel %s: RefFunc is required for the threshold guard", k.GoName)
	}
	if k.Group == "" || k.Field == "" {
		return fmt.Errorf("kernel %s: Group and Field are required to register it", k.GoName)
	}
	if k.Result != nil && k.Result.Type.IsSlice() {
		return fmt.Errorf("kernel %s: a slice result is not supported", k.GoName)
	}
	return nil
}

func (p Part) String() string {
	switch p {
	case Base:
		return "base pointer"
	case Len:
		return "length"
	case Cap:
		return "capacity"
	case Value:
		return "value"
	case ResultAddr:
		return "result address"
	}
	return "?"
}
