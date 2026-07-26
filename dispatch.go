package simd

import (
	"github.com/sebishogun/simd/internal/cpu"
	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// active is the backend every exported function calls through. It is assigned
// once, during package initialization, and never written again, so reads need
// no synchronization.
//
// Backends are swapped as a whole rather than per function. That keeps a tier
// internally consistent, and it lets the differential tests exercise every
// instruction set the host supports inside one process instead of re-running
// the suite once per GOSIMD setting.
var active kernel.Set

// Per-type pointers into active, so ops can hand one back without copying a
// struct full of function pointers on every call.
var (
	opsF32 *kernel.Ops[float32]
	opsF64 *kernel.Ops[float64]
	opsI32 *kernel.Ops[int32]
	opsI64 *kernel.Ops[int64]
)

func init() {
	active = backendFor(cpu.Selected())
	opsF32, opsF64 = &active.F32, &active.F64
	opsI32, opsI64 = &active.I32, &active.I64
}

// ops returns the kernel group for element type T.
//
// Go cannot dispatch a generic call to a type-specific function pointer
// without a type switch somewhere. Putting the only one here, rather than
// repeating it in every exported function, keeps each of those a single line
// and means the mapping from element type to backend exists in exactly one
// place. When T is concrete the switch folds away.
func ops[T Number]() *kernel.Ops[T] {
	var zero T
	switch any(zero).(type) {
	case float32:
		return any(opsF32).(*kernel.Ops[T])
	case float64:
		return any(opsF64).(*kernel.Ops[T])
	case int32:
		return any(opsI32).(*kernel.Ops[T])
	case int64:
		return any(opsI64).(*kernel.Ops[T])
	}
	panic("simd: unsupported element type")
}

// backendFor returns the kernel set for a tier, falling back to the portable
// Go reference for any tier without a backend compiled into this build.
//
// The fallback is not a failure path: on an architecture with no generated
// assembly, and in builds made with the purego tag, it is the only path.
func backendFor(t cpu.Tier) kernel.Set {
	if puregoOnly {
		return ref.Set()
	}
	if s, ok := tierBackends[t]; ok {
		return s
	}
	return ref.Set()
}

// tierBackends maps tiers to their generated backends. Generated code
// registers into it from an init function in the per-architecture package, so
// this file needs no build-tagged imports.
//
// It is empty until the code-generation pipeline lands, so every tier
// currently resolves to the reference implementation: correct, but not fast.
var tierBackends = map[cpu.Tier]kernel.Set{}

// Tier returns the name of the instruction-set tier this process selected,
// such as "avx2", "neon" or "scalar".
func Tier() string { return cpu.Selected().String() }

// AvailableTiers returns the names of every instruction-set tier this CPU can
// execute, weakest first, always beginning with "scalar".
//
// Use it to drive a benchmark or test matrix over the tiers a machine actually
// supports, rather than a hardcoded list that would be wrong elsewhere:
//
//	for _, t := range simd.AvailableTiers() {
//	    exec.Command(...).Env = append(os.Environ(), "GOSIMD="+t)
//	}
//
// Tiers masked out by SIMD_DISABLE are excluded.
func AvailableTiers() []string {
	avail := cpu.Detail().Available
	names := make([]string, len(avail))
	for i, t := range avail {
		names[i] = t.String()
	}
	return names
}

// Describe returns a one-line summary of instruction-set selection: the
// architecture, the chosen tier, every tier the CPU supports, and whether
// GOSIMD or SIMD_DISABLE altered the outcome.
//
// It is intended for logs and bug reports.
func Describe() string { return cpu.Describe() }
