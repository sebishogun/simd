package simd

import (
	"github.com/sebishogun/simd/internal/backend"
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
	active = backendFor(cpu.Detail())
	cpu.SetBackendTier(active.Name)
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

// backendFor picks the best backend this build actually contains for a CPU.
//
// It walks the tiers the CPU supports from strongest to weakest and takes the
// first one with a registered backend, rather than looking up only the
// strongest and giving up. Those are not the same thing: a machine with
// AVX-512 selects the avx512 tier, but if no AVX-512 kernels have been
// generated yet, the right answer is the AVX2 backend — not to fall all the
// way back to portable Go and leave a 9x speedup on the table because the
// best possible tier happened to be missing.
//
// Falling through to the reference is still correct and still happens: on an
// architecture with no generated assembly at all, and in purego builds, it is
// the only path.
func backendFor(sel cpu.Selection) kernel.Set {
	if puregoOnly {
		return ref.Set()
	}
	// Available is ordered weakest first, so walk it backwards. When GOSIMD
	// has pinned a tier, honour that exactly rather than stepping down past
	// it: forcing a tier is how the tests and benchmarks isolate one, and
	// silently substituting a different one would make both meaningless.
	if sel.Forced {
		if s, ok := backend.Lookup(sel.Tier.String()); ok {
			return s
		}
		return ref.Set()
	}
	for i := len(sel.Available) - 1; i >= 0; i-- {
		if sel.Available[i] > sel.Tier {
			continue
		}
		if s, ok := backend.Lookup(sel.Available[i].String()); ok {
			return s
		}
	}
	return ref.Set()
}

// Tier returns the name of the instruction-set tier whose kernels this
// process is actually running, such as "avx2", "neon" or "scalar".
//
// This is the backend in use, which is not always the best tier the CPU
// supports: if no kernels have been generated for that tier yet, the next one
// down is used instead. [Describe] reports both.
func Tier() string { return active.Name }

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
