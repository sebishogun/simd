package simd

import (
	"reflect"
	"sync"

	"github.com/sebishogun/simd/internal/cpu"
	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// Dispatch is a tier index into per-operation tables, not a struct of
// function pointers. Every exported function reads its own static table --
// generated into dispatch_tables_<arch>.go -- at tierIdx. The tables are
// static composite literals of exported guards, which is what lets the
// linker drop every operation a program never calls, assembly included: a
// binary that uses three operations carries three operations' kernels. The
// old form, one Set holding a function value for everything, made every
// kernel reachable from init and therefore alive in every binary.
//
// tierIdx is assigned once, during package initialization, and never written
// again, so reads need no synchronization.
var tierIdx int

// refBase is the complete portable reference set: the fallback the numeric
// overlay starts from, and the direct target for the few operations with no
// generated kernel anywhere. The reference is a third of a megabyte of
// ordinary Go and is linked into every consumer regardless; the six
// megabytes of per-tier assembly are what the tables keep out.
var refBase = func() kernel.Set {
	s := ref.Set()
	ref.FillFastFallbacks(&s)
	return s
}()

func init() {
	tierIdx = tierIndexFor(cpu.Detail())
	cpu.SetBackendTier(dispatchTiers[tierIdx])
}

// tierIndexFor picks the strongest tier this build has tables for among the
// ones the CPU supports -- the same walk the Set registry used to do.
// A machine with AVX-512 whose build carries no AVX-512 kernels gets AVX2,
// not portable Go. When GOSIMD has pinned a tier, that exact tier is used or
// the reference is: substituting a neighbour would make a pinned benchmark
// meaningless.
func tierIndexFor(sel cpu.Selection) int {
	if puregoOnly {
		return 0
	}
	if sel.Forced {
		return tierIndexOf(sel.Tier.String())
	}
	for i := len(sel.Available) - 1; i >= 0; i-- {
		if sel.Available[i] > sel.Tier {
			continue
		}
		if idx := tierIndexOf(sel.Available[i].String()); idx > 0 {
			return idx
		}
	}
	return 0
}

func tierIndexOf(name string) int {
	for i, n := range dispatchTiers {
		if n == name {
			return i
		}
	}
	return 0
}

// opsCache is the per-element-type merge of the reference base with a tier's
// generated partial Ops, built once on first use. The partials are static --
// only the fields with kernels on that tier are set -- because their
// reference fallbacks come from generic constructors whose fields cannot be
// named in a static literal. Merging lazily keeps the liveness scoped: the
// float32 partials are referenced only from the float32 instantiation of
// ops, so a program that never touches float32 never links its kernels.
type opsCache[T Number] struct {
	once  sync.Once
	tiers [len(dispatchTiers)]*kernel.Ops[T]
}

func (c *opsCache[T]) get(base *kernel.Ops[T], partial []*kernel.Ops[T]) *kernel.Ops[T] {
	c.once.Do(func() {
		for i := range c.tiers {
			s := *base
			if i < len(partial) && partial[i] != nil {
				overlay(&s, partial[i])
			}
			c.tiers[i] = &s
		}
	})
	return c.tiers[tierIdx]
}

// overlay copies every non-nil function field of src over dst.
func overlay[T Number](dst, src *kernel.Ops[T]) {
	d := reflect.ValueOf(dst).Elem()
	s := reflect.ValueOf(src).Elem()
	for i := 0; i < d.NumField(); i++ {
		f := s.Field(i)
		if f.Kind() == reflect.Func && !f.IsNil() {
			d.Field(i).Set(f)
		}
	}
}

var (
	cacheF32 opsCache[float32]
	cacheF64 opsCache[float64]
	cacheI8  opsCache[int8]
	cacheI16 opsCache[int16]
	cacheI32 opsCache[int32]
	cacheI64 opsCache[int64]
	cacheU8  opsCache[uint8]
	cacheU16 opsCache[uint16]
	cacheU32 opsCache[uint32]
	cacheU64 opsCache[uint64]
)

// ops returns the kernel group for element type T at the selected tier.
//
// Go cannot dispatch a generic call to a type-specific function pointer
// without a type switch somewhere. Putting the only one here, rather than
// repeating it in every exported function, keeps each of those a single line
// and means the mapping from element type to backend exists in exactly one
// place. When T is concrete the switch folds away, and with it every branch's
// references: the uint16 tables are not linked because a float32 caller's
// instantiation mentions them.
func ops[T Number]() *kernel.Ops[T] {
	var zero T
	switch any(zero).(type) {
	case float32:
		return any(cacheF32.get(&refBase.F32, opsF32ByTier[:])).(*kernel.Ops[T])
	case float64:
		return any(cacheF64.get(&refBase.F64, opsF64ByTier[:])).(*kernel.Ops[T])
	case int32:
		return any(cacheI32.get(&refBase.I32, opsI32ByTier[:])).(*kernel.Ops[T])
	case int64:
		return any(cacheI64.get(&refBase.I64, opsI64ByTier[:])).(*kernel.Ops[T])
	case int8:
		return any(cacheI8.get(&refBase.I8, opsI8ByTier[:])).(*kernel.Ops[T])
	case int16:
		return any(cacheI16.get(&refBase.I16, opsI16ByTier[:])).(*kernel.Ops[T])
	case uint8:
		return any(cacheU8.get(&refBase.U8, opsU8ByTier[:])).(*kernel.Ops[T])
	case uint16:
		return any(cacheU16.get(&refBase.U16, opsU16ByTier[:])).(*kernel.Ops[T])
	case uint32:
		return any(cacheU32.get(&refBase.U32, opsU32ByTier[:])).(*kernel.Ops[T])
	case uint64:
		return any(cacheU64.get(&refBase.U64, opsU64ByTier[:])).(*kernel.Ops[T])
	}
	panic("simd: unsupported element type")
}

// Tier returns the name of the instruction-set tier whose kernels this
// process is actually running, such as "avx2", "neon" or "scalar".
//
// This is the backend in use, which is not always the best tier the CPU
// supports: if no kernels have been generated for that tier yet, the next one
// down is used instead. [Describe] reports both.
func Tier() string { return dispatchTiers[tierIdx] }

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
