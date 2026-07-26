// Package backend is the registry that connects generated assembly to the
// dispatcher.
//
// The generated code lives in internal/<arch> and the dispatcher lives in the
// root package. Neither can reference the other directly — the root package
// would need build-tagged imports of every architecture, and the architecture
// packages would import the root package they are part of. A small registry
// both sides import breaks the cycle.
//
// Registration happens from an init function in the architecture package, so
// the only thing the root package needs is a blank import guarded by the same
// build tag as the generated files.
package backend

import (
	"github.com/sebishogun/simd/internal/kernel"
	"github.com/sebishogun/simd/internal/ref"
)

// registry maps a tier name, as spelled by internal/cpu, to its backend.
//
// It is written only from init functions and read only afterwards, so it needs
// no lock: Go guarantees all package initialization completes before main
// runs, and nothing here is reachable before then.
var registry = map[string]*kernel.Set{}

// For returns the backend being assembled for a tier, creating it from the
// portable reference on first use.
//
// Several generated files contribute to one tier — there is a C source per
// operation family, and each produces its own registration — so they add to a
// shared set rather than each installing a complete one. Starting from the
// reference is what makes that safe at any point: a tier is always a complete
// set of kernels, with the portable implementation standing in for whatever
// has not been generated for it.
func For(tier string) *kernel.Set {
	if s, ok := registry[tier]; ok {
		return s
	}
	s := ref.Set()
	s.Name = tier
	registry[tier] = &s
	return &s
}

// Lookup returns the backend for a tier, and whether one exists.
func Lookup(tier string) (kernel.Set, bool) {
	s, ok := registry[tier]
	if !ok {
		return kernel.Set{}, false
	}
	return *s, true
}

// Tiers returns every registered tier name. Order is unspecified.
func Tiers() []string {
	out := make([]string, 0, len(registry))
	for t := range registry {
		out = append(out, t)
	}
	return out
}

// Base returns a fresh copy of the portable reference set.
//
// Generated code starts from this and overrides only the kernels it actually
// has. That is what lets kernels land a few at a time: a backend is always
// complete, with the reference standing in for everything not yet written, so
// there is never a nil function to call. A partial backend that left holes
// would be a crash waiting for whichever operation nobody got to yet.
func Base() kernel.Set { return ref.Set() }
