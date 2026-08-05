package simd

// KernelThreshold reports the input length below which the named operation's
// generated guard takes the portable reference implementation instead of a
// vector kernel, on the architecture this binary was built for. The name is
// the exported operation name: KernelThreshold("JSONCopyRun"),
// KernelThreshold("IndexByte").
//
// The number is in the operation's own length units — elements for typed
// slices, bytes for byte and text operations — matching the guard, which
// compares against the same length the operation is given.
//
// It exists because the number was previously knowable only by reading
// generated guard code, and callers in other repositories derived their own
// copies of it from memory. A caller that batches work, or that chooses
// between calling a kernel and doing something else entirely, can ask instead
// of remembering: a call below this threshold is not wrong, it is simply a
// portable byte loop wearing the kernel's name.
//
// Unknown names report 0, which is also the value for operations whose kernel
// is worth calling at any length. Under the purego build tag every operation
// is the portable implementation regardless of length; the table still
// reports what the tagged build's guard would do.
//
// The table is generated from the same manifest the guards are generated
// from, and a test holds the two to agreement.
func KernelThreshold(op string) int {
	return kernelThresholds[op]
}
