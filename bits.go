package simd

// Shifts and rotates.
//
// The count is a uint64 and the semantics are Go's, not C's, and that is the
// whole substance of this file.
//
// A shift by the operand's width or more is undefined in C, and the hardware
// disagrees about what it does: x86 masks the count to five or six bits, so
// shifting a uint32 by 32 returns it unchanged; arm64's shift instructions
// saturate to zero; and LLVM may fold the expression to poison. Three answers
// on three architectures is the bit-identity failure this package exists to
// prevent, and it would only ever appear for counts a test has to deliberately
// generate.
//
// Go defines it, and defines it well. These functions match Go's operators
// exactly, for every count including counts far above the width, and the
// kernels clamp explicitly to get there rather than trusting the target.

// ShlInto writes a << s to dst.
//
// A count at or above the element width gives zero, as in Go. The count is
// unsigned because Go panics on a negative shift and a kernel cannot panic.
func ShlInto[T Integer](dst, a []T, s uint64) { ops[T]().Shl(dst, a, s) }

// Shl shifts a left by s in place.
func Shl[T Integer](a []T, s uint64) { ops[T]().Shl(a, a, s) }

// ShrInto writes a >> s to dst.
//
// The shift is arithmetic for the signed types and logical for the unsigned
// ones, as in Go. So a count at or above the width gives zero, except for a
// negative signed value, which gives -1 — sign extension taken to its limit.
func ShrInto[T Integer](dst, a []T, s uint64) { ops[T]().Shr(dst, a, s) }

// Shr shifts a right by s in place.
func Shr[T Integer](a []T, s uint64) { ops[T]().Shr(a, a, s) }

// RotlInto writes each element of a rotated left by s bits to dst.
//
// A rotate has no undefined case: the count is reduced modulo the element
// width, so every count is meaningful and none needs clamping.
func RotlInto[T Integer](dst, a []T, s uint64) { ops[T]().Rotl(dst, a, s) }

// Rotl rotates a left by s bits in place.
func Rotl[T Integer](a []T, s uint64) { ops[T]().Rotl(a, a, s) }

// RotrInto writes each element of a rotated right by s bits to dst.
func RotrInto[T Integer](dst, a []T, s uint64) { ops[T]().Rotr(dst, a, s) }

// Rotr rotates a right by s bits in place.
func Rotr[T Integer](a []T, s uint64) { ops[T]().Rotr(a, a, s) }
