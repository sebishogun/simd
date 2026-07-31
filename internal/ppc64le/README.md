# internal/ppc64le — generated assembly

Plan 9 assembly for POWER, covering the vsx tier(s), compiled from
[`csrc/`](../../csrc) by [`tools/simdgen`](../../tools) and committed so that
using this library needs no C toolchain.

**Do not edit these by hand.** Every `.s` file opens with the generator that
produced it, the C source it came from, and the target it was built for. To
change one, edit the C and run `make codegen`.

The `.go` files declare the assembly symbols and register them with the
dispatch table in [`internal/kernel`](../kernel). A kernel that could not be
generated for this target is simply not registered, and
[`internal/ref`](../ref) runs instead — slower, never wrong.
