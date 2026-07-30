// Command simdinfo reports which instruction-set tier the simd package
// selected on this machine, and which ones it could have used.
//
//	$ simdinfo
//	amd64 tier=avx512 available=[scalar sse2 avx2 avx512]
//
//	$ GOSIMD=sse2 simdinfo
//	amd64 tier=sse2 available=[scalar sse2 avx2 avx512] forced
//
//	$ SIMD_DISABLE=avx512 simdinfo
//	amd64 tier=avx2 available=[scalar sse2 avx2] disabled=[avx512]
//
// It is the first thing to run when a benchmark or a numerical result looks
// wrong, and `simdinfo -tiers` drives the per-tier test matrix in the Makefile.
//
// `simdinfo -require-accelerated` exits non-zero if the portable path was
// selected, which is what every emulated CI lane asserts before running the
// suite. That check exists because its absence cost two backends: the riscv64
// and loong64 lanes were green for months while executing nothing at all,
// because the emulator in the image predated the vector extension and reported
// a CPU that had none. A suite that skips every accelerated tier passes, and
// looks exactly like one that tested them.
//
// `simdinfo -argv0-probe` exits 7 and prints nothing. It exists because the
// paragraph above turned out to describe -require-accelerated itself: the
// qemu-user binaries this repo uses follow the binfmt_misc "preserve argv[0]"
// calling convention, `qemu <path> <argv0> <args...>`, so they consume the
// first argument as argv[0] and every flag after it shifts one place left.
// Under that convention `qemu simdinfo -require-accelerated` passes the guest
// an empty argument list, the assertion is never evaluated, and the lane exits
// 0 whatever the tier. A flag that must change the exit code is the only way
// to notice, since dropped flags are otherwise indistinguishable from flags
// that had nothing to say.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/sebishogun/simd"
)

// argv0ProbeExit is the exit code of -argv0-probe. Any value works as long as
// it is neither 0 nor 1, so that it cannot be confused with either a normal
// run or a failed assertion.
const argv0ProbeExit = 7

func main() {
	tiers := flag.Bool("tiers", false, "print every available tier, one per line, instead of the summary")
	require := flag.Bool("require-accelerated", false,
		"exit non-zero if the selected tier is the portable path")
	probe := flag.Bool("argv0-probe", false,
		"exit 7 immediately, to prove flags survived the emulator's argv handling")
	flag.Parse()

	if *probe {
		os.Exit(argv0ProbeExit)
	}

	if *tiers {
		for _, t := range simd.AvailableTiers() {
			fmt.Println(t)
		}
		return
	}
	fmt.Println(simd.Describe())
	if *require && simd.Tier() == "scalar" {
		fmt.Fprintln(os.Stderr,
			"simdinfo: the portable path was selected, so nothing accelerated would be tested.\n"+
				"On an emulator this usually means the emulated CPU has no vector unit:\n"+
				"QEMU 8.1 is the first with LoongArch LSX/LASX, and RISC-V needs an\n"+
				"explicit -cpu with v=true. See the qemu lanes in the Makefile.")
		os.Exit(1)
	}
}
