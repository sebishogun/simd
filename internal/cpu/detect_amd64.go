//go:build amd64

package cpu

import "golang.org/x/sys/cpu"

// detect reports the tiers this CPU supports, weakest first.
//
// SSE2 is unconditional: it is part of the amd64 baseline, which is why the
// standard library and gonum use it with no runtime check at all.
//
// AVX2 requires FMA as well. Every kernel in the AVX2 tier is free to use
// fused multiply-add, so gating on AVX2 alone would emit VFMADD on a CPU
// without it. vek makes the same pairing.
//
// AVX512 means the F+CD+BW+DQ+VL bundle, matching what the Go standard library
// and simd/archsimd call "AVX512". Nearly every CPU shipping any AVX-512
// support has all five; splitting them buys nothing and multiplies the tier
// matrix.
func detect() []Tier {
	t := []Tier{Scalar, SSE2}
	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		t = append(t, AVX2)
	}
	if cpu.X86.HasAVX512F && cpu.X86.HasAVX512CD &&
		cpu.X86.HasAVX512BW && cpu.X86.HasAVX512DQ && cpu.X86.HasAVX512VL {
		t = append(t, AVX512)
	}
	return t
}
