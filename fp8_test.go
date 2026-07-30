package simd_test

import (
	"math"
	"testing"

	simd "github.com/sebishogun/simd"
)

// An 8-bit format has 256 encodings, so nothing here needs to be sampled.
// These tests are total.

func TestFloat8E4M3Exhaustive(t *testing.T) {
	all := make([]byte, 256)
	for i := range all {
		all[i] = byte(i)
	}
	wide := make([]float32, 256)
	simd.Float8E4M3ToFloat32Into(wide, all)

	// e4m3 has no infinity: every encoding is finite except the two NaNs.
	for i, v := range wide {
		isNaN := byte(i)&0x7f == 0x7f
		switch {
		case isNaN:
			if !math.IsNaN(float64(v)) {
				t.Errorf("e4m3 %#02x = %v, want NaN", i, v)
			}
		case math.IsInf(float64(v), 0):
			t.Errorf("e4m3 %#02x = %v, but e4m3 has no infinity", i, v)
		case math.IsNaN(float64(v)):
			t.Errorf("e4m3 %#02x = NaN, but only 0x7f and 0xff are NaN", i)
		}
	}

	// The documented extremes.
	if got := wide[0x7e]; got != 448 {
		t.Errorf("largest finite e4m3 (0x7e) = %v, want 448", got)
	}
	if got := wide[0xfe]; got != -448 {
		t.Errorf("most negative e4m3 (0xfe) = %v, want -448", got)
	}
	if got := wide[0x01]; got != 1.0/512 { // 2^-9
		t.Errorf("smallest e4m3 denormal (0x01) = %v, want 2^-9 = %v", got, 1.0/512)
	}
	if wide[0] != 0 || !math.Signbit(float64(wide[0x80])) {
		t.Errorf("zeros: +0 = %v, -0 signbit = %v", wide[0], math.Signbit(float64(wide[0x80])))
	}

	// Round trip: widening then narrowing must return the same encoding for
	// every one of the 256, NaN included (as some NaN).
	back := make([]byte, 256)
	simd.Float32ToFloat8E4M3Into(back, wide)
	for i := range all {
		if byte(i)&0x7f == 0x7f {
			if back[i]&0x7f != 0x7f {
				t.Errorf("e4m3 %#02x round-tripped to %#02x, want a NaN", i, back[i])
			}
			continue
		}
		if back[i] != all[i] {
			t.Errorf("e4m3 %#02x -> %v -> %#02x", i, wide[i], back[i])
		}
	}
}

func TestFloat8E5M2Exhaustive(t *testing.T) {
	all := make([]byte, 256)
	for i := range all {
		all[i] = byte(i)
	}
	wide := make([]float32, 256)
	simd.Float8E5M2ToFloat32Into(wide, all)

	// e5m2 IS IEEE-shaped: 0x7c is +Inf, 0xfc is -Inf, and the encodings
	// above them with a non-zero mantissa are NaN.
	if !math.IsInf(float64(wide[0x7c]), 1) {
		t.Errorf("e5m2 0x7c = %v, want +Inf", wide[0x7c])
	}
	if !math.IsInf(float64(wide[0xfc]), -1) {
		t.Errorf("e5m2 0xfc = %v, want -Inf", wide[0xfc])
	}
	for _, e := range []byte{0x7d, 0x7e, 0x7f} {
		if !math.IsNaN(float64(wide[e])) {
			t.Errorf("e5m2 %#02x = %v, want NaN", e, wide[e])
		}
	}
	if got := wide[0x7b]; got != 57344 {
		t.Errorf("largest finite e5m2 (0x7b) = %v, want 57344", got)
	}

	back := make([]byte, 256)
	simd.Float32ToFloat8E5M2Into(back, wide)
	for i := range all {
		v := float64(wide[i])
		if math.IsNaN(v) {
			if back[i]&0x7c != 0x7c || back[i]&0x03 == 0 {
				t.Errorf("e5m2 %#02x round-tripped to %#02x, want a NaN", i, back[i])
			}
			continue
		}
		if back[i] != all[i] {
			t.Errorf("e5m2 %#02x -> %v -> %#02x", i, wide[i], back[i])
		}
	}
}

// The formats differ at the top of their range by design, and that difference
// is the whole reason both exist. Asserting it stops a "fix" that makes e4m3
// IEEE-shaped and silently caps it at 240.
func TestFP8FormatsDifferAtTheTop(t *testing.T) {
	in := []float32{500, 1e5, float32(math.Inf(1))}
	e4 := make([]byte, len(in))
	e5 := make([]byte, len(in))
	simd.Float32ToFloat8E4M3Into(e4, in)
	simd.Float32ToFloat8E5M2Into(e5, in)

	back4 := make([]float32, len(in))
	back5 := make([]float32, len(in))
	simd.Float8E4M3ToFloat32Into(back4, e4)
	simd.Float8E5M2ToFloat32Into(back5, e5)

	// e4m3 saturates; there is no infinity to reach.
	for i, v := range back4 {
		if math.IsInf(float64(v), 0) {
			t.Errorf("e4m3 turned %v into an infinity", in[i])
		}
		if v != 448 {
			t.Errorf("e4m3(%v) = %v, want saturation at 448", in[i], v)
		}
	}
	// e5m2 has the range for 500 and overflows to infinity past 57344.
	if back5[0] == 448 || math.IsInf(float64(back5[0]), 0) {
		t.Errorf("e5m2(500) = %v, but 500 is inside its range", back5[0])
	}
	if !math.IsInf(float64(back5[2]), 1) {
		t.Errorf("e5m2(+Inf) = %v, want +Inf", back5[2])
	}
}

// Rounding is to nearest even, as everywhere else here. e4m3 has three
// mantissa bits, so between 1 and 2 the representable values are 0.125 apart
// and the halfway points fall on 0.0625.
func TestFloat8E4M3RoundsToNearestEven(t *testing.T) {
	in := []float32{1.0625, 1.1875, 1.3125, 1.4375}
	// Halfway between 1.0 and 1.125 -> 1.0 (even mantissa)
	// Halfway between 1.125 and 1.25 -> 1.25 (even)
	// Halfway between 1.25 and 1.375 -> 1.25 (even)
	// Halfway between 1.375 and 1.5 -> 1.5 (even)
	want := []float32{1.0, 1.25, 1.25, 1.5}
	q := make([]byte, len(in))
	back := make([]float32, len(in))
	simd.Float32ToFloat8E4M3Into(q, in)
	simd.Float8E4M3ToFloat32Into(back, q)
	for i := range want {
		if back[i] != want[i] {
			t.Errorf("e4m3(%v) = %v, want %v (round half to even)", in[i], back[i], want[i])
		}
	}
}

func TestFP8NoAlloc(t *testing.T) {
	a := make([]float32, 4096)
	d := make([]byte, 4096)
	w := make([]float32, 4096)
	if n := testing.AllocsPerRun(20, func() { simd.Float32ToFloat8E4M3Into(d, a) }); n != 0 {
		t.Errorf("Float32ToFloat8E4M3Into allocated %v times per run", n)
	}
	if n := testing.AllocsPerRun(20, func() { simd.Float8E5M2ToFloat32Into(w, d) }); n != 0 {
		t.Errorf("Float8E5M2ToFloat32Into allocated %v times per run", n)
	}
}
