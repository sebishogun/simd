package simd_test

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/sebishogun/simd"
)

// strconvShortest is encoding/json's rewrite of strconv's output: the
// same digits, the format rule applied, the e-exponent unpadded.
func strconvShortest(f float64) string {
	abs := math.Abs(f)
	fmtc := byte('f')
	if abs != 0 && (abs < 1e-6 || abs >= 1e21) {
		fmtc = 'e'
	}
	b := strconv.AppendFloat(nil, f, fmtc, -1, 64)
	if fmtc == 'e' {
		n := len(b)
		if n >= 4 && b[n-4] == 'e' && b[n-3] == '-' && b[n-2] == '0' {
			b[n-2] = b[n-1]
			b = b[:n-1]
		}
	}
	return string(b)
}

func TestFormatFloat64MatchesStrconv(t *testing.T) {
	var dst [32]byte
	check := func(f float64) {
		t.Helper()
		n := simd.FormatFloat64(dst[:], f)
		if got, want := string(dst[:n]), strconvShortest(f); got != want {
			t.Fatalf("%x (%g): got %q want %q", math.Float64bits(f), f, got, want)
		}
	}
	for _, f := range []float64{
		0, math.Copysign(0, -1), 1, -1, 3.0, 0.3, 1e15 - 1, 1e15, 1e21, 1e-6,
		9.999999999999999e20, 1.0000000000000001e21, 9.99e-7, 1e-7,
		math.MaxFloat64, math.SmallestNonzeroFloat64, 5e-324, 2.2250738585072014e-308,
		1.7976931348623157e308, 0.1, 0.5, 2.5, 122.416294033786585,
		1e16, 1e17, 123456789012345680000, 6.226662346353213e-309,
	} {
		check(f)
		check(-f)
	}
	rng := rand.New(rand.NewSource(11))
	for i := 0; i < 2_000_000; i++ {
		f := math.Float64frombits(rng.Uint64())
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		check(f)
	}
	for i := 0; i < 500_000; i++ {
		check(rng.NormFloat64() * math.Pow(10, float64(rng.Intn(40)-20)))
	}
}
