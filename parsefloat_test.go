package simd_test

// The oracle is strconv.ParseFloat, and the contract is bit identity — not
// closeness. Clinger's fast path is exactly rounded where it applies, so any
// difference at all is a bug, and math.Float64bits is the comparison.

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

func checkFloats(t *testing.T, parts []string) {
	t.Helper()
	line := []byte(strings.Join(parts, ","))
	idx := boundaries(line, ',')
	dst := make([]float64, len(parts))
	n, ok := simd.ParseFloats(dst, line, idx)

	// The oracle decides both the values and where parsing should stop.
	wantN, wantOK := len(parts), true
	want := make([]float64, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			wantN, wantOK = i, false
			break
		}
		want[i] = v
	}
	if n != wantN || ok != wantOK {
		t.Fatalf("n=%d ok=%v, want %d %v (in %q)", n, ok, wantN, wantOK, line)
	}
	for i := range wantN {
		if math.Float64bits(dst[i]) != math.Float64bits(want[i]) {
			t.Fatalf("field %d = %v (%#016x), want %v (%#016x) from %q",
				i, dst[i], math.Float64bits(dst[i]),
				want[i], math.Float64bits(want[i]), parts[i])
		}
	}
}

func TestParseFloats(t *testing.T) {
	// Hand-picked: the fast path's boundaries, the signs of zero, and the
	// forms only strconv can take.
	checkFloats(t, []string{
		"0", "-0", "+0", "0.0", "-0.0", // signed zero must survive
		"1", "-1", "3.14159", "2.718281828459045",
		"0.1", "0.2", "0.3", // the classic inexact decimals
		"1e22", "1e-22", "1e23", "1e-23", // straddles the exponent bound
		"9007199254740991", "9007199254740992", "9007199254740993", // 2^53 -1,0,+1
		"1234567890123456789",              // 19 digits, over 2^53
		"1.7976931348623157e308", "5e-324", // max and min subnormal
		"1e309", "-1e309", // overflow to Inf, which strconv accepts
		"0.000001", "123456.789", "-98765.4321",
		".5", "5.", "-.5", // leading and trailing dot
		"00000000000000000001", // twenty digits, value 1
		"1e+10", "1E-10", "1E5",
	})

	// Malformed, each stopping where strconv stops.
	for _, in := range []string{
		"1,,2", "1,abc,2", "1,1.2.3,2", "1,1e,2", "1,1e+,2",
		"1,--1,2", "1,1.0e1000000,2", "1, 1,2", "1,0x1p2,2",
	} {
		checkFloats(t, strings.Split(in, ","))
	}

	// Inf and NaN, which strconv accepts and the fast path never can.
	checkFloats(t, []string{"Inf", "-Inf", "+Inf", "inf", "NaN", "nan"})

	// Random values across the whole range, formatted several ways, so both
	// the eligible and the ineligible paths are heavily exercised and their
	// interleaving — which is where the resume logic could go wrong — is too.
	r := rand.New(rand.NewPCG(263, 269))
	for _, format := range []struct {
		name string
		fn   func(int) string
	}{
		{"fixed2", func(i int) string { return strconv.FormatFloat(float64(i%100000)/100, 'f', 2, 64) }},
		{"g17", func(i int) string { return strconv.FormatFloat(math.Sqrt(float64(i+1)), 'g', 17, 64) }},
		{"sci", func(i int) string { return strconv.FormatFloat(float64(i%1000)*1e-3, 'e', 6, 64) }},
		{"random", func(int) string {
			return strconv.FormatFloat(math.Float64frombits(r.Uint64()), 'g', -1, 64)
		}},
		{"mixed", func(i int) string {
			switch i % 4 {
			case 0:
				return strconv.FormatFloat(float64(i)/7, 'f', 3, 64)
			case 1:
				return strconv.FormatFloat(math.Sqrt(float64(i+1)), 'g', 17, 64)
			case 2:
				return fmt.Sprintf("%de%d", i%1000+1, (i%600)-300)
			default:
				return strconv.Itoa(i)
			}
		}},
	} {
		t.Run(format.name, func(t *testing.T) {
			parts := make([]string, 2000)
			for i := range parts {
				parts[i] = format.fn(i)
			}
			// NaN formats as "NaN" and compares unequal to itself, but
			// Float64bits comparison handles that correctly, so nothing is
			// filtered out here.
			checkFloats(t, parts)
		})
	}

	// Short destination stops early without error, as ParseInts does.
	t.Run("shortDst", func(t *testing.T) {
		l := []byte("1.5,2.5,3.5,4.5")
		dst := make([]float64, 2)
		if n, ok := simd.ParseFloats(dst, l, boundaries(l, ',')); n != 2 || !ok {
			t.Errorf("got (%d,%v), want (2,true)", n, ok)
		}
		if dst[0] != 1.5 || dst[1] != 2.5 {
			t.Errorf("got %v, want [1.5 2.5]", dst)
		}
	})

	if n, ok := simd.ParseFloats(nil, "", nil); n != 0 || !ok {
		t.Errorf("empty = (%d,%v), want (0,true)", n, ok)
	}
}
