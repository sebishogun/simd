package benchmarks

// Task 56 says not to attempt ParseFloats naively, and to measure first what
// fraction of realistic input takes strconv's fast path. This is that
// measurement.
//
// strconv implements Eisel-Lemire, so 0.54 GB/s is not evidence that a simple
// kernel beats it. What decides the question is whether a *exactly rounded*
// fast path exists that avoids Eisel-Lemire entirely. It does, and it is
// Clinger's: if the mantissa fits in 53 bits and the decimal exponent is in
// [-22, 22], then mantissa * 10^exp computed in one IEEE double operation is
// exactly the correctly rounded result, because both operands are exact and
// one rounding occurs. That is bit-identity, not an approximation, so it would
// not need a Fast* name.
//
// So the question this measures is: how much realistic input satisfies it, and
// how much faster is the arithmetic when it does.

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

var sinkFloat float64

// floatCorpus returns n comma-separated values of the given flavour.
func floatCorpus(n int, flavour string) ([]byte, []string) {
	p := make([]string, n)
	for i := range p {
		switch flavour {
		case "csv": // sensor readings, prices, coordinates — the common case
			p[i] = strconv.FormatFloat(float64(i%100000)/100, 'f', 2, 64)
		case "sci": // scientific notation, small exponents
			p[i] = strconv.FormatFloat(float64(i%1000)*1e-3, 'e', 6, 64)
		case "long": // full round-trip precision, 17 significant digits
			p[i] = strconv.FormatFloat(math.Sqrt(float64(i+1)), 'g', 17, 64)
		case "hard": // huge exponents, where no fast path can apply
			p[i] = fmt.Sprintf("%de%d", i%1000+1, (i%600)-300)
		}
	}
	return []byte(strings.Join(p, ",")), p
}

// clingerEligible reports whether s satisfies the exact fast path: at most 19
// mantissa digits with a value under 2^53, and a decimal exponent within
// [-22, 22].
func clingerEligible(s string) bool {
	i, neg := 0, false
	if i < len(s) && (s[i] == '-' || s[i] == '+') {
		neg = s[i] == '-'
		i++
	}
	_ = neg
	var mant uint64
	digits, frac := 0, 0
	seenDot := false
	for ; i < len(s); i++ {
		c := s[i]
		if c == '.' {
			if seenDot {
				return false
			}
			seenDot = true
			continue
		}
		if c < '0' || c > '9' {
			break
		}
		if digits >= 19 {
			return false
		}
		mant = mant*10 + uint64(c-'0')
		digits++
		if seenDot {
			frac++
		}
	}
	if digits == 0 || mant >= 1<<53 {
		return false
	}
	exp := -frac
	if i < len(s) && (s[i] == 'e' || s[i] == 'E') {
		i++
		esign := 1
		if i < len(s) && (s[i] == '-' || s[i] == '+') {
			if s[i] == '-' {
				esign = -1
			}
			i++
		}
		e := 0
		for ; i < len(s); i++ {
			if s[i] < '0' || s[i] > '9' {
				return false
			}
			e = e*10 + int(s[i]-'0')
			if e > 400 {
				return false
			}
		}
		exp += esign * e
	}
	if i != len(s) {
		return false
	}
	return exp >= -22 && exp <= 22
}

// pow10exact holds the powers of ten that are exactly representable as a
// float64, which is what makes Clinger's fast path exact.
var pow10exact = [23]float64{
	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
	1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
}

// parseFloatClinger is the fast path, for measuring the arithmetic. It returns
// ok=false for anything it cannot handle exactly, where a real implementation
// would defer to strconv.
func parseFloatClinger(s string) (float64, bool) {
	i := 0
	neg := false
	if i < len(s) && (s[i] == '-' || s[i] == '+') {
		neg = s[i] == '-'
		i++
	}
	var mant uint64
	digits, frac := 0, 0
	seenDot := false
	for ; i < len(s); i++ {
		c := s[i]
		if c == '.' {
			if seenDot {
				return 0, false
			}
			seenDot = true
			continue
		}
		if c < '0' || c > '9' {
			break
		}
		if digits >= 19 {
			return 0, false
		}
		mant = mant*10 + uint64(c-'0')
		digits++
		if seenDot {
			frac++
		}
	}
	if digits == 0 || mant >= 1<<53 {
		return 0, false
	}
	exp := -frac
	if i < len(s) && (s[i] == 'e' || s[i] == 'E') {
		i++
		esign := 1
		if i < len(s) && (s[i] == '-' || s[i] == '+') {
			if s[i] == '-' {
				esign = -1
			}
			i++
		}
		e := 0
		for ; i < len(s); i++ {
			if s[i] < '0' || s[i] > '9' {
				return 0, false
			}
			e = e*10 + int(s[i]-'0')
			if e > 400 {
				return 0, false
			}
		}
		exp += esign * e
	}
	if i != len(s) || exp < -22 || exp > 22 {
		return 0, false
	}
	f := float64(mant)
	if exp >= 0 {
		f *= pow10exact[exp]
	} else {
		f /= pow10exact[-exp]
	}
	if neg {
		f = -f
	}
	return f, true
}

func BenchmarkParseFloatSurvey(b *testing.B) {
	for _, flavour := range []string{"csv", "sci", "long", "hard"} {
		line, parts := floatCorpus(20000, flavour)

		// How much of this corpus the exact fast path covers, and whether it
		// agrees with strconv bit for bit where it applies. Reported once, as
		// a benchmark that does no work, because it is the number the decision
		// turns on.
		eligible, agree := 0, 0
		for _, p := range parts {
			if clingerEligible(p) {
				eligible++
				got, ok := parseFloatClinger(p)
				want, _ := strconv.ParseFloat(p, 64)
				if ok && math.Float64bits(got) == math.Float64bits(want) {
					agree++
				}
			}
		}
		b.Run(fmt.Sprintf("%s/eligible=%d%%_exact=%d%%", flavour,
			eligible*100/len(parts), agree*100/max(eligible, 1)), func(b *testing.B) {
			b.SetBytes(1)
			for b.Loop() {
			}
		})

		b.Run(flavour+"/impl=strconv", func(b *testing.B) {
			b.SetBytes(int64(len(line)))
			for b.Loop() {
				var t float64
				for _, p := range parts {
					v, _ := strconv.ParseFloat(p, 64)
					t += v
				}
				sinkFloat = t
			}
		})
		idxs := boundaries(line, 44)
		dstF := make([]float64, len(parts))
		b.Run(flavour+"/impl=simd", func(b *testing.B) {
			b.SetBytes(int64(len(line)))
			for b.Loop() {
				n, _ := simd.ParseFloats(dstF, line, idxs)
				sinkFloat = float64(n)
			}
		})
		b.Run(flavour+"/impl=clinger", func(b *testing.B) {
			b.SetBytes(int64(len(line)))
			for b.Loop() {
				var t float64
				for _, p := range parts {
					v, ok := parseFloatClinger(p)
					if !ok {
						v, _ = strconv.ParseFloat(p, 64)
					}
					t += v
				}
				sinkFloat = t
			}
		})
	}
}
