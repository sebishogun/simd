package simd

import (
	"strconv"
	"unsafe"
)

// noCopyString views bytes as a string without copying, the mirror of
// [textBytes]. The result must not outlive b and nothing may write through b
// while it is alive; the single caller below satisfies both.
func noCopyString(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(&b[0], len(b))
}

// Float parsing.
//
// # Why there is no kernel here
//
// This was built as a kernel first, and the generator rejected it on every
// target: "LLVM did not vectorize it" for neon, sve2, rvv, vsx, vx and lasx,
// and "needs more argument registers than amd64 has" for the seventh argument.
// Both are correct. Parsing a float is a dependent scan with a decimal point
// and an exponent to find, and the branch on eligibility is the whole
// algorithm — there is nothing for a vector unit to do.
//
// That turned out to be the right answer rather than an obstacle, because the
// speedup measured here never came from vectors. It comes from Clinger's fast
// path, which is scalar arithmetic:
//
//	If the mantissa fits in 53 bits and the decimal exponent is within
//	[-22, 22], then the mantissa and 10^exp are both exactly representable as
//	float64, so mantissa * 10^exp is a single rounding — and a single rounding
//	of an exact product is by definition the correctly rounded result.
//
// So the answer is bit-identical to strconv.ParseFloat, not close to it. That
// is why there is no Fast* variant: there is no accuracy being traded. Fields
// outside those bounds would round twice and could differ in the last place,
// so they are handed to strconv rather than approximated.
//
// Measured over 20,000 fields through the exported API, and the losing rows
// are as much the point as the winning ones:
//
//	corpus                        eligible   strconv       this
//	two-decimal CSV                   100%   462 MB/s   1063 MB/s   2.3x
//	scientific, six decimals          100%   703 MB/s   1431 MB/s   2.0x
//	17 significant digits              10%   497 MB/s    384 MB/s   0.77x
//	exponents beyond 10^22              7%   509 MB/s    395 MB/s   0.78x
//
// Ordinary numeric data is entirely eligible. Data that is mostly round-trip
// precision pays about 23% for being scanned twice, and [ParseFloats] says so
// rather than hiding it.
//
// The eligibility test itself is most of that cost, not the copy it avoids:
// removing the string copy in the fallback moved the losing rows by under 3%.

// pow10Exact holds every power of ten a float64 represents exactly. 10^22 is
// the last: 10^23 needs 77 bits of significand, and multiplying by an inexact
// power would round twice and break the argument above.
var pow10Exact = [23]float64{
	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
	1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
}

// ParseFloats converts the fields of src into float64, writing them to dst,
// and returns how many it converted and whether every one was valid.
//
// It takes the same index slice as [ParseInts]:
//
//	n := simd.IndexAll(idx, line, ',')
//	idx[n] = int32(len(line))
//	count, ok := simd.ParseFloats(vals, line, idx[:n+1])
//
// Results are identical to strconv.ParseFloat's, bit for bit, including the
// sign of zero. Fields the fast path cannot round exactly are passed to
// strconv rather than approximated, so there is no accuracy tradeoff and no
// Fast* variant. See the package comment above for what that costs on input
// that is mostly such fields.
//
// It stops at the first field strconv also rejects and returns that field's
// index, so a caller can report where the input went wrong.
func ParseFloats[S Text](dst []float64, src S, idx []int32) (int, bool) {
	b := textBytes(src)
	n := min(len(dst), len(idx))
	start := 0
	for k := range n {
		end := int(idx[k])
		if end > len(b) || end < start {
			return k, false
		}
		f := b[start:end]
		start = end + 1

		neg := false
		g := f
		if len(g) > 0 && (g[0] == '-' || g[0] == '+') {
			neg = g[0] == '-'
			g = g[1:]
		}
		if mant, exp, ok := clingerParts(g); ok {
			v := float64(mant)
			if exp >= 0 {
				v *= pow10Exact[exp]
			} else {
				v /= pow10Exact[-exp]
			}
			if neg {
				v = -v
			}
			dst[k] = v
			continue
		}
		// Not exactly representable by the fast path — Inf, NaN, a long
		// mantissa, a large exponent, or malformed. strconv decides all of
		// them, including whether it is an error at all.
		//
		// unsafe.String rather than string(f) because this is the path taken
		// by every field of a round-trip-precision corpus, and a copy per
		// field costs 25% there. It is sound: strconv.ParseFloat reads the
		// string and does not retain it, so the view cannot outlive f.
		v, err := strconv.ParseFloat(noCopyString(f), 64)
		if err != nil {
			return k, false
		}
		dst[k] = v
	}
	return n, true
}

// clingerParts splits an unsigned decimal into a mantissa and a base-ten
// exponent, reporting whether the pair is within Clinger's bounds.
//
// It is deliberately strict: anything it is not certain of, down to a leading
// zero count that might push past nineteen digits, is refused and left to
// strconv. A false negative costs one re-scan; a false positive would be a
// wrong answer in the last place.
func clingerParts(f []byte) (mant uint64, exp int, ok bool) {
	digits, frac := 0, 0
	dot := false
	i := 0
	for ; i < len(f); i++ {
		c := f[i]
		if c == '.' {
			if dot {
				return 0, 0, false
			}
			dot = true
			continue
		}
		if c == 'e' || c == 'E' {
			break
		}
		d := c - '0'
		if d > 9 {
			return 0, 0, false
		}
		if digits == 19 {
			return 0, 0, false
		}
		mant = mant*10 + uint64(d)
		digits++
		if dot {
			frac++
		}
	}
	// 1<<53 is where a uint64 stops being exactly representable as a float64.
	if digits == 0 || mant >= 1<<53 {
		return 0, 0, false
	}
	exp = -frac
	if i < len(f) {
		i++ // the 'e'
		esign := 1
		if i < len(f) && (f[i] == '-' || f[i] == '+') {
			if f[i] == '-' {
				esign = -1
			}
			i++
		}
		// A three-digit cap keeps e from overflowing and is far outside the
		// [-22, 22] this can serve anyway.
		if i >= len(f) || len(f)-i > 3 {
			return 0, 0, false
		}
		e := 0
		for ; i < len(f); i++ {
			d := f[i] - '0'
			if d > 9 {
				return 0, 0, false
			}
			e = e*10 + int(d)
		}
		exp += esign * e
	}
	if exp < -22 || exp > 22 {
		return 0, 0, false
	}
	return mant, exp, true
}
