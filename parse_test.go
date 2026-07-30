package simd_test

import (
	"bytes"
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

// boundaries builds the index slice ParseInts wants from a separated line.
func boundaries(line []byte, sep byte) []int32 {
	idx := make([]int32, 0, 64)
	for i, c := range line {
		if c == sep {
			idx = append(idx, int32(i))
		}
	}
	return append(idx, int32(len(line)))
}

func TestParseInts(t *testing.T) {
	// The extremes are the point: both int64 limits must convert, and one
	// past either must be rejected rather than wrapped.
	good := []int64{0, 1, -1, 7, -7, 42, 1000000, -1000000,
		math.MaxInt64, math.MinInt64, 999999999999999999, -999999999999999999}
	parts := make([]string, len(good))
	for i, v := range good {
		parts[i] = strconv.FormatInt(v, 10)
	}
	line := []byte(strings.Join(parts, ","))
	idx := boundaries(line, ',')
	dst := make([]int64, len(good))
	n, ok := simd.ParseInts(dst, line, idx)
	if !ok || n != len(good) {
		t.Fatalf("n=%d ok=%v, want %d true", n, ok, len(good))
	}
	for i, want := range good {
		if dst[i] != want {
			t.Fatalf("field %d = %d, want %d", i, dst[i], want)
		}
	}

	// A leading plus is accepted, as strconv does.
	pl := []byte("+5,+0,+9223372036854775807")
	pd := make([]int64, 3)
	if n, ok := simd.ParseInts(pd, pl, boundaries(pl, ',')); !ok || n != 3 ||
		pd[0] != 5 || pd[2] != math.MaxInt64 {
		t.Fatalf("leading plus: n=%d ok=%v %v", n, ok, pd)
	}

	// Every rejection, each returning the index of the offending field.
	bad := []struct {
		line string
		at   int
	}{
		{"1,x,3", 1},
		{"1,,3", 1},
		{"1,-,3", 1},
		{"1,+,3", 1},
		{"1,1 2,3", 1},
		{"1,12345678901234567890,3", 1}, // 20 digits
		{"1,9223372036854775808,3", 1},  // MaxInt64+1
		{"1,-9223372036854775809,3", 1}, // MinInt64-1
		{"1,2,99999999999999999999999", 2},
		{"1e3,2,3", 0},
	}
	for _, c := range bad {
		l := []byte(c.line)
		d := make([]int64, 8)
		n, ok := simd.ParseInts(d, l, boundaries(l, ','))
		if ok {
			t.Errorf("%q accepted, want rejected", c.line)
		}
		if n != c.at {
			t.Errorf("%q stopped at %d, want %d", c.line, n, c.at)
		}
	}

	// Against strconv on generated data, at lengths past the threshold.
	for _, count := range []int{1, 7, 64, 1000, 20000} {
		p := make([]string, count)
		for i := range p {
			p[i] = strconv.Itoa(i*7919 - 500000)
		}
		l := []byte(strings.Join(p, ","))
		ix := boundaries(l, ',')
		d := make([]int64, count)
		t.Run(fmt.Sprintf("n=%d", count), func(t *testing.T) {
			n, ok := simd.ParseInts(d, l, ix)
			if !ok || n != count {
				t.Fatalf("n=%d ok=%v, want %d true", n, ok, count)
			}
			for i, str := range p {
				w, _ := strconv.ParseInt(str, 10, 64)
				if d[i] != w {
					t.Fatalf("field %d of %q = %d, want %d", i, str, d[i], w)
				}
			}
		})
	}

	// A short destination bounds the work rather than overflowing it.
	l := []byte("1,2,3,4,5")
	d := make([]int64, 2)
	if n, ok := simd.ParseInts(d, l, boundaries(l, ',')); n != 2 || !ok {
		t.Errorf("short dst: n=%d ok=%v, want 2 true", n, ok)
	}
}

func TestFormatInts(t *testing.T) {
	// Round-trip with ParseInts is the defining property; strconv is the
	// per-value oracle. Both int64 extremes, zero, and single-digit values,
	// at lengths on both sides of the guard.
	cases := [][]int64{
		{},
		{0},
		{5},
		{-5},
		{math.MaxInt64},
		{math.MinInt64},
		{0, 1, -1, 10, -10, 99, 100, -101},
		{math.MinInt64, math.MaxInt64, 0, -9999999999999999},
	}
	big := make([]int64, 20000)
	for i := range big {
		big[i] = int64(i*7919-500000) * int64(1+i%3)
	}
	cases = append(cases, big)

	for ci, vals := range cases {
		want := make([]byte, 0, 21*len(vals))
		for i, v := range vals {
			want = strconv.AppendInt(want, v, 10)
			if i != len(vals)-1 {
				want = append(want, ';')
			}
		}
		dst := make([]byte, 21*len(vals))
		n := simd.FormatInts(dst, vals, ';')
		if n != len(want) || !bytes.Equal(dst[:max(n, 0)], want) {
			t.Fatalf("case %d: n=%d want %d; %q vs %q", ci, n, len(want),
				dst[:max(n, 0)], want)
		}
		if len(vals) == 0 {
			continue
		}
		// Round trip.
		idx := boundaries(dst[:n], ';')
		back := make([]int64, len(vals))
		cnt, ok := simd.ParseInts(back, dst[:n], idx)
		if !ok || cnt != len(vals) {
			t.Fatalf("case %d: round-trip parse cnt=%d ok=%v", ci, cnt, ok)
		}
		for i := range vals {
			if back[i] != vals[i] {
				t.Fatalf("case %d: round-trip [%d] = %d, want %d", ci, i, back[i], vals[i])
			}
		}
		// Exact-fit destination succeeds through the reference path.
		exact := make([]byte, len(want))
		if got := simd.FormatInts(exact, vals, ';'); got != len(want) {
			t.Fatalf("case %d: exact-fit = %d, want %d", ci, got, len(want))
		}
		// One byte short fails cleanly.
		if len(want) > 0 {
			short := make([]byte, len(want)-1)
			if got := simd.FormatInts(short, vals, ';'); got != -1 {
				t.Fatalf("case %d: short dst = %d, want -1", ci, got)
			}
		}
	}
}

func TestParseUints(t *testing.T) {
	// The extremes are the point again, and the upper one is the whole reason
	// this is not a wrapper around ParseInts: MaxUint64 is more than twice
	// MaxInt64, and a 20-digit field is one digit longer than any int64.
	good := []uint64{0, 1, 7, 42, 1000000, 9223372036854775807,
		9223372036854775808, 18446744073709551614, math.MaxUint64,
		999999999999999999, 10000000000000000000}
	parts := make([]string, len(good))
	for i, v := range good {
		parts[i] = strconv.FormatUint(v, 10)
	}
	line := []byte(strings.Join(parts, ","))
	dst := make([]uint64, len(good))
	n, ok := simd.ParseUints(dst, line, boundaries(line, ','))
	if !ok || n != len(good) {
		t.Fatalf("n=%d ok=%v, want %d true", n, ok, len(good))
	}
	for i, want := range good {
		if dst[i] != want {
			t.Fatalf("field %d = %d, want %d", i, dst[i], want)
		}
	}

	// Rejections, each stopping at the field that is wrong. No sign is
	// accepted at all, which is strconv.ParseUint's rule and not ParseInts'.
	for _, c := range []struct {
		in   string
		stop int
	}{
		{"1,2,-3,4", 2},                  // minus rejected
		{"1,+2", 1},                      // and so is plus
		{"1,,2", 1},                      // empty field
		{"1,18446744073709551616,2", 1},  // one past MaxUint64
		{"1,99999999999999999999,2", 1},  // 20 digits, too large
		{"1,184467440737095516150,2", 1}, // 21 digits, too long
		{"1,2a,3", 1},                    // trailing junk
		{"1,0x10,2", 1},                  // no base prefixes
		{"1, 2,3", 1},                    // no leading space
	} {
		line := []byte(c.in)
		dst := make([]uint64, 8)
		n, ok := simd.ParseUints(dst, line, boundaries(line, ','))
		if ok {
			t.Errorf("ParseUints(%q) accepted, want rejection at field %d", c.in, c.stop)
			continue
		}
		if n != c.stop {
			t.Errorf("ParseUints(%q) stopped at %d, want %d", c.in, n, c.stop)
		}
	}

	// Against strconv over a long line, which is the only oracle that matters.
	// Values are spread across the range so both the 19-digit and 20-digit
	// paths are exercised, the latter being where the kernel holds the leading
	// digit out of the weighted sum to keep it from wrapping.
	var sb strings.Builder
	want := make([]uint64, 0, 5000)
	for i := range 5000 {
		v := uint64(i) * 3689348814741910323 // strides the whole range
		want = append(want, v)
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(strconv.FormatUint(v, 10))
	}
	big := []byte(sb.String())
	got := make([]uint64, len(want))
	n, ok = simd.ParseUints(got, big, boundaries(big, ','))
	if !ok || n != len(want) {
		t.Fatalf("long line: n=%d ok=%v, want %d true", n, ok, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("long line field %d = %d, want %d", i, got[i], want[i])
		}
	}

	// Round trip through FormatInts is not available for unsigned, but the
	// zero-length and short cases still have to behave.
	if n, ok := simd.ParseUints(nil, "", nil); n != 0 || !ok {
		t.Errorf("empty = (%d,%v), want (0,true)", n, ok)
	}
	short := make([]uint64, 2)
	l := []byte("1,2,3,4")
	if n, ok := simd.ParseUints(short, l, boundaries(l, ',')); n != 2 || !ok {
		t.Errorf("short dst = (%d,%v), want (2,true)", n, ok)
	}
}
