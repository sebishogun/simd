package simd_test

import (
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
