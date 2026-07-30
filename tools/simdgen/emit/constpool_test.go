package emit

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

// TestPoolRenderCoversEveryByte reads the rendered DATA directives back and
// reconstructs the pool from them, which is what the assembler does.
//
// The bug this exists for: render emitted only whole 8-byte directives, so a
// pool whose length was not a multiple of eight lost its tail. GLOBL still
// declared the true length and the assembler zero-filled the difference, so
// nothing failed to build — the last 4-byte constant simply read as zero.
// 46 of 223 ppc64le pools were affected, and simd_quantize_u8 was the one
// where it showed: the missing constant was the upper clamp, so every element
// its scalar tail touched came out 0. One wrong element in 17, on one
// architecture, from a generator that reported success.
func TestPoolRenderCoversEveryByte(t *testing.T) {
	// Lengths 1..40 plus the four sizes that actually occur in internal/ppc64le.
	lens := []int{436, 1724, 4700, 6684}
	for n := 1; n <= 40; n++ {
		lens = append(lens, n)
	}

	for _, n := range lens {
		want := make([]byte, n)
		for i := range want {
			// Distinct and non-zero, so a dropped byte cannot pass by
			// coincidentally matching the assembler's zero fill.
			want[i] = byte(i%251 + 1)
		}

		p := newPoolSet()
		name := p.add("pool", want)
		got, size := replayPool(t, p.render(), name)

		if size != n {
			t.Errorf("n=%d: GLOBL declares %d bytes", n, size)
		}
		if len(got) != n {
			t.Errorf("n=%d: DATA covers %d bytes, want %d", n, len(got), n)
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("n=%d: byte %d = %#02x, want %#02x", n, i, got[i], want[i])
				break
			}
		}
	}
}

// replayPool interprets the DATA and GLOBL directives for one symbol the way
// the assembler does, returning the bytes they define and the declared size.
// It deliberately does not zero-fill: a gap left by a missing directive shows
// up as a short result rather than being papered over.
func replayPool(t *testing.T, s, name string) (data []byte, size int) {
	t.Helper()
	covered := map[int]byte{}
	high := -1

	for _, line := range strings.Split(s, "\n") {
		if !strings.HasPrefix(line, "DATA "+name+"<>+") && !strings.HasPrefix(line, "GLOBL "+name+"<>") {
			continue
		}
		if strings.HasPrefix(line, "GLOBL") {
			// GLOBL sym<>(SB), RODATA|NOPTR, $436
			n, err := strconv.Atoi(line[strings.LastIndex(line, "$")+1:])
			if err != nil {
				t.Fatalf("GLOBL size: %v in %q", err, line)
			}
			size = n
			continue
		}
		// DATA sym<>+0x1a8(SB)/8, $0x...
		var off, width int
		var val uint64
		body := line[len("DATA "+name+"<>+"):]
		if _, err := fmt.Sscanf(body, "0x%x(SB)/%d, $0x%x", &off, &width, &val); err != nil {
			t.Fatalf("parse %q: %v", line, err)
		}
		if width != 1 && width != 2 && width != 4 && width != 8 {
			t.Errorf("DATA width /%d is not a directive the assembler has", width)
		}
		if off%width != 0 {
			t.Errorf("DATA at 0x%x is not %d-byte aligned", off, width)
		}
		for j := 0; j < width; j++ {
			if _, dup := covered[off+j]; dup {
				t.Errorf("byte %d written twice", off+j)
			}
			covered[off+j] = byte(val >> (8 * j)) // little-endian, as Plan 9 assembles it
			if off+j > high {
				high = off + j
			}
		}
	}

	for i := 0; i <= high; i++ {
		b, ok := covered[i]
		if !ok {
			return data, size // hole: stop, so the caller sees a short pool
		}
		data = append(data, b)
	}
	return data, size
}
