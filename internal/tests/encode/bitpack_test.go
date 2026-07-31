package encode

import (
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

// Round trip at every width, which is the whole contract: pack then unpack
// must return the input for any bit width and any count.
func TestBitPackRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(701, 709))
	for bits := int32(1); bits <= 32; bits++ {
		mask := uint32(0xffffffff)
		if bits < 32 {
			mask = 1<<uint(bits) - 1
		}
		// Counts either side of a word boundary for this width, so the
		// straddling case runs rather than only the aligned one.
		for _, n := range []int{0, 1, 2, 3, 7, 8, 31, 32, 33, 64, 100, 1000} {
			a := make([]uint32, n)
			for i := range a {
				a[i] = r.Uint32() & mask
			}
			words := (n*int(bits) + 31) / 32
			packed := make([]uint32, words+1) // +1 for the straddle read
			back := make([]uint32, n)

			simd.BitPackInto(packed, a, bits)
			simd.BitUnpackInto(back, packed, bits)
			for i := range a {
				if back[i] != a[i] {
					t.Fatalf("bits=%d n=%d i=%d: %d -> %d", bits, n, i, a[i], back[i])
				}
			}
		}
	}
}

// The density claim: the output really is bits*n bits and not more.
func TestBitPackIsDense(t *testing.T) {
	const n = 1000
	a := make([]uint32, n)
	for i := range a {
		a[i] = uint32(i) & 0x3ff // fits in 10 bits
	}
	words := (n*10 + 31) / 32
	packed := make([]uint32, words+1)
	simd.BitPackInto(packed, a, 10)

	// 1000 ten-bit values is 10000 bits, 313 words, against 1000 unpacked.
	if words != 313 {
		t.Fatalf("expected 313 words, got %d", words)
	}
	back := make([]uint32, n)
	simd.BitUnpackInto(back, packed, 10)
	for i := range a {
		if back[i] != a[i] {
			t.Fatalf("i=%d: %d != %d", i, back[i], a[i])
		}
	}
}

// Values wider than the field are truncated, not allowed to corrupt the next
// one — the failure mode that makes a whole block unreadable.
func TestBitPackTruncatesOverwideValues(t *testing.T) {
	a := []uint32{0xffffffff, 0, 0xffffffff, 0}
	packed := make([]uint32, 4)
	back := make([]uint32, len(a))
	simd.BitPackInto(packed, a, 4)
	simd.BitUnpackInto(back, packed, 4)
	want := []uint32{15, 0, 15, 0}
	for i := range want {
		if back[i] != want[i] {
			t.Errorf("i=%d: got %d want %d", i, back[i], want[i])
		}
	}
}

func TestBitPackRejectsBadArguments(t *testing.T) {
	dst := make([]uint32, 8)
	for i := range dst {
		dst[i] = 0xdeadbeef
	}
	a := make([]uint32, 8)
	for _, bits := range []int32{0, -1, 33} {
		simd.BitPackInto(dst, a, bits)
		for i := range dst {
			if dst[i] != 0xdeadbeef {
				t.Fatalf("bits=%d wrote to dst[%d]", bits, i)
			}
		}
	}
	// Too small a destination must also do nothing.
	simd.BitPackInto(dst[:1], a, 32)
	if dst[0] != 0xdeadbeef {
		t.Error("a short destination was written to")
	}
}

func TestBitPackNoAlloc(t *testing.T) {
	a := make([]uint32, 1024)
	packed := make([]uint32, 1024)
	back := make([]uint32, 1024)
	if n := testing.AllocsPerRun(20, func() { simd.BitPackInto(packed, a, 12) }); n != 0 {
		t.Errorf("BitPackInto allocated %v times per run", n)
	}
	if n := testing.AllocsPerRun(20, func() { simd.BitUnpackInto(back, packed, 12) }); n != 0 {
		t.Errorf("BitUnpackInto allocated %v times per run", n)
	}
}
