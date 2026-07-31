package encode

import (
	"encoding/binary"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// The whole point of these is agreeing with encoding/binary, since that is
// what the bytes will be read back by.
func TestVarintLenAgainstStdlib(t *testing.T) {
	var vals []uint64
	for _, v := range []uint64{
		0, 1, 0x7f, 0x80, 0x3fff, 0x4000, 0x1fffff, 0x200000,
		0xfffffff, 0x10000000, 1<<35 - 1, 1 << 35, 1<<63 - 1, 1 << 63,
		^uint64(0),
	} {
		vals = append(vals, v)
	}
	r := rand.New(rand.NewPCG(21, 22))
	for range 500 {
		// Spread across every width rather than uniformly, which would put
		// almost every value in the ten-byte bucket.
		vals = append(vals, r.Uint64()>>(r.UintN(64)))
	}

	buf := make([]byte, binary.MaxVarintLen64)
	got64 := make([]int32, len(vals))
	simd.VarintLenInto(got64, vals)
	for i, v := range vals {
		want := binary.PutUvarint(buf, v)
		if int(got64[i]) != want {
			t.Fatalf("VarintLen(%#x) = %d, binary.PutUvarint wrote %d",
				v, got64[i], want)
		}
	}

	v32 := make([]uint32, len(vals))
	for i, v := range vals {
		v32[i] = uint32(v)
	}
	got32 := make([]int32, len(v32))
	simd.VarintLenInto(got32, v32)
	for i, v := range v32 {
		want := binary.PutUvarint(buf, uint64(v))
		if int(got32[i]) != want {
			t.Fatalf("VarintLen(uint32 %#x) = %d, want %d", v, got32[i], want)
		}
	}
}

func TestVarintSizeIsTheSumOfTheLengths(t *testing.T) {
	r := rand.New(rand.NewPCG(23, 24))
	// Lengths past the eight-lane fold and its tail.
	for _, n := range []int{0, 1, 7, 8, 9, 17, 1000, 4096} {
		a := make([]uint64, n)
		for i := range a {
			a[i] = r.Uint64() >> (r.UintN(64))
		}
		lens := make([]int32, n)
		simd.VarintLenInto(lens, a)
		want := 0
		for _, l := range lens {
			want += int(l)
		}
		if got := simd.VarintSize(a); got != want {
			t.Errorf("n=%d: VarintSize = %d, sum of lengths = %d", n, got, want)
		}

		a32 := make([]uint32, n)
		for i := range a32 {
			a32[i] = uint32(a[i])
		}
		simd.VarintLenInto(lens, a32)
		want = 0
		for _, l := range lens {
			want += int(l)
		}
		if got := simd.VarintSize(a32); got != want {
			t.Errorf("n=%d uint32: VarintSize = %d, sum of lengths = %d", n, got, want)
		}
	}
}

// AppendVarints must produce exactly what a binary.PutUvarint loop would, and
// must land in exactly the space VarintSize predicted — a size that was one
// byte out would be caught here rather than by a corrupt stream later.
func TestAppendVarintsRoundTrips(t *testing.T) {
	r := rand.New(rand.NewPCG(25, 26))
	a := make([]uint64, 777)
	for i := range a {
		a[i] = r.Uint64() >> (r.UintN(64))
	}

	var want []byte
	buf := make([]byte, binary.MaxVarintLen64)
	for _, v := range a {
		want = append(want, buf[:binary.PutUvarint(buf, v)]...)
	}

	got := simd.AppendVarints(nil, a)
	if string(got) != string(want) {
		t.Fatalf("AppendVarints produced %d bytes, binary produced %d; they differ",
			len(got), len(want))
	}
	if len(got) != simd.VarintSize(a) {
		t.Errorf("VarintSize said %d, AppendVarints wrote %d",
			simd.VarintSize(a), len(got))
	}

	// Reading it back with the standard library closes the loop.
	rest := got
	for i, v := range a {
		x, n := binary.Uvarint(rest)
		if n <= 0 || x != v {
			t.Fatalf("at %d: decoded %#x (n=%d), want %#x", i, x, n, v)
		}
		rest = rest[n:]
	}
	if len(rest) != 0 {
		t.Errorf("%d bytes left over after decoding", len(rest))
	}
}

// Appending onto a buffer that already has content, and onto one with spare
// capacity, are the two cases where an off-by-one in the grow would corrupt
// what was already there.
func TestAppendVarintsPreservesTheBuffer(t *testing.T) {
	a := []uint64{1, 300, 1 << 40}
	prefix := []byte("keep me")

	got := simd.AppendVarints(append([]byte(nil), prefix...), a)
	if string(got[:len(prefix)]) != string(prefix) {
		t.Errorf("the prefix was clobbered: %q", got[:len(prefix)])
	}

	roomy := make([]byte, len(prefix), 512)
	copy(roomy, prefix)
	got2 := simd.AppendVarints(roomy, a)
	if string(got2) != string(got) {
		t.Errorf("appending into spare capacity gave a different result")
	}
}

func TestVarintNoAlloc(t *testing.T) {
	a := make([]uint64, 4096)
	lens := make([]int32, len(a))
	if n := testing.AllocsPerRun(50, func() { simd.VarintLenInto(lens, a) }); n != 0 {
		t.Errorf("VarintLenInto allocated %v times per run", n)
	}
	if n := testing.AllocsPerRun(50, func() { _ = simd.VarintSize(a) }); n != 0 {
		t.Errorf("VarintSize allocated %v times per run", n)
	}
	// AppendVarints into a buffer with room must not allocate either; the
	// whole reason it asks VarintSize first is to grow at most once.
	dst := make([]byte, 0, simd.VarintSize(a))
	if n := testing.AllocsPerRun(50, func() { _ = simd.AppendVarints(dst, a) }); n != 0 {
		t.Errorf("AppendVarints into a sized buffer allocated %v times per run", n)
	}
}
