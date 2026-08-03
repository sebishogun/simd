package text

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
)

// The five regions JSONMasks writes must be exactly what five separate
// MaskBits calls would write. That is the contract, and it is the only reason
// fusing them is allowed to be faster.
func separately(b []byte) [][]byte {
	n := (len(b) + 7) / 8
	q := make([]byte, n)
	e := make([]byte, n)
	st := make([]byte, n)
	ct := make([]byte, n)
	ws := make([]byte, n)
	simd.MaskBits(q, b, '"')
	simd.MaskBits(e, b, '\\')
	simd.MaskBitsAny(st, b, "{}[]")
	simd.MaskBitsLess(ct, b, 0x20)
	simd.MaskBitsAny(ws, b, " \t\n\r")
	return [][]byte{q, e, st, ct, ws}
}

func fused(b []byte, want uint32) [][]byte {
	n := (len(b) + 7) / 8
	dst := make([]byte, 5*n)
	simd.JSONMasks(dst, b, want)
	out := make([][]byte, 5)
	for i := range out {
		out[i] = dst[i*n : (i+1)*n]
	}
	return out
}

var regionNames = []string{"quote", "escape", "structural", "control", "space"}

func checkMasks(t *testing.T, b []byte) {
	t.Helper()
	want := separately(b)
	got := fused(b, simd.JSONMaskAll)
	for i := range want {
		if !bytes.Equal(got[i], want[i]) {
			t.Errorf("len=%d %s mask differs\n got %x\nwant %x",
				len(b), regionNames[i], got[i], want[i])
		}
	}
}

func TestJSONMasksMatchesSeparateCalls(t *testing.T) {
	// The interesting lengths are around the vector width and its multiples,
	// because the tail loop is where a fused kernel gets the padding wrong.
	for n := 0; n <= 200; n++ {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte(i)
		}
		checkMasks(t, b)
	}
	for _, n := range []int{255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096, 65537} {
		b := make([]byte, n)
		r := rand.New(rand.NewSource(int64(n)))
		for i := range b {
			// Weighted towards the bytes the masks care about, so the bits are
			// dense rather than almost all zero.
			switch r.Intn(8) {
			case 0:
				b[i] = '"'
			case 1:
				b[i] = '\\'
			case 2:
				b[i] = []byte("{}[]")[r.Intn(4)]
			case 3:
				b[i] = []byte(" \t\n\r")[r.Intn(4)]
			case 4:
				b[i] = byte(r.Intn(0x20))
			default:
				b[i] = byte(r.Intn(256))
			}
		}
		checkMasks(t, b)
	}
}

// A real document, which is what the kernel is for.
func TestJSONMasksOnRealJSON(t *testing.T) {
	doc := []byte(`{"a":[1,2,{"b":"x\"y"}],"c":  {"d":null},
	"e":"tab\there and \\ backslash","f":[true,false]}`)
	checkMasks(t, doc)
	// And repeated, so it crosses several vector blocks.
	big := bytes.Repeat(doc, 200)
	checkMasks(t, big)
}

// want must select regions without moving them: a caller that asks for two
// masks must find them where it would have found them asking for five.
func TestJSONMasksWantSelectsWithoutMoving(t *testing.T) {
	b := bytes.Repeat([]byte(`{"a": "b\\c"} `), 40)
	all := fused(b, simd.JSONMaskAll)
	for _, want := range []uint32{
		simd.JSONMaskQuote,
		simd.JSONMaskEscape,
		simd.JSONMaskQuote | simd.JSONMaskEscape,
		simd.JSONMaskStructural,
		simd.JSONMaskControl | simd.JSONMaskSpace,
		simd.JSONMaskQuote | simd.JSONMaskStructural | simd.JSONMaskSpace,
	} {
		got := fused(b, want)
		for i := range all {
			if want&(1<<uint(i)) == 0 {
				continue // not asked for; contents are not promised
			}
			if !bytes.Equal(got[i], all[i]) {
				t.Errorf("want=%05b: %s region differs from the all-regions run\n got %x\nwant %x",
					want, regionNames[i], got[i], all[i])
			}
		}
	}
}

func FuzzJSONMasksMatchesSeparateCalls(f *testing.F) {
	f.Add([]byte(`{"a":1}`))
	f.Add([]byte(``))
	f.Add(bytes.Repeat([]byte(`"\ {} `), 30))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<16 {
			return
		}
		want := separately(b)
		got := fused(b, simd.JSONMaskAll)
		for i := range want {
			if !bytes.Equal(got[i], want[i]) {
				t.Fatalf("len=%d %s differs\n got %x\nwant %x",
					len(b), regionNames[i], got[i], want[i])
			}
		}
	})
}
