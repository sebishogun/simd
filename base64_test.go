package simd_test

// Base64 against encoding/base64, which defines the answer.
//
// Every length from 0 to 200 rather than a sample: the encoder's tail is one
// of three shapes depending on len(src)%3 and the decoder's is one of three
// depending on the padding, and a sampled length would miss two of each.

import (
	"bytes"
	"encoding/base64"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

func TestBase64AgainstStdlib(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 11))
	for n := range 201 {
		src := make([]byte, n)
		for i := range src {
			src[i] = byte(r.UintN(256))
		}
		want := base64.StdEncoding.EncodeToString(src)

		dst := make([]byte, simd.Base64EncodedLen(n))
		got := simd.Base64Encode(dst, src)
		if got != len(want) {
			t.Fatalf("n=%d: Base64Encode wrote %d, want %d", n, got, len(want))
		}
		if string(dst[:got]) != want {
			t.Fatalf("n=%d: got %q want %q", n, dst[:got], want)
		}

		// And back, from both spellings of the encoded form.
		out := make([]byte, simd.Base64DecodedLen(len(want)))
		k := simd.Base64Decode(out, want)
		if k != n || !bytes.Equal(out[:k], src) {
			t.Fatalf("n=%d: round trip gave %d bytes %x, want %d %x", n, k, out[:max(k, 0)], n, src)
		}
		k = simd.Base64Decode(out, []byte(want))
		if k != n || !bytes.Equal(out[:k], src) {
			t.Fatalf("n=%d: []byte round trip gave %d bytes", n, k)
		}
	}
}

// TestBase64RejectsBadInput checks the failure path, which is the half of a
// decoder that a round-trip test never reaches.
func TestBase64RejectsBadInput(t *testing.T) {
	out := make([]byte, 256)
	for _, bad := range []string{
		"A", "AB", "ABC", // not a multiple of four
		"AB!D", "!!!!", "AAAA AAAA", // characters outside the alphabet
		"AAAAAAA=", "AB=C", // padding in the wrong place
		strings.Repeat("A", 64) + "AB!D",
	} {
		if got := simd.Base64Decode(out, bad); got >= 0 {
			// Anything the standard library also accepts is not a bug here.
			if _, err := base64.StdEncoding.DecodeString(bad); err == nil {
				continue
			}
			t.Errorf("Base64Decode(%q) = %d, want -1 (stdlib rejects it)", bad, got)
		}
	}
	// A short destination is a rejection, not a buffer overrun.
	if got := simd.Base64Encode(make([]byte, 3), []byte("hello")); got != -1 {
		t.Errorf("Base64Encode into a short dst = %d, want -1", got)
	}
	if got := simd.Base64Decode(make([]byte, 1), "aGVsbG8="); got != -1 {
		t.Errorf("Base64Decode into a short dst = %d, want -1", got)
	}
}

func TestBase64DoesNotAllocate(t *testing.T) {
	src := []byte(strings.Repeat("the quick brown fox ", 64))
	enc := make([]byte, simd.Base64EncodedLen(len(src)))
	n := simd.Base64Encode(enc, src)
	dec := make([]byte, simd.Base64DecodedLen(n))
	for _, c := range []struct {
		name string
		fn   func()
	}{
		{"Base64Encode", func() { sinkTextInt = simd.Base64Encode(enc, src) }},
		{"Base64Decode", func() { sinkTextInt = simd.Base64Decode(dec, enc[:n]) }},
	} {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %.0f times, want 0", c.name, a)
		}
	}
}
