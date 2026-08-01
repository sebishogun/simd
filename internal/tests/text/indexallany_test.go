package text

import (
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/sebishogun/simd"
)

// IndexAllAny must agree with calling IndexAll once per byte of the set and
// merging, which is the thing it replaces.
func TestIndexAllAnyMatchesPerByte(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	alphabet := "abc{}[]:,\"\\ \n"
	for _, n := range []int{0, 1, 7, 16, 17, 64, 70, 1000, 5000} {
		b := make([]byte, n)
		for i := range b {
			b[i] = alphabet[r.IntN(len(alphabet))]
		}
		for _, set := range []string{"{", "{}", "{}[]:,", "abc", "\"\\", "{}[]:,\"\\"} {
			want := make([]int32, 0, n)
			for i := 0; i < len(b); i++ {
				if strings.IndexByte(set, b[i]) >= 0 {
					want = append(want, int32(i))
				}
			}
			got := make([]int32, n+1)
			k := simd.IndexAllAny(got, b, set)
			if k != len(want) {
				t.Fatalf("n=%d set=%q: found %d, want %d", n, set, k, len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("n=%d set=%q: position %d is %d, want %d", n, set, i, got[i], want[i])
				}
			}
		}
	}
}

// A short destination bounds the work rather than being an error.
func TestIndexAllAnyTruncates(t *testing.T) {
	b := []byte(strings.Repeat("a,b,c,", 100))
	dst := make([]int32, 5)
	if k := simd.IndexAllAny(dst, b, ","); k != 5 {
		t.Errorf("got %d, want the destination length 5", k)
	}
}

// More than eight distinct bytes takes the portable path and must still be
// correct.
func TestIndexAllAnyOverEight(t *testing.T) {
	b := []byte("abcdefghijklmnop")
	dst := make([]int32, 32)
	k := simd.IndexAllAny(dst, b, "acegikmo")
	if k != 8 {
		t.Fatalf("eight-byte set: got %d, want 8", k)
	}
	k = simd.IndexAllAny(dst, b, "acegikmoq")
	if k != 8 {
		t.Fatalf("nine-byte set: got %d, want 8", k)
	}
}
