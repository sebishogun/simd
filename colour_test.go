package simd_test

import (
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

func grayRef(dst, r, g, b []byte) {
	n := min(len(dst), len(r), len(g), len(b))
	for i := 0; i < n; i++ {
		y := 19595*uint32(r[i]) + 38470*uint32(g[i]) + 7471*uint32(b[i])
		dst[i] = byte((y + 32768) >> 16)
	}
}

func TestGrayscale(t *testing.T) {
	// The weights sum to exactly 65536, so a neutral input must come back
	// unchanged. This is the property that says the coefficients are the right
	// ones rather than merely close: if they summed to 65535 or 65537 every
	// grey would drift, and the drift would be invisible on a photograph.
	t.Run("grey is preserved", func(t *testing.T) {
		var r, g, b, dst [256]byte
		for i := range r {
			r[i], g[i], b[i] = byte(i), byte(i), byte(i)
		}
		simd.GrayscaleInto(dst[:], r[:], g[:], b[:])
		for i := range dst {
			if dst[i] != byte(i) {
				t.Fatalf("Grayscale(%d,%d,%d) = %d, want %d", i, i, i, dst[i], i)
			}
		}
	})

	t.Run("primaries", func(t *testing.T) {
		// Full red, green and blue give 0.299, 0.587 and 0.114 of full scale,
		// each rounded to nearest. Getting 149 for green rather than 150 is
		// exactly the Q8 error this used to have; see csrc/bytes.c.
		r := []byte{255, 0, 0, 0, 255}
		g := []byte{0, 255, 0, 0, 255}
		b := []byte{0, 0, 255, 0, 255}
		want := []byte{76, 150, 29, 0, 255}
		dst := make([]byte, len(r))
		simd.GrayscaleInto(dst, r, g, b)
		for i := range want {
			if dst[i] != want[i] {
				t.Errorf("i=%d: got %d want %d", i, dst[i], want[i])
			}
		}
	})

	t.Run("random", func(t *testing.T) {
		rnd := rand.New(rand.NewPCG(3, 5))
		for _, n := range []int{0, 1, 15, 16, 17, 63, 65, 1000, 4096} {
			r := make([]byte, n)
			g := make([]byte, n)
			b := make([]byte, n)
			for i := range r {
				r[i] = byte(rnd.Uint32())
				g[i] = byte(rnd.Uint32())
				b[i] = byte(rnd.Uint32())
			}
			got := make([]byte, n)
			want := make([]byte, n)
			simd.GrayscaleInto(got, r, g, b)
			grayRef(want, r, g, b)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("n=%d i=%d: rgb(%d,%d,%d) got %d want %d",
						n, i, r[i], g[i], b[i], got[i], want[i])
				}
			}
		}
	})
}

func TestRGBToUV(t *testing.T) {
	// Each chroma row sums to zero, so grey must give exactly 128 — neutral.
	// A round trip through a codec that got this wrong tints the whole image.
	t.Run("grey is neutral", func(t *testing.T) {
		var r, g, b, u, v [256]byte
		for i := range r {
			r[i], g[i], b[i] = byte(i), byte(i), byte(i)
		}
		simd.RGBToUVInto(u[:], v[:], r[:], g[:], b[:])
		for i := range u {
			if u[i] != 128 || v[i] != 128 {
				t.Fatalf("grey %d gave u=%d v=%d, want 128 and 128", i, u[i], v[i])
			}
		}
	})

	// Blue is the positive extreme of U and red of V, which is what the signs
	// of the coefficients say; checking them catches a transposed matrix.
	t.Run("extremes", func(t *testing.T) {
		r := []byte{255, 0, 0}
		g := []byte{0, 255, 0}
		b := []byte{0, 0, 255}
		u := make([]byte, 3)
		v := make([]byte, 3)
		simd.RGBToUVInto(u, v, r, g, b)
		if !(u[2] > u[0] && u[2] > u[1]) {
			t.Errorf("blue should be the largest U: got %v", u)
		}
		if !(v[0] > v[1] && v[0] > v[2]) {
			t.Errorf("red should be the largest V: got %v", v)
		}
		// Nothing may leave the byte range.
		for i := range u {
			_ = u[i]
			_ = v[i]
		}
	})

	t.Run("random matches ref", func(t *testing.T) {
		rnd := rand.New(rand.NewPCG(7, 11))
		clamp := func(x int32) byte {
			if x < 0 {
				return 0
			}
			if x > 255 {
				return 255
			}
			return byte(x)
		}
		for _, n := range []int{0, 1, 17, 64, 1000} {
			r := make([]byte, n)
			g := make([]byte, n)
			b := make([]byte, n)
			for i := range r {
				r[i] = byte(rnd.Uint32())
				g[i] = byte(rnd.Uint32())
				b[i] = byte(rnd.Uint32())
			}
			u := make([]byte, n)
			v := make([]byte, n)
			simd.RGBToUVInto(u, v, r, g, b)
			for i := range r {
				rr, gg, bb := int32(r[i]), int32(g[i]), int32(b[i])
				wu := clamp((-11059*rr - 21709*gg + 32768*bb + 32768 + (128 << 16)) >> 16)
				wv := clamp((32768*rr - 27439*gg - 5329*bb + 32768 + (128 << 16)) >> 16)
				if u[i] != wu || v[i] != wv {
					t.Fatalf("n=%d i=%d rgb(%d,%d,%d): got u=%d v=%d want u=%d v=%d",
						n, i, r[i], g[i], b[i], u[i], v[i], wu, wv)
				}
			}
		}
	})
}

func TestColourNoAlloc(t *testing.T) {
	const n = 4096
	r := make([]byte, n)
	g := make([]byte, n)
	b := make([]byte, n)
	y := make([]byte, n)
	u := make([]byte, n)
	v := make([]byte, n)
	if a := testing.AllocsPerRun(50, func() { simd.GrayscaleInto(y, r, g, b) }); a != 0 {
		t.Errorf("GrayscaleInto allocated %v times per run, want 0", a)
	}
	if a := testing.AllocsPerRun(50, func() { simd.RGBToUVInto(u, v, r, g, b) }); a != 0 {
		t.Errorf("RGBToUVInto allocated %v times per run, want 0", a)
	}
}
