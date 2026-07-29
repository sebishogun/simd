package simd_test

// The counts at and above the element width are the whole point of this test.
//
// A shift by the width or more is undefined in C, and x86, arm64 and LLVM each
// do something different with it. A test that stops at width-1 exercises
// nothing that was ever in doubt.

import (
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

func shiftCounts(width uint64) []uint64 {
	return []uint64{0, 1, 3, width - 1, width, width + 1, 2 * width, 255}
}

// checkShift compares every accelerated shift and rotate against the Go
// operator, at lengths straddling the dispatch threshold.
func checkShift[T simd.Integer](t *testing.T, name string, width uint64,
	rotl, rotr func(x T, s uint64) T,
) {
	t.Helper()
	r := rand.New(rand.NewPCG(5, 9))
	for _, n := range []int{0, 1, 15, 16, 17, 64, 257, 1000} {
		a := make([]T, n)
		for i := range a {
			a[i] = T(r.Uint64())
		}
		dst := make([]T, n)
		for _, s := range shiftCounts(width) {
			simd.ShlInto(dst, a, s)
			for i, v := range a {
				if want := v << s; dst[i] != want {
					t.Fatalf("%s Shl n=%d s=%d [%d]: %v << %d = %v, want %v",
						name, n, s, i, v, s, dst[i], want)
				}
			}
			simd.ShrInto(dst, a, s)
			for i, v := range a {
				if want := v >> s; dst[i] != want {
					t.Fatalf("%s Shr n=%d s=%d [%d]: %v >> %d = %v, want %v",
						name, n, s, i, v, s, dst[i], want)
				}
			}
			simd.RotlInto(dst, a, s)
			for i, v := range a {
				if want := rotl(v, s); dst[i] != want {
					t.Fatalf("%s Rotl n=%d s=%d [%d]: %v = %v, want %v",
						name, n, s, i, v, dst[i], want)
				}
			}
			simd.RotrInto(dst, a, s)
			for i, v := range a {
				if want := rotr(v, s); dst[i] != want {
					t.Fatalf("%s Rotr n=%d s=%d [%d]: %v = %v, want %v",
						name, n, s, i, v, dst[i], want)
				}
			}
		}
	}
}

func TestShiftsAndRotates(t *testing.T) {
	// The oracles rotate through the unsigned view of the same width, which is
	// what a rotate means: no sign extension, bits wrap.
	t.Run("int8", func(t *testing.T) {
		checkShift[int8](t, "int8", 8,
			func(x int8, s uint64) int8 {
				u, k := uint8(x), s%8
				if k == 0 {
					return x
				}
				return int8(u<<k | u>>(8-k))
			},
			func(x int8, s uint64) int8 {
				u, k := uint8(x), s%8
				if k == 0 {
					return x
				}
				return int8(u>>k | u<<(8-k))
			})
	})
	t.Run("uint16", func(t *testing.T) {
		checkShift[uint16](t, "uint16", 16,
			func(x uint16, s uint64) uint16 {
				k := s % 16
				if k == 0 {
					return x
				}
				return x<<k | x>>(16-k)
			},
			func(x uint16, s uint64) uint16 {
				k := s % 16
				if k == 0 {
					return x
				}
				return x>>k | x<<(16-k)
			})
	})
	t.Run("int32", func(t *testing.T) {
		checkShift[int32](t, "int32", 32,
			func(x int32, s uint64) int32 {
				u, k := uint32(x), s%32
				if k == 0 {
					return x
				}
				return int32(u<<k | u>>(32-k))
			},
			func(x int32, s uint64) int32 {
				u, k := uint32(x), s%32
				if k == 0 {
					return x
				}
				return int32(u>>k | u<<(32-k))
			})
	})
	t.Run("uint64", func(t *testing.T) {
		checkShift[uint64](t, "uint64", 64,
			func(x uint64, s uint64) uint64 {
				k := s % 64
				if k == 0 {
					return x
				}
				return x<<k | x>>(64-k)
			},
			func(x uint64, s uint64) uint64 {
				k := s % 64
				if k == 0 {
					return x
				}
				return x>>k | x<<(64-k)
			})
	})

	// The cases the C contract exists for, named explicitly so a failure says
	// what broke rather than which random element differed.
	t.Run("overWidth", func(t *testing.T) {
		u := []uint32{1, 0xffffffff, 0x80000000}
		d := make([]uint32, len(u))
		simd.ShlInto(d, u, 32)
		for i := range d {
			if d[i] != 0 {
				t.Errorf("uint32 << 32 = %#x, want 0 (x86 would give %#x)", d[i], u[i])
			}
		}
		simd.ShrInto(d, u, 40)
		for i := range d {
			if d[i] != 0 {
				t.Errorf("uint32 >> 40 = %#x, want 0", d[i])
			}
		}
		// Arithmetic right shift of a negative value saturates to -1, not 0.
		s := []int32{-1, -2, -0x80000000, 5}
		ds := make([]int32, len(s))
		simd.ShrInto(ds, s, 100)
		want := []int32{-1, -1, -1, 0}
		for i := range ds {
			if ds[i] != want[i] {
				t.Errorf("int32(%d) >> 100 = %d, want %d", s[i], ds[i], want[i])
			}
		}
	})

	// In-place must agree with the Into form.
	t.Run("inPlace", func(t *testing.T) {
		for _, s := range []uint64{0, 7, 64, 1000} {
			a := []uint64{1, 2, 3, 0xdeadbeefcafe}
			b := append([]uint64(nil), a...)
			out := make([]uint64, len(a))
			simd.RotlInto(out, a, s)
			simd.Rotl(b, s)
			for i := range b {
				if b[i] != out[i] {
					t.Fatalf("s=%d: in-place %#x, Into %#x", s, b[i], out[i])
				}
			}
		}
	})
}
