//go:build goexperiment.simd && amd64

package simd_test

import (
	"math"
	"math/rand/v2"
	"testing"

	simd "github.com/sebishogun/simd"
)

// The tail is what these helpers exist for, so every length either side of a
// block boundary is checked rather than a round number.
func TestMapFloat32x8(t *testing.T) {
	for n := 0; n <= 40; n++ {
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(i)*0.5 - 3
		}
		dst := make([]float32, n)
		simd.MapFloat32x8(dst, a, func(v simd.F32x8) simd.F32x8 { return v.Mul(v) })
		for i := range a {
			want := a[i] * a[i]
			if math.Float32bits(dst[i]) != math.Float32bits(want) {
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], want)
			}
		}
	}
}

// A partial store must not touch the elements past the end. Filling the
// destination with a sentinel first is the only way to see that: an
// over-store lands inside the same allocation and would otherwise look fine.
func TestMapFloat32x8DoesNotOverStore(t *testing.T) {
	const cap = 64
	for n := 1; n < 40; n++ {
		buf := make([]float32, cap)
		for i := range buf {
			buf[i] = -12345
		}
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(i)
		}
		simd.MapFloat32x8(buf[:n], a, func(v simd.F32x8) simd.F32x8 { return v })
		for i := n; i < cap; i++ {
			if buf[i] != -12345 {
				t.Fatalf("n=%d: element %d past the end was overwritten with %v", n, i, buf[i])
			}
		}
	}
}

func TestMapFloat64x4(t *testing.T) {
	for n := 0; n <= 20; n++ {
		a := make([]float64, n)
		for i := range a {
			a[i] = float64(i) - 7
		}
		dst := make([]float64, n)
		simd.MapFloat64x4(dst, a, func(v simd.F64x4) simd.F64x4 { return v.Add(v) })
		for i := range a {
			if dst[i] != a[i]+a[i] {
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], a[i]+a[i])
			}
		}
	}
}

func TestZipFloat32x8(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 37))
	for n := 0; n <= 40; n++ {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(r.NormFloat64())
			b[i] = float32(r.NormFloat64())
		}
		dst := make([]float32, n)
		simd.ZipFloat32x8(dst, a, b, func(x, y simd.F32x8) simd.F32x8 { return x.Mul(y) })
		for i := range a {
			want := a[i] * b[i]
			if math.Float32bits(dst[i]) != math.Float32bits(want) {
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], want)
			}
		}
	}
}

func TestZipFloat64x4(t *testing.T) {
	for n := 0; n <= 20; n++ {
		a := make([]float64, n)
		b := make([]float64, n)
		for i := range a {
			a[i], b[i] = float64(i), float64(2*i)
		}
		dst := make([]float64, n)
		simd.ZipFloat64x4(dst, a, b, func(x, y simd.F64x4) simd.F64x4 { return x.Add(y) })
		for i := range a {
			if dst[i] != a[i]+b[i] {
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, dst[i], a[i]+b[i])
			}
		}
	}
}

// Shorter of the two, like every other operation in this package.
func TestVecHelpersClampLength(t *testing.T) {
	dst := make([]float32, 5)
	a := make([]float32, 100)
	for i := range a {
		a[i] = 1
	}
	simd.MapFloat32x8(dst, a, func(v simd.F32x8) simd.F32x8 { return v })
	for i := range dst {
		if dst[i] != 1 {
			t.Fatalf("i=%d: got %v want 1", i, dst[i])
		}
	}
}

func TestLanes(t *testing.T) {
	// Whatever the CPU, the counts must be consistent with each other: a
	// float64 is twice a float32, and a byte is four times.
	f32, f64 := simd.Lanes[float32](), simd.Lanes[float64]()
	u8 := simd.Lanes[uint8]()
	if f32 == 0 || f64 == 0 || u8 == 0 {
		t.Fatalf("Lanes reported zero on a build that has the vector type: f32=%d f64=%d u8=%d", f32, f64, u8)
	}
	if f32 != 2*f64 {
		t.Errorf("Lanes[float32]=%d is not twice Lanes[float64]=%d", f32, f64)
	}
	if u8 != 4*f32 {
		t.Errorf("Lanes[uint8]=%d is not four times Lanes[float32]=%d", u8, f32)
	}
	if f32 != 4 && f32 != 8 && f32 != 16 {
		t.Errorf("Lanes[float32]=%d is not an amd64 vector width", f32)
	}
	if !simd.HasVectorType {
		t.Error("HasVectorType is false in a build that compiled vec.go")
	}
	t.Logf("Lanes: float32=%d float64=%d uint8=%d", f32, f64, u8)
}

// The escape hatch has to agree with the slice API, or one of them is wrong.
func TestVecAgreesWithSliceAPI(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 43))
	for _, n := range []int{1, 7, 8, 9, 17, 100, 1000} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(r.NormFloat64())
			b[i] = float32(r.NormFloat64())
		}
		viaVec := make([]float32, n)
		simd.ZipFloat32x8(viaVec, a, b, func(x, y simd.F32x8) simd.F32x8 { return x.Add(y) })

		viaSlice := make([]float32, n)
		simd.AddInto(viaSlice, a, b)

		for i := range viaVec {
			if math.Float32bits(viaVec[i]) != math.Float32bits(viaSlice[i]) {
				t.Fatalf("n=%d i=%d: vector path %v, slice path %v",
					n, i, viaVec[i], viaSlice[i])
			}
		}
	}
}
