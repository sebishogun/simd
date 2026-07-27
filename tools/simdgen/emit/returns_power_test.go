package emit

import (
	"encoding/binary"
	"testing"

	"github.com/sebishogun/simd/tools/simdgen/target"
)

func ppc64leTarget(t *testing.T) target.Target {
	t.Helper()
	tgt, ok := target.Find(target.PPC64LE, "vsx")
	if !ok {
		t.Fatal("no ppc64le/vsx target")
	}
	if len(tgt.Epilogue) == 0 {
		t.Fatal("ppc64le has no epilogue, so returns would not be retargeted")
	}
	return tgt
}

func code(words ...uint32) []byte {
	b := make([]byte, 4*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(b[4*i:], w)
	}
	return b
}

func decode(b []byte) []uint32 {
	out := make([]uint32, len(b)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return out
}

// The encodings under test, as the ISA writes them.
const (
	blr    = 0x4e800020 // bclr 20,0    — branch always
	beqlr  = 0x4d820020 // bclr 12,2    — branch if CR0 EQ
	blelr  = 0x4c810020 // bclr 4,1     — branch if CR0 not GT
	bltlr  = 0x4d800020 // bclr 12,0    — branch if CR0 LT
	blrl   = 0x4e800021 // bclrl        — a call through the link register
	bctr   = 0x4e800420 // bcctr 20,0   — branch through the count register
	nop    = 0x60000000
	addi10 = 0x394a0001 // addi r10, r10, 1
)

func TestRetargetReturns(t *testing.T) {
	tgt := ppc64leTarget(t)

	// Three instructions, the middle one an unconditional return. It should
	// become a branch forward to just past the last word: from index 1 that is
	// (3-1)*4 = 8 bytes.
	got, err := retargetReturns(code(nop, blr, nop), tgt)
	if err != nil {
		t.Fatalf("retargetReturns: %v", err)
	}
	want := []uint32{nop, 18<<26 | 8, nop}
	for i, w := range decode(got) {
		if w != want[i] {
			t.Errorf("word %d = %#08x, want %#08x", i, w, want[i])
		}
	}
}

// TestRetargetKeepsCondition checks the part that would be easy to get wrong
// and impossible to notice: bc and bclr take BO and BI in the same bit
// positions, so a conditional return must come out as the same condition
// branching forward, not as an unconditional one.
func TestRetargetKeepsCondition(t *testing.T) {
	tgt := ppc64leTarget(t)

	body := code(beqlr, blelr, bltlr, addi10, blr)
	got, err := retargetReturns(body, tgt)
	if err != nil {
		t.Fatalf("retargetReturns: %v", err)
	}
	n := uint32(len(body) / 4)
	for i, w := range decode(got) {
		d := (n - uint32(i)) * 4
		src := binary.LittleEndian.Uint32(body[4*i:])
		switch src {
		case addi10:
			if w != addi10 {
				t.Errorf("word %d: an ordinary instruction was rewritten to %#08x", i, w)
			}
		case blr:
			if w != 18<<26|d {
				t.Errorf("word %d: blr became %#08x, want an unconditional b to +%d", i, w, d)
			}
		default:
			bo, bi := (src>>21)&0x1f, (src>>16)&0x1f
			if w != 16<<26|bo<<21|bi<<16|d {
				t.Errorf("word %d: %#08x became %#08x, want bc %d,%d,+%d", i, src, w, bo, bi, d)
			}
		}
	}
}

func TestRetargetLeavesOtherTargetsAlone(t *testing.T) {
	amd64, ok := target.Find(target.AMD64, "avx2")
	if !ok {
		t.Fatal("no amd64/avx2 target")
	}
	// Bytes that happen to decode as a ppc64le return, on a target that is not
	// ppc64le. Nothing may change.
	in := code(blr, blr)
	got, err := retargetReturns(in, amd64)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range decode(got) {
		if w != blr {
			t.Errorf("word %d = %#08x, want it untouched", i, w)
		}
	}
}

func TestCanRetargetRefusesRegisterExits(t *testing.T) {
	for _, tc := range []struct {
		name string
		body []byte
		ok   bool
	}{
		{"plain returns", code(nop, beqlr, blr), true},
		{"call through LR", code(nop, blrl, blr), false},
		{"branch through CTR", code(nop, bctr), false},
		{"truncated", []byte{0x20, 0x00, 0x80}, false},
	} {
		if got, why := canRetargetPPC64(tc.body); got != tc.ok {
			t.Errorf("%s: canRetargetPPC64 = %v (%s), want %v", tc.name, got, why, tc.ok)
		}
	}
}
