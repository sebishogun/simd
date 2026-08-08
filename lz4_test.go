package simd_test

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/sebishogun/simd"
	"github.com/sebishogun/simd/internal/ref"
)

// compressLZ4 is a minimal greedy block compressor: hash-table matches,
// four-byte minimum, correct token/extension/offset encoding, final
// sequence literals-only. It exists so the round trip is hermetic; its
// output need not be optimal, only well-formed.
func compressLZ4(src []byte) []byte {
	var out []byte
	var table [1 << 12]int
	emit := func(lits []byte, matchLen, offset int) {
		lt, mt := len(lits), 0
		if matchLen > 0 {
			mt = matchLen - 4
		}
		tok := byte(0)
		if lt >= 15 {
			tok = 15 << 4
		} else {
			tok = byte(lt) << 4
		}
		if matchLen > 0 {
			if mt >= 15 {
				tok |= 15
			} else {
				tok |= byte(mt)
			}
		}
		out = append(out, tok)
		if lt >= 15 {
			for r := lt - 15; ; r -= 255 {
				if r >= 255 {
					out = append(out, 255)
				} else {
					out = append(out, byte(r))
					break
				}
			}
		}
		out = append(out, lits...)
		if matchLen > 0 {
			out = binary.LittleEndian.AppendUint16(out, uint16(offset))
			if mt >= 15 {
				for r := mt - 15; ; r -= 255 {
					if r >= 255 {
						out = append(out, 255)
					} else {
						out = append(out, byte(r))
						break
					}
				}
			}
		}
	}
	i, lit := 0, 0
	for i+4 <= len(src) {
		h := (binary.LittleEndian.Uint32(src[i:]) * 2654435761) >> 20
		cand := table[h]
		table[h] = i
		if cand < i && i-cand <= 65535 && cand >= 0 &&
			bytes.Equal(src[cand:cand+4], src[i:i+4]) {
			ml := 4
			for i+ml < len(src) && src[cand+ml] == src[i+ml] && ml < 65535 {
				ml++
			}
			emit(src[lit:i], ml, i-cand)
			i += ml
			lit = i
		} else {
			i++
		}
	}
	emit(src[lit:], 0, 0)
	return out
}

func TestLZ4RoundTripAndDifferential(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	corpora := [][]byte{
		nil,
		[]byte("a"),
		bytes.Repeat([]byte("abc"), 5000),
		bytes.Repeat([]byte{0}, 100000),
	}
	long := make([]byte, 200000)
	for i := range long {
		long[i] = byte(rng.Intn(4)) // compressible
	}
	corpora = append(corpora, long)
	rnd := make([]byte, 65536)
	rng.Read(rnd) // incompressible
	corpora = append(corpora, rnd)
	jsonish := bytes.Repeat([]byte(`{"key":"value","n":12345},`), 3000)
	corpora = append(corpora, jsonish)

	for ci, orig := range corpora {
		comp := compressLZ4(orig)
		dst1 := make([]byte, len(orig))
		dst2 := make([]byte, len(orig))
		if n := ref.LZ4BlockDecode(dst1, comp); n != len(orig) || !bytes.Equal(dst1[:max(n, 0)], orig) {
			t.Fatalf("corpus %d: ref decode %d want %d", ci, n, len(orig))
		}
		if n := simd.LZ4BlockDecode(dst2, comp); n != len(orig) || !bytes.Equal(dst2[:max(n, 0)], orig) {
			t.Fatalf("corpus %d: kernel decode %d want %d", ci, n, len(orig))
		}
		// Mutations: kernel and reference must agree exactly, including -1,
		// and neither may write outside dst (Go bounds guard the ref; the
		// kernel is held to agreement).
		for m := 0; m < 300 && len(comp) > 0; m++ {
			mut := append([]byte(nil), comp...)
			for k := 0; k < 1+rng.Intn(3); k++ {
				mut[rng.Intn(len(mut))] = byte(rng.Intn(256))
			}
			short := mut[:rng.Intn(len(mut)+1)]
			r1 := ref.LZ4BlockDecode(dst1, short)
			r2 := simd.LZ4BlockDecode(dst2, short)
			if r1 != r2 {
				t.Fatalf("corpus %d mut %d: ref %d kernel %d", ci, m, r1, r2)
			}
			if r1 > 0 && !bytes.Equal(dst1[:r1], dst2[:r1]) {
				t.Fatalf("corpus %d mut %d: outputs differ at n=%d", ci, m, r1)
			}
		}
	}
}
