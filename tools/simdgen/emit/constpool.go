package emit

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/objfile"
	"github.com/sebishogun/simd/tools/simdgen/target"
)

// Constant pools, resolved by patching rather than by re-spelling.
//
// # Why not re-spell
//
// The obvious approach is to rewrite the one instruction that reads the pool
// as a real Plan 9 mnemonic referring to a DATA symbol, and let Go's linker
// compute the displacement. That is what the first implementation did, and it
// is wrong in a way that does not show up until a kernel is long enough:
// the replacement is not the same length as the instruction it replaces, so
// every PC-relative branch spanning it now jumps to the wrong place. The
// symptom was a reciprocal kernel whose remainder loop was simply never
// entered — no crash, just untouched output at the tail.
//
// # What this does instead
//
// The constant pool is appended to the function's own body, inside the same
// TEXT symbol, and the four-byte displacement in the instruction is patched to
// point at it. Nothing changes length, so every branch that was correct in the
// object file is still correct here.
//
// That leaves alignment. clang reads constant pools with aligned moves —
// movaps and friends — because .rodata.cst16 is 16-byte aligned, and an
// aligned move faults on an address that is not. Go promises no particular
// alignment for anything inside a TEXT symbol. So the aligned move is patched
// to its unaligned counterpart, which is a single opcode byte and therefore
// also length-preserving: movaps is 0F 28 /r and movups is 0F 10 /r. They
// compute exactly the same thing; only the alignment requirement differs.

// alignedToUnaligned maps the opcode byte of an aligned SSE/AVX move to the
// unaligned form of the same instruction.
//
// Both encodings are the same length, which is the whole point: patching one
// byte cannot shift anything after it.
var alignedToUnaligned = map[byte]byte{
	0x28: 0x10, // movaps/movapd/vmovaps/vmovapd -> movups/movupd/vmovups/vmovupd
	0x29: 0x11, // the store direction, for completeness
}

// alignedLoads are the mnemonics whose opcode byte needs the patch above.
var alignedLoads = map[string]bool{
	"movaps": true, "movapd": true,
	"vmovaps": true, "vmovapd": true,
}

// alignedInteger moves are aligned too, but the unaligned form differs by the
// mandatory prefix rather than the opcode: movdqa is 66 0F 6F and movdqu is
// F3 0F 6F. Still the same length, so still patchable in place.
var alignedInteger = map[string]bool{
	"movdqa": true, "vmovdqa": true,
	"movdqa32": true, "movdqa64": true,
	"vmovdqa32": true, "vmovdqa64": true,
}

// unalignedLoads already tolerate any address and need no patch.
//
// leaq is in the list because it is not a load at all: it computes the
// address and never dereferences it, so alignment cannot apply. It was being
// rejected as "a legacy SSE instruction that requires 16-byte alignment",
// which is both wrong and the reason cbrt ran on the portable path.
// The scalar forms — the ones suffixed ss or sd — are here for the same
// reason: an alignment requirement applies to a 128-bit access, and these
// touch four or eight bytes. Only the packed legacy forms (movaps, mulps,
// pand, pxor and the rest) genuinely need the pool 16-byte aligned, and those
// are the ones that stay on the portable path.
var unalignedLoads = map[string]bool{
	"lea": true, "leaq": true, "leal": true,
	"addsd": true, "addss": true, "subsd": true, "subss": true,
	// The scalar compares, for the same reason: four or eight bytes touched,
	// so no 128-bit alignment requirement applies. Each is one instruction
	// under eight names, the predicate being an immediate.
	"cmpeqss": true, "cmpltss": true, "cmpless": true, "cmpunordss": true,
	"cmpneqss": true, "cmpnltss": true, "cmpnless": true, "cmpordss": true,
	"cmpeqsd": true, "cmpltsd": true, "cmplesd": true, "cmpunordsd": true,
	"cmpneqsd": true, "cmpnltsd": true, "cmpnlesd": true, "cmpordsd": true,
	// movd moves four bytes and movq eight, so neither carries the 128-bit
	// alignment requirement either.
	"movd":  true,
	"mulsd": true, "mulss": true, "divsd": true, "divss": true,
	"minsd": true, "minss": true, "maxsd": true, "maxss": true,
	"sqrtsd": true, "sqrtss": true,
	"ucomisd": true, "ucomiss": true, "comisd": true, "comiss": true,
	"cvtsi2sd": true, "cvtsi2ss": true, "cvtsd2ss": true, "cvtss2sd": true,
	"cvttsd2si": true, "cvttss2si": true,
	"movups": true, "movupd": true, "movss": true, "movsd": true,
	"vmovups": true, "vmovupd": true, "vmovss": true, "vmovsd": true,
	"vmovddup": true, "vbroadcastss": true, "vbroadcastsd": true,
	"vpbroadcastd": true, "vpbroadcastq": true,
}

// resolvePool rewrites a function so its constant-pool references point at a
// copy of the pool appended to the body.
//
// It returns the new body, which is the original code followed by the pools,
// and reports whether anything needed doing.
func resolvePool(fn *objfile.Func, instrs []Instr, tgt target.Target) ([]byte, error) {
	return resolvePoolExcept(fn, instrs, tgt, nil)
}

// resolvePoolExcept is resolvePool with some references left alone.
//
// The skipped ones are being re-spelled as mnemonics reading a separate RODATA
// symbol (see constpool_sse2.go), so they need neither an appended copy of
// their pool nor a patched displacement — their bytes are replaced wholesale.
// Everything else is handled exactly as before, which is what lets the two
// mechanisms live in the same function.
func resolvePoolExcept(fn *objfile.Func, instrs []Instr, tgt target.Target,
	skip []poolInstr) ([]byte, error) {

	skipped := make(map[uint64]bool, len(skip))
	for _, r := range skip {
		// The relocation offset is carried rather than derived from the
		// instruction's length. Deriving it as "four bytes from the end" is
		// right for most of the family and wrong for the compares, which put
		// a predicate byte after the displacement — and being wrong meant the
		// appended-pool path still tried to patch an instruction that was
		// about to be replaced, which is how this was found.
		skipped[r.relOff] = true
	}
	switch tgt.Arch {
	case target.AMD64:
		// Handled below: one RIP-relative displacement per reference.
	case target.ARM64:
		return resolvePoolARM64(fn)
	case target.S390X:
		return resolvePoolS390X(fn)
	case target.LOONG64:
		return resolvePoolLoong64(fn)
	case target.RISCV64:
		return resolvePoolRISCV64(fn)
	default:
		return nil, fmt.Errorf("constant pools are not resolved on %s; the address is "+
			"built from a high/low instruction pair that this generator does not yet "+
			"rewrite", tgt.Arch)
	}

	code := append([]byte(nil), fn.Code...)
	// Each distinct pool section is appended once and shared by every
	// reference into it.
	appended := map[string]uint64{}
	var tail []byte

	for _, rel := range fn.Relocs {
		if isSelfRelative(rel.TypeName) || skipped[rel.Off] {
			continue
		}
		if len(rel.Target) == 0 {
			return nil, fmt.Errorf("relocation at +0x%x refers to %s, which is not a "+
				"constant pool this generator can copy", rel.Off, rel.Sym)
		}
		if rel.Off+4 > uint64(len(code)) {
			return nil, fmt.Errorf("relocation at +0x%x runs past the end of the body", rel.Off)
		}

		base, seen := appended[rel.TargetSection]
		if !seen {
			base = uint64(len(code)) + uint64(len(tail))
			tail = append(tail, rel.Target...)
			appended[rel.TargetSection] = base
		}

		// The instruction reading the pool must tolerate an unaligned address,
		// because nothing guarantees the alignment of bytes inside a TEXT
		// symbol. Patching the opcode is length-preserving; re-spelling the
		// instruction would not be.
		if err := makeUnaligned(code, rel, instrs); err != nil {
			return nil, err
		}

		// R_X86_64_PC32: the displacement is measured from the end of the
		// four-byte field, which is why the addend is -4.
		targetOff := base + rel.TargetOff
		disp := int64(targetOff) - int64(rel.Off+4)
		if disp < -(1<<31) || disp >= 1<<31 {
			return nil, fmt.Errorf("constant pool is %d bytes away, out of range for a "+
				"32-bit displacement", disp)
		}
		binary.LittleEndian.PutUint32(code[rel.Off:], uint32(int32(disp)))
	}
	return append(code, tail...), nil
}

// makeUnaligned patches an aligned vector move to its unaligned counterpart.
//
// For a RIP-relative operand the ModRM byte is 0x05 with the register in bits
// 3-5, and it sits immediately before the four-byte displacement. The opcode
// is the byte before that. Both facts are fixed by the encoding, so the opcode
// can be located from the relocation offset alone.
func makeUnaligned(code []byte, rel objfile.Reloc, instrs []Instr) error {
	in, ok := instrAt(instrs, rel.Off)
	if !ok {
		return fmt.Errorf("no disassembly for the instruction at +0x%x", rel.Off)
	}
	switch {
	case alignedLoads[in.Mnemonic], alignedInteger[in.Mnemonic]:
		// Handled below: these keep an alignment requirement even under VEX.
	case strings.HasPrefix(in.Mnemonic, "v"):
		// VEX and EVEX encodings drop the alignment requirement on memory
		// operands. Only the explicitly-aligned moves — vmovaps, vmovapd,
		// vmovdqa — still demand it, and those are handled above. So a
		// v-prefixed arithmetic instruction reading the pool, such as
		// vandps for a sign mask, needs no patch at all.
		return nil
	case unalignedLoads[in.Mnemonic]:
		return nil
	}
	switch {
	case unalignedLoads[in.Mnemonic]:
		return nil
	case alignedLoads[in.Mnemonic]:
		if rel.Off < 2 {
			return fmt.Errorf("%s at +0x%x is too close to the start to patch",
				in.Mnemonic, rel.Off)
		}
		op := rel.Off - 2
		repl, ok := alignedToUnaligned[code[op]]
		if !ok {
			return fmt.Errorf("%s at +0x%x has opcode 0x%02x, which has no unaligned "+
				"counterpart in the table", in.Mnemonic, rel.Off, code[op])
		}
		code[op] = repl
		return nil
	case alignedInteger[in.Mnemonic]:
		return makeIntegerUnaligned(code, rel, in)
	}
	return fmt.Errorf("instruction %q at +0x%x reads a constant pool and is neither a "+
		"known aligned nor unaligned move. Add it to alignedLoads or unalignedLoads in "+
		"emit/constpool.go, after checking that the unaligned form is the same length",
		in.Mnemonic, rel.Off)
}

// makeIntegerUnaligned turns movdqa into movdqu, or the VEX-encoded
// vmovdqa into vmovdqu.
//
// For the legacy encoding the difference is the mandatory prefix — 66 becomes
// F3 — which sits before the two-byte 0F escape and any REX byte. For a VEX
// encoding the same choice lives in the two-bit pp field of the last VEX byte,
// where 01 means 66 and 10 means F3. Both are single-byte edits, so neither
// shifts anything after it.
//
// Every edit is checked against what the byte currently holds. A patch that
// guessed wrong would produce a different, valid instruction, and the failure
// would be a wrong number rather than a crash.
func makeIntegerUnaligned(code []byte, rel objfile.Reloc, in Instr) error {
	if rel.Off < 4 {
		return fmt.Errorf("%s at +0x%x is too close to the start to patch",
			in.Mnemonic, rel.Off)
	}
	// [ ... 0F 6F ModRM disp32 ] — the opcode is two bytes before the
	// displacement, and the 0F escape one before that.
	if code[rel.Off-2] != 0x6F || code[rel.Off-3] != 0x0F {
		// Not the legacy form; try VEX, whose last prefix byte sits where the
		// 0F escape would otherwise be.
		return makeVEXUnaligned(code, rel, in)
	}
	// The 66 prefix sits four bytes before the displacement, or five when a
	// REX byte separates them:
	//
	//	66       0F 6F ModRM disp32   -> 66 at Off-4
	//	66 REX   0F 6F ModRM disp32   -> 66 at Off-5
	//
	// A REX byte is 0x4_, which cannot be confused with 0x66, so one look
	// decides which layout this is. Only xmm8 and above need the REX, which
	// is why this went unnoticed until a kernel used enough registers — the
	// count-byte reduction is the first one that does.
	i := rel.Off - 4
	if code[i]&0xF0 == 0x40 {
		if rel.Off < 5 {
			return fmt.Errorf("%s at +0x%x: a REX byte but no room for the 66 prefix",
				in.Mnemonic, rel.Off)
		}
		i = rel.Off - 5
	}
	if code[i] != 0x66 {
		return fmt.Errorf("%s at +0x%x: expected a 66 prefix at +0x%x, found 0x%02x",
			in.Mnemonic, rel.Off, i, code[i])
	}
	code[i] = 0xF3
	return nil
}

// makeVEXUnaligned flips the pp field of a VEX or EVEX prefix from 66 to F3.
//
// All three prefix forms end the same way — the opcode two bytes before the
// displacement and the ModRM one — so the position of pp is found by walking
// backwards from there:
//
//	C5 P0          6F ModRM disp32   2-byte VEX,  pp in P0 at Off-3
//	C4 P0 P1       6F ModRM disp32   3-byte VEX,  pp in P1 at Off-3
//	62 P0 P1 P2    6F ModRM disp32   EVEX,        pp in P1 at Off-4
//
// EVEX is the one that does not follow the pattern, because its third payload
// byte holds the mask register and the vector length rather than the prefix
// selector. Reading Off-3 on an EVEX instruction lands on that byte instead,
// whose low bits are the aaa mask field — which is zero for an unmasked move,
// so the symptom is "pp=0" rather than a wrong patch. That is what
// vmovdqa64 does, and it is why this is checked by form rather than assumed.
func makeVEXUnaligned(code []byte, rel objfile.Reloc, in Instr) error {
	pp, err := ppOffset(code, rel.Off)
	if err != nil {
		return fmt.Errorf("%s at +0x%x: %w", in.Mnemonic, rel.Off, err)
	}
	const ppMask = 0x03
	switch code[pp] & ppMask {
	case 0x01: // 66
		code[pp] = code[pp]&^ppMask | 0x02 // F3
		return nil
	case 0x02: // already F3
		return nil
	}
	return fmt.Errorf("%s at +0x%x: prefix byte 0x%02x at +0x%x has pp=%d, which is "+
		"neither 66 nor F3", in.Mnemonic, rel.Off, code[pp], pp, code[pp]&ppMask)
}

// ppOffset locates the prefix byte holding pp for the instruction whose
// displacement begins at off.
//
// Each form is identified by its leading byte at the offset that form would
// put it at, and EVEX is confirmed by two bits the encoding fixes: P0[3:2] are
// reserved zero and P1[2] is reserved one. Checking those matters because 0x62
// is also a perfectly ordinary byte to find inside a preceding instruction,
// and a false positive here would silently patch the wrong byte — turning a
// load into a different valid load, which fails as a wrong answer rather than
// as a crash.
func ppOffset(code []byte, off uint64) (uint64, error) {
	if off >= 6 && code[off-6] == 0x62 && code[off-5]&0x0C == 0 && code[off-4]&0x04 != 0 {
		return off - 4, nil // EVEX: 62 P0 P1 P2
	}
	if off >= 5 && code[off-5] == 0xC4 {
		return off - 3, nil // 3-byte VEX: C4 P0 P1
	}
	if off >= 4 && code[off-4] == 0xC5 {
		return off - 3, nil // 2-byte VEX: C5 P0
	}
	return 0, fmt.Errorf("no VEX (C5/C4) or EVEX (62) prefix found before the opcode; " +
		"this is not a form makeVEXUnaligned knows how to patch")
}

// instrAt finds the instruction containing a byte offset.
func instrAt(instrs []Instr, off uint64) (Instr, bool) {
	for i, in := range instrs {
		end := ^uint64(0)
		if i+1 < len(instrs) {
			end = instrs[i+1].Offset
		}
		if off >= in.Offset && off < end {
			return in, true
		}
	}
	return Instr{}, false
}
