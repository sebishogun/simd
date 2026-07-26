# What Go's assembler can actually express, per architecture

*Measured against Go 1.26.2 (`GOROOT=~/.local/share/mise/installs/go/1.26.2`), July 2026.
Mnemonic counts come from `$GOROOT/src/cmd/internal/obj/<arch>/anames.go`; acceptance was
verified empirically by feeding instructions to `go tool asm`.*

This document exists because the answer determines the entire code-generation strategy. The
short version: **amd64 is excellent, arm64 is crippled, and SVE is impossible.**

## Summary table

| GOARCH | SIMD mnemonics | Verdict |
|---|---|---|
| amd64 | 668 `V*` + 161 SSE; 1863 EVEX entries | **Excellent.** Full AVX-512. |
| riscv64 | 652 `V*` | **Complete RVV 1.0.** |
| s390x | 480 `V*` | z13 vector facility (VX) + VXE. |
| loong64 | 266 LSX + 268 LASX | Both widths. |
| ppc64le | 162 VMX + 78 VSX | Good. |
| **arm64** | **66 `V*`** | **Crippled.** No float vector arithmetic at all. |
| wasm | **0** | No SIMD128 in the assembler, and no assembler path will ever exist. |

## amd64 — excellent

`cmd/internal/obj/x86/avx_optabs.go` is 266 KB / 4628 lines with **1863 EVEX entries**.
`evex.go` implements the full EVEX prefix: `{z}` masking (`evexZeroing`), `{bcst}` broadcast
(`evexBcstN4`/`N8`), `{rn-sae}` embedded rounding (`evexRoundingEnabled`), `{sae}`.

Registers: `REG_X0..X31`, `REG_Y0..`, `REG_Z0..Z31` (`a.out.go:137,139,172,203`), mask registers
`REG_K0..K7` (`a.out.go:97,104`).

Verified accepted by `go tool asm`:

```
VPADDD Z1, Z2, Z3                 // AVX-512F
VPDPBUSD Z1, Z2, Z3               // AVX512-VNNI
VGF2P8AFFINEQB $0, Z1, Z2, Z3     // GFNI
VPCOMPRESSD Z1, K1, Z2            // AVX-512 compress
```

Also present: `VPTERNLOGD`, `VPCONFLICTD`, `VP4DPWSSD`, `VAESENC`, `VPCLMULQDQ`, `VPMADD52LUQ`,
`VPERMB`, `VPMULTISHIFTQB`, `VRANGEPD`, `VFIXUPIMMPD`, `VGATHERDPD`, `VSCATTERDPD`,
`VRSQRT28PD`, `VEXP2PD`.

**Missing on amd64:** AVX512-FP16 (`VADDPH`, `VFMADD132PH`), AVX512-BF16 (`VDPBF16PS`,
`VCVTNE2PS2BF16`), AVX512-VP2INTERSECT, SHA512/SM3/SM4-NI, **AMX entirely** (no `REG_TMM` tile
registers), AVX10.

**Note:** `asm6.go` does *not* gate on `GOAMD64` — the only `buildcfg` use is `isAndroid`
(`asm6.go:2493`). The assembler will emit AVX-512 regardless of `GOAMD64=v1`. **Guarding is
entirely the library's responsibility at runtime.** This is precisely the seam where the SIGILL
bugs in [03-competitive-analysis.md](03-competitive-analysis.md) live.

## arm64 — the wall

The complete 66-item `V*` list:

```
VADD VADDP VADDV VAND VBCAX VBIF VBIT VBSL VCMEQ VCMTST VCNT VDUP VEOR VEOR3
VEXT VFMLA VFMLS VLD1 VLD1R VLD2 VLD2R VLD3 VLD3R VLD4 VLD4R VMOV VMOVD VMOVI
VMOVQ VMOVS VORR VPMULL VPMULL2 VRAX1 VRBIT VREV16 VREV32 VREV64 VSHL VSLI VSRI
VST1 VST2 VST3 VST4 VSUB VTBL VTBX VTRN1 VTRN2 VUADDLV VUADDW VUADDW2 VUMAX
VUMIN VUSHLL VUSHLL2 VUSHR VUSRA VUXTL VUXTL2 VUZP1 VUZP2 VXAR VZIP1 VZIP2
```

Plus 26 crypto-extension ops with no `V` prefix (per rule 5 of `arm64/doc.go`):
`AESD AESE AESIMC AESMC SHA1{C,H,M,P,SU0,SU1} SHA256{H,H2,SU0,SU1} SHA512{H,H2,SU0,SU1}
CRC32{B,H,W,X,CB,CH,CW,CX}`.

### What you cannot write

| Needed | Status |
|---|---|
| `VFADD` `VFSUB` `VFMUL` `VFDIV` — **float vector arithmetic** | **ALL ABSENT** |
| `VMUL` — integer vector multiply | **ABSENT** (only `VPMULL`/`VPMULL2`, polynomial, for GHASH) |
| `VCMGT` `VCMGE` `VCMHI` `VCMHS` — signed/unsigned compares | **ABSENT** (only `VCMEQ`, `VCMTST`) |
| `VSMAX` `VSMIN` — signed min/max | **ABSENT** (only unsigned `VUMAX`/`VUMIN`) |
| `VSSHR` `VSSHLL` — signed shifts | **ABSENT** (only unsigned) |
| `VABS` `VNEG` `VABD` `VADDL` `VSUBL` `VCLZ` `VMVN` `VSQXTN` `VADDHN` | **ABSENT** |
| `VSDOT` `VUDOT` `VUSDOT` — dot product | **ABSENT** |
| I8MM (`SMMLA`/`UMMLA`), BF16 (`BFDOT`/`BFMMLA`), `FCMLA`/`FCADD` | **ABSENT** |
| `DUP` from a general register | **ABSENT** — [golang/go#65310](https://github.com/golang/go/issues/65310) |

**You cannot write a NEON `float32` add in Go assembly.** Not awkwardly — the mnemonic does not
exist. Only `VFMLA`/`VFMLS` (fused multiply-accumulate) touch floats.

Note also the spellings are Go-specific, not ARM's: `VFMLA` not `FMLA`, `VADDV` not `ADDV`, and
`VUMAXV` does not exist at all despite `VUMAX` and `VADDV` both existing.

Arrangement operands that do work: `ARNG_8B/16B/4H/8H/2S/4S/1D/2D/1Q/B/H/S/D`
(`a.out.go:1045-1057`), i.e. `VADD V5.H8, V18.H8, V9.H8`.

### SVE/SVE2: entirely unsupported, and deferred upstream

`WHILELT P0.S, R0, R1` and `LD1W (Z0.S), P0/Z, [R1]` are both rejected. `grep -rnw SVE` across
all of `$GOROOT/src` returns exactly **2 hits**, both in *feature detection*
(`vendor/golang.org/x/sys/cpu/cpu_arm64.s:32`, `cpu_netbsd_arm64.go:133`). There are **no Z or P
registers** in the assembler. (The `C_ZREG` you'll find in `arm64/anames7.go` is the *zero*
register — `a.out.go:347`: `C_ZREG // R0..R30, ZR` — not an SVE Z register.)

Upstream status: [golang/go#73787](https://github.com/golang/go/issues/73787) states scalable
vectors are deferred — *"we plan to add support for scalable vectors in Go, although currently
we're not ready to propose a concrete design."* Earlier assembler-syntax CLs
([1](https://groups.google.com/g/golang-codereviews/c/R74opAeP3ck),
[2](https://groups.google.com/g/golang-codereviews/c/udibL-Wyitg)) were abandoned. Go 1.27's
release notes do not mention SVE.

Meanwhile `golang.org/x/sys/cpu` **does** expose `ARM64.HasSVE` and `HasSVE2`. You can detect it.
You just cannot emit it.

## riscv64 — best-in-class after amd64

All of `VSETVLI`, `VSETIVLI`, `VSETVL`, `VLE{8,16,32,64}V`, `VSE*V`, `VLSE*V` (strided),
`VLUXEI*V`/`VLOXEI*V` (indexed), `VLE*FFV` (fault-only-first), `VADDVV`, `VFMACCVV`,
`VREDSUMVS`, `VSLIDEUPVX`, `VMSEQVV`. Registers `REG_V0..V31` (`riscv/cpu.go:112,143`).
Validation helpers `validateRVV`/`wantVectorReg` at `riscv/obj.go:1216-1543`.

Gated by `GORISCV64=rva20u64|rva22u64|rva23u64` (`internal/buildcfg/cfg.go:315-325,449-456`);
`rva23u64` implies V.

Note `VSETVLI` requires Go's special-operand syntax — `VSETVLI X10, X11, E32, M2, TA, MA` is
rejected because `X11` is not a special operand there.

## s390x / loong64 / ppc64le

- **s390x**: `REG_V0..V31`, V0:15 aliased to F0:15 (`s390x/a.out.go:86,117,170`). 480 vector ops
  covering z13 VX + VXE. `VAF V1, V2, V3` accepted.
- **loong64**: `REG_V0..V31` and `REG_X0..X31` (`loong64/a.out.go:156,187,190,221`). Full LSX
  (`VADDB/H/W/V/Q`, `VSADD*`) and LASX (`XVADDB`) as mirrored pairs. `VADDB V1, V2, V3` accepted.
- **ppc64le**: `REG_VS0..VS63` overlapping F0-F31 and V0-V31 (`ppc64/a.out.go:187,250,323-324`).
  VMX + VSX loads `LXV`, `LXVD2X`, `LXVW4X`, `LXVB16X`, `LXVL`/`LXVLL` (length-truncated,
  POWER9), `MTVSRD`/`MFVSRD`. `VADDUWM V1, V2, V3` accepted.

## wasm — nothing

`cmd/internal/obj/wasm/anames.go` has 208 opcodes and **zero** `v128`/`i8x16`/`f32x4`. `GOWASM`
offers only `satconv` and `signext`. There is no assembler path to wasm SIMD and there never will
be — Go 1.27's `simd/archsimd` intrinsics are the only route.

## The escape hatch: raw encoding

`WORD $0x...` / `BYTE $0x..` is **accepted on every architecture**. Verified:

```
arm64    WORD $0x04a01000                              // SVE: ADD Z0.S, Z0.S, Z1.S
amd64    BYTE $0x62; BYTE $0xf1; ... (EVEX VPADDD)
riscv64  WORD $0x0e0080d7                              // RVV encoding
```

Go's own stdlib already uses **13 raw `WORD` escapes** in arm64 assembly, so there is precedent.

Cost of raw encoding: no `go vet` asmdecl checking of the body, no disassembly in Go tooling, and
the register allocator has no knowledge of what you did. Mitigations are in
[05-decisions.md](05-decisions.md).

## Load-bearing conclusion

For arm64, raw-encoded instruction bytes are not a corner-case escape hatch — they are the
**only** mechanism for essentially all numeric SIMD, NEON and SVE alike. Any pipeline for this
library must therefore emit *encodings*, not mnemonics. See
[02-codegen-pipelines.md](02-codegen-pipelines.md).

## Feature detection, per architecture

`internal/cpu` is unavailable to third-party modules. Use **`golang.org/x/sys/cpu`**, which is
also strictly richer:

| Struct | `internal/cpu` | `x/sys/cpu` |
|---|---|---|
| X86 | 34 fields, incl. `HasAVX512{F,CD,BW,DQ,VL,GFNI,VAES,VNNI,VBMI,VBMI2,BITALG,VPOPCNTDQ,VPCLMULQDQ}` | comparable |
| **ARM64** | 11 fields — **no `HasASIMD`, no `HasSVE`, no `HasSVE2`, no `HasDotProd`, no `HasI8MM`** | **`HasASIMD, HasSVE, HasSVE2, HasASIMDDP, HasI8MM, HasFPHP, HasASIMDHP, …`** |
| RISCV64 | 3 fields (`HasFastMisaligned, HasV, HasZbb`) | + `HasZba/Zbs/Zbc/Zvbb/Zvbc/Zvkb/Zvkt/Zvkg/Zvkn/Zvks…` |
| S390X | `HasVX`, `HasVXE` | — |
| Loong64 | `HasLSX`, `HasLASX` | — |
| PPC64 | `IsPOWER8/9/10` (no per-feature VSX bit — ISA level implies it) | `IsPOWER8/9` |

NEON is **not** a detection bit on arm64 because ASIMD is architecturally mandatory on AArch64 —
assume it unconditionally.

## Sources

- `$GOROOT/src/cmd/internal/obj/*/anames.go`, `avx_optabs.go`, `evex.go`, `a.out.go`
- [golang/go#73787 — simd/archsimd proposal](https://github.com/golang/go/issues/73787)
- [golang/go#65310 — no DUP in arm64 asm](https://github.com/golang/go/issues/65310)
- [golang.org/x/sys/cpu](https://pkg.go.dev/golang.org/x/sys/cpu)
- [sys: cpu: add sve2 detection](https://github.com/golang/sys/commit/77580903240cde87369d3ea876dbb47e76e48905)
