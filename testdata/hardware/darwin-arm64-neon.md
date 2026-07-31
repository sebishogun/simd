# darwin/arm64, tier=neon

Reported by the maintainer from a machine that cannot push, and transcribed
here from the output of `make hardware-report` rather than committed by it.
That is the ordinary path rather than an exception — the directory README asks
for the output and offers to have someone else file it, because the data is the
point and the paperwork is not.

Run against `24a0f72`, before the benchmarks moved to `internal/benchmarks`,
which is why that package is absent from the package list below.

## Machine

```
goos:   darwin
goarch: arm64
go:     go version go1.26.5 darwin/arm64
cpu:    Apple M1 Pro
os:     Darwin 25.5.0 arm64
```

## Tier selected

```
$ go run ./cmd/simdinfo
arm64 tier=neon available=[scalar neon]
```

M1 has NEON and no SVE2, so this run says nothing about the sve2 tier, which
stays emulated.

## Correctness

- accelerated (`go test ./...`): **Passed**
- portable (`GOSIMD=scalar go test ./...`): **Passed**

<details><summary>accelerated output</summary>

```
ok      github.com/sebishogun/simd                          9.011s
?       github.com/sebishogun/simd/cmd/simdinfo              [no test files]
ok      github.com/sebishogun/simd/cmd/site                 10.187s
?       github.com/sebishogun/simd/docs/examples/csvscan     [no test files]
?       github.com/sebishogun/simd/docs/examples/standardise [no test files]
?       github.com/sebishogun/simd/internal/arm64            [no test files]
ok      github.com/sebishogun/simd/internal/asmcheck         1.589s
?       github.com/sebishogun/simd/internal/backend          [no test files]
ok      github.com/sebishogun/simd/internal/conformance     69.073s
ok      github.com/sebishogun/simd/internal/cpu              1.801s
?       github.com/sebishogun/simd/internal/kernel           [no test files]
?       github.com/sebishogun/simd/internal/perf             [no test files]
ok      github.com/sebishogun/simd/internal/ref              1.513s
```

</details>

<details><summary>portable output</summary>

```
ok      github.com/sebishogun/simd                          (cached)
?       github.com/sebishogun/simd/cmd/simdinfo              [no test files]
ok      github.com/sebishogun/simd/cmd/site                 (cached)
?       github.com/sebishogun/simd/docs/examples/csvscan     [no test files]
?       github.com/sebishogun/simd/docs/examples/standardise [no test files]
?       github.com/sebishogun/simd/internal/arm64            [no test files]
ok      github.com/sebishogun/simd/internal/asmcheck        (cached)
?       github.com/sebishogun/simd/internal/backend          [no test files]
ok      github.com/sebishogun/simd/internal/conformance     (cached)
ok      github.com/sebishogun/simd/internal/cpu             (cached)
?       github.com/sebishogun/simd/internal/kernel           [no test files]
?       github.com/sebishogun/simd/internal/perf             [no test files]
ok      github.com/sebishogun/simd/internal/ref             (cached)
```

</details>

## Wall-clock

Not measured. Correctness alone moves the verification table; timing needs a
quiet machine and `make hardware-bench`.

## Anything odd

Nothing. `internal/conformance` — the differential suite that runs every tier
against the portable reference and against the other tiers — took 69 seconds
and passed, which is the part of this run that matters. It is the first time
any of these kernels has executed on an arm64 chip rather than under qemu.
