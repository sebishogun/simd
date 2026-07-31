# linux/amd64, tier=avx512

## Machine

```
goos:   linux
goarch: amd64
go:     go version go1.26.2 linux/amd64
cpu:    AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
os:     Linux 7.1.4-arch1-1 x86_64
```

## Tier selected

```
$ go run ./cmd/simdinfo
amd64 tier=avx512 available=[scalar sse2 avx2 avx512]
```

## Correctness

- accelerated (`go test ./...`): **Passed**
- portable (`GOSIMD=scalar go test ./...`): **Passed**

<details><summary>accelerated output</summary>

```
ok  	github.com/sebishogun/simd	9.235s
?   	github.com/sebishogun/simd/cmd/simdinfo	[no test files]
ok  	github.com/sebishogun/simd/cmd/site	(cached)
?   	github.com/sebishogun/simd/docs/examples/csvscan	[no test files]
?   	github.com/sebishogun/simd/docs/examples/standardise	[no test files]
ok  	github.com/sebishogun/simd/internal/amd64	(cached)
ok  	github.com/sebishogun/simd/internal/asmcheck	(cached)
?   	github.com/sebishogun/simd/internal/backend	[no test files]
ok  	github.com/sebishogun/simd/internal/conformance	(cached)
ok  	github.com/sebishogun/simd/internal/cpu	(cached)
?   	github.com/sebishogun/simd/internal/kernel	[no test files]
?   	github.com/sebishogun/simd/internal/perf	[no test files]
ok  	github.com/sebishogun/simd/internal/ref	(cached)
```

</details>

<details><summary>portable output</summary>

```
ok  	github.com/sebishogun/simd	(cached)
?   	github.com/sebishogun/simd/cmd/simdinfo	[no test files]
ok  	github.com/sebishogun/simd/cmd/site	(cached)
?   	github.com/sebishogun/simd/docs/examples/csvscan	[no test files]
?   	github.com/sebishogun/simd/docs/examples/standardise	[no test files]
ok  	github.com/sebishogun/simd/internal/amd64	(cached)
ok  	github.com/sebishogun/simd/internal/asmcheck	(cached)
?   	github.com/sebishogun/simd/internal/backend	[no test files]
ok  	github.com/sebishogun/simd/internal/conformance	(cached)
ok  	github.com/sebishogun/simd/internal/cpu	(cached)
?   	github.com/sebishogun/simd/internal/kernel	[no test files]
?   	github.com/sebishogun/simd/internal/perf	[no test files]
ok  	github.com/sebishogun/simd/internal/ref	(cached)
```

</details>

## Wall-clock

Not measured. `make hardware-bench` adds it, on a quiet machine.

## Anything odd

<Anything that surprised you, including nothing.>
