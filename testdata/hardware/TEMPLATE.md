# <goos>/<goarch>, <CPU name>

## Machine

- CPU:
- Cores / threads:
- OS and kernel:
- Go version:

## Tier selected

```
$ go run ./cmd/simdinfo
<paste the output>
```

## Correctness

```
$ go test ./...
<paste the last few lines, or the whole failure>
```

```
$ GOSIMD=scalar go test ./...
<the portable path, which must also pass>
```

## Wall-clock (optional)

Benchmarks need a quiet machine — close everything else first. `-count 6`
because the minimum is what gets used, not the mean; benchmark noise is
one-sided, so the fastest run is the one least interfered with.

```
$ go test -run '^$' -bench . -count 6 ./... > report.txt
<attach or paste report.txt>
```

## Anything odd

<Anything that surprised you, including nothing.>
