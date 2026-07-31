# internal/benchmarks

Every benchmark in the repository, except a handful at the root that need
test-only hooks the package boundary hides.

They are here rather than beside the code for two reasons. They are a **gate**,
not a test — `make bench-check` compares them against a recorded baseline — and
they only ever call the public API, so a package boundary costs nothing. Being
under `internal/` keeps them off pkg.go.dev.

```
make bench-run     # run them, write the raw output
make bench-check   # and compare against testdata/bench/<goarch>.txt
```

`benchcheck` matches on benchmark **name**, not package path, so moving these
files did not invalidate the recorded baseline.

**A benchmark needs a quiet machine to mean anything.** Close everything else.
`-count 6` or more, because the minimum is what is used rather than the mean:
benchmark noise is one-sided, so the fastest run is the one least interfered
with. Run it twice and trust only what both runs agree on. Entry 48 of
[`docs/wrong.md`](../../docs/wrong.md) cost twenty-one phantom regressions to
learn that.
