# testdata

| | |
|---|---|
| [`bench/`](bench) | the recorded benchmark baseline, one file per `GOARCH`. `make bench-check` compares against it; the threshold is deliberately wide, to block a real regression rather than police noise. |
| [`hardware/`](hardware) | one report per machine that has run the suite on real silicon. This is what the README's verification table is built from, and the contribution the project most needs. See its README. |
