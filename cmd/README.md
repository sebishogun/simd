# cmd

| | |
|---|---|
| [`simdinfo/`](simdinfo) | prints the instruction-set tier actually selected on this machine. Every test lane runs it with `-require-accelerated` first, because a suite that executed no accelerated code looks exactly like one that did — that mistake cost this project two backends. |
| [`site/`](site) | a local benchmark site. `go run ./cmd/site` serves the `docs/tutorial.md` comparisons live and reports the minimum of several samples with the detected tier and load average beside them. Nothing contacts a CDN; Datastar is vendored in `site/assets/`. |
