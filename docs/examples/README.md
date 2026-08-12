# Examples

Complete programs. Each one runs as it stands:

```
go run ./docs/examples/csvscan
```

| | What it shows |
|---|---|
| [`csvscan`](csvscan/) | Both halves of a real use: `IndexAll` to find the structure of each line in one pass, then the reductions and a comparison-plus-`CompressInto` filter to summarise the column. Also shows the allocation discipline — the offset buffer is passed in, so parsing a million rows allocates once. |
| [`standardise`](standardise/) | Struct-of-arrays in practice. Four passes per column, no allocation, and the contrast the tutorial argues: the same data as `[]struct{...}` puts each dimension 32 bytes apart and cannot be loaded into a vector register at all. |

Start with **[the tutorial](../tutorial.md)** if you have not written against
this library before. It is about shaping data rather than about the API — which
is the part that decides whether any of this helps you.

Use the **[API guide](../api.md)** to find an operation by task and the
**[task guides](../guide/)** for sizing, workspace, and crossover details.

Shorter examples live in [`example_test.go`](../../example_test.go) at the
repository root. Those are `Example` functions with `// Output:` comments, so
`go test` checks them on every build and pkg.go.dev renders each one beside the
function it documents. That is deliberate: a snippet in a README can drift out
of date silently, and one of these cannot.
