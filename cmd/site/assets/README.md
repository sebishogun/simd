# Vendored assets

`datastar.js` is the [Datastar](https://github.com/starfederation/datastar)
browser bundle, v1.0.0-beta.11, MIT licensed — see
[LICENSE-datastar.md](LICENSE-datastar.md), which is Datastar's own copyright
notice and is reproduced here because MIT requires it.

It is committed rather than fetched at build time so that `go run ./cmd/site`
works with no network and nothing to install, and so that the exact bytes
served to a browser are the bytes in the repository. Nothing here is
CDN-loaded; the site has no external requests at all.

`site_test.go` checks that this file is the built bundle and not the source —
it asserts on the size and on the absence of bare relative imports, which is
what would break if someone replaced it with the unbundled entry point.
