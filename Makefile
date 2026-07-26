# Build and verification targets.
#
# Nothing here is required to *use* the library — consumers just `go get`. The
# generated assembly is committed. These targets are for contributors, and for
# CI, which runs `make verify`.

GO      ?= go
PKG     ?= ./...
DOCKER  ?= docker

# Tiers this host can actually execute, discovered at runtime rather than
# assumed, so `make test-tiers` does the right thing on any machine.
TIERS := $(shell $(GO) run ./cmd/simdinfo -tiers 2>/dev/null || echo scalar)

.PHONY: all
all: verify

# ---------------------------------------------------------------- correctness

.PHONY: verify
verify: fmt-check vet test test-purego test-tiers
	@echo "OK"

.PHONY: test
test:
	$(GO) test $(PKG)

# The reference must be self-consistent before anything is compared against it.
.PHONY: test-purego
test-purego:
	$(GO) test -tags purego $(PKG)

# Run the whole suite once per instruction-set tier this CPU supports. This is
# what catches a kernel that is correct on AVX-512 and wrong on SSE2.
.PHONY: test-tiers
test-tiers:
	@for t in $(TIERS); do \
		echo "--- GOSIMD=$$t"; \
		GOSIMD=$$t $(GO) test $(PKG) || exit 1; \
	done

.PHONY: test-race
test-race:
	$(GO) test -race $(PKG)

.PHONY: fuzz
fuzz:
	$(GO) test -run '^$$' -fuzz FuzzDifferential -fuzztime $(or $(FUZZTIME),60s) .

# Every architecture with a backend, under emulation. Slow; CI runs it nightly
# rather than per pull request.
.PHONY: test-cross
test-cross:
	@for p in linux/arm64 linux/riscv64 linux/s390x linux/ppc64le; do \
		echo "--- $$p"; \
		$(DOCKER) run --rm --platform $$p -v "$(PWD)":/src -w /src \
			golang:1.26 go test $(PKG) || exit 1; \
	done

# ------------------------------------------------------------------ generated
# Assert every generated .s only contains instructions permitted by the CPU
# feature tier its file is gated on — no EVEX in an _avx2.s. This is the check
# that mechanically prevents the SIGILL class of bug; see
# docs/research/05-decisions.md, D7.

.PHONY: check-emission
check-emission:
	$(GO) run ./tools/cmd/checkemission ./internal/...

.PHONY: codegen
codegen:
	cd tools && $(GO) run ./cmd/simdgen -out ../internal

# ---------------------------------------------------------------- performance

.PHONY: bench
bench:
	$(GO) test -run '^$$' -bench . -benchmem $(PKG)

# Accelerated versus portable Go, same benchmarks, compared properly.
# Requires: go install golang.org/x/perf/cmd/benchstat@latest
.PHONY: benchcmp
benchcmp:
	$(GO) test -run '^$$' -bench . -count 10 $(PKG) > /tmp/simd-asm.txt
	$(GO) test -run '^$$' -bench . -count 10 -tags purego $(PKG) > /tmp/simd-pure.txt
	benchstat /tmp/simd-pure.txt /tmp/simd-asm.txt

# ---------------------------------------------------------------------- hygiene

.PHONY: fmt
fmt:
	$(GO) fmt $(PKG)

.PHONY: fmt-check
fmt-check:
	@out=$$(gofmt -l . | grep -v '^tools/goat/' || true); \
	if [ -n "$$out" ]; then echo "not gofmt'd:"; echo "$$out"; exit 1; fi

.PHONY: vet
vet:
	$(GO) vet $(PKG)

.PHONY: tidy
tidy:
	$(GO) mod tidy
	cd tools && $(GO) mod tidy

.PHONY: clean
clean:
	$(GO) clean -testcache
