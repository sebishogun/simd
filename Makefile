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

# Two fuzz targets. FuzzDifferential drives the public API; the one under
# internal/conformance drives every generated kernel of every tier against the
# portable implementation, with the fuzzer choosing the bit patterns rather
# than a table choosing the values. For floating point that distinction is the
# whole point: the inputs that break a kernel are not large or small, they are
# a signalling NaN or a denormal at the exact exponent boundary.
.PHONY: fuzz
fuzz:
	$(GO) test -run '^$$' -fuzz FuzzDifferential -fuzztime $(or $(FUZZTIME),60s) .
	$(GO) test -run '^$$' -fuzz FuzzKernelsAgainstReference \
		-fuzztime $(or $(FUZZTIME),60s) ./internal/conformance/

# Every architecture with a backend, under emulation.
#
# This is not optional extra assurance. Three separate memory-corruption bugs
# shipped past a green amd64 suite and only appeared here: kernels clobbering
# the register the Go runtime keeps the current goroutine in, kernels writing
# into the caller's frame through a save area the s390x ABI expects the caller
# to provide, and a reference that computed different bits on architectures
# where Go fuses a multiply into an add. None of the three is visible on
# amd64, and none would have been found by reading the code.
#
# -short skips the repetition benchmarks, which measure a minimum over many
# runs and are meaningless under emulation anyway — and would take hours.
# Slow even so; allow half an hour. Run before any release, and nightly in CI.
.PHONY: test-cross
test-cross: cross-setup
	@for p in linux/arm64 linux/riscv64 linux/s390x linux/ppc64le; do \
		echo "--- $$p"; \
		$(DOCKER) run --rm --platform $$p -v "$(PWD)":/src -w /src \
			-e GOFLAGS=-buildvcs=false golang:1.26 \
			go test -short ./... || exit 1; \
	done

# Two architectures cannot be tested by the docker lane above, for two
# different reasons, and both are tested here instead: cross-compile on the
# host, then run the static binary under qemu-user directly.
#
#   loong64  has no golang image at all. The official Go images cover amd64,
#            arm64, 386, arm/v7, ppc64le, riscv64 and s390x, and that is the
#            list, so `docker run --platform linux/loong64` cannot work.
#   riscv64  has an image, and the qemu inside it emulates a CPU with no vector
#            extension. `simdinfo` there reports available=[scalar]: the whole
#            RVV backend is skipped as unexecutable and the lane passes having
#            run none of it.
#
# The second is the more dangerous shape and it is why -require-accelerated
# exists. A suite that skips every accelerated tier is green, and reads
# identically to one that tested them. Both backends were in that state; the
# first run that actually executed them found a segfault in one and wrong
# answers from every constant-reading kernel in the other.
#
# The emulator has to be recent and has to be told which CPU to be. LSX and
# LASX arrived in QEMU 8.1, and RISC-V's V extension is off unless the -cpu
# string asks for it.
QEMU_LOONG   ?= qemu-loongarch64
QEMU_RISCV   ?= qemu-riscv64
QEMU_PKGS    ?= . ./internal/conformance ./internal/ref ./internal/cpu

.PHONY: test-loong64
test-loong64:
	@$(MAKE) --no-print-directory qemu-run \
		ARCH=loong64 QEMU=$(QEMU_LOONG) CPU=la464

.PHONY: test-riscv64
test-riscv64:
	@$(MAKE) --no-print-directory qemu-run \
		ARCH=riscv64 QEMU=$(QEMU_RISCV) CPU=rv64,v=true,vlen=256,zba=true,zbb=true

# qemu-run is the shared body. It asserts an accelerated tier was selected
# before it trusts a single PASS.
.PHONY: qemu-run
qemu-run:
	@command -v $(QEMU) >/dev/null || { \
		echo "$(QEMU) not found. Extract a recent one with:"; \
		echo "  cid=\$$(docker create tonistiigi/binfmt:latest)"; \
		echo "  docker cp \$$cid:/usr/bin/$(QEMU) ~/.local/bin/"; \
		echo "  docker rm \$$cid"; \
		exit 1; }
	@bin=$$(mktemp); \
	GOARCH=$(ARCH) GOOS=linux $(GO) build -o $$bin ./cmd/simdinfo || exit 1; \
	QEMU_CPU=$(CPU) $(QEMU) $$bin -require-accelerated || exit 1; \
	rm -f $$bin
	@for p in $(QEMU_PKGS); do \
		bin=$$(mktemp); \
		GOARCH=$(ARCH) GOOS=linux $(GO) test -c -o $$bin $$p || exit 1; \
		printf "%-9s %-28s " $(ARCH) $$p; \
		QEMU_CPU=$(CPU) $(QEMU) $$bin -test.short | tail -1 || exit 1; \
		rm -f $$bin; \
	done

# Register the qemu interpreters, without which docker --platform fails with
# "exec format error" rather than anything that names the real problem.
.PHONY: cross-setup
cross-setup:
	@[ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ] || \
		$(DOCKER) run --rm --privileged multiarch/qemu-user-static --reset -p yes

# ------------------------------------------------------------------ generated
# Assert every generated .s only contains instructions permitted by the CPU
# feature tier its file is gated on — no EVEX in an _avx2.s. This is the check
# that mechanically prevents the SIGILL class of bug; see
# docs/research/05-decisions.md, D7.

.PHONY: check-emission
check-emission:
	cd tools && $(GO) run ./simdgen -n

.PHONY: codegen
codegen:
	cd tools && $(GO) run ./simdgen

# ---------------------------------------------------------------- performance
# Every performance number in this repository was measured once by hand, which
# is how they were arrived at honestly and also how they rot. `make bench-check`
# makes the remembering the machine's job: it runs the suite, compares against
# the stored baseline for this GOARCH, and fails on anything more than 25%
# slower. Regenerate a baseline with `make bench-update` and say why in the
# commit — a baseline moved without a reason is a regression with paperwork.
BENCH_BASELINE = testdata/bench/$(shell $(GO) env GOARCH).txt
BENCH_COUNT   ?= 6
BENCH_OUT     ?= /tmp/simd-bench-$(shell $(GO) env GOARCH).txt

.PHONY: bench-run
bench-run:
	$(GO) test -run '^$$' -bench . -count $(BENCH_COUNT) $(PKG) > $(BENCH_OUT)
	@echo "wrote $(BENCH_OUT)"

.PHONY: bench-check
bench-check: bench-run
	cd tools && $(GO) run ./benchcheck -baseline ../$(BENCH_BASELINE) $(BENCH_OUT)

.PHONY: bench-update
bench-update: bench-run
	cd tools && $(GO) run ./benchcheck -baseline ../$(BENCH_BASELINE) -update $(BENCH_OUT)



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
