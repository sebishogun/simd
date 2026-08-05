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
all: verify ## The default: everything that must pass before a commit

# An interactive picker over everything below, which checks each target
# against the machine it is being shown on.
#
# A third of these targets cannot run on any given machine — the qemu lanes
# need five emulators and a Linux host, the cross lane needs docker with
# binfmt, codegen needs clang and llvm-objdump, perf-model needs llvm-mca —
# and there is otherwise no way to tell which is which except by running them
# and reading the failures. `make menu` marks each one and says what is
# missing, or that the platform cannot do it at all.
#
# It uses fzf when present and falls back to a numbered list when not, because
# the point is that this works wherever the library does.
.PHONY: menu
menu: ## Interactive picker: every target, marked for THIS machine
	@bash scripts/menu.sh

.PHONY: targets
targets: ## One line per target, for grep and for scripts
	@bash scripts/menu.sh --list | column -t -s "$$(printf '\t')"

# ---------------------------------------------------------------- correctness

.PHONY: verify
verify: fmt-check vet test test-purego test-vec test-tiers ## fmt-check, vet, test, test-purego, test-vec, test-tiers
	@echo "OK"

.PHONY: test
test: ## Run the test suite on the host
	$(GO) test $(PKG)

# The reference must be self-consistent before anything is compared against it.
.PHONY: test-purego
test-purego: ## Run the suite against the portable reference (-tags purego)
	$(GO) test -tags purego $(PKG)

# The vector type in vec.go is behind GOEXPERIMENT=simd, so the default build
# never compiles it. A build tag nothing exercises is the vacuously-green lane
# docs/wrong.md entry 41 is about: it looks covered and is not.
#
# simd/archsimd is amd64-only in Go 1.26, so everywhere else this compiles
# vec_stub.go instead — which is still worth running, because the stub is what
# five of the six architectures get and it has to keep building.
.PHONY: test-vec
test-vec: ## Run the suite with GOEXPERIMENT=simd, which compiles vec.go
	@if GOEXPERIMENT=simd $(GO) list ./... >/dev/null 2>&1; then \
		GOEXPERIMENT=simd $(GO) test $(PKG); \
	else \
		echo "  skipping: this toolchain has no simd experiment"; \
	fi

# Run the whole suite once per instruction-set tier this CPU supports. This is
# what catches a kernel that is correct on AVX-512 and wrong on SSE2.
.PHONY: test-tiers
test-tiers: ## Run the suite once per instruction-set tier this CPU has
	@for t in $(TIERS); do \
		echo "--- GOSIMD=$$t"; \
		GOSIMD=$$t $(GO) test $(PKG) || exit 1; \
	done

.PHONY: test-race
test-race: ## Full suite under the race detector
	$(GO) test -race $(PKG)

# Two fuzz targets. FuzzDifferential drives the public API; the one under
# internal/conformance drives every generated kernel of every tier against the
# portable implementation, with the fuzzer choosing the bit patterns rather
# than a table choosing the values. For floating point that distinction is the
# whole point: the inputs that break a kernel are not large or small, they are
# a signalling NaN or a denormal at the exact exponent boundary.
.PHONY: fuzz
fuzz: ## Fuzz the public API and the tier-vs-reference differential
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
# -vet=off is load-bearing, not a shortcut. `go test` runs vet automatically,
# and under qemu-user the vet subprocess exits without being reaped: it becomes
# a zombie and the parent blocks in do_wait forever. The lane does not fail, it
# hangs — 32 minutes at 0.1% CPU with a `[vet] <defunct>` child was how this was
# found. Nothing is lost by disabling it here, because vet has already run
# natively as part of `make verify` and its findings do not depend on GOARCH.
#
# -short skips the repetition benchmarks, which measure a minimum over many
# runs and are meaningless under emulation anyway — and would take hours.
# Slow even so; allow half an hour. Run before any release, and nightly in CI.
#
# Each lane asserts an accelerated tier was selected before it believes a PASS,
# for the reason spelled out under QEMU_PKGS below: a suite that skipped every
# accelerated tier is green and reads exactly like one that tested them. The
# three lanes here reach sve2, vxe and vsx respectively, so a run that comes
# back reporting scalar means the emulator changed under us, not that the code
# is fine.
#
# riscv64 is absent deliberately. It has an official image, but the qemu inside
# it emulates a CPU with no vector extension, so the lane can only ever report
# scalar and would verify nothing. It is covered by test-riscv64 instead.
#
# CGO_ENABLED=0 is also load-bearing. The image defaults it on, and the first
# package to pull in net/http — cmd/site — made the container's gcc die with
# "internal compiler error: Segmentation fault" compiling net's resolver under
# emulation. Turning cgo off is not a workaround for that: this library
# promises to need no C toolchain, so the cross lane testing it without one is
# the stronger check, and the lane should never have had cgo available.
#
# CROSS_PKGS is the library and nothing else, and that is the second half of
# the same problem. With cgo off, building cmd/site under emulation stopped
# segfaulting and started *hanging* — the container sits at 0% CPU forever,
# somewhere in net/http's dependency graph, and `make test-cross` never
# returns. Naming the packages explicitly is not a way of avoiding a failing
# test: cmd/site is a benchmark server that runs on the developer's machine,
# it contains no kernel and no dispatch, and there is nothing about it that
# an emulated s390x can tell us. The library is . and ./internal/..., and that
# is what this lane exists to check.
CROSS_PKGS = . ./internal/...

.PHONY: test-cross
test-cross: cross-setup ## Every architecture with a backend, under docker + qemu
	@for p in linux/arm64 linux/s390x linux/ppc64le; do \
		echo "--- $$p"; \
		$(DOCKER) run --rm --platform $$p -v "$(PWD)":/src -w /src \
			-e GOFLAGS=-buildvcs=false -e CGO_ENABLED=0 golang:1.26 \
			sh -c 'go run ./cmd/simdinfo -require-accelerated && \
			       go test -short -vet=off -timeout 600s $(CROSS_PKGS)' \
			|| exit 1; \
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

# Every qemu invocation below passes the binary path TWICE, and that is not a
# typo.
#
# These emulators are extracted from the binfmt images, so they follow the
# binfmt_misc "preserve argv[0]" calling convention: the kernel invokes the
# interpreter as `qemu-ARCH <path> <argv0> <args...>`, and the binary therefore
# consumes its first argument as the guest's argv[0]. Run by hand as
# `qemu-ARCH ./x.test -test.short`, the guest sees argv[0]="-test.short" and an
# empty argument list. Flags do not fail — they vanish. A Go test binary with no
# flags runs its full default suite and prints PASS, so the lane stays green
# while testing something other than what was asked for.
#
# That cost the -require-accelerated assertion, which is the guard against a
# vacuously green lane and had itself never once been evaluated. Repeating the
# path puts a throwaway in the argv[0] slot and every real flag lands where the
# guest expects it. qemu-run-probe proves it still does before any lane trusts
# a PASS; without that, a future emulator built the other way would silently
# restore the same blindness.

.PHONY: test-loong64
test-loong64: ## Full suite on loong64 LASX under qemu
	@$(MAKE) --no-print-directory qemu-run \
		ARCH=loong64 QEMU=$(QEMU_LOONG) CPU=la464

.PHONY: test-riscv64
test-riscv64: ## Full suite on riscv64 RVV under qemu
	@$(MAKE) --no-print-directory qemu-run \
		ARCH=riscv64 QEMU=$(QEMU_RISCV) CPU=rv64,v=true,vlen=256,zba=true,zbb=true

# test-gates runs the same binaries on a CPU that LACKS the vector extension.
#
# This is the half of hardware verification that does not need hardware, and it
# is the half that catches the bug this library was designed against. Every
# other emulated lane runs a CPU with everything switched on, so a kernel that
# is gated on a feature it does not actually require, or worse gated on one
# feature and built from another, is selected and runs fine. The failure only
# appears on a machine that is missing the feature — as SIGILL, at the first
# call, in production.
#
# So: no v extension on riscv64. The dispatcher must fall back to the portable
# path and the whole suite must still pass. -require-accelerated is deliberately
# NOT passed here; selecting a scalar tier is the correct outcome and asserting
# otherwise would invert the test.
.PHONY: test-gates
test-gates: ## riscv64 with NO vector unit: the fallback must still pass
	@echo "--- riscv64 with no V extension: must fall back and still pass"
	@$(MAKE) --no-print-directory qemu-run-plain \
		ARCH=riscv64 QEMU=$(QEMU_RISCV) CPU=rv64

# qemu-run-plain is qemu-run without the accelerated-tier assertion, for the
# gate lanes where the portable path is the expected answer.
.PHONY: qemu-run-plain
qemu-run-plain:
	@command -v $(QEMU) >/dev/null || { echo "$(QEMU) not found"; exit 1; }
	@$(MAKE) --no-print-directory qemu-run-probe ARCH=$(ARCH) QEMU=$(QEMU) CPU=$(CPU)
	@bin=$$(mktemp); \
	GOARCH=$(ARCH) GOOS=linux $(GO) build -o $$bin ./cmd/simdinfo || exit 1; \
	printf "%-9s tier: " $(ARCH); \
	QEMU_CPU=$(CPU) $(QEMU) $$bin $$bin || exit 1; \
	rm -f $$bin
	@for p in $(QEMU_PKGS); do \
		bin=$$(mktemp); \
		GOARCH=$(ARCH) GOOS=linux $(GO) test -c -o $$bin $$p || exit 1; \
		printf "%-9s %-28s " $(ARCH) $$p; \
		QEMU_CPU=$(CPU) $(QEMU) $$bin $$bin -test.short | tail -1 || exit 1; \
		rm -f $$bin; \
	done

# qemu-run-probe asserts that a flag passed to a guest binary is actually seen
# by it. -argv0-probe exits 7 and does nothing else, so a 0 here means the flag
# was swallowed and every assertion in the lanes below is decoration.
.PHONY: qemu-run-probe
qemu-run-probe:
	@bin=$$(mktemp); \
	GOARCH=$(ARCH) GOOS=linux $(GO) build -o $$bin ./cmd/simdinfo || exit 1; \
	QEMU_CPU=$(CPU) $(QEMU) $$bin $$bin -argv0-probe >/dev/null 2>&1; \
	rc=$$?; rm -f $$bin; \
	[ $$rc -eq 7 ] || { \
		echo "$(QEMU): flags are not reaching the guest (probe exited $$rc, want 7)."; \
		echo "This emulator handles argv[0] differently from the binfmt builds"; \
		echo "these lanes assume. Every -test.run, -test.short and"; \
		echo "-require-accelerated below would be silently discarded, and the"; \
		echo "lane would pass having tested something else. Fix the invocation"; \
		echo "in the Makefile before trusting any result from it."; \
		exit 1; }

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
	@$(MAKE) --no-print-directory qemu-run-probe ARCH=$(ARCH) QEMU=$(QEMU) CPU=$(CPU)
	@bin=$$(mktemp); \
	GOARCH=$(ARCH) GOOS=linux $(GO) build -o $$bin ./cmd/simdinfo || exit 1; \
	QEMU_CPU=$(CPU) $(QEMU) $$bin $$bin -require-accelerated || exit 1; \
	rm -f $$bin
	@for p in $(QEMU_PKGS); do \
		bin=$$(mktemp); \
		GOARCH=$(ARCH) GOOS=linux $(GO) test -c -o $$bin $$p || exit 1; \
		printf "%-9s %-28s " $(ARCH) $$p; \
		QEMU_CPU=$(CPU) $(QEMU) $$bin $$bin -test.short | tail -1 || exit 1; \
		rm -f $$bin; \
	done

# Register the qemu interpreters, without which docker --platform fails with
# "exec format error" rather than anything that names the real problem.
.PHONY: cross-setup
cross-setup: ## Register qemu interpreters so docker --platform works
	@[ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ] || \
		$(DOCKER) run --rm --privileged multiarch/qemu-user-static --reset -p yes

# ------------------------------------------------------------------ generated
# Assert every generated .s only contains instructions permitted by the CPU
# feature tier its file is gated on — no EVEX in an _avx2.s. This is the check
# that mechanically prevents the SIGILL class of bug; see
# docs/research/05-decisions.md, D7.

.PHONY: check-emission
check-emission: ## Dry-run codegen: report what each target would emit or skip
	cd tools && $(GO) run ./simdgen -n

# perf-model estimates kernel throughput on the architectures this machine
# cannot time.
#
# Every benchmark in this repository runs on amd64, because that is the only
# architecture here that executes natively. qemu emulates semantics and not
# timing, so the emulated lanes prove the other five are CORRECT and say
# nothing about whether they are FAST — which is most of the point of a SIMD
# library. llvm-mca closes that with LLVM's own scheduling tables.
#
# It is a model: L1-resident, one core, no memory system. Read the ratios, not
# the cycle counts. Validated against measured amd64 to within 5-12% on the
# avx512-versus-avx2 comparison; see the package comment.
.PHONY: perf-model
perf-model: ## Model kernel throughput on architectures this machine cannot time
	@command -v llvm-mca >/dev/null || { \
		echo "llvm-mca not found; it ships with clang, which codegen already needs"; \
		exit 1; }
	@cd tools && $(GO) run ./perfmodel $(PERFMODEL_ARGS)

.PHONY: codegen
codegen: ## Regenerate every kernel from csrc for all six architectures
	cd tools && $(GO) run ./simdgen
	cd tools && $(GO) run ./simdgen/thresholds ..

.PHONY: gen-thresholds
gen-thresholds: ## Regenerate the KernelThreshold tables alone (no clang needed)
	cd tools && $(GO) run ./simdgen/thresholds ..

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

# Benchmarks are pinned to one L3 domain so that other work on this machine can
# run at the same time without landing in the same cache.
#
# This box is 16 cores and 32 threads across two CCXs with 32 MiB of L3 each.
# Sharing a domain is what actually perturbs a measurement here — most of these
# kernels are memory-bound, so a compile on a neighbouring core evicts the
# working set rather than merely competing for issue slots. Two runs were thrown
# away earlier to exactly that, once from a stray `go vet` and once from two
# benchmark runs overlapping.
#
# Pinning does not isolate DRAM bandwidth, which is shared whatever happens, so
# a heavy neighbour can still move the largest sizes. It removes the L3 and
# core contention, which is most of it.
#
# Override with BENCH_PIN= to disable, or BENCH_PIN='taskset -c 0-3' to narrow.
#
# taskset is Linux-only. macOS has no supported way to pin a process to a core
# at all — thread_policy_set's affinity hints are advisory and ignored on
# Apple Silicon — so there the default is empty and the answer to a noisy
# measurement is a quiet machine rather than a flag. Leaving `taskset` in the
# default here would simply make `make bench` fail to start on a Mac, which is
# a worse first experience than an unpinned run.
ifeq ($(shell uname -s),Linux)
BENCH_PIN ?= taskset -c 0-7
else
BENCH_PIN ?=
endif

.PHONY: bench-run
bench-run: ## Run the benchmarks and write the raw output
	$(BENCH_PIN) $(GO) test -run '^$$' -bench . -count $(BENCH_COUNT) $(PKG) > $(BENCH_OUT)
	@echo "wrote $(BENCH_OUT)"

.PHONY: bench-check
bench-check: bench-run ## Benchmark and compare against the recorded baseline
	cd tools && $(GO) run ./benchcheck -baseline ../$(BENCH_BASELINE) $(BENCH_OUT)

.PHONY: bench-update
bench-update: bench-run ## Re-record the baseline (read the warning first)
	cd tools && $(GO) run ./benchcheck -baseline ../$(BENCH_BASELINE) -update $(BENCH_OUT)



.PHONY: bench
bench: ## Run every benchmark once, with allocation counts
	$(GO) test -run '^$$' -bench . -benchmem $(PKG)

# Accelerated versus portable Go, same benchmarks, compared properly.
# Requires: go install golang.org/x/perf/cmd/benchstat@latest
.PHONY: benchcmp
benchcmp: ## Accelerated vs portable Go, through benchstat
	$(GO) test -run '^$$' -bench . -count 10 $(PKG) > /tmp/simd-asm.txt
	$(GO) test -run '^$$' -bench . -count 10 -tags purego $(PKG) > /tmp/simd-pure.txt
	benchstat /tmp/simd-pure.txt /tmp/simd-asm.txt

# ---------------------------------------------------------------------- hygiene

.PHONY: fmt
fmt: ## Format all Go source
	$(GO) fmt $(PKG)

.PHONY: fmt-check
fmt-check: ## Fail if anything is not gofmt-clean
	@out=$$(gofmt -l . | grep -v '^tools/goat/' || true); \
	if [ -n "$$out" ]; then echo "not gofmt'd:"; echo "$$out"; exit 1; fi

.PHONY: vet
vet: ## go vet, including asmdecl over every generated .s
	$(GO) vet $(PKG)

.PHONY: tidy
tidy: ## go mod tidy in both modules
	$(GO) mod tidy
	cd tools && $(GO) mod tidy

.PHONY: clean
clean: ## Remove build artefacts and generated scratch files
	$(GO) clean -testcache

# ------------------------------------------------------------------- hardware
# Produce a hardware run report, ready to attach to an issue or commit.
#
# CONTRIBUTING calls this two commands and then asked people to transcribe the
# output into a template by hand, which is neither two commands nor something a
# stranger with a board owes anybody. This does the transcription.
#
# It deliberately does NOT fail when the suite fails. A failing run on real
# silicon is the more valuable of the two outcomes — three memory-corruption
# bugs in this library's history were invisible on amd64 — and a target that
# exits non-zero before writing the file would throw away exactly the report
# worth having.
.PHONY: hardware-report
hardware-report: ## Run the suite and write testdata/hardware/<goos>-<goarch>-<tier>.md
	@set -e; \
	info=$$($(GO) run ./cmd/simdinfo 2>&1) || { \
		echo "cmd/simdinfo did not run:"; echo "$$info"; \
		echo; echo "That is itself worth reporting — open an issue with this output."; \
		exit 1; }; \
	goos=$$($(GO) env GOOS); goarch=$$($(GO) env GOARCH); \
	tier=$$(printf '%s' "$$info" | sed -n 's/.*tier=\([a-z0-9]*\).*/\1/p'); \
	[ -n "$$tier" ] || tier=unknown; \
	out=testdata/hardware/$$goos-$$goarch-$$tier.md; \
	mkdir -p testdata/hardware; \
	echo "running the suite on $$goos/$$goarch, tier=$$tier — this takes a few minutes"; \
	acc=$$($(GO) test $(PKG) 2>&1) && accres=Passed || accres=FAILED; \
	sca=$$(GOSIMD=scalar $(GO) test $(PKG) 2>&1) && scares=Passed || scares=FAILED; \
	{ \
	  echo "# $$goos/$$goarch, tier=$$tier"; echo; \
	  echo "## Machine"; echo; \
	  echo '```'; \
	  echo "goos:   $$goos"; \
	  echo "goarch: $$goarch"; \
	  echo "go:     $$($(GO) version)"; \
	  echo "cpu:    $$(sysctl -n machdep.cpu.brand_string 2>/dev/null \
	                 || awk -F': ' '/model name/{print $$2; exit}' /proc/cpuinfo 2>/dev/null \
	                 || echo '<fill this in>')"; \
	  echo "os:     $$(uname -srm)"; \
	  echo '```'; echo; \
	  echo "## Tier selected"; echo; \
	  echo '```'; echo "\$$ go run ./cmd/simdinfo"; echo "$$info"; echo '```'; echo; \
	  echo "## Correctness"; echo; \
	  echo "- accelerated (\`go test ./...\`): **$$accres**"; \
	  echo "- portable (\`GOSIMD=scalar go test ./...\`): **$$scares**"; echo; \
	  echo '<details><summary>accelerated output</summary>'; echo; \
	  echo '```'; echo "$$acc" | tail -40; echo '```'; echo; echo '</details>'; echo; \
	  echo '<details><summary>portable output</summary>'; echo; \
	  echo '```'; echo "$$sca" | tail -40; echo '```'; echo; echo '</details>'; echo; \
	  echo "## Wall-clock"; echo; \
	  echo "Not measured. \`make hardware-bench\` adds it, on a quiet machine."; echo; \
	  echo "## Anything odd"; echo; \
	  echo "<Anything that surprised you, including nothing.>"; \
	} > $$out; \
	echo; echo "wrote $$out  (accelerated: $$accres, portable: $$scares)"; \
	echo "Attach it to https://github.com/sebishogun/simd/issues/new/choose,"; \
	echo "or open a pull request adding it. A FAILED run is the more useful one."

# Benchmarks are a separate target because they have a precondition the
# correctness run does not: the machine has to be quiet. -count 6 because the
# minimum is what gets used rather than the mean — benchmark noise is one-sided,
# so the fastest run is the one least interfered with. See entry 48 of
# docs/wrong.md, which cost twenty-one phantom regressions to learn.
.PHONY: hardware-bench
hardware-bench: ## Append benchmarks to the hardware report (quiet machine only)
	@set -e; \
	goos=$$($(GO) env GOOS); goarch=$$($(GO) env GOARCH); \
	tier=$$($(GO) run ./cmd/simdinfo 2>&1 | sed -n 's/.*tier=\([a-z0-9]*\).*/\1/p'); \
	out=testdata/hardware/$$goos-$$goarch-$$tier.md; \
	[ -f $$out ] || { echo "run 'make hardware-report' first"; exit 1; }; \
	echo "benchmarking — close everything else; this takes a while"; \
	bench=$$($(GO) test -run '^$$' -bench . -count 6 $(PKG) 2>&1) || true; \
	{ echo; echo "## Wall-clock"; echo; \
	  echo '```'; echo "$$bench"; echo '```'; } >> $$out; \
	echo "appended benchmarks to $$out"
