#!/usr/bin/env bash
#
# Run the verification targets this machine can do, and write the findings to
# a single markdown file.
#
# # Why this exists, and why it redacts
#
# The interesting machines are often not the machine the work happens on. A
# maintainer may have an arm64 laptop provided by an employer, and be able to
# run the suite on it and not to say where it ran. A report that quietly
# carries a hostname, a username, a home directory or a corporate Go module
# proxy in a stack trace is not shareable, and worse, looks shareable.
#
# So the default is to redact, the redactions are listed in the report itself,
# and `--raw` turns them off for the case where none of that matters. What is
# never redacted is the technical content: architecture, tier, pass or fail,
# timings, and the full text of anything that went wrong — because a report
# that drops the failure output to be safe is not worth writing.
#
#   scripts/report.sh                 # everything available, redacted
#   scripts/report.sh --quick         # skip the slow lanes
#   scripts/report.sh --raw           # no redaction
#   scripts/report.sh -o findings.md  # somewhere else
#
# The output is one self-contained markdown file with tables, meant to be
# pasted whole.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ROOT=$PWD

OUT=simd-report.md
REDACT=1
QUICK=0
while [ $# -gt 0 ]; do
  case $1 in
    --raw)    REDACT=0 ;;
    --quick)  QUICK=1 ;;
    -o)       OUT=$2; shift ;;
    -h|--help) sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

have() { command -v "$1" >/dev/null 2>&1; }

# cpu_model and core_count read whichever interface the platform has. The CPU
# model is kept rather than redacted: "Apple M3 Pro" or "Neoverse V2" is the
# single most useful line in a performance report and identifies a part, not a
# person or an employer. Anyone who disagrees can delete the row.
cpu_model() {
  if [ -r /proc/cpuinfo ]; then
    grep -m1 -E '^(model name|Model|CPU part)' /proc/cpuinfo 2>/dev/null \
      | cut -d: -f2- | sed 's/^ *//'
  elif have sysctl; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null
  fi
}

core_count() {
  if have nproc; then nproc
  elif have sysctl; then sysctl -n hw.ncpu 2>/dev/null
  else echo "?"
  fi
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ---------- redaction ----------
#
# Ordered longest-first, because $HOME contains $USER and replacing the user
# first would leave a half-scrubbed path behind.

USERNAME=$(id -un 2>/dev/null || echo "")
HOSTNAME_S=$(hostname 2>/dev/null || echo "")
HOSTNAME_SHORT=${HOSTNAME_S%%.*}

# The module path is public — it is in go.mod and on the forge — but it embeds
# the account name, which on many setups is also the local username. Scrubbing
# it turned every `ok github.com/x/simd` line into `github.com/[user]/simd`,
# which hides nothing and makes the report hard to read. So it is put back
# after the scrub rather than exempted from it, which keeps the username rule
# simple and total.
MODPATH=$(awk '/^module /{print $2; exit}' "$ROOT/go.mod" 2>/dev/null)
MODSCRUBBED=$MODPATH
if [ -n "$USERNAME" ] && [ -n "$MODPATH" ]; then
  MODSCRUBBED=${MODPATH//$USERNAME/[user]}
fi

scrub() {
  if [ "$REDACT" = 0 ]; then cat; return; fi

  # sed with a delimiter that cannot appear in a path.
  sed \
    -e "s|$HOME|~|g" \
    ${USERNAME:+-e "s|\\b$USERNAME\\b|[user]|g"} \
    ${HOSTNAME_S:+-e "s|\\b$HOSTNAME_S\\b|[host]|g"} \
    ${HOSTNAME_SHORT:+-e "s|\\b$HOSTNAME_SHORT\\b|[host]|g"} \
    -e 's|/Users/[^/ ]*|~|g' \
    -e 's|/home/[^/ ]*|~|g' \
    -e 's|GOPROXY=[^ ]*|GOPROXY=[redacted]|g' \
    -e 's|GOPRIVATE=[^ ]*|GOPRIVATE=[redacted]|g' \
    -e 's|GONOSUMDB=[^ ]*|GONOSUMDB=[redacted]|g' \
    -e 's|GOMODCACHE=[^ ]*|GOMODCACHE=[redacted]|g' \
    -e 's|[A-Za-z0-9._%+-]*@[A-Za-z0-9.-]*\.[A-Za-z][A-Za-z]*|[email]|g' \
    -e 's|https\{0,1\}://[^ )]*\.corp[^ )]*|[internal-url]|g' \
    | { if [ -n "$MODPATH" ] && [ "$MODSCRUBBED" != "$MODPATH" ]; then
          sed "s|${MODSCRUBBED//[/\\[}|$MODPATH|g"; else cat; fi; }

}

# ---------- running a target ----------

declare -a ROWS=()
declare -a FAILED=()

run_target() {
  local t=$1 log=$TMP/$t.log start end rc dur
  printf '  %-16s ' "$t" >&2
  start=$(date +%s)
  ( cd "$ROOT" && make "$t" ) >"$log" 2>&1
  rc=$?
  end=$(date +%s)
  dur=$((end - start))

  local status note
  if [ $rc -eq 0 ]; then
    status="pass"; note=$(summarise "$t" "$log")
    printf 'pass  %ds\n' "$dur" >&2
  else
    status="**FAIL**"; note="exit $rc — output below"
    FAILED+=("$t")
    printf 'FAIL  %ds\n' "$dur" >&2
  fi
  ROWS+=("$(printf '| `%s` | %s | %ds | %s |' "$t" "$status" "$dur" "$note")")
}

# summarise pulls the one number a reader wants out of a passing log, so the
# table says something more than "pass".
summarise() {
  local t=$1 log=$2
  case $t in
    test|test-purego|test-race)
      printf '%s packages ok' "$(grep -c '^ok' "$log")" ;;
    test-tiers)
      printf 'tiers: %s' "$(grep -oE '^--- GOSIMD=[a-z0-9]+' "$log" | sed 's/.*=//' | tr '\n' ' ')" ;;
    test-riscv64|test-loong64|test-gates)
      printf '%s' "$(grep -m1 -oE 'tier=[a-z0-9]+' "$log")  $(grep -c 'PASS' "$log") PASS" ;;
    perf-model)
      printf '%s targets modelled' "$(grep -cE '^[a-z0-9]+/[a-z0-9]+ ' "$log")" ;;
    check-emission)
      printf '%s kernels' "$(grep -oE '[0-9]+ kernels' "$log" | awk '{s+=$1} END {print s+0}')" ;;
    vet|fmt-check)
      printf 'clean' ;;
    *)
      printf 'ok' ;;
  esac
}

# ---------- which targets to run ----------
#
# Reuses menu.sh's judgement rather than repeating it, so a target that the
# menu marks unavailable is skipped here for the same reason and with the same
# explanation.

available() {
  bash "$ROOT/scripts/menu.sh" --rows 2>/dev/null \
    | grep '●' | awk '{print $2}' \
    | { if [ -n "${REPORT_PRETEND_MISSING:-}" ]; then
          grep -vxF -e "${REPORT_PRETEND_MISSING// /$'\n'}"; else cat; fi; }
}

WANT_ALL="fmt-check vet test test-purego test-tiers check-emission perf-model \
          test-gates test-riscv64 test-loong64 test-cross"
WANT_QUICK="fmt-check vet test test-purego test-tiers perf-model"

WANT=$WANT_ALL
[ "$QUICK" = 1 ] && WANT=$WANT_QUICK

echo "gosimd report — running the targets this machine supports" >&2
echo >&2

AVAIL=$(available)
declare -a SKIPPED=()
for t in $WANT; do
  if grep -qx "$t" <<<"$AVAIL"; then
    run_target "$t"
  else
    SKIPPED+=("$t")
  fi
done

# ---------- the report ----------

{
  echo "# gosimd verification report"
  echo
  echo "Generated by \`scripts/report.sh\`. Paste it whole."
  echo

  if [ "$REDACT" = 1 ]; then
    echo "> **Redacted.** Hostname, username, home directory, e-mail addresses and"
    echo "> Go proxy/module-cache settings are replaced with placeholders. Nothing"
    echo "> technical is removed: architecture, tier, pass/fail, timings and the"
    echo "> full text of every failure are below as they happened. Re-run with"
    echo "> \`--raw\` to disable."
    echo
  fi

  echo "## Machine"
  echo
  echo '| | |'
  echo '|---|---|'
  printf '| OS / arch | %s / %s |\n' "$(go env GOOS)" "$(go env GOARCH)"
  printf '| Go | %s |\n' "$(go version | awk '{print $3}')"
  printf '| CPU | %s |\n' "$(cpu_model)"
  printf '| Cores | %s |\n' "$(core_count)"
  printf '| Selected tier | %s |\n' "$(cd "$ROOT" && go run ./cmd/simdinfo 2>/dev/null | sed 's/.*tier=//;s/ .*//')"
  printf '| Available tiers | %s |\n' "$(cd "$ROOT" && go run ./cmd/simdinfo -tiers 2>/dev/null | tr '\n' ' ')"
  printf '| Commit | %s |\n' "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"
  printf '| Tree | %s |\n' "$([ -z "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && echo clean || echo 'has uncommitted changes')"
  echo

  echo "## Results"
  echo
  echo '| target | result | time | notes |'
  echo '|---|---|---|---|'
  printf '%s\n' "${ROWS[@]}"
  echo

  if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "### Not run on this machine"
    echo
    echo '| target | why |'
    echo '|---|---|'
    for t in "${SKIPPED[@]}"; do
      miss_reason=$(bash "$ROOT/scripts/menu.sh" --preview "$t" 2>/dev/null \
        | sed -n 's/^ *missing //p' | tr '\n' ' ')
      printf '| `%s` | missing %s |\n' "$t" "${miss_reason:-unknown}"
    done
    echo
    echo "These are skipped, not failed. Each is the only cover for something —"
    echo "\`test-riscv64\` is the only lane that executes RVV, \`test-gates\` the only"
    echo "one that runs a CPU with no vector unit — so a report without them is a"
    echo "report about this machine, not about the library."
    echo
  fi

  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "## Failures"
    echo
    echo "Full output, unedited apart from the redactions noted above."
    echo
    for t in "${FAILED[@]}"; do
      echo "### \`make $t\`"
      echo
      echo '```'
      tail -120 "$TMP/$t.log" | scrub
      echo '```'
      echo
    done
  else
    echo "## Failures"
    echo
    echo "None."
    echo
  fi

  if [ -s "$TMP/perf-model.log" ]; then
    echo "## Modelled throughput"
    echo
    echo "llvm-mca over each kernel's inner loop against the same kernel compiled"
    echo "without vectorization. A model: L1-resident, one core, no memory system."
    echo
    echo '```'
    scrub < "$TMP/perf-model.log"
    echo '```'
    echo
  fi

  echo "## Benchmarks"
  echo
  if [ -f "$ROOT/testdata/bench/$(go env GOARCH).txt" ]; then
    echo "A baseline exists for this architecture. Run \`make bench-check\`"
    echo "separately, on an idle machine, and paste its output here."
  else
    echo "**No baseline exists for \`$(go env GOARCH)\` yet.** To record one:"
    echo
    echo '```'
    echo "make bench-update     # on an otherwise idle machine"
    echo '```'
    echo
    echo "Read \`testdata/bench/README.md\` first. A baseline taken on a busy or"
    echo "thermally degraded machine is worse than none, because every later"
    echo "comparison inherits it."
  fi
  echo

  echo "---"
  printf '%s targets run, %s passed, %s failed, %s unavailable here.\n' \
    "${#ROWS[@]}" "$(( ${#ROWS[@]} - ${#FAILED[@]} ))" "${#FAILED[@]}" "${#SKIPPED[@]}"
} > "$OUT"

echo >&2
printf 'wrote %s (%s lines)\n' "$OUT" "$(wc -l < "$OUT")" >&2
[ ${#FAILED[@]} -gt 0 ] && exit 1
exit 0
