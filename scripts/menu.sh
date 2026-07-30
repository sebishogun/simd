#!/usr/bin/env bash
#
# An interactive picker for this repository's make targets.
#
# The problem it solves is not discoverability, or not only that. This
# Makefile has twenty-eight targets and roughly a third of them cannot run on
# any given machine: the qemu lanes need five emulators, the cross lane needs
# docker with binfmt registered, codegen needs clang and llvm-objdump, and
# perf-model needs llvm-mca. Someone cloning this on an Apple Silicon Mac has
# no way to know which is which except by running them and reading the
# failures.
#
# So every entry is checked against the machine it is being shown on, and the
# preview says either "runs here" or exactly what is missing and how to get it.
#
# The target list, the descriptions and the explanations all come from the
# Makefile. Nothing is duplicated here: a new target with a `##` comment
# appears in this menu without touching this file, and if it has a comment
# block above it, that block is its preview.
#
# fzf is used when present and is not required. Without it the same list is
# printed and picked by number, because the point is that anyone on any
# architecture can see what is available.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ROOT=$PWD
MAKEFILE=$ROOT/Makefile

# ---------- colours, only when the output is a terminal ----------

if [ -t 1 ] && [ "${NO_COLOR:-}" = "" ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'; CYA=$'\033[36m'
else
  B=; DIM=; R=; GRN=; YEL=; RED=; CYA=
fi

# ---------- what this machine can do ----------

have() { command -v "$1" >/dev/null 2>&1; }

GOOS=$(go env GOOS 2>/dev/null || echo unknown)
GOARCH=$(go env GOARCH 2>/dev/null || echo unknown)
TIERS=$(cd "$ROOT" && go run ./cmd/simdinfo -tiers 2>/dev/null | tr "\n" " ")

# requirements_for prints the tools a target needs, one per line. Anything not
# listed needs only Go, which is true of the majority.
requirements_for() {
  case $1 in
    codegen|check-emission)  echo clang; echo llvm-objdump ;;
    perf-model)              echo clang; echo llvm-mca ;;
    test-riscv64)            echo qemu-riscv64 ;;
    test-loong64)            echo qemu-loongarch64 ;;
    test-gates)              echo qemu-riscv64 ;;
    test-cross|cross-setup)  echo docker ;;
    benchcmp)                echo benchstat ;;
    *)                       echo go ;;
  esac
}

# install_hint gives the one thing a reader actually wants when a tool is
# missing: the command that gets it, for the platform they are on.
install_hint() {
  case $1 in
    go)
      echo "https://go.dev/dl — this repository needs Go 1.26 or newer" ;;
    clang|llvm-objdump|llvm-mca)
      case $GOOS in
        darwin) echo "brew install llvm   (Apple's clang omits llvm-mca and llvm-objdump)" ;;
        *)      echo "install LLVM: pacman -S clang llvm, or apt install clang llvm" ;;
      esac ;;
    qemu-riscv64|qemu-loongarch64)
      case $GOOS in
        darwin)
          # Not an install hint, because there is nothing to install. These
          # lanes use qemu-USER, which translates Linux syscalls and therefore
          # only runs on Linux; Homebrew's qemu is qemu-system-*, a full
          # machine emulator that will not run a Linux static binary directly.
          echo "unavailable on macOS: qemu-user emulates Linux syscalls and needs a Linux host." ;;
        *)      echo "extract a recent one: cid=\$(docker create tonistiigi/binfmt:latest) && docker cp \$cid:/usr/bin/$1 ~/.local/bin/ && docker rm \$cid" ;;
      esac ;;
    docker)
      case $GOOS in
        darwin) echo "Docker Desktop, or: brew install --cask docker" ;;
        *)      echo "install docker or podman, then: make cross-setup" ;;
      esac ;;
    benchstat)
      echo "go install golang.org/x/perf/cmd/benchstat@latest" ;;
    *)
      echo "install $1" ;;
  esac
}

# missing_for prints the requirements of a target that this machine lacks.
missing_for() {
  local t=$1 r
  while read -r r; do
    [ -z "$r" ] && continue
    have "$r" || echo "$r"
  done < <(requirements_for "$t")
}

# ---------- reading the Makefile ----------
#
# Targets are discovered rather than listed, so this script does not go stale
# when one is added. A `##` comment on the target line is its description; a
# comment block directly above it is its preview.

targets() {
  grep -oE '^[a-z][a-z0-9-]*:' "$MAKEFILE" | tr -d ':' | sort -u
}

# Internal targets are the ones other targets call with arguments. They cannot
# be run on their own and showing them would be an invitation to a confusing
# failure.
is_internal() {
  case $1 in
    qemu-run|qemu-run-plain|qemu-run-probe) return 0 ;;
    *) return 1 ;;
  esac
}

# describe_target returns the `##` text on the target line, or failing that the
# last sentence-like line of the comment block above it, or a stock line.
describe_target() {
  local t=$1 line desc
  line=$(grep -nE "^$t:" "$MAKEFILE" | head -1 | cut -d: -f1)
  [ -z "$line" ] && { echo "no description"; return; }

  desc=$(sed -n "${line}p" "$MAKEFILE" | sed -n 's/.*## *//p')
  if [ -n "$desc" ]; then echo "$desc"; return; fi

  # Walk up through the comment block and take its first line, which in this
  # Makefile is reliably the summary sentence.
  local i=$((line - 1)) first=""
  while [ "$i" -gt 0 ]; do
    local l
    l=$(sed -n "${i}p" "$MAKEFILE")
    case $l in
      \#*) first=$(printf '%s' "$l" | sed 's/^# *//') ;;
      .PHONY:*) ;;
      *) break ;;
    esac
    i=$((i - 1))
  done
  if [ -n "$first" ]; then printf '%s\n' "$first"; else echo "(no description in the Makefile)"; fi
}

# preview_body prints the whole comment block above a target, which is where
# this Makefile keeps its reasoning.
preview_body() {
  local t=$1 line
  line=$(grep -nE "^$t:" "$MAKEFILE" | head -1 | cut -d: -f1)
  [ -z "$line" ] && return

  local i=$((line - 1)) start=$line
  while [ "$i" -gt 0 ]; do
    local l
    l=$(sed -n "${i}p" "$MAKEFILE")
    case $l in
      \#*|.PHONY:*) start=$i ;;
      *) break ;;
    esac
    i=$((i - 1))
  done
  [ "$start" -ge "$line" ] && return
  sed -n "${start},$((line - 1))p" "$MAKEFILE" | grep '^#' | sed 's/^# \{0,1\}//'
}

# recipe_for prints the target's recipe as written in the Makefile: the
# indented lines that follow it, with the leading @ and the line-continuation
# noise stripped so it reads as the command it is.
recipe_for() {
  local t=$1 line
  line=$(grep -nE "^$t:" "$MAKEFILE" | head -1 | cut -d: -f1)
  [ -z "$line" ] && return
  awk -v start="$((line + 1))" '
    NR < start { next }
    /^\t/ {
      sub(/^\t/, "")
      sub(/^@/, "")
      sub(/[ \t]*\\$/, "")
      if ($0 != "") print
      next
    }
    { exit }
  ' "$MAKEFILE" | head -6 | expand_vars
}

# expand_vars substitutes the handful of Makefile variables that appear in
# recipes, so the preview shows a command a reader could paste rather than one
# they would have to resolve first.
expand_vars() {
  sed -e "s|\$(GO)|go|g" \
      -e "s|\$(PKG)|./...|g" \
      -e "s|\$(DOCKER)|docker|g" \
      -e "s|\$(MAKE)|make|g" \
      -e "s|\$(TIERS)|${TIERS:-scalar}|g" \
      -e "s|\$(QEMU_LOONG)|qemu-loongarch64|g" \
      -e "s|\$(QEMU_RISCV)|qemu-riscv64|g" \
      -e 's|\$\$|$|g'
}

# unique_coverage says what a target is the only cover for, so that a red
# entry reads as a gap rather than as an inconvenience.
unique_coverage() {
  case $1 in
    test-riscv64)  echo "riscv64 RVV — the only lane that executes it" ;;
    test-loong64)  echo "loong64 LSX/LASX — the only lane that executes it" ;;
    test-gates)    echo "the fallback path on a CPU with NO vector unit, which -cpu max cannot produce" ;;
    test-cross)    echo "arm64 NEON/SVE2, s390x VX, ppc64le VSX" ;;
    test-tiers)    echo "every tier this CPU has: ${TIERS:-scalar}" ;;
    test-purego)   echo "the portable reference, which every backend starts from" ;;
    perf-model)    echo "modelled throughput on arm64, ppc64le and s390x" ;;
    codegen)       echo "regenerates all six architectures from csrc" ;;
    *)             echo "" ;;
  esac
}

# ---------- the preview pane ----------

do_preview() {
  local t=$1
  printf '%s%s%s\n' "$B" "make $t" "$R"
  printf '%s\n' "$DIM$(printf '%.0s─' {1..64})$R"

  local miss
  miss=$(missing_for "$t")
  if [ -z "$miss" ]; then
    printf '%s✓ runs on this machine%s  (%s/%s)\n\n' "$GRN" "$R" "$GOOS" "$GOARCH"
  else
    printf '%s✗ cannot run here%s  (%s/%s)\n' "$RED" "$R" "$GOOS" "$GOARCH"
    local m
    while read -r m; do
      [ -z "$m" ] && continue
      printf '    %smissing%s %s\n' "$YEL" "$R" "$m"
      printf '%s\n' "$(install_hint "$m")" | fold -s -w 60 | sed "s/^/        $DIM/;s/\$/$R/"
    done <<< "$miss"
    printf '\n'
  fi

  # What this target verifies that nothing else does. Several are the only
  # cover for a whole architecture, which is worth saying before someone
  # decides a red one does not matter.
  local only
  only=$(unique_coverage "$t")
  [ -n "$only" ] && printf '%scovers%s %s\n\n' "$CYA" "$R" "$only"

  # The recipe as written, rather than `make -n` output. The expansion of a
  # target that delegates to another with variables set is a wall of shell
  # that says less than the two lines it came from.
  local recipe
  recipe=$(recipe_for "$t")
  if [ -n "$recipe" ]; then
    printf '%sruns%s\n' "$CYA" "$R"
    printf '%s\n' "$recipe" | sed 's/^/  /' | cut -c1-72
    printf '\n'
  fi

  local body
  body=$(preview_body "$t")
  if [ -n "$body" ]; then
    printf '%sfrom the Makefile%s\n' "$CYA" "$R"
    printf '%s\n' "$body" | fold -s -w 66 | sed 's/^/  /'
  fi
}

# ---------- the machine banner ----------

banner() {
  printf '%sgosimd%s  make targets on %s%s/%s%s\n' "$B" "$R" "$B" "$GOOS" "$GOARCH" "$R"
  if have go && [ -d "$ROOT/cmd/simdinfo" ]; then
    local info
    info=$(cd "$ROOT" && go run ./cmd/simdinfo 2>/dev/null)
    [ -n "$info" ] && printf '        %s%s%s\n' "$DIM" "$info" "$R"
  fi
  local tools=() t
  for t in go clang llvm-mca llvm-objdump docker benchstat \
           qemu-riscv64 qemu-loongarch64 qemu-aarch64; do
    if have "$t"; then tools+=("${GRN}${t}${R}"); else tools+=("${DIM}${t}${R}"); fi
  done
  printf '        %s\n\n' "$(IFS=' '; echo "${tools[*]}")"
}

build_rows() {
  local t d miss mark
  while read -r t; do
    is_internal "$t" && continue
    d=$(describe_target "$t")
    miss=$(missing_for "$t")
    if [ -z "$miss" ]; then mark="${GRN}●${R}"; else mark="${DIM}○${R}"; fi
    printf '%s %-16s %s%s%s\n' "$mark" "$t" "$DIM" "${d:0:60}" "$R"
  done < <(targets)
}

# ---------- entry points ----------
#
# Every function this section calls is defined above it. That is not style:
# bash resolves a function at the moment the call runs, so a flag handled
# before its helper is defined fails with "command not found" — which is how
# ctrl-r once reached `make --rows`.

# --preview is how fzf calls back into this script.
if [ "${1:-}" = "--preview" ]; then
  do_preview "$2"
  exit 0
fi

if [ "${1:-}" = "--list" ]; then
  while read -r t; do
    is_internal "$t" && continue
    printf '%s\t%s\n' "$t" "$(describe_target "$t")"
  done < <(targets)
  exit 0
fi

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '3,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

# --rows is fzf's reload source and must be handled before the branch below
# that treats any argument as a target to run. It was not, once, and `make
# --rows` is a confusing thing to have happen when you press ctrl-r.
if [ "${1:-}" = "--rows" ]; then
  build_rows
  exit 0
fi

# A target named on the command line runs directly, so this is usable as
# `scripts/menu.sh test` too.
if [ "$#" -gt 0 ]; then
  exec make "$@"
fi

build_rows() {
  local t d miss mark
  while read -r t; do
    is_internal "$t" && continue
    d=$(describe_target "$t")
    miss=$(missing_for "$t")
    if [ -z "$miss" ]; then mark="${GRN}●${R}"; else mark="${DIM}○${R}"; fi
    printf '%s %-16s %s%s%s\n' "$mark" "$t" "$DIM" "${d:0:60}" "$R"
  done < <(targets)
}

pick_with_fzf() {
  build_rows | fzf \
    --ansi \
    --height='90%' \
    --layout=reverse \
    --border=rounded \
    --border-label=' make targets ' \
    --prompt='target > ' \
    --pointer='▸' \
    --header=$'● runs here   ○ needs something\nenter run · tab mark · ctrl-r reload · esc quit' \
    --preview="bash '${BASH_SOURCE[0]}' --preview {2}" \
    --preview-window='right,58%,border-left,wrap' \
    --bind="ctrl-r:reload(bash '${BASH_SOURCE[0]}' --rows)" \
    | awk '{print $2}'
}


# pick_by_number writes the menu to the terminal and only the chosen name to
# stdout, because the caller captures stdout. Printing the list there too
# would put the whole menu in the variable and show the user nothing.
pick_by_number() {
  local -a names=() i=1 t d miss
  exec 3>&1                       # 3 is the real stdout, for the answer

  # Prompt on the controlling terminal when there is one, and on stderr when
  # there is not — a pipeline, a CI log, or anything else without a tty. The
  # earlier version opened /dev/tty unconditionally and died with "no such
  # device" the moment it was run non-interactively.
  local out=/dev/stderr in=/dev/stdin
  if [ -e /dev/tty ] && (: >/dev/tty) 2>/dev/null; then
    out=/dev/tty; in=/dev/tty
  fi
  exec 1>"$out"

  echo "fzf is not installed, so here is the same list."
  echo
  while read -r t; do
    is_internal "$t" && continue
    d=$(describe_target "$t")
    miss=$(missing_for "$t")
    names+=("$t")
    if [ -z "$miss" ]; then
      printf '  %s%2d%s %s●%s %-16s %s%s%s\n' "$B" "$i" "$R" "$GRN" "$R" "$t" "$DIM" "${d:0:56}" "$R"
    else
      printf '  %s%2d%s %s○%s %-16s %s%s (needs %s)%s\n' "$B" "$i" "$R" "$DIM" "$R" "$t" \
        "$DIM" "${d:0:40}" "$(echo "$miss" | tr '\n' ' ')" "$R"
    fi
    i=$((i + 1))
  done < <(targets)
  echo
  printf 'number, or q to quit: '
  local choice=""
  read -r choice <"$in" || true
  case $choice in
    ''|q|Q)      return 1 ;;
    *[!0-9]*)    return 1 ;;
  esac
  [ "$choice" -ge 1 ] && [ "$choice" -le "${#names[@]}" ] || return 1
  printf '%s\n' "${names[$((choice - 1))]}" >&3
}

banner

# SIMD_MENU_PLAIN forces the numbered list even where fzf is installed, for
# anyone who would rather not have a full-screen picker and for testing the
# path that everyone without fzf gets.
if have fzf && [ -z "${SIMD_MENU_PLAIN:-}" ]; then
  choice=$(pick_with_fzf)
else
  choice=$(pick_by_number) || { echo "nothing selected."; exit 0; }
fi

[ -z "${choice:-}" ] && { echo "nothing selected."; exit 0; }

miss=$(missing_for "$choice")
if [ -n "$miss" ]; then
  printf '%smake %s cannot run here.%s\n' "$RED" "$choice" "$R"
  while read -r m; do
    [ -z "$m" ] && continue
    printf '  missing %s%s%s — %s\n' "$B" "$m" "$R" "$(install_hint "$m")"
  done <<< "$miss"
  exit 1
fi

printf '%s$ make %s%s\n\n' "$B" "$choice" "$R"
exec make "$choice"
