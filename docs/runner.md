# The self-hosted runner

CI for this project is expensive in ways a hosted runner is bad at. The
emulated lanes compile several megabytes of assembly under qemu, and the
benchmark gate compares against a stored baseline, which is meaningless on a
machine shared with whoever else GitHub put on it. So the emulated and
benchmark jobs run on a self-hosted runner and the rest stay hosted.

## What is installed

- Runner root: `~/actions-runner/simd`, registered against `sebishogun/simd`
  as `simd-omarchy` with the extra label `simd`.
- Service: `~/.config/systemd/user/gh-runner-simd.service`, a **user** unit.
  `loginctl` linger is enabled for the account, so it starts at boot without a
  login.
- `KillMode=process` and `TimeoutStopSec=5min`, because `run.sh` re-execs
  itself when the runner self-updates and a stop during that would leave a
  half-updated runner.
- `Nice=5` rather than a CPU quota. A quota would make the benchmark numbers
  depend on what else happened to be running, which is exactly what the
  baseline exists to hold still.
- An explicit `PATH` rather than the interactive one `config.sh` snapshots into
  `.path`, which carries every mise shim on the machine.

## What the jobs need on it

`go`, `clang`, `llvm-objdump`, `docker`, `benchstat`, and the two hand-fetched
emulators — `qemu-loongarch64` and `qemu-riscv64` in `~/.local/bin`. The last
two are not optional and not interchangeable with the distribution's: the qemu
inside `multiarch/qemu-user-static` predates LoongArch LSX/LASX and emulates a
RISC-V CPU with no vector extension, so both backends are skipped as
unexecutable and the lane passes having tested nothing. That is how two
backends stayed broken for months. Fetch them with:

    cid=$(docker create tonistiigi/binfmt:latest)
    docker cp "$cid":/usr/bin/qemu-loongarch64 ~/.local/bin/
    docker cp "$cid":/usr/bin/qemu-riscv64    ~/.local/bin/
    docker rm "$cid"

## Registering another one

Never copy `.runner`, `.credentials` or `.credentials_rsaparams` from an
existing runner — the last is a private signing key tied to one registration.
Register fresh:

    mkdir -p ~/actions-runner/<repo> && cd ~/actions-runner/<repo>
    tar xzf ~/actions-runner/simd/actions-runner.tar.gz
    TOKEN=$(gh api -X POST repos/<owner>/<repo>/actions/runners/registration-token -q .token)
    ./config.sh --url https://github.com/<owner>/<repo> --token "$TOKEN" \
        --name <repo>-omarchy --labels <repo> --work _work --unattended

then copy `gh-runner-simd.service`, change the description and the two paths,
and `systemctl --user enable --now` it.

## Checking on it

    systemctl --user status gh-runner-simd
    journalctl --user -u gh-runner-simd -f
    gh api repos/sebishogun/simd/actions/runners -q '.runners[] | "\(.name) \(.status)"'
