# scripts

Shell used by the Makefile, not entry points.

| | |
|---|---|
| `menu.sh` | backs `make menu` and `make targets`: lists every verification target and marks the ones this machine can actually run, with what to install for the rest. Uses fzf when present, works without it. |
| `report.sh` | collects the output the Makefile's reporting targets assemble. |
