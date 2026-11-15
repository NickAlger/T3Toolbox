#!/usr/bin/env bash
# Timing experiments for the execution-plan discussion.   Usage: ./run_timings.sh /path/to/T3Toolbox
# INDICATIVE ONLY. The recorded file timings_2core_cpu_3dc15d9.txt is from a 2-core CPU. Do not use these
# numbers to set defaults; re-run on the target hardware. eta_axes.py and sweep_blocks.py need jax.
set -e
REPO="$1"
if [ -z "$REPO" ]; then echo "usage: $0 /path/to/T3Toolbox" >&2; exit 2; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
for s in order_numpy eta_axes sweep_blocks; do
  echo "=== $s.py"
  python3 "$HERE/$s.py" "$REPO"
  echo
done
