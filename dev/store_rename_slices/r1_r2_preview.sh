#!/usr/bin/env bash
# R1: preview of R2 -- the v2026.4.0 tree with the dry run's pass-2 files overlaid (never committed, never the repo).
# Runs the bitwise battery against the reference, the K-gates at stage R2, and (with --suite) the test suite.
#   bash dev/store_rename_slices/r1_r2_preview.sh <scratch dir> [--suite]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCR="$1"; shift || true
PY=/home/nick/miniconda3/envs/t3toolbox/bin/python
cd "$REPO"
$PY dev/store_rename_apply.py --dry-run --out "$SCR/r1out" --scratch "$SCR/dry" | tail -1
rm -rf "$SCR/r2preview" && mkdir -p "$SCR/r2preview"
git archive v2026.4.0 t3toolbox tests examples pyproject.toml | tar -x -C "$SCR/r2preview"
cp -r "$SCR/dry/pass2/t3toolbox" "$SCR/dry/pass2/tests" "$SCR/dry/pass2/examples" "$SCR/r2preview/"
echo "== battery on the R2 preview vs the v2026.4.0 reference"
PYTHONPATH="$SCR/r2preview" $PY dev/store_rename_battery.py run --out "$SCR/battery_r2preview" --b9 dev/store_rename_slices/battery_ref/b9 | tail -2
$PY dev/store_rename_battery.py compare dev/store_rename_slices/battery_ref "$SCR/battery_r2preview" | tail -5 || true
echo "== K-gates at stage R2 on the pass-2 tree"
$PY dev/store_rename_check.py --stage R2 --tree "$SCR/dry/pass2" --hits "$SCR/r1out" --pass1 "$SCR/dry/pass1" \
    --realized --out "$SCR/check_R2preview.md" || true
if [[ "${1:-}" == "--suite" ]]; then
    echo "== test suite on the R2 preview (-n 3)"
    (cd "$SCR/r2preview" && PYTHONPATH="$SCR/r2preview" $PY -m pytest tests/ -q -n 3 -p no:cacheprovider \
        > "$SCR/pytest_r2preview.txt" 2>&1 || true)
    tail -40 "$SCR/pytest_r2preview.txt"
fi
