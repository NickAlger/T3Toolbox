#!/usr/bin/env bash
# Re-run the experiments made during the review of 2026-11-14 and the target decisions of 2026-11-15 against a T3Toolbox checkout.
#   Usage: ./run_review_checks.sh [/path/to/T3Toolbox]   (default: this repository)
# Compare the output with expected_review_output_3dc15d9.txt (numbers agree to rounding; last digits may differ).
# These are correctness checks on small problems. Nothing is timed. The last three need jax and are skipped without it.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${1:-$(cd "$HERE/../../../.." && pwd)}"       # default: the repository this folder is in
if [ ! -d "$REPO/t3toolbox" ]; then echo "usage: $0 [/path/to/T3Toolbox]" >&2; exit 2; fi
SCRIPTS="$(cd "$HERE/.." && pwd)"
run() { echo "=== $1.py"; shift; python3 -I "$@" 2>&1 | grep -v '^WARNING' || true; echo; }
run unbundled_examples   "$HERE/unbundled_examples.py"   "$REPO"
run jet_steps_check      "$HERE/jet_steps_check.py"      "$REPO"
run kind_readout_check   "$HERE/kind_readout_check.py"   "$REPO"
run tilde_adjoint_check  "$HERE/tilde_adjoint_check.py"
run slot_tuple_examples  "$HERE/slot_tuple_examples.py"  "$REPO" "$SCRIPTS"
run pad_reshape_check    "$HERE/pad_reshape_check.py"    "$REPO"
run levels_sketch        "$HERE/levels_sketch.py"        "$REPO" "$SCRIPTS"
if python3 -I -c 'import jax' 2>/dev/null; then
  run slot_tuple_check   "$HERE/slot_tuple_check.py"     "$REPO"
  run levels_jax_check   "$HERE/levels_jax_check.py"     "$REPO"
  run order0_assessment  "$HERE/order0_assessment.py"    "$REPO"
else
  echo "=== slot_tuple_check.py, levels_jax_check.py, order0_assessment.py: skipped (jax is not installed)"
fi
run merged_frontend_prototype "$HERE/merged_frontend_prototype.py" "$REPO"
run tilde_jet_pairing_check "$HERE/tilde_jet_pairing_check.py"
run type_catalog_check     "$HERE/type_catalog_check.py"
run arrow_type_check       "$HERE/arrow_type_check.py"
run gauge_arrow_check      "$HERE/gauge_arrow_check.py" "$REPO"
run tilde_multiplier_check "$HERE/tilde_multiplier_check.py"
run bad_scaling_example    "$HERE/bad_scaling_example.py" "$REPO"
run adapter_doctest_draft  "$HERE/adapter_doctest_draft.py"
if python3 -I -c 'import jax' 2>/dev/null; then
  run order_scan_check    "$HERE/order_scan_check.py"
else
  echo "=== order_scan_check.py: skipped (jax is not installed)"
fi
