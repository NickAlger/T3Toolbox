#!/usr/bin/env bash
# Re-run every check against a T3Toolbox checkout.   Usage: ./run_all.sh [/path/to/T3Toolbox]   (default: this repository)
# Compare the output with expected_output_3dc15d9.txt (numbers agree to rounding; the last digits may differ).
# These are correctness checks and need only numpy. Timing experiments are in run_timings.sh.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${1:-$(cd "$HERE/../../.." && pwd)}"          # default: the repository this folder is in
if [ ! -d "$REPO/t3toolbox" ]; then echo "usage: $0 [/path/to/T3Toolbox]" >&2; exit 2; fi
for s in graded_messages_prototype probe_fit_step_example correspondence_check pairing_check \
         extra_core_experiment index_check index_check2 frame_consistency_check \
         frame_consistency_nondense_check f32_experiment gn_fusion_check plan_mockup reuse_check component_graphs sharing_check; do
  echo "=== $s.py"
  python3 "$HERE/$s.py" "$REPO"
  echo
done
