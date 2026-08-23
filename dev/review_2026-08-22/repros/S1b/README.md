# S1b — the uniform frame on a numerically rank-deficient train (open; Nick working it on paper)

Run each with `PYTHONPATH=<repo>` from outside the repo.

- `s1b_cases.py`  — the six structural/numerical rank combinations; only "structurally minimal AND numerically
  deficient in both families" (the `resize` continuation warm start) loses a direction, and it is a genuine
  rank deficiency of the masked real block (one lost tangent direction), not a skewed basis.
- `s1b_hist.py`   — the same check at v2026.0.0 / v2026.1.0 / HEAD: identical; it never worked (the uniform
  frame dates from July; the April research-project code is the ragged sweep, which has no padding).
- `s1b_steps.py`, `s1b_pin2.py` — the four sweep steps by hand (all orthonormal) and the per-core masked Gram
  terms of the library frame (the last down core's mode slot 2 lives entirely in padded rR slots).
- `s1b_walk.py`   — the 3-core walkthrough: the last core's down-step unfolding (9 x 3, rows = (rL, rR) slots),
  its zero singular value, and LAPACK's completion `e_(a=0,b=2)` landing in a padded row; the ragged twin's
  3 x 3 unfolding forces the completion into the real row `a=2`.

Mechanism in one line: the SVD sorts the sigma > 0 directions ahead of the padding, which is what the prefix
masks rely on -- but the sigma = 0 block is one degenerate eigenspace that mixes the real-but-deficient
directions with the padded ones, and the uniform layer lets the completion land in the latter.
Candidate fix sketched in dev/HANDOFF.md (masked orthonormal completion after each uniform SVD step);
Nick suspects a more elegant solution -- pending.
