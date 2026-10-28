Verbatim copies of the 2026-08-22 review's O1/O2 oracle sweeps (originals archived with the review
record), promoted as permanent tests by `tests/test_oracle_sweep.py` (Phase D). Edit policy: keep
these byte-close to the archived originals. Local changes: `sweep_optimizers.main`'s `case_filter` hook
for the always-on/slow two-tier split, and (2026.3.0) the `GaussNewtonModel` -> `fitting.LocalModel`
rename in `sweep_models.py`'s module docstring (the frontend class was renamed; docstring only, no code
change). Also (2026.3.0) one line at the top of each `run_case` in
`sweep_models.py` / `sweep_optimizers.py`, `np.random.seed(seed)`: the archived originals drew the trial tangents
(`geom.randn`) and the `randn` structure templates from the unseeded global stream, so a run's `grad_fd` could land
on either side of its `1e-6` bar (observed 1.05e-6 once); seeded per case, every case is reproducible and
independent of which cases ran before it (the two-tier `case_filter` and `pytest -n` included).
`sweep_exact.py` is new (2026.3.0, not a review copy): the exact Hessian against a dense extrinsic oracle.
