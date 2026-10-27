Verbatim copies of the 2026-08-22 review's O1/O2 oracle sweeps (originals archived with the review
record), promoted as permanent tests by `tests/test_oracle_sweep.py` (Phase D). Edit policy: keep
these byte-close to the archived originals. Local changes: `sweep_optimizers.main`'s `case_filter` hook
for the always-on/slow two-tier split, and (2026.3.0) the `GaussNewtonModel` -> `fitting.LocalModel`
rename in `sweep_models.py`'s module docstring (the frontend class was renamed; docstring only, no code
change).
