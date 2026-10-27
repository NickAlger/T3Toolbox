# `s0b/` — the S0b downstream measurement (plan §F "Research-repo measurement", 2026-10-27)

`S0b_report.md` is the read-only measurement of T3Polynomial @ 388090d against T3Toolbox main @ 3158778 (after S6):
the §K call-site table with its discrepancies, the x06/x07 FD cross-checks of `hessian='exact'`, exact-vs-GN numbers on
e01/o01/b01 (d05/s05 need FEniCSx datasets and were not run), the hvp cost ratios, and the one T3Toolbox flag (the
safe-mode gauge check on a near-cancelling difference of two gauged tangents — queued in `../slices/DEFERRED.md`).

The run scripts (`s0b_*.py`, minimal-edit copies of the T3Polynomial scripts + their `.s0b.diff`) import the private
`t3polynomial` package, so they are NOT kept here (research never accretes in this repo). They live with the research
repo / T3Polynomial; S14 re-runs them against the release candidate. The numbers go to `verification_records.md` at S13.
