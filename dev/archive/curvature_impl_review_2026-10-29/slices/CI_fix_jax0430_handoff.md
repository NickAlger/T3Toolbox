# CI floor-leg hot-fix (py3.9/numpy<2/jax 0.4.30) -- worktree ci_fix @ a7287e7, UNCOMMITTED; patch impl/ci_fix.patch
## Root cause: not a library bug, not dtype/x64 (x64 off in both; memory_analysis fields present in both)
The XLA in jaxlib 0.4.30 compiles the chunked transpose/two-hole into a *different program*. Its buffer assignment fits the chunk's assembly scratch
into memory the earlier W-sized stages have freed, so one more probe costs exactly one jet slice, d*(order+1)*r*4 B: 384/1024/18432 at (3,4,8,64,o3)/
(24^4,r16,o3)/(128^6,o5), GN == two-hole, and x prod(C). Current XLA (0.12) charges the whole assembly row: 5664/5888/86016 B, two-hole 1.96x GN.
- **memory_analysis is trustworthy in BOTH.** temp_size matched the measured resident peak (RSS polled during execution, W=65536, cs 256 vs 16384)
  within about 1 MB. 0.4.30: 241.3/247.5 MB temp vs 241.3/247.5 RSS (+6.2 MB = 384 B x 16128 rows). 0.12: 297.2/388.5 vs 298.2/389.1
  (+91 MB = 5664 B x 16128 rows). So the estimator does what S3b/M2 intended on both. The tests pinned one compiler's numbers:
  (1) balanced `peak > 0.5*floor` even with cs clamped at W (floor/per-row is ~1.7W current, ~8W old; it passed only by luck);
  (2) frame stack: budget 2^22 @ W=8192 leaves the free chunk clamped at W on old; (3) `th > 1.5*gn`, and the W=32000 budgets also clamp on old;
  (4) in-situ: a hard `<=` at the estimator's MAXIMAL chunk vs XLA's ~0.1% span-dependence of the packed temp (+0.075% old, -0.14% current: a coin-flip).
- **HIDDEN failure:** CI stops at "Test suite", so the floor has skipped the doc doctests since 4eacbcc; `docs/chunking.md:141` (same 2^22 @ 8192) failed too. Now fixed; all later floor steps run (below).
## Change (estimator CODE untouched, so current-gen answers are bit-identical; docstrings + 1 doc example + tests)
- tests/test_chunk_size_estimator.py: policies against whatever slope is measured. Balanced: within the floor, maximal if cs<W,
  + a mocked 4x-cost run pinning the binding branch (== W//4). Frame stack and hessian budgets at W=10**6, so the budget binds everywhere.
  Hessian: exact == the smaller balance / sized by max(gn,th), maximal; the ratio claim is dropped (it is the compiler's).
  In-situ: documented `PACKING_RTOL = 0.01`, ~10x the packing noise and far below the 1.2-1.5x overshoot it guards. Module docstring explains why.
- docs/chunking.md: the value also depends on the jax/XLA release, and why. Frame-stack example: n_probes=10**6, `2.5 < ratio < 4.5` (old 4.003, current 3.43).
- sampling_derivatives.py docstrings (`_assembly_per_row_bytes`; `estimate_chunk_size`'s 1.3-2.2x line): the slope is the compiler's. CHANGELOG: amended
  the unreleased [2026.3.0] estimator bullet (figures = current jax; 0.4.30 ~5-15x lower); no new Fixed bullet (nothing released broke, no behaviour change).
## Gates (PYTHONPATH=worktree) -- floor = jax 0.4.30/np 1.26.4; current = jax 0.12.0.1/np 2.5.3
- test_chunk_size_estimator.py full: floor 14 passed/20 sub (101 s); current 14 passed/20 sub (59 s; 2 pre-existing UserWarnings from TestFittingChunkSizeWiring).
- test_probe_derivatives.py (the sampling_derivatives file): 17/188 sub both. test_dispatch `-k chunk` selects 0 tests, so ran
  `-k "probe_derivatives or uniform_derivatives or JetDtype"`: 4/22 sub both. doctest-modules sampling_derivatives.py: pass both. chunking.md doctest: rc 0 both.
- Floor FULL suite (halves, -n 2): 302 passed (7:08) + 728 passed, 1 skipped (9:24); module doctests (all) 210 passed; quickstart + all doc pages rc 0. Current: sphinx -W clean.
**Pre-existing, NOT fixed (scope):** on current XLA a remainder chunk (W % cs != 0) is allocated ON TOP of the full chunk (W=8192, cs=2600: +392 rows x
5664 B exactly), so max_chunk_size_within can overshoot by (cs+rem)/cs, up to ~2x. The in-situ test avoids it by design (W=8*cs). Ledger line / follow-up.
## Draft commit message
tests: the chunk-size estimator tests pin the policies, not one XLA's slope (CI floor leg, jax 0.4.30)

The floor leg has failed since 4eacbcc (S3b). The estimator measures the compiled chunked program's slope, and that slope is the compiler's:
jaxlib 0.4.30 packs the chunk loop into memory the W-sized stages free, leaving one jet slice per row (two-hole == GN), while current XLA
charges the whole assembly row (5-15x more; two-hole ~2x GN). memory_analysis matches the resident peak on both (RSS-checked), so the
estimator is right on both. The tests now assert the policies against the measured slope: budgets that bind on every compiler, a mocked
cost for the binding balanced branch, the two-hole sized by max(gn, th), and a documented 1% packing tolerance in the in-situ check. Also
fixed: the same budget mistake in docs/chunking.md's frame-stack example, which failed on the floor too; CI never reached that step. The
docstrings, the note and the unreleased CHANGELOG bullet now say the measured cost depends on the jax release. Estimator code unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017qDbbe5KVzCbMP39xChGE3
