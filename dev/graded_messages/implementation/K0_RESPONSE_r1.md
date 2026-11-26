# Response to `K0_REVIEW_r1.md`

Implementer, 2026-11-25. The three minor findings are fixed; nothing else changed but what the sweep of m3 found.
The orchestrator confirmed the third floor at 150.

- **m1.** Measured on the built tree with the test module's own `scan_vocabulary()`: **84** (the simulated tree's 86
  still held the two dead scan bodies). Both floor comments now say 84
  (`tests/test_contractions_interpreter.py:552`, `tests/test_contractions_sharding.py:103`).
- **m2.** `tests/backend/test_uniform_curvature.py`'s new docstring line is rewrapped under 120 columns.
- **m3.** `tests/test_dispatch.py`, `test_jit_uniform_derivatives`: its two comments described the deleted kernels
  (the d-prefixed jet contractions, the binomial tensor, `build_input_jets`). They now say what the test jits: the
  forward jets and the transposes along a line through `_derivative_data` (the merged methods at an input jet, the
  engine), the supercores through the engine's jet sweeps, the masks host constants.
- **The sweep of m3.** The comments and docstrings of `tests/` against the 315 names of `k0_names.json` and the word
  `sampling_derivatives` (a tokenizer pass, `APPLY`/`ENTRIES`/`PROBE` left out as words): 4 hits, all deliberate. Two
  are the frozen `HISTORICAL` rows of `test_contractions_interpreter.py` (`:237`, `:241`). The other two are K0's own
  statements that the module is gone (`test_import_graph.py:21`, `test_jets.py:896`). Since the names miss prose that
  describes the old kinds without naming them, I also grepped for "P20", "old kinds" and "old kernels". Two more
  plainly described deleted code as present, and are fixed:
  - `tests/test_transposes_typed.py:6-7`: "The old layer keeps its own name until it is deleted (P17, P20): the old
    kinds' `transpose` / `corewise_hessian` and the four `utv_*_transpose*_from_sweep` helpers". The sentence is
    removed.
  - `tests/backend/test_curvature.py:1854-1855`: "(The old kinds' sixth operation `corewise_hessian` is no longer
    consulted: the kinds go in P20.)". The parenthesis is removed.

  Listed, not changed (history, or a statement that stays true):
  - `tests/oracle_sweeps/oracle.py:146`: "the old forms above stay for the sweeps until the old kinds go". These are
    the oracle's own helper forms, which the sweeps still use. The clause dates them but names nothing deleted as
    present; it is for P17/P19 to reword or drop.
  - `tests/_fits.py:16-18`: "what the old derivative kinds fitted", "the old kinds' exactly". A description of what the
    helper reproduces, true as history.
  - `tests/test_fitting.py:39`: "the six old kinds", the oracles' purpose, as history.
  - `tests/test_frontend_sampling.py:309`: "outlives P20". True.
  - `tests/backend/test_uniform_fitting.py:339`: "in place of the old kind", as history.

**Rerun:** `k0_inventory.py --check`, and the tests of the touched files: `test_contractions_interpreter`,
`test_contractions_sharding`, `test_uniform_curvature`, `test_dispatch`, `test_transposes_typed` and `test_curvature`.
`--check`: **0 references to deleted names**, 15 allowed strings. The six files at `-n 2`: **254 passed, 2 skipped,
5,832 subtests passed** (21:07); the 2 skips are the two `_SLOW` tests of `test_uniform_curvature.py`, as in the baseline.
