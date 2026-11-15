# N10 handoff: acceptance (plan §8 row N10, §11)

T3Toolbox code is unchanged in N10: the checklist exposed no failure. This file is the only T3Toolbox change.

## (1) The §11 / CI checklist, on 9bacdaa5

| item | result |
|---|---|
| full suite, t3toolbox env (py3.11, numpy 2.x, jax 0.10), `-n 3` | 1,338 passed, 4 skipped, 48,697 subtests (28 m 38 s, under the concurrent Darcy runs) |
| full suite, floor env `tttt` (py3.9, numpy 1.22, jax 0.4.30; single process, no xdist) | 1,337 passed, 5 skipped (38 m 58 s). The fifth skip is the expected one on jax 0.4.30: it lacks `unregister_event_duration_listener` (N6) |
| padded-frame files in the numpy-1.26 / OpenBLAS venv (`floor126`): `test_padded_frame`, `test_padded_start`, `test_uniform_padded_frame`, `test_aligned_padded_frame_frontend`, `test_linalg`, `test_import_graph` (`-n 2`) | 154 passed, 2,216 subtests |
| `test_import_graph`, `test_dispatch` and `test_contractions_interpreter`, run alone (`-n 3`) | 99 passed, 2,766 subtests |
| every HISTORICAL row present | yes. The feature modules (`padded_frames`, `uniform_padded_frames`, `linalg`, `optimizers`, `geometry`) use 46 distinct contraction strings, all in `HISTORICAL`. The interpreter test's own vocabulary check (every scanned library subscript has a row) passes |
| `-W` docs build (`sphinx -W --keep-going -b dummy`) | build succeeded, exit 0 |
| module doctests (`pytest --doctest-modules t3toolbox`) | 241 passed (t3toolbox); 241 passed (tttt) |
| markdown doctests (the CI command, every `docs/*.md` and `docs/contributor/*.md` except `doctest_style.md`) and `getting_started.rst` | pass on both envs |

## (2) T3Polynomial adopts the library (commit eced8e3, not pushed)

**The ladder change** (`t3polynomial/fitting.py`).
- New keyword `fit_rank_continuation_adaptive(padded_start=False)`. When it is True, every level after the first
  passes the previous level's point as `newton_cg(padded_from=)`.
- `g0_reference='cone'` and the `padded_*` options pass through `newton_kwargs`, as every `newton_cg` option
  already does.
- `_newton_fit` repacks `padded_from` at `x0`'s `N` on the uniform backends.
- Each record now carries `g0_reference` and `g0norm`, plus `start_frame` / `start_movements` / `cone_movements`
  when the library reports them.
- **The default is OFF, so the ladder's behaviour is unchanged.** Whether to flip it is a decision for Nick.
  With `'cone'` the reference is larger, so `gtol_ref` must drop (D-24 suggests ~1e-4; g04/g06 used 3e-4).

**`d10_darcy_o01_fit.py`.** New knobs `START` (`library` / `aligned` / `hint_free`) and `G0_REF` (`zero` / `cone`),
both defaulting to the old behaviour. The script prints and stores the reference and the alignment passes for each
level.

**Retiring `cone.py`.**
- `t3polynomial/cone.py` is now a one-line shim that raises `ImportError`, pointing at
  `newton_cg(padded_from=, g0_reference='cone')` / `padded_start`. The shim is needed because
  `scripts/g04_start_frames.py` still imports it.
- `tests/test_cone.py` is deleted.
- New test `test_fitting_selection.test_padded_start_and_cone_reference` (ragged and uniform backends).
- T3Polynomial's suite: 67 passed, 33 skipped, 547 subtests.

**`pyproject.toml` still says `t3toolbox >= 2026.3.0`.** The new path needs 2026.4.0, which is unreleased. I left the
pin alone, because the default path still works on 2026.3.0.

**Verification on the o01 configuration** (`d10`, TAG `n10_aligned_cone`). Settings: ragged numpy, exact Hessian,
CG power 1.0, `max_newton` 30, `START=aligned`, `G0_REF=cone`, `GTOL_REF=3e-4`, three levels.

| level | ranks | Newton / CG | stop | cone reference | alignment passes | time | test p50 |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 12 / 82 | gradient | 21.031 | — (no previous level) | 6 s | .4712 |
| 1 | 1 → 6 | 23 / 286 | gradient | 46.161 | 10 (cap; last movement 0.49) | 164 s | .1899 |
| 2 | 6 → 11 | 30 / 408 | **max_newton** (‖g‖ 0.124 against the stop 0.0169) | 56.170 | 10 (cap; last 0.37) | 514 s | .1152 |

- Both levels run, and the `'cone'` references are reported. The level-1 value, 46.16, is in line with the prototype
  cone (48.4 / 47.3 at r = 6).
- Context only, not like-for-like: the library-frame / zero-reference / gtol 1e-3 run `ex_p10_ragged` had level 1 at
  17 / 164 (test .1897) and level 2 at 29 / 588 (test .1175).
- At 6 → 11 the 30-iteration cap binds under the cone stop. The g06 Newton runs used `max_newton` 60.
- The alignment never reached tol 0.05 within 10 passes. The cap was the effective stop, as the docs say.

## (3) The capture baselines on the Gaussian convention (for the plan's §2.3 / §11)

**Method.** `scripts/g06_aligned_frame.py PART=capture` (T3Polynomial, new in eced8e3) on the o01 checkpoints
`d10_ex_p10_jit_level0{0..3}`. It runs the SHIPPED `backend.optimizers.padded_start` with three settings:
- `max_passes=0`: the hint-free start;
- the defaults: two-colour, p 5 for 2, cap 10, tol 0.05;
- 10 fixed passes (`tol=0`).

The capture is `‖P_F ∇f‖ / ‖∇f‖`.
- The ambient norm `‖∇f‖` was computed afresh at 1 → 6: 81.050, which reproduces the library's .189.
- At 6 → 11 and 11 → 16 it was taken from the g06 runs on the same checkpoints (46.874, 31.003). The ambient norm does
  not depend on the frame or the convention.

**Results.** Seeds 0 / 1:

| start | 1 → 6 | 6 → 11 | 11 → 16 |
|---|---|---|---|
| library padded frame | .189 | .117 | .106 |
| hint-free, Gaussian (D-41) | **.055 / .071** | **.091 / .092** | **.094 / .093** |
| aligned, shipped default (stopped at the cap, 10 passes) | **.451 / .464** | **.353 / .347** | **.262 / .260** |
| aligned, 10 fixed passes | identical to the default | identical | identical |
| *(plan, superseded)* hint-free, coordinate convention | .104 | .093 | .094 |
| *(plan)* aligned, 10 passes, simultaneous prototype | .464 | .354 | .270 |

**Reading.**
- The Gaussian hint-free start captures about half of what the coordinate convention captured at 1 → 6, and about the
  same at 6 → 11 and 11 → 16.
- At 1 → 6 the Gaussian start is below the library frame, as the coordinate start already was at every level.
- The shipped two-colour default matches the prototype's simultaneous 10-pass capture to within 0.01 at every level.
- The default never stopped on tol: its last movements at pass 10 were 0.17–0.50, so the default and the fixed
  10 passes give identical frames.
- Frames were orthonormal to ≤ 2.4e-15 and represented the base point to ≤ 4.5e-15.
- Each alignment took 2–7 s (ragged numpy).

Raw results are in T3Polynomial `data/g0_reference/g06_L{1,2,3}_capture_n10.json` and `.log` (gitignored).

## Not run in N10 (§11, outside this brief)

- The laptop Newton comparison across start arms (library / hint-free / aligned at 4, 10 and 40 passes / no
  oversampling, two seeds).
- Compile logging and peak memory on one level.
- The Runpod 20 × 1000 run.

The g06 `PART=newton` arms still use the prototype; adapting them to `padded_start` is a small change if wanted.

## Round 2 (`N10_review.md`: REVISE on T3Polynomial only)

T3Polynomial commit **120b856**, on top of eced8e3; neither is pushed.

**MAJOR fixed.** Under `padded_start=True` the ladder owns `padded_from`. On a level with no predecessor (the first),
it now drops the `padded_*` keys forwarded through `newton_kwargs`, so `START=hint_free` runs.

**Tests.**
- A `_newton_fit` spy over a 3-level ladder with `newton_kwargs={'padded_max_passes': 0}`:
  - level 0 gets no `padded_*` key;
  - levels 1 and 2 get `padded_from` equal to the previous level's point (`x0` is that point, zero-padded) and
    `padded_max_passes=0`;
  - the records show `start_frame='padded'` with zero passes.
- The `uniform_jax` route through the ladder: `'cone'`, two tiny levels, padded start.

**Checked on o01.** d10 with `START=hint_free G0_REF=cone` ran levels 0 and 1: 1 → 6 took 17 Newton / 302 CG, cone
46.16, test p50 .1810.

**MINORs and NITs.**
- `dev/HANDOFF.md` has a current-status block.
- The pyproject comment keeps 2026.3.0 as the floor and notes 2026.4.0 for the opt-in path.
- The `cone.py` shim names the three library successors.
- Not changed: NIT 4 (`tests/test_dispatch.py:177` still mentions "coordinate seeds") and NIT 6 (the plan's §2.2
  wording). Both are in T3Toolbox, where nothing is to change in N10.

T3Polynomial's suite: 69 passed, 33 skipped, 547 subtests.
