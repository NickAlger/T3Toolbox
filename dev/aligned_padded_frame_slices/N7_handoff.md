# N7 handoff: the optimizer seam

Slice N7 of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§3.5, §4.7, §5.5, §5.6, D-21, D-25–D-27, D-33,
D-37, D-38, D-39). Implementer: Claude Opus 5.5, 2026-11-09, on top of 2e9e7193. Everything is uncommitted; nothing
outside the repository was touched. First, the maintainer's ruling: the uniform loop kernel's docstring now says that
the `functools.partial` binding is accepted in that instance only and sets no precedent.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/geometry.py` | `ManifoldGeometryOps` and `UniformManifoldGeometryOps` gain `with_frame`, `aligned_padded_frame` and `require_frame_structure`. Two private helpers: `_sharing_labels` (groups → per-mode labels) and `_structurally_zero` (D-21's rule per element; raises on a mixed stack). New imports: `ranks`, `padded_frames`, `uniform_padded_frames`, `ZERO_FRAME_SEED`, the `ALIGN_*` constants. |
| `t3toolbox/backend/optimizers.py` | `Problem.local_model(frame=)`; `_misfit_gradient_at`; the public `padded_start` (in `__all__`, with a doctest of the frontend `least_squares_problem` route); `_alignment_options`, `_cone_options`, `_cone_reference`, `_movement_record`; `'cone'` in `_G0_REFERENCES`; `newton_cg`'s fourteen keyword-only keywords (§5.6 names and defaults) plus its docstring; `gradient_descent`'s six `cone_*` keywords. No new imports (T18's `{common, regularization, corewise}` holds). |
| `t3toolbox/backend/uniform_padded_frames.py` | The maintainer's sentence in the loop kernel's docstring. |
| `tests/backend/test_padded_start.py` | T14 (a)–(h): `TestFrameStructure`, `TestNewtonKeywords`, `TestSeamNumerics`, plus the helpers `fit_case`, `dense_probe_gradient` and `dense_tied_project`. |
| `tests/test_dispatch.py` | `TestPaddedStartSeam`: the T16 rows for the members and `padded_start`, and T14 (i). |
| `tests/test_optimizers_frontend.py` | The keyword-mirror test exempts the fourteen new backend keywords until N8 (judgment call 3). |

## The code

- **Members.**
  - `with_frame`: ragged returns `self`. Uniform returns `from_frame(frame, (masks[0], masks[2]))` with this
    geometry's `groups`.
  - `aligned_padded_frame(x0, x_old, project, hint, …, use_jit)`:
    - target ranks come from `x0`'s core shapes (ragged) or `self.masks` sums (uniform);
    - the old frame is the empty frame when `x_old` is `None` or structurally zero (D-21, per element; a mixed
      stack raises), else `self.frame(x_old)` (ragged) or `ut3_orthogonal_representations(x_old)[0]` (uniform);
    - sharing labels from `groups`;
    - uniform compaction widths from `self.var_masks`;
    - then the layer's loop, `use_jit` passed through.
  - `require_frame_structure(x_cores, frame)`, integers and held masks only (D-37):
    - ragged: each store's core shapes against `compute_orthogonal_representation_ranks` of `x`'s ranks, plus `C`;
    - uniform: `shape`; the variation masks against `var_masks`, family by family; the two boundary entries; each
      supercore's widths against its masks; `N`.
    - Every message names the fix.
- **`Problem.local_model(frame=)`.** It checks the frame through the member and linearizes there. A geometry
  without the member raises.
- **`_misfit_gradient_at(frame, (problem, residual))`** is §3.5 verbatim.
- **`padded_start`.**
  - A geometry without the member raises.
  - Options default to `None` and are forwarded only when set.
  - D-38: `max_passes=0` with `oversampling`, `oversampled_passes`, `tol` or `alternate` raises; `seed` is
    allowed.
  - The residual is at `x0`, or at the zero tensor for `x_old=None`.
- **`newton_cg`.**
  - The fourteen keywords are keyword-only, appended.
  - Structural errors: both `start_frame` and `padded_from`; a `padded_*` option without `padded_from`; a `cone_*`
    option without the cone being computed (not `'cone'`, or `g0norm_newton` given); zero passes with an unread
    option.
  - `padded_from` → `padded_start(..., use_jit=use_jit)`. Iteration 0 linearizes at the start frame; a rejected
    trust-region trial keeps it, as today.
  - `'cone'` without `g0norm_newton` → `_cone_reference`:
    - `padded_start(problem, x0, None, cone_*, use_jit)`;
    - then `‖_misfit_gradient_at(F, r(0))‖` in `geom.with_frame(F)`'s inner product;
    - a vanishing value raises; a geometry without the member raises.
    - With `g0norm_newton` given, nothing runs.
  - Stats: `start_frame`, `start_movements` and `cone_movements` appear only with their keywords. History rows say
    `g0_reference='cone'`.
- **`gradient_descent`.** The same `'cone'`, run eagerly (it has no `use_jit`).

### Judgment calls

1. **T14 (f) is not a library defect** (round 2; the review's root cause, confirmed).
   - `ufv_shared_frame_data` and the uniform tied projection are correct: on the same frame, uniform equals ragged
     to 6e-17.
   - The cause: `share()` gives orthonormal Tucker factors, with every singular value exactly 1. At that tie the two
     layers' orthogonal representations choose different bases of the same subspaces, and the hint-free start
     reads its coordinate seeds in those bases. So the two layers' STARTS differ in span (0.999 here).
   - Round 1's test built the dense reference from the ragged start for both layers. It is now built from each
     layer's own old frame and start, both layers pass at ≤ 1e-9, and the expected-failure marker is gone.
   - It is a basis dependence of the hint-free start at singular-value ties, not a defect. The aligned loop washes it
     out; see the extra check below.
2. **The dense tied oracle.**
   - My first version summed the members' down-frame Tucker columns, which wrongly assumes equal down variations;
     I caught it against the library at a regular tied frame.
   - The correct tied directions vary the shared factor itself: `Σ_{i∈g} Y_i ×_i δU` with `Y_i = X ×_i U_iᵀ`. With
     those, the ragged library matches to 1e-9.
3. **The frontend mirror test.** `test_optimizers_frontend.test_newton_cg_keywords_mirror_backend` requires every
   backend `newton_cg` keyword on the frontend twin. The frontend keywords arrive with N8 (§3.6), so the fourteen new
   ones are exempted, with a comment naming N8. N8 removes the exemption. (The file is not named for N7, but the
   alternative was a red suite.)
4. **`_movement_record`.** The uniform record's NaN tail is dropped (trailing NaNs only), so `start_movements` /
   `cone_movements` are a tuple of floats, one per pass. As N6 noted, a genuine NaN on the last pass is
   indistinguishable.
5. **Conversion under `use_jit`.** `newton_cg` converts `x0` and the problem as before (`_prepare_jit_inputs`). The
   old point is converted by the loops (N3b/N6). `start_frame` is not converted in the backend: the frontend twin
   converts it (§3.6).
6. **T14 (h) through the frontend** waits for N8's twins. The backend members are tested here, on both layers.

### Round 2 (the review: REVISE on one MAJOR)

- **MAJOR:** T14 (f) is fixed as above.
- **NIT:** `_cone_reference` computed the zero tensor's residual twice. It now calls the member directly with its
  one residual; the options are still validated by `_cone_options`. The no-loop test now mocks
  `_misfit_gradient_at`.
- **Extra check: does the aligned loop wash out the start's basis dependence?** Script
  `n7_start_washout.py` in the session scratchpad, not committed.
  - Setup: tied old points, three seeds each of a `t3svd` output (unshared) and a `share()`d point (sharing
    `(0, 1, 1, 0)`); shape `(5, 6, 6, 5)`, old ranks 2, target `((3, 4, 4, 3), (1, 3, 4, 3, 1))`; a full-rank T3
    hint projected by `tv_/utv_project_t3/ut3_onto_tangent_space`; each layer's loop from its own old frame.

  | old point | seed | starts differ | default schedule: passes (ragged / uniform), final span | tol 1e-10: passes, final span |
  |---|---|---|---|---|
  | t3svd output | 0 | 0.98 | 10 / 10, 4.9e-3 | 141 / 139, 7.4e-11 |
  | t3svd output | 1 | 0.27 | 10 / 10, 2.3e-5 | 89 / 89, 3.3e-14 |
  | t3svd output | 2 | 0.98 | 9 / 9, 5.7e-4 | 109 / 109, 1.1e-12 |
  | shared point | 0 | 1.00 | 10 / 10, 1.3e-3 | 91 / 95, 4.6e-10 |
  | shared point | 1 | 0.27 | 10 / 10, 5.5e-4 | 105 / 105, 2.3e-12 |
  | shared point | 2 | 0.27 | 10 / 10, 5.4e-4 | 85 / 85, 4.0e-13 |

  - **Answer: yes, in practice.** The fixed point is set by the gradient. From starts that differ by up to 1.00 in
    span, the final frames agree to 2e-5–5e-3 after the default 9–10 passes, with the same pass counts. At tol
    1e-10 they agree to 3e-14–5e-10, with pass counts within 4.
  - At the default `tol = 0.05` the residual difference is the loop's own stopping accuracy, not the start's.
- **Checks:**
  - the `-W` Sphinx dummy build: exit 0, no warnings;
  - an AST scan of the three touched modules: no shadowed imports or duplicate definitions;
  - the touched tests: 115 passed (no xfail now), 1,449 subtests;
  - the doctests: 4 passed;
  - floor env (`tttt`): the touched tests 115 passed in 140.21s (0:02:20); `TestPaddedStartSeam` 2 passed, 75 deselected in 70.11s (0:01:10).

## Verification

- **(a)** One rejection per family through `local_model(frame=)`:
  - ragged: each store;
  - uniform: each variation-mask family, each boundary entry, `shape`, a padded width.
  - It accepts `geom.frame(x)` and the aligned frame on both layers; the corewise geometry raises.
- **(b)** `newton_cg(padded_from=)` gives the same losses as `newton_cg(start_frame=padded_start(...)[0])`, on both
  layers.
- **(c)** The G-bit.
  - Key sets: the stats and history keys are unchanged without the new keywords; the new keys appear only with
    them.
  - **Before/after:** the same seeded runs on HEAD's tree (`git archive`) and on N7's were compared on both
    layers: Gauss-Newton, exact Hessian, `'warm_start'`, `'zero'` with `g0norm_cg`, and `gradient_descent`. **10 of 10
    runs are identical**: the cores bitwise, and `repr(stats)` equal apart from `wall_time`.
- **(d)**
  - The cone is deterministic, independent of `x0`'s values, and larger than `'zero'`.
  - `cone_max_passes=0` equals `'zero'` to 1e-12, at either oversampling.
  - `gradient_descent`'s cone movements equal `newton_cg`'s.
  - `padded_max_passes=0` starts from `fv_padded_frame` / `ufv_compact_frame(ufv_padded_frame(...))` at ≤ 1e-14
    with no projection (`_misfit_gradient_at` mocked to raise).
  - Zero passes with an unread option raises in all four places, one option at a time; the seed is accepted.
  - The structural errors above, and `'cone'` with `g0norm_newton` running no loop.
- **(e)** `probe_derivatives` at d = 4, uniform against ragged `padded_start`: the same pass counts and movements
  ≤ 1e-9; spans ≤ 1e-9 at tol 1e-10 and ≤ 0.2 at the default stop.
- **(f)** Both layers, each from its own old frame and start: one pass against the dense tied oracle ≤ 1e-9.
- **(g1)** A gauge-rotated frame (the frame of another core representation of the same tensor):
  - matched quadratics equal ≤ 1e-10 relative;
  - the FD criterion along its own retraction holds ≤ 1e-4.
- **(g2)** Exact Newton from a padded start is finite and descends.
- **(h)** A rank-0 and a partially-zero-rank old point through both members give the zero tensor's start (bitwise
  ragged, ≤ 1e-14 uniform). A mixed stack raises.
- **(i)** In the dispatch file, uniform `newton_cg(use_jit=True, padded_from=…, g0_reference='cone')` equals
  `use_jit=False`:
  - the same pass counts;
  - the cone value ≤ 1e-12 relative;
  - the iteration-0 frame ≤ 1e-12 by the frame measure;
  - the problem's gradient is traced inside the whole-loop jit.
- **T16:** the members and `padded_start` on numpy and jax inputs, both layers (jax out, movements equal numpy's ≤
  1e-10); `with_frame` type; `require_frame_structure` accepts jax frames.
- **Mutations, 16, all killed:**
  - the ragged store check; the uniform shape, var-mask, boundary and width checks;
  - the zero-old rule; the mixed-stack guard; `local_model` ignoring `frame`;
  - both keywords; `padded_*` without its group; `cone_*` unchecked; zero passes unchecked;
  - the cone at the zero frame; the cone's residual at `x0`; the start frame unused; the vanishing cone.
- **The existing optimizer, geometry and frontend tests** pass. The doctests of `optimizers.py` and `geometry.py`
  pass.

## Floor env

`tttt` (jax 0.4.30, numpy 1.22, Python 3.9):

- the new jax tests (`TestPaddedStartSeam`, incl. T14 (i)): 2 passed;
- `test_padded_start.py`, `test_optimizers.py`, `test_optimizers_frontend.py` and the import graph (round 1): 115
  passed, 1 xfailed; round 2 below.

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n auto`: **1,326 passed, 4 skipped, 1 xfailed, 48,669 subtests
passed** in 16 min 22 s (N6: 1,308 passed), on round 1's code. Round 1's xfail is gone in round 2 (the touched
tests rerun); the one warning is the existing `adam` one.

## Plan problems

- §2.4's "the committed convention, both layers" holds for equal old frames only. At singular-value ties (a
  `t3svd` output, a shared point) the two layers' hint-free starts differ; the aligned loop washes it out (round 2).
  This is the review's design question for Nick.
- §5.6 lists T15 for N8; the coordinator's N7 brief mentions T15. The frontend twins are N8, so T15 waits.
