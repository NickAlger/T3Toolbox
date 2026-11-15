# N7 review: the optimizer seam

Reviewer: Claude Opus 5.5, 2026-11-09. I reviewed the uncommitted working tree on top of 2d686e2d against:
- the plan: §2.4, §3.5, §3.6, §4.7, §5.5, §5.6, D-21, D-25–D-27, D-33, D-37–D-39, §6 T14 and T16, §12;
- `N7_handoff.md`;
- the session's CI rules: dead or shadowed imports and duplicate definitions; the `-W` docs build; suspicion of
  1e-14 cross-path tolerances.

## Summary

**T14(f)'s "expected failure" is not a library defect.** `ufv_shared_frame_data` and the uniform tied projection are
correct. The test compares the uniform layer's pass with a dense pass started from the RAGGED layer's start, and the
two layers' starts differ.

Why they differ:
- The tied old point has orthonormal Tucker cores (`share()` returns them; all singular values exactly 1).
- At that tie the two layers' orthogonal representations choose different bases of the same subspaces.
- The hint-free start is basis-dependent.

From the uniform layer's own start, its pass matches the dense tied pass to 1.3e-14 (frames) and 4.4e-16 (movement).
The fix is in the test (finding 1).

**A design question for Nick that matters beyond this test.** The same basis-dependence makes the two layers' hint-free
starts differ (subspace distance 0.98) for EVERY old point with orthonormal Tucker cores, unshared too. That includes
`t3svd` and down-orthogonalized points, i.e. the `x_old` the frontend's reduction to minimal ranks produces (item 1 of
"For the maintainer").

The seam itself is clean:
- style per §3.5, with no new imports in `backend/optimizers.py` (T18);
- the G-bit holds: 8 of 8 seeded runs are bitwise identical to HEAD, my own check;
- the `-W` Sphinx build passes, with no shadowed imports and no duplicate definitions in the package.

## (1) T14(f): the root cause

What I ran, all in scratch scripts (paths at the end):

1. **The tied projection itself, same frame and same raw variations, ragged vs uniform** (`n7_tied.py`):
   - at the padded frame of the zero-padded tied point: **5.7e-17** relative;
   - at an ordinary tied point's frame, which in uniform layout has junk slots, i.e. zero-size real blocks: **3.1e-16**;
   - at `t3_orthogonal_representations(x0)`: **5.4e-17**.

   Group spectra agree; the uniform layer adds only the padded zeros. Untied, the projections agree to 1e-16.
2. **The padded frame scattered GAPPY in both blocks**, with garbage in the frame and the raw variations, 20 seeds
   (`n7_gappy.py`). Uniform tied projection against ragged: **3.7e-16**.

   So `ufv_shared_frame_data` and `ufv_share_tucker_variations` are correct at rank-deficient, zero-padded, gappy
   frames.
3. **The seam's frames** (`n7_seam.py`, `n7_seam2.py`):
   - The two layers' OLD frames span the same subspaces (1.5e-14), and both are tied.
   - Their bases differ by genuine rotations, not signs. Up cores: max off-diagonal of `A_rᵀ A_u` 0.19 and 0.98.
   - The shared `x_old`'s Tucker singular values are all exactly 1.0, because `share()` returns orthonormal factors.
     At that tie, the ragged `t3_orthogonal_representations` and the uniform `ut3_orthogonal_representations`
     (pad-safe SVD) legitimately choose different bases.
   - The hint-free convention is basis-dependent: its coordinate seeds name coordinates in the neighbours' bases. So
     the two starts differ in span: **0.999**.
   - `fv_padded_frame` run on the uniform old frame reproduces the uniform start to **9.0e-16**. So the uniform start
     is right for ITS old frame; N4's guarantee ("given the same old frame") holds.
   - Unshared generic points have distinct singular values, so the bases differ by signs and rotations that do not
     matter, and the starts agree (2.6e-15). That is why the unshared comparison passed.
4. **The test with the reference fixed** (`n7_seam3.py`). Build the dense pass from the uniform layer's own old frame
   and start, read back. The uniform one-pass frame matches it to **1.30e-14**, and the movement to **4.4e-16**.
   Against the ragged start (the test as written) the distance is 0.995. That is the reported 0.31 failure, seen at
   the frame level.

**Verdict on the suspicion:** no pre-existing bug in `ufv_shared_frame_data`, and no frame built wrongly by the seam.
The test compares across layers through a basis-dependent convention.

## (2) The N7 review

### Style (checked first)

- **Geometry members** (§3.5, the documented relaxation: they may read their own masks' and the point's shapes).
  - `with_frame`.
  - `aligned_padded_frame`. The old frame is the layer's orthogonal representation, or the empty frame by D-21's rule
    per element; a mixed stack raises.
  - `require_frame_structure`. Integers and held masks only, never an orthogonalization (D-37). Every message names
    the fix.
  - `_sharing_labels` and `_structurally_zero` are small module-level helpers with annotations.
- **`backend/optimizers.py`** imports nothing new: T18's `{common, regularization, corewise}` holds. `math` was
  already imported.
  - `_misfit_gradient_at` is §3.5 verbatim, module-level.
  - `padded_start` is public, in `__all__`, with a doctest of the frontend route.
  - Options default to `None` and are forwarded only when set.
  - D-38: zero passes with an unread option raises in all four places, and `seed` is allowed.
- **`newton_cg` and `gradient_descent`.** The fourteen and six keywords are keyword-only and appended, with the
  §5.6 names and defaults. The structural errors are as §3.6 lists.
- **Labels and imports.** No plan labels in the diff. My AST scan of `t3toolbox/` finds no name both imported and
  defined, and no function defined twice. The new imports in `geometry.py` are all used. One unused import exists
  but is pre-existing: `bgeo` in the frontend `t3toolbox/optimizers.py`, outside N7.
- **The frontend mirror-test exemption.** It covers exactly the fourteen keywords and names N8 as the slice that
  removes it. Accepted: the alternative was a red suite, and N8's T15 restores the mirror.

### Correctness

- **The non-breaking rule (G-bit).**
  - Without the new keywords nothing changes: the stats and history key sets are as before, and the new keys appear
    only with their keywords.
  - I ran 8 seeded runs on HEAD (`git archive` into scratch) and on N7's tree: ragged and uniform; Gauss-Newton,
    exact with `'zero'`, exact with `'warm_start'`; and `gradient_descent`. **All 8: cores bitwise, `repr(stats)`
    equal** (times excluded).
- **The `'cone'` reference.**
  - It is `padded_start(problem, x0, None, cone_*)`, then the misfit gradient's norm at that frame in
    `with_frame(F)`'s inner product. The residual is at the zero tensor, and a vanishing value raises.
  - With `g0norm_newton` given, nothing runs: the history says `'override'` and there is no `cone_movements`.
  - The cone starts from the EMPTY frame, so it has no old basis and is layer-independent.
- **The start frame.** Iteration 0 linearizes at `start_frame` and later iterations at `geom.frame(x)`; a rejected
  trust-region trial keeps the model, as before.
- **Judgment calls.**
  - 1: see (1). The test should be corrected, not left as an expected failure (finding 1).
  - 2: the dense tied oracle varies the shared factor itself. That is right, and the ragged layer matches it at
    ≤ 1e-9.
  - 3–6: accepted.

### Findings

#### 1. MAJOR: T14(f) records a library defect that does not exist, and leaves the uniform tied pass unverified

`tests/backend/test_padded_start.py`, `TestSeamNumerics.shared_one_pass` and
`@unittest.expectedFailure test_shared_one_pass_uniform`.

**Problem.** `shared_one_pass` builds the dense reference from the RAGGED old frame and start for both layers. Its
docstring (and the handoff's judgment call 1) blames `ufv_shared_frame_data`, which is correct (section (1)).

**Fix.**
- For each layer, take the old frame and the start the layer itself uses: the ragged ones as now, and for the uniform
  layer the read-back of `ut3_orthogonal_representations(uold)[0]` and of `padded_start(..., max_passes=0)[0]`.
- Run the dense pass from those.
- Remove `@unittest.expectedFailure`, which would otherwise turn into an unexpected success, and correct the docstring.
- My check of exactly this: 1.30e-14 / 4.4e-16 against the ≤ 1e-9 bound.

#### 2. NIT: `_cone_reference` computes the residual at the zero tensor twice

Once inside `padded_start` (with `x_old=None`) and once for the norm. It costs one `point_forward`.

**Fix.** Optional: let `padded_start` take a precomputed residual privately, or accept the duplicate.

## What I ran (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2; `OPENBLAS_NUM_THREADS=1` for the scripts)

1. **Root cause:** `n7_tied.py`, `n7_gappy.py`, `n7_seam.py`, `n7_seam2.py`, `n7_seam3.py` (results above).
2. **Layer dependence of the start, unshared** (`n7_ortho.py`, 5 seeds each). Ragged against uniform hint-free start,
   subspace distance:
   - generic randn `x_old`: **3.2e-15**;
   - `t3svd(x_old)`: **0.98**;
   - `down_orthogonalize_tucker_cores()`: **0.98**.
3. **G-bit** (`gbit.py`): 8 of 8 bitwise against HEAD.
4. **Docs.**
   - `python -m sphinx -W --keep-going -b dummy docs <scratch>`: **exit 0, no warnings**; the repository was
     unchanged afterwards.
   - The AST scan for shadowed or duplicate definitions: clean.
5. **Tests.**
   - `pytest tests/backend/test_padded_start.py tests/backend/test_optimizers.py tests/test_optimizers_frontend.py
     tests/test_import_graph.py -n 6`: **115 passed, 1 xfailed** (the T14(f) uniform half), **1,447 subtests**.
   - `tests/test_dispatch.py -k TestPaddedStartSeam` (T16, T14 (i)): **2 passed**.
   - Doctests of `backend/optimizers.py` and `backend/geometry.py`: **3 passed**.
6. **Floor env** `tttt` (py3.9, numpy 1.22.4, jax 0.4.30).
   - `test_padded_start.py` and `test_import_graph.py`: **29 passed, 1 xfailed**.
   - `TestPaddedStartSeam`: **2 passed**.
   - Both 1e-14 bounds in N7's tests compare the same code path on the same inputs, not two paths: the iteration-0
     frame of `padded_max_passes=0` against the direct `fv_padded_frame` / `ufv_padded_frame` call (test_padded_start
     line 381), and the zero tensor's uniform start through both entries (line 540). Neither is a suspect cross-path
     tolerance, and both pass on both environments.

## For the maintainer

1. **The hint-free start is layer-dependent at orthonormal old points: a design decision.**
   - **What happens.** The convention reads coordinates in the old frame's bases. At a degenerate spectrum (orthonormal
     Tucker cores, all singular values 1) the ragged orthogonal representation and the uniform pad-safe SVD choose
     different bases.
   - **How much.** The two layers' starts differ in span by ~0.98 (unshared) or 0.999 (shared).
   - **When.** Exactly the inputs the pipeline produces: `x_old` after `t3svd` or `rank_adjustment_sweep`, and every
     `share()`d point.
   - **Each start is valid** for its own layer, and deterministic.
   - **What breaks.** Plan §2.4's "the committed convention, both layers" holds only given the same old frame. The
     cross-layer agreement of `padded_max_passes=0` and of short alignment runs is lost at realistic inputs. T14(e)'s
     uniform-vs-ragged check passes only because `fit_case` draws a generic randn `x_old`.
   - **The `'cone'` reference** is unaffected (it starts from the empty frame).
   - **Options:**
     - (a) Accept and document it: a start is a convention per layer, equal across layers only for equal old frames.
       Add a T14(e) case with an orthonormal `x_old` so the doc's claim is measured.
     - (b) Canonicalize the old frame's bases where both layers can agree, e.g. a sign-fixed Householder QR for the
       Tucker down-orthogonalization, which keeps an already-orthonormal basis up to signs; the convention is
       sign-covariant (N1). The bond bases (ragged QR sweeps against uniform SVD sweeps) would need the same care,
       because they also differ by rotations.
   - I lean to (a) for this arc and (b) as a later audit item. It is your call before N8's twins promise anything
     cross-layer.
2. **Judgment call 3 (the mirror exemption).** N8 must remove it, and T15 restores the keyword mirror.
3. **CI checks.** The `-W` docs build and the floor env are run above. Please run the full suite on the committed tree
   as usual.

## Counts

MAJOR 1, MINOR 0, NIT 1; and one design decision for the maintainer.

VERDICT: REVISE
