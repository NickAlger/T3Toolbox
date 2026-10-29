# S1 + S2 review (reviewer) -- COMPLETE -- VERDICT: APPROVE (0 BLOCKER, 0 MAJOR, 1 MINOR must-fix doc-only, 6 NIT)

Base `main @ 0ec8d45`; tree uncommitted; patches `impl/S1.patch`, `impl/S2.patch`.
Reviewer scratch: `scratchpad/review_s1s2/` (own scripts, base/S1 trees).

## 0. Provenance checks (done)

- `S1.patch` applies cleanly to a fresh `git archive HEAD`; the result equals the split repo's S1 commit (`diff -rq`).
- `S2.patch` applied on top equals the working tree (t3toolbox/, tests/, CHANGELOG.md) byte for byte.
- The split repo's base commit equals `git archive HEAD` (t3toolbox/tests/docs).

## Findings (appended as found)

### 1. Math and numerics (independent re-derivation; reviewer script `review_s1s2/indep_math.py`, ~110 checks; the only failures are the 2 expected padded-restart lift-identity rows, see (b) / m1)

Oracles that share nothing with the kernels: `E v` as a dense sum of the frame's single-replacement terms (via
`t3_to_dense` only), `DPhi[theta_hat]` by the dense multilinear expansion, and `K_z theta_hat` as
`jax.jvp(jax.grad(theta -> <Z, Phi(theta)>))` on a hand-written dense `Phi` (jnp, float64).

- (a) `fv_center_factors` on minimal, Tucker-slack (nD [2,4,2] < nU 5), TT-slack (rL [1,3,7] vs rR [1,3,3]),
  both-slack d=4, d=1, d=2, a zero-padded `resize`, unsquashed boundary ranks; C = () and (2,3): `T_i ==
  zipper[i+1]`, `S_i^T == V_i U_i^T`, `H_i ==` the construction's TT variation, and (independent) the center
  reconstructs X at its orthogonality centre; spectra == dense unfoldings; `sum s(S_0)^2/||X||^2 - 1`: all
  <= 9.9e-15. **Agrees with F1/F17.**
- (b) lift identity `DPhi[Lv] = Ev` at eps = 0, gauged and ungauged: <= 2.8e-15 on minimal / Tucker-slack /
  TT-slack / both-slack / d=1 / d=2 / unsquashed. Adjoint per K element, eps_rel in {0, 1e-8, 1e-3, 0.3, 5}:
  <= 9e-15 (padded included). Last TT slot undamped at eps_rel = 10: bitwise. Symmetry of `L^T K_z L` (K_z the
  new T3-T3 two-hole) for eps_rel in {0, 1e-3, 1}: Gram asymmetry <= 7.5e-16, padded included.
  **At the zero-padded restart the identity FAILS (0.63 rel)** -- expected (the clipped, rank-growing directions
  have no first-order raw preimage); I verified `DPhi[Lv] = E(Q v)` to 4e-16 with `Q = (S^T S^{T+}, . T^+ T)`
  the projection off the clipped directions. Only the docstrings overclaim (finding m1).
- (c) `filtered_reciprocal(s, None, rcond)` is literally `_tied_solve`'s two `where`s (bitwise; the implementer's
  test also covers a padded shared restart through `_tied_solve` itself). `damped_pseudo_inverse` on a matrix
  padded in rows AND columns, eps in {None, 0, (1e-3, 0.7)}: padded block exactly 0 and real block bitwise ==
  the unpadded pinv (numpy LAPACK; not guaranteed bitwise in general, the test's 1e-15 is the right bar).
  Clip: over 200 random zero-padded restarts (d 2-4) every "zero" singular value of every `S_i^T`/`T_i` is
  exactly 0.0 (ratio to the default clip 0.0), and no pinv entry exceeds 1e8 -- the default clip
  (`dtype eps * max(m,n) * s_1`) is adequate there. Defaults: `filtered_reciprocal` `eps*k` is the plan's C.1
  text; `damped_pseudo_inverse` `eps*max(m,n)` matches `numpy.linalg.pinv` and `_tied_solve`; consistent.
  `||X||_HS = ||S_0^T||_F` is F17 (`sum s(S_0)^2 = ||S_0^T||_F^2`), exact at every frame I tried incl. slack and
  padded -- correct, and cheaper than the SVD-based form.
- (d) T3-T3 two-hole vs the jax autodiff oracle: <= 7.1e-16 on all 8 structures (incl. d=1, padded,
  unsquashed z AND theta), K = () and (2,); the one-hole gradient vs `jax.grad`: <= 5.8e-16; K=(2,2) x C=(2,3)
  stacked == per element bitwise. **Agrees with F10 (<= 1e-15).** The implementer's T5 also pins FD (O(h^2))
  and the block-sum oracle; T6 symmetry.
- F7 / E4 / E5 on the final tree: `c4` (vertical null space `|K_N th_v|` 2.4e-11 vs `|K_r th_v|` 2.5e4; FD
  252.85288182 = F6) and `c10` (1.3e-11 vs 1.3e4) and `check_full_rank_N0` (E5: 6.9e-16 / 8.4e-16) via the
  renamed copies (item 6); `check_d1_normal` (E4: 4.6e-17, 2.6e-16) unmodified. The d = 1 branch is kept in
  `_inner_product_frame_transpose` and in the sweep / from-sweep (zero tangents); d = 1 with UNSQUASHED
  boundary ranks (theta (2,3), z (3,2)) vs the jax oracle: 4.2e-16 (`review_s1s2/d1_unsq.py`).
- Re-runs: `c1` (final tree) reproduces F1; `c12` 6.3e-16/9.2e-16; `c40` 90/90 strings, 143/143 HISTORICAL
  rows; `c34` acyclic. `c2`, `c6` on the BASE tree reproduce the plan's numbers exactly (6.04e-15/6.21e-15,
  1.20e-15/1.27e-15; -393.9702957658 / -34939.8090286372; c6 <= 5.8e-16); on the final tree they raise
  `AttributeError` (item 6). The implementer's `repro_s1.py` / `repro_s2.py` reproduce their handoff numbers
  on the final tree.

### 2. Bit-identity of the refactor (reviewer `review_s1s2/bitid.py`, `bitid2.py`; base vs final trees, two processes)

`array_equal` on every array: `fv_shared_frame_data` (5 cases: d 3-4, groups (0,0,0)/(0,0,1)/(0,1,0,1),
C up to (2,3), ranks to 10 so the summed axes exceed SIMD widths), `ufv_shared_frame_data` (the same 5, packed
with extra padding), `tv_project_t3_onto_tangent_space` (7 cases incl. Tucker slack, TT slack, d = 1 unstacked
and stacked, C = (2,3), unsquashed z), and the shared projection with `shared_data` (3 cases, C up to (2,2)).
**All bitwise.** `manifold.py` / `shared_geometry.py` untouched by the diff (they call the same function).

### 3. Batching (§D)

- Stack contracts as specified: filters `C+(k)` / `C+(m,n)` with eps `C` (elementwise `where` + batched SVD,
  `'...'` -- allowed by §D for per-element work on one opaque batch); `fv_center_factors` / `fv_lift_factors`
  C -> C (grouped, C-only); `tv_corewise_lift[_transpose]` factors C, variations K+C -> K+C (C pinned by the
  factors); T3-T3 z / theta / sweep C, directions K+C -> K+C; tangent zippers chains + zippers C, directions
  K+C. The d = 1 zero tangents are shaped `K+C+(rZ, r)` from the directions. Verified per element on
  K = (2,2) x C = (2,3) (bitwise, item 1d).
- The 8 strings beyond the plan's §D list (`'Czn,Cun->Czu'`, `'Cazb,Czu->Caub'`, the env
  `'Cxa,Cxzc,Ccb->Cazb'` and its two dotted twins, `'KCazb,Caub->KCzu'`, `'Cazb,KCaub->KCzu'`,
  `'KCzu,Czn->KCun'`) are each identifiable with `''`: every one has a C-only operand of known rank that pins
  C, and K is the remainder. All 20 new rows are in HISTORICAL (c40: 143 = 123 + 20 pass; the
  supplement-contract and vocabulary-completeness tests run them). §D's "kernels record their own strings"
  covers the env/Tucker-block additions -- fine.
- **Plan gap, confirmed (record for the plan, no code action):** `tests/test_contractions_sharding.py`
  sizes every lowercase letter from its own `SINGLE` table (`self.SINGLE[ch]`, line ~111), independent of
  the interpreter's `AXIS_SIZE`; without `l/n/x/z` there the multi-device test raises `KeyError` on the 20 new
  strings. M5/§D/§H(S1) name only `AXIS_SIZE`. The implementer's extension (same sizes) is the right fix.
- NIT: in the tangent-zipper strings `j = a = k = 2` in the pre-existing `AXIS_SIZE`, so a j/k transposition
  there is caught by values (loop oracle) only, not by shape. Harmless; no action.

### 4. Protocol / pytree / jax

- `LiftFactors`: `@dc.dataclass(frozen=True, eq=False)`; registered at the module bottom under
  `if jax_available:` (flatten -> 5 children, aux `None`); `eps` a leaf. `test_jit_curvature_lift` traces
  ONCE across 2 frames x 3 dampings (damping passed traced). Two-hole from-sweep: one trace across two points.
- Direct annotation `fv_operations.LiftFactors` in `tv_operations` (no string annotations). New edges
  `fv_operations -> {contractions, linalg}`, `sharing -> fv_operations`, `tv_operations -> {contractions,
  fv_operations, t3_linalg}`, `t3_linalg -> {contractions, fv_conversions}`, `tt_operations -> contractions`.
  My AST scan of the real import lines (module level + `if` blocks) finds **no backend cycle** (the only
  cycle is the pre-existing package `__init__` <-> `optimizers` one, identical on base). `t3_linalg` does
  not import `utv_operations`; `fv_operations` imports neither `tv_operations` nor `sharing`.
  (`tt_operations -> contractions` is not in §M's list but is implied by C.3's grouped zipper strings.)
- Dispatch inferred everywhere (`tree_contains_jax` on the operands, or `contract`'s own); no `use_jax`.
  `_tt_zipper_tangent_step` is module-level and closure-free. jax-hidden subprocess: importing the backend,
  `fv_lift_factors`, the lift/transpose and the two-hole all run (`JAX_ABSENT_S1S2_OK`).

### 5. Tests

- T1 (factors, spectra, F17, eps scale), T2 (lift identity: 8 structures x 5 stacks, gauged + ungauged),
  T3 (adjoint, 4 dampings, padded included), T4 (continuity bound, padded restart finite + rank of the pinvs,
  zero point, filters, `_tied_solve` bitwise incl. a padded shared restart, dpinv incl. pad transparency),
  T5 (one-hole vs dense gradient; two-hole vs block-sum on 6 cases x 5 stacks, vs the dense multilinear
  Hessian, vs FD), T6 (Gram symmetry per C), T20 backend half (flat structures + vertical null space +
  the oracle's own Lagrangian null space). All seeded (`setUp` seeds + `default_rng`), `subTest`,
  numpy-only. The tangent zipper has its own FD test (ragged, K/C stacks, uniform == ragged).
- `_curvature_oracles.py`: every draw is the caller's `rng` (`vertical_direction`); the block-sum is a
  generalization of c2's (offsets from theta's shapes, stack-aware, d = 1 case added), not a verbatim port of
  the unseeded block. The `_trs` reference twins are S3's, correctly absent.
- **Mutation sensitivity (reviewer, `review_s1s2/mutate.sh`)** -- each red on `test_curvature.py`:
  drop the `env x BUd` two-hole term (18 failed); drop the zipper's B-direction term (4 failed); `S_i^T`
  contracted with `L_i` instead of `O_i` (same shapes on minimal frames; 24 failed); `||X||` from the wrong
  factor (24 failed); lift transpose scaled 0.5 on the Tucker slot (108 failed); Tikhonov `eps` instead of
  `eps^2` (5 failed); a 1e-9 relative perturbation of one sweep environment (30 failed).
- Dispatch rows: `test_jit_curvature_lift`, `test_jit_inner_product_two_hole` (ragged + uniform scan +
  d = 1 + the delegating projection).

### 6. Committed review scripts broken by the S2 move -- adjudication

Scope is larger than the handoff says. **13 files call the old names directly**: checks `c2`, `c4`, `c6`,
`c15`, `_restart_tr2_power_half`; critic `check_float32_cancellation`, `check_full_rank_N0`,
`check_objective_floor`, `check_restart_tr{,2,3}`, `check_small_sigma{,_signs}`. **Two more break
transitively**: `c10` (importlib-loads c4) and `c29c` (exec's `checks/_restart_tr2_power_half.py`). (`c17`
reads c6 as text and still runs.) So the E2 restart harness (`check_restart_tr*`) and the c4/c10/c15 oracles
-- what S3-S5's reviewers re-run -- are dead on the final tree.

**Pre-existing, not S2's (aside for the orchestrator):** ten more scripts -- `c20`, `c21`, `c24`, `c29`,
`c29b`, `c30`, `c31`, `c31b`, `c31c`, `c42` -- already fail on the BASE tree with `FileNotFoundError`: they
`open(os.path.join(SCR, 'check_restart_tr2.py'))` with `SCR` = the review root, but the file lives in
`critic_checks/` (the folder reorganisation broke them). The same dev-only commit could fix that path
(`os.path.join(SCR, 'critic_checks', 'check_restart_tr2.py')`); they then also need the rename below,
which they inherit from the harness file they exec.

- (a) **No alias.** The names are private, the plan says "move", and a shim in `tv_operations` for dev
  scripts would put a dead private name into shipped code of a breaking release. The implementer is right.
- (b) **Recommend the one-line fix now, by the orchestrator, as a dev-only commit right after S2** (not a
  DEFERRED note): the scripts are the plan's [V]/[E] evidence and are needed again within the next three
  slices. Verified on a copy (`review_s1s2/devfix/`): this single command makes c2, c4, c6, c10 and
  `check_full_rank_N0` reproduce the plan's numbers on the final tree:

  ```
  cd dev/curvature_impl_review && sed -i \
    -e 's/tv\._project_t3_rebasis_step/t3_linalg._inner_product_rebasis_step/g' \
    -e 's/tv\._project_t3_variations_step/t3_linalg._inner_product_variations_step/g' \
    -e 's/^import t3toolbox\.backend\.tv_operations as tv$/import t3toolbox.backend.tv_operations as tv; import t3toolbox.backend.t3_linalg as t3_linalg/' \
    $(grep -l '_project_t3' checks/*.py critic_checks/*.py)
  ```
  plus one README sentence ("S2 moved `tv._project_t3_*` to `t3_linalg._inner_product_*`; the scripts are
  updated"). If the orchestrator prefers not to touch the record, the fallback is a DEFERRED/S13 note that
  lists all 15 S2-broken scripts (not only c2/c4/c6/c10) plus the 10 path-broken ones, so S3-S5 know to run
  them on the base tree.

### 7. House style, CHANGELOG, separability

- Signatures: three columns, one argument / return element per line, `len=d, elm_shape=...` grammar,
  module docstrings and `__all__` updated in every touched module; references cited (Absil-Mahony-Trumpf
  2013, Psenka-Boumal arXiv:2011.13395, Alger et al. Section 6.3); no `dev/` paths in shipped text; no
  formatter churn (the diff touches only the new code and the two extraction sites).
- CHANGELOG: one `Added` bullet per slice, as S0c-S0e did -- **acceptable**; it gives S13 a skeleton. (One
  wording fix rides with m1, one NIT n2.)
- `sphinx -W --keep-going` (sphinx 9.2.0 + autoapi in a scratch venv, on a copy of the final tree): **exit 0,
  no warnings** (the implementer could not run it).
- Separability: see section 8 (S1-only tree).

## Findings

### BLOCKER
None.

### MAJOR
None.

### MINOR

- **m1 -- the lift's exactness is overclaimed at rank-deficient frames.** `tv_corewise_lift`'s docstring
  ("for **any** coordinates `v` ... on every frame `t3_orthogonal_representations` builds, minimal or not"),
  `fv_lift_factors`' "Exactness" bullet ("exact on every frame from `t3_orthogonal_representations`, minimal or
  not") and the S1 CHANGELOG bullet ("exact at zero damping on every orthogonal frame, minimal or not") are
  false at a zero-padded restart, which IS a `t3_orthogonal_representations` frame: there `DPhi[Lv]` misses
  `Ev` by 0.63 relative (reviewer `review_s1s2/padded_lift.py`). What holds -- verified to 4e-16 -- is
  `DPhi[Lv] = E(Qv)` with `Q = (S^T (S^T)^+ V_i, H_i T_i^+ T_i)`, the projection off the clipped directions;
  `Q = I` exactly when every `S_i^T` has full row rank and every `T_i` full column rank (minimal, Tucker-slack
  and TT-slack frames of a generic point: what F2/F3 measured). The plan's B ("on every frame from
  `t3_orthogonal_representations` the lift is exact") carries the same imprecision; F4 only claims finiteness
  for the padded case. **Fix (wording only):** in the two docstrings and the CHANGELOG bullet say "on every
  orthogonal frame whose factors have full rank -- minimal or slack (`S_i^T` wide, `T_i` tall); at a
  rank-deficient point (a zero-padded restart) the clipped, rank-growing directions have no first-order
  preimage and the lift drops them: `DPhi[Lv] = E(Qv)`, `Q` the projection onto the factors' ranges". A T2/T4
  row pinning `DPhi[Lv] = E(Qv)` at the padded restart would make the statement tested (optional).

### NIT

- n1 `fv_lift_factors` signature: the two scalar comments (`# RELATIVE...` at col 57, `# relative clip` right
  after `= None,`) are neither in the nested-tuple comment column (52) nor in their own aligned group -- the
  staircase `signature_style.md` rules out. Align them, or blank-line-separate the scalar group.
- n2 CHANGELOG S2 bullet: "now gauges that **shared** kernel" -- in this library "shared" means SF-T3 sharing;
  say "now gauges the same kernel" (or "delegates its ungauged core to it").
- n3 `fv_lift_factors`' negative-damping guard checks only Python `int`/`float`; a negative `np.float32` or a
  0-d array slips through (harmless numerically -- the filter uses eps^2 -- but `LiftFactors.eps` is then
  negative). `np.ndim(cd) == 0 and not tree_contains_jax(cd) and float(cd) < 0` covers numpy scalars too.
- n4 `_check_lift_lengths(frame, variations, ...)`: unannotated parameters (private helper; fine to leave).
- n5 `tv_project_t3_onto_tangent_space` (public) calls `t3_linalg._inner_product_frame_transpose`
  (private, cross-module) -- exactly as plan C.3 specifies, so no change; recorded only because the
  ungauged `E^T z` at an arbitrary frame is a capability a raw-`.data` user cannot reach publicly (the
  backend razor). A candidate for the deferred ledger, not for this slice.
- n6 Tangent-zipper interpreter rows: `j = a = k = 2` in the pre-existing `AXIS_SIZE` (section 3).

## 8. Runs (last; final tree unless stated; 2 CPU cores, `-n 2`)

| run | result | implementer |
|---|---|---|
| `tests/backend/test_curvature.py` | 22 passed / 270 subtests | 22 / 270 |
| curvature + interpreter + manifold + sharing | 165 passed / 1,846 subtests | -- |
| `tests/test_contractions_sharding.py` (standalone, 4 virtual devices) | 1 passed / 716 subtests | 1 / 716 |
| `test_dispatch -k "curvature_lift or inner_product_two_hole or shared"` | 4 passed | -- |
| `pytest --doctest-modules t3toolbox` | 206 passed | 206 |
| `getting_started.rst` + every `docs/*.md` / `docs/contributor/*.md` page (minus `doctest_style.md`) | OK | OK |
| `sphinx -W --keep-going` (scratch venv, sphinx 9.2.0) | exit 0, no warnings | not run |
| full suite, half A (`tests/backend`, dispatch, fitting, optimizers_frontend, oracle_sweep, interpreter) | 319 passed, 1 skipped / 15,839 subtests | |
| full suite, half B (the rest) | 634 passed / 27,940 subtests | |
| **full suite total** | **953 passed / 1 skipped / 43,779 subtests** | 953 / 1 / 43,779 |
| slow tier `T3TOOLBOX_SLOW_TESTS=1 tests/test_oracle_sweep.py`, run 1 | **1 failed** (O2 `grad_fd` 1.05e-6, 4.9e-6 vs `< 1e-6`), 4 passed / 12,790 | 5 / 12,790 |
| slow tier, run 2 | 5 passed / 12,790 subtests | |
| **S1-only tree** (`S1.patch` on a fresh `git archive HEAD`): full suite | 943 passed / 1 skipped / 43,531 subtests | 942 (333 + 609) |
| S1-only: module doctests / sharding / doc pages / `sphinx -W` | 203 / 1 (666) / OK / exit 0 | 203 / 1 (666) |

**The slow-tier failure is pre-existing flakiness, not S1/S2.** `sweep_models.run_case` draws the tangent `p`
from `geom.randn` and `A0` from `TuckerTensorTrain.randn`, i.e. the UNSEEDED global numpy RNG (nothing in
`TestO2ModelOracleSweep` seeds it), and the Richardson-FD check `grad_fd < 1e-6` is crossed on some draws.
Reviewer `review_s1s2/o2_seeded.py` runs `sweep_models.main()` with the global RNG seeded: seed 0 gives the
SAME single failure on the BASE tree and the final tree (`(2, ragged, manifold, probe_derivatives, 2)`,
`grad_fd = 9.391905244294256e-06`, identical to all digits); seed 1 gives none on either. So the S1/S2 code
does not move a single number on that path, and the "tests are per-test seeded" claim does not hold for the O2
slow tier. **Aside for the orchestrator (outside S1/S2; the slow tier is a REQUIRED release-gate step, S13):**
seed it (e.g. `np.random.seed(seed)` at the top of `run_case`, or a `setUp` in `TestO2ModelOracleSweep`),
and/or give `grad_fd` a scale-aware bar; record in HANDOFF/DEFERRED.

S1 alone is green (separable); S1 + S2 equals the working tree byte for byte.

## Must-fix before commit (doc-only, fold into the S1 commit; no re-review needed)

1. **m1** -- qualify the lift's exactness claim (in `tv_corewise_lift`'s and `fv_lift_factors`' docstrings and
   the S1 CHANGELOG bullet) to frames whose factors have full rank, and state what holds at a rank-deficient
   (zero-padded restart) frame: `DPhi[Lv] = E(Qv)`, the clipped rank-growing directions dropped.

Optional: n1-n3 (signature comment column, "shared kernel" wording, numpy-scalar negative guard).

## For the orchestrator (not slice-agent actions)

- Item 6: apply the one-line `sed` to the 13 `dev/curvature_impl_review/` scripts (+ README sentence) in a
  dev-only commit after S2; optionally also fix the 10 pre-existing `SCR/check_restart_tr2.py` paths.
- Plan gap (item 3): `tests/test_contractions_sharding.py`'s `SINGLE` table also needs every new lowercase
  letter (M5 named only `AXIS_SIZE`); note it beside M5 so S3/S8's new letters (if any) extend both.
- Pre-existing O2 slow-tier flakiness (section 8): seed the sweep before the S13 release gate.
- Draft commit messages in the handoff are accurate; the S1 one should mention the m1 wording if amended.

VERDICT: APPROVE
