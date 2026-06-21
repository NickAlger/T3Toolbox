# Handoff — `t3svd` / `rank_adjustment_sweep` redesign (session ending 2026-06-16)

This session started as "investigate the flagged `t3svd` non-minimal-ranks bug" and turned into a full
redesign of how T3-SVD and rank minimization are exposed, across **both** the ragged and uniform layers.
It is **complete, committed, and pushed to `main`** (commits `265d3d40..ab76bb4b`). This note records the
final state, what to read, and where to resume.

## ⚠️ Read the git history with care — the early commits were superseded

The first half of the session (`265d3d40`, `ac8cd300`, Slices A/B `846db101..9c4b52a1`, `d1983b74`)
built an *internal* re-tighten with a `minimize_ranks` flag and a string `assume_orthogonal`. The second
half (`3820f6df`, `e7ce8bfc`, `fe5cc958`) **replaced** that with the clean split below. So if you read the
history top-to-bottom you'll see `minimize_ranks` appear and then disappear — the final design is what
matters; don't reintroduce the flag.

## Final API (both layers)

- **`t3svd(max_tt_ranks, max_tucker_ranks, [rtol, atol — ragged only], assume_orthogonal=False)`** — the
  basic algorithm (Algorithm 10 / Oseledets TT-SVD). Orthogonalize + one L→R truncating sweep. **Always
  left-orthogonal**; returns the raw sweep ranks with **no minimal-rank guarantee** under truncation.
  `assume_orthogonal=True` skips the orthogonalization, asserting the input is already right-orthogonal
  (not enforced). Uniform has **no** `rtol`/`atol` (data-dependent shapes).
- **`rank_adjustment_sweep(direction='right_to_left')`** — the separate, opt-in lossless minimization.
  `'right_to_left'` → right-orthogonal; `'left_to_right'` → left-orthogonal. One sweep reaches minimal
  **only if the input is orthogonal in the opposite direction** (a `t3svd` result is left-orthogonal, so
  `'right_to_left'` minimizes it). Precondition **not enforced**; verify with `is_left_orthogonal` /
  `is_right_orthogonal`. Wrong direction: ragged under-minimizes (lossless), uniform is **lossy** (static
  shapes) — both documented + doctested.
- Checkers `has_minimal_ranks` / `minimal_ranks` / `is_left_orthogonal` / `is_right_orthogonal` exist on
  **both** `TuckerTensorTrain` and `UniformTuckerTensorTrain` now.

Uniform == ragged verified in tensor, ranks, AND gauge for `t3svd` and the common
`rank_adjustment_sweep('right_to_left')` path (stacked + asymmetric/bond-orphan caps).

## What to read (the design is documented)

- **[`t3svd_design_rationale.md`](t3svd_design_rationale.md)** — *why* the split, the gauge-inconsistency
  + parity argument (reversing can't fix it), single-pass-vs-robust, the ragged/uniform asymmetry. Read
  this before "fixing" `t3svd` to return minimal ranks again.
- **[`t3svd_minimal_ranks.md`](t3svd_minimal_ranks.md)** — user-facing: the orphan mechanism, what
  "minimal" means (matricization ranks of a tree TN), how to minimize. (Companion verification doc:
  `t3svd_verification.md`.)

## Key files

- `backend/t3_svd.py` — `t3svd`, `rank_adjustment_sweep`.
- `backend/ut3_svd.py` — `ut3svd`, `uniform_t3_svd` (the masked scan), `ut3_rank_adjustment_sweep`,
  `_reduce_left_to_right`.
- `backend/ranks.py` — `compute_minimal_ranks`, `compute_raw_sweep_ranks` (used by uniform `ut3svd` to
  shrink the supercore to the actual content ranks).
- `backend/t3_orthogonalization.py` / `ut3_orthogonalization.py` — `t3_orthogonality_residual` /
  `ut3_orthogonality_residual` (back the checkers).
- Tests: `tests/test_tucker_tensor_train.py` (`test_t3svd_is_left_orthogonal_not_necessarily_minimal`,
  `test_rank_adjustment_sweep`, `test_t3svd_assume_orthogonal`, the `compute_minimal_ranks` foundation
  tests), `tests/test_uniform_tucker_tensor_train.py` (same shape), `tests/test_dispatch.py` (jit).

## Open follow-ups / next steps

1. **Resume the broader uniform port.** This whole redesign was a detour from the
   `UniformTuckerTensorTrain` port — see **[`uniform_slice_handoff.md`](uniform_slice_handoff.md)** and
   `uniform_port_plan.md`. Uniform `t3svd`/`ut3svd` is now done (and the option-a/option-b divergence the
   handoff flagged is resolved). Remaining: the deferred uniform **basis/variations/tangents** (`ubv_*`,
   `uniform_*`) and supercore tangent ops, jax pytree registration / `test_dispatch` coverage for the
   uniform ops (slice 7), constructors+IO (slice 8), uniform `t3m` (slice 9).
2. **`t3m` is left as raw rounding** (Nick's call): it rounds via `t3svd`, so its output is correct and
   within the max ranks but **no longer auto-minimized**. If a minimal `t3m` is wanted later, have it call
   `rank_adjustment_sweep('right_to_left')` after its rounds.
3. **Uniform `rank_adjustment_sweep` is lossy on wrong-direction misuse** (vs ragged's lossless-partial).
   Accepted + documented as a not-enforced precondition. If that footgun ever needs removing, the robust
   alternative (orthogonalize internally → always minimal) is sketched in the rationale doc.
4. **Full suite is green** (`test_tucker_tensor_train` 64, `test_uniform_tucker_tensor_train` 33,
   `test_dispatch` 5, `test_t3m` 9, `test_manifold` 37, `test_basis_variations_format` 23,
   `backend/test_contractions` 29); module doctests clean. The `common.py` `NUMPY SCAN`/`MAP` debug prints
   were removed, so the old `grep -vE "^(RAGGED|NUMPY)"` test filter is no longer needed.
