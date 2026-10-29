# S13b code review (reviewer A, CODE lane) -- main @ ede08fd

No BUG-class finding: every exact-mode path I exercised (ragged / uniform / uniform-corewise, eager + `use_jit`,
verbose TR display with validation, corewise exact with line search, the validation raises) runs and behaves as
documented. What I found is small and docstring-level. Everything in `slices/DEFERRED.md` is left out.

## WRONG-DOCSTRING

1. **`rcond` is typed `Optional[float]` but the uniform lift passes per-block host arrays.**
   - `backend/linalg.py:725` (`filtered_reciprocal`) and `:777` (`damped_pseudo_inverse`), and `backend/sharing.py:651`
     (`_tied_filter`), declare `rcond: typ.Optional[float]`.
   - `ufv_operations.py:468/481` pass `_real_rcond(...)`, an array of shape `(slots,)+C+(1,)`, and `:577` passes
     `eps*dims`, shape `C+(1,)`. It works because the array broadcasts against `s[..., :1]`, but the signature
     comment (the shape contract) is wrong.
   - Fix: `typ.Optional[typ.Union[float, NDArray]]  # scalar, or broadcastable to C+(1,) (per block)`.
2. **`fitting.py:802-803`: `apply_derivatives_model`'s signature still says ragged-only.**
   - It reads `geometry  # MANIFOLD / COREWISE` and `x: t3.TuckerTensorTrain`, but the body (`:847`) takes a
     `UniformTuckerTensorTrain`, and its sibling factories carry the `Union[...]` plus "(or the UNIFORM_* twin)".
   - This predates a62e621, but the arc realigned this signature.
   - Fix: copy `entries_derivatives_model`'s two lines.

## INCONSISTENCY

3. **`backend/geometry.py:44` overstates when `curvature_damping` raises.** The module docstring says "a value
   where there is no lift to damp raises". `ManifoldGeometryOps.curvature_precompute` (`:359-364`, uniform
   `:733-738`) validates the value, then returns `None` at `d = 1` without raising. Repro:
   `apply_model(MANIFOLD, d=1 x, ..., hessian='exact', curvature_damping=1e-3)` builds the model silently. The
   corewise geometries do raise.
   - Fix: say "on a corewise geometry", which is what `fitting.py:582/912` and the CHANGELOG already say. Or
     raise at `d = 1` too.
4. **A float32 numpy manifold fit still comes back float64 in exact mode.** This was already true at a62e621,
   so it is not an arc regression.
   - The cause is upstream of the arc: `t3_orthogonal_representations` returns the frame's O/P/Q in float64
     from float32 cores, so the sweep, the GN gradient and now the lift factors / `z` / the two-holes all
     promote.
   - The CHANGELOG `[2026.3.0]` Fixed entry ("a float32 jet pipeline ... stays float32") holds kernel by kernel
     but not end to end on `MANIFOLD`. `CorewiseGeometryOps` stays float32.
   - The same default-dtype pattern sits at `backend/tt_operations.py:190` (`Z0 = xnp.ones(...)`, no dtype),
     which every lift and T3-T3 sweep reads.
   - Fix: a ledger line next to the S3b M7 "small dtype leftovers", or pass
     `dtype=xnp.result_type(coresA[0], coresB[0])` to the zipper boundary.

## LEFTOVER

None found:
- No slice or plan names in exception messages. The only `review S9` / `review S11` mentions are comments
  from the 2026-08-22 library review, not arc slices, and none were added by the arc.
- No window-stub remnants: `grep window|stub` finds only the `abc` marker text and mc_sgd's iteration window.
- `__all__` audit over the 23 changed modules: every listed name exists, no private name is exported, and no
  new public def is missing. The only unexported public defs are the `Geometry` Protocol and
  `fitting.ragged_backend_geometry`, both pre-arc.

## Checked and clean
- **`backend/optimizers.py`:**
  - The CG step and `newton_cg`:
    - `_cg_body` / `_cg_core` are branch-free, radius `inf` gives plain truncated CG, and the radius dtype
      follows `rs0`.
    - `cg_converged` / `cg_boundary` / `cg_truncated` are mutually exclusive (verified per row on a 12-step
      exact run).
    - TR accept/shrink/expand and both floors match the docstring and the CHANGELOG. The descent guard
      counts as a boundary step.
  - Run fields and validation:
    - `NewtonInfo` defaults match what the loop sets (`regularization` is None when unregularized; step and
      `tr_*` fields are None on converged rows; `lift_*` are set only in exact mode).
    - `stop_reason` is set on the last row only. `losses[k] == history[k]['objective']` holds across rejections.
    - `tr_radius0` (finite > 0, TR only), `curvature_damping` (raises in GN mode), `step_control` defaults,
      `g0_reference` / overrides and `_check_hessian` (rejects bool) all agree with their docstrings.
    - `_require_exact_capable` reads the abstractmethod marker. Pytree registration covers `LocalModel`,
      `CurvatureState` and `ManifoldCurvatureData`; the exact mode is static by tree structure.
- **`optimizer_display.py`:** the rejected-trial table reuse is keyed on the model object, the `TR Δ ✓/✗`
  segment and the `κ σ ε` cells render correctly, and there is no off-by-one in iteration or stop rows (live
  verbose run).
- **Geometries (`backend/geometry.py`):**
  - All four exact members. `_lift_sigma_min` and its uniform twin use real-slot counts from host-`np` masks,
    with `s_g` in place of tied modes' spectra.
  - `from_frame`; `_var_masks_for` / `_masks_over_stack` broadcast over K.
  - Corewise `()` curvature data with the damping guard; d = 1 shortcut.
- **Kinds (`backend/fitting.py`, `uniform_fitting.py`):** `corewise_hessian` on all 6 + 6 kinds, ω² applied
  exactly where `transpose` applies it. The uniform overrides mask θ and θ̂ with the FRAME's (up, left) masks
  via `_frame_theta` and re-mask the output.
- **Uniform two-holes (`ut3_sampling.py`, 12 wrappers):** argument order matches the ragged kernels and the
  kind call sites. Signature shape comments are consistent (W+C residuals, K+C directions, W summed).
- **Lift and tied factors:**
  - `tv_corewise_lift[_transpose]` and the tied variants are exact adjoints (same filter, T^T sum).
    `utv_corewise_lift[_transpose]` structurally checks the companion's row splits.
  - `ufv_lift_factors` / `ufv_tied_lift_factors`: masks on host `np`, supercores on `xnp`, and the boundary
    init cast to the factor dtype.
- **T3-T3 two-hole:** `t3_linalg` uniform branches (d-prefixed contractions) and `ut3_linalg` wrappers.
- **Gauge scale (`tv_gauge_residual` / `utv_gauge_residual` `scale=`):** structural shape check,
  floor-not-replace. The frontend `_record_*_gauge_scale` covers `+ - * / neg stack_tangents` on both layers
  and records only when checks are active.
- **Chunk estimator `hessian=` plumbing:** frontend `least_squares_problem`, `newton_cg`,
  `_resolve_chunk_size`, `estimate_chunk_size` and `max_chunk_size_within`. Only uniform `probe_derivatives`
  chunks; the others resolve to None.
- **New jet scan inits:** all pass `xnp.result_type(...)` dtypes.
