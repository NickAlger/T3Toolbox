# P13b: the derivative methods removed — handover

The second commit of P13, built to P13a's handover list and the coordinator's two carried items. Written by the
implementer, 2026-11-21, against `graded-messages` at `986247bb` (P13a committed). Nothing is committed.
`docs/chunking.md` is deleted in the working tree; it was staged by `git rm` and then unstaged, so it shows as
deleted, not staged.

## 1. What changed

**The 24 methods are gone:** the six `*_derivatives`, `*_derivatives_transpose` and `*_corewise_derivatives_transpose`
of each of `TuckerTensorTrain`, `T3Tangent`, `UniformTuckerTensorTrain` and `UT3Tangent`. Along with them:
- their section banners;
- the four frontend modules' imports of `sampling_derivatives`, now unused;
- three `See Also` entries that named them (now `probe_corewise_tangent` and `apply_corewise_hessian`, or dropped).

**Callers in the tests.** One parser-based rewrite moved every call onto a new test-only helper,
`tests/_derivative_data.py`:
- `dd.forward(obj, kind, sample, pp, n)` is the merged method at `affine_jet(sample, pp)` and `(n,)`, then
  `taylor_to_derivatives`. Entries is apply at the one-hot jet.
- `dd.tangent_transpose(cls, kind, residual, sample, pp, frame, n, sum_over_probes)` and
  `dd.corewise_transpose(x, kind, ...)`: the old transposes took the residual in the derivatives' convention
  (`D_t = t! T_t`, so `D_t^T c_t = T_t^T (t! c_t)`). They are the merged transposes of `taylor_to_derivatives` of the
  residual.
- A packed probe vector keeps its form. `chunk_size` is accepted and ignored.

This is the plan's prescription for callers that still use the old data form. The helper's names hold no removed
name, for P20's grep.

The calls moved, 162 in all:
- `test_uniform_probing` 52;
- `o1_uniform` 21;
- `o1_ragged` 18;
- `test_dispatch` 18;
- `test_fitting` 11;
- `test_manifold` 11;
- `test_tucker_tensor_train` 9;
- `test_probe_derivatives` 6;
- `test_optimizers` 4;
- `sweep_models` 3;
- `test_optimizers_frontend` 3;
- `test_uniform_fitting` 2;
- one each in `test_chunk_size_estimator`, `test_padded_start` and `test_zero_reference`;
- by hand: one `pairs` table in `test_uniform_probing` that held the static methods by reference, and the subprocess
  script of `test_jax_absent`, which now uses an inline `taylor_to_derivatives` and `affine_jet`.

**Verified before the rewrite:** the helper against the old methods, while they still existed:
- all four classes, the three kinds, `n` = 1 to 3, both values of the sum flag, `K = (2,)` and `C = (2,)`;
- a packed probe sample;
- worst error 4.7e-16, with the uniform padding included.

**Doctests:**
- `t3toolbox/optimizers.py` (2) and `t3toolbox/fitting.py` (1), and the pages: `fitting_and_optimization.md`, 5 calls
  through one page helper `derivative_data`, defined once in the first block; and `numerical_contracts.md`, 2;
- each now makes its data as `taylor_to_derivatives(kind, (n,), X.<kind>(affine_jet(ww, pp), (n,)))`, through
  `t3toolbox.fitting`'s re-exports;
- the pages pass `python -m doctest`;
- `naming_conventions.md` names `v.apply_derivatives(...)` in prose only (P19).

**The examples.** The six examples are rewritten:
- **Dense ground truth.** The per-sample loops over `pd.dense_*_derivatives` became one call of `dense_apply` or
  `dense_probe` at `affine_jet(ww, pp)` and `(order,)`, then `taylor_to_derivatives`.
- **Helpers renamed,** with no removed name in them: `dense_apply_jets` (in the three apply examples) and
  `dense_probe_jets`; `dense_jets` was already clean.
- **Forward and transpose.** The apply examples' operator calls `Z.apply(jet, (order,))` and
  `T3Tangent.apply_transpose(frame, jet, taylor_to_derivatives(r / s_vec), (order,))`.
- **The rest.** The probe examples' misfit lines and `fit_probe_display.py` use a local `derivative_data`.
- The docstrings no longer mention the removed methods or `sum_over_probes`. No example imports
  `sampling_derivatives` any more.

**`docs/chunking.md` is removed,** with:
- its `design_notes.rst` entry;
- 3 markdown links in `fitting_and_optimization.md`, and 2 links plus one file-name mention in `glossary.md`;
- the `:doc:` links: 3 in `optimizers.py` and 4 in `backend/sampling_derivatives.py` (the one in `manifold.py` went
  with its method).

The plain `docs/chunking.md` mentions in code comments wait for P19, as planned.

**`tests/test_probe_derivatives.py` is not folded.** Its frontend calls moved to the helper. Its subject is the
backend `sampling_derivatives` kernels (`pd.t3_*_derivatives`, `pd.tv_*_derivatives_transpose`), which stay until
P20, so the file goes with them there. The dense ground truth for the merged methods is the new conversion pin
below.

## 2. The carried items

1. **Nested-list `index`:** the test now covers the five entries methods it missed:
   - `TuckerTensorTrain.entries_corewise_hessian`;
   - `UT3Tangent.entries_transpose`;
   - `UniformTuckerTensorTrain.entries_corewise_transpose`, `entries_corewise_tangent` and `entries_corewise_hessian`.

   All 13 entries methods are now covered.
2. **The derivative-conversion pin** (`TestTheDerivativeConversion`, in `test_frontend_sampling.py`). Its reference
   uses neither `affine_jet` nor `taylor_to_derivatives`:
   - the `t`-th derivative in `s` of the dense tensor contracted with `w_j + s p_j`, every mode for apply and all but
     one for probe;
   - computed by an exact polynomial fit of degree `d` on `d + 1` points, times `t!`, at `n` = 1 to 4 (slot 4 > d
     vanishes).

   It is compared with `taylor_to_derivatives` of the merged method and of `dense_probe` / `dense_apply`, to 1e-10.
   It survives P20.

## 3. The examples, old against new

All six ran in a worktree at `986247bb` and in the new tree; every run exits 0.
- **Ground truth:** the new dense helpers equal the old loops to rounding (apply 6.2e-16, probe 6.4e-16).
- **`fit_probe_display`:** output identical.
- **`fit_hilbert_uniform_probe_derivatives_newton_cg`:** one Newton count differs (10 against 9), with the same errors.
- **The SGD and rank-adaptive examples** (the three apply ones, `fit_shared_factors_jetted_probes`): their final digits
  and some rank picks differ. They are stochastic or rank-adaptive, so rounding-level differences in the forward and
  transpose (now the engine) are amplified. The conclusions they print (best level, noise floor) agree.
- **For the coordinator:** `fit_hilbert_from_apply_derivatives.py`'s fit time went from 5.8 s to 10.9 s. Its operator
  now makes the jet and runs the engine through `T3Tangent.apply` and `apply_transpose` on each minibatch, where the
  old methods ran the dedicated derivative kernels. I did not profile it. It is the first place the engine's
  apply-with-jets cost shows at the frontend, and might belong with the performance work, not with P13.

## 4. Mutation pass (scratch copy, deleted)

6 mutants, all killed by `tests/test_frontend_sampling.py`:
- the five entries methods above without the list conversion;
- the jets' factorial off by one (the conversion pin).

## 5. Gate

The full gate at `-n 2` (logs in `scratchpad/gate_P13b`) passed every step:
- **suite**: 1836 passed, 5 skipped, 1 xfailed, 90,493 subtests, in 1928 s; peak 8.3 GB; **lowest MemAvailable
  3678 MB**;
- **doctest-modules**: 315 (the removed methods' examples are gone);
- **quickstart**: pass;
- **pages**: 47 (`chunking.md` is gone);
- **examples**: 14;
- **sphinx**: 0 warnings.

On the floor `tttt`: `test_frontend_sampling`, `test_manifold` and `test_jax_absent` pass (74), and so do the two
edited pages' doctests.
