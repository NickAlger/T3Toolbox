# Critic pre-checks for round 5 (written before plan_v5.md exists)

These are facts v5 must address, measured read-only. All runs used `python -I -B`.

## P1 — the `g0_reference='zero'` default

### P1a — on COREWISE the zero reference is identically 0 (`check_zero_reference_geometries.py`)

| geometry | ‖g(zero tensor)‖ |
|---|---|
| ragged MANIFOLD | 3.80e3 |
| uniform MANIFOLD | 4.99e3 |
| ragged COREWISE | **0.0** |
| uniform COREWISE | **0.0** |

The corewise map is multilinear in the cores, so for d ≥ 2 its Jacobian at all-zero cores vanishes. Today's fallback
(`computed_g0norm = gnorm if gnorm > 0 else 1.0`, `backend/optimizers.py:716`) would then silently use 1.0 as the
reference: an arbitrary absolute scale. **v5 must say what `'zero'` means on COREWISE.** It could be `'warm_start'`,
an error, or a documented substitute.

### P1b — the zero reference is not intrinsic to the problem (`check_zero_reference_invariance.py`)

- Rotating mode 0 by an orthogonal Q (an exact isometry: same data, same landscape) changes ‖g(0)‖ from 8.70e3 to the
  range 5.71e3–9.12e3, a 1.6× spread.
- A nonzero start's ‖g‖ is invariant to the last digit (3.973169e3 both ways).

The cause: the frame at X = 0 is LAPACK's fixed completion, which does not rotate with the problem. The g0 plan's own
§3 "trap" (layer-dependent) is the same phenomenon. So "the problem's intrinsic data-gradient scale"
(`zero_reference_g0_plan.md` §4.2) is only approximately true: it holds to an O(1) factor, which is harmless for a
relative stop but not a precise scale.

### P1c — the default breaks the O2 ragged-vs-uniform pairing (`check_zero_reference_layers.py`, `check_zero_reference_o2_pairs.py`)

Zero references differ by layer at d = 3: 8 % (apply), 23 % (probe) and 33 % (probe_derivatives); they are equal at
d = 2.

Simulating the default on the O2 optimizer-sweep pairing (d = 3, manifold, unshared, 6 Newton steps, each layer using
its own zero reference) gives these ragged-vs-uniform `newton_losses` errors:

| kind | today | zero default |
|---|---|---|
| entries | 2.0e-15 | **5.2e-4** |
| probe | 2.3e-15 | **3.9e-3** |
| apply | 1.3e-15 | 1.3e-15 |
| apply_derivatives | 3.5e-15 | 3.5e-15 |
| probe_derivatives | 4.9e-15 | 4.9e-15 |

The pair threshold is 1e-7 (`tests/oracle_sweeps/sweep_optimizers.py:143`). Those files are verbatim copies that must
stay byte-close (their README), so the O2 always-on and slow tiers would fail unless they pin
`g0_reference='warm_start'` (or a common float), which needs an edit to the copies or to the driver.

v5 must:

- name the fix (the driver passes `g0_reference='warm_start'`; the README edit-policy exception is recorded); and
- audit every test and example that pairs ragged against uniform trajectories.

Mechanism: CG's tolerance η·‖g‖ changes with g0, so CG stops at a different iterate.

## P2 — the frontend rename

- `t3toolbox.LocalModel` does not collide with anything in the package namespace: backend `LocalModel` and `Problem`
  are not re-exported (`t3toolbox/__init__.py:39-48`).
- Blast radius of `GaussNewtonModel` (grep, excluding `dev/archive`): `t3toolbox/fitting.py` 27, `tests/test_fitting.py`
  10, `tests/test_dispatch.py` 8, `CHANGELOG.md` 6, backend `optimizers.py` 5 (docstrings), `docs/fitting_and_optimization.md`
  4, `docs/contributor/parameters_not_closures.md` 4, `CLAUDE.md` 4, `tests/backend/test_optimizers.py` 3,
  `examples/fit_hilbert_from_apply_derivatives.py` 3, `t3toolbox/__init__.py` 2, `docs/release_notes.md` 2, plus 1 each
  in `tests/test_sharing.py`, `tests/test_oracle_sweep.py` (docstring), `tests/oracle_sweeps/sweep_models.py`
  (**docstring only**; byte-close copy), `shared_geometry.py`, `manifold.py`, `backend/fitting.py`, `user_guide.rst`,
  `numerical_contracts.md`, `numerical_contract_catalog.md`, `fitting_internals.md`, `batching_and_stacking.md`,
  `api_reference.rst`, `paper_scope.md`.
- T3Polynomial: 0 code references; 3 in `dev/t3toolbox-upstream-notes.md`.
- Its code uses `tfit.probe_derivatives_model` (2×), `topt.newton_cg` (2×), `topt._geometry_ops` (2×), `topt._setup`
  (1×), `bopt.least_squares_problem` (2×) and `bfit.probe_derivatives_kind` (2×).

## P3 — the T3Polynomial g0 machinery the default would supersede

- `t3polynomial/fitting.py:86-112` (`_zero_gradient_norm`, using `_geometry_ops`), `:189-197`, `:316-345`, `:476-501`
  (`gtol_ref`, and `g0norm_newton=None` as the warm-start spelling).
- `tests/test_fitting_selection.py:60-91`.
- **Semantics collision:** today T3Polynomial passes `g0norm_newton=None` to mean "warm start". If v5 makes `None`
  mean "use `g0_reference`" (default `'zero'`), that call silently flips to the zero reference.
  `test_fitting_selection.py:91` (`st_warm['newton'] >= st['newton']`) and the warm-start warning path would then
  test nothing.

## P4 — the jax-absent test

`tests/test_jax_absent.py` is a subprocess script that hides jax and runs `newton_cg(use_jit=True)`. "Classes added to
`test_jax_absent.py`" should mean importing every module that registers a new pytree (`fv_operations`, `geometry`,
`optimizers`, `fitting`) and constructing each new class there. A bare import is not enough.

## P5 — "production kernels are banded recurrences"

The existing production probe_derivatives transpose already mixes forms:

- the tildes are the scanned order-convolution (`_adj_sweep_scanned`), not bidiagonal;
- `compute_dxi_tilde_jets` contracts `trs` densely (`sampling_derivatives.py:1640-1655`);
- `assemble_tt_variation_jets` delegates to `assemble_tt_variation_jets_trs` per W-chunk (`:1942-1990`).

So "dense trs only as `_trs` test references" contradicts today's production code (`deferred_and_rejected.md:85-96`:
"the genuinely-full convolutions (eta/deta/tilde) scan the summed order axis instead"). v5 must state the rule as the
house rule actually is:

- bidiagonal recurrences where the input jet is affine;
- an order scan where the convolution is genuinely full;
- otherwise mirror whatever the GN kernel of the same term uses.

Applying the rule literally would force the dotted kernels to beat the GN kernels they differentiate.
