SIGN OFF: the merged signatures, the new corewise methods, the checks and the rewritten call sites are right. Two small items carry into P13b, neither a silent wrong answer.

# P13a review, round 1: the frontend sampling signatures

Reviewer, 2026-11-21. This reviews the uncommitted P13a on top of `f827a784`, against the plan's P13 and 2.5, my
design review, and the rulings in `LOG.md`, including the maintainer's on finding A (positional `residual`,
documented). Everything was run single-process. The mutant copies have been deleted, and the working tree is untouched
apart from this file.

## Runs

| Run | Result |
| --- | --- |
| `test_frontend_sampling`, `test_tucker_tensor_train`, `test_manifold`, `test_uniform_probing`, `test_fitting` (current env) | 229 passed, 25,440 subtests |
| The module doctests of the five changed modules | 133 passed |
| `test_frontend_sampling` on the floor | 11 passed |
| The full gate (the coordinator's report) | passed, 1,835 |

## The settled design, point by point

**The signatures.** All four classes follow N16:
- the sample;
- what is applied;
- `order=()` (none on entries);
- the sum flag, with `True` on the corewise and tangent transposes and `False` on the ambient ones.

The six new corewise methods per tensor-train class are one backend call each, and the tests show them bitwise equal
to their functions. The `*_derivatives*` methods are untouched, as split.

**`check_sample`.** It runs in every sample-taking method, always. On the uniform classes it runs on a ragged sample
only, and the engine checks a packed one. My review agreed with this. The ambient transposes read `N` off `ww`.

**The direction guard.** It is generalized as I asked: a pair of core sequences (or two supercores on the uniform
class), and a `TypeError` naming `y.data` / `y.data[:2]` otherwise. The tests cover a `TuckerTensorTrain`, a
`UniformTuckerTensorTrain`, a `T3Tangent`, three parts, and two arrays on the ragged class.

**`np.asarray` on entries.** It is in all 13 entries entry points, each followed by `check_sample`.

**Finding A, as the maintainer ruled.** The right-and-wrong doctests at `d = 3` are in place, each checking the
adjoint relation at the intended sample:
- `t3_probe_corewise_transpose`;
- `t3_probe_ambient_transpose`;
- `TuckerTensorTrain.probe_corewise_transpose`;
- `TuckerTensorTrain.probe_ambient_transpose`.

The swap test (`test_swapping_the_sample_and_the_residual_changes_the_result`) guards against symmetric test
problems. The CHANGELOG note is P19's.

**The 98 rewritten calls.** I sampled every `transpose` line of the test diff and the hand edits. The mapping is
right in every case:
- the arguments are moved into N16's order, with the residual by keyword;
- an old explicit `sum_over_probes=X` becomes `sum_over_samples=X` (the same polarity);
- an old default (`False`) on the corewise and tangent transposes becomes an explicit `sum_over_samples=False`;
- an old explicit `True` becomes the new default;
- the ambient transposes keep their default `False` with no flag added.

The hand edits are right too:
- the `UT3Tangent` hygiene pairs compare clean against dirty at one flag;
- `test_fitting`'s reference transposes use the summed default, as their old `True` did;
- the negative-index wrap in `test_tucker_tensor_train` passes `sum_over_samples=False` on both sides.

## Mutants (13, mine): one per index conversion

I replaced `_as_index(index)` with `index`, one site at a time, and ran `test_frontend_sampling.py`:
- **8 killed:** `T3Tangent.entries` and `.entries_transpose`; `UT3Tangent.entries`;
  `UniformTuckerTensorTrain.entries`; `TuckerTensorTrain.entries`, `.entries_ambient_transpose`,
  `.entries_corewise_transpose`, `.entries_corewise_tangent`.
- **5 missed:**
  - `UT3Tangent.entries_transpose`;
  - `UniformTuckerTensorTrain.entries_corewise_transpose`, `.entries_corewise_tangent`, `.entries_corewise_hessian`;
  - `TuckerTensorTrain.entries_corewise_hessian`.

The code at the five missed sites is right, which I read. The nested-list test simply lists 8 of the 13 entry points.
Losing a conversion would be loud, a `ValueError` from `check_sample` on a list, not silent.

## Carried into P13b

1. **Add the five missed entry points to `test_entries_takes_no_order_and_converts_a_nested_list`**, so all 13 are
   pinned. It is a one-line addition each.
2. **The derivative-conversion pin, as noted for P13b:** a test, independent of `affine_jet` and
   `taylor_to_derivatives`, that outlives P20. For example, the `n`-th derivative in `t` of `<T, ⊗_j (w_j + t p_j)>` by
   exact polynomial interpolation, against `taylor_to_derivatives` of `dense_probe` and `dense_apply` at
   `affine_jet(ww, pp)`.
