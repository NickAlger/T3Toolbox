# N8 handoff: the frontend twins (review stop)

Slice N8 of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§3.6, §5.7, D-8, D-9, D-35). Implementer: Claude Opus
5.5, 2026-11-10, on top of 46f7b7fd (N7 committed). Everything is uncommitted; nothing outside the repository was
touched. The frontend carries the descriptions, validation and dispatch; the backend stays bare.

## Files

| file | change |
|---|---|
| `t3toolbox/frame_variations_format.py` | `T3Frame.padded` (staticmethod); `fv_aligned_padded_frame` (in `__all__`); the private `_old_frame_data` and `_structurally_zero`. |
| `t3toolbox/uniform_frame_variations_format.py` | `UT3Frame.padded` (staticmethod; returns `(frame, split)`); `ufv_aligned_padded_frame` (in `__all__`); the private `_frame_of` and `_old_frame_data`. |
| `t3toolbox/manifold.py` / `t3toolbox/uniform_manifold.py` | `T3Tangent.align_padded` / `UT3Tangent.align_padded`. |
| `t3toolbox/tucker_tensor_train.py` / `t3toolbox/uniform_tucker_tensor_train.py` | `oversampled_ranks(p, sharing=None)`. |
| `t3toolbox/optimizers.py` | `newton_cg`'s fourteen keyword-only keywords, mirrored exactly (names, defaults, order). A docstring section with a doctest. The private `_padded_start_keywords` and `_old_point_data`. `gradient_descent`'s docstring covers `'cone'` and the `cone_*` options through its `**kwargs` (§5.7: its signature is unchanged). |
| `tests/test_optimizers_frontend.py` | N7's exemption removed: the keyword-mirror test passes unexempted. |
| `tests/test_aligned_padded_frame_frontend.py` | **New**: T15 and the frontend half of T14 (h) (judgment call 1). |
| `tests/test_dispatch.py` | `TestAlignedPaddedFrameFrontend`: the frontend dispatch rows. |

Every twin has a user-level doctest.

## The twins

- **`T3Frame.padded(x_old, tucker_ranks, tt_ranks, sharing, seed)`**: `fv_padded_frame` of `x_old`'s frame.
- **`UT3Frame.padded(...)`**: `ufv_padded_frame`, returning `(UT3Frame, split)`.
- **The loop twins `fv_aligned_padded_frame` / `ufv_aligned_padded_frame(x_old, …, hint, project=None, …, use_jit)`**.
  - Projection by the hint's class:
    - ragged: a `TuckerTensorTrain` → `tv_project_t3_onto_tangent_space`; a dense array of `x_old`'s shape →
      `tv_project_dense_onto_tangent_space`;
    - uniform: a `UniformTuckerTensorTrain` with `x_old`'s `N` → `utv_project_ut3_onto_tangent_space`.
  - Errors:
    - the other layer's hint → TypeError naming `.to_t3()` / `from_t3(hint, N=…)`;
    - a wrong `N` → ValueError naming `from_t3(…, N=)`;
    - anything else without `project` → TypeError.
  - A custom `project` receives the raw frame data.
  - The uniform twin's default `level_widths` is `ufv_frame_widths(x_old.N, max target Tucker rank, max target TT
    rank)`.
- **The old point, one rule for every twin** (`_old_frame_data`).
  - Layer check.
  - A structurally zero `x_old` (any Tucker or interior TT rank 0, per element; a mixed uniform stack raises) is the
    empty frame, with no reduction.
  - Otherwise the reduction to minimal ranks: the three methods on the ragged layer (`get_minimal_ranks`, `t3svd`,
    `rank_adjustment_sweep`), `uniform_minimal` on the uniform one. Then its orthogonal representation.
  - Safe mode: tied factors under `sharing`.
- **`T3Tangent.align_padded` / `UT3Tangent.align_padded`**: one backend step on the tangent's own frame and
  variations.
- **`newton_cg`** (`_padded_start_keywords`).
  - Only set options are forwarded, so no new keyword means the old call.
  - A corewise geometry with `start_frame`, `padded_from` or `'cone'` raises.
  - `start_frame` must be the layer's frame class. In safe mode it must be orthonormal and represent `x0` after its
    reduction. Under `use_jit` its arrays are converted to jax.
  - `padded_from` (`_old_point_data`): the same layer, shape, stack and `N` as `x0` (errors name the conversion). A
    structurally zero one goes to the backend as rank-0 data. Otherwise, in safe mode: tied factors under the
    geometry's sharing, and `x0.allclose(padded_from)`. Then it is reduced as the setup reduces `x0`.
  - The rest is the backend's (both keywords, group options, zero passes).

### Judgment calls

1. **One new test file** (`tests/test_aligned_padded_frame_frontend.py`) instead of spreading T15 over the seven
   frontend test files the plan lists. The twins form one feature across the frame, tangent, tensor and optimizer
   modules, and their cross-layer and zero-point cases share setup. The keyword mirror stays in
   `test_optimizers_frontend.py`.
2. **The frontend doesn't canonicalize a structurally zero `padded_from`** to `TuckerTensorTrain.zeros(...)`. Its
   data goes to the backend as is, where the members apply the same test (N7, D-21). The plan's "convert first"
   exists to skip the reduction (which breaks at rank 0) and the orthogonalization (which raises), and both are
   skipped. The frame twins build the empty frame directly. T14 (h)'s frontend half confirms both spellings give the
   same start.
3. **`gradient_descent`**: per §5.7, its signature is unchanged and the cone options reach the backend through
   `**kwargs`. Only its docstring and its `g0_reference` comment changed.
4. **The dispatch file releases jax programs after its heavy padded-frame tests** (see the full suite). It is test
   hygiene for the `-n 3` rule. It touches tests from N2–N7 too, one cleanup line each.

## Verification

- **T15, twin ≡ backend call (bitwise):**
  - both frame twins, incl. a stack C = (2,), a shared ragged point (tied factors bitwise) and per-element uniform
    targets;
  - both loop twins, ragged with T3 and dense hints, uniform with the default widths;
  - both `align_padded`;
  - both `oversampled_ranks`;
  - `newton_cg(padded_from=)` against the backend with the old point's data, both layers (losses and movements
    equal); without the keywords no new stats key.
- **Hint dispatch:** every error above. A custom `project` receives raw tuples.
- **T14 (h), the frontend half:** a rank-0 and a partially-zero-rank `x_old` (Tucker `(0,2,2)`, TT `(1,0,2,1)`).
  - Through the frame twins on both layers: bitwise equal to each other and to the empty frame's start.
  - Through `newton_cg(padded_from=)` on both layers: the iteration-0 frames bitwise equal.
- **The §3.6 messages:** layer, shape, `N`, the start frame's class, corewise (`padded_from`, `'cone'`), both
  keywords, a group option without its group, a cone option unread, zero passes with an option.
- **Safe-mode checks:** a different tensor, a non-orthonormal start frame, a start frame of another point, untied
  factors under sharing. Each raises in safe mode and runs under `safety.unsafe()`.
- **Dispatch** (`TestAlignedPaddedFrameFrontend`):
  - every twin on jax inputs gives jax outputs (uniform masks host);
  - the loop twins eager and with `use_jit`, equal to numpy by the frame measure (≤ 1e-12);
  - `newton_cg` with `padded_from` and `'cone'` under `use_jit` on both layers: jax result, the same pass count,
    the cone value to 1e-10.
- **CI rule:**
  - the `-W` Sphinx dummy build: exit 0, no warnings;
  - an AST scan of the seven touched frontend modules: no shadowed imports or duplicate definitions;
  - their doctests: 145 passed;
  - no 1e-14 cross-path tolerance: the bitwise checks compare one code path against its own call, and cross-path
    checks use 1e-12 or looser.

## Floor env

`tttt` (jax 0.4.30, numpy 1.22, Python 3.9):

- `test_aligned_padded_frame_frontend.py` and `test_optimizers_frontend.py`: 53 passed;
- the new jax tests (`TestAlignedPaddedFrameFrontend`): 2 passed.

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n 3` (the coordinator's rule for this machine):

- **Result:** **1,340 passed, 4 skipped, 48,713 subtests passed** in 15 min 49 s (N7: 1,326 passed). The one warning
  is the existing `adam` one.
- **The first `-n 3` run crashed a worker** inside an XLA compile while it ran `TestAlignedPaddedFrameFrontend.test_twins`
  (1 failed, 1,339 passed).
- **Cause, measured:**
  - every eager-jax alignment test adds 8,000–9,500 process memory mappings (one per compiled executable section);
  - a worker running a third of `tests/test_dispatch.py` passed `vm.max_map_count` (65,530) and aborted, the failure
    `scan_body_principles.md` describes;
  - four of these tests in one process took it from 449 to 21,088 mappings, and `jax.clear_caches()` brought it back
    to 905.
- **Fix (judgment call 4):** the module-level `_release_jax_programs()` (`jax.clear_caches()`), registered with
  `addCleanup`.
  - Where: the eight padded-frame tests of `TestDispatch` (N2–N8) and in `setUp` of the four padded-frame jit
    classes (12 sites).
  - The compile-count tests measure deltas within one test, so they are unaffected.
  - The rerun passed with no crash.

## Plan problems

- §3.6 lists T15 over seven files (judgment call 1).
- The plan's "the twins convert a structurally zero `x_old` to the canonical rank-0 tensor first" is implemented as
  "skip the reduction and pass it on" (judgment call 2). The behaviour is the same and verified.
