# M3 design note: the block sums and the display

Implementer, 2026-11-23, on `68c10621`. Read only; nothing built yet. Implements plan §4 "The block sums, and the
display", the M3 row of §7, the maintainer's note "Defaults, and the block sums", and `M_ARC_DECISIONS.md` 13(c).

## 1. The changes

- **Engine.** The `r_block_sumsq` row reads `('r', 'weight')`, as `f` does. `r_block_sumsq_step(r, weight, ...)`
  becomes `Σ r · data_metric(r, ω, None)` per array, summed over the axes after the `m` jet axes (`W`, `C`, the
  edge). This is the same pairing as `_weighted_sum`, with other axes kept. There is no `γ`: the engine never sees
  it. The variable table, the step docstring and the "the weight's shape" text (`sampling_steps.py`) are updated.
  The weight now counts for the dtype when only `r_block_sumsq` is wanted, as it does for `f`.
- **Reference** (`sampling_definitions.r_block_sumsq`): the sum gains `_weight_squared(S, weight, i, r.shape)`, the
  module's own `ω²`. It does not import `jets`.
- **Oracle** (`backend/fitting.py`).
  - `y_block_sumsq` becomes `<y, M y>` per block over `z`'s shape: `y` is broadcast to `F+W+C+(N_i,)`, then
    `Σ y · data_metric(y, ω, None)` over the axes after `m`.
  - `_y_metric_y` (`f_resolution`) and the new block form share one private helper that returns the per-mode
    products at `z`'s shape. Each then applies its own reduction, so `f_resolution` is unchanged to the bit.
  - `C` is read off the frame by a private helper (`D[0]`'s or `B[0]`'s leading axes; the supercore's axes after
    `d`). `f`'s shape cannot be used, because `f` need not be wanted.
  - Both block sums join the scaled outputs (`_SCALED`), so `γ` is inside `M`.
  - The class docstring's "plain sums of squares" is rewritten, and so is the `Oracle` protocol's docstring in
    `optimizers.py`.
- **Display.**
  - `relative_errors` is unchanged in code. Its docstring says `‖ω⊙r_b‖/‖ω⊙y_b‖`, with `γ` cancelling.
  - The `(unwt …)` figure goes: `_format_header(info)`, `format_newton_iter` without `obj_unweighted`, and
    `errors()` returns no third value.
  - The text "the weight's shape" becomes "the block shape" throughout (§4's sites, `data_preparation.py:134-137`,
    and the frontend's `optimizers.py:529`).
- **Dense oracle** (`tests/oracle_sweeps/oracle.py`), additive.
  - `weighted_full(kind, like, value, weight)` gives `ω⊙value`, both written out to `like`'s shapes by `expand`,
    with no broadcasting.
  - `block_sumsq_full(kind, values, m)` sums the squares, keeping the mode and `F`.

## 2. What each reader sees

| Reader | After M3 |
| --- | --- |
| `make_newton_display` / `relative_errors` (and so `verbose=True`, `stats['diagnostics']`) | the weighted ratio. A block-constant weight (every `blocks_to_weight` weight, every old form) gives today's ratios to rounding. A sample with `ω = 0` drops out of both sums. A block with `ω = 0` throughout shows `—`. `γ` cancels: the numerator and denominator come from the same batch. |
| the header | no `(unwt …)`. `obj = misfit + reg` stays. |
| `f_resolution` | **unchanged.** Since M2 it reads `<y, M y>` (`_y_metric_y`), not the block sums. |
| optimizers, trust region, `LocalModel` | unchanged: no code reads the block sums except the display, and the display is read-only. Trajectories are bitwise identical. |
| held `r_block_sumsq` (`HeldVariables`) | means the weighted value (unreleased). |
| the old `SamplingKind.block_sumsq`, `block_sumsq_over_*`, `has_block_sumsq` | untouched. They are another layer, deleted with the old kinds, and the display no longer reads them. |

**What M3 fixes along the way.** Today `y_block_sumsq` is summed over `y`'s own shape. With a `y` shared by the
stack (a size-1 `C` axis) the ratio is too large by `√|C|`. With a `y` shared by the samples (a size-1 `W` axis) it is
too large by `√|W|`. Taking the sum over `z`'s shape fixes both.

## 3. Tests, each against an independent oracle

1. **Dense weighted block sums.**
   - Cases: probe, apply and entries; `order` `()`, `(2,)` and `(1,1)`.
   - Weights: `None`, random per entry, per sample with zeros, and block-constant. `y` with size-1 `W` or `C` axes.
     The sizes of `F`, `W` and `C` are distinct.
   - What is compared: the engine's `r_block_sumsq` (ragged, and uniform with garbage-padded `y` and `ω`), and the
     oracle's two sums, all against `oracle.py`'s `block_sumsq_full`.
   - The reference's formula test (`test_sampling_definitions.py:476`) becomes the weighted brute-force sum. The
     engine-against-reference matrix, which passes weights already, then covers the jet matrix.
2. **`ω = 1`.**
   - `r_block_sumsq` with `weight=None`, with full-form ones and with `blocks_to_weight(ones)` equals `Σ r²` written
     in the test, bitwise.
   - `y_block_sumsq` with a full-size `y` equals `data_block_norms(y)**2` to rounding.
3. **The scale.**
   - On a drawn batch, each sum is `n/b` times the dense sum over the subset; it moves to the scaled list of
     `test_a_drawn_batch`.
   - Law: `½ Σ_b r_block_sumsq = Σ_C f`, weighted, drawn and undrawn. This compares two routes, `f_step` against
     the block step.
4. **Block-constant weights reproduce today's display.**
   - Committed: `test_the_errors_against_dense_residuals` keeps its dense unweighted reference. Its block-constant
     weight makes the reference exact, and its docstring says so.
   - Gate: records (`train_err`, `val_err`) from a fixed set of weighted runs (probe `(1,)`, apply `(2,)`, entries,
     regularized, drawn). They are recorded on `68c10621` in the scratch folder and compared after M3 to
     `rtol 1e-12`. The iteration counts and stop reasons must be identical.
5. **A zero-weighted outlier.**
   - Setup: exact data with one sample corrupted by `1e3·‖y‖` and `ω = 0` there; apply, and probe `(1,)`.
   - Each row's `train_err` must equal, to `1e-8`, the dense unweighted ratio on the batch with that sample deleted
     (`np.delete`).
   - The last row's `train_err` must be below `1e-6`.
   - Teeth: the dense plain ratio must be above `0.1`. A mutant that leaves either sum unweighted fails.
6. **`√|C|`**, at oracle level, since the optimizers refuse a stacked fit.
   - Setup: `C = (3,)`; `y` shared over `C`, then shared over `W`.
   - `y_block_sumsq` must be `Σ_c` of the dense per-element block sums, which is `|C|` times the shared `y`'s.
     `relative_errors` must match the dense ratio.
   - The mutant "over `y`'s own shape" fails by the factor `|C|`.
7. **The header.**
   - The `_format_header` tests lose `obj_unweighted`, and their pinned strings lose `(unwt …)`.
   - No row of a weighted, regularized or drawn run prints `unwt`.
   - `format_newton_iter(..., obj_unweighted=...)` raises `TypeError`.
8. **`test_dispatch`:** `r_block_sumsq` with a weight under `jit`, if it is not already covered. Float32 `ω` with
   float64 `y` gives float32 sums.

Before converting tests, I record the expected-red set from one run (M2's practice).

## 4. Unchanged

- The names and the block shape.
- `optional_outputs`; no new outputs and no new engine variables.
- `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp`, `r`, `Jp` and `f_resolution`, all bitwise.
- `data_block_norms` and `equalizing_weight`, which are plain per-block quantities.
- The table's layout and the records' keys.

## 5. Decisions for Claude

- **D1. `format_newton_iter` is public and released (2026.5.0) with `obj_unweighted` in fourth position.**
  - The plan drops this plumbing, so removing the parameter shifts `fmt` by one position for positional callers.
    Such a call fails loudly (a `TypeError` in the formatting), not silently.
  - **Recommendation:** remove it with no shim, and record it for P19's release notes. The alternative, an ignored
    parameter, is dead API.
- **D2. Prose about the dropped figure and the unweighted ratio.**
  - `examples/fit_probe_display.py:12-13, 86-87` describes the `(unwt …)` header, and becomes false at M3.
    **Recommendation:** fix those four lines in M3.
  - `docs/fitting_and_optimization.md:114` gives the ratio as `‖S(x)_ij − y_ij‖/‖y_ij‖`. That stays true for the
    page's block-constant weights. **Recommendation:** leave it to P19, with the §4.6 prose already listed there,
    and add it to that list.
