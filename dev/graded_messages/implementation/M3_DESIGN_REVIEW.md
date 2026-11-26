SOUND: one should-fix point (documentation), the rest minor; no second design round needed.

# M3 design review: the block sums and the display

Reviewer (the sub-arc's second), 2026-11-23, on `M3_DESIGN_NOTE.md` at `68c10621`, read only. I read it against:
- plan §4 ("The block sums, and the display") and the M3 row of §7;
- the maintainer's note "Defaults, and the block sums";
- `M_ARC_DECISIONS.md` 13(c);
- today's code: `sampling_steps.r_block_sumsq_step`, the engine's row, `backend/fitting.py` (`query`, `_y_metric_y`,
  `_SCALED`), `backend/optimizer_display.py` (`relative_errors`, `_format_header`, `format_newton_iter`,
  `make_newton_display`), the frontend `optimizers.newton_cg` (verbose), and every reader found by grep in
  `t3toolbox/`, `examples/` and `docs/`.

## The points Claude asked me to judge

**1. `r_block_sumsq` through `data_metric` in the engine, with no `γ`; the oracle adds `γ` to both sums. Agree.**
- This is exactly how `f` is built: the engine computes `<r, M r>` at `γ = 1`, and the oracle owns `γ` (O-21; the
  engine never sees it).
- The block step is `_weighted_sum` with other axes kept. Summing over `W`, `C` and the edge per block of `(d,) + F`
  is today's block reduction, so the block shape does not move.
- Putting both sums in `_SCALED` makes `γ` part of `M`, as decided. The display's ratio then has the same `γ` above
  and below, since numerator and denominator both come from the batch the display holds.
- A consequence worth one sentence in the oracle's docstring: on a drawn batch whose `y` is constant over `W` (kept as
  one axis of 1 by `take`), `y_block_sumsq` is `γ · b · y² = n · y²`, the full batch's value. So the denominator of
  a drawn batch estimates the full one without bias, as `f` does.
- Counting the weight for the dtype when only `r_block_sumsq` is wanted is right: `f` does the same.

**2. `y_block_sumsq` sharing a helper with `f_resolution`, which stays bitwise. Agree, under two conditions.**
- The helper returns the per-mode products `y · data_metric(y, ω, None)` at `z`'s shape, and each caller reduces.
- `f_resolution` stays bitwise provided:
  - the products are formed exactly as now (`broadcast_to(y)` first, then `data_metric` on the broadcast array);
  - `f_resolution`'s reduction keeps its current axes and order (the jet axes, `W`, the edge, per mode, then the sum
    over modes).
- Both reductions keep or drop `C`, but never sum `C` inside the shared part, so this is easy to keep.
- Pin it: after M3, the battery's `f_resolution` arrays (`orc/…`) against `68c10621` should be bitwise. That is a
  two-line comparison of two records of the existing `battery.py new`, much cheaper than a test.

**3. `C` read off the frame. Right, and there is a better home for it (minor 1).**
- `f`'s shape is not available when `f` is not wanted, as the display's denominator query shows: it asks for
  `('y_block_sumsq',)` alone. `y` and `ω` cannot give `C` either: a shared `y` has size-1 `C` axes.
- So the frame is the only source, and it should be the one source for both sums. Let `f_resolution` read it there
  too, rather than keep `np.shape(f)` beside a second reader; the values are equal.
- See minor 1 for where the reader should live.

**4. Is the display really the only consumer? Of the values, yes. Of the meaning, no (should-fix 1).**

In the library, `grep` finds three kinds of reader:
- **The display**, which reads them only in `make_newton_display`'s `errors()`, and through it `verbose=True` and
  `stats['diagnostics']`.
- **Text.** The `Oracle` protocol's docstring and the engine's table and docstrings.
- **The `HeldVariables.r_block_sumsq` slot.** It is unreleased, holds the engine's value, and has no `γ`.

No optimizer, trust-region or `LocalModel` code reads them, so trajectories cannot move. The old kinds'
`block_sumsq` / `has_block_sumsq` are a separate layer.

Outside the library:
- `examples/fit_probe_display.py` (D2);
- `docs/fitting_and_optimization.md:114` and its timing block (`:747-765`; not checked, block-constant, unaffected);
- the frontend docstring of `newton_cg` (`optimizers.py:528`, "in the shape of the weight"), which is a docstring and
  M3's to fix, not P19's;
- the module docstring of `optimizer_display` (`:26-30`, "UNWEIGHTED … independent of any residual weight"), which the
  note covers under "§4's sites".

What the note's table misses is the **validation table**:
- `errors()` takes the validation numerator and denominator from `val_batch`, which after M3 means `val_batch`'s own
  weight.
- The frontend translation builds the validation batch with **no** weight (`_old_fit(..., None)` for `val_sample`),
  and a user's `val_batch` may well have none.
- With a weight that is not constant per block (per sample, per entry), the `train|val` cells then compare a
  weighted ratio with an unweighted one, in one cell, with nothing to say so. With a block-constant weight the two
  agree, which is why no current test or page would notice.

**5. The tests' independence: good, with one gap and one remark.**
- Tests 1, 5 and 6 are against independent oracles:
  - `oracle.py`'s new `weighted_full` and `block_sumsq_full` (`expand`, no broadcasting);
  - the dense ratio with the sample deleted;
  - `Σ_c` of per-element dense sums.

  Test 5's teeth (the plain ratio above 0.1) and test 6's mutant (over `y`'s own shape: a factor of `|C|`) are the
  right ones.
- Test 2 (`ω = 1` bitwise) pins `None`, full-form ones and `blocks_to_weight(ones)` against `Σ r²`. Good.
- **Remark.** Test 3's law, `½ Σ_b r_block_sumsq = Σ_C f`, compares two engine routes that share `data_metric`. It is a
  consistency check, not an oracle. Keep it, but count on tests 1 and 5 for correctness.
- **Gap** (minor 2). Test 1 puts garbage in the padding for the engine's `r_block_sumsq` only. The oracle's
  `y_block_sumsq` on a **packed** batch, with garbage in the padding of `y` and `ω` and edges both 1 and `N`, is also
  needed. It is the display's denominator on every uniform fit, and it goes through `_arrays_of` and the new
  per-block reduction, which `f_resolution`'s part-4 tests reach only through a sum over `F`.
- Mutants to run at build: `γ` applied to one sum only (the drawn ratio off by `√γ`: test 3 and the drawn gate
  record must see it); `r_block_sumsq` unweighted (test 5); `y_block_sumsq` over `y`'s own shape (test 6).

**6. The gate's comparison of display records recorded on `68c10621`. Agree, with one condition (minor 3).**
- Recording `train_err` and `val_err` for a fixed set of weighted runs before any change, and comparing to `rtol
  1e-12` with exact iteration counts and stop reasons, is the right test of "block-constant weights reproduce today's
  ratios".
- 1e-12 is ample: with `ω` constant per block, `ω²` factors out of both sums, and the regrouping is `O(1e-16)`.
- **The condition.** Every recorded run must have `y` (and `ω`) of `z`'s full shape on its `W` axes, and must be
  unstacked. With a size-1 `W` or `C` axis today's ratio is wrong by `√|W|` or `√|C|`, which M3 fixes on purpose, so
  such a run would differ by design.
- Include one run with a validation batch, so that `val_err` is in the record (point 4).
- `stats['history']` and the losses should be bitwise across the two trees: nothing that steers the run changes. Add
  that to the comparison; it costs nothing.

## The two decisions

**D1: drop `obj_unweighted` from `format_newton_iter` with no shim, recorded for P19's release notes. Agree.**
- The parameter's only meaning was the dropped figure. The maintainer's line on old calls is to break them cleanly
  rather than compromise the design (TARGET_DECISIONS, "Old positional calls … documented, not guarded"). An ignored
  parameter would be dead API in a public signature.
- One correction to the note's claim that a positional caller "fails loudly". With a table, a fourth positional
  number lands in `fmt` and the formatting raises (`len(fmt % 1.0)`). With **no** table (`train_err=None` or
  `error_table=False`), `fmt` is never used, and the call passes with the figure silently gone.
- That outcome is harmless, since dropping the figure is the intended behaviour, but the upgrade note should say
  "positional callers: remove the fourth argument", not promise an error.
- Record it in `M_ARC_DECISIONS.md` and in P19's list.

**D2: fix the `unwt` lines of `examples/fit_probe_display.py` in M3, and leave the page's ratio sentence to P19.
Agree.**
- The example's lines `:12-13` and `:86-87` become false the moment M3 lands, and the examples run in the gate, so
  M3 owns them.
- The page's `‖S(x)_ij − y_ij‖/‖y_ij‖` (`:114`) stays true for every weight the page uses (block-constant), so it is
  prose for P19. Add it to P19's list beside §4.6, worded as "the weighted ratio `‖ω⊙r_b‖/‖ω⊙y_b‖`; the same as the
  plain one for a weight constant within each block".

## Findings

### 1. SHOULD FIX: state what the validation table means, and make the default comparable

**Evidence.** Point 4: after M3, the train cells use the run's weight and the val cells use `val_batch`'s own weight,
and the translation's validation batch has none. For a weight that is not constant per block, the two halves of a
`train|val` cell are different quantities.

**Resolve** (a documentation decision for Claude; recommendation first):
- **(a)** Document it in `make_newton_display`, in the frontend `newton_cg` docstring and in the oracle docstring:
  "each table is the weighted ratio of its own batch: give `val_batch` a weight of the same form, or both tables are
  not comparable."
- **(b)**, in the translation only (deleted in P17): give the validation batch the training weight's blocks through
  `blocks_to_weight` on the validation data, which is possible because the old weights are per block.

I recommend (a). With (b) only as an option: the old display showed unweighted ratios for both, and for the old
block weights the weighted ratio equals the unweighted one, so the translation's output is unchanged either way.

A test: one display run with a per-sample training weight and an unweighted `val_batch`, whose `val_err` equals the
dense unweighted ratio on the validation data. It pins the meaning and catches a mutant that reuses the training
batch's weight for validation.

### 2. MINOR

1. **The `C` reader.** Make it public, `frame_stack_shape(frame)` beside `frame_shape` in `sampling_variables`.
   - It reads `D[0]`'s or `B[0]`'s leading axes, and the supercore's axes after `d`, for any of the four forms.
   - The maintainer added `frame_shape` publicly for exactly this reason: reading a uniform frame by position is "as
     easy to get wrong as the count of four" (TARGET_DECISIONS, 2026-11-21). A backend user writing an oracle needs
     `C` for the same reason the oracle does (the razor).
   - Then use it for both sums (point 3). If Claude prefers to keep the public surface still, a private helper is
     acceptable.
2. **The packed `y_block_sumsq` test** of point 5: garbage in the padding, edges 1 and `N`, uniform against ragged and
   against `block_sumsq_full`.
3. **The gate records' condition** of point 6: full-shape `y` and `ω` on `W`, unstacked, one run with a validation
   batch, and the history compared bitwise.
4. **Wording.**
   - The oracle's class docstring should say that both block sums carry `γ` and that their ratio does not depend on
     it.
   - The `Oracle` protocol's docstring should say that a user oracle's block sums are what the display divides, so a
     user oracle returning plain sums gets plain ratios.
   - The engine's `r_block_sumsq` row and table text should name the weight among its reads.
5. **D1's upgrade note**, as corrected above.

## What I will check at the build

- The weighted block sums against my own dense sums (`np.repeat`), with:
  - every kind, and the orders `()`, `(2,)`, `(1,1)`;
  - per-sample and per-entry weights, with zeros;
  - `y` shared over `W` and over `C`;
  - drawn and full batches;
  - ragged and packed batches.
- `f_resolution` bitwise against `68c10621` (the battery's `orc/` arrays).
- The gate records' comparison, and the history bitwise.
- The validation table's meaning (finding 1).
- The mutants of point 5.
- The expected-red set against a run, and the tally.
