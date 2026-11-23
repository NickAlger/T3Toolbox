SIGN OFF

# M3 review, round 1: the block sums and the display

Reviewer (the sub-arc's second), 2026-11-23, on the uncommitted M3 on `graded-messages` (over `deedf72d`) and
`M3_HANDOVER.md`, against the plan's M3 row, my design review and Claude's rulings (`M_ARC_DECISIONS.md` 20 to 23).

Two files are left out, as Claude asked. `tests/backend/test_curvature.py` and `tests/backend/test_uniform_curvature.py`
carry the CI-fix agent's edits and are not M3's.

My scripts are in `scratchpad/m2rev2/`:
- `check_m3.py`: the block sums against my own dense per-block loops;
- `check_m3_display.py`: the validation cells;
- `battery_m3.py`, with its records `b68.npz` and `bm3.npz`, and `base68/` (a `git archive` of `68c10621`);
- `mutants_m3.sh`, `mutants_m4only.sh` and `mutants_m3.out`.

**Verdict: SIGN OFF.**
- Every number I built independently agrees.
- `f_resolution` is bitwise unchanged.
- The validation cells mean what the ruling says.
- All five of my mutants are killed.
- The gate, the red set and the tally check out.
- There are no findings above minor.

## 1. The code against the plan, the rulings and my design review

**The engine.**
- `r_block_sumsq_step(r, weight)` is `Σ r · data_metric(r, ω, None)` per block. It sums over every axis after the
  jet axes (`W`, `C`, the edge) and keeps the mode and `F`.
- The row reads `('r', 'weight')`, as `f` does, and there is no `γ`: the engine never sees it.
- The reference multiplies by its own `_weight_squared` and still has no `jets` import.

**The oracle.**
- `_SCALED` gains `r_block_sumsq`, and the scaling loop adds `y_block_sumsq`, so both carry `γ` and their ratio does
  not.
- `_y_times_metric_y` is the old `_y_metric_y` body unchanged: `y` is broadcast to `z`'s shape first, then multiplied
  by `data_metric`.
- `_y_metric_y` (`f_resolution`) and `_y_block_metric_y` each reduce it. `f_resolution` keeps its axes and order, and
  is bitwise (§3).

**`C`.** Ruling 22 keeps the reader private, as `_stack_of_frame`. It reads all four forms correctly: `D[0]` or
`B[0]`'s leading axes on the ragged layout, and the supercore's `[1:-2]` on the uniform one. It is now the one source
of `C` for both sums.

**The display.**
- `(unwt …)` and `obj_unweighted` are gone (ruling 20). `errors()` returns two values, and the callback plumbing
  follows.
- The docstrings of `make_newton_display`, the frontend `newton_cg` and the oracle state the weighted ratio and the
  validation rule (ruling 21).
- The `Oracle` protocol says that a user oracle's plain sums give plain ratios (my minor 4).

**The text.**
- "The block shape" replaces "the weight's shape" in every file M3 touches.
- `grep` finds `UNWEIGHTED` only in the old layer: `block_sumsq_over_*`, the old kinds' `block_sumsq`, and
  `probing.py`'s note. That layer is left alone by design and deleted with the old kinds.

**The example** (ruling 23).
- `fit_probe_display.py` loses its `unwt` text.
- Its comment now says the per-order weight is constant within each block and so leaves the ratios as they are. That
  is true, and it is why its docstring's `‖S(x)_ij − y_ij‖ / ‖y_ij‖` stays correct.
- The page's sentence (`:114`) is on P19's list.

## 2. The block sums against my own dense loops (`check_m3.py`): 888 checks, 0 failures

**The setup.**
- `C = ()` and `(2,)`, with `W = (2, 3)`.
- Probe and apply at `()` and `(1,)`, and entries.
- `z` is my own einsum of `to_dense()`.

**The data and weights.**
- `y`: full, shared over `W` (size-1 `W` axes), shared over `C`, and (probe) an edge of 1.
- The weight: none; per entry with zeros; per sample with a zero; constant per block.
- Every expansion is by `np.repeat`.
- The expected sums are, per block, `Σ ω²(z − y)²` and `Σ ω²y²` over every axis after the jet axes.

| Check | Cases | Tolerance |
| --- | --- | --- |
| the oracle's `r_block_sumsq` and `y_block_sumsq` on the ragged frame | 116 + 116 | 1e-12, shapes exact |
| the engine's `r_block_sumsq`, called directly | 116 | 1e-12 |
| `½ Σ_b r_block_sumsq = Σ_C f` | 116 | 1e-12 |
| uniform: `pack_batch` on the packed frame, finite garbage (`5e2`) in the padding of the sample and of every `N` edge of `y` and the weight | 96 + 96 | 1e-12 |
| a drawn batch (2 of 6): both sums equal `γ = 3` times my dense sums over the drawn samples, `y` and `ω` drawn by my own index loop | 116 + 116 | 1e-12 |

The `y`-shared cases are those where the old `y_block_sumsq` was wrong by `|W|` or `|C|`, and they now agree with the
dense sum over `z`'s shape.

## 3. `f_resolution` bitwise

I ran M2's `battery.py new` myself on a `git archive` of `68c10621` and on this tree, and compared the records
bytewise:
- every `f_resolution` array is identical (84);
- every array that is not a block sum is identical (24,871);
- all 437 `r_block_sumsq` and 84 `y_block_sumsq` arrays differ, as designed: the battery's weights are not constant per
  block, and its stacked cases share `y`.

This reproduces the handover's account exactly.

## 4. The display's validation cells (`check_m3_display.py`)

**The setup.** Backend `newton_cg` with `make_newton_display`, apply at `()` and probe at `(1,)`.
- The training weight is per sample.
- `val_batch` has no weight in one run, and its own per-sample weight in another.
- Row 0 (the start point) is compared with my dense ratios, `√(Σ ω²(z − y)² / Σ ω²y²)` per block.

**The results.**
- `train_err` equals the dense ratio under the training weight, and `val_err` the dense ratio under `val_batch`'s own
  weight, both to 1e-10, in all four runs.
- The ratio under the other weight differs by 0.3 to 3, so the check has teeth.

## 5. Mutants (5 of 5 killed)

Each was run on a scratch copy against `test_block_sums`, `test_least_squares_oracle` and `test_optimizer_display`, at
`-n 2`. Unmutated, the copy passes 65 tests with 590 subtests.

| Mutant | Result |
| --- | --- |
| `γ` on `y_block_sumsq` only (`r_block_sumsq` out of `_SCALED`) | killed (11, incl. the display's drawn test) |
| `γ` on `r_block_sumsq` only (`y_block_sumsq` out of the scaling loop) | killed (11) |
| the engine's `r_block_sumsq` unweighted | killed (141) |
| `y` not broadcast to `z`'s shape (summed over its own shape) | killed (115) |
| the oracle's `y_block_sumsq` unweighted | killed (148) |

The implementer's further mutants (the reference unweighted; the validation table lent the training weight) are also
recorded killed. My own `check_m3_display.py` shows the second has teeth.

## 6. The tests' independence

**`test_block_sums.py`.**
- It writes out the weighting, the broadcast and the per-block reduction by `np.repeat` and loops.
- It takes `z_c` from `sampling_definitions.z`, the library's dense reference, which M3 does not change. That is
  acceptable: it is the pinned dense definition, not the engine under test.
- The law `½ Σ_b r_block_sumsq = Σ_C f` is labelled a consistency check, as my design review asked.

**The display tests.**
- The outlier test compares with a plain dense ratio on the batch with the outlier deleted (`np.delete`), and it has
  teeth: the plain ratio with the outlier is above 0.5.
- The validation test compares with dense weighted and unweighted ratios.

**The display records** (§7) are an independent before-and-after comparison.

## 7. The gate, the red set and the tally

**The log.**
- `m3impl/gate/suite_v.log` ran from about 01:36 to 02:14, after every M3 file's last change (01:34). Only the two
  CI-fix files are newer.
- 1,960 passed, 0 failed.

**My tally against M2 part 5's log.** 1,960 nodes against 1,966, and exactly 10 differ:
- the 2 removed `unwt` tests;
- the 4 new `test_block_sums` nodes (164, 49, 6 and 3 subtests);
- the 4 new display tests.

**What I take from the handover.**
- The expected-red account: 117 failures, all the header argument, the old `y_block_sumsq` formula, or the
  reference's formula.
- The other gate steps: doctests, pages, examples, Sphinx with 0 warnings.
- The display records: 40 error arrays within 2.3e-16, and every other field, the cores, stop reasons and counts
  bitwise.

**Doctests I reran myself.** `optimizer_display.py` and `sampling_definitions.py` (the new weighted example) pass under
plain `python -m doctest`.

## Minor

1. **The upgrade note for D1** (ruling 20) and the page sentence at `:114` (ruling 23) are on P19's list. Make sure the
   list in `HANDOFF.md` carries both, with the wording of my design review.
2. **The `C` reader** stays private (ruling 22), pending the maintainer. If `frame_stack_shape` is approved, it is a
   rename of `_stack_of_frame` plus a doctest.

Nothing of mine is left running; the scratch copies are deleted. The working tree is as I found it, apart from this
file.
