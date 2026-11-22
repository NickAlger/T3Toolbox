NOT YET, for D9 only: the two acceptance tests stay red until it is ruled and built. I recommend option (d), which I prototyped (section 2). I found nothing else to fix in the code.

# P16 part 2 review, round 1: the backend model and the optimizers

Reviewer, 2026-11-21, on the uncommitted part 2 over `5be5dc28`. I read it against the plan's P16 item 2, the design
review and the rulings, and my orientation risk list. I judged the diff, not the handover's account of it.

Runs were one at a time, at `-n 2`. My scratch copies are in `reviewer2/` (the `d9` copy is the prototype of option
(d)). The working tree is untouched apart from this file.

## 1. What I checked

**The red list.** My own full suite (32 min): 243 failed, 1796 passed, 5 skipped, 1 xfailed.
- The red ids are **exactly** the 105 of `P16_part2_EXPECTED_RED.txt`: none missing, none extra.
- **The reasons, grouped by first error line, are the stated ones:**
  - `least_squares_problem` or `uniform_least_squares_problem` missing (the frontend; 176 subtests);
  - the old `pack_sample` form (the frontend);
  - "`<OldKind>` does not supply 'Kp'": the frontend factories pass an old kind where the backend model now takes
    an oracle;
  - `PhaseSeconds(sweep=...)`: the display, part 3;
  - the corewise `base_point` given the frontend's four-store frame;
  - the two D9 assertions.
- **The odd lines are frontend cascades:**
  - `test_fitting.py:1143` raises `UnboundLocalError` after its earlier subtests failed;
  - `sweep_exact` produces 0 rows because it builds every row through the `fitting.*_model` factories.

**The side-by-side against the pre-P16 library.**
- *The implementer's 36 cases, re-run:* the same table, 35 SAME and one DIFF (`u_switch_probe_manifold_reg`).
- *That DIFF is the radius tie:* row 0's `tr_radius` agrees to the last bit, and `pHp` differs in its last digit
  (`...038` against `...036`). The boundary flag decides whether the radius doubles for row 1 (61.17 against 30.59);
  the trajectories then differ, and both end at the same point (1.4e-6). It is a pre-existing fragility of the
  default radius, not this part's.
- **My own 18 cases** (`reviewer2/sbs_mine.py`): **18 SAME.** They cover:
  - the **shared** geometries: manifold and corewise, ragged and uniform, Gauss-Newton and exact, regularized;
  - **uniform corewise** Gauss-Newton on entries; regularized exact corewise on probe;
  - an exact uniform `padded_from`; a weighted, regularized uniform probe;
  - **order-2 derivative** data, exact corewise and with a full `(mode, order)` weight;
  - `gradient_descent` with `on_line_search_failure='accept'`;
  - **a regularized custom-draw MC-SGD**; uniform MC-SGD and Adam, regularized;
  - switch on a regularized corewise fit; the cone with exact Newton.

  Losses and final points agree to 1e-9 (most to 1e-13), with one exception: the regularized shared corewise
  Gauss-Newton case agrees to 9e-5 in the final point. I traced that to rounding amplification, not a regression: the
  **old** library against itself, with `x0` perturbed by 1e-14, differs by 6.8e-5 on the same case. It is an
  ill-conditioned corewise fit far from its optimum, and CG iterations grow along it.

**The silent cost.** `computed_names` of the `hvp` query given `model.held`, at a manifold frame with centers and at a
corewise point:
- Gauss-Newton recomputes **no** plain sweep;
- exact recomputes `nu_L` only, the parity case I noted in the design review.

**Compile keys.**
- A jitted `gn_quadratic` of a rebuilt model compiles once across a new point and a new weight.
- The first drawn batch compiles once more: its scale goes from `None` to a float, and `W` changes.
- After that, a new draw does not recompile.

**Routing, read in the code.**
- `jacobian`, `gn_quadratic` and `hvp` pass `Pi p`.
- Exact `hvp` is one query with `variations=Pi p` and `core_directions=lift(Pi p)`. On a corewise point the lift
  returns the same object, which the oracle requires.
- The regularizer's terms are unscaled, in `point_objective` and in the stochastic steps.
- Trials, the zero reference, the cone and the alignment projector query with nothing held. The projector takes
  `with_frame(frame)` at every pass.

The implementer's 15 mutants cover each of my risks: `held` across draws and trials, the scale, uniform `point_data`
with garbage, dropped centers, `eps`, `eps |rho|`, the `Kp` check, `with_frame`, swapped directions, an unprojected
`hvp`, and the regularizer dropped from `point_objective`. I did not repeat them.

## 2. D9: compare like with like

**The problem**, confirmed:
- The current point's objective is read off the frame through the centers: the model's `f`, plus `rho` at
  `base_point(frame)`.
- A trial's objective is read at its raw cores: `point_objective`, plus `rho(x_trial)`.
- The two are equal to rounding but rounded differently. At the floor, that offset is as large as the changes being
  tested.

**(d) is sound. I recommend it over (a).** I prototyped it in a scratch copy (`reviewer2/d9`):
- one `point_objective(x)` when a run starts;
- after an accepted step, the carried value is that step's `f_new`;
- the same in `gradient_descent`.

What the prototype shows:
- **Both D9 tests pass:** the line-search exhaustion stop fires, and float32 exact Newton stops at its floor with no
  spurious step.
- **The implementer's 36 side-by-side cases are unchanged:** the same 35 SAME and the same tie.
- **That is expected:** 2026.5.0 compared point route with point route. Its build's `point_forward(x_new)` is
  exactly the trial value (d) carries, so (d) reproduces the old comparison without the old extra forward.

The three points you asked about:
- **Armijo and the trust-region ratio with a regularizer.** Both sides become `point_objective`, misfit plus
  `rho(x)` from the raw cores: like with like in both terms. Today the `rho` term is mixed as well. `slope`, `pHp` and
  the predicted change stay the model's, which is right: they are its first- and second-order terms, and are not
  compared with a rounded value.
- **Paths where the next point is not the accepted trial:**
  - A rejected trust-region trial keeps `x` and the carried value.
  - `'stop'` after Armijo exhaustion ends the run.
  - `'accept'` after exhaustion moves to the last trial, whose `f_new` is the one carried.
  - The `-g` fallback is a trial like any other.
  - Switching to exact rebuilds the model at the same point; the value is unchanged.
  - The start (`x0`, a `padded_from` or `start_frame` start, every new call) costs one `point_objective`.
  - `retract` returns the next `x` itself, the very arrays scored, so the carried value is the value of `x`
    bitwise.
  - **The invariant to document:** the reference is the point-route value of exactly the current `x`. Any future
    step that changes `x` between iterations (for example the re-gauging in the backlog) must recompute it.
- **What to report.** The history's `objective` (and `losses`) should be the carried value, the one the acceptance
  used. Then `losses[k] + delta_f[k] == losses[k+1]` exactly on accepted rows, as in 2026.5.0. Split `misfit` and
  `regularization` on the same route: a private helper returning `point_objective`'s two parts (misfit, `rho(x)`),
  with the public function their sum. The model's frame-route `f` stays `LocalModel.misfit`, which the display's
  "unwt" line and the floor's dtype read.
  - The floor's scale is unaffected, since `f_resolution` is a size, not a comparison.
  - The switch's `e_GN = |delta_f - m_GN|` loses a bias of the floor's size, which can only help its rounding gate.

**Cost.** (a) costs one forward per accepted point, which scales with `W`. (d) costs one per run. The correctness is
the same. Row 4.3's purpose (no extra forward per accepted point) is kept by (d), so I read it as consistent with
row 4.3: your call rather than the maintainer's.

**Either option breaks tests that relied on the mix.** My prototype turns 9 tests of `test_optimizers.py` red besides
fixing the two:
- **The `_OffsetOracle` tests.** These are the trust-region floor tests, the shrink test, and the switch's reset,
  count and record tests. `_OffsetOracle` adds its offset to `f` at a corewise point to force rejections. Under (d) or
  (a) the reference is also read at a point, so the offset cancels. They need a route-independent rejection mechanism,
  for example a test oracle that raises `f` at points other than the run's start.
- **The history split test.** The prototype does not split the carried value; the helper above fixes it.
- **The timing attribution test.** The start's `point_objective` needs a lap; `reference` or `trial_objective` both
  fit.

The handover's "(a) restores both" holds for the two D9 tests only. With (a), the offset tests break the same way.

## 3. Minor

- `_alignment_projector`'s `lru_cache` is unbounded: one entry per (geometry, oracle), so one per rank level.
  Negligible; say so in a comment.
- The default trust-region radius tie (handover section 6) is real and pre-existing. A relative tolerance on the
  boundary test would remove it. It belongs in the backlog, not here.
