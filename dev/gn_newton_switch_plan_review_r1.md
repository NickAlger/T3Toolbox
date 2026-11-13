# Critic review r1 — `dev/gn_newton_switch_plan_2026-11-13.md` (draft 1)

Read for this review: `CLAUDE.md`; `dev/HANDOFF.md`'s 2026.5.0 bundle paragraph (Nick's nine answers); the source notes;
`backend/optimizers.py` (`_check_hessian`, `CurvatureState`, `LocalModel`, `Problem.local_model`, `_cg_*`, `NewtonInfo`,
the step-control constants, `_resolution_eps`, `_initial_radius`, `_lift_diagnostics`, `newton_cg` whole, the pytree
registrations); `backend/geometry.py` (`curvature_precompute`, ragged and uniform, manifold and corewise);
`backend/optimizer_display.py` (`_curvature_segment`, `_format_header`); the frontend `newton_cg` /
`least_squares_problem` / `_resolve_chunk_size`; `tests/test_optimizers_frontend.py` (mirror and forwarding tests);
`tests/backend/test_optimizers.py` (`_NEWTON_SCALAR_FIELDS` row test, `_OffsetProblem`, `TestTrustRegionNewton`);
`tests/test_dispatch.py` (the `_cache_size` rows); `parameters_not_closures.md`, `scan_body_principles.md`,
`naming_rules.md`, `signature_style.md`; `t3_to_dense` / `t3_inner_product` as the style reference; the timing plan
draft 1 (for §3.3); T3Polynomial `scripts/d10_darcy_o01_fit.py` (knobs) and `t3polynomial/fitting.py`.

**(checked)** = run on a tiny problem (rank-2 fit of a noisy rank-3 `probe_derivatives` order-1 target, d = 3), numpy:
`dc.replace(exact_model, curvature=None)` gives `hvp`, `quadratic` bitwise equal to a Gauss-Newton model's, and the exact
model's `gradient` bitwise equal to the GN gradient, on `ManifoldGeometryOps` and `CorewiseGeometryOps`, with and
without `IdentityRegularizer`; under jax, `tree_structure(gn_model) == tree_structure(gn_view)`.

## Summary

The plan is faithful to the nine answers and to the source's rule, and its mathematics is right. Verified:

- `m_N(s) = m_GN(s) + ½α²·switch_pKp`. `curvature_quadratic(p) = ⟨Πp, LᵀK_N LΠp⟩` is the exact quadratic minus
  `gn_quadratic`. Exact mode's `pHp` is `gn_quadratic + pKp`, the form is quadratic in `p` (so `α²`), and the sign is
  `+`. Regularization sits in `gn_quadratic`, so it is in both models. `e_GN`/`e_N` read the row's own `predicted`
  (LS: `α·slope + ½α²pHp`; TR: `slope + ½pHp`).
- The GN view is the Gauss-Newton operator, including the gauge (same `geom_aux`) and the regularizer. It has the
  GN treedef, so under `use_jit` it reuses pure GN's compiled `_cg_core` (checked). The exact build has no side effect
  on the step: `curvature_precompute` is pure, the lift is finite at any `ε` including the zero tensor, and the
  gradient is the same expression. So T1 (`switch_tau=0` is bitwise GN for either step control) and T2 (`switch_k=0`
  is bitwise exact) hold by construction. T3's offsets (references, `tr_radius0`, `Δ_max`) are the right ones.
- The rounding gate's counting equals the source's. The library's scale `ε(‖ωy‖‖ωr‖ + |f|)` is the right one: it
  is the cancellation error of `f = ½‖S(x) − y‖²`, which the source's `max(1, |f|)` ignores.
- The compile budget is at most 2 `_cg_core` programs per level, which is within the house rule.
- `cg_forcing_power` is today's spelling, and the `switch_*` group follows the `padded_*`/`cone_*` rules. History
  keys of non-switch runs are unchanged, following the aligned-frame review's precedent.

There is no MAJOR finding. The MINOR findings: a cost saving with identical switching, two stale coordination points
with the timing plan, a biased basis for G-4, one benchmark item made optional, the frontend's validation order, and
three spec gaps.

---

## MINOR

### 1. MINOR — §1.1 steps 5 and 7, §1.2, §1.5, §3.1: the tracking product is only needed on a countable row

Every non-countable row resets the count whatever `e_N` is. That covers a rejected TR trial, an LS backtrack, a forced
`'accept'` and the −g fallback. So the product, about 2–3 GN applies, decides nothing on those rows. Countability is
known after step control, and the fallback before CG. Rejections and backtracks cluster in the noisy transient, where
the GN phase lives, so the saving is real at no change in when the switch fires.

**Fix.** Compute `switch_pKp` after step control and only when the row is countable. Report `switch_e_gn` on every
GN-phase step row (it is free: `|Δf − predicted|`), and `switch_pKp`/`switch_e_n` as None on non-countable rows.
Update T4 (each reset row has `switch_e_n is None` and the count is 0), the display (`—`), §1.5's cost, and the T9/§6
product counts. This departs from the source's accounting ("one product per GN-phase iteration, rejected trials
included") though not from its rule, so list it in §9 as G-7 with this as the recommendation.

### 2. MINOR — §3.3, G-5, W1: stale against the timing plan's draft 1

Timing draft 1 flattened the records (its W-1), so `ModelSeconds` no longer exists. Its Q-1(a) already clocks exact
mode's misfit transpose in `gradient` and makes `curvature` the `curvature_precompute` alone. The requested split is
therefore already there: on a GN-phase row, the per-point tracking cost is `seconds.curvature`. Timing §2.7 puts the
tracking product in `forms` "unless the switch plan says so".

**Fix.** Keep the one remaining request: `PhaseSeconds.tracking`, None except on rows that ran the product. With
finding 1 that means countable GN-phase rows. The overhead per row is then `seconds.curvature + seconds.tracking`,
with the compile twins beside it. This is a consistency matter between the two plans, not a decision for Nick: settle
it in this critic loop (I am raising it on the timing plan too), drop G-5 from §9, and have W1 lap
`'tracking'`/`'curvature'` instead of adding "(A)'s two timing fields".

### 3. MINOR — §7: §4.13 is claimed by both plans

Timing draft 1 writes `docs/fitting_and_optimization.md` §4.13 "Timing a Newton run" and refers to "the §4.13
example" in its tests and figures. This plan takes §4.13 and moves timing to §4.14. Timing lands first and ships
regardless of W3's outcome.

**Fix.** The switch takes §4.14, opening with a pointer to §4.12, and nothing landed is renumbered. Likewise the
CHANGELOG's `[2026.5.0]` has no `### Added` yet: the timing slice creates it and this plan appends.

### 4. MINOR — §1.5, §6.1, G-4: the default would be fixed on a biased measurement

Under `use_jit`, the GN phase's CG is one compiled call, while the tracking runs eagerly, op by op: the exact build at
every GN-phase point and the product on every row. At 10×100 on the laptop, the tracking's wall-time share is then
largely dispatch overhead. The standing rule is that small-run timings do not predict large-scale cost (contractions
take over at scale). G-4's "~15 % of GN-phase wall time" criterion inherits that bias.

**Fix.**
- W3 reports the overhead in both currencies: wall time (the timing split, compile seconds separate) and
  apply-equivalents (tracking products and builds × the multipliers measured at W3). G-4's rule reads both.
- Add a middle option to G-4 for when eager dispatch dominates: jit the tracking product as a module-level
  `_curvature_quadratic(lm, p)` through `_jitted`. That is one more program per level, still within the budget. Exact
  rows keep today's eager `pKp`, so T2 stays bitwise. Weigh it before `switch_track_every`.
- Put "does the feature ship in 2026.5.0 if 10×100 is not promising?" on the W3 stop's agenda.

### 5. MINOR — §6.1/§6.2: the padded-start stress case is made optional (fidelity, answer 9)

Answer 9 lists three benchmark items: Darcy 10×100; then 20×1000 if promising; and the padded-start levels as the
no-switch stress case. The plan folds the third into §6.2/W5 (GPU, optional). The stress case is the worst case for
"track every row" (the tracking is paid and nothing is gained), so it is exactly the input G-4 needs, and it is cheap
on the laptop.

**Fix.** Add `START=aligned` arms to §6.1/W3 (switch, GN-TR, tracking-only), reporting how often the switch fires on
padded levels and the tracking share there. Keep the 20×1000 repeat in W5.

### 6. MINOR — §2.1, §2.3: "raised at the call before any work" does not hold on the frontend path

The frontend runs `least_squares_problem` before the backend sees the switch options. That includes the minimal-rank
reduction and, for a uniform `probe_derivatives` fit with `chunk_size='auto'`, the estimator's compile. Today's
`_check_hessian` call carries the comment "structural, named for the caller, before the setup work".

**Fix.** The frontend calls `bopt._check_switch_options(hessian, switch_tau, switch_k, switch_cg_forcing_power)`
right after `_check_hessian` (it is a pure function on basic types), and the backend checks again for raw-`.data`
callers. T7 then asserts the frontend raise happens before `least_squares_problem` is called (mock spy, as the
misspelt-option test does).

### 7. MINOR — §1.1, §3.1: `NewtonInfo.lm` on GN-phase rows is unspecified

A callback receives `info.lm`. With the tracking model, `info.lm.is_exact` is True on a row whose `hessian` is
`'gauss_newton'`. With the GN view, a callback's `info.lm.gradient` redoes the transpose.

**Fix.** Specify the GN view: it is the model that solved the step and agrees with the row's `hessian`. It is the
same object across a rejected trial, which keeps `test_rejection_keeps_x_and_reuses_the_model`'s identity contract in
switch mode (add a switch subtest). Note the transpose cost in the `NewtonInfo` docstring.

### 8. MINOR — §3.1, §3.2: `switched_at` and the converged row's switch fields at the edges

- Firing on the last step row (`it = max_newton − 1`) gives `switched_at = max_newton`, which names no row.
- A firing row followed by the converged row gives a `switched_at` row that "solved" nothing, against §3.2's wording
  ("the first iteration solved with the exact Hessian").
- The converged row's `switch_count` in the GN phase is undefined: carried, or None?

**Fix.** Define `switched_at` as "the index of the first row whose model is exact, equal to `len(history)` when the
switch fired on the last row", or as None in that case. Pick one and assert it in T8. State the converged row's
fields: `switch_fired=False`, `switch_count` the carried count (or None), the rest None.

### 9. MINOR — §1.1 step 3, T9: the GN-view compile claim has no gate

"The GN phase runs pure GN's compiled `_cg_core`" is the compile-budget claim, but T9 only bounds growth over a firing
run.

**Fix.** Add to T9: after a pure `gauss_newton`, `step_control='trust_region'`, `use_jit` run at a structure, a
`switch_tau=0` run at the same structure grows `_jitted(_cg_core)._cache_size()` by 0. I checked the treedef equality
it rests on.

---

## NIT

1. **§1.3.** `informative = max(e_GN, e_N) > floor` counts exactly as the source's `e_GN > floor`, as the plan shows.
   The source's form is simpler, and its "invalid" label suits a row whose GN error is at the floor better than
   "informative miss". Either works; prefer the simpler one unless "applied to both errors" was meant to change
   behaviour, which it cannot here.
2. **§1.6.** Add float32. Under `use_jit`'s default float32, `floor ≈ 1.2e-4·(‖ωy‖‖ωr‖ + |f|)`, so the switch can only
   fire on large early steps. Advise x64 for the switch (T5 already exercises the gate).
3. **§3.1.** State the reading of answer 7 in one sentence: the fields are None on `NewtonInfo` outside switch mode
   and are omitted from non-switch history rows. The precedents are the `extra` stats keys and the aligned-frame
   review's "history keys on default runs". (Timing §2.7 assumes the fields are always present and lists four, not
   five; I am raising that on the timing plan.)
4. **T4.** "The count update unit-tested", but the update is inline in §1.1 step 7. Either name a helper
   (`_switch_count(count, countable, hit) -> int`) or test it through runs only.
5. **T3.** Besides `Δ_max`, the `trust_region_floor` (`1e-12·Δ0`) also differs between the two runs. Assert that
   neither floor fires, or compare with `stop_reason` aside.
6. **T6.** "Converges in ≤ GN's Newton steps": name the baseline (GN with its default line search, or GN-TR) and note
   that `newton_steps` counts rejected trials. Per the house rule, verify the property on the problem before
   asserting it.
7. **§2.1.** With `switch_tau=0`, `switch_k` is unread in effect: the count never moves. Raise it like
   `switch_cg_forcing_power`, or say why not.
8. **§2.1.** D-33/D-38 are cited from a `dev/` plan. Cite the shipped statement instead: `newton_cg`'s docstring, "an
   option of a group set without its group engaged ... raises".
9. **§2.3/T11.** Forwarding only when not None exists to keep the forwarding test unedited, while
   `hessian`/`curvature_damping`/`tr_radius0` are always forwarded. Either way is fine, but that test's purpose ("none
   dropped or swapped") is best served by adding the three switch keywords with non-default values.
10. **§3.1.** GN-phase rows drop the tracking model's `lift_sigma_min`/`lift_eps`, which is the invariant's price.
    Those values are what explains §1.6's non-firing. If `info.lm` is the GN view (finding 7), a callback cannot reach
    them. Acceptable; say so in §1.6.
11. **§6.2.** "CG-equivalent work" leaves out the per-Newton-step costs the switch saves: model build, trial
    retractions and objectives, and the per-point tracking build. Include them, or say the currency is a lower bound.
12. **W3.** T3Polynomial's level records copy `newton`, `newton_accepted` and `newton_stop_reason` only. Per-level
    `switched_at` then comes from the rows (the first `'exact'` row) unless the record copies it; say which.

## §9 decisions

- **Real decisions with sound recommendations:** G-1, G-2, G-3, G-6.
- **G-4:** real, but needs finding 4's two currencies and middle option.
- **G-5:** settled here (finding 2), not Nick's.
- **Add:** G-7 (finding 1).

VERDICT: REVISE
