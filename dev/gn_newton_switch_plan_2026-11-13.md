# Gauss-Newton → exact-Hessian switch for `newton_cg` — plan (draft 4, 2026-11-14)

> **Status: DRAFT 4: W1 IMPLEMENTED and reviewed (ca13b37c, 2026-11-14; CHANGES 4.x); W2 next.** Draft 3: Nick's §9 decisions applied (2026-11-13; CHANGES 3.x); every §9 item RESOLVED. Draft 2 was
> approved by the critic (review r2, `dev/gn_newton_switch_plan_review_r2.md`) after answering review r1
> (`dev/gn_newton_switch_plan_review_r1.md`: no MAJOR, 9 MINOR, 12 NIT; CHANGES 2.x). Next: implementation, after the
> timing feature lands (HANDOFF's order). Signature-level. Binding inputs: Nick's nine answers (`dev/HANDOFF.md`,
> "2026.5.0 BUNDLE", item B), his §9 decisions, and the source notes `dev/gn_newton_switch_source_notes_2026-11-13.md`
> (the rule as built in nicks-optimizers, and its mapping onto `newton_cg`). §9 records the decisions on what those
> left open. References are to `main`
> (R6's working-tree edits leave these line numbers unchanged): `backend/optimizers.py` (`_check_hessian` 150–163,
> `LocalModel` 204–331, `Problem.local_model` 360–443, `NewtonInfo` 1148–1218, `_check_step_control` 1245–1257,
> `_resolution_eps` 1273–1280, `newton_cg` 1313–1670), `backend/optimizer_display.py` (`_curvature_segment` 149),
> frontend `optimizers.py` (`least_squares_problem` 202, `newton_cg` 498). Bundle order: the timing plan (A,
> `dev/newton_timing_plan_2026-11-10.md`, draft 1) lands first; §3.3 says what this plan adds to it.

## CHANGES

| round | by | what |
|---|---|---|
| 1 | writer | first draft |
| 2.1 | r1 M1 | The tracking product runs only on **countable** rows (the switching is unchanged: every other row resets whatever `e_N` is). `switch_e_gn` on every GN-phase step row; `switch_pKp` / `switch_e_n` only on countable rows. §1.1, §1.2, §1.5, §3.1, T4, the display; G-7 added |
| 2.2 | r1 M2 | §3.3 against timing draft 1: the per-point build is timing's `curvature` phase and the transpose its `gradient`, so the one remaining request is `PhaseSeconds.tracking`. Settled in the critic loop; G-5 removed; W1 laps `'tracking'` |
| 2.3 | r1 M3 | Docs: the switch takes §4.14 (timing keeps §4.13); the CHANGELOG appends to timing's `### Added` |
| 2.4 | r1 M4 | W3 reports the overhead in two currencies (wall time, apply-equivalents); G-4 gains a middle option (jit the tracking product); "ship in 2026.5.0 if 10×100 is not promising?" on the W3 agenda |
| 2.5 | r1 M5 | The padded-start stress case (`START=aligned` arms) moves into W3 at 10×100; W5 keeps the 20×1000 repeat |
| 2.6 | r1 M6 | The frontend runs `_check_switch_options` before `least_squares_problem`; T7 checks it with a spy |
| 2.7 | r1 M7 | `info.lm` on GN-phase rows is the GN view, the same object across a rejected trial (T8) |
| 2.8 | r1 M8 | `switched_at` = the index of the first row whose model is exact (`len(history)` when it fired on the last row); the converged row's switch fields defined (§3.1, §3.2, T8) |
| 2.9 | r1 M9 | T9 gains the gate "a `switch_tau=0` run after a pure GN-TR run compiles no new `_cg_core`" |
| 2.10 | r1 NITs | N1 not adopted: answer 4 says "applied to both errors"; §1.3 states that the counting equals the source's. N2 float32 advice (§1.6). N3 the reading of answer 7 (§3.1). N4 the count update tested through runs, no helper. N5 T3 asserts neither TR floor fires. N6 T6's baseline is GN-TR, verified before asserting. N7 `switch_k` with `switch_tau=0` raises. N8 the rule cited from `newton_cg`'s docstring. N9 a new forwarding subtest with non-default values; the existing test stays unedited. N10 lift diagnostics on GN-phase rows (§1.6). N11 the apply-equivalent bar is a lower bound (§6.2). N12 per-level `switched_at` read from the rows (§6.1) |
| 2.11 | r2 NIT | W1 laps `'tracking'`; the field ships with timing (its §2.7), and W1 adds it only if timing's draft 2 does not |
| 3.1 | Nick G-1 | Adopted as recommended: `cg_forcing_power` is the GN phase's power, `switch_cg_forcing_power` (None = 1.0) the exact phase's. No text change beyond §9 |
| 3.2 | Nick G-2 | `switch_tau = 0` and `switch_k = 0` allowed; T1 and T2 are public-API bitwise tests. No text change beyond §9 |
| 3.3 | Nick G-3 | `switch_fired` kept as a row field. No text change beyond §9 |
| 3.4 | Nick G-4 | Track every countable row. `switch_track_every`, the jitted tracking product and the measurement-gated default are removed (§1.5); the cost is still measured and reported at W3 in both currencies, as information. W3's review stop becomes a report (its agenda goes with it, the ship question included); W3b removed (§6.1, §8) |
| 3.5 | Nick G-6 | `_switch_sample` and `_gauss_newton_view` private. No text change beyond §9 |
| 3.6 | Nick G-7 | New `switch_track: Optional[str] = None` (None = `'countable'`; `'countable'` \| `'all'`, case-insensitive). Spelled as the `switch_*` group's other options (None = the default, so a value set outside switch mode raises) and validated as `step_control` is; raises with `switch_k=0`; checked in `_check_switch_options`, so on the frontend before `least_squares_problem`; reported as `stats['switch_track']`. `'all'` runs the product on every GN-phase step row and moves no count. §0, §1.1, §1.2, §1.5, §2.1–§2.3, §3.1–§3.3, T4, T7, T8, T11, §7 |
| 3.7 | Nick | §9 lists every item RESOLVED (2026-11-13); the status header |
| 4.1 | W1 handoff/review | W1's readings, recorded: `_require_exact_capable(..., hessian='exact')` so the message names the mode; two backend messages change (the invalid-`hessian` message lists `'switch'`; by the session's ruling the GN `curvature_damping` message reads "... only meaningful with hessian='exact' or 'switch'"); option types `numbers.Real` / `numbers.Integral` with bools refused; the new `NewtonInfo` fields follow `timing`, the new `stats` keys precede `stats['timing']`; the shared test helper `check_timing(..., switch=False)` (12/12 timing mutations caught identically before and after) |
| 4.2 | W1 handoff/review | Tests as realized: T4 by a forced hit, a scripted sample and CG stubs; T5 records the floor AND the gate's verdict (the plan's literal criterion alone is vacuous at float32, where `e_N ≥ e_GN`); T6 on `newton_cg`'s docstring problem; G-bit by the battery, not digests; beyond §5: one transpose and one companion per point, a capped CG step counting, the count resetting from 2 |
| 4.3 | W1 handoff | The self-expiring frontend mirror exemption (`not_yet`) and the `hessian` regex in `TestFrontendExactNewton.test_options_with_no_effect_and_unbuilt_geometries_raise`: W2 MUST delete/update them |
| 4.4 | W1 handoff/review | Printed run on the docstring example: ratios 0.013 and 0.003 on two clean steps, then the switch (`switched_at=2`); 6 Newton steps against GN-TR's 25. Compile budget confirmed: two CG programs per level, none recompiled on a repeat |

## 0. What and why

Gauss-Newton converges linearly on a nonzero-residual fit, at a rate set by the curvature term it drops; the exact
Hessian gives the superlinear tail but is indefinite far from a minimizer and costs 3–4× per apply (docs §4.12). Today
the user picks one per call, and T3Polynomial hand-schedules "exact past rank 10". `hessian='switch'` makes the choice
inside one `newton_cg` call: start with Gauss-Newton, measure after each clean step whether the exact model would have
predicted the actual change better, and switch for good once it clearly would have. With `s` the trial step (`αp` on
the line search, `p` in the trust region), `K_N = K_{Aᵀω²r} − K_{grad f}` and `L` the lift (on a corewise geometry
`L = I` and `K_N = K_{Aᵀω²r}`):

```
Δf      = f(R_x(s)) − f(x)                                    (known: the row's delta_f)
m_GN(s) = ⟨g, s⟩ + ½⟨s, H_GN s⟩                                (known: the row's `predicted`)
m_N(s)  = m_GN(s) + ½⟨s, Π Lᵀ K_N L Π s⟩ = m_GN(s) + ½α²·switch_pKp      (ONE curvature product per tracked row)
e_GN    = |Δf − m_GN(s)|,    e_N = |Δf − m_N(s)|
switch for good once e_N < switch_tau · e_GN on switch_k consecutive counted steps   (defaults 0.25, 2)
```

Why it works: `e_GN = ½|⟨s, K_N s⟩| + O(‖s‖³)` while `e_N = O(‖s‖³)` along a second-order retraction, so near a
nonzero-residual minimizer `e_N / e_GN → 0`; in the noisy transient the ratio is about 1 or more, and a threshold well
below 1, required twice in a row, rarely fires early. In words: **the exact model must have been at least four times
better than Gauss-Newton at predicting the objective, on two clean steps in a row.**

## 1. The algorithm inside `newton_cg`

### 1.1 State and the per-row steps (switch mode only)

Run state: `switched` (False; True from the start iff `switch_k = 0`), `count = 0`, `switched_at` (None; 0 iff
`switch_k = 0`). Every other mode runs today's loop unchanged.

1. **A new point** (`lm is None`). Switched: `lm = problem.local_model(x, frame=…, hessian='exact', …)`, exactly exact
   mode. Not switched: the same call gives the **tracking model** `lm_track`, and `lm = _gauss_newton_view(lm_track)`
   (§2.2) is what CG, the step control and the row's `info.lm` use. `g = lm_track.gradient` reads the model's one
   misfit transpose, which is the Gauss-Newton gradient bit for bit, so a point costs ONE transpose, as in either pure
   mode. A rejected trust-region trial keeps both models.
2. **Forcing.** `power = switch_cg_forcing_power if switched else cg_forcing_power`;
   `eta = min(0.5, (gnorm / cg_ref) ** power)` as today. `cg_ref` and `newton_ref` are resolved once, at iteration 0.
3. **CG** on `lm`, as today. The GN view has the Gauss-Newton treedef, so under `use_jit` the GN phase reuses pure GN's
   compiled `_cg_core` (T9 gates it), and the switch adds one compile of the exact structure: at most two per call
   (compile budget: a small fixed number inside a loop).
4. **The −g fallback** as today; a local flag `fallback` records it.
5. **`pHp`** as today for `lm`'s mode, so `ρ`, the radius update and every existing row field are Gauss-Newton's.
6. **Step control** as today. Before the switch `ρ` and the radius rules read `m_GN`, after it `m_N` (each phase is
   its pure mode).
7. **Tracking** (GN-phase rows with a step). `e_gn = |delta_f − predicted|` on every such row (free). If the row is
   **tracked** (countable, §1.2; under `switch_track='all'`, every step row):
   `switch_pKp = float(lm_track.curvature_quadratic(p))`, `curvature = ½α²·switch_pKp`,
   `(e_gn, e_n, informative, hit) = _switch_sample(delta_f, predicted, curvature, floor, switch_tau)` (§1.3).
   Otherwise `switch_pKp = switch_e_n = None`. Then `count = count + 1 if (countable and hit) else 0`. When
   `count ≥ switch_k`: `switched = True`, `switched_at = it + 1`, the row's `switch_fired = True`.
8. **The row** (§3). **Accepted:** `x, lm, lm_track = x_trial, None, None`.

The switch lands on a fresh linearization: only accepted steps count, so the next row rebuilds the model anyway (now
exact); no model changes mid-point. The radius carries over; `Δ0` (the GN Cauchy length) and `Δ_max` stay iteration 0's.

### 1.2 What counts, what resets

A GN-phase row is **countable** iff it is a full accepted model step:
`accepted and not fallback and (tr_accepted if trust_region else (ls_steps == 0 and not ls_failed))`.

| row | effect |
|---|---|
| countable, `e_N < τ·e_GN` | `count += 1` |
| countable, a miss, or uninformative (§1.3) | reset |
| a rejected trust-region trial; a line-search backtrack (`α < 1`); the −g fallback; a forced accept (`'accept'` after 40 halvings) | reset, with no product (one under `switch_track='all'`, which moves no count) |
| CG stopped on the boundary, truncated, or at its cap | **no special effect**: counted like any step (both models are evaluated at the step actually taken) |
| the converged row | none (no step) |

A reset is "something went wrong with the local quadratic model" (Nick, answer 5). By default
(`switch_track='countable'`) the tracking product runs only on countable rows: a non-countable row resets whatever
`e_N` is, so the product would decide nothing there. `switch_track='all'` runs it on every GN-phase step row, as the
source does (one per GN-phase iteration, rejected trials included, under the same rule): it reports `e_N` on the
non-countable rows too, at one product each, and moves no count, so the switch fires at the same row (G-7). Tracking
starts at iteration 0, with no delay (answer 6). Rows after the switch track nothing (one-way: no switching back).

### 1.3 The rounding gate

`Δf` is a difference of two objective values, so below the objective's resolution neither error can be measured. The
gate uses the library's dtype-aware scale (the trust region's `objective_floor` scale, the cancellation error of
`f = ½‖S(x) − y‖²`) times the source's constant:

```
resolution  = ε · (‖ω⊙y‖ · ‖ω⊙r‖ + |f|)        ε = _resolution_eps(lm.residual): float32's under default jit
floor       = _SWITCH_ROUNDING · _TR_RESOLUTION_FACTOR · resolution  = 100 · 10 · resolution
informative = isfinite(e_GN) and isfinite(e_N) and max(e_GN, e_N) > floor
hit         = informative and e_N < τ · e_GN
```

Applied to both errors (answer 4): both at the floor is uninformative, a reset. `e_GN` at the floor with `e_N` above it
is a miss, also a reset, so the counting equals the source's single gate `e_GN > floor`; only the label differs. `e_N`
at the floor with `e_GN` well above it is a hit. `‖ω⊙y‖` is computed once per run, lazily, as today. The expression
moves into `_objective_resolution` (§2.2), which the `objective_floor` test then calls with the same arithmetic.

### 1.4 Continuation, step control, forcing

- **Each `newton_cg` call starts in the GN phase with `count = 0`** (answer 8). There is no resume state, so a ladder
  passing `hessian='switch'` per level gets this by construction; after a rank increase we may have left the basin.
- **Default step control is the trust region** in switch mode (answer 3): `_check_step_control` maps `None` to
  `'trust_region'` for `'exact'` and `'switch'`. `'line_search'` is allowed, as with either pure mode.
- **Forcing power 0.5 before the switch, 1.0 after**, both settable (answer 2): `cg_forcing_power` (default 0.5,
  unchanged) is the GN phase's, `switch_cg_forcing_power` (None = 1.0) the exact phase's (names: G-1).

### 1.5 Cost, measured and reported

Per GN-phase **point**, the tracking model adds the geometry's `curvature_precompute` to a Gauss-Newton build (the
transpose is shared, step 1). On the manifold that is the lift's `2d − 1` small SVDs plus one small T3-to-T3 sweep:
W-free. Per **tracked** GN-phase row (countable, by default), one `curvature_quadratic`: lift, the kind's
`corewise_hessian` (W-sized, the dominant term), the tangential term, lift transpose; about one exact apply minus a GN
one, i.e. 2–3 GN applies. A GN row runs `cg_iters` GN applies plus a transpose and the retractions, so the overhead
ranges from a few percent (many CG iterations per step) to the source's estimate of 30–40 % (1–5 per step). Under
`use_jit` both tracking calls run eagerly in the host loop, where exact mode's build and `pKp` run today; on a small
problem that eager dispatch, not arithmetic, can dominate their wall time.

**The plan measures the overhead and reports it, as information** (slice W3, §6.1): no default or option depends on it
(G-4). The product runs on every countable row (every step row under `'all'`), with no thinning option and no jitted
variant; `switch_tau = 0` runs GN-TR's exact trajectory plus the tracking (T1), so its excess over GN-TR IS the
overhead. It is reported in **two currencies**: wall time (timing's split, compile seconds apart) and
apply-equivalents (tracking products and per-point builds × the multipliers measured at W3).

### 1.6 When it does not fire (documented, safe failures)

- **Flat sets**: manifold `d = 1` (the curvature term is 0, `e_N = e_GN`, never a hit); maximal-rank slots.
- **A first-order retraction**: the T3-SVD retraction is second order on a minimal, numerically full-rank frame. At a
  zero-padded start or near small singular values the damped lift (`ε`) and the filtered new levels keep `e_N` at
  `O(‖s‖²)`, and the switch holds off; the run stays Gauss-Newton and pays the tracking. The rows cannot show why:
  `lift_sigma_min` / `lift_eps` follow the row's model, None on GN-phase rows, and `info.lm` is the GN view. Accepted
  as the price of the row invariant (§3.1); the exact rows after a switch carry them.
- **Zero residual**: both errors shrink together and reach the floor; the switch may never fire, at no harm.
- **float32** (the default under `use_jit` without x64): `floor ≈ 1.2e-4 · (‖ω⊙y‖·‖ω⊙r‖ + |f|)`, so only large early
  steps are informative and the switch rarely fires. The docs advise jax x64 with `hessian='switch'`.

## 2. API

### 2.1 Keywords (backend and frontend, identical)

`hessian` gains the value `'switch'` (`newton_cg` only). Four keyword-only options, appended after `cone_seed` and
after (A)'s `timing`, one blank-line group:

```python
def newton_cg(
        problem, x0, ...,                                           # unchanged
        hessian:                    str = 'gauss_newton',           # or 'exact' | 'switch' (case-insensitive)
        ...,
        *,
        start_frame=None, padded_from=None, ..., cone_seed=None,    # unchanged
        timing:                     bool = False,                   # (A)

        switch_tau:                 typ.Optional[float] = None,     # None = 0.25; in [0, 1); 0 = track, never switch
        switch_k:                   typ.Optional[int]   = None,     # None = 2; int >= 0; 0 = exact from iteration 0
        switch_cg_forcing_power:    typ.Optional[float] = None,     # None = 1.0: the forcing power from the switch on
        switch_track:               typ.Optional[str]   = None,     # None = 'countable'; 'countable' | 'all' (§1.2)
) -> typ.Tuple[Tangent, dict]:                                      # (x_cores, stats)
```

Rules, structural, raised before any work (on the frontend before `least_squares_problem`, §2.3):

- The shipped rule in `newton_cg`'s docstring, "an option of a group set without its group engaged … raises": each
  `switch_*` option with `hessian` other than `'switch'` raises. So does an option the run would not read:
  `switch_tau` or `switch_track` with `switch_k=0` (nothing is tracked); `switch_k` or `switch_cg_forcing_power` with
  `switch_tau=0` (the count never moves, the switch never fires).
- `switch_k` must be an `int`, not a `bool`. `switch_cg_forcing_power` is unchecked, as `cg_forcing_power` is.
  `switch_track` is `'countable'` or `'all'`, case-insensitive; anything else raises, a non-`str` included, as with
  `step_control`.
- `hessian='switch'` runs `_require_exact_capable(kind, geom)` up front: a kind or geometry without the exact Hessian
  raises at the call, naming `'switch'`.
- `curvature_damping` is legal in switch mode (the tracking model and the exact phase read it); `tr_radius0` with the
  trust region; `on_line_search_failure='accept'` only with `step_control='line_search'`, as today.
- `Problem.local_model(hessian='switch')` and the frontend `*_model` factories raise: a model has one Hessian.

**Non-breaking**: with `hessian` not `'switch'`, every run is bitwise unchanged: `x`, the whole `stats` dict, and
every history row's keys and values (the G-bit procedure, T8).

### 2.2 Backend helpers (private, module-level, pure functions on basic types)

```python
_HESSIAN_MODES = ('gauss_newton', 'exact')                     # unchanged: a local model's modes
_NEWTON_HESSIAN_MODES = _HESSIAN_MODES + ('switch',)            # newton_cg's
_SWITCH_TAU, _SWITCH_K, _SWITCH_CG_FORCING_POWER = 0.25, 2, 1.0
_SWITCH_TRACKS = ('countable', 'all')                           # switch_track's values; None = the first
_SWITCH_ROUNDING = 100.0                                        # floor = this · _TR_RESOLUTION_FACTOR · resolution
_SWITCH_FIELDS = ('switch_pKp', 'switch_e_gn', 'switch_e_n', 'switch_count', 'switch_fired')

def _check_hessian(                                             # gains `modes`; otherwise as today
        hessian:  str,                                          # the selector (case-insensitive)
        who:      str,                                          # the caller, for the message
        modes:    typ.Tuple[str, ...] = _HESSIAN_MODES,         # the caller's allowed modes
) -> str:                                                       # the lower-cased mode

def _check_switch_options(                                      # §2.1's rules; also called by the frontend
        hessian:                  str,                          # the validated mode
        switch_tau:               typ.Optional[float],
        switch_k:                 typ.Optional[int],
        switch_cg_forcing_power:  typ.Optional[float],
        switch_track:             typ.Optional[str],
) -> typ.Optional[typ.Tuple[
    float,                                                      # tau
    int,                                                        # k
    float,                                                      # the forcing power after the switch
    typ.Optional[str],                                          # the tracked rows: 'countable' | 'all'; None iff k = 0
]]:                                                             # None outside switch mode

def _objective_resolution(
        eps:        float,                                      # _resolution_eps(residual)
        data_norm:  float,                                      # ‖ω⊙y‖
        misfit:     float,                                      # ½‖ω⊙r‖²
        f:          float,                                      # the objective at x
) -> float:                                                     # ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|), today's expression verbatim

def _switch_sample(
        delta_f:    float,                                      # f(trial) − f
        predicted:  float,                                      # m_GN(s): α·slope + ½α²·pHp, or slope + ½·pHp
        curvature:  float,                                      # m_N(s) − m_GN(s) = ½α²·switch_pKp
        floor:      float,                                      # the rounding floor (§1.3), >= 0
        tau:        float,                                      # switch_tau
) -> typ.Tuple[
    float,                                                      # e_gn = |Δf − m_GN(s)|
    float,                                                      # e_n  = |Δf − m_N(s)|
    bool,                                                       # informative: both finite, not both <= floor
    bool,                                                       # hit: informative and e_n < tau·e_gn
]:

def _gauss_newton_view(
        local_model:  LocalModel,                               # an exact-mode model
) -> LocalModel:                                                # the same linearization in Gauss-Newton mode
```

`_gauss_newton_view` is `dc.replace(local_model, curvature=None)`: frame, sweep, residual and companion shared, so its
`hvp` / `quadratic` / `gn_quadratic` / `retract` are Gauss-Newton's bit for bit, and its treedef is GN's (both checked
by the critic, manifold and corewise, with and without a regularizer). Its `gradient` would redo the transpose; the
loop reads `lm_track.gradient` instead. `_check_step_control(step_control, hessian)` keeps its signature; its default
rule widens to `'switch'`.

### 2.3 Frontend mirror (`optimizers.py`)

- `newton_cg` gains the four keywords with the backend's defaults (the existing mirror test enforces this). It calls
  `bopt._check_hessian(hessian, 'newton_cg', bopt._NEWTON_HESSIAN_MODES)` and then
  `bopt._check_switch_options(hessian, switch_tau, switch_k, switch_cg_forcing_power, switch_track)` before
  `least_squares_problem` (its minimal-rank reduction and the chunk estimator's compile are work); the backend checks
  again for raw-`.data` callers. It forwards each switch keyword only when not None, as the `padded_*` seam does.
- `least_squares_problem(hessian=)` accepts the three solver modes; `'switch'` sizes `chunk_size='auto'` as `'exact'`
  (the tracking product and the exact phase run the two-hole assemblies). Chunking is otherwise unaffected, and
  `estimate_chunk_size` keeps its two modes.
- The docstring gains one paragraph (the rule in words, the keywords and defaults, "each call starts in Gauss-Newton").

## 3. Reporting

### 3.1 Rows (`NewtonInfo`)

`hessian` becomes **the model of this row** (`'gauss_newton'` or `'exact'`; the model that solved the step, or on the
converged row the model built there). Outside switch mode that is the run's mode, as today; the docstring's "run-level
strings" list moves it to a per-row entry. `info.lm` is the row's model: on GN-phase rows the GN view, the same object
across a rejected trial (the display's reuse and the rejection contract hold); its `gradient` recomputes the transpose,
which the docstring notes. Five fields appended:

```python
    switch_pKp:   typ.Optional[float] = None   # tracked GN-phase rows: ⟨p, Lᵀ K_N L p⟩, tracking model (NOT in pHp)
    switch_e_gn:  typ.Optional[float] = None   # GN-phase rows with a step: |Δf − m_GN(s)|
    switch_e_n:   typ.Optional[float] = None   # tracked GN-phase rows: |Δf − m_N(s)|
    switch_count: typ.Optional[int]   = None   # GN-phase rows: the consecutive-hit count AFTER this row (0: reset)
    switch_fired: typ.Optional[bool]  = None   # every switch-mode row: True on the row whose count reached switch_k
```

- **The reading of answer 7**: the five fields are None on `NewtonInfo` outside switch mode and are omitted from
  non-switch history rows (the precedents: the `extra` stats keys, and the aligned-frame review's "history keys on
  default runs"). `_SWITCH_FIELDS` are excluded from `_NEWTON_SCALAR_FIELDS`; `_newton_scalar_record(info)` appends
  them iff `info.switch_fired is not None`, which is a bool on every switch-mode row. The existing
  `set(row) == set(_NEWTON_SCALAR_FIELDS)` test stands.
- **The converged row**: `switch_fired = False`; in the GN phase `switch_count` is the carried count; the other three
  are None. Exact-phase rows: `switch_fired = False`, the other four None.
- The tracking curvature lives in `switch_pKp`, so `pKp` stays None on GN-phase rows and the display's
  `κ = pKp / (pHp − pKp)` (which reads `pKp`, and assumes `pHp` contains it) is untouched. **The invariant: a GN-phase
  row equals the pure Gauss-Newton run's row, and an exact-phase row the pure exact run's, apart from the `switch_*`
  fields and `wall_time`.**

### 3.2 Run (`stats`) and the display

- Switch mode only: `stats['hessian'] = 'switch'`; `stats['switch_track']`, the resolved `'countable'` or `'all'`
  (None with `switch_k = 0`, which tracks nothing); and `stats['switched_at']`, **the index of the first row whose
  model is exact**: `it + 1` of the firing row, so `len(history)` when the switch fired on the run's last row; that row
  may be the converged row. 0 with `switch_k = 0`; None if it never fired. Other modes: no new key.
- **Display** (`optimizer_display.py`): a pure `_switch_segment(info)`, None unless the row is a GN-phase switch row
  (`switch_count is not None`): `sw 3.1e-01 ×1`, the ratio `e_N/e_GN` (`—` when `switch_e_n` is None, non-finite, or
  `e_GN = 0`), then the count. The firing row appends `⇒ exact`. Exact-phase rows show today's `κ σ ε` segment.
  Nothing changes outside switch mode (same strings). The `format_newton_iter` docstring gains one switch-row example.

### 3.3 Coexistence with timing (A)

Timing's draft 1 already places the switch's per-point build: the exact-mode build at a GN-phase point is its
`curvature` phase (`curvature_precompute` alone), and the transpose is its `gradient` in both modes (its Q-1(a)). One
request remains, settled in this critic loop and raised on the timing plan too: **`PhaseSeconds.tracking:
Optional[float]`**, a tracked GN-phase row's `curvature_quadratic` (countable rows; every step row under
`switch_track='all'`), None on every other row, so `forms` keeps its meaning. A row's tracking overhead is then
`seconds.curvature + seconds.tracking`, with the compile twins beside it.
`NewtonInfo.timing` is not a history field, like `_SWITCH_FIELDS` outside switch mode; the two do not interact. Timing
§2.7 lists four switch fields; it should list five (raised there by the critic).

## 4. Scope

- In: `newton_cg` on every geometry and representation with `hessian='exact'` (`MANIFOLD`, `COREWISE`, the uniform
  twins, the shared wrappers of all four), every kind, regularized or not. Chunking unaffected (§2.3).
- Out: `gradient_descent`, `mc_sgd`, `adam`; switching back; a delayed tracking start; a trigger with no product
  (Fletcher–Xu); carrying the state across continuation levels; a secant `S` (NL2SOL). Recorded in
  `deferred_and_rejected.md` (§7).

## 5. Tests

Numerics numpy-only, jax in `test_dispatch.py` (house rule). New class `TestSwitchNewton` in
`tests/backend/test_optimizers.py` unless noted. "Row-equal" = every history field except `_SWITCH_FIELDS` and
`wall_time`, compared with `==`. Each run property is verified on its problem by a script before it is asserted.

| # | test | criterion |
|---|---|---|
| T1 | **Threshold disabled = Gauss-Newton.** `hessian='switch', switch_tau=0` vs `hessian='gauss_newton'` with the same `step_control` (each of the two) and `cg_forcing_power`; ragged and uniform; manifold, corewise, shared manifold | `x` bitwise; `losses`, `newton_steps`, `newton_accepted`, `stop_reason` equal; rows row-equal; every row `hessian == 'gauss_newton'`; `switched_at is None` |
| T2 | **Forced at 0 = exact.** `switch_k=0, switch_cg_forcing_power=q` vs `hessian='exact', cg_forcing_power=q`, same matrix | as T1, every row `'exact'`, `switched_at == 0`, no tracking fields set |
| T3 | **The junction.** A run that fires at `j`: rows `< j` row-equal a GN-TR run's first `j` rows (`stop_reason` aside); rows `≥ j`, offset by `j`, equal (numeric fields) an exact run from `x_j` with `tr_radius0` = row `j`'s `tr_radius`, `g0norm_newton` / `g0norm_cg` = the switch run's references, `cg_forcing_power = q`, on a problem where neither `Δ_max` binds nor a TR floor fires in either run (asserted: they differ between the runs) | bitwise (numpy) |
| T4 | **Each reset condition**, through runs (the existing fixtures where they apply: `_OffsetProblem`, the descent-guard and exhaustion tests): a rejected TR trial, an LS backtrack, the −g fallback, a forced `'accept'`, each with `switch_count == 0`, `switch_e_gn` set and `switch_pKp is switch_e_n is None`; a countable boundary-CG hit increments the count. Under `switch_track='all'` the same reset rows carry `switch_pKp` and `switch_e_n` with `switch_count == 0`, and the run equals the default run (`x` bitwise; rows row-equal; `switch_count`, `switch_fired`, `switched_at` equal; `switch_pKp`, `switch_e_n` equal on countable rows). `_switch_sample` unit-tested over hit / miss / uninformative / non-finite | as stated |
| T5 | **The rounding gate under float32** (`test_dispatch.py`). A noise-free, true-rank fit run to its floor with `use_jit=True`, x64 off: the floor is computed with float32's ε; every countable row whose two errors are at or below it (recomputed from the row) has `switch_count == 0`; if a switch fires, its last `switch_k` rows are informative | as stated |
| T6 | **It fires where it should, and not where it should not.** The §4.12 doctest problem (rank-3 noisy target, rank-2 fit, probe derivatives order 1): fires; the `switch_k` rows before `switched_at` have `e_N < 0.25·e_GN`; `newton_steps` (rejected trials included) ≤ GN-TR's. Manifold `d = 1`: never fires, `switch_e_n == switch_e_gn` on countable rows | as stated |
| T7 | **Options.** Each §2.1 rule raises (message checked), on the backend and the frontend, the frontend's before `least_squares_problem` is called (mock spy, as the misspelt-option test); `step_control=None` resolves to `'trust_region'`; `switch_track='ALL'` resolves to `'all'`; `Problem.local_model` and a `fitting` factory reject `'switch'`; a kind without `corewise_hessian` raises at the call | as stated |
| T8 | **Stats and rows contract.** Switch mode: `stats` keys = today's + `{'hessian', 'switch_track', 'switched_at'}`, `stats['switch_track']` as §3.2; every row has the five switch keys; `switch_fired` True on exactly one row iff the switch fired (`switched_at` not None, `switch_k > 0`); `switched_at == len(history)` when it fired on the last row (`max_newton` set to end there); the converged row's fields as §3.1; a rejected trial keeps `x` and passes the same `info.lm` object (the GN view) to the next row. **G-bit**: a seeded ragged and uniform run per existing mode, recorded on the parent commit in the W1 handoff, equals the new code's `x` and whole `stats` bitwise | as stated |
| T9 | **Dispatch rows** (`test_dispatch.py`): `newton_cg(hessian='switch', use_jit=True)` ragged, uniform and shared uniform: jax in, jax out; after a pure `gauss_newton`, `step_control='trust_region'`, `use_jit` run at a structure, a `switch_tau=0` run at the same structure grows `_jitted(_cg_core)._cache_size()` by 0; a firing run grows it by ≤ 1 more; a second firing run by 0; eager-jax vs jit rows within the existing jit tolerances | as stated |
| T10 | **Display** (`test_optimizer_display.py`): the `sw` segment, `—` on a row with `switch_e_n` None, the `⇒ exact` mark, on hand-built rows; no `κ` on a GN-phase row carrying `switch_pKp`; output unchanged outside switch mode | strings |
| T11 | **Frontend** (`test_optimizers_frontend.py`): the mirror test passes unedited; a new forwarding subtest with `hessian='switch'` and the four keywords at non-default values (none dropped or swapped), and absent from the backend call when not given; `least_squares_problem(hessian='switch')` sizes as `'exact'`; a frontend switch run on ragged and uniform | as stated |
| T12 | The existing optimizer tests, display tests and doctests pass **unedited**; the full suite on both envs (`-n 3` at most on the laptop); the slow tier serially at release | green |

## 6. Benchmarks

### 6.1 Darcy 10×100, o01 configuration (laptop; slice W3)

T3Polynomial `scripts/d10_darcy_o01_fit.py` at its defaults (10 points × 100 directions, jet order 4, `N_CHUNK=5`,
`MAX_RANK=24`, `GTOL_REF=1e-3`) with `MAX_NEWTON=60` (so no arm is censored at 30), `BACKEND=uniform_jax`, `X64=1`,
`timing=True`; one job at a time. The knobs go in the script (`HESSIAN=switch`, a new `CG_POWER_AFTER`), never in
library code. Arms, all at equal tolerance, each with the default start (`START=library`) and, for the starred ones,
again with `START=aligned` (`padded_from` at every level: the no-switch stress case of answer 9):

| arm | settings |
|---|---|
| GN | `gauss_newton`, power 0.5, line search (today's default) |
| GN-TR\* | `gauss_newton`, power 0.5, trust region (separates the step control from the switch) |
| exact | `exact`, power 1.0, trust region |
| switch\* | `switch`, 0.5 → 1.0, trust region |
| tracking only\* | `switch`, `switch_tau=0`: GN-TR's trajectory plus tracking; its excess over GN-TR is the overhead |

Reported per level and in total: Newton rows and accepted steps, CG iterations split by row Hessian, tracking
products, `switched_at` (read from the level record's `history`: the first `'exact'` row; T3Polynomial's records are
unchanged), stop reasons, train / val / test errors (must not be worse than GN's), and the overhead in both currencies
(§1.5): wall time with timing's split (`curvature + tracking` share of the GN phase, compile seconds apart) and
apply-equivalents. On the aligned arms: how often the switch fires on padded levels and the tracking share there. The
table goes in the W3 handoff as a report: no decision hangs on it (G-4 and G-7 are resolved, §9).

### 6.2 Then, if 10×100 is promising

"Promising": the switch arm's apply-equivalents (GN applies + m_N × exact applies + m_t × tracking products, with
`m_N`, `m_t` measured at W3) are below GN's at no loss in val/test error. This is a **lower bound** on the switch's
advantage: it leaves out the per-step costs a shorter run also saves (model builds, trial retractions and objectives).

- **20×1000** on a Runpod GPU (scripts and notes in `nicks_research_experiments/tensor/runpod`, never in
  T3Polynomial), the same arms including the aligned ones, full ladder (early-level timings do not predict
  large-scale cost).

## 7. Docs

- **`docs/fitting_and_optimization.md`, a new §4.14 "Switching from Gauss-Newton to the exact Hessian"** after
  timing's §4.13, opening with a pointer back to §4.12; nothing landed is renumbered. Contents: the rule in math and
  words; the counting table; the rounding gate; the four keywords in §4.12's table style; the defaults (TR,
  0.5 → 1.0); cost; reporting (row fields, `stats` keys, the display segment); when it does not fire (§1.6, with the
  x64 advice); each call starts in Gauss-Newton. A platform-stable doctest on §4.12's problem: `stats['hessian']`,
  `switched_at` not None, rows' `hessian` before and after, Newton steps ≤ GN-TR's. §5's "The Newton schedule is yours"
  bullet names `'switch'` as the automatic schedule.
- **Docstrings**: backend `newton_cg` (the Hessian paragraph, the keywords, `stats`), `NewtonInfo` (`hessian` per row,
  `lm` on GN-phase rows, the five fields), frontend `newton_cg` and `least_squares_problem`, `format_newton_iter` (an
  example row).
- **CHANGELOG `[2026.5.0]`**, appended to the `### Added` that the timing slice creates: `newton_cg(hessian='switch')`
  with `switch_tau` / `switch_k` / `switch_cg_forcing_power` / `switch_track`; the `NewtonInfo` switch fields;
  `stats['hessian']` / `stats['switch_track']` / `stats['switched_at']`; the display segment. Release notes: in the
  bundle's release commit.
- **Contributor**: `deferred_and_rejected.md` (I-36 gains the switch; §4's out-of-scope list as deferred);
  `curvature_internals.md` (one paragraph on the GN view: one build, one transpose, two compiled CG programs).

## 8. Slices

Process (HANDOFF, item B): a persistent implementer and reviewer (Opus, xhigh), each first reading a handful of
`TuckerTensorTrain` methods and their backend functions, `CLAUDE.md`, the signature / naming / testing docs, then this
plan. Per slice: `dev/gn_newton_switch_slices/W<k>_handoff.md` → `W<k>_review.md` (style section first; `VERDICT:
APPROVE` / `REVISE`) → fixes → the session reads the diff, runs the full suite, commits by name, pushes. Unforeseen
issues: the session decides and records them in `dev/gn_newton_switch_judgment_calls_2026_11.md`.

| slice | content | files | gates |
|---|---|---|---|
| W1 | Backend: modes and checks, `_check_switch_options`, `_objective_resolution` (bitwise refactor), `_switch_sample`, `_gauss_newton_view`, the loop (§1), the `NewtonInfo` fields and `_newton_scalar_record`, the `stats` keys; with timing landed, lap `'tracking'` (the field ships with timing, its §2.7; W1 adds it only if timing's draft 2 does not) and the tracking build as `curvature` (§3.3) | `backend/optimizers.py`, `tests/backend/test_optimizers.py`, `tests/test_dispatch.py` | T1–T9, T12, G-bit |
| W2 | Display segment; frontend keywords, early checks, forwarding, `least_squares_problem` sizing; docstrings of the touched functions | `backend/optimizer_display.py`, `optimizers.py`, `tests/backend/test_optimizer_display.py`, `tests/test_optimizers_frontend.py` | T7 (frontend), T10, T11, T12, doctests |
| W3 | Measurement and the 10×100 benchmark (§6.1, both starts), run by the session in T3Polynomial (its own suite and commits). **A report, not a stop**: §6.1's table in the W3 handoff, as information (G-4 and G-7 are resolved) | T3Polynomial `scripts/`, `data/` | §6.1's table |
| W4 | Docs, CHANGELOG, contributor records (§7). **Review stop** (the bundle's last before the release commit) | `docs/`, `CHANGELOG.md` | doctests; `sphinx -W` |
| W5 | Optional, before the tag if §6.2's bar is met: 20×1000 on GPU, both starts | research repo, T3Polynomial `data/` | the table |

## 9. Decisions (all RESOLVED, Nick, 2026-11-13)

- **G-1, the forcing-power keyword names. RESOLVED 2026-11-13: as recommended.** `cg_forcing_power` (default 0.5,
  unchanged) stays the Gauss-Newton phase's power; `switch_cg_forcing_power` (None = 1.0), in the `switch_*` group, is
  the exact phase's (§1.4, §2.1). No existing default or meaning changes.
- **G-2, the boundary values. RESOLVED 2026-11-13: both allowed.** `switch_tau = 0` (track, never switch) and
  `switch_k = 0` (exact from iteration 0) are legal, so T1 and T2 are public-API bitwise tests; `switch_tau = 0` is
  also the overhead measurement (§6.1).
- **G-3, `switch_fired`. RESOLVED 2026-11-13: added** (§3.1), so the display's `⇒ exact` mark stays stateless.
- **G-4, the tracking default. RESOLVED 2026-11-13: track every countable row** (the default of G-7). No
  `switch_track_every`, no jitted tracking product, no measurement-gated default ("no need for more elaborate options
  at this time"). The cost is still measured and reported at W3 in both currencies, as information; no decision hangs
  on it (§1.5, §6.1, §8).
- **G-5**: removed in draft 2 (settled in the critic loop, §3.3).
- **G-6, public or private helpers. RESOLVED 2026-11-13: private.** `_switch_sample` and `_gauss_newton_view` stay
  private in this release; public when a user asks.
- **G-7, where the tracking product runs. RESOLVED 2026-11-13: the user's choice, default countable.**
  `switch_track: Optional[str] = None` (None = `'countable'`; `'countable'` | `'all'`, case-insensitive), validated in
  `_check_switch_options` (so on the frontend before `least_squares_problem`) and reported as `stats['switch_track']`
  (§1.2, §2.1–§2.3, §3.2). The switching is the same under either value: only countable rows count, so the switch
  fires at the same row; `'all'` adds `e_N` on the non-countable step rows, at one product each.
