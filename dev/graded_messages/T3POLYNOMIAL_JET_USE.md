# How T3Polynomial uses jetted fitting, and what it implies for derivative coordinates

Read-only study, 2026-11-22. T3Polynomial at `dbda0de` (`/home/nick/repos/T3Polynomial`; it calls released
T3Toolbox 2026.5.0). Question: should derivative coordinates be exposed as (i) Taylor everywhere with one
conversion at the boundary, (ii) a first-class derivative-coordinate option, or (iii) something else
(`DATA_METRIC_PLAN.md` §1 and Q7; `SAMPLING_AUDIT.md` D4). Paths below are relative to the T3Polynomial repo.

## 1. Summary of the use

T3Polynomial fits one degree-`g` homogeneous polynomial `p(z) = T(x̂,…,x̂,·)`, `x̂ = (1, z)`, to derivative jets of a
whitened PDE map. The coefficient tensor `T` has `g+1` modes: `g` input modes of size `n+1` and one output mode of
size `m`. It uses exactly one jetted kind: **`probe_derivatives` with one jet variable** (`F = (J+1,)`), along lines.

- **The data.** At sample point `z`, unit direction `v` and unit output functional `ω`, implicit_probing returns
  forward jets `y_t = D^t f(z)[v^t]` for `t = 0..J`, and reverse jets `ψ_{t+1} = ω^T D^{t+1} f(z)[v^t, ·]`
  (`jets.py:9-13`, `datagen.py:105-111`, `toy.py:88-116`). The maps are Darcy, eigenproblem, OT, DEQ (via
  `jax.experimental.jet`) and cubic Poisson. `S` points × `D` directions are flattened to `W = S·D` rows
  (`jets.py:243-283`).
- **The seam** is `assemble_probe_fit_data` (`jets.py:120-240`), returning `ProbeFitData` (`jets.py:77-117`). It
  pivots the two measured families into one target per tensor mode:
  - Output mode, slot `j`: `y_j`.
  - Input modes, slot `j`: `G^(j)`. The z-block is `ψ_{j+1}/g`. The e₀ entry comes from the Euler identity
    `ω·y_j − ⟨ψ_{j+1}, z⟩/g − (j/g)⟨ψ_j, v⟩` (`jets.py:219-224`; `docs/offset_orders.tex` eqs. out/zblock/euler).
- **Sample.**
  - `ww = [x̂]*g + [ω]`, `pp = [v̂]*g + [0]`, with `v̂ = (0, v)`.
  - `data = [G]*g + [y]`: blocks of shape `(J+1, W, N_i)`, slot axis first. The `g` input entries are the same
    array object.
  - `J` is in `1..g`, default `J = g = 4`. At `J = g` the input-mode row at slot `g` is zero-filled and given
    weight 0 (`live` mask, `jets.py:226-229`).
- **The fit.**
  - `topt.newton_cg(geometry, 'probe_derivatives', (ww, pp), data, x0, order=J, weight=weight, ...)`, ragged or
    uniform, numpy or jax (`fitting.py:225-226, 236-237`), inside rank-continuation ladders (`fitting.py:383, 595`).
  - The model is evaluated with `X.probe_derivatives(ww, pp, J)` at about 20 sites, always compared with
    `fit.data` (`fitting.py:384-386, 597-599`; scripts `d05:155,224`, `d06:89-90`, `d09:40`, `e01:286,291`,
    `o01:361,366`, `n01:311`, `s05:138`, `x02:136`, `x06:68`, `x07:87,91`, `g05:84-85`, `s00b:152`).
  - Prediction uses `T.probe(ww)[-1]`, with no jets (`fitting.py:648-667`).
- **Research scripts** go below the frontend.
  - `least_squares_problem` / `local_model`: `h01:63-68`, `d07:65,120,140`, `d08/d11`, `g06*`, `h03/h04`.
  - The model factory `tfit.probe_derivatives_model(..., residual, weight=, hessian=)` with a hand-formed **primal**
    residual `model − data`: `x06:68-69`, `x07:86-88`.
  - The backend `bfit.probe_derivatives_kind(J, weight)` and its `kind.transpose(r, sample, frame, sweep)`, which
    applies `ω²` inside, on hand-formed residuals: `g01:48, 58-59, 136`, `g06:84`, `g06b:41`, `g06c:99,105`. One
    hand-built **covector**: `h01:119-125` divides a weighted residual by `ω` so that the internal `ω²` yields
    `J^T ω s`.
- **No other jetted use.** Nothing calls `apply_derivatives`, `entries_derivatives`, `prepare_derivative_data` or
  `taylor_to_derivatives`; it predates them.

## 2. Coordinates and weights, and why

**Coordinates: derivatives throughout**, and stated as such: "jets are jets — no factors" (`jets.py:31`,
`offset_orders.tex:117`). The sources produce derivatives: implicit_probing (`D^k q[…]`, no `1/k!`;
`implicit_probing/docs/jax_hook.md:71-72`) and `jax.experimental.jet`. The released `probe_derivatives` returns
`d^j/ds^j`. These match, so no conversion exists anywhere. The input jets are lines (slots 0 and 1 only), and
`0! = 1! = 1`, so **the sample is the same in both coordinate systems; only the data and the weight carry
factorials.**

**Weights.** `weight` is the probe block form `(g+1, J+1)`. It is passed as `ω`, not `ω²` (matching plan Q6).
- **Production:** `per_block_weights` (`jets.py:390-416`) gives `1/‖data[i][t]‖` per (mode, slot) on live blocks,
  exact 0.0 on dead cells, and raises on a near-zero live block. The floor was removed on purpose: it had turned a
  dead block into a weight of 1e12 (open-issues 9; `dev/jet_order_design.md` item 4). Used by d10, e01, o01, n01,
  g01, h01, x06, x07 and others.
- **Legacy:** `per_order_weights` (`jets.py:367-387`), `1/RMS_t`, a **bare `(J+1,)` vector**: `s05_fit.py:98`,
  `s00_toy_validation.py:71,101`, `tests/test_fitting_recovery.py:51`.
- **Why per block:** the jets decay by orders of magnitude per slot, up to seven decades on the eigen map
  (eigen design note §8.3). Each block should count by its *relative* error.
- **No per-sample or per-entry weight exists.** The per-row control in open-issue 29 is achieved by scaling the
  directions (§4).

**The key structural fact: everything T3Polynomial computes is invariant under any positive per-slot rescaling of
the data, provided the weight is computed from the data in the same coordinates.** The weights are data-relative
per block (or per slot), so `ω_b ⊙ r_b` does not change. Therefore none of these change:
- the objective, `g = J^T M r`, the GN and exact Hessians, the Newton and CG trajectory, and the zero/cone
  reference `‖P J^T M r(0)‖`;
- the display;
- every diagnostic, all of which are per-slot relative: `per_order_misfits`, `refined_misfits`,
  `forward_summed_reverse_misfit`, `weighted_misfit` (`jets.py:286-436`), and `d09:38-45`;
- the selection rule.

`k!` is one such rescaling. So are `κ^t` (`input_scale`, output modes) and `‖v‖^t` (`normalize`). For T3Polynomial,
**the coordinate choice is bookkeeping, not mathematics**, as long as data and weight stay together.

**Its mathematics is more natural in Taylor coordinates.** Divide the slot-`j` identities by `j!`, with
`Ψ[j] = ψ_{j+1}/j!` and `c_y[j] = y_j/j!`.
- The Euler e₀ entry becomes a plain Cauchy product with **no Leibniz factor `j`**:
  `G_T[j][0] = ω·c_y[j] − (⟨Ψ[j], z⟩ + ⟨Ψ[j−1], v⟩)/g`. Verified independently with a random symmetric degree-4
  tensor to 4e-14 (scratch script, not T3Polynomial code).
- The toy's falling factorials `(g)_t` (`toy.py:95-116`) become binomials `C(g,t)`.
- Open-issue 29's per-row radius estimator `(t+1)·RMS(y_t)/RMS(y_{t+1})` is already a ratio of Taylor coefficients.

## 3. Implications and recommendation

**Recommendation: (i).** Taylor coordinates everywhere, with T3Polynomial converting once at **its own seam**
(`assemble_probe_fit_data`). That is, ProbeFitData holds Taylor data, and the weights are computed from it.

- **Why (i) costs T3Polynomial almost nothing.**
  - The fit and every diagnostic are unchanged (§2), and per-block weights computed on Taylor data are exactly the
    `k!`-converted derivative weights.
  - `prepare_derivative_data` is not even needed: dividing `G` and `y` by `j!` in the assembler suffices, or
    rewriting it in the factor-free Taylor form.
- **Where the risk under (i) actually sits.** It is *primal* mixing, not the dual trap.
  - If T3Polynomial converted only at the `newton_cg` call (`prepare_derivative_data` inside `_newton_fit`), its
    ~20 comparisons of the model's (now Taylor) outputs against the stored derivative `fit.data` would be silently
    wrong by `j!` per slot.
  - So would `x06`/`x07`'s hand residual `model − fit.data`.
  - So would a `per_block_weights(derivative data)` passed beside Taylor data: slot 4 weighted 24× too low, 576×
    in the objective.
  - The library's documentation should therefore say: **convert once, at assembly, data and weight together, and
    keep every stored copy in Taylor coordinates**. For a data-relative weight, the simplest correct recipe is
    "convert the data, then compute the weight", which makes the weight conversion unnecessary.
- **The dual trap (D4) in T3Polynomial.** It is confined to scripts that will break anyway on the new transposes
  (`g01`, `g06*`, `h01`, all private backend API).
  - Under (i) with the data converted at the seam, `J^T(data_metric(kind, order, r, weight, None))` with Taylor
    `r` and Taylor `ω` is right, with no covector conversion.
  - The trap appears only if a script forms `c_d = ω_d² r_d` from leftover derivative arrays.
  - `h01`'s formulation is worth noting: it works in the weighted space `A = ωJΠ`, `s = ω⊙r` (`h01:17-21`), and
    `s` is coordinate-invariant (`ω_T r_T = ω_d r_d`). Only `c = ω⊙s` transforms (`c_T = k! c_d`).
- **(ii) buys T3Polynomial nothing.** A derivative-coordinate option would let it keep its current arrays, but its
  objective is coordinate-free, so that saves only the one-time edit at the seam.
  - It would add a flag to forwards, transposes, `data_metric`, the weight rules and the display.
  - Done consistently it has no trap: `J_d^T c_d = J_T^T(k! c_d)` with the plain slot-sum pairing in derivative
    coordinates. But it doubles a surface whose only consumer does not need it.
- **(iii): no new mechanism; two documentation points that this use surfaces.**
  1. *A data-relative weight makes the coordinate choice moot.* `equalizing_weight(data_block_norms(Taylor y))`
     equals `k! × equalizing_weight(data_block_norms(derivative y))`, except where the relative floor bites. Say so
     beside `prepare_derivative_data`. Its `weight=None` default (ones on derivatives = `k!` on Taylor) is not what a
     user with decaying jets wants; T3Polynomial never fits unweighted.
  2. *`equalizing_weight`'s relative floor is coordinate-dependent.* For T3Polynomial-like data (derivatives
     decaying by decades), Taylor coordinates *widen* the block spread by up to `J!` (24 at `J = 4`). The eigen map
     at σ/g = 0.05 already spans `‖y₀−ȳ‖/‖y₄‖ ≈ 1.3e6` in derivative coordinates, past the default `floor = 1e-6`.
     The library's own advice ("rescale the direction first, then equalize") covers it. T3Polynomial does exactly
     that with `κ` and uses an explicit live mask instead of a floor.

## 4. Surprises and weight uses

- **The bare per-slot vector is gone (decision 2.5).** `s05:98`, `s00:71,101` and `tests/test_fitting_recovery.py:51`
  pass a `(J+1,)` weight. `weighted_misfit` and `_live_data_count` implement the old bare-vector rule locally
  (`jets.py:427-429`, `fitting.py:147-150`). These break **loudly**. At `J = g` the length `J+1` equals the mode
  count `d`, so the vector parses as a `d`-leading weight with rank-0 rows, which rule 2 rejects. A raise, not a
  misread.
- **Open-issue 29** (`dev/open_issues_2026_08_27.md:742-767`) wants a per-row, per-slot weight of the tied form
  `c_w^t` ("trust each jet to its radius"). Today it is only reachable by scaling the directions with
  `normalize=False`. The plan's full form `(J+1, W, 1)` per mode expresses it directly.
  - `c_w^t` is the same in both coordinate systems.
  - A zero per-row weight would also let near-crossing rows (the eigen map's `min_gap` discards) stay in the dataset
    without dominating.
  - **Served.**
- **The e₀/z balance inside the input-mode rows is a per-component weight T3Polynomial cannot express today.**
  `κ` conflates two things: a preconditioner (output modes) and an e₀-versus-z reweighting (input modes), "NOT
  loss-neutral on the input modes; it is a reweighting there" (`jets.py:155-159`; eigen note §8.3 "Correction").
  - The plan's full form with an edge axis and size-1 sample axes, `(J+1, 1, n+1)` per input mode and `(J+1, 1, 1)`
    for the output, decouples the two at no `W` cost. **Served, but only by rule 3's size-1 axes.**
  - Input edges `n+1` differ from the output edge `m`, so a uniform run exercises plan §4's `pack_batch` padding path.
    T3Polynomial runs `uniform_jax` in production (`fitting.py:228-237`).
- **`κ_auto` would change silently in a migration.** It is `(‖y₀−ȳ‖/‖y_J‖)^{1/J}`, read off the *assembled*
  output-mode block norms (`e01_eigen_fit.py:226-229`, `d10_darcy_o01_fit.py:117-120`, `g01_g0_common.py:90-92`).
  - If ProbeFitData moves to Taylor coordinates, `κ_auto` grows by `(J!)^{1/J}` (2.21 at `J = 4`), and `κ` is not
    loss-neutral.
  - This is the one silent behaviour change a T3Polynomial port would make. It is T3Polynomial's to pin, not the
    library's.
  - The docstring calls it "≈ the Taylor radius" (`jets.py:153`), but the true Taylor-radius estimate is the Taylor
    ratio.
- **`input_scale` is, by my reading, the library's `rescale_coordinates` composed with `rescale_direction`.** The
  first scales the input modes' z-coordinates by `1/κ`, the second gives length `κ`. The result matches z-block
  `κ^{t+1}`, e₀ `κ^t`, output `κ^t` (`jets.py:210-217`). T3Polynomial re-derives `backend/data_preparation.py`, a sign
  that that module is the right home for "change of data coordinates" in general.
- **The plan's §2 trap applies only to tiny tests.** A per-sample probe weight written as `(W,)` per mode is read as
  the block form when `W == J+1`. T3Polynomial has `m = 1` and one flat `W` axis, so production sizes (`W` in the
  thousands) raise on the size check.
- **Terminology hazard at the source.** implicit_probing's README calls its outputs "a Taylor coefficient"
  (`implicit_probing/README.md:63-64`), though they carry no `1/k!`. The boundary where (i) requires a conversion is
  exactly where a user is told their data are already Taylor coefficients.
- **Other breakage, expected and loud:**
  - `tests/test_jets_identities.py` uses `sd.dense_probe_derivatives` as the derivative-coordinate oracle
    (`:16, 48, 67, 134, 146-148, 234`).
  - The scripts above use private backend API (`topt._geometry_ops`, `bfit.probe_derivatives_kind`,
    `kind.transpose` with `ω²` inside).
  - `docs/offset_orders.tex` eqs. (out, zblock, euler) and `jets.py:28-37` need their Taylor forms.
