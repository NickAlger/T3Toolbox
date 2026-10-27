# S0b — T3Polynomial downstream measurement on T3Toolbox 2026.3.0-rc (main @ 3158778), 2026-10-27
**Setup.** T3Toolbox is a `git archive 3158778` snapshot (`t3toolbox_3158778/`), so S7's in-progress edits are not in it. T3Polynomial @ 388090d is used read-only. Runs are single-thread, from `source env.sh` (`ORIG=1` = the untouched clone).
**`implicit_probing` is not installed** (e01/o01 datagen and `sketch_gradients` import it), so `shim/implicit_probing` stands in for it: the same API, with jets by nested jax AD of the implicit map. `check_shim{,_ot}.py` validates it: y1 and ψ1 equal the exact Jacobian to 1e-16, ⟨ψ_{t+1},v⟩=⟨ω,y_{t+1}⟩ to 1e-14, and y2 agrees with an FD.
d05/s05 need FEniCSx datasets (`data/nx24/darcy_*`, `data/probes.npz`), so they were **not run**. Every run script is in `scripts/`, the minimal edits are in `scripts/*.s0b.diff` and `t3p_patched_fitting.diff`, and the logs are in `runs/*/`.

## 1. Call-site inventory (§K checked against the code)
| file:line | call → keys | on 2026.3.0 | S14 fix |
|---|---|---|---|
| fitting.py:105-112 | `bopt.least_squares_problem(topt._geometry_ops(..), bfit.probe_derivatives_kind(J,w),..)`, `local_model(zero).gradient` | runs (private helper kept) | K.1 deletion; moves stops ×0.73–2.18 (a) |
| fitting.py:189-197 | `g0norm_newton=<own zero>`, `gtol_rel=1e-3`; the opt-out `g0norm_newton=None` drops the key | **SILENT**: the opt-out now runs with `g0_reference='zero'` (`s0b_p3_warmstart.py`), while the warning still says "warm start" | `g0_reference='warm_start'` (P3) |
| fitting.py:199,205,208 | `topt.newton_cg(geom,'probe_derivatives',s,d,X0,order,weight,**kw)`; kw = max_newton, verbose, val_sample/val_data, g0norm_newton, gtol_rel, use_jit + newton_kwargs {chunk_size, callback, hessian} | OK (all are explicit keywords) | — |
| fitting.py:353,364,513,525 | `stats['newton']` | **BREAKS**: KeyError after level 0 (observed in e01, o01, b01, tests) | `['newton_steps']` (+ record `newton_accepted`/`stop_reason`; TR rejections count as steps) |
| e01:255,293 / o01:329,368 | `newton_kwargs=dict(chunk_size=None)`; `rec['newton']` | break through fitting.py:513 (observed) | none (`newton_kwargs['hessian']='exact'` already works) |
| d05:142,149,232,237,272 | `newton_kwargs={chunk_size}`, SHARING env; reads ls_steps (filters None), cg_truncated, rec['newton'] | breaks through fitting.py (not run) | K.3 as drafted |
| s05:102-112 | no newton_kwargs; SHARING env | breaks through fitting.py (not run) | none (**not in §K**) |
| x06:59-61,65-113 | `tfit.probe_derivatives_model(M,Y,ww,pp,J,res,weight=)`; `.objective_value/.gradient/.gn_hessian/.gn_quadratic/.frame` | runs **unmodified** (checked) | exact column (in the copy) (d) |
| x07:72-74,102-106,117,154 | the same model; `_newton_fit(X0,..,max_newton=N_ITER,verbose=False)` reads `st['history']` only | runs **unmodified** (checked) | HESS flag; controls need `gtol_rel` (c) |
| b01:147,232,275-278 | `fit_rank_continuation(newton_kwargs={'chunk_size':None})`; `_newton_fit(**NEWTON_KWARGS)`; callback `info.ls_steps` | :147 breaks (KeyError at :353, observed); :275 OK (exact: every row is None, so the histogram is `{}` "over 71 iterations") | count LS rows only |
| tests/test_fitting_selection.py:86,89,91 | `st['newton']`; `g0norm_newton=None` | 5/5 ERROR on the original package; 1/5 on the patched one; OK after the K.2 rename (`test_fitting_recovery` OK) | K.2. :91 passes **vacuously** (1 ≥ 1), so the label assert is needed |

**Discrepancies with §K.**
(a) K.1 treats the library `'zero'` reference as a drop-in for `_zero_gradient_norm`. It is evaluated at a different zero frame (Haar vs `zeros()`' LAPACK completion). On e01 the library/downstream ratio is **2.18 (L0, rank 1), 0.73, 0.94, 0.97**, and every stop moves by that factor.
(b) s05 is absent from §K. It needs no edit, but it breaks through fitting.py.
(c) In K.3, x07's GN control (`_newton_fit` at its default gtol 1e-3) is a 1-step no-op from a converged checkpoint, so it needs `gtol_rel`.
(d) The x06 exact column must not take `MANIFOLD.norm(H_fd - Hp)`, which raises at t ≤ 1e-5 (§5).
(e) Every §K line number checked out exactly. (f) The K.4 "e01 smoke run" needs `implicit_probing` (or this shim).

## 2. Run status (small sizes; GN = the default `hessian='gauss_newton'`)
| script | size (what was cut) | original package | patched package (GN) |
|---|---|---|---|
| e01 | N_U=16, N=6, S×D=16×16, J=4, N_FLOOR=3000, N_TEST=500, S_TE=20, MAX_LEVELS=4, N_CHUNK=2 | KeyError fitting.py:513 at the end of L0 | 30/30/30/30 steps (all at the cap), fit 39 s, test q 0.178. With MAX_NEWTON=100: 80/44/100/100, 166 s, q 0.154 |
| o01 | N_GRID=16, N=4, S×D=6×24, MAX_LEVELS=3, MAX_RANK=6, N_FLOOR=2000 | KeyError fitting.py:513 | 30/30/23, fit 22 s, L2 dof 380, q 0.048 |
| b01 arms | `--cells 16x6 --max-levels 3` (Darcy dims kept) | KeyError fitting.py:353 | ragged 20/17/10 (4.0 s); uniform 20/30/13 (7.7 s; the known backend divergence at L2) |
| x06 / x07 | e01 MAX_NEWTON=100 checkpoint L3 (dof 1001; GN ρ 0.032, \|g\|/g0 1e-3) | run | run (§3) |

## 3. FD cross-checks (`x06` copy + exact column; ‖H_fd p − hessian(p)‖/‖hessian(p)‖, with H_fd = x06's central difference of P_X grad f(R(±tp)))
| e01 L3 direction | t=1e-2 | 1e-3 | 1e-4 | 1e-5 | 3e-6 | GN rel (every t) |
|---|---|---|---|---|---|---|
| steepest descent | 4.3e-2 | 4.8e-4 | 4.8e-6 | 4.8e-8 | 4.3e-9 | 0.14 |
| GN Newton-CG dir. | 1.9e-2 | 2.0e-4 | 2.0e-6 | 2.0e-8 | 4.9e-9 | 0.94 (pᵀHp: exact 160.2 vs GN 81.3) |
| random ×2 | 1.0e-2 / 9.7e-3 | 1.0e-4 / 9.9e-5 | 1.0e-6 / 9.9e-7 | 1.0e-8 / 9.9e-9 | 2.5e-9 / 1.0e-9 | 0.10 / 0.11 |

The error is **O(t²) down to a ~1e-9 floor**. `quadratic(p)` matches d2 = (f₊−2f₀+f₋)/t² with the same O(t²) behaviour (2.7e-6 at t=1e-4). L2 (dof 425) at t=1e-4 gives 3.1e-6, 5.1e-6, 3.7e-7 and 4.6e-7, while GN is off by 0.11–1.08.

**x07 Newton loop** (10 iterations, CG_MAX 25, η 0.1; from the same checkpoint):
| solver | outer | H-actions / CG | \|g\| 4.2e-2 → | wall |
|---|---|---|---|---|
| x07 with its FD Hessian (H_STEP 1e-3), original script | 10 | 260 (every CG hits its cap; resid/\|g\| 0.2–0.9) | 2.1e-6 (linear, ratio 0.13–0.87) | 28 s |
| x07 loop with `hessian(p)` (the C.13 payoff) | 10 | 260 (the same cap) | 9.5e-7 (ρ = 1.00 every step) | 39 s |
| library exact `newton_cg` (TR, gtol 2e-8) | **6**, 0 rejected | 205 | **1.1e-8** | 31 s |
| library GN (gtol 2e-8) | 10 (cap) | 258 | 3.3e-2 | 13 s |

## 4. Exact vs GN (ragged, unshared, probe_derivatives; default TR and damping)
- **Tail** (e01 L3 checkpoint, gtol 1e-9, max 30 steps).
  - GN takes 30 steps (ρ 0.02–0.03), 817 CG, and only gets \|g\| from 4.2e-2 to 2.2e-2 (43 s). GN+TR gets to 1.5e-3.
  - Exact takes **6 steps** and 205 CG: …2.5e-3 → 1.8e-5 → 1.1e-8 (30 s). With p=1 it reaches 3.5e-12 (322 CG).
  - Damping 0, 1e-6 and 1e-3 give the same steps and CG (206/205/207), which **supports O-1: 1e-6 ≡ 0**. o01 restarts with damping 0, 1e-3 and 1e-2 show no consistent gain either.
- **Per-level restarts from identical warm starts** (`s0b_{e01,o01}_restart.py`; gtol 1e-3 against the zero reference). Each cell is steps (rejected) / CG / final f:
  - e01 L0: GN 60(cap)/149/9.0992 vs **exact 15/54/9.0992**.
  - e01 L1: 44/310/4.0045 vs **21(1)/91/4.0045**.
  - e01 L2: 60(cap)/584/1.8050 vs **23(3)/203/1.7999**.
  - e01 L3: 60(cap)/888/**0.8354** vs 24(4)/284/0.8367.
  - e01 wall time is about equal (2/9/24/48 s vs 2/9/24/45 s).
  - o01 L0: 30(cap)/78/8.237 vs 26(3)/77/8.199.
  - o01 L1: 30(cap)/247/**0.987** vs 27(4)/238/1.108.
  - o01 L2: 23/544 vs 22(3)/338, the same f 0.04829, 14 vs 24 s.
- **Exact can converge to a worse local minimum** from zero-padded starts. This is not a defect: the exact-mode endpoints are verified minima. At o01 L1 the dense exact Hessian is symmetric to 2e-16 and PSD (81 zero eigenvalues = the gauge, none negative), at f 1.108 against GN's 0.987. Exact with LS and GN+TR land at the same 1.108. At e01 L3 (Lanczos) the endpoint is PSD too. The o01 exact *ladder* then grew one edge only (L2 dof 138 vs 380), ending at q 0.131 vs GN's 0.048.
- **Global phase.** The TR starts at Δ0 = the Cauchy length (2.3e-3 at o01 L1) and doubles per boundary step, so about 13 steps are radius-limited where GN+Armijo takes full steps (relevant to **O-2**). On the small-residual b01 toy, exact takes 27/26/15 steps against GN's 20/17/10, at 2.4× the wall time. Exact with LS takes 26/21/18 and GN+TR 20/18/12, so the extra steps come from the curvature term far from the tail, not from the TR.
- **e01 ladder end to end** (MAX_NEWTON 30). Exact takes 15/21/26/30 steps (L3 at the cap, 6 of 30 trials rejected) against GN's 30/30/30/30. Train misfit 0.285 vs 0.287, test q 0.165 vs 0.178, fit time 91 s vs 39 s.
- **Cost** (`s0b_hvp_cost.py`, numpy, W=208). An exact hvp costs 3.0×, 1.9× and 3.5× a GN hvp at dof 153, 425 and 1001; an exact model build costs 2.5–2.7× a GN one. CG per step: exact 4–12, GN 7–15; the tail's last two steps took 64–162 (risk J.1).
- **Not yet supported** (`runs/exact_unsupported.log`). A shared ragged manifold raises `NotImplementedError` ("…tied lift lands in slice S7"). `uniform` and `uniform_jax` raise `ValueError` ("UniformProbeDerivativesKind does not implement corewise_hessian()"). So SHARING=input runs, every `backend='uniform*'` level (b01's uniform arms, the e01/o01 BACKEND knob) and n01 are GN-only. Their GN numbers above are the S7/S8 targets.

## 5. Possible T3Toolbox defect and correctness findings
- **FLAG (minor, safe-mode false positive).** `MANIFOLD.norm` / `inner` raise on a near-cancelling difference of two gauged tangents. `tv_gauge_residual` is relative to the *operand's* norm. With both terms gauged to 1e-16, `H_fd - hessian(p)` (agreement 5e-8) measures 1.3e-9 at t=1e-5, above atol 1e-9, and 1.9e-8 at t=3e-6. This is not specific to exact mode; it hits any FD-verification pattern.
  - Repro: `scripts/s0b_gauge_residual.py`, run from `runs/e01_gn100`.
  - Workaround: `corewise_norm` or `safety.unsafe()`.
  - Fix: document it, or scale the check by the operands.
- **No correctness defect found.**
  - `hessian(p)` matches the FD of the Riemannian gradient along the retraction (two levels, four directions each), and `quadratic(p)` matches d²f(R(tp)).
  - Gauge residuals of `hessian`, `gn_hessian` and `curvature_hessian` are ≤ 1e-16, and the dense exact Hessian is symmetric to 2e-16.
  - Unsupported configurations fail up front with clear messages.

## 6. What S14 must change
1. `t3polynomial/fitting.py:353,364,513,525`: `stats['newton']` → `['newton_steps']`. This **blocks** every ladder script (e01, o01, d05, s05, b01, n01, x02).
2. `fitting.py:195-197`: add `g0_reference='warm_start'` on the `g0norm_newton=None` opt-out. Otherwise the opt-out silently runs with the zero reference (P3).
3. `tests/test_fitting_selection.py:86,91` → `newton_steps`, plus `:89` and a `g0_reference` label assert (K.2).
4. `fitting.py:86-112,189-194,338-341,498-501`: the K.1 deletion is optional and shifts each stop by the ratios in §1(a).
5. `scripts/x07:154`: pass `gtol_rel` to the control and add a HESS flag. `scripts/x06`: an exact column using `corewise_norm`. The diffs are in `scripts/`.
6. `d05:232` and `b01:278`: count LS rows explicitly (cosmetic). `d08:113`: `topt._setup` → `least_squares_problem` (not run).
7. Policy, as data for S13: keep GN as the continuation default and use exact for the tail and polish, where it is superlinear (6 vs 30+ steps). From zero-padded starts exact can pick a worse basin and is radius-limited (§4).
