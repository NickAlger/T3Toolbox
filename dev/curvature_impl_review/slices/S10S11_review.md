# S10 + S11 review (independent) -- plain probe / apply / entries two-holes, ragged + uniform
Base 7acabc5 (main now a006d88, dev/-only commits); tree UNCOMMITTED (16 files + my whitespace fixes). Scripts: review/s10s11/.
**Verdict: APPROVE-WITH-FIXES** (the fixes are cosmetic, already applied; no blocking finding).

## Independent checks (mine; none reuse the implementer's oracles)
- **Math** (`oracle.py`): own dense multilinear two-hole -- f = <A, Phi(theta)>, A the dense back-projection, (H th)_k =
  sum_{j!=k} df/dX_k with X_j <- th_j; one einsum per (k, j), NO FD, no library kernel. probe/apply/entries x d=1, d=1 raw
  tails (2,3), d=3, d=3 raw tails, d=4, Tucker slack, TT slack; K=(2,3), C=(2,): max 1.25e-15 over 6 seeds.
- `check2.py`: Gram asym (K=4) <= 1.0e-15. **Containment at the two-hole level**: entries(c, idx) == apply(c, onehot(idx))
  **bitwise**; apply(c, w) == probe(z~_i = alpha_i c w_i, sum alpha = 1) <= 1.2e-15. Kind op at a MANIFOLD frame (model
  sweep reused; probe omega^2 r, per-mode row) vs the oracle <= 5.1e-16. Plain probe == probe_derivatives order 0 <= 3.6e-16.
- **T6**: the two-hole is the Hessian of a scalar, so fully symmetric (no "symmetric part" caveat); the exact hvp is
  Pi(...)Pi + H_R, symmetric in the coordinate metric. The tests assert both, correctly; shared ragged hvp asym <= 1e-15.
- **Kernels vs C.5 / I-33** (read): all recurrences/maps in the GN forms -- tau~dot = two `_sigma_step` per step (K
  placement checked term by term), dx~i dot = two `compute_deta`, mu/nu/eta dot = `compute_sigma/tau/deta(Gdot,G,G,..)`,
  nu re-swept through G, no corewise eta / dx~i; apply/entries: `compute_sigma_hat` through G, `compute_tau(Gdot,G,G,xi,
  xi_dot,sigma_hat)`, `compute_deta` for dxi_hat dot. No trs, closure-free bodies, no manual W/K/C reorder. Factorings
  derived: probe mu.(xi(x)s~ + deta~(x)nu) + tau~.(xi(x)nu) -> 4 finals; apply mu_dot.E + mu.E_dot -> 2. 12 HISTORICAL
  rows present; the sharding test is AST-driven (no table) and passes.
- **Sweep reuse** (`spy_sweeps.py`): per exact hvp, 0 frame-sweep precomputes and no xi/mu recompute; probe +1
  `compute_mu` (the specified nu re-sweep), apply/entries +1 `compute_sigma_hat` (through G). Ragged and uniform.
- **Mutants** (`mutate.py`, tree copies): 9/9 RED -- theta_hat mask removed, `self.masks` for the frame's, no output
  re-mask, drop tau~.E2dot, drop dx~i's mu half, nu from the manifold sweep, drop sigma_hat_dot, (Uniform)ProbeKind unweighted.
- **Bit-identity vs 7acabc5** (`bitident.py`, git archive, separate processes): 194 arrays, 0 differ -- GN grad/hvp/4-step
  newton_cg of the 3 kinds, manifold+corewise, ragged+uniform; probe_derivatives EXACT hvp/curvature_hvp/quadratic +
  exact newton_cg, ragged and uniform, both geometries (covers the `_frame_theta` refactor).
- **T17** (`uniform_corewise.py`, public uniform problem): forced padding, 1e3 garbage in every padded slot incl. the
  first/last TT boundary-bond slots: exact hvp + curvature_hvp bitwise invariant (UNIFORM_COREWISE + UNIFORM_MANIFOLD);
  corewise per element vs ragged <= 2.1e-16; padding exactly 0 against ragged-derived extents.
- **Ragged SHARED** (no direct test in the slice; `shared_hsweep.py`): hvp vs d/dt P_x grad_sh f(R_sh(tv)) is clean O(h^2)
  down to 1e-9; GN / the per-mode lift miss by 1e-2..1. `use_jit` exact newton_cg (MANIFOLD, UNIFORM, shared) == eager to f32.

## Rulings on the 5 judgment calls
1. **Factored TT assemblies: ACCEPT.** Algebra verified; pinned to unfactored at 1e-14 and to my oracle at 1e-15.
2. **No `_trs` twins: ACCEPT.** No order axis, no binomial structure; T5's twin clause vacuous. The FD leg is robust: the
   test's exact Richardson logic over 40 seeds x 3 kinds x d=1,3,4 maxes at 1.04e-14 (bar 1e-12; `fd_robust.py`).
3. **T11 vacuous for apply/entries: ACCEPT.** No weight field exists, and §H's S11 row lists no T11. Probe's per-mode
   omega^2 r is checked independently (check2) and by the unweighted mutants.
4. **chunk_size='auto' -> None: ACCEPT, no regression.** `_resolve_chunk_size` (optimizers.py:164) gives None for every
   kind but probe_derivatives in BOTH modes; the plain kinds have no chunk field and GN never chunks them.
5. **Shared uniform helper refactor: ACCEPT.** Bit-identical (above).
T7's sigma_min guard is an ASSERT on fixed per-case seeds, never a skip: acceptable. It trips on ~4% of random draws
(11/300 measured), so a reseed of the ~27 cases would likely fail one; S12/S13 could draw until sigma_min > 1e-3.

## F27 / E2
- **F27 reproduced independently** (`f27_check.py`, c29b's seeded restarts, warm ref, p=1.0): library exact+TR == the
  harness in every row. A: 18/1/117 (1e-8), 17/1/92 (1e-3). B: 4/1/12, `objective_floor` (the harness says
  "unconverged"). C: 11/1/53, 10/1/33. warm/zero = 1.9e-2, 1.6e-14, 3.1e-1, matching F27.
- **Why E2's 13/69/0 vs GN 54 do not reproduce: `check_restart_tr.py` never seeds the legacy `np.random` stream** that
  `TuckerTensorTrain.randn` draws the target and level-1 start from (it seeds only `default_rng(7)`, :173). So every run
  is a new problem, at ANY commit: two runs today give level-1 g0 2.18e5 / 8.71e4, two at 920d776 give 1.04e5 / 6.86e4
  (`e2_level1.py`). This is not drift and not a regression; the handoff's "on the 920d776 restart" is imprecise. tr2,
  tr3, c29b and c29c do seed. S0d's zero reference also moves the level-1 stop, so the implementer's
  `g0_reference='warm_start'` re-run is the right control.
- The `st['newton']` KeyError in check_restart_tr*.py / c29b / c29c is the S0d rename (3e5bc92): expected; S13 re-pins.

## Findings (none blocking)
- P3 FIXED: comment-column staircases (signature_style.md "don't use the staircase") split into blank-line groups:
  apply.py:336,338,347,367,369; entries.py:278; ut3_sampling.py:406,451,472,474. Whitespace only.
- P3 FIXED: tests/test_optimizers_frontend.py:749-752 had three blank lines before `_plain_fit_problem`; now two.
- P3 open: the T7 guard's fragility to a reseed (above); no ragged-shared plain-kind test (covered by composition and my check).

## Plan-text amendments for S13
- F27: "reproduced by the library exact newton_cg (S11) == the harness, row for row", with the counts above.
- E2: mark as ONE unseeded draw (irreproducible by construction); cite the seeded tr2 / c29b rows instead.
- DEFERRED row 28 (F22/F28/F29's apply numbers "historical until S11"): these can now be re-measured with the library.
- naming_conventions.md: the `*_tangent` dotted-twin suffix and `t3_<kind>_corewise_hessian[_from_sweep]` / `ut3_*`.
## Items for S12
- The same five-step wrapper + `_frame_theta`; delete the `_UniformKind` stub and `_UNIFORM_HESSIAN_SLICE`; remove T23's
  window cases; flip fitting.py:819/873 ("not implemented ... yet") and the CHANGELOG / LocalModel text; add one
  ragged-shared subcase per kind.

## Gates (mine; post-fix unless noted)
- 13 targeted files, -n 2, pre-fix: 456 passed, 2 skipped / 29893 sub. test_dispatch: 54 passed / 43 sub.
- 222 module doctests; getting_started.rst + every docs/*.md and docs/contributor/*.md page OK; sphinx -W: 0 warnings.
- Floor leg (py3.11 / numpy 1.26.4 / jax 0.4.30): 197 passed, 2 skipped / 1225 sub; docs/chunking.md doctest OK.
- Full suite in two halves: backend + dispatch 386 passed, 2 skipped / 2615 sub; the rest 752 passed, 1 skipped / 43048
  sub (1 warning, the pre-existing adam one). `git diff --check` clean; no dev/ or CLAUDE.md edits; no new import edge.
- Slow tier, SERIAL: 56 passed / 13087 sub, 3 pre-existing warnings (adam, jax f64, S9 n4).
