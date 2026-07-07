# T3Toolbox — current handoff

_Updated 2026-07-07._

## Where we are

**The uniform optimizer layer — the 1.0 centerpiece — is BUILT and tested (backend complete).** The four
optimizers (`gradient_descent` / `mc_sgd` / `adam` / `newton_cg`) now run on the uniform layer, **fully
packed**, **jit-compile once**, and are **robust to non-minimal input**, all verified against the ragged
optimizer. Increment 3b (the uniform tangent + probing + jets) closed 2026-07-01; this session
(2026-07-06 → 07) wired the optimizers/fitting onto it — the whole reason the uniform layer exists
(speed). Branch `main`, direct commits, pushed; full suite green.

**→ The one remaining slice is U7: the frontend surface + docs** (see "Next steps").

## Done this session (2026-07-06 → 07) — optimizers/fitting on the uniform layer

Sliced U1–U6 + U5.6; plan + slicing history in **`dev/uniform_optimizers_plan.md`**. All backend-first
(the reused geometry-generic optimizer bodies in `backend/optimizers.py` are unchanged apart from an
`inner`-seam swap); the new code lives in `backend/uniform_fitting.py`, `backend/ubv_sampling.py`,
`backend/ut3_operations.py`. Tests in `tests/backend/test_uniform_fitting.py` (+ jit in `test_dispatch.py`).

- **U1** — `ubv_corewise_inner` (the masked/stacked raw-tuple twin of `UT3Tangent.corewise_inner`) + a
  `GeometryOps.inner` seam on the optimizer bodies (ragged = `corewise_dot`, byte-identical). This was
  Nick's redirect from an earlier "masking-project" hack — the honest masked-reduction is the right design.
- **U2** — uniform `GeometryOps` factories (`uniform_{manifold,corewise}_ops`): **bare-supercore-pair in/out,
  fixed-rank masks closed over** (Arch-B1: masks are loop-invariant state, only supercores traced). Verified
  == the frontend `UNIFORM_MANIFOLD` / `UNIFORM_COREWISE` `.data` path + mask loop-invariance across points.
- **U3 / U3′** — uniform `SamplingKind` builders (plain apply/entries/probe + the derivative/jet twins): the
  split `precompute → from_sweep` seam in `ubv_sampling` (the sweep carries the mask-once base + packed
  vectors), reusing the ragged layer-agnostic fields via `dataclasses.replace`; geometry-agnostic (variation
  masks derived from the frame). Verified == ragged `SamplingKind` + adjoint identity + garbage-robust.
- **U3.5 — the packedness-mirror pivot** (a design change, agreed with Nick). The user-facing sampling ops
  **infer input packedness and mirror it** — ragged in → ragged out (== ragged), packed in → packed out
  (== `pack`(ragged)); the fitting **split-seam is packed-only**, so the optimizer inner loop keeps probe
  residuals packed (no per-matvec unpack/repack; `d` stays a single scan axis, not a Python list).
  `ut3_operations.{is_packed,pack_if_ragged}`; `sumsq_over_probes` + `_make_order_weight(order_axis=)` made
  packed-aware (ragged path byte-identical). Doc: `docs/uniform_equivalence_contract.md` § vector-I/O mirror.
- **U4** — `uniform_least_squares_problem` (packs the loop-invariant sample+data **once** → the reused
  backend `Problem`/`LocalModel` run fully packed; optimizer state = the bare supercore pair). LocalModel
  objective/gradient/gn_quadratic verified == ragged for every kind × both geometries.
- **U5** — all four optimizers run on uniform (fully packed) + the packed-aware minibatch `take` (`_ptake_*`,
  so mc_sgd/adam keep minibatches packed). gradient_descent matches ragged; newton_cg + mc_sgd track ragged;
  adam descends. (Optimizer tests are deliberately small/short — correctness needs a few iterations, not
  deep convergence — with **tolerance-based** assertions, never bit-exactness.)
- **U6 — the jit path.** The per-step kernel **compiles ONCE** across iterations with changing supercores
  (verified `traces==1`): masks closed over as host-numpy constants, only supercores traced, frame masks
  re-derived inside `local_model` constant-fold. Test: `test_jit_uniform_optimizer_wholestep` in
  `tests/test_dispatch.py`. Reusable GPU benchmark: **`dev/bench_uniform_vs_ragged.py`**.
- **U5.6 — minimal-rank requirement** (a real bug found during U6's benchmark investigation). The uniform
  optimizer crashed on a **non-minimal** base: the retraction truncates to the realizable rank and desyncs
  from the fixed masks → a cryptic mid-loop crash (ragged tolerates non-minimal; uniform cannot, its masks
  are fixed). Fix (enforce the precondition, since we own the optimizer): **`uniform_minimal(x0)`** reduces
  to minimal (`t3svd` → right-to-left `rank_adjustment_sweep`; no-op if already minimal), and
  `uniform_least_squares_problem` validates + raises a clear error pointing to it. Docstring + doctests
  (fail non-minimal → succeed after `uniform_minimal`) + `TestUniformMinimalRank`.

### Two investigation findings (from Nick's benchmark questions) — carry forward
- **Eager timing:** uniform-eager ≈ ragged-eager is NOT a bug. The uniform supercores carry ~2.6× more
  elements (padding) yet pipeline better; the two roughly offset (pipelining slightly wins on CPU). The real
  speedup (scan over `d` + recompile-free step) is a **GPU** story — re-run `dev/bench_uniform_vs_ragged.py`
  on a GPU server (bump `SHAPE`/`TR`/`W`).
- **No hidden unrolled loops in the major ops:** the TT sweeps use `lax.scan` (8 scans), the tucker
  orthogonalizations use a batched SVD, the sampling ops are fully vectorized (all constant in `d`). A small
  **residual ~6 jaxpr-eqns/mode** in orthogonalization+retract remains (not a naive Python loop — subtler;
  linear, not ballooning; amortized since compile-once). A follow-on only if very-high-`d` compile matters.
  Aside: `d ≥ 24` hits einsum's 26-letter label budget (a shared ragged/uniform edge case).

## Next steps

**U7 — the frontend surface + docs (closes optimizers-on-uniform):**
- Extend `t3toolbox/fitting.py` (`apply_model` &c.) and `t3toolbox/optimizers.py` to accept a
  `UniformTuckerTensorTrain` `x0` + the uniform geometries (`UNIFORM_MANIFOLD` / `UNIFORM_COREWISE`), or add
  uniform frontend factory functions. **The frontend optimizer MUST call `uniform_minimal(x0)`
  transparently** so frontend users never meet the minimal-rank requirement (the backend
  `uniform_least_squares_problem` raises a clear error; the frontend should just reduce and proceed).
- Doctests to the reference-module standard (`docs/doctest_style.md`); a worked uniform example alongside
  `examples/fit_hilbert_*`.
- On close: refresh this handoff + sweep `dev/uniform_optimizers_plan.md` into `dev/archive/` (dated).

Then the **release-hygiene roadmap** (R1–R6) below.

## The 1.0 roadmap (mid-level-toolkit scope) — summary
- **R1** packaging correctness (`readme = README.md`; create `CHANGELOG.md`; numpy range).
- **R2** public API surface (curate `__init__.py`) **+ the naming/organization review** (`dev/naming_review.md`:
  backend prefix grammar; `T3Basis→T3Frame` / `bv_→fv_` / `ubv_→ufv_` — its own mechanical, suite-gated pass).
  Also the deferred cosmetic `Sequence`→`Union` hint relaxation.
- **R3** README + quickstart (remove the "DO NOT USE" banner **only at the moment of shipping**).
- **R4** docs build (fix autoapi exclusions + `modules.rst` title; **fold design rationale from `docs/` into
  user-facing Sphinx docs**).
- **R5** test CI (pytest numpy-1.x/2.x matrix + **wire doctests in**); no auto-formatter near the curated style.
- **R6** cleanup — delete `OLD_*` / stray artifacts **only after confirming functionality is preserved**.
- **R7 — DONE this session** (the uniform tangent layer + optimizers/fitting on it), except **U7** (frontend)
  above. Still: document the absent weighted layer; do **not** ship research caveats as user guidance.
- **→ 1.1:** the Goal-1 `fit(...)` facade (auto geometry/optimizer/ranks/`x0` + rank-continuation).

## Don't-trip constraints (the maintainer's standing rules)
- **The uniform optimizer requires a minimal-rank base** — a non-minimal (unrealizable) nominal rank desyncs
  the retraction from the fixed masks and crashes mid-loop. `uniform_least_squares_problem` rejects it with a
  clear error; call `uniform_minimal(x0)` first (the frontend U7 will do this transparently).
- **The packedness-mirror convention** (U3.5): user-facing sampling ops infer & mirror packedness; the
  fitting split-seam is packed-only (the optimizer inner loop stays packed). Don't "normalize" it to a flag.
- **A uniform op needs more than dense-vs-ragged** — also exact output masks + garbage-robustness
  (`docs/testing_strategy.md`). Masks are host numpy (`np`, not `xnp`); supercores are `xnp`.
- Numerical test assertions use a **tolerance** (`np.allclose`/`np.isclose`); exact comparison only for
  structure (shapes, masks, types, counts) — never rely on floating-point bit-exactness across hardware.
- Never delete an `OLD_*` (or anything) until its functionality is **confirmed preserved**.
- "DO NOT USE" banner stays until the literal moment of shipping.
- **No automated tool rewrites the code style** (esp. the shape comments). No `manifold.py` rename.
- Research caveats are not user-facing. Notes are preserved/relocated, never lost.
