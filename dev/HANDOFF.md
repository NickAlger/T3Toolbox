# T3Toolbox — current handoff

_Updated 2026-08-19 (shared Tucker factors: design settled, implementation started — see Active
threads). Prior history: `dev/archive/handoff_2026-07-12_1.0_complete.md` (the 1.0 completion:
R1–R7, the R4 doc pass, the docs user/dev split S1–S5, the cordon/ETT/literature morning)._

## Where we are — 2026.0.0 SHIPPED ✅

**2026.0.0 is live on PyPI (2026-07-13) — `pip install t3toolbox`** (+ the `[jax]` extra).
REL-1 → REL-4 all done; the full release history is archived at
`dev/archive/release_plan_2026-07-13.md`. The ship (REL-4): CHANGELOG `[Unreleased]` →
`[2026.0.0]`; install recipe flipped to `pip install t3toolbox`; gates green (593 tests /
40,215 subtests, docs `-W`, doctests, wheel + `twine check` + fresh-venv numpy-only smoke); commit
`21f7b6fb`, tag `v2026.0.0` → the trusted-publishing workflow (approved on the `pypi` environment)
→ published. **Verified against real PyPI**: numpy-only and `[jax]` fresh-venv installs, quickstart
smoke, and the full `getting_started.rst` doctest (64/64) against the installed package.

Loose ends: the **GitHub Release** for `v2026.0.0` — Nick to create via the web UI (notes = the
CHANGELOG `[2026.0.0]` section); **Zenodo DOI** still deferred (Nick, later); the dead `gh-pages`
branch can be deleted (optional).

## Standing open question

- **None.** The `contractions.py` architecture question (Nick's 2026-07-15 unease) was **RESOLVED
  2026-07-17 by the grouped-einsum interpreter** and archived with its resolution banner:
  `dev/archive/OPEN_QUESTION_contractions_architecture_RESOLVED_2026-07-17.md`.

## Active threads

- **Shared Tucker factors (SF-T3) — DESIGN SETTLED (2026-08-19), implementation in progress.**
  Optimize over T3s whose Tucker factors are tied within user-specified mode groups (the SF-ETT
  of Molozhavenko & Rakhuba 2026, generalized to arbitrary partitions). **The spec is
  `dev/shared_factors_handoff.md` (v3)**, with the math in `dev/shared_t3_math.tex` (+pdf) —
  both updated 2026-08-19 after two design-review rounds (all decisions AGREED; the v3 header
  lists what changed vs v2: the corewise post-pass is the per-group MEAN not the Gram solve;
  `S_i` is recomputed from the frame by RE-SWEEP (bit-identical to construction; companion holds
  a thin SVD of stacked `S^T`, never a Cholesky/Gram — float32-measured); grouped `t3svd` is the
  paper-faithful two-phase with an all-singleton dispatch anchor; the retraction embedding is
  built tied (`[U_g | Udot]` + center cores) — the v2 mean-re-share was unsound; `GeometryOps`
  gains a `precompute` slot (breaking, sanctioned); zero-padded restarts escape via the untied
  TT channel — full shared rank is a diagnostic, never a precondition). Commit sequence in v3
  §8 (13 slices: ragged 1–8, uniform 9–11, docs 12, symmetric jetted-probes example 13).
  The three reference papers are in the research repo (`tensor/references/`); copies in `dev/`
  stay untracked (third-party PDFs).

- **The grouped-einsum interpreter `contractions.contract` — SLICE 1 BUILT (2026-07-17; committed,
  NOT pushed).** Nick's resolution of the standing `contractions.py` architecture question: ONE
  public interpreter, `contract('WCo,WCa->Cao', *ops, len_W=...)` — standard einsum format,
  UPPERCASE = a group of zero-or-more axes, lowercase = single axes. Solves each group's axis count
  from the operand ndims (exact linear system over Fractions; identifiability = a rank condition on
  the SUBSCRIPTS, never instance shapes, so a call site either always needs a `len_<G>` or never
  does; co-traveling groups need only their run total — the old "don't demand a split you don't
  need" rule, now mechanical), expands groups into fresh letters, runs ONE einsum on the operands
  **as given — no reshape ever**, which strengthens the shardability contract from
  "leading sub-axis" to **ANY sub-axis of any group** (compiler-verified:
  `TestInterpreterAnyAxisSharding`, 24 cases, 0 all-gathers) and makes block fusion inexpressible.
  numpy path reuses `_pairwise_path` computed on the GROUPED string (identical paths to the named
  fns, size-independent, cached); jax gets the single expanded einsum. Perf parity measured
  (≤1.06x; the 4-operand trs combine is ~0.8x = FASTER, empty groups no longer leave size-1 axes).
  Evidence: `tests/test_contractions_interpreter.py` — the definitional loop oracle (map the
  single-axis contraction over group indices), a differential sweep vs ALL ~100 named contractions
  over the empty/single/multi block matrix, and the supplement meta-test (the rank analysis
  reproduces every hand-written `n_probe`/`n_frame` decision); jit dispatch in `test_dispatch.py`;
  green on the compat floor env (numpy 1.22 / jax 0.4.30).
  **END STATE (Nick, 2026-07-17): make `contract` the public grouped-contraction surface and DELETE
  the 104 named contractions** (breaking OK — release is days old, known users don't call them).
  **Slice 2 — DONE (2026-07-17): all 108 library call sites flipped to `contract(...)`** —
  `probing.py` (49), `sampling_derivatives.py` (47), `apply.py` (11), `entries.py` (1); the
  mechanical rewrite mapped the trailing `n_probe`/`n_frame` args to `len_W=`/`len_C=` keywords,
  and the ~10 function-object *selection* sites (`f = contractions.X if sum_over_probes else
  contractions.Y` in the two variation assemblies + the apply/entries adjoint tail) now select the
  SUBSCRIPT STRING instead. Zero named-contraction references remain outside `contractions.py` and
  its tests. Gates: jet/probe suites, full `test_dispatch` (jit through every flipped hot path),
  module doctests, compat-floor env, full suite.
  **Slice 4 — DONE (2026-07-17): the 104 named contractions are DELETED and `contract` is the
  surface.** Phase A (pre-deletion, separate commit): the standing evidence — the loop oracle over
  the AST-scanned vocabulary (123 strings; the scan caught the 19 slice-3 strings that had no
  coverage) + the frozen identifiability table (`HISTORICAL`, seeded from the signatures and
  cross-verified before deletion) + call-site consistency + split invariance + the any-axis sharding
  sweep over the whole vocabulary (678 compiled cases, 0 all-gathers); also surfaced+fixed a latent
  `\dots` invalid-escape in a `tucker_tensor_train` docstring. Phase B: `contractions.py` truncated
  to the interpreter (~3200 → ~450 lines, `__all__ = ['contract']`, module docstring + SHARDING
  block rewritten); deleted `tests/backend/test_contractions.py`, `test_contractions_unfused.py`,
  `test_contractions_lean_jets.py`, the migration cross-checks, and the named-fn sharding classes;
  `test_dispatch` entries converted to `contract` strings; CHANGELOG Added + BREAKING-Removed
  entries. Phase C (docs): `batching_and_stacking.md` (§4b rewritten for the interpreter; the
  "shard the leading axis" rule DELETED — any axis now; legend/§8/§10 updated),
  `batching_internals.md` (interpreter-era extension rule; the two failure eras recorded; the
  "generating the subscript per call — REJECTED" record flipped with its new information),
  `signature_style.md` + `reading_signatures.md` references, CLAUDE.md (machinery-2 + code-style
  bullets + the open-question pointer). **The standing architecture question is RESOLVED and
  archived**: `dev/archive/OPEN_QUESTION_contractions_architecture_RESOLVED_2026-07-17.md`.
  **Docs — DONE (2026-07-17 evening): the framework write-up.** `docs/grouped_contractions.md`
  (user tier: the problem, the dialect, usage with run-and-pasted examples, the `len_<G>`
  supplement story + its three guarantees, what-you-get, motivation-in-brief) and
  `docs/contributor/contractions_internals.md` (the pipeline; the exact linear system + the
  rank/kernel identifiability story + the greedy-supplement argument; the static-not-instance
  decision; co-travel runs incl. the output-inclusion counterexample and the no-cycles argument;
  letter assignment with real expanded strings; why the pairwise path is computed on the grouped
  string; validation-for-free; the evidence structure + the one HISTORICAL-table duty). Wired into
  both toctrees; cross-refs from `batching_and_stacking.md` §4 and the `contract` docstring.
  **Slice 3 — DONE (2026-07-17): the lean-jet inline einsums are `contract` call sites.** The four
  scan-step bodies (`compute_eta_jets`, `compute_deta_jets`, `_sigma_banded_step`,
  `_adj_tilde_step`) now pass W/K/C UNFLATTENED to `contract(...)` (`len_C` supplied where the W|C
  split is unpinned) — the pre-flatten reshapes at the top of each step are gone, so the any-axis
  sharding property now reaches the jet hot paths; ZERO `xnp.einsum` remain in
  `sampling_derivatives.py`. Verified: lean == trs == the definition (float64-exact small,
  float32-noise at W=3000); the uniform-jit **memory win survives the rewrite** (eta measured
  0.03 vs 0.84 GiB XLA temp = 28x at d=4, order 5, W=3000, r=48); jet/probe suites, compat floor,
  full suite. The resume note is superseded → `dev/archive/contraction_cleanup_resume_2026-07-16.md`.
  Remaining slice: (4) delete the named fns + their oracle tests
  (the differential test's reference then retires with them — keep the loop oracle + sharding
  sweeps), rewrite the module docstring/SHARDING block for the interpreter era, update
  `docs/batching_and_stacking.md` §4 (drop "shard the leading axis" → any axis) +
  `docs/contributor/batching_internals.md` (the "generate the subscript per call" REJECTED entry is
  now the design — record the new information: call-site strings stay literal/greppable) + CLAUDE.md
  + `dev/OPEN_QUESTION_contractions_architecture.md` (resolution). Stale doc ref found on the way:
  `batching_internals.md` + the resume note cite `tests/test_contractions_naming.py`, which never
  existed (no git history).

- **Jet recurrence/convolution forms — the zippering direction, WIRED IN as the standard (2026-07-16;
  committed, NOT pushed).** The concrete realization of the `trs`-as-sparse-convolution idea in the
  standing architecture question, now the default implementation. **The lean forms own the canonical
  names; the dense forms are the `_trs` reference twins** (Nick's design, 2026-07-16): `compute_mu_jets`
  (recurrence/scan/chunk, wired into every call site) vs `compute_mu_jets_trs` (dense binomial-tensor
  einsum, numerically equal to tolerance, kept for reference + tiny/memory-abundant regimes + the
  `test_jet_recurrence.py` oracle). Both public. The non-fused `compute_mu_jets_banded` was dropped
  (dominated by the fused form). Testing: `test_probe_derivatives` (vs dense/adjoint) now anchors the
  lean path; `test_jet_recurrence` keeps `standard == trs`, so `_trs` stays anchored transitively.
  Research record + benchmarks: separate repo (`nicks_research_experiments/t3_jet_experiments/`,
  esp. `findings.md`). **Names below are the pre-rename `_banded_fused`/`_scanned` labels (history).**
  - **Done so far:** the **plain chain** — `compute_mu_jets_banded_fused` (affine two-term recurrence,
    folded into one GEMM) and `compute_eta_jets_scanned` (full-convolution order-scan); and the **forward
    tangent Jacobian** — `compute_sigma_jets_banded` / `compute_tau_jets_banded` (three affine
    pushthroughs) and `compute_deta_jets_scanned` (three full-convolution combines). All correct to
    ~1e-12 across K-stacked shapes; swapping all five in passes the full probe + uniform suite (temporary
    monkeypatch, 137 tests / 582 subtests).
  - **The two load-bearing lessons** (both cost a wrong prediction first, both measured): (1) a memory
    win from a scan needs a **real `lax.scan`/`lax.map`, not a Python loop** — unrolled loops let XLA keep
    everything co-resident (measured ~1.2× vs the intended 14–28×), and this **only works on the uniform
    path** (ragged `xmap`/`xscan` are Python loops; uniform dispatches to the real jax primitives, and
    `lax.map` is *sequential*, not `vmap`). (2) deta's higher memory than eta is the **K stack, linear in
    K** — *not* the three terms; reordering the accumulation is a no-op (XLA schedules it identically,
    V1=V2=V3 byte-identical).
  - **Wins (uniform jit, r=128 W=32000):** mu fused — numpy ~2.9×, jax 1.03–1.10×, memory constant vs
    trs-linear. eta scanned — **14–28× less XLA temp** (~4.5 GB const vs 64–128 GB). deta scanned —
    **3.3–6.9× at K=4 / 28× at K=1** (~9 GB at K=1). Timing (W=3000): deta scanned **1.5–1.8× faster at
    K=1**, 0.6–0.7× at K=4 (leaner but the scan serializes order); sigma ~parity.
  - **Also DONE (2026-07-16 cont.):** the **transpose/tilde jets** (`compute_{sigma,tau}_tilde_jets_scanned`
    — the banded prop + scanned src *mix*; a nested scan, jit-clean; 2–3.4× less memory) and **both
    variation assemblies** (`assemble_{tt,tucker}_variation_jets_scanned` — W-chunked with a
    **reducer seam**: add if the chunked axis is summed / concat if kept, so C-batching is a parameter
    not a rewrite; tt-core 168→8 GB / 21×). The math is recorded in
    `docs/symmetric_probe_derivatives.tex` §recurrence. The whole forward Jacobian + its transpose are
    now in recurrence/scan/chunk form. All experimental, module-private, verified vs the trs originals
    in `tests/test_jet_recurrence.py`; polymorphic (numpy/jax × ragged/uniform correct; the memory win
    is jax+uniform). Research record: `nicks_research_experiments/t3_jet_experiments/findings.md`.
  - **Copy-free W-chunking — DONE (2026-07-16).** Nick pinpointed the `moveaxis` (used to put the chunk
    axis leading for `xmap`) as a transpose that copies the whole operand — which made chunked
    assemble_tucker *worse* than dense at N>>n. Replaced pad+reshape+moveaxis with `_wchunked_reduce`:
    slice each chunk in place (`lax.scan`/`dynamic_slice` on jax, eager loop on numpy), reduce add/concat.
    assemble_tt now 168→2.63 GB (**64×**, was 21×); N>>n fixed (assemble_tucker 51.9→1.6 GB at N=10000).
  - **Contraction cleanup — SUPERSEDED by the grouped-einsum interpreter (2026-07-17, Nick's
    decision).** The named-function extraction (phase 1 mu done, `6e9a55d8`) stops here; the
    remaining inline einsums in `sampling_derivatives.py` instead became `contract(...)` call
    sites (interpreter slice 3, done). The note is archived:
    `dev/archive/contraction_cleanup_resume_2026-07-16.md`.
  - **Rename + wire-in — DONE (2026-07-16).** Lean forms took the canonical names, dense → `_trs`
    (single-pass whole-word rename + surgical seam re-point; the chunked assemblies still call
    `assemble_*_trs` per chunk *intentionally* — chunking runs the dense contraction on each W-slice).
    Added canonical `compute_nu_jets`; `__all__` carries both name sets. Full suite green;
    `test_jet_recurrence` gained a `nu` equivalence check.
  - **`chunk_size` wiring + estimator — DONE (2026-07-16; committed, NOT pushed).** Three slices, all
    green (docs clean under `-W`):
    - **A** (`d84054a1`): `chunk_size` (`Optional[int]`, default `100`, `None`=dense) threaded through the
      whole probe-transpose chain (both assemblies → the backend transpose fns + corewise → `utv_` twins →
      `T3Tangent`/`UT3Tangent.probe_derivatives_transpose`). New user doc `docs/chunking.md`.
    - **B** (`83dd4c70`): `estimate_chunk_size` / `max_chunk_size_within` — eager, shape-param estimators.
      `per_row` is **measured** (dual `ShapeDtypeStruct` lowerings + `memory_analysis().temp_size`, max over
      TT + Tucker, cached) not analytic (~20× off); balanced default = assembly ≈ jet floor; `n_shards`
      sizes the per-device shard. `tests/test_chunk_size_estimator.py`.
    - **C** (this commit): `chunk_size='auto'` threaded through the 4 optimizers + `_setup` + the two kind
      builders; `_resolve_chunk_size` calls the estimator for a uniform `probe_derivatives` fit (shapes off
      `x0`, minibatch size for `mc_sgd`/`adam`), `None` for ragged/non-probe. `Problem` stores the kind
      directly so the closed-over `chunk_size` survives jit.
    - **Design decisions** (Nick, 2026-07-16): no `'auto'` at the low levels — plain int + safe fixed
      default; the estimator is a separate eager (outside-jit) function so it sidesteps GSPMD's trace-time
      sharding-blindness; **balanced** (assembly ≈ edge-var memory) is the default policy, `max_chunk_size_within`
      the opt-in absolute-budget one. Full record: `docs/chunking.md`.
  - **Deferred:** (1) `probe_derivatives_model` / `_uniform_model` (the direct model-builder frontend) still
    uses the fixed `100` default — threading `chunk_size` there needs a static field on the pytree
    `UniformGaussNewtonModel` (its `kind` cached-property rebuilds from fields); the optimizers (the main
    path) are fully wired. (2) **Sharded fitting** — the `shard_map` boundary + `psum` in the optimizer
    (slice D), **postponed 2026-07-16** (Nick wants to weigh scope); full design + the genuine decisions
    are in **`dev/sharded_fitting_plan.md`**. The estimator already takes `n_shards` and
    `docs/chunking.md` ships the manual recipe.
    Also still open from the jet thread: **C-chunking** (the reducer seam supports it), uniform
    mask-strict/garbage-robust tests, extracting the inline per-step grouped einsums into `contractions.py`.

- **`contractions.py` unfusing — DONE (2026-07-15, `dae52839` + `f65b341d`; committed, NOT pushed).**
  No named block is fused with another any more: 14 sites, 112/112 bit-identical, full suite 712 passed /
  40,532 subtests. The rule is now **flatten only what einsum forces you to** — a *shared* block (on
  several operands) must be a letter; a *passive* block (one operand, rides to the output) rides as
  `'...'`. Rule + evidence: `docs/contributor/batching_internals.md`. Plans archived (**and wrong in
  places — read their banners**).
  - **Two things worth not re-deriving.** (1) "Every named block gets its own einsum letter" is
    **impossible** — the `W|K` and `K|C` splits are not recoverable from the operands, so the fusion was
    forced by the signature, not laziness. (2) Option B *beat* the plan: `W`-minor sharding on a
    multi-axis block, which the plan accepted as inherent, is now free (3 all-gathers → 0). Only a
    *shared* block's flatten survives.
  - **The shardability contract — BUILT & DONE (2026-07-16).** *Every grouped index must be shardable
    over its first sub-axis.* `TestShardabilityContract` in `tests/test_contractions_sharding.py`: an
    **automatic sweep** over every public contraction (enumerated from the module's functions, **not**
    `__all__`), 280 (function, block) pairs, ~37s, holds everywhere with nothing exempted. The existing
    hand-written tests are kept — the sweep proves the rule holds, they record what broke it and what it
    cost. Rule + the two non-obvious details: `docs/contributor/batching_internals.md`. Plan archived.
    - **The feasibility probe answered the design question decisively:** 78/78 names parsed (101/101 now
      `__all__` is fixed), **200/200** agreeing with the independently hand-written shape comments, 78/78
      constructing → calling → matching the predicted output shape. Far past the "50 clean + 10
      exceptions → automatic" bar. One exception: `trs` (a family tag; letters from the body einsum;
      carries no grouped index, so never sharded).
    - **Still open from the watch-list: the latent `_pairwise_path`/`'...'` trap.** It builds `set(term)`
      per operand, so an ellipsis's `.` counts as a shared index and skews the greedy pairing. Harmless
      today (every `'...'` rewrite is 2-operand, and `_grouped_einsum` bypasses `_pairwise_path` at ≤2
      operands) — **it would bite the first 3+-operand `'...'` contraction anyone writes.** No guard was
      added; the contract did not require one.

- **Weighted layer (edge weights) — COMPLETE & SHIPPED, ragged + uniform (2026-07-15). Thread closed;
  committed, NOT pushed.** Diagonal weights on the internal edges, as a lightweight data format +
  `absorb` into cores. `T3Weights`/`UT3Weights` weight a **tensor**; `T3FrameWeights`/`UT3FrameWeights`
  are a **metric on a tangent's coordinates** (Grasedyck-Kramer). All four carry `absorb` /
  `weighted_norm`/`weighted_inner` / `reciprocal`/`sqrt` / `concatenate`/`kronecker`, plus
  `from_*svd` / `from_*weights` and ragged<->uniform conversions; frontend free functions are
  family-prefixed (`t3_`/`ut3_`/`fv_`/`ufv_absorb_weights`) and the whole surface is exported at the
  package root. Tests: `tests/test_weighted.py`, `tests/test_uniform_weighted.py`.
  - **Durable knowledge is now in the rendered docs** (build notes archived): user usage ->
    `docs/weighting.md`; design records -> **`docs/contributor/weighted_internals.md`** (the two-classes
    reasoning, the metric-on-variations change, the frame-like stack model + the two-level check, the
    uniform mirror's three traps, placement notes, and what's deferred); the testing lesson ->
    `docs/contributor/testing_strategy.md` ("Exercising a mask check is not testing it"); the naming rule
    -> `docs/naming_conventions.md`; the `ut3_norm`/`ut3_inner` gap -> `docs/contributor/deferred_and_rejected.md`.
    Build records: `dev/archive/weighted_layer_design.md` (ragged), `dev/archive/uniform_weighting_design.md`
    (uniform; §8 = its decision log). **The rendered docs are authoritative where the archives disagree.**
  - **Side-fixes that landed with it** (all committed): `common.prefix_mask` extracted (~8 duplicates
    across 4 modules); `require_concrete_masks` moved to `common` (it is the mask-representation contract,
    not a `ut3_` family member); a stale CI `--ignore` for the S4-deleted parked module removed; and
    `docs/batching_and_stacking.md`'s "weighted is parked" line corrected.
  - **The guard gap it surfaced is now FIXED** (2026-07-15): the frame/variation side never called
    `require_concrete_masks` at all — `ufv_apply_frame_masks`, `ufv_apply_variations_masks`,
    `ut3frame_to_t3frame`, `ut3variations_to_t3variations` all took traced masks and died with jax's
    cryptic `TracerArrayConversionError` instead of the actionable message. All four guard now, and
    `tests/test_dispatch.py::TestTracedMaskGuard` pins all ten chokepoints (plain / frame / variations /
    weights) — the plain-layer guard had no test at all before, only a doctest.
  - **Deferred (reachable from the primitives):** weighted `+`/`-`/scale/`⊙` as operations + an optional
    thin container; the Grasedyck-Kramer `SingularValueRegularizer` — **both layers now have every
    primitive it needs**, so it is the natural next consumer.

- **Regularization framework — COMPLETE & SHIPPED (S1–S5, 2026-07-14).** Identity (Tikhonov)
  regularization on the fitting objective `min ½‖ω⊙(S(X)−y)‖² + ρ(X)`, composing with every optimizer /
  kind / geometry / representation: `regularizer=IdentityRegularizer(λ)` on any optimizer, ragged + uniform,
  backend-homed (`backend/regularization.py`). Plus the **`obj = misfit + reg` display split** (`verbose=` +
  `stats['history']`/`['diagnostics']`; also fixed a latent `(unwt …)` mislabel bug). Commits: S1–S4 + the
  split **pushed** (`d9700056`, `54d752d2`); **S5 docs + this cleanup uncommitted → next: commit.**
  - **Durable knowledge now in the rendered docs** (design note archived): user usage →
    `fitting_and_optimization.md` §4.9 (+ the §5 "rank is the primary regularizer" note); contributor design
    decisions + the `v_X`/`value` derivations + uniform mask-safety + stochastic scaling + the deferred items
    (Grasedyck–Kramer seam, base-point-as-tangent public op, `already_left_orthogonal` amortization) →
    `docs/contributor/fitting_internals.md` ("Regularization" + "What's deferred"). Full build record:
    `dev/archive/regularization_design.md`.
  - Worked example: `examples/fit_hilbert_regularized.py` (Option A — Hilbert denoising, fit ~0.30, λ by
    validation). Small open optimization: masked-last-core `point_norm_sq` (uniform, cheaper + ragged-consistent).

- **`use_jit=True` auto-convert (silent-drop bug) — DONE + PUSHED to `main` (2026-07-14, `3ce68fea`).**
  Fixes: `use_jit=True` with numpy inputs used to silently run eager (the flag looked accepted but did
  nothing — meaningless "jit" benchmarks). Root cause was the `_maybe_jit` / `xwhile` guard requiring all
  inputs to already be jax. **Key technical finding:** it's *not* a jax limitation — a jit tracer *is* a
  `jnp.ndarray`, so our type-inference dispatch routes to `jnp` during tracing; forcing jit on numpy
  inputs runs correctly and (with x64) matches eager bit-for-bit. The guard's real job was preventing a
  silent **float32** downgrade (jax's default), not a crash. Nick's call: **auto-convert** — requesting
  jit is opting into jax-world precision. Fix (`_prepare_jit_inputs` in `backend/optimizers.py`): when
  `use_jit=True`, `jnp.asarray` `x0` + `problem.sample`/`data` (masks/weight left alone), so both jit
  mechanisms engage; returns a **jax-backed** result (float32 unless x64); **raises** if jax absent
  (the one un-honorable case). New `common.tree_to_jax`. Verified end-to-end ragged + uniform (masks stay
  numpy) + no-jax raise. Tests: backend `test_use_jit_requires_jax` + jax-backed assertions in
  `test_newton_cg_recovers_to_high_accuracy`; frontend `test_newton_cg_use_jit_returns_jax`. Docs:
  `fitting_and_optimization.md` §4.5, frontend module docstring, CLAUDE.md shipped-surface. Two refinements
  (Nick, same session): (a) **`use_jit` promoted to an explicit frontend kwarg** on `newton_cg`/`mc_sgd`/
  `adam` (was implicit via `**kwargs` — justified singling-out: it's the only kwarg that changes the
  return type/precision); (b) a **3-part precision doctest** in `optimizers.newton_cg` — numpy float64
  (~1e-10) vs jit float32 (~1e-7) vs jit + `jax_enable_x64` (float64 restored, ~1e-10). Verifiable via the
  dtype contrast + straddle-1e-8 booleans (raw floats aren't bit-reproducible). **x64 leak avoided**:
  `jax.experimental.enable_x64` is gone in jax 0.10, so the doctest uses the global `jax.config.update`
  but captures dtype/err as plain Python values *while x64 is on*, then restores `x64=False` BEFORE
  asserting — a green run guarantees no leak into the single-process `--doctest-modules` sweep (verified:
  full sweep 169 passed; also green on the jax 0.4.30 compat-floor env). **Next: commit.**

- **Newton-CG warm-start reference overrides — DONE + PUSHED to `main` (2026-07-14, `649edb62`).**
  `newton_cg` now takes three optional kwargs so a warm-start continuation loop isn't hurt by a
  misleadingly-small initial `‖g0‖`: `g0norm_newton` / `g0norm_cg` override the reference norm the
  Newton stop (`‖g‖ ≤ gtol_rel·‖g0‖`) and the CG forcing term (`η = min(0.5, (‖g‖/‖g0‖)**power)`) are
  relative to (chained fallback: `g0norm_newton` also feeds CG unless `g0norm_cg` is given; `g0norm_cg`
  alone touches only CG; neither → the computed initial norm as before), and `cg_forcing_power`
  (default `0.5`) trades CG iters per Newton step for fewer Newton steps (raise it when the manifold
  retraction is expensive vs a Hessian-apply). **Backend-only change** (`backend/optimizers.py`); the
  frontend forwards via `**kwargs`, uniform inherits it for free. `NewtonInfo.g0norm` now reports the
  effective Newton reference. Tests: `test_optimizers.py::test_newton_cg_g0norm_and_forcing_overrides`
  (four fallback cases + power direction, reconstructing each ref from the reported η) +
  `test_optimizers_frontend.py::test_newton_cg_g0norm_kwargs_forward`. Docs: `fitting_and_optimization.md`
  §5, cross-ref in `rank_continuation.md`, CLAUDE.md shipped-surface. Suites green (57 opt/display/dispatch
  + doctests). **Next: commit** (message per §Workflow); nothing else open on this thread.

- **Newton-CG diagnostic display — DONE + MERGED to `main` (2026-07-13).**
  `optimizers.newton_cg(..., verbose=True)` prints a per-iteration block (objective/gradient, CG stats,
  line search, ρ, wall time) + a per-`(mode, order)` relative-error table (`‖r_ij‖/‖y_ij‖`), with an
  optional `val_sample`/`val_data` validation column; records also returned in `stats['diagnostics']`.
  **Backend-owned** (anti-drift): a raw-`.data` user gets the identical display via
  `backend.optimizer_display.make_newton_display` + `newton_cg(callback=...)`. Works on ragged **and**
  uniform (the `block_sumsq` reduction is dual-path; validation auto-packed). Table layout follows the
  kind's axes (plain probe: mode cols; probe_derivatives: mode rows × order cols, train|val cells).
  Example `examples/fit_probe_display.py` shows both layouts. Full suite green (619 tests). Design record
  + slice list: `dev/newton_display_plan.md`; merged fast-forward (commits `53aab004`..`545653ce`).
  Thread closed.

- **Per-mode residual weighting — DONE + MERGED to `main` (2026-07-13).**
  The fitting layer's residual weight `ω` generalized from a per-order vector to an `ω[mode, order]`
  matrix; **per-mode weighting** added to the probe models (probe is the only kind with a per-mode
  axis — apply/entries stay order-only). `probe_model(weight=(d,))`,
  `probe_derivatives_model(weight=(d,order+1))`, topt threads it, uniform mirror is compile-once
  (nested-tuple aux). New example `examples/fit_per_mode_weight_probes.py`; docs §4.6 rewritten.
  Design record + the full slice list: `dev/per_mode_weighting_plan.md`. Full suite green, docs `-W`
  clean; merged fast-forward (commits `02972a86`..`1dcd84ce`). Thread closed.

- **The toolbox reference paper** (independent): scope + curation in `dev/paper_scope.md`.
  Next: walk the groups starting at Group 6 (`docs/symmetric_probe_derivatives.tex` is nearly a
  drop-in chapter). Paper-grade material queued there from the archive sweep: the two-spaces
  geometry picture; the apply/entries sweep-level scatter derivations.

## Backlog (not scheduled)

- **Standalone grouped-contractions library — PARKED (Nick, 2026-07-17).** Splitting `contract`
  into a small independent repo (T3 would VENDOR a single-file copy — no runtime dependency).
  Revisit trigger: the next time a grouped contraction is needed in another project. The einx
  feasibility check (einx.dot covers the headline case, compiles to the same expanded einsum,
  shards clean; residue = co-travel merging + the string-level identifiability contract + zero-dep
  minimalism) and the plan of record live in the research repo:
  `grouped_contractions_vs_einx/findings.md`.
- **Base-point-as-tangent as a public library op** (Nick, 2026-07-14) — representing a base point `X` as a
  gauged tangent `v_X` within its own tangent space is broadly useful; expose it as a first-class op
  (frontend `T3Tangent`/`UT3Tangent` factory + backend helper) with the direct construction (last TT
  variation `= P_last`, else zero; already gauged). Reg's `_manifold_point_tangent` /
  `uniform_manifold_ops`'s closure is the current internal impl to promote/share.
  (Now also in `docs/contributor/fitting_internals.md` "What's deferred"; full context:
  `dev/archive/regularization_design.md` §11b.)
- **Default-path doctest pass** for undocumented public functions (Nick wants this).
- **`core_shapes` (property, strips stack) vs `get_core_shapes` (static, includes stack)**
  inconsistency — verified live 2026-07-12; a code decision for Nick.
- **Zenodo DOI** — Nick, at a later date.
- Delete the dead `gh-pages` branch (optional; Pages deploys from artifacts now).
- Per-test seeding → `pytest -n auto`; trimming `test_dispatch` jit time (deferred niceties).

## Post-1.0 (1.1) threads

- The Goal-1 **`fit(...)` facade** (auto geometry/optimizer/ranks/`x0` + rank continuation —
  "standard user, no fiddling").
- **Weighted-layer revival/redesign** (currently parked + cordoned with warnings).

## Standing constraints

The durable rules live where they belong: project-wide conventions and gotchas in **CLAUDE.md**;
contributor-facing conventions and decision records in the rendered **Contributor guide**
(`docs/contributor/` — naming rules, refactoring methodology, testing strategy, the
deferred/rejected ledger). Two operational ones worth repeating here: the docs build must stay
at **zero warnings** (`sphinx -W` in CI), and doctest outputs are **run-and-pasted, never
hand-written**.
