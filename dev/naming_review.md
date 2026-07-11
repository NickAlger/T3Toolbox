# Naming / organization review — decisions log

_Started 2026-06-21._

> **§2 EXECUTED 2026-07-10: `basis`/`base` → `frame`.** The `T3Basis→T3Frame` / `UT3Basis→UT3Frame` /
> `.basis→.frame` / `basis_*→frame_*` / `bv_→fv_` / `ubv_→ufv_` rename **and** the `base`-half (frame
> accessors `model.base`→`.frame`, `geometry.base(x)`→`geometry.frame(x)`, and the **C stack → frame stack**:
> `base_stack_shape→frame_stack_shape`, `n_base→n_frame`) shipped in two suite-gated commits. `T3Variations`
> kept. Math "basis" preserved (variation-core standard/orthonormal basis, "basis vectors", the **Tucker
> basis** factor matrices); `base` kept for the plain manifold **point** (`base_point`/`base_masks`/prose)
> and the named **`base-inner`** convention. Decision refinements settled during execution: (a) `fv_`-coarse
> for frame/variation-only ops; (b) singular alignment (`compute_mus→compute_mu`); (c) accept long morphemes;
> (d) the C stack is structurally a batch of *frames* → "frame stack". **Still open:** the cross-class
> method-name sweep (§3, verify-before-merge), the §1 auto-fixes (`probe_t3→t3_probe` etc.), and §4's backend
> module reorg (**un-deferred 2026-07-10** — file-level reorg only, polymorphism triage dropped).
>
> **§1/§3/§4 EXECUTED 2026-07-11** — the full naming pass + backend module reorg shipped (plan,
> token inventory, and execution deltas: `dev/naming_pass_plan.md`; user-facing conventions catalog:
> `docs/naming_conventions.md`). Sampling grouped **by type** (Nick's call — option C); chains got
> the **`tt`** family with the polymorphic `tt_reverse`/`tt_squash_tails` merges; `has_jax` →
> `jax_available`; `t3toolbox/__init__.py` curated (R2). See `dev/HANDOFF.md`.

_The log below records the agreed conventions + open concerns (planning notes; some now executed per the
banner above)._

## Locked decisions

### 1. Backend prefix grammar
- **Grammar:** `[w][u]<family>_` — modifiers stack in order **weighted → uniform → family**;
  `w` valid only on `t3`/`tv`.
- **Families:** `t3_` (T3 tensor), `tv_` (tangent vector), `fv_` (frame–variation pair).
- **Full set:** `t3_/ut3_/wt3_/wut3_` · `tv_/utv_/wtv_/wutv_` · `fv_/ufv_` (no weighted frame/variation).
- **Prefix = primary operand representation; always a prefix, never a suffix.**
- **Unprefixed:** representation-agnostic infra (`common`, `stacking`, `linalg`, `contractions`).
- **Frontend OO layer: no prefix** (the class is the namespace).
- **Auto-fixes:** `tucker_tensor_train_apply`→`t3_apply`, `probe_t3`→`t3_probe`,
  `bv_to_t3(ind,…)`→`fv_to_t3(ind,…)`, `tangent_to_t3`→`tv_to_t3`.
- `fv_to_t3(ind,…)` (indexed *single term*) and `tv_to_t3(…)` (efficient *sum* of all terms) are
  DISTINCT ops on the same `(frame,variation)` data — both kept; the prefix disambiguates.

### 2. `T3Basis → T3Frame` (full propagation)
- **Rationale:** the orthogonal representation is overcomplete/redundant and minimal rank is NOT
  required — so "frame" (admits redundancy; idiomatic in manifold optimization) is correct, while
  "basis" (implies minimality) is not. No paper conflict (T4S avoids naming the frames; it does use
  "variation").
- **Scope:** `T3Basis→T3Frame`, `UT3Basis→UT3Frame`, module `basis_variations_format.py →
  frame_variations_format.py`, prefix `bv_→fv_` (`ubv_→ufv_`), `basis_*`→`frame_*`.
- `T3Variations` **stays** ("variation" is paper-confirmed).
- **Careful execution** (it's everywhere, ~160 refs): scoped, scripted rename in one pass; full
  suite + per-module doctest sweep as the gate.

### 3. Cross-class consistency
- **Rule:** parallel frontend classes (`TuckerTensorTrain` / `UniformTuckerTensorTrain` / the weighted
  twin) expose **identical method names** for the same operation.
- `squash` / `squash_tails` → **`squash_tails`** uniformly (self-documenting). Backend follows the
  prefix grammar: `t3_squash_tails` / `ut3_squash_tails` / `wt3_squash_tails`.
- jax predicate: keep **`contains_jax`** (object/property) + **`tree_contains_jax`** (backend pytree
  walk); **drop `has_jax`**.

### 4. Backend module organization — TARGET shape (executed during the uniform-layer fix, not standalone)
- **Organize the backend as a (representation-family) × (operation-kind) matrix**, making the ragged
  side conform to the template the uniform side already exemplifies (`ut3_{constructors, operations,
  linalg, orthogonalization, svd, sampling, conversions, masking}`).
- **Op-kind columns:** constructors · operations · linalg · orthogonalization · svd · sampling ·
  sampling-derivatives · conversions · masking (uniform-only). **Family rows:** t3 · tv · fv (+ `u`/`w`
  variants). Rep-agnostic infra (`contractions`, `stacking`, dispatch, generic `linalg`, `ranks`) sits
  **outside** the matrix, unprefixed.
- **Sampling cut:** Euclidean t3 sampling → `t3_sampling`; tangent sampling (the bulk of today's
  `probing.py`) → `tv_sampling`; derivatives → `*_sampling_derivatives`. Dissolves the
  `apply.py`/`entries.py`/`probing.py` asymmetry + the kitchen sink.
- **Polymorphism refines the matrix (KEY):** an op that is *polymorphic over ragged/uniform* (dispatches
  internally via `get_backend(is_uniform, …)`) lives **unadorned in the base family** (`t3_*`/`tv_*`);
  there is **no separate `ut3_` twin**. So `u` marks **uniform-*specific*** ops only (masking, supercore
  packing, mask-aware logic) — polymorphism *thins* the `u`-rows (less duplication = the goal). Whether
  a `t3_*` fn accepts uniform input is a documented *capability*, not name-encoded.
- **Therefore the reorg is NOT a standalone naming task** — which cells are polymorphic-base vs
  uniform-specific is decided by a **per-operation polymorphism triage** (*already-poly / make-poly /
  can't-or-shouldn't*), which **is part of fixing the uniform layer**. Adopt the matrix as the *target*;
  do the reorg + triage *together* during the uniform fix.
- **`ut3_sampling` is a triage item:** prior stopgap with a packing bug; intent is to make the t3 probing
  backend polymorphic via existing hooks → likely **eliminate the separate `ut3_sampling`**.
- **RESOLVED — ragged/uniform is *inferred*, not threaded** (a "ragged/uniform dispatch", parallel to
  numpy/jax). `get_backend` is a two-axis inferred dispatch (representation × array-backend), both read
  from inputs at the lowest level: `use_jax = tree_contains_jax(...)`;
  `is_uniform = is_ndarray(up_tucker_cores)` (uniform = one stacked supercore ndarray; ragged = a tuple
  of per-mode arrays); `xnp, xmap, xscan = get_backend(is_uniform, use_jax)`. The hooks already exist in
  `probing.py` — making it polymorphic *completes* that pattern. Once an op dispatches this way it needs
  **no `ut3_` twin**. (Fold this into CLAUDE.md's dispatch convention alongside the numpy/jax rule.)
- **Two `MANIFOLD`/`COREWISE` names:** geometry singletons (frontend `manifold.py`) vs `GeometryOps`
  (backend `optimizers.py`) — same names, different things. Give the backend pair distinct names / its own
  `geometry` module (no-prefix rule is frontend-only). Resolve in the reorg.
- **`common.py` grab-bag:** split rep-agnostic infra (dispatch / io / random) vs keep as catch-all — decide.

## Smaller open items
- **Long-name morphemes** (e.g. `entries_jacobian_derivatives_from_sweep`, 39 chars) — shorten morphemes
  or accept as the price of a regular scheme; mostly falls out of the reorg pass.
- **Plurality:** `compute_xis`/`compute_mus` (plural) vs `compute_mu_jets` (singular) — align.
- **Frame/variation-only ops:** share `fv_` (coarse — recommended) vs split `f_`/`v_`.
- **Cross-class method-name sweep:** run as an execution checklist (align all parallel classes).
