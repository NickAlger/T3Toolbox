# Whitepaper scope — the T3 algorithms & implementation reference

**Status:** working scope doc (2026-07-09). A living scaffold we annotate group-by-group as we decide
what goes in the paper. Not a plan of record yet.

## Purpose & framing

> **This is a DIFFERENT paper from T4S — do not conflate them.** The **T4S paper** (`t4s.pdf`;
> arXiv:2603.21141, *"Tucker Tensor Train Taylor Series"*, Alger–Christierson–Chen–Ghattas 2026) is the
> existing **research preprint** — a historical algorithm reference. **This** document scopes a *separate,
> not-yet-written* **software / algorithms reference paper for the toolbox** (ACM TOMS target): a reusable
> **library reference**, not a research contribution. Some content was *cut from* T4S and finds its home
> here (e.g. `docs/symmetric_probe_derivatives.tex`) — a relationship, not an identity. Throughout this
> doc, "the paper" / "the whitepaper" = the **toolbox** paper unless it explicitly says T4S.

A companion whitepaper laying out, in a clean mathematical framework, the algorithms and hard-won
implementation details for **Tucker tensor trains (T3)** — the "meat and potatoes" operations that
today are scattered across the tensor-train / hierarchical-Tucker literature and must be re-derived
for T3, plus the design decisions and correctness insights we found the hard way. Explicitly **a
useful reference, not an originality claim**: the value is consolidation + T3-specific rigor.

**Target venue:** ACM TOMS (algorithm + software paper; no theoretical-novelty gate; direct
precedent = Kressner & Tobler's htucker toolbox, TOMS 2014). Zenodo release-DOI for the citable
artifact. (JOSS is out until the repo has ~6 mo public open-dev history + external adoption — see the
chat thread that produced this doc.)

**Scope of the sweep:** the *validated frontend* surface — every public method/property across
`tucker_tensor_train.py`, `basis_variations_format.py`, `manifold.py`, `corewise.py`, `fitting.py`,
`optimizers.py`, `safety.py`, and the three `uniform_*` files. **Excluded** (not validated / deferred):
`OLD_orthogonalization.py`, `weighted_tucker_tensor_train.py`.

**Reading convention below:** each group splits **Algorithm content** (candidate numbered-algorithm
boxes / theorems) from **API/plumbing** (mentioned, not paper-central).

---

## The 11 groups

### Group 1 — Data structures & their invariants
*The `TuckerTensorTrain` two-network object; and the **distinguished** frame/variation format — reusing
the pure-TT format for frames/variations is mathematically wrong unless ranks are minimal.*
- **Algorithm content:** `TuckerTensorTrain` (`.data=(tucker_cores, tt_cores)`); `T3Basis` (`U/O/P/Q`)
  + `T3Variations` (`V/H`) as a distinct format + `check_bv_pair`; `T3Tangent=(basis,variations)`;
  structural derivation `structure`/`core_shapes`/`variation_shapes`/`minimal_ranks`/`manifold_dim`/
  `tangent_space_dimension`; the always-error structural contracts (`validate`/`__post_init__`).
- **API/plumbing:** `randn/zeros/ones/unit`, `from_canonical`, `from/to_tensor_train`, `to/from_vector`,
  `save/load`, `to_jax/to_numpy/copy`, `random_orthogonal(_like)`, `size/data_size`, `__repr__`, `to_dense`.

### Group 2 — Core T3 linear algebra ("meat & potatoes")
*Lift TT arithmetic to the two-network T3, with the rank-growth laws stated once.*
- **Algorithm content:** `__add__`/`__sub__`/`__neg__` (ranks add); `__mul__` (Hadamard — ranks
  multiply `n_x·n_y`, `r_x·r_y`); **`t3m`** (truncated Hadamard, three interchangeable algorithms
  `form_then_round`/`inplace_fused`/`swap` — a real implementation contribution); `inner`/`norm`;
  `sum`/`sum_stack` (genuine tensor sum, ranks grow ×S)/`sum_stack_corewise`; structural surgery
  `segment`/`concatenate`/`squash`/`reverse`/`resize`.

### Group 3 — Orthogonalization & orthogonal representations
*Gauge-fix the two-network T3; generalize TT left/right-orthogonalization to four directions; produce
the frame/variation representation (Alg. 11).*
- **Algorithm content:** per-core `down_svd_tucker_core`, `{left,right,up}_svd_tt_core`,
  `orthogonalize_relative_to_{tucker,tt}_core`; sweeps `down_orthogonalize_tucker_cores`,
  `up_orthogonalize_tt_cores`, `{left,right}_orthogonalize_tt_cores`; **`t3_orthogonal_representations`**
  (Alg. 11; note the code/paper sweep-order divergence); `T3Basis.orthogonalize`.
- **Checkers (→ Group 10):** `is_{left,right}_orthogonal`, `is_orthogonal`, `orthogonality_residual`,
  `is_consistent`.

### Group 4 — T3-SVD & rank management
*The T3-SVD, truncation, lossless minimal-rank reduction, rank continuation — with the two-sided
correctness test (→ Group 11).*
- **Algorithm content:** `t3svd`/`t3svd_dense`; `rank_adjustment_sweep` (lossless minimal-rank
  reduction — **separate** from t3svd, which does not minimize); `continuation_ranks`+`resize` (the
  §5.4.1 rank-continuation outer loop, driven by edge condition numbers); rank concepts
  `tucker_ranks`/`tt_ranks`/`ranks`/`minimal_ranks`/`has_minimal_ranks`/`has_numerically_minimal_ranks`.

### Group 5 — The manifold: tangent-space geometry
*The fixed-rank T3 manifold; **two** metrics (HS `MANIFOLD` vs Euclidean-coordinate `COREWISE`);
projection/gauge/retraction/transport; tangent↔ambient maps. **Contains the canonical-random-tangent
theorem — see Addendum A.***
- **Algorithm content:** tangent algebra `T3Tangent.{__add__,__sub__,__neg__,__mul__,sum_tangents,
  normalized,corewise_inner,corewise_norm}` + same-tangent-space guard; `bv_to_t3` +
  `T3Tangent.to_t3`/`to_dense` (embed in ambient); **`ManifoldGeometry`**: `project`/`project_oblique`/
  `project_ambient`/`inner`/`norm`(HS)/`retract`/`transport`/`base`; **`CorewiseGeometry`**:
  `project`(=id)/`inner`/`norm`/`retract`(additive); gauge `gauge_residual`/`is_gauged`/`Π`; the
  `corewise.py` parameter-space vector algebra (`corewise_{add,sub,scale,dot,stack_dot,norm,...}`).

### Group 6 — Sampling operations + jets + transposes
*The unified `probe ⊃ apply ⊃ entries` family, its symmetric Riemannian derivatives (𝒥/𝒥ᵀ), the
**three** transpose families (ambient/corewise/tangent), the derivative (jet) versions, and the
memory-vs-compute adjoint-state choice.*
- **Algorithm content (on T3):** `entries`/`apply`/`probe`; `*_ambient_transpose`; `*_corewise_transpose`;
  `{probe,apply,entries}_derivatives` + `*_corewise_derivatives_transpose`.
- **Algorithm content (on T3Tangent, 𝒥/𝒥ᵀ):** `probe`/`apply`/`entries` + `*_transpose` +
  `*_derivatives` + `*_derivatives_transpose`.

### Group 7 — Fitting & optimization
*Geometry-generic Gauss–Newton on the manifold; least-squares from any sampling op **and its
derivatives**; the four optimizers; the probing-fit recipe.*
- **Algorithm content:** `GaussNewtonModel` (`objective_value`, `gradient=Π𝒥ᵀr`, `jacobian`,
  `gn_quadratic`, `gn_hessian`, `evaluate`); builders `{apply,entries,probe}_model` +
  `*_derivatives_model`; optimizers `gradient_descent`/`mc_sgd`/`adam`/`newton_cg`.

### Group 8 — The stacking layer (batching)
*Mostly design exposition: three meanings of "stack", the `C`/`W`/`K` blocks, the base-inner `W+K+C`
convention and **why**, heterogeneous stacks, the two batch machineries (leading-`'...'` vs
grouped-block contractions). The base-inner convention + grouped-block contractions are the
contribution.*
- **Dedicated methods:** `stack`/`unstack` (every class), `stack_tangents`/`unstack_tangents`,
  `stack_basis`/`unstack_basis`, `sum_stack`/`sum_tangents`, `{,base_,tangent_}stack_shape`;
  backend `stacking.{stack,unstack,tree_zip,apply_func_to_leaf_subtrees}`.

### Group 9 — The uniform layer (SIMD/vectorized representation) — a RATIONALE section
*Mirrors Groups 1–7 on stacked supercores; the paper content is the **design rationale**, not a
re-listing. (Confirmed framing: uniform is a faster ragged layer, `to_uniform → op → to_ragged ==
op_ragged`.)*
- **Objects (mirror the core):** `UniformTuckerTensorTrain`, `UT3Basis`/`UT3Variations`, `UT3Tangent`,
  `UNIFORM_MANIFOLD`/`UNIFORM_COREWISE`; masks `UT3Masks`/`UT3BasisMasks`/`UT3VariationsMasks`;
  converters `from/to_t3`, `from/to_t3basis`, `from/to_t3tangent`, `from/to_ut3`, `apply_masks`,
  `squash_tails`.
- **The design decisions (each a subsection):** (1) **mode-index-first** supercore layout
  `(d,)+stack_shape+(…)` — locality (cores touched one-at-a-time for `lax.scan`) + polymorphism;
  (2) **boolean rank masks** not integer ranks — closed under add=concat / multiply=Kronecker with no
  data movement; the **determinantal-variety** view; (3) *why* masks exist — enforce variable rank by
  zeroing variation padding (rank control); (4) **SVD-prefix orthogonalization** (masks stay a
  deterministic prefix); (5) value-hashed masks as static pytree aux → **jit-compile-once** across base
  changes; (6) masks-on-host-`np` / supercores-on-device-`xnp` (the jit story); (7) the **equivalence
  contract** (correctness *and* test strategy).

### Group 10 — Numerical contract & safety mode — a HARD-WON contribution
*The **structural-vs-numerical** dividing line, precondition-vs-caveat, and **precisely which requirement
each operation actually needs** (minimal-rank / orthogonal / gauged / same-frame). One of the strongest
distinctive contributions.*
- **Machinery:** `safety.py` — safe/unsafe mode, `checks_active`, `require`, `effective_rtol` (two
  tolerances: float64 numpy vs float32 jax), `is_tracing`, `frames_equal` ("same tangent space" is
  **numerical**, not identity).
- **The catalog itself:** orthogonal-frame ops (project/retract/transport/project_ambient); gauged ops
  (HS inner/norm); same-frame ops (±/stack/matvec); minimal-rank (a `retract` **caveat** only —
  reconfirmed in Addendum A: not a precondition for the Gaussian-tangent fact either); precondition-free
  (T3 / corewise / probing).
- **Diagnostic checkers:** `is_orthogonal`, `is_gauged`, `has_{,numerically_}minimal_ranks`,
  `is_consistent`, `frames_equal`, `orthogonality_residual`, `gauge_residual`.

### Group 11 — Testing methodology — a HARD-WON contribution
*The verification strategies (a methodology section, not methods).*
- The **two-sided T3-SVD test**; dense-ground-truth verification (rebuild + hand einsum, residual
  ~1e-12); **mask-strictness + garbage-padding robustness** for uniform (the phantom-rank /
  doubled-boundary bug class); **adjoint-identity** tests for transposes (`⟨Ax,y⟩=⟨x,Aᵀy⟩`); jit-clean
  dispatch tests (stray `np.*` on a tracer raises); backend-agnostic numerics (numpy-only numerical +
  separate jax dispatch). *The covariance = projector test (Addendum A) is a clean new exemplar.*

---

## Cross-cutting threads (candidate standalone sections)

### Design-principle thread: "regular representation + explicit reduction"
Recurring choice: prefer a **uniform / regular (redundant) representation** with an **explicit
canonicalizer**, over baking a canonical-but-special-cased form into the type. Instances:
- **Arbitrary boundary TT ranks** (don't force `r0=rd=1`) so every TT core is identically shaped
  `(r_i, n_i, r_{i+1})` → one uniform code path, no boundary special-casing; `squash`/`squash_tails`
  canonicalizes to `r0=rd=1` losslessly, only when a caller needs it. Load-bearing for `segment`/
  `concatenate` (a segment's boundary bonds are another train's interior bonds) and *essential* for the
  uniform supercore (a shared core shape is required — an `r0=rd=1` exception would poison the SIMD
  layout). **(Nick's item (2).)**
- **Boolean rank masks** in the uniform layer (regular structure, no integer-rank special cases).
- **Doubled-rank tangent embedding** (`to_t3`/`retract`) — regular embedding, then truncate.
- **Non-minimal-rank tolerance** — ops stay correct on non-minimal input; `rank_adjustment_sweep` /
  `t3svd` / gauge `Π` are the explicit reductions.

Placement: a short **Design principles** section; boundary-rank is its cleanest worked example.

---

## Addendum A — The canonical random tangent (Nick's item (1))

**Proposition.** At a point `X` of the fixed-rank T3 manifold with an **orthonormal, gauged** frame,
the linear map `δ` from gauged variation cores (coordinate/Frobenius inner product) to `T_XM` (HS inner
product) is an **isometry**. Hence i.i.d. `N(0,1)` variation cores + gauge projection `Π` produce the
**standard isotropic Gaussian on `T_XM`** = orthogonal projection onto `T_XM` of an ambient standard
normal. In particular `E‖v‖²_HS = dim T_XM`.

*Why worth stating:* naively "random cores" pushes forward to an **anisotropic** ambient Gaussian
shaped by the parametrization's Jacobian; it is only because an orthonormal+gauged frame makes that
Jacobian an **isometry** that the naive random tangent is the coordinate-free / canonical one. This
licenses randomized methods on the manifold (sketching, dimension/trace estimation, random directions)
and is the same isometry that lets `MANIFOLD.inner` be the cheap coordinate dot.

**Conditions — settled empirically (scratchpad scripts `gaussian_tangent_check.py`,
`gaussian_tangent_nonminimal.py`; move to the research repo):**
- **Orthogonality — necessary & sufficient.** It *is* the isometry.
- **Minimal rank — NOT required.** Verified on a genuinely non-minimal orthonormal frame
  (`has_minimal_ranks=False`; Tucker rank 3 exceeding its TT-bond product 2): the gauge projection
  absorbs the redundancy, so draws span **exactly** the true tangent dimension (36), covariance =
  projector = covariance of the projected ambient normal, isometry exact to `1e-14`. Consistent with the
  `has_minimal_ranks` catalog note.

Numbers (minimal frame `shape=(4,5,6)`, ranks `(2,3,2)/(1,2,2,1)`, dim 30, M=40000):
`‖cov(randn)−P‖/‖P‖ = 2.9e-2` (= `√((dim+1)/M)`, pure MC noise); `‖cov(proj. ambient normal)−P‖ =
2.8e-2`; `‖cov(randn)−cov(proj)‖ = 4.1e-2`; `E‖v‖²/dim = 1.0002`; isometry `6.6e-13`.

**Follow-up (code, small):** `MANIFOLD.randn`'s docstring says "orthogonal, minimal-rank" — **overstated**;
tighten to "orthogonal." `COREWISE.randn` is the deliberate contrast: raw/ungauged, **not** HS-isotropic.

Placement: **Group 5** (a "canonical random tangent" result), with spurs to Group 1 (`MANIFOLD.randn`
gauged vs `COREWISE.randn` raw), Group 10 (precondition-vs-caveat exemplar), Group 11 (covariance =
projector test pattern).

---

## docs/ curation — what is whitepaper material

Full read of all 21 `docs/*.md` (5 parallel readers, 2026-07-09). Tiers: **CORE** (paper body) ·
**RATIONALE** (design/discussion section) · **METHODOLOGY** (methods/appendix) · **PROCESS** (our code
conventions / jax plumbing — exclude, implementation footnote at most).

### Paper-body & discussion (keep)

| Doc | Tier | Group | The transferable nugget |
|---|---|---|---|
| `symmetric_probe_derivatives.tex` | **CORE** | 6 (→5/8) | **Net-new math cut from T4S**, self-contained (Def/Lemma/Thm **with proofs** + algorithm box + dense oracle): symmetric derivative (jet) probing from one jet-Leibniz rule (binomial convolution); transpose = binomial correlation; arity = internal-edge count. Strongest single contribution. *(missed by the `.md`-only sweep)* |
| `ttm_t3m_ht_note.tex` | **CORE** | 2 (→3/4) | **Sole full home** for the T3M Hadamard product: KR-at-leaves / Kronecker-inside, three algorithms (form/fused/swap) + cost table, the leaf-frame-coupling theorem + oversample-then-round. Reframe §1–4 "idea capture"→"shipped". *(missed by the `.md`-only sweep)* |
| `entries_apply_probe.md` | CORE | 6 | `probe ⊃ apply ⊃ entries` via 3 exact identities; monotone cost. Cleanest, most paper-ready doc. |
| `transposes.md` | CORE | 6 | Three-adjoint taxonomy (ambient/corewise/tangent) as a **projection hierarchy**; return ambient adjoint as CP to defer the `|W|²` blowup; transposes are `Ω(N)` (N lives in the output). |
| `t3svd_minimal_ranks.md` | CORE | 4 | Minimal-rank inequalities = single-edge matricization ranks (T3 is a **tree** ⇒ clean bipartition); a hard rank cap in a sweep orphans upstream ranks. Paper-ready. |
| `rank_continuation.md` | CORE | 4/7 | Edge-condition-number growth: equalize `κᵢ=σ₁/σ_last` across edges vs grow uniformly (~3.3× DOF savings). |
| `uniform_ranks_and_varieties.md` | CORE | 9/4/5 | **Standout / near-theorem:** variable-rank stacked T3 = batch in the bounded-rank **determinantal variety** (closure of fixed-rank manifolds, stratified by them); mask = stratum label; rank varies across a base stack, shape locked. |
| `numerical_contract_catalog.md` | RATIONALE | 10 | **Precondition-vs-caveat** line + per-op requirement table (SF/ORTH/GAUGE/minimal). Headline: minimal rank is a precondition for *nothing*. De-tangle from safe-mode machinery. |
| `uniform_masks_vs_ranks.md` | RATIONALE | 9/2/4 | Store rank as **boolean projector masks**, not integer prefixes — the unique form closed under ⊕=concat / ⊗=Kronecker with no data movement. |
| `uniform_rank_masks_rationale.md` | RATIONALE | 9/5/7 | Masks are a **functional rank constraint**: inflate-equivalence holds for a *fixed* tangent, but a fresh gradient at a rank-deficient frame points into rank-growing directions — masks zero exactly those. |
| `uniform_svd_prefix_orthogonalization.md` | RATIONALE | 3 | Prefix-rank contract is correct **only under SVD** orthogonalization; QR's Q is ordered by input columns not importance → a fixed prefix can drop real content. |
| `uniform_equivalence_contract.md` | RATIONALE | 9/11 | Define the vectorized layer by a **round-trip equivalence** to the reference layer (real parts only) — simultaneously correctness spec + test oracle (ragged twin, not dense). |
| `t3svd_design_rationale.md` | RATIONALE | 4/3 | **Gauge-parity:** truncation and lossless minimization each flip the orthogonality gauge → can't fuse into one gauge-predictable sweep; hence separate ops. |
| `fitting_and_optimization.md` | RATIONALE | 7 | **Geometry-generic Gauss–Newton:** `J=𝒥∘Π`, `grad=Π𝒥ᵀr`; swap the single gauge `Π` → Riemannian (PD H) or raw-core Euclidean (gauge-singular H). Rest is plumbing. |
| `batching_and_stacking.md` | RATIONALE | 8 | Batch taxonomy (`C`/`K`/`W` on different parts); **broadcast (aligned) vs multi-index contraction (independent on disjoint operands)**; transpose-of-a-broadcast-is-a-sum. **Reframe base-inner** as "base is outer-shared over its fiber" (right-aligned-broadcast is a 1-line artifact). ~30–40% paper-worthy. |
| `testing_strategy.md` | METHODOLOGY | 11 | **Phantom-rank blind spot** (dense tests can't see too-permissive masks), the **tautology trap**, cured by exact-mask assertions + garbage-padding (clean==dirty). |
| `ambient_derivative_transpose_note.md` | METH. (appendix) | 6 | Derivative ambient adjoint has **intrinsically exponential CP rank** (`2^d` apply, `d·2^{d-1}` probe) — a property of the tensor, not the encoding; why derivative-fitting routes through tangent/corewise. |

### Process-only (exclude; implementation footnote at most)
`uniform_backend_jit_recipe.md` (jit closure recipe), `uniform_supercore_layout.md` (d-leading =
lax.scan-driven), `uniform_pytree_composition.md` (pytree children/aux + value-hashing) — collapse the
masks-are-numpy/jit story into one note. `probing_section6_notes.md` (code↔paper Rosetta worklog —
harvest the recurrences + `J=𝒥∘Π` / §6.3-substitution from the paper's own §6). `signature_style.md`,
`doctest_style.md` (code conventions — salvage two sentences: "shape *is* the type in array code";
"verify gauge-defined outputs by their relationship, not their values").

### Actionable findings
- **The two-sided T3-SVD write-up EXISTS (proofs and all) — it was misfiled, not unwritten.**
  *(Correction to the subagent curation, which checked only `docs/` and reported it missing.)* It lives at
  **`dev/archive/t3svd_verification.md`** — moved there by the knowledge-reorg (`055c5744`) and its
  references left dangling. CORE-tier (Group 4 + Group 11), paper-grade: **accuracy** (generalized
  Oseledets Thm 2.2 — TT unfoldings *and* Tucker matricizations add in quadrature, evaluated at the
  chosen ranks) with proof; **parsimony** (`r̂_k ≤ #{σ⁽ᵏ⁾_j ≥ τ}`) with the projection Lemma
  (`σ_j(PAQ) ≤ σ_j(A)`) and full proof; plus the accuracy-at-`r̂`-vs-rank-at-`ρ` consistency pitfall.
  Referenced by live code docstrings (`tucker_tensor_train.py:3991,4318`) as the authority. By the routing
  rule (proofs/rationale = user-facing → `docs/`) it belongs in `docs/`, not archive. **Action: restore to
  `docs/t3svd_verification.md` and fix the dead links** (2 docs + 2 code docstrings) — no authoring needed.
- **CLAUDE.md TODO is stale:** the flagged `entries_apply_probe.md` §4-table / derivative-dimension
  staleness was **already fixed** (commit `7c8d1818`, 2026-06-18). Strike "refresh
  docs/entries_apply_probe.md" from CLAUDE.md's doc-pass TODO.
- **Cosmetic (pre-paper):** unify `base`/`basis` between `transposes.md` and `entries_apply_probe.md` §4;
  strip hard-coded residual magnitudes from working docs.

### Merge plan for Group 9 (uniform rationale)
One arc: *what it represents* (determinantal variety / strata) → *how rank is stored* (boolean masks,
closed under ⊕/⊗) → *why masks can't be dropped* (functional rank constraint) → *what "correct" means*
(round-trip contract). SVD-prefix orthogonalization → Group 3. jit/pytree/host-numpy → one impl note.

---

## Archive scan results (dev/archive/, 2026-07-09)

Full read of all 22 archive files + the 2 `docs/*.tex` (6 parallel readers + a `.tex` reader).

**Misfiling — only ONE.** `t3svd_verification.md` is the sole genuinely misfiled reference doc →
**restore to `docs/`**. The contested `t3m_plan.md` / `t3m_swap_plan.md` are **correctly-archived build
plans** whose durable theory **and cost table** already live in `docs/ttm_t3m_ht_note.tex` — so the fix
there is to **repoint the code docstrings to the `.tex`**, not restore. Everything else
(plans/handoffs/session-logs) is correctly archived.

**The real defect — a systematic reorg dead-link cluster (~30 refs in ~15 files).** Live code/doc
references still say `docs/<name>` for files the reorg moved to `dev/archive/`. Fix per target:
- `t3svd_verification.md` (`docs/t3svd_minimal_ranks.md:100`, `docs/doctest_style.md:75`,
  `tucker_tensor_train.py:3991,4318`) → **auto-fixed by the restore** (they already say `docs/…`).
- `t3m_plan.md` / `t3m_swap_plan.md` (`tucker_tensor_train.py:1963,1980`, `backend/t3_linalg.py:288,313,493`)
  → **repoint to `docs/ttm_t3m_ht_note.tex`** (theory + cost-table home).
- `symmetric_probe_derivatives.tex`'s own dead ref to `derivatives_mirror_plan.md` → **drop**.
- all others (history/plan pointers) → **repoint `docs/<name>` → `dev/archive/<name>`**:
  `safe_unsafe_mode_plan` (`numerical_contract_catalog.md:3`, `batching_and_stacking.md:368`,
  `fitting_and_optimization.md:274`, `safety.py:6`, `manifold.py:1433`); `apply_entries_handoff`
  (`entries_apply_probe.md:231`, `batching_and_stacking.md:492`); `optimizers_plan`
  (`optimizers.py:12`, `backend/optimizers.py:10`, `tests/backend/test_optimizers.py:1`,
  `fitting_and_optimization.md:7,243,247,269`); `derivative_fitting_plan` (`fitting.py:277`,
  `backend/fitting.py:227`, `fitting_and_optimization.md:7,269`); `geometry_refactor_plan`
  (`fitting.py:30`, `tests/test_fitting.py:12`, `fitting_and_optimization.md:8,270`); `uniform_port_plan`
  (`uniform_equivalence_contract.md:112`, `ut3_orthogonalization.py:63`,
  `uniform_tucker_tensor_train.py:605,609`); `uniform_slice_handoff` (`uniform_pytree_composition.md:85`);
  `derivatives_mirror_plan` (`ambient_derivative_transpose_note.md:12`, `entries_apply_probe.md:193`,
  `manifold.py:758`).
  *(Note: my initial grep's "`fitting_plan.md` referenced" was a substring false-positive of
  `derivative_fitting_plan.md`; plain `fitting_plan.md` has no live refs.)*

**Extract-worthy nuggets (durable content NOT in live docs):**
- `geometry_refactor_plan.md` §2/§3/§5.1 — the submersion picture `π:P→M`, the **gauge-singular
  Gauss–Newton-Hessian derivation** (`𝒥(gauge)=0 ⟹ gauge∈ker(𝒥ᵀ𝒥)`, `Π` the cure), Manopt/Pymanopt/
  Geomstats vs TensorLy/t3f positioning, the metric-on-tangent argument → Group 5/7 "why".
- The **contraction dense-tangent-projection formal proof** (`t3m-swap-planning_2026-06-13.md` L942–1000;
  results are in the `tangent_operations.py` docstring, the cross-term-cancellation proof is not) → Group 5.
- The **bidiagonal-`trs` perf optimization + 32×/78× benchmarks** (`derivatives_mirror_plan.md:205–212`)
  → dev perf-tracking, not `docs/`.

`method_porting_plan.md` self-labels "delete once implemented" → deletion candidate (Nick's call).

---

## Open decisions / next steps
- [x] Uniform = rationale section (Group 9). *(Nick agreed.)*
- [x] Tighten `MANIFOLD.randn` docstring. *(Done 2026-07-09.)*
- [x] Curate `docs/` incl. the 2 `.tex` for whitepaper material. *(Done — tables above.)*
- [x] Archive scan (dev/archive/). *(Done — one misfile, a dead-link cluster, 3 extract-nuggets.)*
- [ ] **Live-file cleanup (needs sign-off):** (1) `git mv` restore `t3svd_verification.md` → `docs/`;
  (2) repoint the ~30 dead `docs/<name>` links (t3m → `.tex`; the rest → `dev/archive/`); (3) drop the
  dead `derivatives_mirror_plan.md` ref in the `.tex`; (4) strike the stale `entries_apply_probe.md`
  line in CLAUDE.md's doc-pass TODO.
- [ ] Extract the 3 nuggets (geometry §2/§3/§5.1; contraction-projection proof; bidiagonal-`trs` perf)
  — during the paper write, not now.
- [ ] Walk the 11 groups one-by-one: numbered-algorithm-box vs prose vs omit.
- [ ] Decide "Design principles" (regular representation + explicit reduction): standalone or distributed.
- [ ] Cosmetic doc cleanups (base/basis; strip captured residual numbers) before paper use.
- [ ] Move the two verification scripts to the research repo.
- [ ] `method_porting_plan.md` deletion (your call).
- [ ] Paper skeleton / section order (after the walkthrough).
