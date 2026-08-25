# T3Toolbox — project guide

A working reference for collaborating on this codebase. Read this first.

## What this is

Pure-Python (NumPy + optional JAX) library for **Tucker tensor trains (T3)** — a Tucker
decomposition whose central core is stored as a tensor train. Nick Alger and Blake Christierson are converting research
code into a standalone package: cleaning up, documenting, restructuring for usability, adding
features. **The goal is a general-purpose library for other people and other use cases** — not to
reproduce or serve any particular application. The research it came from (the T4S paper, below) is
**essentially done** (an arXiv preprint, with only minor revisions still in flight); treat it as
historical reference for the algorithms, never as the design target. (The "WORK IN PROGRESS" banner came off 2026-07-12; 2026.0.0 shipped to PyPI 2026-07-13 — `pip install t3toolbox`.)

- Repo: `github.com/NickAlger/T3Toolbox` (renamed from `TuckerTensorTrainTools`; that rename left
  stale references we've mostly fixed). Branch `main`, direct commits.
- Env: pure-Python (numpy + optional jax), **forward-compatible across numpy 1.22↔2.x and
  jax 0.4.30↔0.10**. Run tests/scripts with the project's env Python and `PYTHONPATH=$PWD`; the optax
  example also needs `jax`+`optax` installed. Authors: Nick Alger, Blake Christierson.
  *(Local env paths + the `pip --only-binary` compiler workaround are maintainer-specific machine setup,
  not package requirements, so they live outside the repo.)*

## Where things live (routing rule)

Keep knowledge sorted by **audience × lifetime**, so it doesn't re-jumble. **Durable knowledge
always lands in `docs/`, tagged by audience; `/dev` holds only what is alive while a thread is
open** (Nick's rule, 2026-07-12):

- **User-facing design docs** (the *why*: architecture, conventions, rationale) → `docs/`
  (rendered Sphinx, the "Design notes" section).
- **Durable contributor references** (test strategy, style/authoring conventions,
  design-decision records, rejected alternatives, refactoring methodology) → `docs/contributor/`
  (rendered as the "Contributor guide" — dev-facing but public and permanent).
- **Ephemeral working notes** (handoffs, status, plans) → `dev/`; dated superseded ones →
  `dev/archive/`. The living current-state/handoff is **`dev/HANDOFF.md`** — read it for
  where-we-are + next steps. If something in a handoff turns out to be durable (a lesson, a
  settled decision), promote it to `docs/contributor/` before it washes out.
- **Research experiments + findings** → a separate research repo (maintainer-local; it *imports*
  the library — research never accretes here).
- **Maintainer-personal** prefs (work style, commit signature, local env) → personal `~/.claude/`, not the repo.

**Handoff ritual:** when wrapping up, refresh `dev/HANDOFF.md` (one living doc by default; more when
threads interleave), sweep superseded notes into `dev/archive/` as dated files, and keep this file's
Current-state pointer accurate. **Exception: a `dev/OPEN_QUESTION_*.md` note is a standing question, not
a thread — it is unresolved, not superseded, and must NOT be archived until resolved.** (None open
today; the `contractions.py` architecture question was resolved 2026-07-17 — the grouped-einsum
interpreter — and the uniform-frame-at-rank-deficiency question was resolved 2026-08-23 — the
pad-safe SVD, review S1b; both archived with resolution banners.)

## The paper (`t4s.pdf` in repo root)

> **Two distinct papers — don't conflate them.** *This* section is about the **T4S paper** (the research
> preprint below). A **separate** software / algorithms **reference paper for the toolbox itself** is being
> scoped in **`dev/paper_scope.md`** (ACM TOMS target; consolidates the T3 algorithms + hard-won
> implementation insights — a library reference, *not* a research contribution). When any note says "the
> paper," check which: **T4S** = the research preprint here; the **toolbox paper** = the new library reference.

**Status: a reference, not the goal.** The T4S paper is an **arXiv preprint** — not yet
journal-published, and still getting minor revisions (an example, some added detail). It sits in the
repo only as the reference for the *algorithms* and as context for why this code exists. It is **not**
the target of this work and **not** authoritative for API or design decisions — when the paper's
specific usage and good general-purpose library design conflict, library design wins (see "What this
is"). Use it to understand the math, not to decide what the library should do.

The math is in *Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series"
(arXiv:2603.21141)* — "T4S". **Appendix A** is the reference for the manifold/tangent code.
Notation map (paper ↔ code): `U`=up_tucker, `P`=left_tt, `Q`=right_tt, `O`=down_tt (outer);
`δU`=tucker_variations (V), `δG`=tt_variations (H). The code matches the appendix; the one known
divergence is the **Algorithm-11 orthogonalization sweep order** (code does left-then-right; the
paper does right-then-left) — the resulting orthogonal representations are equivalent. When code
implements a numbered equation/algorithm, cite it in the docstring — but note **the local `t4s.pdf` is
newer than the public arXiv version**: the numbering differs and some boxes (e.g. the probing
algorithms) aren't on arXiv yet. Cite numbers as they appear in the local `t4s.pdf`; arXiv will be
updated to this version by the time the package is released.

## Architecture

**Thin OO frontend over a pure-functional backend.**
- Frozen dataclasses hold only the cores; everything else is a `@cached_property`/method that
  delegates to `backend.*` functions operating on raw `.data` tuples, then re-wraps the result.

> **The backend/frontend razor (project-wide).** Decide what belongs in the backend from the
> perspective of a user who **bypasses the frontend entirely** and works on raw `.data` tuples — an
> important minority do exactly this, and they must be able to do everything the frontend does. For
> any piece of nontrivial logic ask: *would such a user rather find and call a backend function, or
> just rewrite it themselves?* If rewriting is easier (the logic is trivial), leaving it inline in the
> frontend is fine — don't bloat the backend. If the logic takes real thought to get right and is easy
> to get **wrong** (e.g. the frame-inner stack-axis bookkeeping, a depth-aware tree zip), it belongs in
> the backend so they can find and reuse it. **Exception:** logic *inseparable* from the frontend —
> constructing/validating the `T3*` OO classes (the `.validate()` contracts), the same-frame guard (it
> needs the two `T3Frame` objects) — stays in the frontend regardless, since the backend cannot depend on
> the frontend.
> **Corollary:** don't leave a backend user one fiddly step short of a usable result — if a backend
> function would otherwise force them to know some follow-up call (e.g. `tree_zip` to pair two returned
> trees), fold that step in so the function returns the directly-usable thing.
> **Backend classes: parameters are FIELDS, behaviour is METHODS.** "No classes in the backend" was a
> proxy that the geometry/optimization layers outgrew; the value it protected is that the math stays
> reachable. The sharpened rule: **backend functions implement the math on plain data; backend classes
> bind parameters and name roles, and every line of math in a method is also reachable as a standalone
> function.** A record-of-functions IS a class (dictionary-passing), and it hides the parameters — which
> matters because these objects ride as jax `aux_data`, so their hash/eq are the compilation cache key.
> Value identity comes from the fields (`common.ValueHashedFields`); a hand-maintained `identity` tuple
> beside them is the anti-pattern (it silently miscompiled). Full record:
> [`docs/contributor/parameters_not_closures.md`](docs/contributor/parameters_not_closures.md).
>
> **Corollary (the test is knowledge, not line count):** a one-line wrapper *earns* its place in the
> backend when it encodes a non-obvious **capability** — its value is the name + docstring + test, not
> the code. A trivial-*and*-obvious one-liner (`return a + b`) stays inline; a trivial-*but*-non-obvious
> one belongs in the backend, because a user who doesn't know the trick can't "just rewrite it." Example:
> the `t3_{apply,entries,probe}_corewise_transpose` trio (in `apply`/`entries`/`probing`) are each a
> one-line substitution into the *tangent*
> transpose (`P,Q,O → G`, paper §6.3), but that substitution is the non-obvious part — inlining it in
> the frontend would hide the corewise capability from backend users entirely.
- `TuckerTensorTrain` (`tucker_tensor_train.py`) — the keystone; `.data = (tucker_cores, tt_cores)`.
- `T3Frame` / `T3Variations` (`frame_variations_format.py`) — orthogonal frame + tangent direction.
- `T3Tangent` (`manifold.py`) — bundles `(T3Frame, T3Variations)`: a tangent vector.
- `backend/` — stateless functions, each module with its own `__all__`.

**Three representations** (the organizing principle):
- **ragged** — tuples of variably-shaped arrays. The default, fully working path.
- **uniform** — one stacked supercore array + masks (`ut3_*`, `ufv_*`, `uniform_*`); for
  `jax.lax.scan` vectorization. **The uniform mirror is COMPLETE** — `UniformTuckerTensorTrain`,
  `UT3Frame`/`UT3Variations`/`UT3Tangent`, the uniform geometries, sampling + jets, the weighted layer,
  shared factors, and the optimizers, all jit-wired with host-numpy masks. **Before
  touching uniform code, read the design notes** — governing: [`docs/uniform_equivalence_contract.md`](docs/uniform_equivalence_contract.md)
  (the uniform layer is a *faster ragged layer*: `to_uniform → op → to_ragged == op_ragged` on real
  parts, garbage don't-care — this is correctness *and* the test strategy). Then:
  [`docs/uniform_ranks_and_varieties.md`](docs/uniform_ranks_and_varieties.md) (a
  stacked uniform T3 is a batch in the bounded-rank **determinantal variety** — ranks may vary per
  stack element, shape is fixed), [`docs/uniform_supercore_layout.md`](docs/uniform_supercore_layout.md)
  (core index `d` **leads**: `(d,)+stack_shape+(...)`, for `lax.scan` + locality),
  [`docs/uniform_masks_vs_ranks.md`](docs/uniform_masks_vs_ranks.md) (**how the rank masks
  behave** — gappy under add/multiply, canonical vs working form, the tangent mask algebra; the
  *why-boolean-masks* decision record is
  [`docs/contributor/uniform_rank_masks_rationale.md`](docs/contributor/uniform_rank_masks_rationale.md)),
  [`docs/contributor/uniform_rank_masks_rationale.md`](docs/contributor/uniform_rank_masks_rationale.md) (**why the masks exist at
  all**: they enforce the variable-rank feature by zeroing the variation padding, so the gradient can't
  grow rank — a maskless "inflate to uniform rank" layer is operation-equivalent but loses rank control;
  considered & rejected) and [`docs/contributor/uniform_svd_prefix_orthogonalization.md`](docs/contributor/uniform_svd_prefix_orthogonalization.md)
  (orthogonalization must be **SVD-based** so the masks are a deterministic prefix), and
  [`docs/contributor/uniform_pytree_composition.md`](docs/contributor/uniform_pytree_composition.md) (`UT3 = tucker_supercore +
  tt_supercore + masks`-holder; the holder is a static `aux_data` with **value-based** hash/eq over mask
  **content** — the `common.ValueHashedMasks` mixin — so a rebuilt-but-identical holder is the *same* jit
  cache key and re-orthogonalizing the frame each optimization step does **not** recompile; the
  `T3Frame`↔`T3Tangent` pattern), and
  [`docs/uniform_backend_jit_recipe.md`](docs/uniform_backend_jit_recipe.md) (the **backend** jit story:
  masks can't be traced, so a backend optimizer holds them as loop-invariant state and traces only the
  supercores — jit the whole step closing over the frame masks, and the frame masks fall out as
  constant-folded constants; the design constraint for the 3b uniform optimizer).
- **weighted** — diagonal **edge weights** on the internal edges, a lightweight data format + `absorb`
  into cores (NOT a separate object layer; see [`docs/weighting.md`](docs/weighting.md)). Two classes:
  **`T3Weights`** (`tucker[d], tt[d+1]` = the `t3svd` sval format) weights a **`TuckerTensorTrain` as a
  tensor** (`t3_absorb_weights` + `t3_weighted_norm/inner` + `t3_{concatenate,kronecker}_weights` in
  `t3_operations`/`t3_linalg`; frontend `t3_absorb_weights`/`t3_weighted_norm`/`t3_weighted_inner` +
  `from_t3svd`); **`T3FrameWeights`** (`up/down/left/right`, each len d) is a **metric on a tangent's
  coordinates** (Grasedyck–Kramer preconditioner) — absorbed into the **variation** cores
  (`fv_absorb_weights`; `T3Tangent.weighted_norm/weighted_inner`, frame left orthonormal).
  `concatenate`↔`+`, `kronecker`↔`⊙` (Hadamard, Kronecker-of-weights verified). **Uniform mirrors all of
  it** (`UT3Weights`/`UT3FrameWeights` + `ut3_*`/`ufv_*`/`utv_*`). The old parked
  `wt3_*`/`EdgeVectors`/`WeightedTuckerTensorTrain` layer was **retired**.
  **Batching: a weight is absorbed into its target but batches with the FRAME** — `T3Weights` carries the
  tensor's `C`; `T3FrameWeights` carries the **frame's** `C`, *not* the variations' `K+C` (one metric per
  base point, broadcast over `K` for free). Conflating "what it acts on" with "what it batches with" was a
  real bug — [`docs/contributor/weighted_internals.md`](docs/contributor/weighted_internals.md). All
  machinery-1.

> **Batching/stacking is the most error-prone part of the library. Before touching anything with
> batch/stack axes, read [`docs/batching_and_stacking.md`](docs/batching_and_stacking.md)** — start with
> its **"Start here"** section (one-screen mental model + a concrete shape table + the shape-notation
> legend), then the numbered sections for the full reference (three meanings of "stack", the frame-inner
> convention and *why*, the `'...'`-vs-grouped-contraction machineries, heterogeneous-stack tuples,
> `vmap`/`jit` with the frame as a pytree leaf). The notes below are the terse version.
>
> The three batch **blocks** (mnemonic-and-collision-free letters): **`C`** = frame/core stack (base
> points, on *every* core; `= stack_shape`), **`W`** = probe stack (the `w` vectors, on `ww` only),
> **`K`** = tangent stack (the `k` tangent vectors at *one* frame, on the *variations* only). Order is
> frame-inner **`W + K + C`**. **The most common slip is conflating `C` (base points) with `K` (tangent
> vectors at a frame)** — they live on different operands and mean different things. (These were `F`/`V`/`G`
> before a rename that made them disjoint from the core/variation symbols `U`/`P`/`Q`/`O`/`G`/`H`/`V`.)

**"Stacking" means three different things** — keep them straight:
1. `stack_shape`: leading batch axes on one object's cores (`core.shape = stack_shape + (...)`).
   A leading `'...'` rides these along for free — **but only ONE batch block** (one shared/broadcast
   prefix). "Add stacking to a function" = rewrite its einsums/concats with `'...'`/negative axes.
2. `backend/stacking.py`: convert a Python tree of separate objects ↔ one stacked object
   (`stack`/`unstack`, `tree_zip`, `apply_func_to_leaf_subtrees`).
3. the uniform supercore (the separate, deferred representation).

**Two batch machineries** (full detail + the *why* are in the doc above): (1) **one** broadcastable
prefix → a leading `'...'` einsum, which rides `stack_shape` for free; (2) **two** independent blocks
on *different* operand subsets (canonical case: core/frame stack `C` on the cores vs probe stack `W` on
`ww` only — a single `'...'` can't express it) → the grouped-einsum interpreter
`backend.contractions.contract('WCa,Caib,WCi->WCb', *ops)` (UPPERCASE letter = a group of zero-or-more
axes, solved from the operand ndims; `len_W=`/`len_C=` supplied exactly when the string alone can't pin
a split — the error says which; no reshape ever happens, so every group sub-axis shards freely).
**Convention (library-wide, frame-inner): core stack `C` innermost, extra stacks
`W`/`K` outermost** (`W+C`, `K+C`, `W+K+C`) — because `'...'`-broadcast replicates a frame over the
extras for free only when `C` is innermost. (`apply`/`entries` were the last `C+W` holdout — flipped
in 5b; the whole library is now frame-inner.)

**The three sampling operations** (`entries`, `apply`, `probe`) — evaluate the dense tensor without
forming it, differing in how many modes stay free (`probe`: one → `d` vectors; `apply`/`entries`:
none → scalar) and the test vectors (general vs one-hot). Containment `probe ⊃ apply ⊃ entries`; all
three have tangent (Riemannian `𝒥`/`𝒥ᵀ`) forms. **What differs, costs, and how each fits a Riemannian
solve:** [`docs/entries_apply_probe.md`](docs/entries_apply_probe.md). Probing is the original
exemplar; `apply`/`entries` are the general-purpose all-modes special cases.

**Two gotchas:**
- **Canonical core-tuple orderings (frontend takes precedence).** `TuckerTensorTrain.data =
  (tucker_cores, tt_cores)`; `T3Frame.data = (up, down, left, right) = (U, O, P, Q)` (the `O`/down core
  is called **`down_tt_cores`**, not "outer"); `T3Variations.data = (tucker_variations, tt_variations)`;
  `(frame, variations)` pairs are frame-first. All verified backends (`tv_operations`,
  `fv_conversions` (incl. `t3_orthogonal_representations`), **and the sampling modules
  `probing`/`apply`/`entries`**) take these tuples in this
  exact order — pass `.data` straight through, no reorder. (The weighted layer's `fv_operations.py` — the
  shipped `T3FrameWeights` tangent-metric backend, not parked — uses the canonical order too.)
- **`corewise_dot`/`corewise_norm` collapse EVERY axis** (stacks included) to a scalar. To keep the
  stack (vectorized linalg), use `corewise.corewise_stack_dot(X, Y, n_stack)`.

**Backend dispatch**: `xnp, xmap, xscan = get_backend(is_uniform, use_jax)` (`backend/common.py`).
`xnp` = numpy or jax.numpy; `xmap`/`xscan` = ragged loops / numpy / `jax.lax`.

**numpy-vs-jax dispatch convention: infer it from the input array types at the lowest level — do NOT
thread `use_jax` params around.** Each operation computes `use_jax = tree_contains_jax((its inputs))`
(or `is_jax_ndarray(...)`) and the jax-ness propagates through the computed intermediates (so a
delegating function needn't infer at all — its callees do, from the arrays passed down). Applied
across the verified code: `probing`, `manifold`, `frame_variations_format`, `corewise`, and their
backend deps (`tv_operations`, `fv_conversions`, `t3_operations`, `t3_linalg`,
`t3_orthogonalization`, `linalg`) — **operations carry no `use_jax`**.
**The exception: pure constructors with NO array inputs** — `TuckerTensorTrain.randn/zeros/ones` and
`load`, `t3_corewise_randn`/`t3_zeros`/`t3_ones`, `common.randn`, and the rank-spec helpers in
`ranks.py` — keep a `use_jax` flag (there's nothing to infer from; it chooses the output type).
Factories that DO take an existing object infer from it (e.g. `T3Tangent.zeros/randn` from the frame,
`from_tensor_train`/`from_canonical` from the cores). *(The weighted layer — ragged and uniform — follows
the inferred convention throughout; the old parked layer that threaded `use_jax` is retired.)*
**The other exception — uniform masks are ALWAYS numpy, by `np` not `xnp` (and that is intentional, not
a backend-agnosticism bug).** A `UniformTuckerTensorTrain`'s masks are static *structure* (jax pytree
`aux_data`), so all mask logic — building, rank recurrences, `+`/`×` concat/Kronecker, `int(mask.sum())`
shape/rank extraction — runs on the **host with `np`**, while only the supercores (data) flow through
`xnp`. This is required for jit: inside a trace any `jnp` op on a mask yields a tracer, breaking
`int()` extraction and leaking tracer masks into `aux_data`; numpy masks instead fold into the compiled
program as device constants (zero per-call transfer). **Historically a bare `np.` was a tell that code
wasn't backend-agnostic — that heuristic does NOT apply to uniform mask code; do not "fix" mask `np.*`
to `xnp`.** Rule: **supercores → `xnp`; masks → `np`.** Full reasoning + the deferred eager
`jax.device_put` option: [`docs/contributor/uniform_pytree_composition.md`](docs/contributor/uniform_pytree_composition.md),
[`docs/uniform_masks_vs_ranks.md`](docs/uniform_masks_vs_ranks.md).

## Code style (deliberate and nonstandard — do NOT normalize)

- **Naming:** the family prefix grammar, module map, semantic markers ("corewise",
  "numerically_"), and the cataloged deliberate exceptions live in
  **[`docs/naming_conventions.md`](docs/naming_conventions.md)** — read it before naming anything
  new. Governing principle: names exist to help the user; when convention and clarity conflict,
  clarity wins.

- **Signature shape comments — the trailing comment IS the type the language can't express** (Python's
  `NDArray` says nothing; the *shape* is the real contract). One argument per line, one return element
  per line (expand return tuples even when they'd fit), three vertically-aligned columns (name · type ·
  `#` shape-contract). Micro-grammar `# len=d, elm_shape=...`. **Annotate what Python *can* express**
  (real `Union`/`Optional`, include `None`); the comment carries only what it **can't** (shapes,
  sequence lengths like `len=d+1`, constraints, semantics) — so a `#`'s presence signals an
  inexpressible contract. Names/types always align; **comments align in one column when the
  type-length spread is small, else split into blank-line-delimited groups of similar types and align
  within each group** (never the unpredictable staircase). A **principle, applied within reason**
  (trivial scalars may need no comment). Full rationale + rules + exemplar:
  **[`docs/contributor/signature_style.md`](docs/contributor/signature_style.md)** (reference module: `backend/probing.py`).
- Body locals encode axis layout in the **name suffix** (`C_aib`, `mu_WCa`, `B0_b_j_c`), matching the
  grouped-subscripts scheme (`C`/`W`/`K` = grouped index blocks, lowercase = single axes, leading `d` =
  stacked/derivative axis; contractions are `contract('WCa,Caib,WCi->WCb', ...)` calls).
- `math.prod` (not `np.prod(..., dtype=int)`) for static products of shape ints.
- einsum everywhere with a leading `'...'`; numpy path passes `optimize=path`, jax path omits it.
- Uppercase single-letter core names (`U V G P Q O …`) are intentional — ignore "should be
  lowercase" linter notes.

## House philosophy: structural problems always error; numerical preconditions are checked in safe mode

The dividing line is **structural vs numerical**:

- **Structural problems → hard error, always (both modes).** Wrong shape, inconsistent ranks/lengths,
  mismatched stack shapes — these raise unconditionally. `TuckerTensorTrain`, `T3Frame`, and
  `T3Variations` validate in `__post_init__`. If something is the wrong shape: error. (Structural
  consistency is *not* governed by the safety mode below.)

- **Numerical problems → enforced as PRECONDITIONS in safe mode, skipped in unsafe / under jit.**
  *(This supersedes the older "numerical problems → warn, never enforce" rule.)* The library carries an
  ambient **safety** setting (`t3toolbox/safety.py`, a `contextvars` var — the one sanctioned global,
  justified because it is correctness-neutral): a tolerance pair is **safe mode** (the default), `None`
  is **unsafe mode**. In safe mode an op with a genuine numerical **precondition** (its result is
  *wrong* without it) checks it and raises; `with safety.unsafe():` or a jax trace (jit/grad/vmap — you
  cannot branch on a tracer) **skips** it. This is **correctness-neutral** (the `assert`/`-O`
  precedent): the *numbers* are identical, only error-catching differs. The non-enforcing checkers
  (`is_orthogonal`, `is_gauged`, `has_minimal_ranks`, `safety.frames_equal`) still exist and *are* the
  checks. Master plan: [`dev/archive/safe_unsafe_mode_plan.md`](dev/archive/safe_unsafe_mode_plan.md); the full
  precondition-vs-caveat sweep: [`docs/contributor/numerical_contract_catalog.md`](docs/contributor/numerical_contract_catalog.md).

  - **Precondition vs caveat (only preconditions are enforced).** A **caveat** — the op is valid and
    correct *as computed*, the property only governs what the result *means* (e.g. "this coordinate dot
    equals Hilbert–Schmidt") — is **never** enforced; it would reject legitimate use. The enforced
    preconditions are concentrated in the manifold/tangent surface: **same-frame** (`T3Tangent`
    `+`/`-`/`stack_tangents`, `GaussNewtonModel` matvecs), **orthogonal frame**
    (`MANIFOLD.project`/`project_oblique`/`retract`/`project_ambient`/`transport`), **gauged variations**
    (`MANIFOLD.inner`/`norm`). `TuckerTensorTrain` / `corewise` / `probing` are **precondition-free**
    (exact for any cores). The check sites use `safety.checks_active(*operand_data)` (safe **and** not
    tracing) → `safety.require(<checker>(atol=safety.effective_rtol(...)), msg)`, routed through cached
    residuals (`T3Frame.orthogonality_residual`, `T3Tangent.gauge_residual`) so a fixed frame/tangent in
    an inner loop is contracted once. **Frontend-only** for now (a backend mirror for raw-`.data` users
    is deferred).

  - **"Same tangent space" is NUMERICAL, not structural** (the key correction). Two tangents share a
    tangent space iff their frames are *numerically equal* (`safety.frames_equal`, with an `is`-identity
    fast path) — **not** object identity. The old `self.frame is other.frame` identity guard was a
    numerical check faked as structural; it forced `T3Tangent`'s frame to be jax **aux_data** (→ jit
    recompiled on every frame change) and false-failed on a jit round-trip. The numerical guard let the
    frame become a pytree **leaf** (frame flows as traced data → no recompile). Likewise `GaussNewtonModel`
    is a registered pytree (frame/sweep/sample/residual leaves; geometry/kind aux) — you can `jit` the
    frontend matvec directly and it compiles **once** across all bases.

  - **Two tolerances + minimal-rank naming.** A check picks `rtol_jax` (looser) when any input is a jax
    array, else `rtol_numpy`, because jax runs **float32** by default — this supports jax-but-not-jit
    usage in safe mode (autodiff prototyping). Naming: bare **`minimal_ranks` = structural** (cheap
    integer arithmetic, `has_minimal_ranks`); **`numerically_minimal` = numerical** (an SVD;
    `has_numerically_minimal_ranks`). **Minimal rank is NOT a correctness precondition for any op**
    (settled empirically — catalog); it survives only as a `retract` caveat + diagnostic checkers.

  - **inner/norm live on the *Geometry*, not the tangent.** The Hilbert–Schmidt metric is
    `MANIFOLD.inner`/`norm` (checks same-frame + orthogonal + gauged); the Euclidean coordinate metric is
    `COREWISE.inner`/`norm` (same-frame only). The raw coordinate ops on `T3Tangent` are
    `corewise_inner`/`corewise_norm` (no HS claim, no orth/gauge check).

## Verification & testing

- Correctness is checked against **dense ground truth**: rebuild via `.to_dense()` + a hand-written
  `np.einsum`, compare residual norms (~1e-12..1e-16). Verify a math property empirically with a
  quick script before asserting it in a test.
- **The deeper rationale — and a real trap — is [`docs/contributor/testing_strategy.md`](docs/contributor/testing_strategy.md):**
  dense/numerical tests on clean-padding inputs are **blind to too-permissive masks** (phantom rank — the
  doubled-boundary bug class). For any uniform op, dense-vs-ragged is **not enough**; also assert **exact
  output masks** (derived non-circularly) and **garbage-padded-input robustness**. Read it before adding
  uniform tests.
- `unittest`, in `tests/`. Pattern: `subTest` over structures × stack_shapes. **Numerical
  correctness is checked numpy-only** — the backend is backend-agnostic (`xnp`, `'...'` einsums,
  inferred dispatch), so jax computes the same numbers; duplicating every sweep in jax was wasted
  time. **jax invocation is covered separately by `tests/test_dispatch.py`** (which jit-compiles each
  op — a stray `np.*` on a tracer raises, proving no hidden numpy — plus a jax-in→jax-out check for
  dynamic-shape rtol/atol ops, plus a few numerical smoke tests). This cut the full suite from ~550s
  to ~50s. When adding a numerical test, write it numpy-only and add the op to `test_dispatch` if its
  jax dispatch isn't already covered. (Frozen dataclasses are registered jax pytrees; `T3Tangent` carries
  the **frame as a leaf** — the same-frame guard is a *numerical* `frames_equal` check that survives the
  jit round-trip, so the frame flows as traced data with no per-frame recompile — see `manifold.py`.)
  A few tests still build explicit jax operands where that *is* the thing under test (e.g.
  `test_contains_jax`); those keep the `jnp` import.
- **Doctests = reproducible examples**, CI-enforced (the `tests` workflow runs module doctests, the
  quickstart page, **and every `docs/*.md` + `docs/contributor/*.md` page**, on both numpy
  generations — so a documented example that the library outgrows fails the build; the one exclusion
  is `doctest_style.md`, whose fragments are illustrative. Page examples are `>>>` sessions inside the
  ```` ```python ```` fence, and **need a blank line before the closing fence** or doctest swallows it
  into the expected output): **examples first** (teach the API), not coverage. Seed (`np.random.seed(0)`) or fixed inputs; value-match via
  `np.allclose`; print **structure** (shapes/ranks) not raw values; show **gotchas** (structural →
  traceback `+IGNORE_EXCEPTION_DETAIL`; numerical → wrong-vs-right). One distinct behavior per option
  (no cross-product); long tail → prose. **Run the example and paste the real output — never hand-write
  it.** Full convention + exemplar (`manifold.py`): **[`docs/contributor/doctest_style.md`](docs/contributor/doctest_style.md)**.
  (Supersedes the old "illustrative captured values" convention.)
- **Running tests/scripts**: use the project's env Python with `PYTHONPATH=$PWD`, e.g.
  `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q -n auto` (tests are per-test seeded and
  order-independent; CI runs `-n auto` too). The review's oracle sweeps are permanent two-tier tests
  (`tests/test_oracle_sweep.py`): always-on plus `T3TOOLBOX_SLOW_TESTS=1` for the full optimizer
  matrix -- the slow tier is a REQUIRED release-gate step. The optax example needs `jax`+`optax` installed.
  (The maintainer's exact env path lives in personal `~/.claude/`.)

## Workflow

*(Maintainer collaboration style + the commit co-author line are personal — see `~/.claude/`. The
project engineering practices below are shared.)*

- **Work in incremental, reviewable slices** — propose the plan and the genuine decisions, confirm,
  then implement; slice big restructures into reviewable units.
- **When *designing* a format/API, reason from the math/algorithms first** — set aside
  consistency-with-existing-conventions and code-change cost; design the *correct* thing, then change
  the code to match. Treat the library as **general-purpose**: do NOT privilege the T4S paper's
  optimization tasks (Riemannian fitting/CG) as "the" use case — a batched or exotic operation is as
  legitimate as a fitting one.
- **Prefer minimal dataclasses** — cores only; derive shapes/splits rather than storing redundant
  fields (e.g. the `C`/`K` stack split is recovered from the (frame, variations) pairing).
- **Commit per logical chunk.** Verify tests pass first; write a descriptive message; stage only the
  relevant files (leave unrelated stray edits alone).
- **Changing a backend convention has a wide blast radius — grep ALL consumers.** An axis-ordering or
  signature change ripples through the OO wrappers that delegate to it (e.g. `TuckerTensorTrain.probe`
  → `t3_probe`) and *their* tests/doctests, not just the file you edited. After such a change, run the
  full suite (`test_tucker_tensor_train` + `test_manifold` + `test_frame_variations_format` +
  `test_contractions_interpreter`), not only the directly-touched tests.
- Don't ship a possibly-wrong result with a weak test — if something looks off, dig in or flag it.

## Current state

**2026.1.0 is SHIPPED to PyPI (2026-08-20) — `pip install t3toolbox`** (2026.0.0 shipped
2026-07-13; the checklist both followed is `dev/archive/release_plan_2026-07-13.md`; live status:
`dev/HANDOFF.md`). "Tested" = *numerical correctness in numpy* (vs dense ground truth) **plus** *jax
dispatch* covered by `tests/test_dispatch.py` (jit each op; a stray `np.*` on a tracer raises) — not a
duplicate numerical sweep. Full suite green (899 tests / 42,539 subtests; ~7 min with ``pytest -n auto``, ~14 min serial --
tests are per-test seeded and order-independent, Phase D);
docs at zero warnings with `-W` in CI; doctests CI-enforced on both numpy generations — **module
doctests, `getting_started.rst`, AND every `docs/*.md` + `docs/contributor/*.md` page** (the one
exclusion is `doctest_style.md`, whose fragments are illustrative).

- **The shipped surface (all solid/tested):** `TuckerTensorTrain` + backend (arithmetic,
  `to_dense`, `t3svd`, `t3m`, save/load; the three sampling ops + their symmetric derivatives +
  the ambient/corewise/tangent transposes); `frame_variations_format`; `manifold` (`T3Tangent`,
  `MANIFOLD`/`COREWISE`) + safe/unsafe mode; the geometry-generic `GaussNewtonModel` + the four
  optimizers + fitting from all sampling kinds and their derivatives, with an optional **`ω[mode,order]`
  residual-weight matrix** in the objective `½‖ω⊙r‖²` (per-**order** for the derivative kinds — the
  Gauss-Newton conditioner; per-**mode** for probe — probe is the only kind with a per-mode axis, so
  apply/entries stay order-only; a bare vector = per-order, backward compatible; **not** the parked
  weighted *layer* — this is a fitting-objective weight; `docs/fitting_and_optimization.md` §4.6); plus an
  optional **`regularizer=`** objective term `ρ(x)` — shipped: `IdentityRegularizer(λ)` = `½λ‖x‖²` (HS
  ridge on `MANIFOLD`, weight decay on `COREWISE`), composes with every optimizer/kind/geometry/representation,
  λ auto-scaled by `batch/n` in minibatch steps, extensible via the small `Regularizer` protocol
  (`backend/regularization.py`; Grasedyck–Kramer prior is future work); `docs/fitting_and_optimization.md`
  §4.9; plus an optional
  **`verbose=` per-iteration diagnostic display** for `newton_cg` (CG/line-search stats + a
  per-`(mode,order)` relative-error table, train/validation; a regularized run splits the objective as
  `obj = misfit + reg`, the `misfit`/`regularization` fields also carried in `stats['history']`) — backend-owned in
  `backend/optimizer_display.py` (`make_newton_display` + a `callback=` hook), so a raw-`.data` user gets
  the identical display; plus `newton_cg` **warm-start reference overrides** — `g0norm_newton` /
  `g0norm_cg` pin the reference `‖g0‖` the Newton stop / CG forcing term are relative to (default = the
  initial `‖g‖`, misleadingly small after a continuation warm start; `g0norm_newton` also feeds CG unless
  `g0norm_cg` is given), and `cg_forcing_power` (default `0.5`) tunes CG effort per Newton step
  (`docs/fitting_and_optimization.md` §5); plus **`use_jit=True` auto-converts** (`mc_sgd`/`adam`/`newton_cg`):
  requesting jit moves `x0`/`sample`/`data` onto jax and compiles, so it returns a **jax-backed** result
  (jax float32 unless x64); if jax is absent it **warns and runs eager** (the library-wide jax-absent policy, `common.jax_or_warn`) — loud, never the old silent drop
  (`_prepare_jit_inputs`; `docs/fitting_and_optimization.md` §4.5); **the full uniform mirror
  of all of it** — `UniformTuckerTensorTrain`, `UT3Frame`/`UT3Variations`/`UT3Tangent`, the
  uniform geometries, uniform sampling + jets, and the optimizers running fully packed,
  jit-compile-once, ragged-vs-uniform inferred from `x0` (per-element verified vs ragged +
  adjoint-identity + mask-strict + garbage-robust + jit-clean; `docs/contributor/testing_strategy.md`).
  Worked examples in `examples/fit_hilbert_*` (+ `examples/fit_per_mode_weight_probes.py` for the
  per-mode residual weight). (Build history: the archived plans in
  `dev/archive/` — `uniform_fix_plan`, `uniform_optimizers_plan`, `naming_pass_plan`,
  `docs_pass_plan`, `docs_split_plan`.)
- **Whole-library pre-release review (2026-08-22) — FULLY landed AND test-hardened (fix phase complete
  2026-08-24; Phase D complete 2026-08-25).** 19-lane review, 186 findings (ledger + repros: `dev/review_2026-08-22/`);
  every silent-wrong-answer and crash cluster is fixed with a regression test, **S1b included**
  (2026-08-23: the uniform frame at a *numerically* rank-deficient point — fixed by the mask-aware
  `backend.linalg.pad_safe_svd` threaded through every uniform sweep; the frame is now gauge-EQUIVALENT
  to ragged, no longer bit-identical — `docs/uniform_equivalence_contract.md` §"Gauge-carrying
  operations"), **and the deferred E-clusters worked through with Nick's rulings 2026-08-23/24**:
  explicit equality (`==`/hash raise on all twelve runtime classes; `allclose` + `corewise_equal`),
  jit cache-key hygiene, the docstring/stale-text sweep (no `dev/` paths in shipped text; derivative
  order is spelled `order`, axis letter `t`, `K` = tangent stack only — `docs/naming_conventions.md`
  §"Index letters"), the masked uniform same-frame guard, and the probe-only trace detector (+ the
  finding that a stacked UT3 cannot `vmap` — masks are static aux; `batching_and_stacking.md` §7).
  Rulings worth knowing are listed in the HANDOFF; the behavioural changes are in the CHANGELOG's
  `[2026.2.0]` Fixed / Changed sections (jax-absent → warn; `d = 1` degenerate; per-mode weight rows;
  adam-on-manifold warning; the frame gauge; the equality flip).
- **Optimization layer restructured (2026-08-21; in 2026.2.0).** The geometry, the sampling kind and the local model are frozen
  dataclasses whose **parameters are fields**, not records of closures, so they hash/compare by value
  and are stable jax `aux_data`. New `backend/geometry.py`; `SamplingKind` is a class hierarchy;
  `UniformGaussNewtonModel` merged into `GaussNewtonModel`; sharing is a `groups` field.
  `backend/optimizers.py` now imports no T3-specific module. Inner CG compiles **once per fitting run
  instead of once per Newton iteration**. Fixed along the way: a silent miscompile where a
  `dc.replace`-derived kind reused its parent's compiled program, and a stacked+regularized fit that
  silently mis-weighted (now raises, as does stacked optimization generally). Breaking — see the
  CHANGELOG's `[2026.2.0]` and the upgrade notes. **Why:
  [`docs/contributor/parameters_not_closures.md`](docs/contributor/parameters_not_closures.md).**
- **Design references:** the rendered docs are the reference — user tier (`docs/*.md` +
  the user guide) and the Contributor guide (`docs/contributor/`); `entries_apply_probe.md` §8
  carries the probing paper↔code map.
- **Shared Tucker factors (SF-T3) — BUILT, ragged AND uniform (2026-08-19/20; shipped in 2026.1.0).** Optimize over T3s whose Tucker factors are tied within
  user-specified mode groups (SF-ETT, Molozhavenko & Rakhuba 2026, generalized to arbitrary
  partitions — the arbitrary-partition dimension/smoothness is OUR extension). Surface: `sharing=`
  on `t3svd`/`rank_adjustment_sweep`/`get_minimal_ranks`/`manifold_dim`/`continuation_ranks`/
  `resize` (+ uniform twins), `x.share(...)`, `has_shared_tucker_factors` (a METHOD — checker
  grammar), the `shared(base, sharing)` geometry wrapper (`shared_manifold`/`shared_corewise`,
  uniform bases included, compile-once), and the (breaking) `Geometry.precompute` aux slot (the protocol in `backend/optimizers.py`).
  **User doc: [`docs/sharing.md`](docs/sharing.md)** (incl. "What the group spectrum is" — the
  four faces of `s_g`; sharing ≠ symmetry); **design records:
  [`docs/contributor/sharing_internals.md`](docs/contributor/sharing_internals.md)** (the S_i
  re-sweep/SVD-not-Gram measurements, the two-phase decision, the tied embedding, the
  padded-restart analysis — full shared rank is a DIAGNOSTIC, never a precondition). Example:
  `examples/fit_shared_factors_jetted_probes.py`. The derivation note is `docs/shared_t3_math.tex`
  (+pdf); the build spec is archived at
  `dev/archive/shared_factors_handoff_2026-08-20_complete.md`.
- **Weighted layer — SHIPPED, ragged AND uniform** (the edge-weight redesign + its uniform mirror):
  `T3Weights`/`UT3Weights` (tensor) + `T3FrameWeights`/`UT3FrameWeights` (tangent metric), each with
  `absorb`/`weighted_norm`/`weighted_inner`/`reciprocal`/`sqrt`/`concatenate`/`kronecker`, the
  `from_*svd`/`from_*weights` constructors, and ragged↔uniform conversions; the old parked `wt3_*` layer
  retired. **User doc: `docs/weighting.md`; design records:
  [`docs/contributor/weighted_internals.md`](docs/contributor/weighted_internals.md)** (build records
  archived in `dev/archive/`). **Three things not to re-derive** (all in that note): a weight is absorbed
  into the **variations** but batches with the **frame** (stack `C`, not `K+C` — conflating those was a real
  bug); uniform `absorb` is **garbage-transparent** so it needs no masking (which is why the weighted code
  never touches the masking layer); and uniform `reciprocal` **must** guard the padding (`1/0 = inf` →
  `0*inf = nan`; the GK metric *is* a reciprocal) but deliberately does not guard real-slot zeros.
  Deferred (reachable from the primitives): weighted `+`/`⊙`/scale operations and the Grasedyck–Kramer
  `SingularValueRegularizer` — both layers now have everything it needs.

## Open questions / TODO

Live status + backlog: **`dev/HANDOFF.md`**. The durable open items:

- **Goal-1 `fit(...)` facade** — auto geometry/optimizer/ranks/`x0` + rank-continuation/validation
  ("standard user, no fiddling"). **Deferred to 1.1**; 1.0 ships as an honest mid-level toolkit.
- **Weighted tensor-network layer — SHIPPED, ragged + uniform** (`docs/weighting.md`; design records
  `docs/contributor/weighted_internals.md`). Follow-ups (deferred, reachable from the primitives):
  weighted `+`/`−`/scale/`⊙` operations + an optional thin container, and the Grasedyck–Kramer
  `SingularValueRegularizer` — both layers now have every primitive it needs.
- **Minimal-rank audit — RESOLVED** (the settled verdict, per the catalog and the enforced check
  sites): minimal rank is a correctness precondition for **nothing** — gauge projections +
  `project`/`project_dense_onto_tangent` need orthogonality only; `inner`/`norm` HS-faithfulness
  needs orthogonal + gauged (NOT minimal); `retract` preserves frame ranks only on a minimal frame
  (a caveat, not a precondition). User statement: `docs/numerical_contracts.md`; decision record:
  `docs/contributor/numerical_contract_catalog.md`.
- **Deferred/rejected design items** are cataloged in the rendered ledger
  (`docs/contributor/deferred_and_rejected.md`) — incl. the ambient derivative transpose and the
  structured-`K` open idea. Smaller niceties (per-test seeding → `pytest -n auto`; trimming
  `test_dispatch` jit time) live in the HANDOFF backlog. **No auto-formatter near the
  deliberately-nonstandard code style.**
