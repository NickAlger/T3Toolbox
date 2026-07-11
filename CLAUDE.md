# T3Toolbox — project guide

A working reference for collaborating on this codebase. Read this first.

## What this is

Pure-Python (NumPy + optional JAX) library for **Tucker tensor trains (T3)** — a Tucker
decomposition whose central core is stored as a tensor train. Nick Alger and Blake Christierson are converting research
code into a standalone package: cleaning up, documenting, restructuring for usability, adding
features. **The goal is a general-purpose library for other people and other use cases** — not to
reproduce or serve any particular application. The research it came from (the T4S paper, below) is
**essentially done** (an arXiv preprint, with only minor revisions still in flight); treat it as
historical reference for the algorithms, never as the design target. The README still says "WORK IN PROGRESS DO NOT USE."

- Repo: `github.com/NickAlger/T3Toolbox` (renamed from `TuckerTensorTrainTools`; that rename left
  stale references we've mostly fixed). Branch `main`, direct commits.
- Env: pure-Python (numpy + optional jax), **forward-compatible across numpy 1.22↔2.x and
  jax 0.4.30↔0.10**. Run tests/scripts with the project's env Python and `PYTHONPATH=$PWD`; the optax
  example also needs `jax`+`optax` installed. Authors: Nick Alger, Blake Christierson.
  *(Local env paths + the `pip --only-binary` compiler workaround are maintainer-specific machine setup,
  not package requirements, so they live outside the repo.)*

## Where things live (routing rule)

Keep knowledge sorted by **audience × lifetime**, so it doesn't re-jumble:

- **User-facing design docs** (the *why*: architecture, conventions, rationale) → `docs/` (rendered Sphinx).
- **Internal working notes** (handoffs, status, plans) → `dev/`; dated superseded ones → `dev/archive/`.
  The living current-state/handoff is **`dev/HANDOFF.md`** — read it for where-we-are + next steps.
- **Research experiments + findings** → a separate research repo (maintainer-local; it *imports*
  the library — research never accretes here).
- **Maintainer-personal** prefs (work style, commit signature, local env) → personal `~/.claude/`, not the repo.

**Handoff ritual:** when wrapping up, refresh `dev/HANDOFF.md` (one living doc by default; more when
threads interleave), sweep superseded notes into `dev/archive/` as dated files, and keep this file's
Current-state pointer accurate.

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
  `jax.lax.scan` vectorization. **`UniformTuckerTensorTrain` (the plain layer) is built through slice 8
  and jit-wired with host-numpy masks** (uniform frame/variations/tangents still deferred). **Before
  touching uniform code, read the design notes** — governing: [`docs/uniform_equivalence_contract.md`](docs/uniform_equivalence_contract.md)
  (the uniform layer is a *faster ragged layer*: `to_uniform → op → to_ragged == op_ragged` on real
  parts, garbage don't-care — this is correctness *and* the test strategy). Then:
  [`docs/uniform_ranks_and_varieties.md`](docs/uniform_ranks_and_varieties.md) (a
  stacked uniform T3 is a batch in the bounded-rank **determinantal variety** — ranks may vary per
  stack element, shape is fixed), [`docs/uniform_supercore_layout.md`](docs/uniform_supercore_layout.md)
  (core index `d` **leads**: `(d,)+stack_shape+(...)`, for `lax.scan` + locality),
  [`docs/uniform_masks_vs_ranks.md`](docs/uniform_masks_vs_ranks.md) (rank metadata is **boolean
  masks**, not integer ranks — closed under add=concat / multiply=Kronecker with no data movement),
  [`docs/uniform_rank_masks_rationale.md`](docs/uniform_rank_masks_rationale.md) (**why the masks exist at
  all**: they enforce the variable-rank feature by zeroing the variation padding, so the gradient can't
  grow rank — a maskless "inflate to uniform rank" layer is operation-equivalent but loses rank control;
  considered & rejected) and [`docs/uniform_svd_prefix_orthogonalization.md`](docs/uniform_svd_prefix_orthogonalization.md)
  (orthogonalization must be **SVD-based** so the masks are a deterministic prefix), and
  [`docs/uniform_pytree_composition.md`](docs/uniform_pytree_composition.md) (`UT3 = tucker_supercore +
  tt_supercore + masks`-holder; the holder is a static `aux_data` with **value-based** hash/eq over mask
  **content** — the `common.ValueHashedMasks` mixin — so a rebuilt-but-identical holder is the *same* jit
  cache key and re-orthogonalizing the frame each optimization step does **not** recompile; the
  `T3Frame`↔`T3Tangent` pattern), and
  [`docs/uniform_backend_jit_recipe.md`](docs/uniform_backend_jit_recipe.md) (the **backend** jit story:
  masks can't be traced, so a backend optimizer holds them as loop-invariant state and traces only the
  supercores — jit the whole step closing over the frame masks, and the frame masks fall out as
  constant-folded constants; the design constraint for the 3b uniform optimizer).
- **weighted** — cores + edge-weight vectors (`wt3_*`, `weighted_*`); weights "absorbed" into cores.
  Tangent weighting (`absorb_weights_into_tangent_cores`) is **parked** in `backend/fv_operations.py`
  pending a redesign of weighted tensor networks.

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
`ww` only — a single `'...'` can't express it) → the named grouped-block contractions in
`backend/contractions.py` (`WCa_Caib_WCi_to_WCb` etc.; each capital block reshaped to one flat axis,
= 1 when empty). **Convention (library-wide, frame-inner): core stack `C` innermost, extra stacks
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
  exact order — pass `.data` straight through, no reorder. (Only the parked `fv_operations.py`
  `absorb_weights` still uses the old `(up, left, right, outer)` probing order.)
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
`from_tensor_train`/`from_canonical` from the cores). The deferred weighted layer still threads
`use_jax` (old pattern) — migrate when repairing it.
**The other exception — uniform masks are ALWAYS numpy, by `np` not `xnp` (and that is intentional, not
a backend-agnosticism bug).** A `UniformTuckerTensorTrain`'s masks are static *structure* (jax pytree
`aux_data`), so all mask logic — building, rank recurrences, `+`/`×` concat/Kronecker, `int(mask.sum())`
shape/rank extraction — runs on the **host with `np`**, while only the supercores (data) flow through
`xnp`. This is required for jit: inside a trace any `jnp` op on a mask yields a tracer, breaking
`int()` extraction and leaking tracer masks into `aux_data`; numpy masks instead fold into the compiled
program as device constants (zero per-call transfer). **Historically a bare `np.` was a tell that code
wasn't backend-agnostic — that heuristic does NOT apply to uniform mask code; do not "fix" mask `np.*`
to `xnp`.** Rule: **supercores → `xnp`; masks → `np`.** Full reasoning + the deferred eager
`jax.device_put` option: [`docs/uniform_pytree_composition.md`](docs/uniform_pytree_composition.md),
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
  **[`docs/signature_style.md`](docs/signature_style.md)** (reference module: `backend/probing.py`).
- Body locals encode axis layout in the **name suffix** (`C_aib`, `mu_WCa`, `B0_b_j_c`), matching the
  contraction-naming scheme (`C`/`W`/`K` = grouped index blocks, lowercase = single axes, leading `d` =
  stacked/derivative axis; functions named `inputs_to_output`, e.g. `WCa_Caib_WCi_to_WCb`).
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
  precondition-vs-caveat sweep: [`docs/numerical_contract_catalog.md`](docs/numerical_contract_catalog.md).

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
- **The deeper rationale — and a real trap — is [`docs/testing_strategy.md`](docs/testing_strategy.md):**
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
- **Doctests = reproducible examples** (not yet wired into CI, but written so they can be): **examples
  first** (teach the API), not coverage. Seed (`np.random.seed(0)`) or fixed inputs; value-match via
  `np.allclose`; print **structure** (shapes/ranks) not raw values; show **gotchas** (structural →
  traceback `+IGNORE_EXCEPTION_DETAIL`; numerical → wrong-vs-right). One distinct behavior per option
  (no cross-product); long tail → prose. **Run the example and paste the real output — never hand-write
  it.** Full convention + exemplar (`manifold.py`): **[`docs/doctest_style.md`](docs/doctest_style.md)**.
  (Supersedes the old "illustrative captured values" convention.)
- **Running tests/scripts**: use the project's env Python with `PYTHONPATH=$PWD`, e.g.
  `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q`. The optax example needs `jax`+`optax` installed.
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
  `backend/test_contractions`), not only the directly-touched tests.
- Don't ship a possibly-wrong result with a weak test — if something looks off, dig in or flag it.

## Current state

**The core library runs and is tested; some advanced layers are deferred.** Live status + the 1.0
roadmap live in **`dev/HANDOFF.md`** — read it for where-we-are and next steps. "Tested" = *numerical
correctness in numpy* (vs dense ground truth) **plus** *jax dispatch* covered by `tests/test_dispatch.py`
(jit each op; a stray `np.*` on a tracer raises) — not a duplicate numerical sweep, and not a guarantee
of every path. Full suite ~50s, green.

- **Solid / tested:** `TuckerTensorTrain` + backend (arithmetic, `to_dense`, `t3svd`, `t3m`, save/load;
  the three sampling ops `apply`/`entries`/`probe` + their symmetric derivatives + the three transpose
  families ambient/corewise/tangent); `frame_variations_format`; `manifold` (`T3Tangent`, the
  `MANIFOLD`/`COREWISE` geometries) + safe/unsafe mode (`safety.py`); the geometry-generic
  `GaussNewtonModel` (`fitting.py`) + the four optimizers
  (`gradient_descent`/`mc_sgd`/`adam`/`newton_cg`, in `optimizers.py`) + least-squares fitting from
  apply/entries/probe **and their derivatives**; the **plain** `UniformTuckerTensorTrain` + `ut3_*`
  backend (through slice 8); **the uniform `UT3Frame`/`UT3Variations` foundation (increment 2c) and the
  uniform *tangent + manifold layer* (increment 3b) — the *tangent backend* (`UT3Tangent` +
  `backend/utv_operations`: doubled-rank `to_ut3`/`to_dense`, `retract`, gauge,
  `utv_project_ut3_onto_tangent_space`, cross-layer converters, stack/unstack), the *two geometries*
  (`UNIFORM_MANIFOLD`/`UNIFORM_COREWISE`, 3b-5), *tangent + corewise probing* (3b-6: the `d`-prefixed
  `WKC` contractions, `UT3Tangent.{probe,apply,entries}` + their `𝒥ᵀ` transposes, and the corewise
  `UniformTuckerTensorTrain.*_corewise_transpose`), and now the *derivative (jet) probing* (**3b-6′**: the
  `d`-prefixed JET `trs_*` contractions, `{UniformTuckerTensorTrain,UT3Tangent}.{probe,apply,entries}_derivatives`
  + their `𝒥ᵀ` transposes + the corewise `*_corewise_derivatives_transpose`) — all per-element verified vs
  ragged + adjoint-identity + mask-strict + garbage-robust + jit-clean** (see `docs/testing_strategy.md`).
  Worked examples in `examples/fit_hilbert_*`.
- **Design references** (the durable *why*, to be folded into user docs in the doc pass):
  `docs/fitting_and_optimization.md`, `docs/batching_and_stacking.md`, `docs/entries_apply_probe.md`,
  `docs/transposes.md`, `docs/numerical_contract_catalog.md`, `docs/probing_section6_notes.md`,
  `docs/signature_style.md`, `docs/doctest_style.md`, the `docs/uniform_*` notes, the `docs/t3svd_*`
  notes. (Historical plans/handoffs are archived under `dev/archive/`.)
- **Uniform tangent layer + optimizers — DONE, INCLUDING the frontend (the 1.0 centerpiece).** The *backend*,
  *geometries*, *tangent + corewise probing*, and the *derivative (jet) probing* (increment 3b), the
  **optimizers/fitting on the uniform layer** (increments U1–U6 + U5.6), AND now the **frontend surface (U7)**:
  all four optimizers run on uniform, **fully packed**, **jit-compile-once**, verified vs the ragged optimizer,
  robust to non-minimal input (`uniform_minimal`). Backend pieces: `backend/uniform_fitting.py`
  (`uniform_{manifold,corewise}_ops`, the `uniform_*_kind` `SamplingKind` builders, `uniform_least_squares_problem`,
  `pack_sample`/`pack_data`), the packedness-mirror convention (`ut3_operations.{is_packed,pack_if_ragged}`), a
  GPU benchmark (`dev/bench_uniform_vs_ragged.py`). **Frontend (U7):** the four `optimizers.*` and the six
  `fitting.*_model` factories **infer** ragged-vs-uniform from `x0`'s type (a uniform `x0` requires a uniform
  geometry singleton; the optimizer path calls `uniform_minimal` transparently). The roll-your-own surface is
  `fitting.UniformGaussNewtonModel` (UT3Tangent-valued gradient/Hessian; value-hashed jit aux → compile-once).
  Worked examples: `examples/fit_hilbert_uniform_{newton_cg,probe_derivatives_newton_cg}.py`. History:
  `dev/archive/uniform_optimizers_plan.md`. Live status: `dev/HANDOFF.md`.
- **Deferred / broken:** the **weighted layer** (parked `absorb_weights`) — deferred past 1.0. Remaining
  `OLD_test_*.py` stray files are dead/superseded and slated for the **R6** cleanup (delete only after
  confirming functionality is preserved elsewhere; `OLD_orthogonalization.py` was deleted 2026-07-11
  after Nick confirmed supersession).

## Open questions / TODO

Live roadmap + next steps: **`dev/HANDOFF.md`**. The durable open items:

- **Uniform layer + optimizers — DONE (the 1.0 centerpiece), frontend included.** The tangent/manifold layer
  (increment 3b), the optimizers/fitting on it (U1–U6 + U5.6), and the **frontend surface (U7)** are built and
  tested: `optimizers.*` / `fitting.*_model` infer ragged-vs-uniform from `x0`, `fitting.UniformGaussNewtonModel`
  is the roll-your-own surface, and worked examples exist. Slicing history: `dev/archive/uniform_optimizers_plan.md`.
  **The uniform layer is closed; the naming pass is DONE; next is the doc pass** (below).
- **Redesign the weighted tensor-network** code structure (deferred past 1.0).
- **Doc pass (Track B / release):** fold the design rationale from `docs/` into user-facing Sphinx
  docs; fix the docs build (`conf.py` autoapi excludes core modules; committed `_build`; `modules.rst`
  still titled "TuckerTensorTrainTools").
- **Public API + naming review — DONE (2026-07-11).** The `basis`/`base` → `frame` rename
  (`dev/naming_review.md` §2), the full naming pass + backend module reorg (plan + inventory:
  `dev/naming_pass_plan.md`; the sampling modules are grouped **by type** — Nick's call — with the
  family carried by function prefixes), and the curated `t3toolbox/__init__.py` public surface.
  **The user-facing conventions catalog (grammar, module map, semantic markers, deliberate
  exceptions) is [`docs/naming_conventions.md`](docs/naming_conventions.md)** — read it before
  naming anything new.
- **Goal-1 `fit(...)` facade** — auto geometry/optimizer/ranks/`x0` + rank-continuation/validation
  ("standard user, no fiddling"). **Deferred to 1.1**; 1.0 ships as an honest mid-level toolkit.
- **Cleanup backlog:** `OLD_*.py` (delete only once functionality is confirmed preserved elsewhere);
  wire doctests into CI; add a test CI workflow (pytest + numpy 1.x/2.x matrix); README + `CHANGELOG`
  + `pyproject` fixes (`readme = "README.md"`). Remove the "WORK IN PROGRESS DO NOT USE" banner **only
  at the moment of shipping**. **No auto-formatter near the deliberately-nonstandard code style.**
- **Minimal-rank audit** (mostly resolved): gauge projections + `project`/`project_dense_onto_tangent`
  need orthogonality only (minimal rank NOT required — confirmed); `inner`/`norm` HS-faithfulness needs
  orthogonal + minimal + gauged; `retract` preserves frame ranks only on a minimal frame.
- **Deferred niceties:** the ambient derivative transpose (exponential-rank, no use case); per-test
  seeding → `pytest -n auto` parallelism; trimming `test_dispatch` jit time + the SVD-truncation grids.
