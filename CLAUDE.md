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
- Env: conda env **`t3toolbox`** (Python 3.11; modern stack — numpy 2.x, scipy, jax 0.10, optax, chex;
  the full suite is green on it, ~3.5× slower than the old env due to jax 0.10's jit). The older `tttt`
  py3.9 env (numpy 1.22 / jax 0.4.30, **no optax**) still works for everything except the optax example.
  **The library is forward-compatible across both** (numpy 1.22↔2.x, jax 0.4.30↔0.10). Run things with the
  env's python explicitly, e.g. `/home/nick/miniconda3/envs/t3toolbox/bin/python` (no need to
  `conda activate`). **`pip install` in `t3toolbox` must use `--only-binary=:all:`** — the inherited
  `CC`/`CXX` point at `tttt`'s (broken) compiler, so source builds fail; wheels sidestep it. JAX available.
  Authors: Nick Alger, Blake Christierson.

## The paper (`t4s.pdf` in repo root)

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
> to get **wrong** (e.g. the base-inner stack-axis bookkeeping, a depth-aware tree zip), it belongs in
> the backend so they can find and reuse it. **Exception:** logic *inseparable* from the frontend —
> constructing/validating the `T3*` OO classes (the `.validate()` contracts), the same-frame guard (it
> needs the two `T3Basis` objects) — stays in the frontend regardless, since the backend cannot depend on
> the frontend.
> **Corollary:** don't leave a backend user one fiddly step short of a usable result — if a backend
> function would otherwise force them to know some follow-up call (e.g. `tree_zip` to pair two returned
> trees), fold that step in so the function returns the directly-usable thing.
> **Corollary (the test is knowledge, not line count):** a one-line wrapper *earns* its place in the
> backend when it encodes a non-obvious **capability** — its value is the name + docstring + test, not
> the code. A trivial-*and*-obvious one-liner (`return a + b`) stays inline; a trivial-*but*-non-obvious
> one belongs in the backend, because a user who doesn't know the trick can't "just rewrite it." Example:
> `probing.{apply,entries,probe}_corewise_transpose` are each a one-line substitution into the *tangent*
> transpose (`P,Q,O → G`, paper §6.3), but that substitution is the non-obvious part — inlining it in
> the frontend would hide the corewise capability from backend users entirely.
- `TuckerTensorTrain` (`tucker_tensor_train.py`) — the keystone; `.data = (tucker_cores, tt_cores)`.
- `T3Basis` / `T3Variations` (`basis_variations_format.py`) — orthogonal frame + tangent direction.
- `T3Tangent` (`manifold.py`) — bundles `(T3Basis, T3Variations)`: a tangent vector.
- `backend/` — stateless functions, each module with its own `__all__`.

**Three representations** (the organizing principle):
- **ragged** — tuples of variably-shaped arrays. The default, fully working path.
- **uniform** — one stacked supercore array + masks (`ut3_*`, `ubv_*`, `uniform_*`); for
  `jax.lax.scan` vectorization. **`UniformTuckerTensorTrain` (the plain layer) is built through slice 8
  and jit-wired with host-numpy masks** (uniform basis/variations/tangents still deferred). **Before
  touching uniform code, read the design notes** — governing: [`docs/uniform_equivalence_contract.md`](docs/uniform_equivalence_contract.md)
  (the uniform layer is a *faster ragged layer*: `to_uniform → op → to_ragged == op_ragged` on real
  parts, garbage don't-care — this is correctness *and* the test strategy). Then:
  [`docs/uniform_ranks_and_varieties.md`](docs/uniform_ranks_and_varieties.md) (a
  stacked uniform T3 is a batch in the bounded-rank **determinantal variety** — ranks may vary per
  stack element, shape is fixed), [`docs/uniform_supercore_layout.md`](docs/uniform_supercore_layout.md)
  (core index `d` **leads**: `(d,)+stack_shape+(...)`, for `lax.scan` + locality),
  [`docs/uniform_masks_vs_ranks.md`](docs/uniform_masks_vs_ranks.md) (rank metadata is **boolean
  masks**, not integer ranks — closed under add=concat / multiply=Kronecker with no data movement), and
  [`docs/uniform_pytree_composition.md`](docs/uniform_pytree_composition.md) (`UT3 = tucker_supercore +
  tt_supercore + masks`-holder; the holder is an `eq=False` identity-hashed static `aux_data`, the
  `T3Basis`↔`T3Tangent` pattern).
- **weighted** — cores + edge-weight vectors (`wt3_*`, `weighted_*`); weights "absorbed" into cores.
  Tangent weighting (`absorb_weights_into_tangent_cores`) is **parked** in `backend/bv_operations.py`
  pending a redesign of weighted tensor networks.

> **Batching/stacking is the most error-prone part of the library. Before touching anything with
> batch/stack axes, read [`docs/batching_and_stacking.md`](docs/batching_and_stacking.md)** — start with
> its **"Start here"** section (one-screen mental model + a concrete shape table + the shape-notation
> legend), then the numbered sections for the full reference (three meanings of "stack", the base-inner
> convention and *why*, the `'...'`-vs-grouped-contraction machineries, heterogeneous-stack tuples,
> `vmap`/`jit` with the basis as a pytree leaf). The notes below are the terse version.
>
> The three batch **blocks** (mnemonic-and-collision-free letters): **`C`** = base/core stack (base
> points, on *every* core; `= stack_shape`), **`W`** = probe stack (the `w` vectors, on `ww` only),
> **`K`** = tangent stack (the `k` tangent vectors at *one* base, on the *variations* only). Order is
> base-inner **`W + K + C`**. **The most common slip is conflating `C` (base points) with `K` (tangent
> vectors at a base)** — they live on different operands and mean different things. (These were `F`/`V`/`G`
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
on *different* operand subsets (canonical case: core/base stack `C` on the cores vs probe stack `W` on
`ww` only — a single `'...'` can't express it) → the named grouped-block contractions in
`backend/contractions.py` (`WCa_Caib_WCi_to_WCb` etc.; each capital block reshaped to one flat axis,
= 1 when empty). **Convention (library-wide, base-inner): core stack `C` innermost, extra stacks
`W`/`K` outermost** (`W+C`, `K+C`, `W+K+C`) — because `'...'`-broadcast replicates a base over the
extras for free only when `C` is innermost. (`apply`/`entries` were the last `C+W` holdout — flipped
in 5b; the whole library is now base-inner.)

**The three sampling operations** (`entries`, `apply`, `probe`) — evaluate the dense tensor without
forming it, differing in how many modes stay free (`probe`: one → `d` vectors; `apply`/`entries`:
none → scalar) and the test vectors (general vs one-hot). Containment `probe ⊃ apply ⊃ entries`; all
three have tangent (Riemannian `𝒥`/`𝒥ᵀ`) forms. **What differs, costs, and how each fits a Riemannian
solve:** [`docs/entries_apply_probe.md`](docs/entries_apply_probe.md). Probing is the original
exemplar; `apply`/`entries` are the general-purpose all-modes special cases.

**Two gotchas:**
- **Canonical core-tuple orderings (frontend takes precedence).** `TuckerTensorTrain.data =
  (tucker_cores, tt_cores)`; `T3Basis.data = (up, down, left, right) = (U, O, P, Q)` (the `O`/down core
  is called **`down_tt_cores`**, not "outer"); `T3Variations.data = (tucker_variations, tt_variations)`;
  `(basis, variations)` pairs are basis-first. All verified backends (`tangent_operations`,
  `bv_conversions`, `orthogonal_representations`, **and now `probing`**) take these tuples in this
  exact order — pass `.data` straight through, no reorder. (Only the parked `bv_operations.py`
  `absorb_weights` still uses the old `(up, left, right, outer)` probing order.)
- **`corewise_dot`/`corewise_norm` collapse EVERY axis** (stacks included) to a scalar. To keep the
  stack (vectorized linalg), use `corewise.corewise_stack_dot(X, Y, n_stack)`.

**Backend dispatch**: `xnp, xmap, xscan = get_backend(is_uniform, use_jax)` (`backend/common.py`).
`xnp` = numpy or jax.numpy; `xmap`/`xscan` = ragged loops / numpy / `jax.lax`.

**numpy-vs-jax dispatch convention: infer it from the input array types at the lowest level — do NOT
thread `use_jax` params around.** Each operation computes `use_jax = tree_contains_jax((its inputs))`
(or `is_jax_ndarray(...)`) and the jax-ness propagates through the computed intermediates (so a
delegating function needn't infer at all — its callees do, from the arrays passed down). Applied
across the verified code: `probing`, `manifold`, `basis_variations_format`, `corewise`, and their
backend deps (`tangent_operations`, `bv_conversions`, `t3_operations`, `t3_linalg`,
`t3_orthogonalization`, `linalg`) — **operations carry no `use_jax`**.
**The exception: pure constructors with NO array inputs** — `TuckerTensorTrain.randn/zeros/ones` and
`load`, `t3_corewise_randn`/`t3_zeros`/`t3_ones`, `common.randn`, and the rank-spec helpers in
`ranks.py` — keep a `use_jax` flag (there's nothing to infer from; it chooses the output type).
Factories that DO take an existing object infer from it (e.g. `T3Tangent.zeros/randn` from the basis,
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
  mismatched stack shapes — these raise unconditionally. `TuckerTensorTrain`, `T3Basis`, and
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
  checks. Master plan: [`docs/safe_unsafe_mode_plan.md`](docs/safe_unsafe_mode_plan.md); the full
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
    residuals (`T3Basis.orthogonality_residual`, `T3Tangent.gauge_residual`) so a fixed frame/tangent in
    an inner loop is contracted once. **Frontend-only** for now (a backend mirror for raw-`.data` users
    is deferred).

  - **"Same tangent space" is NUMERICAL, not structural** (the key correction). Two tangents share a
    tangent space iff their frames are *numerically equal* (`safety.frames_equal`, with an `is`-identity
    fast path) — **not** object identity. The old `self.basis is other.basis` identity guard was a
    numerical check faked as structural; it forced `T3Tangent`'s basis to be jax **aux_data** (→ jit
    recompiled on every base change) and false-failed on a jit round-trip. The numerical guard let the
    basis become a pytree **leaf** (base flows as traced data → no recompile). Likewise `GaussNewtonModel`
    is a registered pytree (base/sweep/sample/residual leaves; geometry/kind aux) — you can `jit` the
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
- `unittest`, in `tests/`. Pattern: `subTest` over structures × stack_shapes. **Numerical
  correctness is checked numpy-only** — the backend is backend-agnostic (`xnp`, `'...'` einsums,
  inferred dispatch), so jax computes the same numbers; duplicating every sweep in jax was wasted
  time. **jax invocation is covered separately by `tests/test_dispatch.py`** (which jit-compiles each
  op — a stray `np.*` on a tracer raises, proving no hidden numpy — plus a jax-in→jax-out check for
  dynamic-shape rtol/atol ops, plus a few numerical smoke tests). This cut the full suite from ~550s
  to ~50s. When adding a numerical test, write it numpy-only and add the op to `test_dispatch` if its
  jax dispatch isn't already covered. (Frozen dataclasses are registered jax pytrees; `T3Tangent` carries
  the **basis as a leaf** — the same-frame guard is a *numerical* `frames_equal` check that survives the
  jit round-trip, so the base flows as traced data with no per-base recompile — see `manifold.py`.)
  A few tests still build explicit jax operands where that *is* the thing under test (e.g.
  `test_contains_jax`); those keep the `jnp` import.
- **Doctests = reproducible examples** (not yet wired into CI, but written so they can be): **examples
  first** (teach the API), not coverage. Seed (`np.random.seed(0)`) or fixed inputs; value-match via
  `np.allclose`; print **structure** (shapes/ranks) not raw values; show **gotchas** (structural →
  traceback `+IGNORE_EXCEPTION_DETAIL`; numerical → wrong-vs-right). One distinct behavior per option
  (no cross-product); long tail → prose. **Run the example and paste the real output — never hand-write
  it.** Full convention + exemplar (`manifold.py`): **[`docs/doctest_style.md`](docs/doctest_style.md)**.
  (Supersedes the old "illustrative captured values" convention.)
- **Running tests/scripts**: use the env's python directly, e.g.
  `PYTHONPATH=$PWD /home/nick/miniconda3/envs/t3toolbox/bin/python -m pytest tests/ -q` (scripts run from
  `/tmp` need `PYTHONPATH=/home/nick/repos/T3Toolbox`). The optax example needs the `t3toolbox` env.

## Workflow (how Nick likes to work)

- **Incremental slices with discussion between steps.** Nick drives the design: propose the plan and
  the genuine decisions, confirm, then implement. Slice big restructures into reviewable units.
- **When *designing* a format/API, reason from the math/algorithms first.** When Nick says so,
  explicitly set aside consistency-with-existing-conventions and code-change cost — design the
  *correct* thing, then change the code to match (he'll pay for big refactors). Treat the library as
  **general-purpose**: do NOT privilege the T4S paper's optimization tasks (Riemannian fitting/CG) as
  "the" use case — a batched or exotic operation is as legitimate as a fitting one.
- **On a subtle design call, lead with the full reasoning, not a bare question.** A multiple-choice
  prompt without the first-principles argument behind it isn't useful; lay out the tradeoff (and your
  recommendation), then let Nick decide.
- **Prefer minimal dataclasses** — cores only; derive shapes/splits rather than storing redundant
  fields (e.g. the `C`/`K` stack split is recovered from the (basis, variations) pairing).
- **Commit per logical chunk and push to `main`.** Verify tests pass first; write a descriptive
  message; stage only the relevant files (leave unrelated stray edits alone). End commit messages
  with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Changing a backend convention has a wide blast radius — grep ALL consumers.** An axis-ordering or
  signature change ripples through the OO wrappers that delegate to it (e.g. `TuckerTensorTrain.probe`
  → `probe_t3`) and *their* tests/doctests, not just the file you edited. After such a change, run the
  full suite (`test_tucker_tensor_train` + `test_manifold` + `test_basis_variations_format` +
  `backend/test_contractions`), not only the directly-touched tests.
- Don't ship a possibly-wrong result with a weak test — if something looks off, dig in or flag it.

## Current state

**This is a conversion-in-progress, and most of it does not run yet.** A large fraction of the files
were copied in (or only partially edited) from the authors' prior research projects: expect stale
APIs, half-applied refactors, broken imports, and doctests whose "outputs" were captured from an
earlier draft. We are working through the codebase **file-by-file**, making each module actually run
and pass tests before moving on. **Do not assume a module works just because it exists and looks
complete — verify by running it.** Worked through and verified so far: `tucker_tensor_train.py`,
`basis_variations_format.py`, `manifold.py` (incl. the `ManifoldGeometry`/`CorewiseGeometry` layer),
`fitting.py` (the geometry-generic `GaussNewtonModel`), `safety.py` (the safe/unsafe-mode mechanism), and
the backend functions those rely on. Treat everything else as copied-in-and-not-yet-working until checked.

- **Solid / tested** — *numerical correctness in numpy*; jax **dispatch** (that jax is actually
  invoked, no hidden numpy) is covered separately by `tests/test_dispatch.py`, **not** a duplicate
  numerical sweep. So "tested" means the numbers are right and the backend dispatches — it does **not**
  guarantee every path; two latent bugs (a `t3svd` test bound, a hidden `np.einsum`) surfaced this way.
  Covered: `TuckerTensorTrain` + its backend; `basis_variations_format`; `manifold` (`T3Tangent` — full
  ragged port: linalg with the same-basis guard, gauge projections,
  `to_dense`/`to_t3`/`zeros`/`randn`/`project`/`retract`, two-axis stacking, probing (incl. the
  K-stacked forward probe and transpose via 3-group contractions), checkers).
  Tests: `tests/test_tucker_tensor_train.py`, `test_basis_variations_format.py`, `test_manifold.py`,
  `test_dispatch.py`, `backend/test_contractions.py` (suite ~45s).
- **Solid / tested — the geometry refactor + safe/unsafe mode (branch `geometry-refactor`).** A
  Manopt-style **geometry abstraction**: `ManifoldGeometry`/`CorewiseGeometry` singletons (`MANIFOLD`/
  `COREWISE` in `manifold.py`) bundle the chart choices (frame `base`, gauge `project`, `retract`, the
  HS `inner`/`norm`); one geometry-generic `GaussNewtonModel` (`fitting.py`) with `apply`/`entries`/
  `probe` factories, `gradient`/`gn_hessian`/`jacobian`/`gn_quadratic`/`evaluate`. **Safe/unsafe mode**
  (`safety.py`, S1–S6): ambient `contextvars` tolerance (safe default / `None` unsafe), numerical
  **preconditions** enforced in safe mode + skipped under unsafe/jit, two tolerances (numpy/jax),
  eager-only. The same-frame guard is numericalized (`frames_equal`), `T3Tangent`'s basis + the whole
  `GaussNewtonModel` are jax **leaves** → `jit(matvec)` compiles once across bases (no per-base
  recompile). ORTH preconditions wired into the manifold projections/retraction; inner/norm moved to the
  geometries; minimal-rank settled (not a precondition). Docs: `docs/safe_unsafe_mode_plan.md`,
  `docs/numerical_contract_catalog.md`, `docs/geometry_refactor_plan.md`. Tests: `test_safety.py` +
  the geometry/safe-mode cases in `test_manifold`/`test_fitting`/`test_dispatch`. **Remaining: G3 —
  `optimizers.py`** (the Cauchy-SGD / Newton-CG loops this enables).
- **Probing + K-stacking (`backend/probing.py`, `manifold.py`)** — done through **slice 5c**; the
  blow-by-blow is in git history, and `docs/probing_section6_notes.md` maps Section 6 of the paper
  (the Riemannian Jacobian) to the code. Delivered: forward/transpose probe
  (`T3Tangent.probe`/`probe_transpose` — bare `J^(s)`/`(J^(s))ᵀ`, no gauge `Π`); the whole pipeline
  flipped to base-inner `W+(K)+C`; K-stacked heavy ops (`to_dense`/`to_t3`/`retract`/`project` on a
  batch of tangents sharing one base); two-axis `T3Tangent.unstack_tangents`/`_basis` + `stack_*`;
  `bv_to_t3` broadcast-on-wrap (`broadcast_t3_to_common_stack`); `apply`/`entries` flipped to `W+C`;
  jax pytree registration (`T3Tangent` basis-as-aux). **The whole batch/stack design lives in
  [`docs/batching_and_stacking.md`](docs/batching_and_stacking.md) — read it before touching anything
  with stack axes.**
  - **5c (done): forward-probe a `K`-stacked tangent.** `T3Tangent.probe` now accepts a `K`-stacked
    input (`m` base points, each carrying `k` tangent vectors sharing its frame), producing probes
    stacked `W + K + C`. Implemented as a genuine **3-group contraction** (`W` probes × `K` tangents ×
    `C` base) — the earlier map-over-`K`/`jax.vmap` plan was **reversed** in favour of low-level
    einsums (consistency with the `contractions.py` toolkit, no numpy `K` loop, clean XLA folding).
    The perturbation sweep (`compute_sigmas`/`detas`, `assemble_tangent_zs`) uses 8 new base-inner
    `W+K+C` contractions; the base sweep is `K`-free. **The `K`/`C`/`W` split is recovered from
    shapes, never threaded** — a `C`-only base core pins `len(C)` and an `W+C` edge var pins `len(W)`,
    so most contractions self-infer; only the three whose sole core operand is a variation core
    (`K+C`) take an `n_base`, recomputed inline in the sweep `_func` from a base core (the `n_probe`
    precedent). They reduce to the 2-group result when `K=()`. See `docs/batching_and_stacking.md`
    §4/§7. (Future idea — exploit orthonormal-`K` structure — parked in `docs/probing_section6_notes.md`.)
  - **K-aware transpose (done): transpose a `K`-stacked forward.** `T3Tangent.probe_transpose` now
    accepts residuals carrying the tangent batch `K` (`W+K+C`, the output space of the K-stacked
    forward) and returns a tangent that carries `K`: `sum_over_probes=True` → tangent stack `K`;
    `=False` → `W+K` (base `C`). `J` and `Jᵀ` are general-purpose / independent (e.g. for `Jᵀ M J`).
    The adjoint sweep **reuses the forward's self-inferring 3-group contractions**; the assembly adds
    10 new outer-product builders (`contractions.py`, keep-`W` and sum-`W` forms — the sum-`W` ones
    generalize `WCo_WCa_to_Cao`/`Wo_WCa_to_Cao`/`WCi_WCa_WCj_to_Ciaj`). **No `K`/`C` inference in
    `probe_tangent_transpose`:** the sweep self-infers, `assemble_tucker` recovers `len(W)` from the
    `W`-only probe vectors, and only `assemble_tt` (no `W`/`C`-only operand) takes `n_probe`. `K=()`
    is exactly the prior transpose.
- **Solid / tested — the plain uniform layer**: `UniformTuckerTensorTrain` + the `ut3_*` backend, built
  through **slice 8** (foundation, orthogonalization, linalg, sampling, ut3svd, jax-wiring/host-masks,
  constructors+IO) and **jit-wired** (pytree registered; host-numpy masks; `tests/test_dispatch.py`
  `test_jit_uniform`). Live status + remaining slices: `docs/uniform_slice_handoff.md`.
- **Solid / tested — symmetric probing derivatives (branch `probe-derivatives`, NOT yet merged to
  `main`; under review).** `d^t/ds^t` of the three sampling ops (`probe`/`apply`/`entries`) in one
  repeated direction `P`, via a Taylor-jet axis + the binomial tensor `trs[t,r,s]=C(t,r)[r+s=t]`. The
  **full grid** is built in `backend/probe_derivatives.py` **and the frontend** (`TuckerTensorTrain` /
  `T3Tangent` `{probe,apply,entries}_derivatives` + their transposes): **forward** (Euclidean
  `*_derivatives_t3` + Riemannian `*_tangent_derivatives`), **tangent (Riemannian) transpose**
  (`*_tangent_derivatives_transpose`), and **corewise transpose** (`*_corewise_derivatives_transpose`,
  the §6.3 `P,Q,O→G` substitution → core gradients). **Full `W+K+C` stacking, base-inner** (order `t`
  outermost; sample stack `W` = the paired `(X,P)` samples = probing's probe stack; tangent stack `K`;
  base stack `C`) — mirrors regular probing via **order-threaded 3-block `(W,K,C)` contractions** in
  `contractions.py`. The transpose is the **jet-ified adjoint-state Lagrangian** (replace each
  contraction by its `trs` version; stationarity → adjoint-hooked-`trs` sweeps + order-less assembly
  whose tensor arity = the core's internal-edge count); apply/entries use the cheaper **`ρ`-seeded
  `sigma_hat` sweep** (`compute_sigma_hat_jets`). The forwards hard-error on an `X`/`P` stack mismatch
  (`check_perturbation_{vectors,index}`). Verified vs the dense subset-expansion oracle,
  `jax.linear_transpose`, the adjoint identity, and `jax.grad` (~1e-16, across `W/K/C/order`). The
  **ambient** transpose is **deferred** (inherently exponential-rank — `docs/ambient_derivative_transpose_note.md`).
  Math note: `docs/symmetric_probe_derivatives.tex`; plan/handoff: `docs/derivatives_mirror_plan.md` +
  `docs/probe_derivatives_handoff.md`. Tests: `tests/test_probe_derivatives.py`,
  `backend/test_contractions.py`, `test_manifold`/`test_tucker_tensor_train` (frontend), `test_dispatch`
  (jit). Naming: forward `*_jets`, transpose adjoints `*_tilde_jets`, the seeded apply sweep `*_hat_jets`;
  sample stack `W` (not `S`), order-count `order` (not `K`).
- **Deferred / broken**: the uniform **tangent** layer (`ubv_*`, `uniform_*` — uniform basis/variations/
  tangents; every `is_uniform` branch in the tangent code was dropped/stubbed). The weighted layer
  (parked `absorb_weights`). `OLD_*.py` files are still tracked.

## Open questions / TODO

- **Symmetric probing derivatives — remaining (branch `probe-derivatives`; plan
  `docs/derivatives_mirror_plan.md`).** The feature is **functionally complete**: apply/entries
  derivatives, `K`-stacking, tangent + corewise transposes, and the full frontend are all done and
  tested (see "Current state"). Remaining: (1) **doc refresh** — `entries_apply_probe.md` (stale §4
  table + add the derivative dimension), `symmetric_probe_derivatives.tex` (add apply/entries + `K`);
  (2) **review + merge to `main`**. **Deferred:** the **ambient** transpose (no use case, exponential-rank
  — `docs/ambient_derivative_transpose_note.md`); the project-once gather optimization. **Deferred
  nice-to-haves (Nick's call):** the **Hessian-conditioning experiment**
  (`docs/derivative_order_information_and_conditioning.md`). *(The derivative **fitting example** is
  done: `examples/fit_hilbert_from_apply_derivatives.py` fits from apply-derivatives via Manifold Cauchy
  SGD — T4S §5.3.2, the tuning-free Cauchy-step stochastic optimizer; ~8× faster than full-batch
  Newton-CG on that demo.)*
- **The transpose grid — DONE & REDESIGNED (ragged).** Each sampling op (`entries`/`apply`/`probe`)
  now has **three** transposes — **ambient / corewise / tangent** — read **[`docs/transposes.md`](docs/transposes.md)**
  (taxonomy, costs, decision guide) and the work log [`docs/transpose_redesign_handoff.md`](docs/transpose_redesign_handoff.md).
  - **ambient** = the literal base-free adjoint, returned as **canonical (CP) factor tuples** (the
    natural type for a multilinear adjoint; convert to a T3 via `from_canonical`):
    `TuckerTensorTrain.{apply,entries,probe}_ambient_transpose` (static). `apply`/`entries` are rank-1
    (or rank-`|W|` summed); `probe` is rank-`d`. Keeps `sum_over_probes` (cheap in CP — the `|W|²`
    dense-T3 cost lives only in `from_canonical`). *(These are the renamed old `*_transpose` methods,
    which used to wrap into a T3 with a copy-tensor — that `O(|W|³)` path is gone.)*
  - **corewise** = gradient w.r.t. the cores (Adam/L-BFGS), raw `(tucker_grads, tt_grads)`:
    `TuckerTensorTrain.{apply,entries,probe}_corewise_transpose` (instance, `self` is the base). The
    paper §6.3 substitution (`P,Q,O → G`) into the tangent backend; `c` must be an array.
  - **tangent** = Riemannian gradient, a `T3Tangent`: `T3Tangent.{apply,entries,probe}_transpose`
    (unchanged; `sum_over_probes=False` keeps `W`, `=True` is `Jᵀr`). Older history:
    [`docs/apply_entries_handoff.md`](docs/apply_entries_handoff.md), `docs/batching_and_stacking.md` §11.
- **A least-squares fitting example/tutorial — wanted, deferred (Nick's call).** Show `apply`/`entries`
  as the forward sampling operator `J` and the summed transpose (`sum_over_probes=True`) as the gradient
  `Jᵀr` and Gauss-Newton Hessian `JᵀJ v` — the worked use case that motivates `sum_over_probes=True`.
  Blocked on bringing Nick's optimization code into the project (cleanup + tests = substantial); come
  back to it later. This is the deferred "S4" follow-up to the §11 transpose docs.
- **Which ops require a minimal-rank basis** (partly answered, full audit pending): gauge
  projections need orthogonality only; `inner`/`norm` Hilbert-Schmidt faithfulness needs orthogonal
  + minimal + gauged; `retract` preserves base ranks only on a minimal base; `project`/
  `project_dense_onto_tangent` works on any orthogonal base — **minimal rank NOT required, CONFIRMED**
  (the exact orthogonal projection holds for non-minimal bases; a flaky test that suggested otherwise
  was a fragile `pinv` *oracle*, not a code bug — see the `pinv-oracle-test-fragility` memory).
- Repair the **uniform layer** — **slices 1–8 DONE** (`UniformTuckerTensorTrain` + `ut3_*`: foundation,
  orthogonalization, linalg, sampling, ut3svd, jax-wiring/host-masks, constructors+IO), rebuilt
  function-by-function (hybrid backend). The host-numpy-mask design + jit story live in the five
  `docs/uniform_*.md` notes (see the **uniform** bullet under Architecture); live status in
  [`docs/uniform_slice_handoff.md`](docs/uniform_slice_handoff.md). **Remaining:** slice 9 (**t3m**); the
  **uniform transpose mirror** (ambient doable now — `from_canonical` just landed; corewise/tangent gated
  on the tangent layer); and the deferred **uniform tangent layer** (`ubv_*`, `uniform_*`).
- Redesign the **weighted tensor network** code structure.
- **⚠️ MUST FIX — numpy-2.0 doctest breakage (surfaced by the `t3toolbox` env move, 2026-06-20).** numpy 2.x
  changed scalar reprs (`np.False_`/`np.True_` not `False`/`True`; `np.float64(x)` wrappers), so doctests
  that **print scalars** now fail on the modern env **even though the test suite is green** (doctests aren't
  wired into CI, so 295-pass does NOT cover them). Confirmed failing: `corewise.corewise_logical_not`
  (prints `np.False_`). **There are almost certainly more across the codebase.** Needs a numpy-2.0 doctest
  sweep: rerun `python -m doctest <module>` per module on the `t3toolbox` env and fix printed-scalar reprs
  (print via `bool(...)`/`float(...)`, or update expected output). **Do not let this slip.**
- Cleanup backlog: `OLD_*.py` + stray `.npz` artifacts; wire doctests into CI; docs (`conf.py` autoapi
  excludes backend/weighted, committed `_build`, `modules.rst` still titled "TuckerTensorTrainTools").
- **Doctests — existing-doctest sweep ✅ DONE.** All verified modules' **existing** doctests reworked
  into reproducible examples (convention: `docs/doctest_style.md`; exemplars `manifold.py` +
  `TuckerTensorTrain.__mul__`/`inner`/`t3svd`) and committed: `manifold`/`corewise`/`stacking` (already
  conformed), `linalg`, `backend/probing`, `backend/dense_t3svd`, `basis_variations_format`,
  `tucker_tensor_train` (161 failing → 0; `python -m doctest` clean per module). The sweep doubled as a
  stale-code detector — it found+fixed wrong captured values and dead imports/`NameError`s the broken
  doctests hid (incl. a nonexistent `t3.t3_corewise_randn` used across ~12 `tucker_tensor_train`
  examples). **Remaining doctest work (deferred, Nick wants it): add default-path doctests to currently
  *undocumented* public functions** — a separate pass (`docs/doctest_handoff.md`); the `linalg` probe
  already did this for `linalg`'s `*_svd`/`*_svd_pair`/`pad_or_truncate`. **Flagged for follow-up** (out
  of the doctest scope, no behavior change): `TuckerTensorTrain.core_shapes` (property) strips the stack
  while `get_core_shapes()` (static) includes it — an apparent inconsistency worth a look.
- **T3M — elementwise multiply with truncation. ✅ DONE & tested.** `TuckerTensorTrain.t3m(other,
  method=..., max_tucker_ranks, max_tt_ranks, rtol, atol, oversample=1)` — three interchangeable
  algorithms generalizing the TTM algorithm (Michailidis et al., arXiv:2410.19747) to T3, all matching
  the dense oracle: **(a)** `t3m_form_then_round` (form full product → round; `*` uses this exact path),
  **(b)** `t3m_inplace_fused` (fused L→R sweep; the `t3m()` default), **(c)** `t3m_swap` (the `r≫d`
  specialist; gauge-managed truncating swaps + KR merge + `oversample`/`t3svd`-cleanup for the
  Tucker leaf-frame tension). Spec: `max_*_ranks` scalar-or-sequence, per-step `rtol`/`atol` (require
  unstacked; max-rank is stacking-OK), SVD-everywhere, **joint** truncation. `oversample` (method='swap'
  only, default 1=off; try 2): trades a little memory for near-(a) quality and is what honors a
  per-position `max_tt_ranks` sequence in (c). Tests `tests/test_t3m.py` + `test_dispatch` jit cases.
  Design/status: **`docs/t3m_plan.md`**; build details **`docs/t3m_swap_plan.md`**; the TTM↔T3M↔HT
  theory (why oversample is forced, why convert-to-balanced-HT doesn't help) **`docs/ttm_t3m_ht_note.tex`**.
- **Further test-speed options (deferred; suite is already ~50s after the numpy-only refactor, so
  low priority):** (1) **per-test seeding → parallelism** — tests share one global `np.random` seeded
  once at import (the source of the t3svd RNG-order flakiness we hit); seed per-test, then run in
  parallel (`pytest -n auto`) for a ~cores× speedup; (2) trim `test_dispatch`'s jit-compile time
  (~12s) — fewer ops, or drop x64 there (it's an invocation check, not a precision one); (3) trim the
  remaining rtol×atol×rank-limit grids in the SVD-truncation tests (`test_tucker_svd_dense`,
  `test_ttsvd_dense`, `truncated_svd`, `t3svd_dense`) to representative combos (each ~16 combos).
