# T3Toolbox — project guide

A working reference for collaborating on this codebase. Read this first.

## What this is

Pure-Python (NumPy + optional JAX) library for **Tucker tensor trains (T3)** — a Tucker
decomposition whose central core is stored as a tensor train. Nick Alger and Blake Christierson are converting research
code into a standalone package: cleaning up, documenting, restructuring for usability, adding
features. The README still says "WORK IN PROGRESS DO NOT USE."

- Repo: `github.com/NickAlger/T3Toolbox` (renamed from `TuckerTensorTrainTools`; that rename left
  stale references we've mostly fixed). Branch `main`, direct commits.
- Env: Python 3.9, conda env `tttt`, JAX available. Authors: Nick Alger, Blake Christierson.

## The paper (`t4s.pdf` in repo root)

The math is in *Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series"
(arXiv:2603.21141)* — "T4S". **Appendix A** is the reference for the manifold/tangent code.
Notation map (paper ↔ code): `U`=up_tucker, `P`=left_tt, `Q`=right_tt, `O`=down_tt (outer);
`δU`=tucker_variations (V), `δG`=tt_variations (H). The code matches the appendix; the one known
divergence is the **Algorithm-11 orthogonalization sweep order** (code does left-then-right; the
paper does right-then-left) — the resulting orthogonal representations are equivalent. When code
implements a numbered equation/algorithm, cite it in the docstring.

## Architecture

**Thin OO frontend over a pure-functional backend.**
- Frozen dataclasses hold only the cores; everything else is a `@cached_property`/method that
  delegates to `backend.*` functions operating on raw `.data` tuples, then re-wraps the result.
- `TuckerTensorTrain` (`tucker_tensor_train.py`) — the keystone; `.data = (tucker_cores, tt_cores)`.
- `T3Basis` / `T3Variations` (`basis_variations_format.py`) — orthogonal frame + tangent direction.
- `T3Tangent` (`manifold.py`) — bundles `(T3Basis, T3Variations)`: a tangent vector.
- `backend/` — stateless functions, each module with its own `__all__`.

**Three representations** (the organizing principle):
- **ragged** — tuples of variably-shaped arrays. The default, fully working path.
- **uniform** — one stacked supercore array + masks (`ut3_*`, `ubv_*`, `uniform_*`); for
  `jax.lax.scan` vectorization. **Currently broken / deferred.**
- **weighted** — cores + edge-weight vectors (`wt3_*`, `weighted_*`); weights "absorbed" into cores.
  Tangent weighting (`absorb_weights_into_tangent_cores`) is **parked** in `backend/bv_operations.py`
  pending a redesign of weighted tensor networks.

**"Stacking" means three different things** — keep them straight:
1. `stack_shape`: leading batch axes on one object's cores (`core.shape = stack_shape + (...)`).
   A leading `'...'` rides these along for free — **but only ONE batch block** (one shared/broadcast
   prefix). "Add stacking to a function" = rewrite its einsums/concats with `'...'`/negative axes.
2. `backend/stacking.py`: convert a Python tree of separate objects ↔ one stacked object
   (`stack`/`unstack`, `tree_zip`, `apply_func_to_leaf_subtrees`).
3. the uniform supercore (the separate, deferred representation).

**The custom-contraction toolkit (`backend/contractions.py`)** — when **two** independent batch
blocks live on *different subsets* of operands, a single `'...'` can't express it (right-aligned
broadcasting would force the two blocks to align). The canonical case is probing: the **core/base
stack `G`** (on the cores) and the **probe stack `F`** (on the probe vectors only). So probing is
built on named grouped-block contractions — `inputs_to_output` with a capital letter per grouped
block (e.g. `FGa_Gaib_FGi_to_FGb`): each block is reshaped to one flat axis (`math.prod(shape)`, = 1
when empty, so no-stack / one-stack / both-stack collapse to the same code), einsum'd with the
capitals, then reshaped back. **Stacking-axis convention (library-wide, base-inner): the core/base
stack `G` is innermost (adjacent to the indices); extra stacks — probe `F`, tangent `V` — are
outermost** (`F+G`, `V+G`, `F+V+G`). Why: the `'...'`-broadcast ops (`to_dense`, gauge, linalg)
replicate a base over the extra axes for free only when `G` is innermost; the custom contractions
are flops-neutral to order, so they follow the same convention for one consistent layout (no
boundary-transpose copies). (`apply`/`entries` are the lone holdout still on `G+F`.)

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
`from_tensor_train`/`from_canonical` from the cores). The deferred uniform/weighted layers still
thread `use_jax` (old pattern) — migrate when repairing them.

## Code style (deliberate and nonstandard — do NOT normalize)

- Shape/structure comments are **trailing comments on the same line** as each array argument and
  each return-type element; put **one array per line** (expand return tuples even when they'd fit),
  and **vertically align** argument names with their type annotations.
- Body locals encode axis layout in the **name suffix** (`G_aib`, `mu_XIa`, `B0_b_j_c`), matching the
  contraction-naming scheme (`G`/`F` = grouped index blocks, lowercase = single axes, leading `d` =
  stacked/derivative axis; functions named `inputs_to_output`, e.g. `GFa_Gaib_GFi_to_GFb`).
- `math.prod` (not `np.prod(..., dtype=int)`) for static products of shape ints.
- einsum everywhere with a leading `'...'`; numpy path passes `optimize=path`, jax path omits it.
- Uppercase single-letter core names (`U V G P Q O …`) are intentional — ignore "should be
  lowercase" linter notes.

## House philosophy: hard guards for structural problems, warnings for numerical ones

The dividing line is **structural vs numerical**:

- **Structural problems → hard error.** Wrong shape, inconsistent ranks/lengths, mismatched stack
  shapes, operating across different tangent spaces — these raise. `TuckerTensorTrain`,
  `T3Basis`, and `T3Variations` validate in `__post_init__`; `T3Tangent` `+`/`-`/`inner` require
  the **same `T3Basis` object** (identity, not value equality, which would be ambiguous on ndarray
  fields). If something is the wrong shape: error.

- **Numerical problems → warn, don't enforce.** If something is supposed to be orthogonal / gauged /
  minimal-rank but isn't, let it through and let the result be wrong — do not check at construction.
  Instead provide non-enforcing checkers (`is_orthogonal`, `is_gauged`, `has_minimal_ranks`) and
  document the requirement + failure mode in the operation's docstring, often with an illustrative
  doctest showing it failing when the property is absent.

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
  jax dispatch isn't already covered. (Frozen dataclasses are registered jax pytrees; `T3Tangent` has
  the **basis as aux_data** so the same-tangent-space identity guard survives jit — see `manifold.py`.)
  A few tests still build explicit jax operands where that *is* the thing under test (e.g.
  `test_contains_jax`); those keep the `jnp` import.
- Doctests are NOT wired into the runner — illustrative (captured) values are the convention;
  deterministic outputs (shapes, ranks, exact `0.0`) should still be accurate.
- **Run tests filtering debug noise**:
  `python -m unittest tests.test_X 2>&1 | grep -vE "^(RAGGED|NUMPY)"`
  (`common.py`'s ragged_map/scan print `RAGGED MAP` / `NUMPY SCAN(` etc. — leftover debug prints,
  not yet removed). Scripts run from `/tmp` need `PYTHONPATH=/home/nick/repos/T3Toolbox`.

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
  fields (e.g. the `G`/`V` stack split is recovered from the (basis, variations) pairing).
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
`basis_variations_format.py`, `manifold.py`, and the backend functions those three rely on.
Treat everything else as copied-in-and-not-yet-working until checked.

- **Solid / tested**: `TuckerTensorTrain` + its backend; `basis_variations_format`; `manifold`
  (`T3Tangent` — full ragged port: linalg with the same-basis guard, gauge projections,
  `to_dense`/`to_t3`/`zeros`/`randn`/`project`/`retract`, stacking, checkers).
  Tests: `tests/test_tucker_tensor_train.py`, `test_basis_variations_format.py`, `test_manifold.py`,
  `backend/test_contractions.py`.
- **In progress — probing (current focus)**: `docs/probing_section6_notes.md` maps the paper
  (Section 6, the Riemannian Jacobian) to the code.
  - **Slice A (DONE)** — `backend/probing.py` backend: removed all edge-weighting (weights are
    absorbed into cores up front, then probe unweighted); harmonized `use_jax` to the house pattern
    so the tangent path runs; kept every `is_uniform` branch; updated paper refs + doctests.
    Verified `probe_tangent` vs `probe_dense` and the adjoint identity `<z, Jv> = <Jᵀz, v>`
    (numpy/jax/stacked). `probe_tangent`/`probe_tangent_transpose` take `base = T3Basis.data`
    (`(U, O, P, Q)`) directly (the old `(U,P,Q,O)` order was reordered to match the frontend).
  - **Slice A.5 (DONE)** — double-stacking. Tangent + transpose probing now work in all four
    stacking cases (no stack / probe stack F / T3 stack G / both) via the custom **G/F contractions**
    in `contractions.py` (G = T3 `stack_shape`, F = probe batch; output ordered G then F). Raw `...`
    einsums only carried F and broke on a stacked T3; the forward path reuses existing contractions,
    the transpose adds `GFo_Gio_to_GFi` (deta_tilde; G *batched*, not compute_xis' outer form) and
    the `sum_over_probes` contractions `GFo_GFa_to_Gao`/`Fo_GFa_to_Gao`/`GFi_GFa_GFj_to_Giaj` (sum F,
    keep G; take `n_probe`). Verified: forward vs dense, adjoint identity (sum + non-sum), np+jax.
    Also: `contractions.py` now uses `math.prod` (was `np.prod(..., dtype=int)`).
  - **V-stacking rework (in progress)** — generalizing so a `T3Variations` may carry an extra
    **tangent stack `V`** (a batch of tangent vectors sharing one base) beyond the basis's base stack
    `G`. Three stacking axes total: `G` = base/core stack (on the cores), `V` = tangent stack (on
    variations only), `F` = probe stack (on `ww` only); in the transpose `V = F`. **Convention
    (decided): base-stack INNERMOST, extra stacks OUTERMOST, everywhere** — i.e. order by how
    core-bound each axis is. Variation cores = `V + G + core`; full probe ordering = `F + V + G`
    (probe_t3 = `F + G`). Rationale: the bulk of tangent ops combine a base (stack `G`) with a
    variation via raw `...` einsums, and right-aligned broadcasting replicates the base over the
    outer extras only when `G` is innermost; probing's custom contractions are flops-neutral to
    order, so flip them to match (one convention ⇒ no boundary-transpose copies). The `G`/`V` split
    is **recoverable from the (basis, variations) pairing** (`G = basis.stack_shape` is the trailing
    suffix of `variations.stack_shape`) — so **no stored field**; `T3Variations` keeps one
    `stack_shape`. Beware: stacked arrays blow up fast — keep stack dims 1–2, core dims small in
    tests. Slices:
    - **Slice 1 (DONE)** — `check_bv_pair`: equality → "`base.stack_shape` is the trailing suffix of
      `variations.stack_shape`". `+test_check_bv_pair_stacking`.
    - **Slice 2 (DONE)** — `T3Tangent`: three properties `base_stack_shape`(`G`) /
      `tangent_stack_shape`(`V`) / `stack_shape`(`V+G`); stack-aware `inner`/`norm` (return a `V+G`
      array, one value per stacked tangent) via new `corewise.corewise_stack_dot`; `zeros`/`randn`
      gain a `V` param (gauge already broadcasts the base over `V`); `add`/`sub`/`inner` require
      matching stacks.
    - **Slice 3 (DONE)** — flipped the whole probe pipeline to base-inner `F+...+G`: the 11 probing
      contractions in `contractions.py` (`GF`→`FG`), `probe_*`/`probe_dense`, order-agnostic scan
      inits. `apply`/`entries` use a DISJOINT contraction set and were left `G+F` (flipping
      `TuckerTensorTrain.apply`/`entries` is a separable follow-up; toolkit names self-document order).
      Values/tests order-invariant.
    - **Slice 4 (DONE)** — `T3Tangent.probe` (`𝒥`, instance method → probes `F+G`) +
      `T3Tangent.probe_transpose` (`𝒥ᵀ`, staticmethod taking a basis → a `T3Tangent`), passing
      `basis.data` straight through (probing's `base` order now matches `T3Basis.data`), **no** gauge
      projector `Π`. Transpose: `sum_over_probes=True` → `V=()`;
      `=False` → `V=F` stacked (wraps with no reorder, since slice-3 made the output `F+G = V+G`).
      `probe()` rejects a `V`-stacked input (needs 3-block contractions).
    - **Slice 5 (THREE independent pieces; do as separate slices in this order):**
      - **5a (DONE) — heavy tangent ops on `V`-stacked** (`to_dense`/`to_t3`/`retract`/`project`). The
        semantic model (agreed with Nick): a `V+G`-stacked tangent is a batch of tangent vectors, one
        per `(v, g)` pair; the base point is ONE object per `g`, **shared/replicated across `V`**.
        Densifying = densify each pair, stacked `V+G+(N…)`; broadcasting a base core (`G`) against a
        variation (`V+G`) IS the faithful vectorization of that sharing (base-inner ⇒ `G` is the
        trailing suffix of `V+G`, so the broadcast is unambiguous). Two code paths, fixed differently
        on purpose:
        - **`to_dense` (backend `t3_operations`) made broadcast-stack-aware** (compute the common
          `np.broadcast_shapes` of all core stacks, `broadcast_to` each core up, then contract). It was
          the lone *reshape*-based primitive that hard-assumed one shared `vs`; the `...`-einsum ops
          (gauge, `project_t3_onto_tangent_space`) already broadcast base-`G`/variation-`V+G` tuples, so
          heterogeneous-but-broadcastable backend tuples are already first-class — `to_dense` just joins
          them. **`bv_to_t3` left as a thin selector (NOT broadcast there).**
        - **`tangent_to_t3` (builder) DOES materialize base→`V+G`** (derive `ss` from a variation core;
          `broadcast_to` every base core before the concats/zeros). Intrinsic, not a workaround: its
          output is a validated `TuckerTensorTrain`, and `validate()` requires a uniform stack — a
          doubled-rank core `[U_i ; V_i]` is one array, so `U` (`G`) must be lifted to `V+G`. The
          asymmetry with `to_dense` is principled (bare-array output vs class instance). `retract`
          follows (`t3svd` stack-agnostic). **`project` already worked unchanged** on a `V`-stacked `x`.
        Tests: `test_tangent_stacked_heavy_ops` + `test_project_tangent_stacked` (compare every `(v,g)`
        slice to the unstacked reference; np+jax). (5a tests slice cores by hand because they predate
        the two-axis stack/unstack below.)
      - **Two-axis stack/unstack (DONE, separate slice — found in a `V`-stack audit of `manifold.py`
        + `basis_variations_format.py`).** The audit found three single-stack assumptions: (i)
        `T3Tangent.unstack` *crashed* on a `V`-stack (zipped a `G`-deep basis tree with a `V+G`-deep
        variations tree); (ii) `T3Tangent.stack` was *silently wrong* (over-stacked the basis to
        `V+G`); (iii) frontend `bvf.bv_to_t3` *crashes* on a `V`-stack (wraps a mixed-stack term in
        `TuckerTensorTrain`, which `validate()` rejects — same broadcast-on-wrap issue as
        `tangent_to_t3`; **fixed in the `bv_to_t3` slice below**). Everything else (`+ - * inner norm`, gauge,
        `is_gauged`, `project`, `probe_transpose`) already handles `V+G` via corewise/`…`-broadcast.
        Resolution (decided with Nick): the monolithic `stack`/`unstack` can't faithfully invert two
        stacks (the `V`/`G` split isn't recoverable from a bare tree), so **replaced them with two
        explicit pairs**, each peeling ONE named stack (so `stack_X` cleanly inverts `unstack_X`):
        - `unstack_tangents`/`stack_tangents` (peel the tangent stack `V`): a `V`-tree of tangents
          that **share the base** (`stack_tangents` reuses it, **guard: same `T3Basis` object** —
          structural identity, as in `inner`/`+`). "For each vector within the basis."
        - `unstack_basis`/`stack_basis` (peel the base stack `G`): a `G`-tree of single-base-point
          tangents at **distinct** bases (`stack_basis` places `G` *innermost* → `V+G`; needs
          interior-axis stacking, which the component `T3Variations.stack` can't do — hence it's a
          real op, not user-assembled). "For each basis."
        Pattern: backend functionals in `tangent_operations.py` (`unstack_tangent_stack`/
        `stack_tangent_stack`/`unstack_base_stack`/`stack_base_stack`, built on `stacking.unstack/
        stack(axes=…)`) do the array/tree work; the `T3Tangent` methods are thin wrappers doing the
        compatibility checks + (un)wrapping. `T3Basis`/`T3Variations` keep their single plain
        `stack`/`unstack`. Tests: `test_unstack_stack_tangents`/`_basis` (round-trip + per-slice dense,
        incl. multi-axis stacks) + `test_stack_tangents_guard`.
      - **`bv_to_t3` V-stack fix (DONE, separate slice — audit finding (iii)).** Extracted the 5a
        `to_dense` broadcast block into a reusable backend helper `broadcast_t3_to_common_stack`
        (`t3_operations.py`): `np.broadcast_shapes` of all core stacks, `broadcast_to` each core up.
        `to_dense` now calls it (behavior unchanged). Frontend `bvf.bv_to_t3` calls it on the
        mixed-stack term (base `G` + one variation `V+G`) before wrapping in `TuckerTensorTrain`, so
        the term is a valid uniform `V+G`-stack T3. Backend `bv_conversions.bv_to_t3` stays a thin
        selector (returns the mixed-stack tuple, consumed by broadcast-aware `to_dense`). Test:
        `test_bv_to_t3_tangent_stacked` (every `(v,g)` slice == unstacked term; np+jax).
      - **5b — flip `apply`/`entries` to `F+G`** (independent of V-stacking; the lone `G+F` holdout =
        `TuckerTensorTrain.apply`/`entries` + their 2 contractions `GFa_Gaib_Fo_Gio_to_GFb` /
        `GFa_Gaib_GiF_to_GFb`). Changes their public output stacking + tests/doctests. Quick.
      - **5c — forward-probe a `V`-stacked tangent** (`J` on a *batch* of tangents; currently rejected
        by the guard in `T3Tangent.probe`). Needs a 2nd private batch block (`V` on the variation,
        alongside `F` probes / `G` base). DECISION: do **map-over-`V`** first (vmap/loop the 2-block
        probe — correct, removes the guard, unvectorized); defer 3-block contractions until a perf need.
- **Deferred / broken**: the uniform layer (`ut3_*`, `ubv_*`, `uniform_*`) — many modules don't even
  import; every `is_uniform` branch in the tangent code was dropped/stubbed. The weighted layer
  (parked `absorb_weights`). `OLD_*.py` files are still tracked.

## Open questions / TODO

- **Which ops require a minimal-rank basis** (partly answered, full audit pending): gauge
  projections need orthogonality only; `inner`/`norm` Hilbert-Schmidt faithfulness needs orthogonal
  + minimal + gauged; `retract` preserves base ranks only on a minimal base; `project` works on any
  orthogonal base.
- Repair the **uniform layer** (fix `ut3_*`/`ubv_*` imports, then add supercore variants of the
  tangent ops).
- Redesign the **weighted tensor network** code structure.
- Cleanup backlog: remove the `common.py` debug prints; `OLD_*.py` + stray `.npz` artifacts; wire
  doctests into CI; docs (`conf.py` autoapi excludes backend/weighted, committed `_build`,
  `modules.rst` still titled "TuckerTensorTrainTools"); the TTM algorithm for `t3_mult` (low
  priority — a named method, needs tolerances, not for stacking).
- **Further test-speed options (deferred; suite is already ~50s after the numpy-only refactor, so
  low priority):** (1) **per-test seeding → parallelism** — tests share one global `np.random` seeded
  once at import (the source of the t3svd RNG-order flakiness we hit); seed per-test, then run in
  parallel (`pytest -n auto`) for a ~cores× speedup; (2) trim `test_dispatch`'s jit-compile time
  (~12s) — fewer ops, or drop x64 there (it's an invocation check, not a precision one); (3) trim the
  remaining rtol×atol×rank-limit grids in the SVD-truncation tests (`test_tucker_svd_dense`,
  `test_ttsvd_dense`, `truncated_svd`, `t3svd_dense`) to representative combos (each ~16 combos).
