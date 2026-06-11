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
   Every backend einsum uses a leading `'...'` so these ride along for free — the vectorization
   mechanism. "Add stacking to a function" = rewrite its einsums/concats with `'...'`/negative axes.
2. `backend/stacking.py`: convert a Python tree of separate objects ↔ one stacked object
   (`stack`/`unstack`, `tree_zip`, `apply_func_to_leaf_subtrees`).
3. the uniform supercore (the separate, deferred representation).

**Backend dispatch**: `xnp, xmap, xscan = get_backend(is_uniform, use_jax)` (`backend/common.py`).
`xnp` = numpy or jax.numpy; `xmap`/`xscan` = ragged loops / numpy / `jax.lax`.

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
- `unittest`, in `tests/`. Pattern: `subTest` over structures × stack_shapes × `[False, True]`
  (numpy/jax). JAX **x64 is enabled in test files**, so `tol=1e-9` works for jax; without x64 jax is
  float32 and gauge/orthogonality residuals are ~1e-6.
- Doctests are NOT wired into the runner — illustrative (captured) values are the convention;
  deterministic outputs (shapes, ranks, exact `0.0`) should still be accurate.
- **Run tests filtering debug noise**:
  `python -m unittest tests.test_X 2>&1 | grep -vE "^(RAGGED|NUMPY)"`
  (`common.py`'s ragged_map/scan print `RAGGED MAP` / `NUMPY SCAN(` etc. — leftover debug prints,
  not yet removed). Scripts run from `/tmp` need `PYTHONPATH=/home/nick/repos/T3Toolbox`.

## Workflow (how Nick likes to work)

- **Incremental slices with discussion between steps.** Nick drives the design: propose the plan and
  the genuine decisions, confirm, then implement. Slice big restructures into reviewable units.
- **Commit per logical chunk and push to `main`.** Verify tests pass first; write a descriptive
  message; stage only the relevant files (leave unrelated stray edits alone). End commit messages
  with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
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
- **In progress — `backend/probing.py` (current focus)**: the math matches the paper (Section 6, see
  `docs/probing_section6_notes.md`), but the tangent path does **not** run yet. `probe_tangent` /
  `probe_tangent_transpose` and their helpers thread `use_jax=` into `compute_xis/mus/nus/etas` and
  `_apply_edge_weights`, which were refactored to *infer* `use_jax` from the array tree and no longer
  accept the kwarg (`probe_t3` path works; tangent path raises `TypeError` on the first call).
  Doctests reference stale APIs (`orth.orthogonal_representations`, `t3m.tangent_randn`, etc.).
  House convention to harmonize to (see `tangent_operations.py`): `use_jax = use_jax or
  tree_contains_jax((...))` inside each function.
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
