# Uniform T3 port — handoff (after slice 5)

Resume note for the `UniformTuckerTensorTrain` port. The living plan with per-function SHARE/REWRITE
verdicts and the 9-slice order is **[`docs/uniform_port_plan.md`](uniform_port_plan.md)**; the design
philosophy is the five **`docs/uniform_*.md`** notes (read those before touching uniform code). This note
records *where we stopped*, the *one open bug to investigate first*, and the immediate next steps.

---

## ✅ RESOLVED — ragged `t3svd` non-minimal ranks under truncation (FIXED; one uniform consequence)

**Fixed.** Ragged `t3svd` now returns structurally minimal ranks for every cap pattern
(`has_minimal_ranks` always `True`). Full write-up: **[`docs/t3svd_minimal_ranks.md`](t3svd_minimal_ranks.md)**.
The original repro (`x.t3svd(max_tt_ranks=2, max_tucker_ranks=3)` on `randn((5,6,7),(4,5,6),(1,3,2,1))`)
now gives tucker `(2,3,2)`, `has_minimal_ranks=True`.

### What it was
At each mode the L→R sweep truncated the **Tucker** rank against the *current* right bond, then shrank
that bond. A **hard** `max_tt` cap forces the bond below its structural value `rL·n`, retroactively
orphaning the just-fixed Tucker rank (`n > rL·rR` → non-minimal). Symmetric on the TT side (a `max_tt`
cap left above the adjacent `tucker·bond` bound). No-truncation / `rtol`/`atol` paths were already
minimal. **This is not Tucker-specific** — plain TT-SVD with hard per-bond caps has the same orphan
(see the note).

### The fix (option (a) — lossless re-tighten)
`backend/t3_svd.py::_shrink_to_minimal_ranks`: after the sweep, a structural **right-to-left** pass
(re-SVD each Tucker edge then each bond, no cap) drops exactly the orphaned directions. **Lossless**
(verified ≤3.7e-15 vs the pre-fix natural-sweep dense) and gated behind a minimality check so the
already-minimal paths pay nothing. New tests in `tests/test_tucker_tensor_train.py`:
`test_t3svd_truncation_is_minimal`, `test_t3svd_lossless_compression_of_degenerate`,
`test_compute_minimal_ranks_matches_matricization`, `test_compute_minimal_ranks_inequalities`.

### ✅ RESOLVED — uniform `ut3svd` now matches ragged option (a) exactly
The post-fix divergence (when the **Tucker ranks are left uncapped while a TT-bond cap bites**, ragged
option (a) keeps the full Tucker rank through the bond SVD for a better approximation, while uniform's
old masked sweep truncated Tucker-first = option (b), ~1% worse) is **fixed**. `ut3svd` was reworked to
build its sweep masks from the *capped* (not minimal) ranks — so the bond SVD sees the full Tucker rank
— then re-tighten to minimal. Now **uniform == ragged exactly in tensor, ranks, AND gauge for both
`minimize_ranks` flags** (incl. no-truncation and the uncapped-Tucker case): verified 48+ cases,
0 mismatch. Also added `assume_orthogonal` (`None`/`'left'`/`'right'`, matching ragged) and the
non-enforcing `is_left_orthogonal`/`is_right_orthogonal` checkers. The re-tighten fires only on a genuine
orphan (raw-sweep content != minimal), as a cheap reverse-based re-SVD (`assume_orthogonal='left'`),
matching ragged's gauge. `test_t3svd_truncation` now asserts exact rank+tensor equality across
asymmetric/divergent caps. (Commits: option-a + `minimize_ranks`; `assume_orthogonal` + checkers;
efficient re-tighten + `compute_raw_sweep_ranks`.) Only `rtol`/`atol` remain unsupported in uniform
(intentional — data-dependent shapes).

---

## Where we are: slices 1–5 DONE, pushed to `main`

`UniformTuckerTensorTrain` (the plain-T3 analog only; uniform basis/variations/tangents still deferred).
`.data = (tucker_supercore, tt_supercore, (shape_mask, tucker_edge_mask, tt_edge_mask))`. Masks live in an
`eq=False` identity-hashed `UT3Masks` holder (the `T3Basis`↔`T3Tangent` aux_data pattern; jax pytree
registration itself is **slice 7**, not done). All frontend methods are thin wrappers over `.data → .data`
backend functions (the **backend/frontend razor**: a backend-only user must be able to do everything on the
raw `.data` tuple; only trivial one-liners stay inline).

**Tests:** `tests/test_uniform_tucker_tensor_train.py` — **30 tests, all green**. Full ragged + jax
regression still green (`test_tucker_tensor_train` 57, `test_manifold` 37, `test_dispatch` 5,
`backend/test_contractions` 29). Numerical tests are **numpy-only**; jax dispatch for the uniform ops is
**not yet covered** (deferred to slice 7).

Done per slice (verified vs ragged/dense, ≤~3e-15):
1. **Foundation** — `.data` layout, `UT3Masks`, frontend skeleton, masking (`ut3_masking`:
   `make_uniform_masks` vectorized prefix masks, `apply_masks_to_cores`), conversions (`ut3_conversions`:
   `t3_to_ut3`/`ut3_to_t3`/`ut3_to_dense`), operations (`ut3_operations`: reverse, squash, pack/unpack,
   stack/unstack). `to_dense` factored to representation-agnostic `t3_operations.t3_to_dense_chain` +
   `absorb_tucker_into_tt` (polymorphic einsum).
2. **Orthogonalization** (`ut3_orthogonalization`) — `down`/`up`/`left`/`right`, `.data → .data`, masks
   recomputed via rank recurrences (`_left/right_orthogonalized_tt_ranks`, `_prefix_mask`). Shares the
   polymorphic `backend/orthogonalization.py` TT sweeps (fixed an `if xs[0]:` truth-of-array bug there).
3. **Linalg** (`ut3_linalg`) — `ut3_scale`, `ut3_add` (block-diag TT via `xnp.block`, concat masks),
   `ut3_sum_stack` (super-diagonal merge via 3 identity einsums), `ut3_inner_product` (mask→squash→absorb→
   scan zipper), `ut3_norm_orthogonalized`.
4. **Sampling** (`ut3_sampling`) — `ut3_entries`, `ut3_apply` (pack vecs to `N`), `ut3_probe`
   (pack + `probe_t3` + `unpack_vectors`), `ut3_full_sum` (apply with ones). Re-mask then call the shared
   ragged-path entries/apply/probing.
5. **ut3svd** (`ut3_svd`) — `ut3svd(data, max_tucker_ranks, max_tt_ranks)`: cap by `min(current,max)` →
   `compute_minimal_ranks` → build truncation masks → `uniform_t3_svd` sweep (orthogonalize + per-edge-SVD
   `lax.scan`, pad back, mask-multiply) → shrink supercore to those ranks. **No rtol/atol** (rejected; would
   make data-dependent shapes). **Per-stack-element `max_*_ranks` allowed** (the variety/rank-sweep —
   verified: one call, elem0→rank2, elem1→rank4). Extracted
   `ut3_orthogonalization.down_orthogonalize_tucker_supercores` (bare batched Tucker SVD), reused by the
   `.data` orthogonalizer and the sweep.

## Remaining slices
- **6 — Transposes.** `apply_transpose`/`entries_transpose`; introduce the `d`-folded named contractions
  (the uniform analog of `backend/contractions.py`, with `d` folded into batched einsums instead of a
  ragged map over the mode index). Oracle: ragged transpose.
- **7 — jax wiring (IN PROGRESS).** `UniformTuckerTensorTrain` is **registered** as a jax pytree
  (children = the two supercores; `aux_data` = the `UT3Masks` holder, identity-hashed via `eq=False`).
  jit-ting the ops then surfaced the key finding (as slice-7 was meant to): **masks must be numpy (host)
  and all mask logic must use `np`, not `xnp`.** Under jit any `jnp` op on a mask is a tracer →
  `int(mask.sum())` shape/rank extraction raises `ConcretizationTypeError` (`to_dense`/`inner`/`t3svd`/
  `probe`), and mask-*recomputing* ops (orthogonalize/svd/`+`/`×`) **leak tracer masks into `aux_data`**
  (silently invalid). Remaining slice-7 build plan, in order:
  1. **`xnp → np` mask refactor** across `ut3_*`: mask builders, rank recurrences, `+`/`×`
     concat/Kronecker, and `int(mask.sum())` shape/rank extraction all use `np` (host); only supercores
     go through `xnp`. `make_uniform_masks` emits numpy (drop its `use_jax`); `to_jax`/`to_numpy`
     convert the **supercores only** (masks stay numpy).
  2. **Tracer guard** at the mask chokepoint (`apply_masks_to_cores` / the shape-extraction helper):
     if a mask is a `jax.core.Tracer`, raise a clear structural error leading with the **functional**
     remedy — *"uniform masks must be concrete host (numpy) arrays … close over the masks and trace
     only the supercores"* — not jax's `ConcretizationTypeError`. No false positives (a mask is never
     legitimately a tracer here). Put the short version as a code comment where the guard lives.
  3. **Signature-comment contract:** tag mask args **`HOST bool, static`** (the `HOST` token defined in
     `docs/signature_style.md`) across the `ut3_*` signatures — the comment states the contract, the
     guard enforces it (poor-man's dtype/placement type, no type-system pain).
  4. **Right/wrong functional doctest** (no frontend needed — the OO-averse backend user is first-class):
     the close-over pattern works; passing masks as traced args raises the guard's stable message
     (`+IGNORE_EXCEPTION_DETAIL`).
  5. **Uniform dispatch tests** in `tests/test_dispatch.py` (jit each op, prove no hidden numpy /
     no tracer-leak) — where uniform jax coverage lands.

  Full reasoning: `docs/uniform_pytree_composition.md` ("Masks are numpy (host) — the jit story" +
  "How to jit … functional, or via the frontend"). **Don't "fix" mask `np.*` to `xnp` — it's intentional.**
- **8 — Constructors + IO.** `zeros`/`ones`/`randn` (pure constructors keep `use_jax`), `from_canonical`/
  `from_tensor_train`, `save`/`load`.
- **9 — t3m.** Elementwise multiply + truncation; both `t3m_form_then_round` and the max-rank
  `t3m_inplace_fused` (mirror the ragged `t3m`).
- **Deferred** (notes in plan): partial `sum` (only full-sum-via-apply-ones done), `to_vector`/`from_vector`
  (route via ragged + jax pytree).

## Conventions to keep (don't re-derive)
- **Dispatch by inference**, not threaded flags: `use_jax = tree_contains_jax(data[:2])` (note `data[:2]`
  = the **supercores** only), then `xnp,_,xscan = get_backend(True, use_jax)` (uniform path always
  `is_uniform=True`). Only the supercore constructors keep a `use_jax=` param.
- **Masks are numpy (host), via `np` not `xnp` — intentional, jit-required.** Masks are static
  `aux_data` structure; under jit any `jnp` op on them is a tracer. So mask logic (builders, rank
  recurrences, `+`/`×`, `int(mask.sum())`) uses `np`; `to_jax`/`make_uniform_masks` keep/emit numpy
  masks; only supercores go through `xnp`/`to_jax`. **Rule: supercores → `xnp`; masks → `np`.** Do not
  "fix" mask `np.*` to `xnp` for consistency. (`docs/uniform_pytree_composition.md`.)
- **`d` (mode index) leads** axis 0, outside the stack: supercores `(d,)+stack+(...)`; `lax.scan` scans
  axis 0. `shape_mask` has no stack `(d,N)`; edge masks carry the stack.
- **Masking schedule**: re-mask on entry to every op; SVD remainder `R = ss·Vᵀ` auto-zeros padded slots so
  no garbage propagates. Ranks shrink to the **structural** minimum the SVD yields (static, jit-safe) —
  never numerical-zero dropping (that would need a forbidden rtol).
- **Razor**: frontend methods are thin wrappers; all nontrivial logic is a `.data → .data` backend fn.
- **Workflow**: commit per slice to `main` after the full regression is green; stage only relevant files
  (leave stray `.npz`/`.idea`/`conf_OLD.py`/`make.bat` untracked); end messages with the
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer. Run tests filtering noise:
  `python -m unittest tests.test_X 2>&1 | grep -vE "^(RAGGED|NUMPY)"`. Scripts from `/tmp` need
  `PYTHONPATH=/home/nick/repos/T3Toolbox`.

## Key files
- Frontend: `t3toolbox/uniform_tucker_tensor_train.py`
- Backend: `t3toolbox/backend/ut3_{masking,conversions,operations,orthogonalization,linalg,sampling,svd}.py`
- Shared-path edits made this port: `t3toolbox/backend/{orthogonalization.py (if xs[0] fix),
  t3_operations.py (t3_to_dense_chain / absorb_tucker_into_tt), apply.py (is_uniform inference)}`.
- Tests: `tests/test_uniform_tucker_tensor_train.py`
- Plan + design notes: `docs/uniform_port_plan.md`, `docs/uniform_{equivalence_contract,
  ranks_and_varieties,supercore_layout,masks_vs_ranks,pytree_composition}.md`
- The ragged `t3svd` to fix: `t3toolbox/backend/t3_svd.py` (sweep ~lines 74–94).
