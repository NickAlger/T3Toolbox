# Uniform T3 port — handoff (after slice 5)

Resume note for the `UniformTuckerTensorTrain` port. The living plan with per-function SHARE/REWRITE
verdicts and the 9-slice order is **[`docs/uniform_port_plan.md`](uniform_port_plan.md)**; the design
philosophy is the five **`docs/uniform_*.md`** notes (read those before touching uniform code). This note
records *where we stopped*, the *one open bug to investigate first*, and the immediate next steps.

---

## ⚠️ FLAGGED BUG — ragged `t3svd` returns NON-minimal ranks under truncation (investigate first)

**Ragged `t3svd` is supposed to return structurally minimal ranks. Under truncation it does not — this
is a bug and needs a thorough investigation before relying on the uniform-vs-ragged rank comparison.**

### Reproduction (confirmed)

```python
import numpy as np, t3toolbox.tucker_tensor_train as t3
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5,6,7), (4,5,6), (1,3,2,1))
x2, _, _ = x.t3svd(max_tt_ranks=2, max_tucker_ranks=3)
x2.tucker_ranks                                            # (3, 3, 2)   <-- tucker[0] = 3
t3.TuckerTensorTrain.get_minimal_ranks(x2.shape,
        x2.tucker_ranks, x2.tt_ranks)                      # ((2,3,2), (1,2,2,1))  <-- should be 2
x2.has_minimal_ranks                                       # False  <-- BUG
```

`tt_ranks = (1,2,2,1)`, so the Tucker rank at mode 0 is bounded by `rL·rR = r0·r1 = 1·2 = 2`. Ragged
leaves it at **3** → strictly non-minimal (`has_minimal_ranks` is `False`).

### Mechanism (located — strong head-start for the fix)

`t3toolbox/backend/t3_svd.py`, the L→R sweep at lines ~74–94. At each mode `ii` it does, in order:
1. `ragged_orth.down_svd_tt_core(..., max_rank=max_tucker_ranks[ii])` — truncate the **Tucker** rank at `ii`;
2. then (if `ii < last`) `ragged_orth.left_svd_tt_core(..., max_rank=max_tt_ranks[ii+1])` — truncate the
   **adjacent TT bond** `ii+1`.

So the Tucker truncation at mode `ii` is computed against the **un-truncated** neighbor bond `rR`. When
step 2 subsequently shrinks that bond below the Tucker rank, the Tucker rank is left at `> rL·rR` →
non-minimal. (No-truncation is unaffected: every SVD keeps `min(rows,cols)`, i.e. the structural rank, so
the result is already minimal — and uniform matches it exactly there. Only the interaction of a Tucker cap
with a smaller adjacent TT cap exposes it.)

### Investigation checklist
- [ ] **Characterize**: sweep `has_minimal_ranks` over many truncation patterns (vary which of
      `max_tucker`/`max_tt` bites, both directions, multi-mode) to map exactly when it triggers. Also check
      the **TT** ranks for the symmetric failure (a `max_tt` cap left larger than the adjacent
      `tucker·bond` bound) — the same sweep-order argument predicts it can happen on the TT side too.
- [ ] **Decide the fix.** Candidates: (a) a final minimality cleanup pass (cheap orthogonalize/SVD sweep
      that drops to `compute_minimal_ranks`), (b) bound each Tucker truncation by the *post-truncation*
      `rL·rR` (needs lookahead to the next bond cap), or (c) reorder/iterate the sweep. Prefer the one that
      keeps the represented tensor identical (it must — only redundant ranks are being removed).
- [ ] **Verify**: `has_minimal_ranks` holds after the fix across the swept patterns; `to_dense` unchanged
      (≤1e-14); kept singular values unchanged. Re-run `tests/test_tucker_tensor_train.py` (+ `test_t3m`,
      which rounds via t3svd).
- [ ] **Reconcile uniform.** Uniform `ut3svd` is *already* minimal (it builds masks from
      `compute_minimal_ranks` of the capped target — see `ut3_svd.py`). So **once ragged is fixed, the two
      agree exactly** and the "uniform is tidier than ragged under truncation" caveat disappears. At that
      point, strengthen `tests/test_uniform_tucker_tensor_train.py::test_t3svd_truncation` to assert exact
      rank-equality with ragged (it currently only asserts `to_dense` match + ranks ≤ cap, precisely
      because of this bug). The note in `uniform_port_plan.md` slice 5 should then be revised.

**Until fixed, the uniform behavior is the correct/desired one** (truly minimal, per the agreed
"shrink-to-minimal-structural" decision + the minimal-for-free principle). Do not "fix" uniform to
reproduce ragged's non-minimal output — fix ragged.

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
- **7 — jax wiring.** Register the `UT3Masks` holder / `UniformTuckerTensorTrain` as jax pytrees (masks as
  static `aux_data`, identity-hashed — slice-1 holder is already `eq=False` for this). Add the uniform ops
  to `tests/test_dispatch.py` (jit each, prove no hidden numpy). This is where uniform jax coverage lands.
- **8 — Constructors + IO.** `zeros`/`ones`/`randn` (pure constructors keep `use_jax`), `from_canonical`/
  `from_tensor_train`, `save`/`load`.
- **9 — t3m.** Elementwise multiply + truncation; both `t3m_form_then_round` and the max-rank
  `t3m_inplace_fused` (mirror the ragged `t3m`).
- **Deferred** (notes in plan): partial `sum` (only full-sum-via-apply-ones done), `to_vector`/`from_vector`
  (route via ragged + jax pytree).

## Conventions to keep (don't re-derive)
- **Dispatch by inference**, not threaded flags: `use_jax = tree_contains_jax(data[:2])` (or `is_ndarray`),
  then `xnp,_,xscan = get_backend(True, use_jax)` (uniform path always `is_uniform=True`). Only pure
  constructors with no array inputs keep a `use_jax=` param.
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
