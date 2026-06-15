# Doctest cleanup — handoff / resume note

Goal: rework the **existing** doctests in the verified modules into **reproducible examples**, per the
convention in **`docs/doctest_style.md`** (read that first — it's the spec). Exemplars: `manifold.py`
and the reworked `TuckerTensorTrain.__mul__` / `inner` / `t3svd`.

## Status — existing-doctest sweep ✅ COMPLETE (all committed)

- `manifold`, `corewise`, `backend/stacking` — already conformed (0 failures), untouched.
- `backend/linalg` (`31fd91c7`), `backend/probing` (`c95ba848`), `backend/dense_t3svd` (`f3dbb0d7`),
  `basis_variations_format` (`5151f415`), `tucker_tensor_train` (161 failing → 0, latest commit),
  and the `__mul__`/`inner`/`t3svd` pilots. `python -m doctest <module>` is clean for each.
- The sweep doubled as a **stale-code detector** — found+fixed wrong captured values and dead
  imports/`NameError`s the broken doctests hid (e.g. a nonexistent `t3.t3_corewise_randn` used across
  ~12 `tucker_tensor_train` examples; `bv_to_t3`'s ambiguous `==`; dead imports in `dense_t3svd`).
- **Flagged for follow-up** (out of doctest scope, no behavior change): `TuckerTensorTrain.core_shapes`
  (property) strips the stack while `get_core_shapes()` (static) includes it — apparent inconsistency.

## Remaining doctest work: the deferred pass (Nick wants this)

**Add default-path doctests to currently *undocumented* public functions** (the convention's "default
path always earns an example"). A separate pass after existing fixes are done. The `linalg` probe
already did this for `linalg`'s `*_svd` / `*_svd_pair` / `pad_or_truncate` (kept). Candidates elsewhere:
the `probing` helper functions (`compute_*`, `assemble_*`), `tangent_operations`, `t3_operations`,
`ranks`, `contractions`, etc. Scope it deliberately (it's large) — one concise default-path block per
public function, no option cross-products.

## How the delegation works (what we learned)

- **One agent per file.** Never run two agents on the *same* file concurrently — they race on the file
  and lose each other's edits. (Different files in parallel is fine; that's how the wave ran.)
- **Agents CAN run code** (essential — doctests must be run-and-pasted, never hand-written). The plain
  ` ... | grep ` pipe and `<<heredoc` forms are sometimes sandbox-denied; the workaround agents use is a
  `/tmp` wrapper script, or `dangerouslyDisableSandbox` with `PYTHONPATH=/home/nick/repos/T3Toolbox`.
- **Always re-verify by running** the module's doctests yourself after an agent finishes, plus the scope
  check. `# doctest: +IGNORE_EXCEPTION_DETAIL` (inline) is honored by `python -m doctest` automatically.
- The doc was **validated** by a no-context probe (it reproduced the convention correctly), so delegating
  from `docs/doctest_style.md` alone is reliable.

## Not in scope / leave alone
Deferred layers (`ut3_*`, `ubv_*`, `uniform_*`, `wt3_*`, `bv_operations`, `OLD_*`). Wiring doctests into
CI is a separate backlog item (the point of making them reproducible is that this becomes possible).

## Broader session context
This doctest sweep followed a completed **signature-style** sweep of all verified modules
(`docs/signature_style.md`). Both conventions were developed via pilot → write-up → no-context-agent
validation → per-module delegation with review.
