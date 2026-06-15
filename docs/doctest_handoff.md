# Doctest cleanup — handoff / resume note

Goal: rework the **existing** doctests in the verified modules into **reproducible examples**, per the
convention in **`docs/doctest_style.md`** (read that first — it's the spec). Exemplars: `manifold.py`
and the reworked `TuckerTensorTrain.__mul__` / `inner` / `t3svd`.

## Status

**Existing-doctest fixes — done & committed:**
- `manifold`, `corewise`, `backend/stacking` — already conformed (0 failures), untouched.
- `backend/linalg` (commit `31fd91c7`), `backend/probing` (`c95ba848`), `backend/dense_t3svd`
  (`f3dbb0d7`), `basis_variations_format` (`5151f415`).
- `TuckerTensorTrain.__mul__` / `inner` / `t3svd` (`022216a5`-era + `e17f90df`) — the hand pilots.

**IN PROGRESS — `tucker_tensor_train.py` (the big one): NOT committed.**
- A sub-agent (`general-purpose`, fix-existing-only) was reworking it when the session ended. It took it
  from **161 → ~23** failing examples, but the file has **uncommitted edits** and the agent may not have
  finished (it was still running; no completion report seen).

## Resume steps for `tucker_tensor_train.py`

1. **Check the agent.** If a completion notification arrived, read its report. The working tree already
   has its partial edits (`git diff --stat t3toolbox/tucker_tensor_train.py`). Do NOT assume it's done.
2. **Run the doctests:** `python -m doctest t3toolbox/tucker_tensor_train.py 2>&1 | grep -vE "^(RAGGED|NUMPY)"`
   (and `... | grep -cE "^Failed example:"` for the count). Find the remaining failures
   (`... | grep -E "line [0-9]+, in"`).
3. **Finish the remaining ~23** by hand or with a follow-up agent (same prompt shape as the others).
   They'll be the same patterns: unseeded random → `np.random.seed(0)`; raw residual floats →
   `np.allclose(result, reference)`; raw/gauge-ambiguous arrays → relationship-check + structure prints;
   stale deterministic values → fix to the real value (RUN it); failure modes → traceback blocks.
4. **Review before committing** (doctests are higher-stakes than signatures — a wrong pasted value is a
   silent lie, so *run*, don't eyeball):
   - `python -m doctest t3toolbox/tucker_tensor_train.py` → **0 failures**;
   - scope check — **no new `Examples` added to previously-undocumented methods** (this pass is
     fix-existing-only): `git diff | grep -E "^\+.*Examples"` should only correspond to methods that
     already had examples; leave `__mul__`/`inner`/`t3svd` untouched (already done);
   - spot-check a couple of reworked blocks for teaching quality.
5. **Commit** `t3toolbox/tucker_tensor_train.py` (per-module commit, message like the others). That
   **completes the existing-doctest sweep** of the verified modules.

## Then: the deferred pass (Nick wants this)

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
