# Padded frame / cone reference plan — critic review, round 3

**Plan:** `dev/padded_frame_cone_impl_plan_2026-11-02.md` (round 3, 1761 lines; CHANGES rows :57-63).
**Previous reviews:** `dev/padded_frame_cone_plan_review_r1.md`, `_r2.md`. **New probes** (read-only, `t3toolbox` env,
session scratchpad): `probe_prefix.py` (inclusion vs prefix checks), `probe_imports.py` (AST import graph with the planned module).

## Overall assessment

Round 3 resolves everything left open in round 2:

- **m32.** The writer's pushback is right. The inclusion check I proposed passes my own probe's gappy masks; the prefix-mask
  requirement rejects them in both modes, `unsafe()` included. What remains is the numerical zero-padding precondition,
  which the ragged layer already has.
- **m33.** It is now a documented judgment call. The recommended placement (option C) is acyclic: I checked it on the real import
  graph with the new module added. It also respects the backend/frontend razor and the naming rules.
- **n35–n39.** All taken.

Four optional NITs remain (n40–n43), none of which needs another round.

## m32 — the prefix-mask requirement (§3.4 :630-652, §3.6 :734-736, §3.8 :773-774, T5(g), I-34, contracts row :1366)

**The pushback is correct.** My round-2 change, "the old-rank prefix masks lie inside `x0`'s masks", would have passed
the very case I reported. `probe_prefix.py`, with old ranks `(1, 1, 1)` / `(1, 1, 1, 1)`:

| `x0` | mode-0 mask | inclusion | prefix | ‖X_old − x0‖ / ‖x0‖ |
|---|---|---|---|---|
| round-2 case: `uz + ua` (`from_t3(n=r=2)`) | `[1 0 1 0]` | **passes** | fails | 1.00 |
| control: `from_t3(resize(a))` | `[1 1 0]` | passes | passes | 0.00 |
| residual: `from_t3(ragged z + a)` | `[1 1 0]` | passes | passes | 1.00 |

**Coverage.**

- **Gappy masks** (row 1). The requirement turns them into a structural `ValueError` in `padded_structures`, which raises in
  `safe()` and `unsafe()` alike. T5(g) tests both modes on exactly this case.
- **Padding inside the old prefix.** Prefix masks with `old ≤ ranks` put the old prefix inside the real slots, so the
  "garbage gathered as old content" case is also gone.
- **The residual** (row 3). This is a prefix-layout `x0` whose content is not in the leading slots. It is the ragged layer's
  `z + a` case carried over by `from_t3`. On both layers it is the numerical zero-padding precondition: checked in safe
  mode, unchecked under `unsafe()`. The plan says so ("Zeros outside the old block are numerical on both layers"), and
  that matches the house rule.

**The remedy.** The error message names the right one: `from_t3` of a ragged `resize`. Dropping my round-2 `ut3svd`
suggestion is also right, because `ut3svd` recompresses rather than restoring the zero-padded layout.

**Bookkeeping.** The decision is recorded in §3.8's two new rows, I-34 and the `numerical_contracts.md` row. The
zero-padding message now says "in the prefix layout `resize` produces".

**Resolved.**

## m33 / n34 — placement (§3.2 :562-576, §1 :178, I-32, G-acyclic :1285-1288)

**Option C, a new `backend/padded_frames.py` (recommended).**

- **Acyclic.** `probe_imports.py` builds the backend import graph from every `import` node, module- and function-level. It
  then adds a `padded_frames` node with the §1 import list and the edge `geometry → padded_frames`. Results:
  - no cycle;
  - `padded_frames`' import closure reaches none of `geometry`, `optimizers`, `fitting`, `uniform_fitting`;
  - the only non-backend module in that closure is `t3toolbox.corewise`, which imports only `backend.common` and is
    already imported by nine backend modules.
- **No new importers.** Only `geometry` (the members) and the docs import the module; the T3-free driver in
  `optimizers` does not.
- **The imported names exist.** In `fv_operations`: `ZERO_FRAME_SEED` :259, `fv_variation_shapes` :94, `_identity_prefix`
  :262, `_haar_rows` :274. In `ufv_operations`: `ufv_stack` :187, `ufv_unstack` :173. In `ufv_conversions`: the three
  converters :141-257.
- **Cross-module private use has precedent.** The cited precedent is real: `geometry.py:538` calls
  `ufv_operations._real_group_levels`.
- **Razor.** The module holds backend standalone functions only. The frontend surface is still `padded_start`, with no
  frontend constructor (I-22), so the backend/frontend razor is unchanged.
- **Naming.** The `fv_` / `ufv_` prefixes follow the data and the module is named for the topic. That is the `sharing.py` rule:
  "a feature is not a family" (`naming_conventions.md` :46-49). The module map gets a line (:1406-1409). Every reference
  moved with the code: §5.2–§5.3 headings, S2–S4 files, the `frame_variations.md` text, `api_reference.rst`, §12.

**Option A** (the function-level import) is described accurately: it is the first upward function-level edge, and it is
recorded as an exception if chosen. **n34** is fixed: `_bfit_pd` and `_jitted` are now described as lazy / optional
imports.

**I endorse C.** The choice stays Nick's ("(weigh)").

**Resolved.**

## Round-2 NITs

| r2 | status | where |
|---|---|---|
| n35 | taken | §4.4 :913, §5.4 :1127 ("containing the old frame's subspaces, §2.3") |
| n36 | taken | §2.3 :314 ("and `X_old` has full rank at them (generic, not structural)") |
| n37 | taken | §2.7: the bracket is "at least the projection onto `F*`'s own rank-`r` space, possibly above the cone value, measured 0.99–1.04×". The "free last core" subspace is indeed inside `T_{F*}` (the `δG_{d−1}` term). The tex paragraph says the same. |
| n38 | taken | fitting §5 :1316-1317, CHANGELOG :1488: "exact for matrices; for T3 … within a few percent". "At least 15–76×" at :129, :758, :1325, :1362, :1459, :1501, I-25. |
| n39 | taken | `stats['start_frame_source']` in 11 places; the old name survives only in the historical CHANGES rows |

## New NITs (optional; no further round needed)

- **n40** — §3.4 :634, "Uniform layer: the layout is checked structurally".
  - Only the *mask* layout is structural. Whether `X_old` sits in the leading real slots stays numerical on the uniform
    layer too, as row 3 of the table shows: `from_t3` of a ragged `z + a` has prefix masks and is caught only by the
    safe-mode check.
  - Suggested heading: "the mask layout is checked structurally; the content layout is the shared numerical
    precondition".
  - Also at :650: "which with prefix masks are all the slots outside the old prefix" → "the real slots past the old
    prefix, up to `x0`'s ranks". The uniform padding beyond the masks is never read, as §3.6 :734 already says.
- **n41** — G-acyclic :1286-1287 says "an AST scan sees module-level edges only".
  - Not true: an AST scan sees function-level `import` nodes as well, and `probe_imports.py` uses them.
  - Simpler gate: build the graph from every import node; assert it is acyclic; assert that the *backend* has no
    function-level library import. Both hold today (the probe found none).
  - That gate keeps holding for the next feature too, not just for `padded_frames`.
- **n42** — `naming_conventions.md` :76-78. The "Outside the matrix" list names infrastructure and the role-named fitting
  stack, but not `sharing`, the analog the new line cites (it is described at :46-49).
  - Listing `padded_frames` alone there makes the list look exhaustive.
  - Either put the sentence beside the `sharing` paragraph, or list both.
- **n43** — S5's content (:1518) names the minimal-old-ranks check but not the prefix-mask check (I-34), which lives in
  the same member. Add it; the gate already covers it through T5(g).

## Checked and found correct (round 3)

- **The m32 mechanism.**
  - Prefix masks with `old ≤ ranks` put the old prefix inside the real slots.
  - Gappy masks are rejected in both modes.
  - The residual is the shared numerical precondition, as on the ragged layer.
  - The control (`from_t3` of a `resize`) passes and gathers `X_old` exactly (`probe_prefix.py`).
  - `common.prefix_mask` (:772) is the right primitive for `mask == prefix_mask(mask.sum(-1), size)`.
- **The option-C import graph.**
  - Acyclic, and the closure of the new module stays below `geometry`.
  - The backend has no function-level library import today, so Option C keeps that.
- **The decisions are internally consistent.**
  - I-32 is marked "(weigh)" and both options are complete.
  - I-34 agrees with §3.4, §3.8, T5(g) and the contracts row.
  - Every option-C relocation propagated to §1, §3.1, §4.4, §5, §7, §8 and §12.
- **Still in force.** The round-1 and round-2 "checked" lists, including:
  - the dense per-step oracle (T3(h));
  - the `K` lift;
  - the bitwise ragged-path uniform old frame on prefix inputs;
  - G-bit and the stats keys under decision 2.

VERDICT: APPROVE
