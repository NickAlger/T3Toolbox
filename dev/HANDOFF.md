# T3Toolbox — current handoff

_Updated 2026-07-11._

## Where we are

**The naming pass + backend module reorg is DONE (2026-07-11)** — on top of the two things closed
2026-07-10 (the uniform frontend U7, and the `basis`/`base` → `frame` rename). Branch `main`,
direct commits. Full suite green after every sub-slice (593 tests / 40,215 subtests, exit-code
checked).

**→ Next: the doc pass (R3/R4)** — README/quickstart + fix the Sphinx build + fold the `docs/`
design rationale into user-facing docs. Then the rest of the release hygiene (R5 CI, R6 cleanup).

## Done this session (2026-07-11) — the naming pass + backend module reorg

Plan + complete old→new token inventory + execution deltas: **`dev/archive/naming_pass_plan.md`**. The
**user-facing conventions catalog** (prefix grammar, module map, semantic markers, deliberate
exceptions): **[`docs/naming_conventions.md`](../docs/naming_conventions.md)**. Decisions log:
`dev/archive/naming_review.md` (§1/§3/§4 now executed). Suite-gated per sub-slice; one commit each.

**Governing principle (Nick):** naming exists to help the user — orient, find, understand
inputs/outputs. When convention and clarity conflict, clarity wins. Corollary: parameter names
deliberately encode representation (`xx` ragged collection vs `uxx` uniform; `x` t3 data vs
`data` ut3 data) — keep them.

- **Slice 1:** `TuckerTensorTrain.squash()` → `.squash_tails()` (+ the bare `squash` bool param on
  the orthogonal-representations wrappers); `common.has_jax` → **`jax_available`** (31 refs).
- **Slice 2a — the sampling cut (option C, Nick's by-type grouping):** `apply.py` / `entries.py` /
  `probing.py` each hold their type's t3+tv+dense ops, transposes, and sweeps (tangent-apply/entries
  + sweeps moved OUT of `probing.py` into their type files; shared machinery stays in `probing.py` —
  containment `probe ⊃ apply ⊃ entries`, specializations import the general, never the reverse);
  `probe_derivatives.py` → **`sampling_derivatives.py`** (all jets together — shared machinery).
  **Settled with Nick (2026-07-11, confirmed after reflection): the uniform sampling modules stay
  grouped by OBJECT type (`ut3_sampling`/`utv_sampling`) — deliberately asymmetric.** The uniform
  sampling functions are thin wrappers (mask-once / pack / delegate to the shared polymorphic ragged
  machinery), not independent math, so they group by the object they wrap; the algorithm story lives
  in the ragged type files. Rationale recorded in `docs/naming_conventions.md`.
  Function renames to grammar: `probe_t3`→`t3_probe`, `tucker_tensor_train_*`→`t3_*`,
  `*_tangent*`→`tv_*`, sweeps→`tv_*`, `*_dense`→`dense_*`, and the plural→singular helper chains
  (`compute_xis`→`compute_xi` etc.).
- **Slice 2b — the `tt` chain family (Nick's tt/utt proposal):** new `tt_operations.py`
  (**merged polymorphic** `tt_reverse` + `tt_squash_tails` from the one-line same-math
  ragged/uniform twins — the one sanctioned twin-merge; `tt_change_core_shapes`, the zippers);
  `orthogonalization.py` → **`tt_orthogonalization.py`** (`tt_left/right_orthogonalize`, killing the
  3-way same-name collision); `utt_` reserved, **no members** (ragged name = polymorphic name).
  Added `t3_squash_tails(data)` (razor mirror of `ut3_squash_tails`). Then the t3 family:
  `t3_operations` **split 3-way** (`t3_constructors`, `t3_conversions` (incl. `to_dense`→
  **`t3_to_dense`**), slimmed `t3_operations`), `t3_orthogonalization` prefix-normalized,
  `dense_t3svd.py` merged into `t3_svd.py` (`dense_tucker_svd`/`dense_ttsvd`/`dense_t3svd`),
  `t3_inner_product_t3`→`t3_inner_product`, `rank_adjustment_sweep`→`t3_rank_adjustment_sweep`.
- **Slice 2c — ragged tv/fv:** `tangent_operations.py`→**`tv_operations.py`** (`tv_to_t3`,
  `tv_retract`, `tv_*gauge*`, …); `orthogonal_representations.py` **dissolved** along the uniform
  precedent (`t3_orthogonal_representations` → `fv_conversions`; the frame residuals →
  `fv_operations` as `fv_frame_*`); the fv micro-grammar (`fv_variations_zeros`,
  `fv_frame_reverse`, …).
- **Slice 2d — uniform mirrors:** `ufv_sampling.py`→**`utv_sampling.py`** with `ut3tangent_*`→
  **`utv_*`**; `ufv_tangent_operations.py`→**`utv_operations.py`** (`utv_retract`,
  `utv_corewise_inner`, …); `ut3_orthogonalization` data-level ops → `ut3_*`;
  `uniform_t3_svd`→`ut3svd_supercores`; `ut3_randn`→`ut3_corewise_randn`; masking prefixes
  (`ut3_make_masks`, `ufv_make_frame_masks`, …); stale pre-frame aliases (ubvt/ubto) normalized.
- **Slice 2e:** backend `GeometryOps` singletons `MANIFOLD`/`COREWISE` → **`MANIFOLD_OPS`**/
  **`COREWISE_OPS`** (frontend singletons keep their names); `__all__` hygiene.
- **Slice 3 (R2):** **`t3toolbox/__init__.py` curated** — classes, geometry singletons, GN models +
  the six factories, the four optimizers, safety, `__version__`. `backend/__init__.py` stays
  deliberately empty (docstring only): backend users import submodules explicitly.
- **Slice 4:** docs reconciled (cumulative token sweep + word-level review over `docs/*.md`,
  `CLAUDE.md`, `dev/paper_scope.md`, `docs/index.rst`); **`base-inner` → `frame-inner`** done;
  **`docs/naming_conventions.md`** written (Nick: "clearly document" — includes the cataloged
  deliberate exceptions). `dev/archive/`, `dev/naming_*` records, and all `OLD_*`/weighted files
  untouched by sweeps.
- Also: **deleted `OLD_orthogonalization.py`** (Nick confirmed superseded; suite-gated).

**Kept/guarded during the pass (don't undo):**
- The **A4 watchlist** (`dev/archive/naming_pass_plan.md`): same-looking, different-math near-twins —
  `sum_stack` (rank-growing tensor sum) vs `sum_stack_corewise` (rank-preserving core sum);
  `fv_to_t3` (single term) vs `tv_to_t3` (sum); the HS-vs-coordinate inner-product family;
  "corewise"/"numerically_" are semantic markers no rename may add or drop.
- **A3 gaps are INTENDED** (Nick): the frontend class asymmetries (e.g. `UT3Tangent` without
  `save`/`to_vector`) are design, not TODOs.
- **Weighted layer fully untouched** (incl. the unexported `wt3_squash_tails` copy in
  `t3_operations.py` and old-name references inside weighted docstrings — fix when reviving).

> **METHODOLOGY LESSONS (add to the frame-rename lessons; reuse for any future rename).**
> (1) **Frontend-method collisions**: many backend names double as frontend method names
> (`apply_corewise_transpose`, `retract`, `rank_adjustment_sweep`) — those need *qualified*
> (alias-scoped) renames + bare renames only inside the defining module; blind bare renames would
> mangle the frontend. (2) **Inventory BOTH import forms** — `import x.y as z` AND
> `from x import y as z`; a missed `from`-alias (`ufv_tops`) briefly broke the uniform seam.
> (3) **`getattr(module, 'name')` string references** in tests are invisible to token renames —
> the suite catches them (loudly). (4) **Slash-glob prose** (`compute_xis/mus/...`) and
> **brace-glob docstrings** (`probing.{a,b,c}_*`) need hand fixes. (5) **Substring module names**:
> `orthogonalization.py` ⊂ `t3_orthogonalization.py` — use lookbehinds. (6) **Never pipe the
> gate** (`pytest | tail` eats the exit code) and **never edit tracked files while a gate runs**
> (one mixed-tree incident → reset + deterministic replay; scripts made that cheap).

## Prior sessions (2026-07-10) — U7 uniform frontend + the frame rename

Summarized; details in the git log and `dev/archive/uniform_optimizers_plan.md`.
1. **Uniform layer CLOSED** — backend, optimizers, and the U7 frontend (four `optimizers.*` + six
   `fitting.*_model` factories infer ragged-vs-uniform from `x0`; `fitting.UniformGaussNewtonModel`
   is the roll-your-own surface; jit-compile-once via value-hashed aux; two worked examples).
   Standing constraints: the uniform optimizer needs a **minimal-rank base**
   (`uniform_minimal`, called transparently by the frontend); the **packedness-mirror** convention
   (user-facing ops mirror packedness; the fitting split-seam is packed-only).
2. **`basis`/`base` → `frame` rename DONE** (`naming_review.md` §2): `T3Basis`→`T3Frame`,
   `bv_`→`fv_`/`ubv_`→`ufv_`, the C stack → frame stack. Math "basis" and the plain manifold
   "base point" preserved.
3. Packaging fix (`pip install -e .` works).

## Active thread: the T3Toolbox software reference paper (independent)

Scope + curation: **`dev/paper_scope.md`** (11 operation groups; two settled findings: the
canonical-Gaussian-tangent result, and minimal rank is a correctness precondition for nothing).
**Not the T4S paper** (that is the existing arXiv preprint, a historical algorithm reference).
Next on this thread: walk the groups starting at Group 6 (`docs/symmetric_probe_derivatives.tex`
is nearly a drop-in chapter). Note: `paper_scope.md`'s code references were updated by the naming
sweep; the `.tex` files were not (they cite math, not code).

## Next steps

1. **Doc pass (R3/R4):** README + quickstart ("DO NOT USE" banner stays until shipping); fix the
   Sphinx build (`conf.py` autoapi exclusions, committed `_build`, `modules.rst` title,
   `docs/index.rst` examples are stale and should be regenerated); fold the `docs/` design
   rationale into user-facing Sphinx docs. `docs/naming_conventions.md` is written and should be
   wired into the doc tree.
2. **R5 test CI** (pytest + numpy 1.x/2.x matrix; wire doctests in — they are green module-by-module
   but not in CI).
3. **R6 cleanup:** the remaining `OLD_test_*.py` files (delete only after confirming preserved
   coverage), stray `docs/make.bat`, `.idea/`.
4. **→ 1.1:** the Goal-1 `fit(...)` facade; revive/redesign the weighted layer.

## The 1.0 roadmap (mid-level-toolkit scope) — summary
- **R1** packaging correctness — mostly done (`readme`, packages.find); `CHANGELOG.md` still to create.
- **R2 — DONE (2026-07-11):** public API surface (`__init__.py`) + the naming/organization review
  (frame rename + the full naming pass + backend module reorg + `docs/naming_conventions.md`).
- **R3** README + quickstart (banner off only at shipping).
- **R4** docs build + fold design rationale into user-facing Sphinx docs.
- **R5** test CI (numpy matrix + doctests). No auto-formatter near the curated style.
- **R6** cleanup — `OLD_test_*` etc., delete only after confirming preserved.
- **R7 — DONE** (uniform layer + optimizers + U7 frontend).

## Don't-trip constraints (the maintainer's standing rules)
- **Naming: read [`docs/naming_conventions.md`](../docs/naming_conventions.md) before naming
  anything new**; user-over-convention; the semantic markers ("corewise", "numerically_") and the
  representation-encoding parameter names are load-bearing.
- **The uniform optimizer requires a minimal-rank base** (`uniform_minimal`; frontend calls it
  transparently; `uniform_least_squares_problem` rejects non-minimal x0).
- **The packedness-mirror convention** (U3.5) — don't "normalize" it to a flag.
- **A uniform op needs more than dense-vs-ragged** — exact output masks + garbage-robustness
  (`docs/testing_strategy.md`). Masks are host numpy (`np`), supercores `xnp`.
- Numerical test assertions use a **tolerance**; exact comparison only for structure.
- Never delete an `OLD_*` (or anything) until functionality is **confirmed preserved**.
- "DO NOT USE" banner stays until the literal moment of shipping.
- **No automated tool rewrites the code style** (esp. the shape comments). No `manifold.py` rename.
- **Weighted layer is out of scope** until its post-1.0 revival — no renames, no deletions, only
  reference-fixes required to keep it importable.
- Research caveats are not user-facing. Notes are preserved/relocated, never lost.
