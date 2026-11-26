# M5 handover: the types (comments only)

Implementer, 2026-11-23, on `2ac0013c` (M4). Built from `M5_DESIGN_NOTE.md` and Claude's rulings (D1 to D7 all agreed,
no separate design review). Nothing is committed.

## 1. What changed

**No library code.** `m5/ast_equal.py` compares every `.py` under `t3toolbox/` at `HEAD` and in the tree, docstrings
removed (comments are not in the AST), by `ast.dump(include_attributes=False)`:
- 60 files compared;
- 17 differ in text and **0 in code**;
- the same set of files (git's view, ignored files left out), and no non-`.py` file changed.

Its teeth (`scratchpad/m5impl/ast_teeth.log`): a changed comment or docstring reads as identical; a changed default,
annotation or code string literal reads as different.

**Step A, the signature comments, by script** (`m5/types.py`; its log `m5/edit_log.tsv`):
- 313 edits in 99 signatures, 16 files.
- The script reads each in-scope signature block (from `def` to the body) and replaces each trailing comment it finds
  in a table of old → new texts. The `#` stays in its column. A line over 120 characters wraps with `#   `
  continuations (D6).
- The 219 comments it leaves alone are listed in the log as `LEFT`: the `order` and flag lines, and the lines
  M1, M2 and M4 already typed.
- The nine `c` lines whose `#` was one column off are realigned (M4 review, minor 2).

| Group | Signatures | Types written |
| --- | --- | --- |
| forwards: `t3_`, `ut3_` ×3; TT, UTT `.probe/.apply/.entries` | 12 | the point `X`; `ww` `P_in_i⊗V_i`; `index` `index^d`; `z: P⊗V_i*` / `P` / `ℝ` |
| tangents: `tv_`, `utv_`, `t3_/ut3_…_corewise_tangent`; `T3Tangent`, `UT3Tangent`; TT, UTT `…_corewise_tangent` | 24 | the directions `X`; the frame a role only (D2); `Jp` / `J x_dot` in `Y*` |
| the 42 (the lines M4 left) | 42 | `X*` blocks; `E*` CP factors (ambient, whose `ww` is `V_i`); the tangent transposes `X*` "in a T3Tangent (UT3Tangent) at frame (before the projection)" (D5) |
| `dense_*` | 3 | `T` in `E` |
| `jets` | 8 (5 already typed) | the converters `Y*` derivative ↔ Taylor, "or `Y` the other way"; `prepare_derivative_data`; `affine_jet` `V_i` → `P_in_i⊗V_i`; `index_to_one_hot` `e_index: V_i`; the checkers' `sample` |
| `data_preparation` | 4 | `y` `Y*`; blocks `ℝ` per block; the sample `P_in_i⊗V_i` |
| frontend `fitting.rescale_coordinates` | 1 | as its backend twin |
| `sampling_variables` | 1 | `sample`, `y` `Y*`, the directions `X`, `Vc` a role |
| `uniform_fitting` `pack_sample`, `pack_data` | 2 | |
| `LeastSquaresOracle.query`, `Oracle.query` | 2 | returns: `r`, `Jp` `Y*`; `g`, `JtJp`, `Kp` `X*` (before the projection); `f`, `Jp_norm_sq`, `f_resolution` `ℝ`; the block sums `ℝ` per block |

**Step B, by hand** (`m5/prose.py`, exact replacements, each checked to occur once in its function or class):
- **The four corewise Hessians** (M4 review, minor 1): "for the curvature term of a weighted least-squares fit
  (`Kp` at `c = M r`), pass `data_metric(…)`". The three backend docstrings, and TT's `probe_corewise_hessian`; the
  other eight point to them.
- **D5:**
  - the six tangent transposes say that the result is `X*` carried in a `T3Tangent` (`UT3Tangent`) before the
    projection;
  - the three uniform return comments name `UT3Tangent`.
- **The engine's table:**
  - a `Type` column, by rule (`TYPE`, checked against the rows; 38 typed): `P`, `Y*`, `Y turned`, `P* turned`, `X*`;
  - a paragraph **The types** after it, giving each type's meaning and its jet law;
  - the prose after it types `f`, `Jp_norm_sq`, `r_block_sumsq` (`ℝ`) and `g`, `JtJp`, `Kp` (`X*`).

  `test_sampling_graph`'s parse of cells 1 to 3 is unchanged.
- **The `sampling_definitions` sentence (D7):** each variable's type is the engine table's last column.
- **The oracle:**
  - `LeastSquaresBatch`'s `sample` (`P_in_i⊗V_i | index^d`) and `minibatch_scale` (`gamma`);
  - a paragraph **The outputs, typed** in `LeastSquaresOracle`'s docstring.
- **D7, `LocalModel`, backend and frontend:**
  - the comment lines and the class docstrings: `𝒥ᵀ ω² r` → `Jᵀ(M r)`, `‖ω J p‖²` → `<M J p, J p>`,
    `K_{Aᵀω²r}` → `K_{Aᵀ M r}`, with `M = gamma * omega**2` stated once;
  - the objective's letter `c` (now the transposes' argument) → `f` or `½<r, M r>`.
- **D4, docs only:**
  - the glossary's `P` row gains `P_in,i` (written `P_in_i`) and `index^d`;
  - `signature_style.md`'s vocabulary gains both, with one sentence: the corewise point is `X`; a manifold frame,
    `Vc` and `omega` take a role; a transpose at a frame returns `X*` in a tangent's container.

**`tests/_typing_laws.py`** gains two movers:
- `variations`, a direction at a frame (`V_i Δ_i^-1`);
- `dense`, `E` (`⊗Δ_i^-1`).

## 2. Tests: `tests/test_sampling_typed.py` (new; 14 tests, 559 subtests, 4 s)

| Class | What it establishes |
| --- | --- |
| `TestTheForwardsAndTangents` | the input law on all 36 forwards and tangents (counted); the jet law on the 24 with jets at `(2,)` and `(1, 1)` (48) |
| `TestTheDenseFunctions` | `E × inputs -> Y*`: the input law with `T` moved as `E`; the jet law |
| `TestTheHelpers` | `affine_jet` is the jet law (exactly); the Taylor converters move `P` and `P*` by the jet law and `V_i*` by the input law, and **pair contragrediently**: `<taylor_to_derivatives(c), r> = <c, taylor_to_derivatives(r)>` and `<derivatives_to_taylor(c), taylor_to_derivatives(r)> = <c, r>`, while the wrong pairing differs (orders `(2,)`, `(2, 1)`, where `k! ≠ 1`); the weight converter keeps the misfit; `prepare_derivative_data` by `P` / `P*`; `data_block_norms` `P` per block, `equalizing_weight` `P` → `P*`, the recipe's weight `P*` and its misfit invariant |
| `TestTheOracle` | every output of `query` by its type, on a drawn batch (`γ = 2`), all three kinds, a frame and a point, both laws |
| `TestTheEngineTypeColumn` | the column read from the docstring: complete (every row of `_SHAPES`), its vocabulary, structurally consistent (`mode` ↔ `Y*`/`Y turned`, a core-shaped size ↔ `X*`, `tilde` ↔ turned); each public name of probe and apply, on a frame and a point, under both laws by its cell (`X*` on the mode axis read from the Size cell); the prose's `ℝ` and `X*` outputs |

`index_to_one_hot` and the checkers are not law-tested: nothing they take moves under either law.

**Mutation pass** (`scratchpad/m5impl/mutants.sh`, each on a fresh copy; `mutants.log`):

| Mutant | Result |
| --- | --- |
| test: the probe forwards' output moved as `V_i` (not `V_i*`) | 15 failures |
| test: `equalizing_weight`'s output moved as `P` (not `P*`) | 4 |
| test: the dense `T` moved as `E*` (not `E`) | 3 |
| docstring: `mu_tilde` typed `P` (not `P* turned`) | 5 |
| docstring: `Jp` typed `P` (not `Y*`) | 5 |
| docstring: `V_tilde` typed `P* turned` (not `X*`) | 17 |
| library: `taylor_to_derivatives` divides (the converters swapped) | 8 |
| library: `affine_jet`'s slot 1 holds `w + p` | 1 |
| library: `w_tilde` not turned (the seed's flip dropped) | 108 |

## 3. The gate

All on the final tree; logs in `scratchpad/m5impl/gate/`; Sphinx output deleted; nothing left running.

**The suite**, `pytest tests/ -n 2 -v` (under `timed.py`):
- **1979 passed, 5 skipped, 1 xfailed, 93,034 subtests passed**, 35:24;
- lowest MemAvailable 3.9 GB, peak tree RSS 8.1 GB (`suite_v.log`, `suite_v.stats`).

**The node tally** against M4's `-v` log (`m0/tally.py`; `tally_m4_vs_m5.txt`):
- 1,971 → 1,985 nodes;
- exactly 14 nodes differ, all new: `test_sampling_typed.py`, 559 subtests;
- every other node has M4's outcome and subtest count, so no number changed.

**The other steps**, `run_gate.sh -n 2 doctest-modules doctest-quickstart doctest-pages examples sphinx`, one after
another (`gate/docs/`):

| Step | Result | Seconds | Lowest MemAvailable (MB) |
| --- | --- | --- | --- |
| module doctests | PASS, 325 passed | 54 | 10,265 |
| `getting_started.rst` | PASS, 0 failed | 3 | 11,522 |
| pages (47 files) | PASS, 0 failed | 19 | 11,473 |
| examples | PASS, 14, 0 failed | 570 | 11,151 |
| Sphinx `-W` | PASS, 0 warning or error lines | 1,016 | 11,074 |

**The AST check:** `IDENTICAL CODE` (60 files, 17 differ in text only).

## 4. Departures and choices

- **The count is 99 signatures by script, plus two classes by hand** (`LeastSquaresBatch`'s fields; `LocalModel`,
  backend and frontend), against the note's 105. The difference: `LeastSquaresBatch.data_metric` and five `jets`
  functions were already typed in M1 and M2 and needed nothing.
- **`ut3_full_sum` and the unscale functions are left untyped:** they have no data-space argument.
- **M1's long lines** (`data_metric`'s exemplar, quoted in `signature_style.md`) are not rewrapped. D6 applies to
  the lines M5 writes.
- **The engine's `Y turned` / `P* turned`.** The plan's wording ("every tilde variable is `Y` in turned form") is
  refined, as the note proposed: the edge tildes are dual jets on edges, not elements of `Y`. Both laws confirm the
  column name by name.
- **The input law on the engine's edge variables holds for probe and apply only.** An entries index does not move
  with the input law, so `xi` (a column of `D`) does. The types are kind-free; that law is not. This is stated in
  the test.
- **The TT frontend's "gradients" on its transposes' returns now read "blocks: `X*`".** They are `J^T c`, a gradient
  only when `c = M r`.

## 5. Found outside the slice, for P19

- `backend/optimizers.py` beyond `LocalModel` still writes `Π 𝒥ᵀ ω² r` and `½‖ω⊙r‖²` in docstrings and comments:
  - lines near 765, 853, 930, 1107, 1445;
  - the stats record's `objective` / `misfit` comments;
  - `newton_cg`'s Hessian text.

  Each needs `γ` (`M`).
- The frontend `fitting` module text (`:7`, `½‖ω⊙(S(X) − y)‖²`).
- §9.1's table of function types goes to P19's page (D1).
