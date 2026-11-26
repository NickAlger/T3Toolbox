# M5 design note: the types (comments only)

Implementer, 2026-11-23, on `2ac0013c` (M4). Plan §9.1, §9.3 and §7's M5 row; grammar `signature_style.md` (M1).

## 1. The signatures: 105, grouped

Only the surface that survives P20 is typed. The old layer (`compute_*`, `*_from_sweep`, `*_derivatives*`,
`sampling_derivatives`, the old kinds) is left alone, as M4 left it.

| Group | n | The lines typed |
| --- | --- | --- |
| Forwards: `t3_`, `ut3_` ×3; TT, UTT `.probe/.apply/.entries` | 12 | point `X`; `ww` `P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)`; `index` `index^d, shape=(d,)+W, int`; return `z: P⊗V_i*` / `z: P` / `z: ℝ` |
| Tangents: `tv_`, `utv_`, `t3_…_corewise_tangent`, `ut3_…` ×3; `T3Tangent`, `UT3Tangent` `.probe/.apply/.entries`; TT, UTT `.…_corewise_tangent` | 24 | `variations`, `core_directions` `X`; return `Jp: P⊗V_i*` etc. (`J: X -> Y*`) |
| The 42, the lines M4 left | 42 | as above; `c` realigned; returns `X*` (transposes, Hessians), `E*` CP factors (ambient, whose `ww` is `V_i`) |
| Dense `dense_*` | 3 | `T` `E, shape=C+(N0,...)` |
| `jets` | 13 | 8 need lines (the converters, `prepare_derivative_data`, `affine_jet`, `index_to_one_hot`, `check_sample`); 5 already typed by M1/M2, checked |
| `data_preparation` | 4 | `data_block_norms`, `rescale_direction`, `rescale_coordinates` (backend and frontend): `y` `Y*`, blocks `ℝ` per block |
| Engine `sampling_variables` | 1 | `sample`, `y` `Y*`, `weight` `omega`, the directions `X` |
| Oracle | 4 | `LeastSquaresBatch`'s fields and `.data_metric`; `LeastSquaresOracle.query`'s and `Oracle.query`'s returns: `r`, `Jp` `Y*`; `g`, `JtJp`, `Kp` `X*` before the projection; `f`, `Jp_norm_sq`, `f_resolution` `ℝ`; block sums `ℝ` per block |
| `uniform_fitting` `pack_sample`, `pack_data` | 2 | `Y*` / `omega` / inputs |

`derivatives_to_taylor` types as `Y*` in derivative coordinates to `Y*` in Taylor ones, "or `Y` the other way (the
contragredient)", and `taylor_to_derivatives` the reverse (plan §1). Plus M4's minor points: the "gradient" sentence
of the corewise Hessians, which I find in **four** docstrings, not six (the three backend ones and TT's
`probe_corewise_hessian`; the other eight point to them), and the nine `c` columns.

## 2. The engine's column

A fifth column, `Type`, on the shape table (`| Variable | Index | Jet sizes | Size |`; `test_sampling_graph` reads
cells 1 to 3, so it still passes):
- `P`: `xi`, `mu`, `nu`, `nu_L`, `eta`, `eta_D` and every dot;
- `Y*`: `z`, `r`, `Jp`;
- `Y`, turned: `w_tilde`;
- `P*`, turned: every other tilde and tilde-dot (they live on edges, not in `Y`);
- `X*`: `V_tilde`, `H_tilde` and their dots.

The prose after the table gives `f`, `Jp_norm_sq` `ℝ`, `r_block_sumsq` `ℝ` per block, and `g`, `JtJp`, `Kp` `X*`.
`scratchpad/m5impl/check_engine_types.py` checks every public name of probe and apply, on a frame and on a point,
under both laws: 152 cases, worst 2.4e-15.

## 3. Law tests (new `tests/test_sampling_typed.py`)

1. The 36 forwards and tangents, through M4's kind of adapter: the input law on all, the jet law on the 24 with jets
   at `(2,)` and `(1, 1)`.
2. The dense functions: the input law with `E` moved by `⊗Δ_i^-1` (a new mover, `InputLaw.dense`), and the jet law.
3. `jets` and `data_preparation`:
   - `affine_jet`, the jet law's own definition;
   - the converters, `prepare_derivative_data`, `data_block_norms` (`P` per block), `equalizing_weight` (`P` to `P*`);
   - the recipe `blocks_to_weight(equalizing_weight(data_block_norms(y)))` moves as `P*`, so `f` is invariant.
   Vacuous, so skipped: `index_to_one_hot` and the checkers.
4. The oracle's outputs on a drawn batch (`γ ≠ 1`), by their types.
5. **The column tested:** parse it from the docstring, and run each public name under its type's laws. Then the
   column cannot drift.

Teeth: mutants in scratch, as in M4.

## 4. The AST check

`m5/ast_equal.py`: every `.py` under `t3toolbox/`, at `HEAD` (`git show`) and in the tree, parsed, docstrings removed,
`ast.dump(include_attributes=False)` compared. Same set of files, no non-`.py` change. Then the gate: the suite at
`-n 2 -v` with M4's node tally (only new nodes), the module doctests, the quickstart, the pages, the examples, Sphinx.

## 5. Decisions

- **D1. Where `J: X -> Y*` goes.** I recommend per line only: the arrow is read off the argument and return lines.
  The 9.1 table goes to P19's page. The alternative is a "Type:" sentence in about 100 docstrings.
- **D2. The base point.** The corewise point `x` is `X`, "the point" (the input law moves it as `X`). A manifold
  frame and `Vc` get a role and no type: the frame is not in its own `X`.
- **D3. `omega`** keeps a role and no algebraic type (M1's practice). `ω²` is `M`'s diagonal; `ω` itself is not in
  `Y`.
- **D4. Spelling.** `P_in_i` in comments (M0 decision 1's analogue), `index^d` for entries. The glossary's `P` row
  gains `P_in,i` and `index`, so that the vocabulary `signature_style` points to is complete (docs only).
- **D5.** The `T3Tangent` transposes return `X*` "as a `T3Tangent` at the frame, before the projection".
- **D6.** Lines over 120 characters wrap with `#   ` continuations, as `check_sample` does.
- **D7. Out of scope unless ruled in:**
  - `LocalModel`'s comment lines (frontend and backend), which still say `𝒥ᵀ ω² r`; I recommend including them;
  - `sampling_definitions`, which gets one sentence pointing to the column.
