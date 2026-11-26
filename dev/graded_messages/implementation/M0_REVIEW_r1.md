SIGN OFF

# M0 review, round 1: the `J -> P` and `J_in_i -> F_in_i` rename

Reviewer, 2026-11-22, on the uncommitted working tree over `368ba3b4` and `M0_HANDOVER.md`. I checked it
independently of the implementer's scripts (`m0/rename.py`, `battery.py`, `residual.py`, `tally.py`); none of them is
used below. My scripts and outputs are in `scratchpad/m0rev/`:
- `tokcmp.py`: the identity check;
- `rbattery.py`: the calls and the values;
- `mutants.sh`: the mutants.

`old/` is `git archive 368ba3b4`; `new/` is a copy of the working tree's `t3toolbox/` and `tests/`.

**Verdict: SIGN OFF.** Behaviour is unchanged bit for bit on everything I ran. Every changed token is explained by
the plan's map or by a listed hand edit. No `J` meaning the group is left. The tests are not vacuous and catch all
eight of my mutants. The doc entries are as decided. Two minor points follow; neither needs another round.

## 1. Identity: a token-level comparison under my own map

`tokcmp.py` tokenizes each `.py` file of both trees and aligns the two token streams. It requires every differing
token to be the old token under my map, written from plan §7 and Q12, not from `rename.py`:
- grouped terms with `J` among their leading capitals;
- suffixed locals;
- `len_J`, `J_in(_i,_j)`, `J_IN`, `J_i`, and the two renamed test names;
- a bare `J` allowed to become only `P`, `ring`, `jet_sizes` or `F_in`.

Code, strings, docstrings and comments are all tokens, so all are covered.

- **Explained edits:**

  | Kind | Count |
  | --- | --- |
  | term | 587 |
  | suffix | 377, exactly the plan's count |
  | exact name | 216 |
  | bare `J` → `P` | 193 |
  | bare `J` → `ring` | 146 |
  | bare `J` → `F_in` | 41 |
  | bare `J` → `jet_sizes` | 2 |
- **Residual: 8 items, each read.**
  - `sampling_variables.py`, the module docstring. Besides mapped tokens, the hand edit "``P`` its jet axes (of the
    declared sizes below)", reflowed (checked line by line with `git diff -U0`).
  - `ut3_operations.py`, `pack_vectors`' docstring: the added definition of the packed `F_in`.
  - `test_sampling_variables.py:328`: the same wording as the module docstring.
  - Five insertions, all the new tests:
    - the floor `found` / `assertEqual(scan_literal_jet_call_sites(), found)`;
    - `test_no_literal_names_the_retired_J`;
    - the `'JWCa,…'` row of the malformed strings;
    - `GroupLengths(J=…)` raising, with `hasattr(…, 'J')` False.

  Nothing else changed: no deleted file, no other code.
- **None of the bare-`J` edits is the Jacobian.** I read every removed line that holds a bare `J` (292 distinct
  lines, in `scratchpad/m0rev/bareJ_removed.txt`). Each one is:
  - the ring `Jets(order)`;
  - a jet size: `pack_vectors`' `J`, `_shape`'s `J`, the test locals;
  - a `GroupLengths` field, keyword or count;
  - or the grammar and its prose in the group-only files.

## 2. Behaviour: my own battery, call logs and values

`rbattery.py` deliberately covers the surface the implementer's battery did not emphasize. It patches
`numpy.einsum` and `jax.numpy.einsum` to log every call (string, operand shapes and dtypes, `optimize`).

**What it runs:**
- **Frontend on a ragged `TuckerTensorTrain`:** `probe` and `apply`; `*_corewise_tangent`, `*_corewise_transpose` and
  `*_corewise_hessian`, with `K = (2,)`; `sum_over_samples=False`.
- **Tangent layer:** `T3Tangent` `probe`/`apply` with `K = (2,)`, and the static `probe_transpose`/`apply_transpose`.
- **Uniform:** `UniformTuckerTensorTrain` `probe`/`apply`, its corewise transpose and Hessian; `UT3Tangent.probe` and
  `apply_transpose`.
- **Orders:** `()`, `(1,)`, `(3,)`, `(1, 1)` and `(2, 1)`.
- **The engine directly:**
  - `z`, `r`, `f`, `g`, `w_tilde` and `r_block_sumsq` with `y` and a weight;
  - a held `z`;
  - `Jp`, `Jp_norm_sq`, `JtJp` and `Kp` with variations and raw `K`-stacked core directions.
- **Entries:** forward and transpose, ragged and uniform.
- **The fitting layer:**
  - the exact-Hessian `local_model` quadratics (probe at `(1,)`, apply at `(2,)`);
  - `newton_cg(hessian='exact')` (trust region), ragged;
  - `newton_cg` on `UNIFORM_MANIFOLD`;
  - minibatched `adam` on `COREWISE`.
- **jax:** one `jax.jit` engine call in jax's default **float32**; the implementer's ran in x64.

**Result**, on `old` and on `new`:
- **Calls:** the two logs are byte-identical (`cmp`). That is 24,038 `einsum` calls, 60 of them `jnp` from the jit
  trace, sha256 prefix `51daf77c12170b90`.
- **Values:** 743 output arrays, 23,830 numbers in float64, float32, int64 and bool, are bitwise equal: dtype, shape
  and `tobytes()`.

This agrees with the code path I traced at orientation. Groups expand to fresh letters by first appearance, and the
numpy path reads intersection sizes only, so the same `einsum` calls are made in the same order.

## 3. Per-test counts

I relied on the implementer's two `-v` suite logs:
- `m0/baseline_v.log`, the archived HEAD;
- `scratchpad/m0impl/m0_suite_v.log`, the current tree. No file in `t3toolbox/`, `tests/` or `docs/` is newer than
  it.

I tallied them with my own parser, node by node, the two renamed tests mapped. Exactly two nodes differ, both
expected:
- `test_no_literal_names_the_retired_J`: new, passed;
- `test_malformed_strings_raise_at_every_order`: 45 to 48 subtests (the retired `J` string × 3 records).

The renamed `test_every_P_literal_is_a_jet_contract_literal` keeps its 93 subtests, and
`test_counts_are_nonnegative_ints_and_P_any_count` its 8. The summary lines agree: 1,897 → 1,898 passed, 5 skipped, 1
xfailed, 90,942 → 90,945 subtests, no failure. (My parser misses 28 subtest lines whose labels it does not match, the
same 28 in both logs, so the comparison stands.) I did not rerun the whole suite.

## 4. Mutants

Each mutant was applied to a fresh copy of `new/`, then `test_jet_contract`, `test_contractions_interpreter` and
`test_contractions_sharding` were run at `-n 2`. All eight are killed:

| Mutant | Result |
| --- | --- |
| M1: `is_readout`'s filter `'P'` → `'J'` (`_jet_contract_cases`) | 185 failed (`TestTheStrings` among them) |
| M2: `build`'s `keep = 'P' in …` → `'J'` | 378 failed |
| M3: the scan filter of `test_every_P_literal…` → `'J'` (the vacuous-loop case) | the floor fails |
| M4: `J` added back to `_JET_GROUPS` | 3 failed (the retired-`J` row × 3 records) |
| M5: a `J` property alias on `GroupLengths` | `test_fields_in_order` fails |
| M6: `GroupLengths` fields reordered, `W` before `P` | `test_fields_in_order` fails (the positional `GroupLengths(1, 2, 3, 4)` pin) |
| M7: a stale `'JWCa,Caib->JWCb'` literal in `probing.py` | `test_no_literal_names_the_retired_J` fails, and so does `test_every_library_subscript_is_frozen` |
| M8: `stepped()` deletes `'J'` instead of `'P'` (library side) | 5,636 failed |

M1, M2 and M3 are the vacuous-loop class of my design review's finding 1. M5 and M6 are the field-order point. The
scratch copy was deleted, and nothing is left running.

## 5. Leftover `J`s: my own scan

I scanned every token of `t3toolbox/` and `tests/` for words with a capital `J`, leaving out the `Jet`/`JAX`/`Jp`
families.
- **Bare `J`, 244 uses:** every one is the Jacobian (`J^T`, `J x_dot`, `J = dense_core_jacobian(…)`,
  `J = call(…corewise_tangent…)`). The group-only files now hold `J` only in the new retired-`J` tests and their
  docstrings.
- **The short words** (`Jv`, `Jx`, `JTr`, `JTz`, `Jd`, `Jk`, `Jn`, `Jm`, `DJ`, `JT`, `wJ`, `J_`): Jacobian products.
- **`J_TRAIN`, `J_VAL`, `J_val_verbose`:** the pinned case "J" of `test_optimizers_frontend`.
- **`J_t`:** the header of the bridge test (old kernels, deleted in P20).
- **The alphabets** in `sampling_definitions` and `probing`.

None is the jet group or a jet size.

## 6. The doc entries (plan §7, M0)

- **`glossary.md`:**
  - the preamble reserves `P` and `Q` and says `J` is the Jacobian only;
  - the `P` row is extended (ring, group, `P*`), not duplicated;
  - new rows: `Q`, `F` (the box, `F_in,i` written `F_in_i`, the packed `F_in`, and the frame in `P_F`, `T(F)`,
    `Π_{T(F)}`, `F_{x→t}`), `M` (with the manifold collision), `Y*`/`Y` (`*` the algebraic dual), `γ`, `ℓ` (not `λ`)
    and `[G]`;
  - the `m` row gains the number of jet variables.

  All as decided. Symbols are in backticks, and no function is named that does not exist yet.
- **`naming_conventions.md`, "Index letters":** `P`, `Q`, the Jacobian-only `J`, and the shape rule (a concrete shape
  spells the sizes, a family spells the group, `[G]`). One sentence places the old `t` paragraph with
  `sampling_derivatives.py`. Right scope: P19/P20 rewrite the rest.
- **The gate's doc steps** (the implementer's logs, which I read): module doctests 321 passed, quickstart and 47
  pages with no failure, 14 examples, Sphinx `-W` with no warning. The four "error" matches in `sphinx.log` are the
  name `relative_errors`.

## 7. Minor points

1. **`m0/baseline_v.log` is 19 MB**, in a directory meant for the commit. The per-node record it supports is
   `tally_baseline_vs_m0.txt`. Commit the log gzipped (about 1 MB), or leave it in the scratch folder and keep the
   tally. Claude's call.
2. **The glossary's `[G]` example** writes `F + W + [K] + C`. That is right as a concrete shape (full jets). Just
   beside it, `naming_conventions.md` writes a family as `P + W + [K] + C`. Both are correct under the stated rule; if
   the example is meant as a family, `P + W + [K] + C` would match the naming page. Optional.

The handover's account of what changed and why agrees with what I found in the code and the logs.
