# M0 design note: the `J -> P` and `J_in_i -> F_in,i` rename

Implementer, 2026-11-22. Read only so far; counts are from a token scan of `368ba3b4`.

## The map

| Old | New | Where |
| --- | --- | --- |
| the group `J` in grouped strings | `P` | 205 literals (subscripts with `->`), the grammar (`_JET_GROUPS`, the parser's `'J'`, its messages), 28 `'J'`/`'JW'`/`'JWKC'` literals in the test machinery |
| `GroupLengths.J`, `J=` keywords, `lens.J` | `P` | the field, 25 constructions, 9 reads, test locals named for the field |
| 20 suffixed locals (`mu_JWCa`, `w_JWo`, `jet_JX`, ...) + `factors_J`, `lead_J` | `J -> P` in the suffix | 377 uses |
| `len_J` (`pack_vectors`, `pack_if_ragged`) | `len_P` | 33 lines |
| `J_in_i`, `J_in_j`, `J_in`, `J_IN`, `J_i` (`pack_vectors`) | `F_in_i`, `F_in_j`, `F_in`, `F_IN`, `F_in_i` | ~180 lines |
| a shape `(d,)+J+W+(N,)` (packed input jets) | `(d,)+F_in+W+(N,)`, `F_in` the largest `F_in_i` | ~25 comments |
| a shape family `J + W + [K] + C + (size,)` | `P + W + [K] + C + (size,)` (`P` the jet axes, like `W`) | ~10 |
| test-only and local sizes named `J` | `F_in` or `jet_sizes` | 5 sites |

**Spelling of `F_in,i`:** `F_in_i` in code, comments and docstrings (same width as `J_in_i`, so aligned columns do not
move, and the `,` would clash with the comment grammar's separators); `F_in,i` in the glossary's prose, which says so.
**Stays:** the Jacobian (`J`, `Jp`, `JtJp`, `held_Jp`, `J^T`, `Jᵀ`, `J x_dot`), `Jet`/`JET`/`JAX` words, the alphabet
`'IJKL...'` in `sampling_definitions.py`. `dev/` is a record and is not touched.

## How

One script, token-level (`tokenize`, so code, strings and comments are told apart), with an edit log
(file, line, col, old, new). It tells the group from the Jacobian by context, never by the bare letter:
- strings matching the subscripts grammar; keyword/attribute/field `J` on `GroupLengths` (AST);
- an explicit table of identifiers and size words;
- `J` in shape grammar (followed by `+ W`, `+ X`, `+ ...`, `+ stack_shape`; `J^T omega^2 J + K` does not match);
- every bare `J` in the four files that are wholly about the group (`contractions.py`, `_jet_contract_cases.py`,
  `test_jet_contract.py`, `test_contractions_interpreter.py`; none has a Jacobian `J`);
- a short explicit list of prose and local sites (about 15), reviewed by eye.

## Evidence that nothing changes, bit for bit

1. **Round trip.** An inverse pass (the rules reversed, run on a copy of the renamed tree) gives a tree whose diff
   against `git archive HEAD` is empty, apart from the hand-written additions (the two pages, the new tests). This
   shows every edit is a consistent relabelling with no collision. A pre-scan finds no target token already present,
   except six bare `P` in two test files (a projector in comments; a local `P = np.polynomial.polynomial`), which the
   inverse skips by position.
2. **Completeness**, which the round trip cannot show (a missed `J` survives it unchanged): a residual scan
   classifies every remaining capital-`J` token in `t3toolbox/` and `tests/`; none may be the group. The edit log is
   tallied against the plan's counts.
3. **Behaviour.** No group letter ever reaches numpy or jax:
   - `contract` expands groups into fresh letters by position;
   - the only name-dependent part, the uppercase fallback pool, is never reached by the library's strings;
   - the numpy path depends only on index-set sizes.

   So the calls are the same. Checked by one script run against both trees (HEAD from `git archive`, no branch
   change): every public variable of the three kinds × orders `()`, `(2,)`, `(1,1)` × ragged/uniform ×
   manifold/corewise, the transposes, and a short `newton_cg`. Every `einsum` call (string, shapes) is logged: the
   logs must be identical and the outputs `np.array_equal`.
4. **The gate adds** completeness on every path it runs:
   - a stale `J` string now raises in `jet_contract` (not a group);
   - `len_J=`, `GroupLengths(J=)` and `.J` raise;
   - the renamed doctests and the two pages run, and Sphinx `-W` builds them;
   - **the counts**: passed and subtests must equal HEAD's plus exactly the new assertions. A missed `'J'` in the test
     machinery would make loops vacuous, so it shows as fewer subtests, not as a failure. HEAD's counts come from CI's
     current leg on the last push, or from one local run before the build.

   One honest caveat: the test oracles `loop_oracle`/`groups_of` iterate the groups in sorted letter order (`CJKW`
   becomes `CKPW`). So the oracles' own summation order changes, at rounding level and within their tolerances. The
   library's numbers do not change.

## The "old `J` raises" test

Beside the retired `T`:
- `('JWCa,Caib,JWCi->JWCb', 'not groups')` in `test_malformed_strings_raise_at_every_order`;
- `GroupLengths(J=1, W=2, K=0, C=0)` raises `TypeError` in `test_fields_in_order`.

## The doc entries

- **`glossary.md` Symbols:**
  - the preamble reserves `P`, `Q`;
  - the `P` row is extended (the jet ring and group);
  - new rows: `Q` (reserved, Part 2), `F` (the box `F`, `F_in,i`, and the frame in `P_F`, `T(F)`), `M` (the data
    metric `M: Y* -> Y`, the collision with the manifold), `Y`/`Y*` (`*` the algebraic dual), `γ`, `ℓ` (noting the
    regularizer's `λ`), `[G]`.
  - Written by meaning: no function that does not yet exist is named.
- **`naming_conventions.md`, "Index letters":** `P` and `Q` beside `C`, `W`, `K`, and `[G]`.

## What I would change, for a decision

1. **`J = Jets(order)`** in `test_sampling_definitions.py`: about 140 uses of a test-local `J` naming the jet *ring*,
   which the plan's count misses. Q12's aim is that `J` means only the Jacobian, so I recommend renaming it to `ring`
   (`P` is taken there by the problem, `jets` by a module).
2. **Add a guard** in `TestJetLiteralsStayInJetContract`: no subscripts literal in the library names `J`. `contract`
   accepts any letter as a group, so a stale `J` string passed to `contract` would run silently. About 6 lines.
3. Add `m` (the number of jet variables) to the glossary's `m` row. `naming_conventions.md`'s lowercase-`t` jet axis
   describes the old derivative code; I add one sentence pointing to `P`, and leave the paragraph to P19/P20.
4. Keep the script and its edit log under `dev/graded_messages/implementation/m0/`, so that the reviewer can rerun the
   round trip.
