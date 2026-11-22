# M0 handover: the `J -> P` and `J_in_i -> F_in_i` rename

Implementer, 2026-11-22. Built on `M0_DESIGN_NOTE.md`, the reviewer's `M0_DESIGN_REVIEW.md` and Claude's rulings
(decisions 1 to 6 agreed; the review's findings folded in). Nothing committed.

## 1. What changed

**The rename (scripted).** `dev/graded_messages/implementation/m0/rename.py` was run once, forward, on the working tree
at `745231d5`, whose `t3toolbox/` and `tests/` equal `368ba3b4`'s. It made 1,579 edits in 37 files, every one logged
in `m0/edit_log.tsv` (file, line, rule, old, new). The rules are in the script's docstring.

| What | Before (HEAD) | After |
| --- | --- | --- |
| grouped subscript literals naming the jet group | 205 with `J` | 205 with `P`, 0 with `J` |
| `GroupLengths(J=…)` / `lens*.J` | 25 / 9 | `P=` 25 / `.P` 9; `P` stays the **first** field (`test_fields_in_order` pins `['P', 'W', 'K', 'C']` and the positional `GroupLengths(1, 2, 3, 4)`) |
| `len_J` (`pack_vectors`, `pack_if_ragged`, the engine, `uniform_fitting`, tests) | 33 lines | `len_P`, 33 lines |
| suffixed locals (`mu_JWCa`, `w_JWo`, `jet_JX`, … 20 names) + `factors_J`, `lead_J` | 377 uses | `…_P…`, 377 |
| `J_in_i`, `J_in_j`, `J_in`, `J_IN`, `ut3_operations`' `J_i` | 180 lines | `F_in_i`, `F_in_j`, `F_in`, `F_IN`, `F_in_i` |
| the packed shape `(d,)+J+W+(N,)` | 32 | `(d,)+F_in+W+(N,)` |
| a shape family `J + W + [K] + C`, `shape=J+X` | 12 | `P + W + [K] + C`, `shape=P+X` |
| the grammar (`_JET_GROUPS`, the parser's `'J'`, its messages, the doctests) in `contractions.py` | `J` | `P` |
| the test machinery's `'J'` literals (`_jet_contract_cases`, `test_jet_contract`, `test_contractions_interpreter`, `test_contractions_sharding`) | 28 | `'P'` |
| the jet ring `J = Jets(order)` in `test_sampling_definitions.py` (decision 1) | 146 | `ring` |
| test locals for jet sizes (`sampling_variables._shape`, `pack_vectors`, three tests) | `J` | `jet_sizes` / `F_in` |
| two test names | `…_J_…` | `test_every_P_literal_is_a_jet_contract_literal`, `test_counts_are_nonnegative_ints_and_P_any_count` |

**Stays:** the Jacobian everywhere (`J`, `Jp`, `JtJp`, `held_Jp`, `J^T`, `Jᵀ`, `J x_dot`), `Jet`/`JET`/`JAX` words,
the alphabet `'IJKL…'` in `sampling_definitions.py`. `dev/` is not touched (so
`dev/graded_messages/scripts/review/term_stacks_check.py`, a record that uses `len_J` and `GroupLengths(J=…)`, **no
longer runs** after M0; it is not run by CI or the gate). No example, page or CHANGELOG entry named any of it.

**Hand edits on top of the script** (the whole of `diff(working tree, forward(HEAD))`):
- `t3toolbox/backend/ut3_operations.py`, `pack_vectors`' docstring: defines `F_in` once, "the largest `F_in_i` on
  each jet axis", the meaning of `(d,)+F_in+W+(N,)` in the uniform signatures (review finding 5).
- `t3toolbox/backend/sampling_variables.py`, module docstring: "`P` its jet axes (of the declared sizes below)" in
  place of "`P` its declared jet sizes" (the rule: a family spells the group), with the paragraph reflowed.
- `tests/test_sampling_variables.py:328`, the same wording in a docstring.
- The new tests (section 2).
- `docs/glossary.md` and `docs/naming_conventions.md` (below).

**The docs.**
- `glossary.md`, Symbols:
  - the preamble reserves `P` (jet axes) and `Q` (dual-jet axes), and says `J` is the Jacobian only;
  - the `P` row leads with the jets (the ring and its group, `P*`), and its old meanings follow;
  - new rows: `Q`, `F` (the box, `F_in,i` written `F_in_i`, the packed `F_in`, and the frame in `P_F`, `T(F)`,
    `Π_{T(F)}`, `F_{x→t}`), `M` (with the manifold collision), `Y*`/`Y` (`*` the algebraic dual), `γ`, `ℓ` (not
    `λ`) and `[G]`;
  - the `m` row gains "the number of jet variables";
  - all written by meaning, no function named that does not exist; `*` and `[G]` in backticks.
- `naming_conventions.md`, "Index letters":
  - one paragraph: `P`, `Q`, `J` the Jacobian only, and the shape rule, **a concrete shape spells the sizes (`F`,
    `F_in_i`, `min(F_in_i, F)`), a family spells the group (`P + W + [K] + C + (size,)`)**, with `[G]`;
  - one sentence saying the lowercase `t` paragraph describes `sampling_derivatives.py`'s kernels. The paragraph
    itself is left for P19/P20.

## 2. Tests added or changed

- `test_jet_contract.py`, `test_malformed_strings_raise_at_every_order`: `('JWCa,Caib,JWCi->JWCb', 'not groups')`, the
  retired letter beside the retired `T` (3 new subtests, one per record).
- `test_jet_contract.py`, `test_fields_in_order`: `GroupLengths(J=1, W=2, K=0, C=0)` raises `TypeError`, and a record
  has no attribute `J`.
- `test_contractions_interpreter.py`, `test_every_P_literal_is_a_jet_contract_literal` gains a **floor** (finding 1):
  the set of literals its scan finds must equal `scan_literal_jet_call_sites()`. A filter left at another letter
  finds nothing and fails.
- `test_contractions_interpreter.py`, new `test_no_literal_names_the_retired_J` (decision 2): no grouped string literal
  in the library names `J`. `contract` would run one, so this is the static guard. It has its own floor (more than
  100 literals scanned).
- The case helpers of `_jet_contract_cases.py` already had floors: `TestTheStrings` counts 12 products, 18 readouts
  and 8 rides.

**Mutants** (each in a scratch copy, each killed):
- the `'P'` filter of `jet_operands` back to `'J'`: `TestTheStrings` fails;
- the scan filter of `test_every_P_literal…` back to `'J'`: the floor fails;
- `J` added to `_JET_GROUPS`: the malformed-strings test fails 3 times;
- a stale `contract('Cio,Jo->JCi', …)` in `probing.py`: the new guard fails.

## 3. The evidence and the gate

**Collision check.** The inverse rules, run on untouched HEAD (`rename.py <head> inverse … --dry`), find **0** sites.
One pre-existing kind of word matched at first (`CP`, `CPs`: canonical polyadic). The group-word rule now requires `W`
in the word, as every group term of the library has it. The six pre-existing bare `P`s in two group-only test files
(a projector in comments; a local `np.polynomial.polynomial`) are skipped by line.

**Round trip.**
- `forward` on a fresh `git archive 368ba3b4`, then `inverse` on a copy: `diff -r` against the archive is **empty**,
  1,579 edits each way.
- The first attempt failed on the packed `F_in`, which the inverse read as the size word. It was fixed by applying
  the packed rule first.
- The working tree equals `forward(HEAD)` except the hand edits listed in section 1.

**Residual scan** (`m0/residual.py`, on the renamed tree). Every remaining word with a capital `J`:
- 1,349 uses of 108 known non-group words;
- 236 bare `J` and the short words (`Jv`, `Jx`, `JTr`, `Jd`, `DJ`, …). I read them all. Each is the Jacobian (`J^T`,
  `J x_dot`, `J = dense_core_jacobian(…)`, `J = call(…corewise_tangent…)`), except `J_TRAIN`/`J_VAL`/`J_val_verbose`,
  which are the pinned case "J" of `test_optimizers_frontend`.

None is the group.

**Behaviour, bit for bit** (`m0/battery.py`, run on the archive and on the renamed tree).
- Covered:
  - every public variable of probe, apply and entries;
  - manifold frame and corewise point, ragged and uniform;
  - orders `()`, `(1,)`, `(2,)`, `(1, 1)`, `(2, 1)`, with lines and a mode that does not move;
  - `K` = `()` and `(2,)`, `C` = `()` and `(2,)`, `sum_over_samples` True and False;
  - a 3-step `newton_cg` on jetted probe data, ragged and uniform;
  - one `jax.jit` run (x64; uniform probe, order `(2, 1)`, `K = C = (2,)`) with `jnp.einsum` logged during the trace.
- Result: **23,577 output arrays bitwise equal** (dtype and shape included). **154,550 einsum calls** (strings,
  operand shapes, numpy `optimize` paths; 328 of them `jnp`) are **identical**: the two call logs have the same
  sha256.
- The claim is for the library's outputs only. Test oracles that iterate groups in sorted letter order
  (`loop_oracle`, `groups_of`, the sharding sweep) now visit `C K P W` instead of `C J K W`. They run the same cases
  at the same tolerances, but their own summation order and RNG draw order may differ.

**Per-test counts** (findings 1 and 2).
- Baseline: one local run of the suite on `git archive 368ba3b4` at `-n 2 -v`, before any edit, kept as
  `m0/baseline_v.log` (19 MB). Result: 1,897 passed, 5 skipped, 1 xfailed, 90,942 subtests, exit 0 (35:10).
- `m0/tally.py` compares node by node; the two renamed tests are read under their new names. Output:
  `m0/tally_baseline_vs_m0.txt`. **Exactly two nodes differ**, both expected:
  - the new `test_no_literal_names_the_retired_J` (1 passed);
  - `test_malformed_strings_raise_at_every_order`: 45 to 48 subtests (the retired `J` string × 3 records).

  Every other node has the same outcome and the same subtest count.

**The gate** (`-n 2`).

| Step | Result |
| --- | --- |
| suite (run with `-v`, outside `run_gate.sh` for the per-node log) | 1,898 passed, 5 skipped, 1 xfailed, 90,945 subtests, exit 0, 36:44; no failure (the baseline had none either); peak RSS of the parent 4.3 GB |
| doctest-modules | PASS, 321 passed |
| doctest-quickstart | PASS |
| doctest-pages | PASS, 47 files, 0 failures |
| examples | PASS, 14 examples, 0 failed |
| sphinx `-W` | PASS, 0 warning/error lines |

The lowest free memory in the five `run_gate.sh` steps was 9.8 GB; the logs are in the scratch folder
(`…/scratchpad/m0impl/gate/`, its sphinx build deleted).

## 4. Departures and choices

- **Spelling.** `F_in_i` in code, comments and docstrings; `F_in,i` only in the glossary prose, which gives the
  mapping. The packed `F_in` is defined in `pack_vectors`. The engine's per-mode `_Setting.F_in` keeps that name (a
  tuple of the `F_in_i`).
- **Shapes.** As ruled: a concrete shape spells the sizes, a family spells `P`.
- **Bare-`J` size locals.** `sampling_variables._shape`'s `J` became `jet_sizes`, not `P`, since it holds sizes, not
  the group. The test locals became `F_in`.
- **The suite ran outside `run_gate.sh`.** I ran it directly, to keep the `-v` per-node log the comparison needs.
  The other five steps ran through `run_gate.sh`.
- **`baseline_v.log` is 19 MB** under `dev/…/m0/`, as ruled. Compress it before committing if that is too large for
  the repository.

## 5. Found outside the slice

- `dev/graded_messages/scripts/review/term_stacks_check.py` no longer runs (it is a record; untouched).
- `naming_conventions.md`'s lowercase-`t` paragraph still describes the old kernels; it is pointed to `P` and left for
  P19/P20.
