# M0 design review: the implementer's note on the `J -> P` and `J_in_i -> F_in,i` rename

Reviewer, 2026-11-22, on `M0_DESIGN_NOTE.md`, read only, at `368ba3b4`. I read it against the plan's §7 (M0) and
9.0, Q12, the round-6 review's minor points 3 and 4, my orientation list, and the code. I traced the engine path
`probe` → `sampling_variables` → `sampling_steps` → `jet_contract` → `contract` → `einsum` and checked the note's
claims with greps and AST scans. I ran no tests.

**Verdict on the design: sound; build it, with the two should-fix points below folded in.** The map is complete in
kind, the spelling is right, the evidence is in three independent layers, and the six decisions are all agreeable.
Two points strengthen the evidence where a missed rename would be silent (findings 1 and 2); the rest are minor.

## What I checked, and what holds

- **Behaviour (the note's evidence 3).** Confirmed from the code:
  - `_expanded_subscripts` expands groups into fresh letters in order of first appearance, so the expanded `einsum`
    string does not depend on the group's letter;
  - `_pairwise_path` reads only the sizes of index-set intersections;
  - `_jet_roles` chooses carrier and pair by jet sizes and position;
  - `_jet_group_lens` sorts its keywords by letter (`C J K W` becomes `C K P W`). That changes only the order of the
    rows in `_solve_group_lengths`, whose solution is unique, and of the cache key.
  - The uppercase fallback pool is the one place where the letter matters, and the library's strings never reach it.

  So the calls are identical, which is what makes numpy's own (size, letter) ordering of intermediates identical too.
- **The four "group-only" files have no Jacobian `J`.** `contractions.py`, `_jet_contract_cases.py`,
  `test_jet_contract.py` and `test_contractions_interpreter.py` contain no `Jacobian`, `J^`, `Jᵀ`, `J.T` or `J @`.
  So the rule "every bare `J` there is the group" is safe.
- **The regex assertions follow.** `test_jet_contract.py:535` (`'entries, but lens.J'`) and `:577` (`term 'JWCi'`)
  are in a group-only file, so they are renamed with the messages they match. The two `'JtJp' needs ...` asserts are
  the Jacobian and stay.
- **Outside the code.** No doc page, example, `CLAUDE.md`, CHANGELOG entry or T3Polynomial file names a `J` group,
  `len_J`, `GroupLengths` or `J_in`, and `v2026.5.0` has none of them. The `J`s left in the pages are the Jacobian.
- **`F_in_i` (my orientation item 5): agreed.** A comma cannot go in an identifier, and in `elm_shape=F_in,i+W+(Ni,)`
  or `min(F_in,i, F)` it would read as the comment grammar's separator. Same width as `J_in_i`, so the aligned
  columns stay put. `F_in,i` only in the glossary's prose, which states the mapping: right.
- **The completeness classes (item 4).** The map covers every class I had found:
  - the packed `(d,)+J+W+(N,)` comments (34 lines with `J the largest J_i`, by my count; the note says about 25);
  - the shape family `J + W + [K] + C`;
  - the bare locals meaning jet sizes (`sampling_variables.py:1188-1189`, `ut3_operations.py:159,171`);
  - the test locals;
  - the doctest annotations and messages in `contractions.py`, by the group-only rule;
  - and the jet ring `J = Jets(order)`, which I had missed.

  The residual scan (evidence 2) is the right closing check. Prose sites the shape-grammar rule will not catch,
  because `J` is not followed by `+`, are for its list:
  - `sampling_variables.py:219` ("with ``J`` its declared jet sizes");
  - `tests/test_sampling_variables.py:328` ("``J`` the input's jet sizes clipped to ``F``").

  The Jacobian `J`s in `utv_sampling.py:93,109,122,134` and `apply.py:225` ("the bare ``J``", "every ``J`` /
  ``Jᵀ``") must stay, and the rules as written leave them alone.

## Findings

### 1. SHOULD FIX: compare subtest counts per test, not in total, and give the source-scanning tests a floor

**Where:** the note's evidence 4, "passed and subtests must equal HEAD's plus exactly the new assertions".

**The risk is real.** Several tests select their cases by the letter:
- `_jet_contract_cases.py:77, 82, 87`: `'J' in t`, `'J' in output_term(...)`;
- `test_contractions_interpreter.py:763`: `test_every_J_literal_is_a_jet_contract_literal` scans the library for
  strings with `'J'`.

A filter left at `'J'` gives zero cases, or misclassifies a product as a readout, and the test passes.

**Why a total is weaker.** A total over about 49,000 subtests can hide a small loss, for example one helper's
misclassification moving cases from one test to another. One string-literal guard losing its 150-odd subtests would
show in the total, but per test it shows by name.

**What would resolve it.**
- Take the counts per test node: `pytest -v` lists each `SUBPASS` with its node id, so the log can be tallied.
- Compare them node by node; the only differences allowed are the new tests.
- Add a floor where a test's cases come from a scan: `test_every_P_literal_is_a_jet_contract_literal` asserts that it
  found at least one literal, or better, exactly as many as `scan_literal_jet_call_sites()` returns. A missed filter
  then fails at any time, not only on the day of this comparison.

### 2. SHOULD FIX (decision 5): take HEAD's counts from a local run of the archived HEAD, not from CI

- CI's current leg is a different machine. Its skips and failures can differ from the laptop's (`BASELINE_laptop.md`
  records failures that are machine-specific), and a CI log is not verbose enough for finding 1.
- So run the suite once on a `git archive 368ba3b4` copy, `-n 2`, with `-v`, before the build, and keep the log.
- Its per-test counts are then the reference for finding 1, and for my own check, so I do not run a second baseline.
  Tell Claude when it starts and ends, so that we do not overlap.

### 3. MINOR: say that `GroupLengths` keeps `P` as its first field (my item 6)

The field order is the meaning of a positional construction (`test_jet_contract.py:820`, `GroupLengths(1, 2, 3,
4)`). A rename that also reorders, for example alphabetically, would swap counts silently. The existing
`test_fields_in_order` pins it, and after the rename it asserts `['P', 'W', 'K', 'C']`. State this in the note: `P`
first, the positional test kept as is.

### 4. MINOR: widen the behaviour battery a little

The note lists orders `()`, `(2,)` and `(1, 1)`. I would add:
- `(1,)`: at order 1 a two-slot leaf ties with a full jet, and the `nu` strings' carrier changes (P4a's recount);
- an unequal order in two variables, `(2, 1)`;
- `K ≠ ()` and `C ≠ ()`, and `sum_over_samples=False`;
- the jax path once, jitted. The log must then capture `jnp.einsum` too.

These are the places where the role choice and the expanded strings are most varied. It costs a few lines in the same
script.

### 5. MINOR: `F_in` gets two meanings; define both once

- `_Setting.F_in` (renamed from `J_in`) is the per-mode tuple of jet sizes.
- `(d,)+F_in+W+(N,)` uses `F_in` for the largest `F_in_i` per axis, the packed size.

Both readings are natural, but write the packed one out once where it is made (`pack_vectors`' docstring: "`F_in`,
the largest `F_in_i` on each axis") and keep the others pointing there.

### 6. MINOR: the test oracles' changes go beyond summation order

The note's caveat (`loop_oracle` and `groups_of` iterate groups in sorted order) is right. More generally, a test
that iterates strings or groups in sorted order and draws from one shared `rng` may now draw its operands in another
order: different inputs, still valid, within tolerance. So the bitwise claim is about the library's outputs only, as
the note scopes it. Say "tests: same cases, same tolerances", not "same test numbers".

### 7. MINOR: `dev/` scripts that use the old names

`dev/graded_messages/scripts/review/term_stacks_check.py` uses the renamed names. It is a record and not run by CI
or the gate, so leaving `dev/` untouched is right. Say in the handover that it no longer runs after M0.

## The six decisions

1. **`J = Jets(order)` → `ring`: agree.** It is the jet ring, the very meaning Q12 takes away from `J`, in about
   140 uses in one file. `ring` is short and free there (`P` is the problem, `jets` a module). The round trip and the
   edit log must cover it like the rest.
2. **A guard that no library subscripts literal names `J`: agree.** It is cheap and static, so it covers paths that
   no test executes. It also covers the case that worries me most: a stale `J` string passed to plain `contract`,
   where `J` is just another group and would run. Put it beside the floor of finding 1.
3. **The two small doc additions: agree.**
   - The `m` row gains "the number of jet variables, `len(order)`". It already lists the maximum order of the `.tex`
     note, so the row then states its own collision, as the table does elsewhere.
   - One sentence in "Index letters" pointing from the old `t`/`trs` paragraph to `P`, leaving the paragraph to P19
     or P20. That is the right scope. The paragraph's "jet-bundle `J^m`" is mathematical notation, not the group, and
     can stay until then.
   - Keep `*` and `[G]` inside backticks so that MyST does not read them as emphasis or a link, and let the gate's
     Sphinx `-W` and page doctests confirm it.
4. **The script and edit log under `implementation/m0/`: agree.** I will rerun the round trip, but I will check
   identity with my own independent pass: an AST comparison of old and new modules under my own map, plus a
   line-by-line reverse mapping of the diff. A round trip shows that the relabelling is consistent; it cannot show
   that it is right or complete. Name nothing there `test_*.py`.
5. **HEAD's counts: a local run, not CI** (finding 2).
6. **The shapes: agree, with the rule written down.**
   - `(d,)+F_in+W+(N,)` is right: the packed input jets are padded to the largest `F_in_i`, not clipped to `F`
     (`ut3_operations.py:159`). See finding 5.
   - `P + W + [K] + C + (size,)` is right where the formula covers variables of different jet sizes (a leaf
     `min(F_in_i, F)`, a full jet `F`). There `P` names the jet axes, as `W`, `K` and `C` name theirs in the same
     comments, and 9.0 defines `P` as "the ring and its group of axes".
   - But the plan's 9.0 also says "shape comments write `F`; strings write the group `P`". State the rule in the
     `naming_conventions.md` entry so that the two do not read as inconsistent: **a signature's concrete shape writes
     the sizes (`F`, `min(F_in_i, F)`); a family of shapes writes the group `P`.**
   - The same rule covers `jet_JX: shape=J+X` (`data_preparation.py:269`), which becomes `jet_PX: shape=P+X`.

## What I will check on the build

As in my orientation, in this order. Nothing heavy runs at the same time as the implementer's runs.
- **Identity**, by my own AST comparison and reverse diff.
- **Calls and values**, by my own battery, with `einsum` logs on both trees (numpy and jax).
- **The per-test subtest counts** against finding 2's reference log.
- **Mutations:**
  - a `'J'` filter left in `_jet_contract_cases.py` and in the source-scanning test (finding 1's floor must catch it);
  - `J` put back into `_JET_GROUPS`;
  - a `J` alias on `GroupLengths`;
  - a stale `J` string passed to `contract` (decision 2's guard).
- **The residual scan**, my own.
- **The doc entries** against plan §7, then the gate.
