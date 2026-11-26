SIGN OFF

# P5 review, round 2

Reviewer, 2026-11-20, against `P5_RESPONSE_r1.md`, the coordinator's ruling on the dtype of held variables, and the
final docstrings. Every finding of round 1 is resolved. Three wording notes below, none blocking.

## Finding 1 (held-shape check on one slot): resolved

`TestChecks.test_a_held_variable_of_a_wrong_form_raises` has the case: `HeldVariables(mu=good, nu=bad)` wanting
`('eta_D',)`, so `nu` is filled second and not read. My mutant (check only the first filled slot) is now caught.

## Finding 2 and the ruling: resolved; the rule is sound

**The rule.** Every output is in the result dtype; the counted inputs include each held variable a row reads *or*
that is wanted; each is cast once; a held variable neither read nor wanted is not counted and not returned. It
follows from O-11 ("every floating input whose entries could affect the output"): a wanted held variable is an
output, so its own entries affect that output. It gives one dtype for the whole returned tuple, which is simpler than
both rules of round 1 (mine left the outputs in two dtypes; the implementer's turned on whether a row read the
value).

The four cases in `TestResultDtype` pin both sides:
- read and wanted;
- neither, in each direction of precision;
- wanted only, which lowers the call.

My mutant (a read held value not cast) is caught.

**The code** (`held_counted = held_names & (read | set(want))`, cast once) is what the rule says.

## The docstrings: final, and right

- **The type fields.** I read the 22 entries in `jets`, `data_preparation` and `t3toolbox.fitting`, and the 31
  return types in `sampling_definitions`. They are now real types, with the meaning moved to the description line:
  - `tuple of NDArray or NDArray`, then "Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array
    otherwise)";
  - `NDArray or tuple of NDArray`, then "One array; with ``i`` None, a tuple of ``d`` of them".

  Nothing is lost, and Sphinx no longer resolves a word of prose as a cross-reference.
- **`docs/conf.py`** is the committed file (`git diff HEAD -- docs/conf.py` is empty).
- **The links claim holds.** In the implementer's rebuilt `gate_P5_sphinx3/sphinx_html` (0 warning lines,
  `results.tsv`), `ManifoldCurvatureData.z` is linked only from its own page (two anchors) and the general index;
  `HeldVariables.z` only from its own page and the index. Before P5 those prose fields linked to
  `ManifoldCurvatureData.z`, a wrong target, so this fixes a latent defect, not only the ambiguity.

## Three wording notes (not findings)

1. **The ruling says "every output", the code says "every output but `w`".** The docstring of `sampling_variables`
   states it (`w` is returned as given), and the reference does the same. If the ruling is copied into the plan's
   P10 or the decision record, write "every output but `w`, the sample, which is returned as given". Otherwise the
   two will read as a contradiction later.
2. **`sampling_variables`' docstring still says "Held variables are used as given and not recomputed".** Since the
   ruling, a read held variable is used after the one cast. "Used, not recomputed; their values trusted" would be
   exact. One phrase; P19 can take it.
3. **`use_jax` reads every held value**, counted or not. So a jax held variable that is neither read nor wanted still
   makes the outputs jax, while its precision does not count. That is the plan's step 4 as written (`use_jax` over
   every array input, the dtype over the counted ones), and harmless. Noting it for P10's dtype matrix, which can
   state the case.

## What I ran (single process)

- My five mutants of round 1 on a fresh scratch copy (deleted after): all caught now, the two of round 1's findings
  included.
- The four new test files, `test_jets`, `test_data_preparation`, and the module doctests of `jets`,
  `data_preparation`, `t3toolbox.fitting`, `sampling_definitions` and `sampling_variables`: 175 passed, 1,388
  subtests.
- Floor env (`tttt`): `test_sampling_variables` and `test_sampling_graph`, 25 passed.

The working tree is as I found it, apart from this file. Nothing of mine is left running.
