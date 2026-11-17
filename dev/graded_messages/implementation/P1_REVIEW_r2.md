SIGN OFF

# P1 review, round 2: `jet_contract` and `GroupLengths`

Reviewer, 2026-11-17. No blocking or should-fix finding remains. Two new minor findings (10, 11), both in the
tests, which can be fixed without another round.

## Findings 1 to 9

| # | Round 1 | Now | How I checked |
| --- | --- | --- | --- |
| 1 | should fix: jax-ness from a subset of the operands not noticed | **resolved** | the mutant `tree_contains_jax(operands[:1])` now fails 6 subtests of `test_jit_jet_contract` |
| 2 | the closure assertion was vacuous | **resolved** | the body's closure is five cells, `int, int, int, dict, str` (read by me); the test whitelists those types |
| 3 | no ride case under jit | **resolved** | `JET_CASES` has a ride, an affine leaf, two full jets, a readout with an affine leaf, a readout of three full jets |
| 4 | which jet carries is not pinned | **resolved as decided** (a comment, no test) | the comment at `carrier = full[0]` and at the pair says the choice is free and why the first is taken |
| 5 | `_JetPlan`, `_jet_plan`, `plan` | **resolved** | `_JetStringCase`, `_jet_string_case`, local `case`, field `kind`; `_jet_scan_body`, `_jet_scan`. Clear, none generic; no "plan" in the new code except the `PART 2` notes |
| 6 | no shape comment on the return | **resolved** | present |
| 7 | the order-axis check was uneven | **resolved** | a one-slot `T` axis on a ride now raises; the mutant that skips the check on a ride fails 6 subtests |
| 8 | a list operand | **no change, as decided** | same error as `contract` |
| 9 | `S` after the order group pinned by a test | **resolved by the maintainer's decision** | below |

## What changed beyond the findings

**The site group is gone.** `GroupLengths` has the fields `T, W, K, C`; `GroupLengths(S=0, ...)` is a `TypeError`;
`'STWa,Sab->STWb'`, `'TSWa,Sab->TSWb'` and `'SWa,Sab->SWb'` (at `T=0`) each raise the unknown-group error. A search
of `contractions.py`, `tests/test_jet_contract.py` and `tests/_jet_contract_cases.py` finds no site group, apart
from the one comment and the one docstring sentence that say there is none, and the tests that it raises. The
mutant that accepts `S` again fails 6 subtests. "The order group leads its term" still holds
(`'WTa,ab->WTb'` raises) and is tested (the mutant without the check fails 4 subtests).

**The group `A` against its definition** (the note "Groups in contraction strings"): the order group in the
two-slot format of a variable of degree at most one; as many axes as `T`; absent exactly when `T` is; no field of
its own. The code gives `len_A = lens.T`, requires 2 slots where `lens.T == 1`, and has no `A` field. The
docstrings and comments say "an affine variable has two [slots]", "the axis count of the order group T, and of the
affine group A", "at order 0 both groups are empty". Nothing contradicts the definition.

**The scanned readout.** Algebra as the response states it, and it is right: with the carrier started `s` slots up
(the orders the looped operands took) and shifted `k` more, `carried[t] = a[t-s-k]` meets `turned[t] = b[order-t]`,
so the orders sum to `order - s - k`.
- My own check, by a different method from round 1 so that it holds at any order (`/tmp/p1rev/check_math2.py`):
  operands as polynomials in `s`, a plain lowercase `np.einsum` at the roots of unity, coefficients by an inverse
  FFT. The 41 strings, with the four `H` strings with a full leaf among them, and 11 beyond (the implementer's
  two; an affine operand first with three full jets; three full jets with the samples kept; two affine and three
  full; four full with an affine), orders 1 to 8, `W` in `()`, `(3,)`, `(2,2)`, `K` in `()`, `(2,)`, `(2,2)`, `C`
  in `()`, `(2,)`, both forms: 5,214 cases, largest relative error 5.9e-15 (3.9e-15 on the four `H` strings).
  Eager jax with x64, orders 1, 2, 5, 8, stacks empty and not: 72 cases, 3.4e-15.
- Under jit, for each of the four `H` strings with a full leaf: exactly one `scan` at orders 3 and 9 under
  `'scan'`, 54 jaxpr lines at both (56 for the string with `K` on the leaf); under `'loop'` no scan and 99 against
  261 lines. No array in either program at order 9 has two axes of length `order+1`.
- `'loop'` does only the triangle: spied, the pair's slices have 4, 3, 2, 1 slots at order 3 and 7 down to 1 at
  order 6. `'scan'` makes `order+1` calls, all on the one derived string, all with the full order axis.
- An equal (string, record) built anew gets the same body object; the closure holds no array.

**The marker for Part 2 (row 5.5).** Honest:
- It fails today for the stated reason: run without the decorator's protection, 16 of the 20 product literals
  have a full-slot step whose first pair is the carried jet with a no-order operand; the four that do not are the
  `nu` strings with an affine leaf, as the response says.
- It turns into a failure when the numpy order is fixed: in a scratch copy I gave `_pairwise_path` a tie-break
  that avoids the carried jet. `tests/test_jet_contract.py` then reports `1 failed` (the unexpected success) and
  no `xfailed`.
- The passing tests beside it: the numpy readout test depends only on the string (the path is a function of
  shared letters), and a mutant path that pairs a readout's jet with the coefficient first fails 4 subtests. The
  jax tests depend on sizes in principle; I ran them at five size sets (bonds 2 to 64, mode rank 3 to 40) times
  three sample counts (2, 200, 5000): on the steps where the carried jets have all their slots, no product step
  pairs the carried jet first and every readout step folds its pair first, in all 15. The restriction to those
  steps is stated in the helper's docstring and in the class docstring's sibling note, and is sound: under
  `'scan'` every step has all slots, and under `'loop'` the first has.
- `contract` and `_pairwise_path` are untouched: the diff of `contractions.py` removes no line.

**The handover's note on `HISTORICAL`.** Accurate: the table maps a string to `''` or one letter; the three tests
that read it are as described; the 12 strings and their smallest sets agree with my round-1 count (ten need two,
`'TWCa,AWKCi,TWCb->KCaib'` needs three).

## New findings

**10. Minor. The start shift of a readout beyond the 41 is tested in float64 only.**
Place: `tests/test_jet_contract.py`, `test_more_jets_than_the_41_have`; the code at
`xnp.zeros_like(carried[:n_slots - 1 - left])`.
Evidence: the mutant `xnp.zeros(carried[:n_slots - 1 - left].shape)` (zeros in the default dtype) passes
`tests/test_jet_contract.py` and the `jet_contract` tests of `test_dispatch.py`. The code is right; no string of
Part 1 reaches that line.
Resolves it: one float32 case in that test, asserting the dtype.

**11. Minor. The expected failure would hide a broken helper.**
Place: `TestOrderInsideAStep.test_on_numpy_no_product_step_pairs_the_carried_jet_with_the_core_first`.
Evidence: `expectedFailure` accepts any failure, and `_first_pairs` on numpy PRODUCT strings is run nowhere else
(the numpy sibling test runs it on readouts only). If the helper ever raised for a product string, the marker
would stay "expected" after the order is fixed.
Resolves it: a plain passing test on numpy for the four `nu` strings with an affine leaf (the first pair is the
core with the coefficient), which runs the helper on numpy products and pins "16, not 20".

## Mutations (round 2, in a scratch copy under /tmp; 24 tried)

Caught by `tests/test_jet_contract.py`: the rank check skipped at order 0 (2,049 subtests; round 1's second
uncaught one); the readout's wrong jet shifted (260); the shift the wrong way (1,028); no shift (1,028); the
second jet not turned under `'scan'` (260); the scan one step short (1,060); coefficient 0 taken twice (1,060);
the start shift dropped (3) and one short (3); `S` accepted (6); the lead check dropped (4); a ride's slots
unchecked (6); the body not memoized (1, and 2 in dispatch); the loop's part in the low slots (1,281); the loop's
pair not turned (696); the numpy path fixed (1, the unexpected success); the numpy readout path spoiled (4).
Caught by `tests/test_dispatch.py` only: jax-ness from the first operand (6); a readout scanned under `'loop'`
(1); a readout never scanned (2); a product always looped (2) or always scanned (1); `'scan'` as a Python loop on
jax (4).

Not caught:
- zeros in the default dtype in the start shift: finding 10.
- (from round 1, unchanged and not meant to be caught) the last full jet carrying instead of the first, and the
  scan body dispatching on the sum alone: both give the same values on every input.

## The gate

I did not run the whole gate again. The implementer's run (`/tmp/gate_P1_r2`) began at 02:32:29, and the six
slice files were last written between 02:26:00 and 02:30:01, so it ran on the tree as it stands; its table reads
1,442 passed, 5 skipped, 1 xfailed, 1 warning, 60,982 subtests, and every other step passes with sphinx at 0
warnings. On the working tree I ran:

- `pytest --doctest-modules t3toolbox/backend/contractions.py`: 3 passed.
- `pytest tests/test_jet_contract.py tests/test_contractions_interpreter.py`: 47 passed, 1 xfailed, 14,233
  subtests.
- `pytest tests/test_dispatch.py -k jet_contract tests/test_contractions_sharding.py`: 3 passed, 328 subtests
  (the two `jet_contract` dispatch tests and the sharding file's `jet_contract` test under that selection).

## What I ran besides

`/tmp/p1rev/check_math2.py` (the FFT check, the jaxpr counts and sizes, the spied slices, the body identity and
closure, the `S` and order-axis errors); `/tmp/p1rev/mutate2.py` (24 mutations of a copy of the tree, against
`tests/test_jet_contract.py` and, where it bears, the `jet_contract` tests of `test_dispatch.py`); a script that
runs the marker's helper directly (the count of 16, and the jax tests at 15 size sets).

The working tree is as I found it apart from this file; the scratch copy is removed; nothing is left running;
nothing committed.
