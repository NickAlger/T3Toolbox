NOT YET

# P4a review, round 1: `jet_contract` over one jet group `J`

Reviewer, 2026-11-19, on the working tree of `graded-messages` (the implementer's uncommitted changes on
`96196427`). One should-fix finding (1): a gap in the jit tests, not a code defect. Three minor ones. The code is
right as far as I can test it. The mathematics agrees with an independent oracle to 2e-15. 25 of 26 code mutations
are caught; the one missed is equivalent. The floor env passes the whole file.

## What is confirmed

- **Against the plan and the rulings.** `GroupLengths(J, W, K, C)`, with `J` any count and the other counts
  validated. `jet_contract(subscripts, *operands, lens, order, over_orders)`: the last three keyword-only, none
  with a default. Each ruling of 2026-11-19 is carried out:
  - a string with no `J` raises at every `m`, `m = 0` included;
  - "most slots" is counted after clipping to the box;
  - with two or more partners under `'scan'`, the first full partner is scanned and the rest looped;
  - an unreachable corner gives zeros of `contract`'s shape, dtype and backend;
  - `order` is read by `_read_order` before any cache lookup: a tuple only, of `numbers.Integral` entries that are
    not `bool`, normalized to Python ints;
  - the jax optimizer pin accepts `{'auto', 'optimal'}`.

  The markers `PART 2 (row 5.4, jets in several variables)` (in the block comment and at the line where `scanned`
  is chosen) and `PART 2 (row 5.5)` (at the product step) are in place. Item 4 of "Carried to Part 2" is in
  `TARGET_DECISIONS.md`.
- **Every branch, read by hand.**
  - **Ride:** the operand is sliced to the box and the result padded on the jet axes only.
  - **Product:**
    - The all-zero offset is always in the box, so the result is never `None`.
    - The carrier is sliced to `F - offset`.
    - Under `'scan'` the carrier has exactly `F` slots: it has at least as many clipped slots as the scanned
      partner, which has `F`. So the shift does the truncation.
  - **Readout fold:**
    - `k` runs over `[max(0, c-sb+1), min(c, sa-1)]` on each axis.
    - The two slices have equal lengths, so einsum cannot broadcast a jet axis of size 1.
    - No slot beyond the corner is read.
  - **Scanned readout:**
    - The pair is cut to the box *before* it is flipped, which avoids the trap I named at orientation.
    - The pre-shift is `n - c`, and the step algebra gives `i + j = c - k`.
  - **Fallback:** an empty sum by one `contract`.
- **Form.**
  - Dispatch comes from all the operands (`tree_contains_jax(operands)`), and the scan body re-derives `xnp`.
  - No `np.` touches an operand. Zeros come from `zeros_like`, `pad` and `concatenate` in the operand's dtype.
  - The scan body is a memoized module-level factory. Its key `(step string, record, carried, scanned)` is
    everything it closes over (5 static cells).
  - Python 3.9-safe.
- **The tests' helpers.**
  - `reference` loops over every multi-index of every jet operand, slots beyond the box included, and accumulates
    where the sum is in the box (or equals the corner). It uses no `contract` or `jet_contract` and shares no step
    with the algorithm (no carrier, no pair, no slicing).
  - `library_shapes` is the plan's table of declared jet sizes (section 2.3), not a rule of the code.
  - P1's counts are pinned by a hard-coded table, independent of `n_contractions` (but see finding 2).
- **The departure** (an input term of `J` alone raises, decided from the string): I agree with it. It keeps the
  rule that a string is right or wrong at every order, and `'JW'` with `W = 0` writes the scalar polynomial (the
  `polymul` test does so).

## Findings

**1. Should fix. Two paths of `jet_contract` are never compiled under jit: the lone readout and the
unreachable-corner fallback.**
- *Place:* `tests/test_dispatch.py`, `test_jit_jet_contract`. Its five shapes at `(3,)`, `()` and `(1, 2)` never
  reach either branch:
  - `roles.carrier is None`: one jet operand, no `J` in the output;
  - `top is None`: no combination reaches the corner, so the pair falls back to empty slices and the looped
    coefficients to zeros.

  Ruling 5 names the backend of these zeros, but only their numpy type is tested
  (`test_a_box_or_a_corner_beyond_every_operand`).
- *Evidence:* three mutants that put host numpy into these branches. The mutated code stays correct on numpy and
  breaks under a trace:
  - `__import__('numpy').zeros_like` in place of `xnp.zeros_like` for the lone operand;
  - the same for the fallback's looped coefficients;
  - `numpy.asarray` on the fallback's empty pair slice.

  All three pass `tests/test_jet_contract.py`, `TestJetContractDerivedStrings` and both `jet_contract` tests of
  `test_dispatch.py` (50 passed each). On the mutated copy, my `jit_paths.py` raises
  `TracerArrayConversionError` at the first of these shapes. The code as written is fine: on the real tree all six
  of my extra shapes jit, return jax arrays, keep float32 and match numpy, with one operand closed over as numpy
  too. The point is that a later edit there would not be seen.
- *Resolves it, cheaply* (as you asked: one shape each, no cross-product). Add two rows to `test_jit_jet_contract`,
  at one jet shape, one form and float32 only, with one operand closed over as numpy:
  - `'JWCi,Cio->WCo'` with jet sizes `(2,)` at `(3,)`: a lone readout with an unreachable corner, which takes the
    `zeros_like` branch;
  - `'JWCa,JWCi,JWCb->Caib'` with jet sizes `(2,)`, `(2,)`, `(2,)` at `(6,)`: the fallback.

  If a reachable lone readout is wanted too, `(4,)` at `(2,)` costs one more jit. Not needed: a pre-shifted carrier
  (a product with two partners, a readout with a looped and a scanned operand) runs through `_shifted_up`, which
  the scan body already compiles.

**2. Minor. `n_contractions` restates the code's own rules, yet both docstrings say it is "written from the plan,
not from the code".**
- *Place:* `tests/_jet_contract_cases.py`, `n_contractions`; `test_the_number_of_contractions_follows_the_rule`.
- It repeats `_jet_roles` and the loop: carrier and pair by clipped slots, the first full partner scanned, the skip
  of a combination whose pair ranges are empty, and `max(count, 1)` for the fallback. The skip and the fallback
  come from the rulings and the code, not from the plan's text. So the test pins today's behaviour, which is
  useful, but it is not independent evidence.
- The independent pin is the P1 table in `test_one_variable_with_the_library_shapes_makes_the_contractions_of_today`,
  and my own count (below) agrees with it.
- *Resolves it:* say "a regression pin of the rule as built", or similar.

**3. Minor. One signature breaks the column rule.** `_jet_roles`: the `jet_sizes` line pushes its comment past the
column of the other two (`docs/contributor/signature_style.md`: when the spread of the types is large, group them,
or put the long type's comment on its own aligned column).

**4. Minor. The docstring of `jet_contract` states "A string must have `J` on some operand" but not the rule on an
input term of `J` alone.** The error message says it well; one clause in the docstring would match it.

## Mutations (scratch copy of `t3toolbox/` + `tests/`, `python -B`, `PYTHONDONTWRITEBYTECODE=1`; deleted after)

Each mutant ran against `tests/test_jet_contract.py` and `TestJetContractDerivedStrings`. The dispatch-type
mutants also ran against the two `jet_contract` tests of `test_dispatch.py`.

**Caught (25):**
- the fold not flipped; flipped on the first jet axis only; its range's low and high bounds each off by one;
- the scanned operand not clipped, in a product and in a readout; the scanned pair flipped before it is cut; the
  carried member of the scanned pair not cut;
- a product's part placed at offset 0; the box check off by one; the ride not sliced;
- the carrier ranked by unclipped slots; any two-slot partner scanned;
- the readout's and the product's pre-shifts zeroed;
- the lone readout without its range check; the zero fallback removed; the shift's zeros in the default dtype;
- dispatch read off the first operand (caught by `test_dispatch.py`);
- `order` not normalized; `bool` accepted; its length unchecked;
- `J` alone accepted; no `J` accepted; a jet axis of size 0 accepted.

**Missed (1, equivalent):** the pair ordered by size instead of string order. That only swaps which member is
carried and which is flipped. The fold is symmetric, and the scanned readout is the same sum.

**Missed (3, finding 1):** the host-numpy mutants above.

## Independent checks (mine, not the implementer's tests)

- **The mathematics.**
  - *How:* `oracle_jet.py`. Each jet operand is clipped to the box and evaluated on a grid of roots of unity, with
    `M_j = 1 + ` the sum of the clipped degrees along each jet axis. The evaluated operands are contracted
    pointwise by one lowercase `np.einsum`, with the groups expanded by hand. The coefficients come back by an
    inverse DFT on each axis, then are truncated to the box or read at the corner. The oracle itself is checked
    against `numpy.polynomial.polymul` and a double loop.
  - *Coverage:* the 29 strings, derived from P1's 41 independently of the implementer's file, plus six beyond them:
    a three-jet product, a three-jet product through a core, a product through a lowercase core, a lone readout, a
    four-jet readout and bare pairs. At `(1,)`, `(3,)`, `(1, 2)`, `(2, 0)` and `(1, 0, 2)`, with jet sizes per
    axis drawn from `{1, F-1, F, F+2}`, with stacks, under both forms.
  - *Result:* 1,296 calls per seed, seeds 0, 1 and 2. Worst relative error 6.9e-16, 1.1e-15 and 2.3e-15.
  - *Slots beyond the box:* the same calls with NaN in every slot beyond the box are bit-identical and NaN-free.
- **Edge cases.** All as defined:
  - an unreachable corner gives exact zeros, float32 kept;
  - a size-1 jet axis folded against a larger one at `m = 2`, at three corners;
  - an output box larger than both operands;
  - a ride sliced on one axis and padded on the other.
- **Counts at `m = 1` against P1** (worked out by hand: ride 1, two-slot leaf 2, two full jets `n + 1`, H with a
  full leaf `n + 1`, H with a leaf 2, V 1): all 41 P1 shapes at orders 1, 2 and 4, under both forms, by a spy on
  `contract`. All agree.
- **jit** (`jit_paths.py`): the lone readout (reachable and not), the pair fallback at `m = 1` and `m = 2`, a
  two-partner product under `'scan'`, and a readout with a looped and a scanned operand all compile. Each gives a
  jax float32 result equal to numpy, also with one operand closed over as numpy.
- **Program sizes** (`jaxpr_sizes.py`, my own, beside the test):

  | Case | Form | Lines at orders 2, 8, 16 | `scan` primitives |
  | --- | --- | --- | --- |
  | `eta` | `'scan'` | 53, 53, 53 | 1 |
  | `eta` | `'loop'` | 77, 257, 497 | 0 |
  | H, three full jets | `'scan'` | 54, 54, 54 | 1 |
  | H, three full jets | `'loop'` | 72, 234, 450 | 0 |

  At `m = 2` there is no `scan` under `'scan'`, and the program grows with the box (167, 257, 467 and 153, 234,
  423), as O-38 says.
- **The floor env** (`tttt`: python 3.9, numpy 1.22.4, jax 0.4.30), on P4a's tree:
  - `pytest tests/test_jet_contract.py -k TestOrderInsideAStep`: 5 passed, 1 xfailed. This includes the two jax
    behaviour tests under jax 0.4.30's `optimize='optimal'`. The pin's restatement is enough.
  - The whole `tests/test_jet_contract.py`: 46 passed, 1 xfailed, in 19.5 s. This covers the zero-length einsum of
    the fallback on numpy 1.22.

## What I ran (single process; nothing else of mine running beside it)

- `pytest tests/test_jet_contract.py` on t3toolbox: 46 passed, 1 xfailed, 12,946 subtests, 27.9 s.
- `pytest --doctest-modules t3toolbox/backend/contractions.py` and `TestJetContractDerivedStrings`: 5 passed, 542
  subtests. The docstring examples are real runs.
- `test_dispatch.py::test_jit_jet_contract`, `test_jet_contract_scan_is_one_lax_scan` and the sharding test's
  `jet_contract` case: 3 passed, 442 subtests, 101 s. The jit test alone takes 59 s.
- The two runs on the floor env above.
- `oracle_jet.py` (three seeds), `jit_paths.py`, `jaxpr_sizes.py` and `mutate.py` (29 mutants), all in
  `/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/reviewer/`.

The full gate was not rerun: the implementer's gate on this tree is green. Finding 1 adds two small rows to one
test, so rerunning the slice's dispatch tests and the module doctests would be enough. The working tree is as I
found it, apart from this file. The scratch copies are deleted, and nothing of mine is left running.
