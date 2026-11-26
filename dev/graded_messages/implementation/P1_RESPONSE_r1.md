# P1, the implementer's answers to the review of round 1

To `P1_REVIEW_r1.md` (verdict NOT YET, one should-fix finding), with Claude's instructions on it: decisions C1
and C2, the hold on C1, the maintainer's decision that replaced C1 ("S should go"), and the addendum on marking
the contraction order for Part 2. 2026-11-17.

Every finding is accepted. Nothing is "not accepted". One instruction (C1) was carried out, reverted on the
hold, and then replaced by the maintainer's decision, which is carried out.

## The findings

**1. Should fix: the dispatch tests do not notice jax-ness inferred from a subset of the operands. Accepted.**
`tests/test_dispatch.py`, `test_jit_jet_contract`: for each case, order, form and dtype, and for EACH operand
position `p` in turn, the call is made under `jax.jit` with operand `p` a numpy array closed over (outside the
traced arguments) and the others traced; the result must be jax, of the dtype, and numpy's numbers; the same
call is also made eagerly. The eager call with one numpy operand that the review named is replaced by these.
Evidence: the mutant `get_backend(True, tree_contains_jax(operands[:1]))` in `jet_contract` now fails 6 subtests
of that test (R1 below); before, it passed everything.

**2. Minor: the closure assertion is vacuous. Accepted.** `test_an_equal_record_gets_the_same_scan_body` now
asserts that the body's closure has exactly five cells and that each is a `str`, an `int` or a `dict` of `str`
to `int` (for a product string and for a readout string). That no ARRAY can reach the body except through the
carry and the scanned input is a property of the factory's signature (it is handed a string and the record,
never an operand), and the test's docstring says so; the eager test in `test_dispatch.py` (a second call makes
one `contract` call only) is what shows that the body is found again.

**3. Minor: the ride case is not in the jit test. Accepted.** `JET_CASES` now has five strings: a ride
(`'TWKCo,Cio->TWKCi'`), an affine leaf, two full jets, a readout with an affine leaf, a readout of three full
jets.

**4. Minor: which jet carries is not pinned. Accepted as instructed: no test, a comment.** The values cannot
see the choice, so no test was invented. `_jet_string_case` now says at `carrier = full[0]` that the choice is
free (the product is the same either way, to rounding; only the split of the work differs) and that the first
is taken so that it is one rule read off the string; the same at the choice of the pair.

**5. Minor: the names `_JetPlan`, `_jet_plan`, `plan`. Accepted.** Renamed `_JetStringCase`,
`_jet_string_case`, local `case`; the record's field `case` became `kind`. No occurrence of "plan" is left in
the new code. `_jet_product_scan_body` became `_jet_scan_body` (it now serves the readout too).

**6. Minor: no shape comment on the return. Accepted.**
`) -> NDArray:   # the output term's shape. Its T axis, if it has one, has order+1 slots`.

**7. Minor: the order-axis check is uneven. Accepted.** The check now runs wherever `lens.T == 1`, before the
ride branch: an `A` axis has 2 slots, every `T` axis the same `order+1 >= 2`. One message for all three failures
(`operand p (term '...') has n slot(s) on its order axis -- ...`). A ride is still exactly one `contract` call.
`test_order_axes_of_the_wrong_length_raise` has 8 cases x 2 forms, three of them rides. The docstring states the
rule.

**8. Minor: a list operand. No change, as instructed.** `contract('ab,bc->ac', [[1., 2.], [3., 4.]], ...)`
raises `AttributeError: 'list' object has no attribute 'ndim'` (measured). `jet_contract` raises the same error
for the same reason (it reads `.ndim` in its rank check). No conversion was added.

**9. Minor: the site group. Resolved by the maintainer's decision: there is no site group.** See C1.

## Claude's decisions

**C1 (the site group): replaced by the maintainer's decision of 2026-11-17, 02:23, "S should go". Done.**
The history, so that the record is exact: (i) Claude's C1 (a string naming `S` beside an order group raises; `S`
with no order group goes through with `lens.S`) was carried out; (ii) Claude then put it on hold and I reverted
exactly that part to the round-0 state; (iii) the maintainer decided that the site group goes, and that is what
the tree now has:
- `GroupLengths` has four fields, `T`, `W`, `K`, `C`, in that order. `S` is gone from the class, its docstring
  and doctests, and from every construction (6 in the library's doctests, 18 in the tests).
  `GroupLengths(S=0, ...)` is a `TypeError`, pinned.
- The groups of a `jet_contract` string are `T`, `A`, `W`, `K`, `C`. A string that names `S` raises the
  unknown-group error (`group(s) ['S'] are not groups of a jet_contract string -- the groups are T, A, W, K, C,
  the ones GroupLengths counts`), at every order, with or without an order group, before or after it. Three
  such strings are in `test_malformed_strings_raise_at_every_order`.
- `test_the_site_group_is_one_more_group` is removed.
- The rule that the order group leads its term stays.
- Nothing in the code, the docstrings or `tests/_jet_contract_cases.py` speaks of a site group, except one
  comment in `contractions.py` and one sentence in the helper's docstring saying that the uniform layout's site
  axis is not a group of these strings (a term is written for one site's arrays, which is what a step sees on
  both layouts).
- The group `A` is exactly as it was.
- Mutation: `S` accepted as a group again (`_JET_GROUPS` with `S`): 6 subtests fail.

For the record of the hold, the behaviour of the round-0 tree (no longer the tree's), measured at `T=0` and
`T=1` alike: `S` with no order group was an ordinary `contract` call with `len_S = lens.S`; an order group with
`S` after it (`'TSWCi,SCio->TSWCo'`, and a product and a readout of that form) was accepted and computed; `S`
before the order group (`'STWCi,SCio->STWCo'`, the layout of row 5.4) raised "the order group must lead its
term".

**C2 (a readout that takes a full jet one coefficient at a time follows the form): done.**
The algebra, checked by a script against explicit loops at orders 1 to 16 before it was written (largest
relative error 5.8e-16; the two wrong variants, the carried jet shifted down and the second jet not turned, were
both off by more than 1e-3). With pair `a`, `b` and the scanned jet `c`:

    top = sum over i+j+k = order of <a[i], b[j], c[k]>
    step k:  carried[t] = a[t-k] (zero for t < k),   turned[t] = b[order-t]
             sum over t of <carried[t], turned[t], c[k]>  =  sum over i of <a[i], b[order-k-i], c[k]>

So `a` (the first of the pair) rides the carry and is shifted one slot UP per step; `b` is turned once; the
zeros the shift brings in remove `t < k`, and what leaves at the top are the `a[i]` with `i > order-k`, which no
`b` could complete. Every step has the same shapes.
- Code: under `'scan'`, `_jet_scan` (coefficient 0 by one `contract` call, then `xscan` over the rest) with the
  same memoized body as the product. Under `'loop'` the readout is as it was: slices, the triangle only. A
  readout with no third full jet is unchanged and the same under both forms.
- Beyond the 41 (four full jets; an affine operand beside a third full jet): the first full jet beyond the pair
  is scanned and the others are taken by a Python loop, the carried jet starting shifted by the orders already
  taken. Tested against explicit loops (`test_more_jets_than_the_41_have`).
- Tests: both forms against the reference for the four `H` strings at orders 1 to 16, float64 and float32,
  dtype kept (`TestTheTwoForms`, now 16 strings); under jit exactly one `scan` in the jaxpr and the same number
  of lines at orders 2 and 8 for `'scan'`, growth for `'loop'`, and the eager scan traces its body once
  (`test_jet_contract_scan_is_one_lax_scan`, for the product and for the readout); the sharding sweep covers
  the scanned readouts (164 compiles, zero all-gathers); on numpy the values are the reference's.
- The docstring and the block comment say what each form does for a product and for a readout.

**The addendum (the order of the contractions inside one step): marked, not fixed.** `contract` and
`_pairwise_path` are untouched. The handover has the section "Carried to Part 2" (section 5). In short:
- the default of `jnp.einsum`'s `optimize` in the installed jax (0.12.0.1) is `'auto'`; for the step string
  `'TWCa,Caib,WCb->TWCi'` `jnp.einsum_path` reports `[(1, 2), (0, 1)]` (the core with the coefficient first);
- numpy is forced to `('einsum_path', (0, 1), (0, 1))` for every product step. For 16 of the 20 product strings
  operands 0 and 1 are the carried jet and the core. **Not all 20:** for the four `nu` strings with an affine
  leaf (`'Caib,AWCi,TWCb->TWCa'` and its three `K` forms) operands 0 and 1 are the core and the coefficient,
  which is the cheap order;
- one thing the finding did not say: jax's path depends on the sizes. On the later steps of `'loop'`, where the
  carried slice is short (down to one slot), jax pairs differently, and the same holds for the pair of a readout.
  So the tests look at the steps where the carried jets have all their slots (the first step of `'loop'`; every
  step of `'scan'`);
- the marker test is an expected failure in every run, and an unexpected success (a failure) once the numpy
  order is fixed: checked in a scratch process with `_pairwise_path` patched.

## D. The `HISTORICAL` table

Handover, section 6, item 1: what the scan collects, the table's format (one letter or none), the three tests
that read it, the 12 strings with their smallest sufficient sets of lengths (ten need two, one needs three; I
had not listed the sets before), the tally of the other 29, and two ways the format could change.

## The reviewer's four uncaught mutations, tried again

| Mutation | Round 0 | Now |
| --- | --- | --- |
| R1: jax-ness read from the first operand only | not caught | **caught**: 6 subtests of `test_jit_jet_contract` fail (the jit calls with a numpy operand closed over) |
| R2: the rank check skipped at order 0 | not caught (`contract` raises with a message that also says `ndim`) | **caught**: 2,051 subtests fail; `TestStrictness` now requires `jet_contract`'s own message (`operand p (term '...') has ndim`) at every order |
| R3: the last full jet carries instead of the first | not caught | **not caught, and not meant to be**: equal values; finding 4, a comment in the code |
| R4: the scan body dispatches on the sum alone | not caught | **not caught: an equivalent mutant.** The sum is the result of `contract` over all the operands, so it is jax whenever any operand is; there is no input on which the two differ |

The four were tried again on the final tree (after the site group was removed): R1 6 subtests fail, R2 2,049,
R3 and R4 none, as in the table (2,051 was the count before the site-group test was removed).

New mutations for what round 1 added, all caught: the scanned readout's second jet not turned (260 subtests);
`'scan'` never scanning a readout (2, the jaxpr test); `'loop'` scanning a readout (1, the jaxpr test); the
carried jet's start shift dropped (3, the test beyond the 41); a ride not checked for its slots (6); `S`
accepted as a group again (6).

## The gate

The whole gate on the final tree (`run_gate.sh --detach -n 2 -l /tmp/gate_P1_r2`); every step passes, and the
six slice files had the same checksums after the run as when it started:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1442 passed, 5 skipped, 1 xfailed, 1 warning, 60982 subtests passed   1029.2      3639       4225
    doctest-modules     PASS   243 passed                                                     29.7      1459       6269
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 1.7       236       7422
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               13.5       454       7249
    examples            PASS   14 examples, 0 failed                                           224       361       7343
    sphinx              PASS   0 warning/error line(s)                                       202.6       690       6887

`1 xfailed` is the marker for Part 2 (row 5.5). Against the baseline: 36 more tests, the same 5 skips and 1
warning, no failure. Before it, on the same tree: the fast slice files through `run_gate.sh quick` (57 passed,
1 xfailed, 15,333 subtests), the module doctests of `contractions.py` (3 passed) and the two `jet_contract`
tests of `test_dispatch.py` (2 passed, 164 subtests). Two earlier full runs of this round were stopped by me
before they finished, each because an instruction arrived that changed the tree; neither had reported a failure.
