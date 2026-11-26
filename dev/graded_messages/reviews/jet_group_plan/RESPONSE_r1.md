# Response r1 to the critic's first review

Written by the writer, 2026-11-17, against `50e2b62` (the draft `a69e497` plus the review). Every finding is
accepted. Where a finding needed a ruling, I followed the orchestrating session's guidance of the same round; where
the ruling is the maintainer's, it is an open point in section 11.2 of `PART1_PLAN.md`. Scripts of mine are in
`/tmp/jg/` (not in the tree).

A vocabulary change runs through the whole revision (finding 19, and the session's ruling): **jet shape** now means
the tuple of maximum orders, `order`, as the record uses it; **jet sizes** means the sizes of an array's jet axes
(`F = n + 1` for a full jet). The draft's "jet order" is gone from the plan; it is listed as the alternative in N40.
The head paragraph of the plan defines both terms once.

## Blocking

**1. A forgotten `order`.** Accepted. New open point **O-44** with three options, each with what a forgotten `order`
then does: (a) `()` the default everywhere (silent: two samples on a method; a wrong fit that passes `check_batch`
at order 1, and at every order with a curve); (b) no default on `LeastSquaresOracle`, the only object built once per
fit with a jet shape (a `TypeError` at construction; methods silent as in (a)); (c) no default anywhere (a
`TypeError` at every call; every order-0 call changes). It says plainly that the hazard is new with input jets (the
pair `(ww, pp)` used to fail on its form) and that row 6.2 is why the array cannot say it. The tests that pin each
option are named in O-44; the plan adds, whatever the answer, a wrong-against-right doctest on `probe` and `apply`
(P13), one in the oracle's docstring and a default test (P14), and the doctest row of section 6. I would propose (b)
and say why in O-44; I did not pick. The `check_sample` paragraph and section 2.2 now point to O-44, and the oracle's
sketch says its default is O-44.

## Should fix

**2. The jet axes on the uniform layout.** Accepted, reasoned from entry 116. Section 7.2 states the rule: there is no
don't-care slot on a jet axis; the zeros written when modes of unequal jet sizes are packed are coefficients (data,
not padding); `ut3_operations.pack_vectors`, through `pack_if_ragged`, writes them (the twins call it, and `pack_sample` through
`pack_batch`). Checked against its code (`backend/ut3_operations.py`, lines 128 to 150): today it pads the mode axis
only, so P11 extends it to bring every axis but the last to its largest size across the modes, which leaves `W`
unchanged; nothing checks them, and no checker is
proposed, because a packed input with a nonzero there is another valid input jet. P11's garbage test now puts
garbage in the masked slots only (core padding and the mode axis `N`), and a new test shows that a nonzero written
into a packed jet slot changes the result as the reference does with that slot added to the ragged input. "The
engine is strict about it" is replaced by what is checked: the packed sample's shapes against one another, nothing
about values (section 2.2). The sentence it adds to `docs/uniform_equivalence_contract.md` is open point **O-45**,
with the alternative (a mask per mode and jet axis) and its cost, marked as not proposed.

**3. The reference module and `affine_jet`.** Accepted. P4b now has the reference build the two-slot input jet itself
(`xnp.stack([w, p])`) and, for entries above order 0, call `apply.dense_apply` on an input jet made from the unit
vectors it already builds (`_setting`, line 233). P4c says the module reads `order` with a private function of its
own and calls nothing of `jets`. The independence test (`independence_violations`, lines 1064 to 1133) is named
and is unchanged.

**4. The old derivative kinds at order 0.** Accepted. P16 now says: the old plain kinds give `()`; the old derivative
kinds at order `n` give `(n,)`, `(0,)` included, with the evidence (`(1,) + W + (Ni,)`, row 2.7). I checked every
other place an old call is translated:
- P13 (`taylor_to_derivatives('probe', (order,), x.probe(affine_jet(ww, pp), (order,)))`), the examples' dense
  ground truth in P13 and P17, the bridge of P6 and P9, the release notes (`(3,)`), the dry port and section 9's
  table (`(J,)`): all already write `(n,)` from the old integer, which is right at 0. No change needed.
- Section 9's sentence "If O-36 accepts an integer ... `(J,)` may be written `J` throughout" was wrong at `J = 0`
  under option (b) of O-36. It now says `(J,)` is right for every `J` and `J` may replace it only where `J >= 1`.
- The cost is added to O-36 (b).

**5. The jet matrix at `m = 2`.** Accepted. Section 6 adds the case order `(1, 2)`, `d = 3`, input jet sizes `(3, 2)`,
`(1, 1)`, `(2, 3)`: a curve in one variable and a line in the other, a discrete mode, and clipping on the first
axis only (`F = (2, 3)`). It runs wherever the matrix does: P4b, P4c, P5 (listed), P9 (apply on the matrix), P11
(packed, so with per-axis zero coefficients).

**6. P4a's dispatch row.** Accepted. P1's `test_jet_contract_scan_is_one_lax_scan` is kept with its three assertions,
restated: one program size at orders `(2,)` and `(8,)` under `'scan'` and growth under `'loop'`; a ride and a readout
with no third full jet have no scan, and a product with a two-slot leaf has no scan above order 1 and one scan of one
step at `(1,)`; the eager body is one stable object. At `m = 2` neither form scans and the program grows (O-38).

**7. Packed inputs.** Accepted. Section 2.4 says `affine_jet` returns the form it is given: a tuple for a tuple or a
list, a packed `(d,) + (2,) * m + W + (N,)` for packed vectors, with the reason (entry 115's tuple rule is for the
functions whose result depends on `kind`; `affine_jet` has none, and keeping the packed form lets the twins mirror
it as today). `index_to_one_hot` returns a tuple. Section 2.4's paragraph on the twins says they mirror a ragged
`J_in_i + W + (Ni,)` sample or a packed `(d,) + J + W + (N,)` one (checked against `ut3_probe`,
`backend/ut3_sampling.py` lines 81 to 94). Tests: P4b (`affine_jet` on packed vectors equals stacking its ragged
result) and P12 (the twins mirror both forms at `(2,)` and `(1, 1)`).

**8. Open points.** Accepted:
- O-39 is withdrawn: rows 2.1 and 2.2 decide it. Section 2.2 keeps one paragraph saying the slots of an input jet are
  Taylor coefficients, with the rows cited; 2.4 drops "(To confirm, O-39.)"; 11.2 says O-39 is withdrawn.
- O-38 is reduced to the confirmation of a loss: at `m >= 2` row 2.9's one program size is given up until Part 2.
- O-36 has the costs the critic names (the type of `order` deciding `m` under (b), a weaker form of what 6.2 forbids;
  the upgrade at order 0 needing `(0,)`; `(J,)` for every caller under (a)).
- Added: O-44 (the default, finding 1), O-45 (the jet axes on the uniform layout, finding 2), O-46 (whether an
  entries oracle has an `order` field), O-47 (whether the dual-jet and derivative-data functions accept
  `kind='entries'`). Each is proposed, not settled.
- Section 5.2 now says which slices wait on which points and names.

## Minor

**9. The readout claim at order 1, and the recount.** Accepted; the account was false at order 1. Corrected in the
account (section 3). I recounted the expected failure myself with `contractions._pairwise_path` on the step strings
the new rule makes (`/tmp/jg/` script, run in the tree): at orders 2 and above the carrier is P1's, and 16 of the 20
cases (12 strings, `mu` and `nu` each with a two-slot and a full leaf) pair the carried jet with the core first, the
4 `nu` strings with a two-slot leaf being cheap, as in P1; at order 1 the leaf is the carrier in those 4 `nu`
strings and they become costly too (20 of 20). P4a's xfail item says this and pins the counts at order 4, as P1's
test does (`ORDER = 4`, `tests/test_jet_contract.py`).

**10. The sums.** Accepted. The table of 2.3 now gives `r_block_sumsq` and `y_block_sumsq` the shape of the weight;
only the blocks, `f` and `Jp_norm_sq` have no jet axes.

**11. Edge cases of `jet_contract`.** Accepted. A string with `J` in the output and no jet operand, and a readout with
no jet operand, raise at `m >= 1` (rule, check and reason in 2.4; P4a's strictness tests cover them through the list
of what is checked). A partner with more slots than the box is sliced to the box first and is then a full partner.

**12. The head against N39.** Accepted. The head now says what N39 says: `F` the full jet sizes, `J` the jet axes in
general, `J_in_i` an input's.

**13. 5.2.** Accepted. O-37 moved to P4d; the names each slice waits on are listed.

**14. Entries leftovers.** Accepted in P16 (records at `()` for every kind, `(2,)` and `(1, 1)` for probe and apply),
P10 and P11 (entries at `()` only).

**15. Call counts.** Accepted. P4a gives the `ast` counts 55, 4, 2, 2, and says `_jet_contract_cases.py` names them in
its docstring only (my own `ast` count agrees with the critic's). P4d gives `ast` and regex counts for
`test_dispatch.py` (45) and `test_jax_absent.py` (16, inside its subprocess script, which `ast` cannot see), in place
of "about thirty" and "eleven".

**16. The consumer's calls.** Accepted, from an `ast` count over `/home/claude/t3polynomial` (read only, not run):
`fitting.py` lines 384, 385, 597, 598 for `probe_derivatives`; `probe_derivatives` also in `d05`, `d06`, `d09`,
`d11`, `e01`, `g05`, `n01`, `o01`, `s00b`, `s05`, `x02`, `x06`, `x07`; frontend `least_squares_problem` in `d07`,
`d08`, `d11`, `g06`, `g06b`, `g06c`, `h01`; backend `least_squares_problem` with `probe_derivatives_kind` in `g01`;
`probe_derivatives_model` in `x06`, `x07`. Section 9's table has rows for all of them, and its weight expression
handles `None`.

**17. Stale historical texts.** Accepted. P1's test item points at the 41 strings of P1's file as built; P1 to P3 are
marked at their heads as built and rebuilt; the doctest row of section 6 lists P4a to P4d.

**18. P4b's size.** Accepted: medium.

**19. Vocabulary.** Accepted, and done in the plan as the session ruled (see the top of this response). The account's
section 7 lists the terminology among the amendments.

**20. Smaller amendments.** Accepted. The account's section 7 adds: the note "Groups in contraction strings" marked
superseded in place; O-10's answer ("five fields"); row 3.4's "the order" holding for entries only as `()`.

## The points the critic says are the maintainer's

1. The default of `order`: O-44. 2. The jet padding: O-45. 3. The entries oracle's field: O-46. 4. Entries in the dual
and derivative functions: O-47. 5. O-38 reduced to a confirmation; O-39 withdrawn.

## The critic's unverified items

- The counts of tests touched (47 of 56, 39 of 42, 15 of 29): verified by an `ast` walk that lists the test functions
  whose source names `pp`, `order` or `entries`; the plan now says how they were counted.
- The `test_dispatch.py` and `test_jax_absent.py` references: replaced by counts of calls (45 by `ast`; 16 by a
  regular expression over the subprocess script).
- The recount of the expected failure: done (finding 9).
- A nested scan for `m >= 2`: not examined; O-38 no longer proposes it.
- T3Polynomial's weight shapes: read from `t3polynomial/fitting.py` (the comments at lines 285 and 414: `(J+1,)` per
  slot or `(degree+1, J+1)` per mode and slot); its runtime not run.

## Disagreements

None of substance. One remark: under option (b) of O-44 the oracle is the only object built once per fit that takes a
jet shape; `pack_batch`, the checkers and the derivative functions already take `order` positionally with no
default, so (b) touches the oracle alone. O-44 says so.
