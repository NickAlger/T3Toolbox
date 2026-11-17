# Review r2 of the jet-group plan change (4f5b2b7)

**Verdict: NOT YET, on one should-fix finding with a two-sentence fix (finding 1).** Once it is resolved as below, I
agree: I would let an implementer start on P4a from this text once the maintainer has answered the open points. No
second full round is needed for that; the minor findings may be fixed or left.

Written by the critic, 2026-11-17. I checked `RESPONSE_r1.md` against `git diff 50e2b62 4f5b2b7` and the code. My
scripts are in `/tmp/critic/`.

## The twenty findings of r1

Each one is resolved, and I checked each rather than taking the response on trust.

- **1 (default of `order`).** It is now O-44, with three options. Each option says what a forgotten `order` does
  and which tests pin it. The doctests that show the wrong call against the right one are in P13 and P14 whatever
  the answer, and the oracle's sketch marks its default as open. No text elsewhere settles the question.
  - Sections 2.2 and 2.4 point to O-44.
  - The sketches keep `order=()` on the sampling functions, which options (a) and (b) share, and 11.2 says so.
  - P14's test is written for either answer.
  - Two gaps remain, under (b) and (c): minor findings 2 and 3 below.
- **2 (jet axes on the uniform layout).** Section 7.2 now states the rule from entry 116: there is no don't-care
  slot on a jet axis. P11's garbage test is restricted to the masked slots. A new test shows that a nonzero written
  into a jet slot changes the result as the definition says. The sentence this adds to the uniform equivalence
  contract is O-45, with the mask alternative and its cost. "The engine is strict about it" is replaced by what is
  actually checked. The writer's reading of `pack_vectors` is right: `backend/ut3_operations.py`, lines 128 to 150,
  pads only the last axis and takes the stack shape of the first vector. The extension belongs in P11, the first
  slice that packs input jets of unequal sizes, before P12's twins and P14's `pack_batch` use it. But its wording
  loses a structural error; see finding 1 below.
- **3 (reference module and `affine_jet`).** P4b and P4c now have the reference build its own input jet and unit
  vectors and read `order` itself. The independence test is named and left unchanged.
- **4 (old derivative kinds at order 0).** P16 now translates the old derivative kinds to `(n,)`, `(0,)`
  included, and the plain kinds to `()`. I checked the other translations:
  - P13's old-data recipe gives jet sizes `(1,)` at `(0,)`, which matches the old axis of length 1.
  - P13 and P17 (examples), the bridge of P6 and P9, the release notes and section 9 all write `(n,)` built from
    the integer, which is right at 0.
  - The `'entries_derivatives'` translation gives `('apply', (n,))`, which is also right at 0.
  - Section 9's sentence on `J` against `(J,)` is corrected.
- **5 (jet matrix at `m = 2`).** The new case is order `(1, 2)` with `F = (2, 3)`, and modes of jet sizes
  `(3, 2)`, `(1, 1)` and `(2, 3)`. It means what it says, and it is legitimate:
  - The `(3, 2)` mode has 3 slots on an axis whose maximum order is 1. That is a valid input whose third slot is
    clipped (`min(J_in_i, F) = (2, 2)`), so it tests per-axis clipping, as the text says.
  - The genuine curve that reaches the outputs is the `(2, 3)` mode's, along `s_2`. "A curve in one variable"
    describes the case as a whole, not the `(3, 2)` mode, whose curve is clipped away. Worth half a sentence; not a
    finding.
  - The reference's grid for this case is `M = (5, 9)`, which is small.
  - Packed on the uniform layout the jet sizes are `(3, 3)`, and the leaves `(2, 3)`.
  - The case runs in P4b, P4c, P5, P9 and P11.
- **6 (one program size).** P4a keeps the three assertions of P1's test, restated: the leaf has no scan at orders
  above 1 and one scan of one step at `(1,)`; at `m = 2` the program grows.
- **7 (packed inputs).** `affine_jet` now keeps the form it is given, and the twins mirror a packed
  `(d,) + J + W + (N,)`. Both are tested in P4b and P12. The reason given for not applying entry 115's tuple rule is
  sound: `affine_jet` has no `kind`.
- **8 (open points).**
  - O-39 is withdrawn, and section 2.2 cites rows 2.1 and 2.2 in its place.
  - O-38 is reduced to confirming a loss.
  - O-36 now carries the costs I named.
  - O-44 to O-47 are new.
  - Section 5.2 lists what each slice waits on.
- **9 (recount of the expected failure).** I checked it against `_pairwise_path`. Every product step string is a
  three-way tie of two shared groups, and the tie goes to the pair `(0, 1)`.
  - In the merged `nu` string `Caib,JWCi,JWCb->JWCa` that pair is the core with operand 1. Operand 1 is a
    coefficient when the full `nu` carries (a two-slot leaf at order 2 or above: cheap). It is the carried jet when
    operand 1 carries (a full leaf, or the leaf at order 1: costly).
  - That gives 16 of 20 at orders 2 and above, and 20 of 20 at order 1, as the plan now says.
- **10 to 20.** All fixed as described:
  - The sums now have the weight's shape.
  - The two edge cases raise, with a reason.
  - The head agrees with N39, and the vocabulary is N40.
  - 5.2 is corrected.
  - The entries leftovers in P10, P11 and P16 are fixed.
  - The `ast` counts are given.
  - The consumer's table is complete, and its weight expression handles `None`.
  - The historical texts and the size of P4b are fixed, and the amendments are added.
  - The vocabulary is consistent through the plan. `grep -i "jet order"` finds only N40's alternative.

## Findings

### Should fix

**1. P11 extends the public `pack_vectors` in a way that silently drops a structural error. The change is not
listed among the changed functions and has no test of its own.**

- *Where.* P11: "P11 extends it to bring every axis but the last to its largest size across the modes, with
  zeros, which leaves `W` as it is, since `W` is the same for every mode." Also 7.2 and the account.
- *What is wrong.* Today a ragged sample whose modes disagree in `W` fails in `xnp.stack` inside `pack_vectors`
  (`backend/ut3_operations.py`, line 150). It is reached through `pack_if_ragged` by every uniform twin
  (`backend/ut3_sampling.py`, lines 78 to 469) and by `uniform_fitting`'s packing (line 380). Raising every axis but
  the last to its largest size would zero-pad a mismatched `W` instead, and the result would be wrong data. "`W` is
  the same for every mode" is what the old code *enforced*; the new wording only *assumes* it. Under "structural
  problems always error", even a backend that gives no guarantees should not turn an error it raises today into a
  silent wrong answer.
- `pack_vectors` is also a public name (`ut3_operations.__all__`) whose contract changes. Section 2.4 does not list
  it. P11 has no "What breaks" entry for it, and no test of the packing itself.
- *What would resolve it.*
  - Extend only the leading jet axes: `pack_vectors` takes the number of jet axes (or the target jet sizes), say
    `m=0` by default, so that every existing caller is unchanged. Every other axis but the last must still agree
    across the modes, and a mismatch raises as today.
  - List the change in section 2.4.
  - Add to P11's tests: unequal jet sizes are packed with zeros; a `W` mismatch still raises; `m = 0` is
    unchanged.

### Minor

**2. O-44 (b) and O-46 interact, and neither point says so.** Suppose O-44 is answered (b), no default on
`LeastSquaresOracle`, and O-46 as proposed, one class with the field. Then every entries user must write
`LeastSquaresOracle('entries', order=())` for a kind that "has no jet shape" (entry 120). Name the interaction in
one line, so the maintainer answers the two together. For example: under (b), `order` could be required only for
`'probe'` and `'apply'`, through a sentinel default that `__post_init__` resolves.

**3. O-44 (c) should name its scope and the revision it implies.** "No default anywhere": does it cover the public
sampling functions and methods only, or also the engine, the dense functions (P4b, about forty order-0 calls in
tests and examples, which I counted in r1's grep) and the reference module? The plan is written for (a) and (b)
(11.2 says so). Under (c) it must say which slices change: P4b, P4c, P12 and P13, and section 9's "`T.probe(ww)` |
unchanged" row. Section 5.2 also lists P13 among the slices on O-44 only through its doctest; under (c) P13
changes far more.

**4. The account's wording.** "Memoization on the jet shape and shapes" (`ACCOUNT.md`, section 2) and "a new input
jet sizes recompiles" are leftovers of the vocabulary change. Cosmetic.

## Are the open points and names all and only the maintainer's?

- **O-36, O-40, O-41, O-42, O-43, O-44, O-45, O-46, O-47.** Each is a genuine choice of interface or contract.
  Each is answerable in a line, with the costs stated.
- **O-37.** It is partly decided already. Reach from `t3toolbox.fitting` follows from the answers of 20:15 ("the
  jet helpers are reachable from `t3toolbox.fitting`"). The open part is the top level. The text proposes both
  places, which is fine, but it could say that only the top level is his choice.
- **O-38.** It is now a confirmation of a loss, which is his to accept.
- **N31 and N34.** They are records of decided names, marked "Decided". The others are proposals: N32, N33, N35
  to N39, and N40 for the vocabulary.
- **Missing.** I found nothing else that is his.

## Not verified

- The recount of 16 of 20 and 20 of 20 was checked by reading `_pairwise_path` against the derived step strings
  for the `mu` and `nu` families, not by running the test.
- Nothing of the suite was run.
