# Review r1 of the jet-group plan change (a69e497)

**Verdict: NOT YET.**

Written by the critic, 2026-11-17, against `graded-messages` at `a69e497` (the plan as changed, `ACCOUNT.md`), the
plan before it at `9456502`, and the library and `tests/` as P4 left them. Scripts of my own are in `/tmp/critic/`
(not in the tree); each is named where its output is used.

The change is sound in its core. The mathematics holds (checked below by independent scripts): the truncated
product on a box, the corner readout, the pairing with a flipped dual jet in two and three variables, the
derivative conversions by `k_1! ... k_m!`, the generalized `rescale_direction`, and the reference module's node
grid, which is exact with the stated `M_j` and fails with one node fewer. The derived table of 29 strings is right
string by string. The contraction counts against P1 hold on all 41 strings at orders 0 to 5, with the one stated
difference (the `V` strings with an input: 1 call where P1 made 2). Entries 116 to 120 are carried out.

What keeps it at NOT YET: two consequences of "inputs are jets" that the plan decides silently, one of them a
silent wrong fit (findings 1 and 2), and a handful of plan statements that the code does not bear out or that would
make an implementer stop (findings 3 to 8).

---

## Findings

### Blocking

**1. Forgetting `order` gives a silently wrong result, and at order 1 a silently wrong fit; the plan decides the
default `order=()` on the oracle without saying so.**

- *Where.* Section 2.2 ("An oracle ... `order` (a tuple, `()` by default"), the `LeastSquaresOracle` sketch in
  2.4 (`order: typ.Tuple[int, ...] = ()`), the merged signatures (`order=()`), and the paragraph under
  `check_sample` ("an input of jet shape `(2,)` passed with `order = ()` and a `W` of one more axis is not
  distinguishable from a two-axis sample stack, and the docstring says so").
- *What is wrong.* Before this change a sample was `(ww, pp)` above order 0, so a forgotten `order` failed on the
  form of the sample. Now `affine_jet(ww, pp)` is an ordinary sequence of arrays `(2,) + W + (N_i,)`. Called with the
  default `order=()`, `x.probe(affine_jet(ww, pp))` reads the jet axis as one more sample axis and returns the plain
  probes at `w` and at `p` as two samples. Slot 0 is right, slot 1 is not a jet coefficient. My script
  (`/tmp/critic/trap.py`, a dense tensor, `d = 3`): same shape as the true order-1 jet, slot 0 equal, slot 1 not
  equal.
  For a fit it is worse. Take `LeastSquaresOracle('probe')` (default `order=()`), a batch
  `LeastSquaresBatch(affine_jet(ww, pp), y)` with `y` of order 1, so `(2,) + W + (N_i,)`, and no weight. Then
  `check_sample` reads `W' = (2,) + W`, `check_data` expects `W' + (N_i,)`, which is exactly the shape of `y`, and
  `check_weight` has nothing to check. So `check_batch` passes, and the fit solves another problem: the tensor is
  fitted to the jet's coefficients as if they were values at `w` and at `p`. With a line input, `check_data` catches
  it only at order 2 and above, because `n + 1 != 2` there. With a curve of `n + 1` slots it is silent at every order.
- *Why it is the maintainer's.* Entry 119 fixes that `order` is never inferred. Whether it has a default is
  another question, and nothing on record answers it for the oracle. N16's `order=0` default (now `order=()`) was
  decided when a forgotten order could not pass a jet sample silently. The plan presents the hazard as a docstring
  note under `check_sample`. It is not among the costs of O-36 and not in 11.2.
- *What would resolve it.* An open point in 11.2 with the options and their costs. For example: (a) keep `()` as
  the default everywhere and document the trap as a doctest gotcha on `probe`/`apply`, in the oracle's docstring and
  in `docs/fitting_and_optimization.md`; (b) give `LeastSquaresOracle` no default `order` (the oracle is built once
  per fit, so the cost is small), keeping the default on the sampling methods; (c) no default anywhere. The
  plan's tests should then pin whichever is chosen: a doctest of the wrong-versus-right call, and a test that the
  oracle's default (if any) is what the docs say.

### Should fix

**2. The uniform layout's jet padding is unmasked data. The plan decides this without listing its consequence:
garbage there changes the result, against the uniform equivalence contract and the tests P11 asks for.**

- *Where.* Section 2.2 ("the jet axes of the modes padded with zeros to their largest size, and the engine is
  strict about it"), 7.2 ("zeros beyond an array's shape change no jet, so the uniform layout needs no mask on the
  jet axes"), P11 ("garbage-padded inputs give the same real parts"), P14 (`pack_batch`).
- *What is wrong.* Today every padded slot of a uniform input is "garbage, don't care". The padding of `N` in a
  packed sample is cut off by the Tucker cores' masks (`docs/uniform_equivalence_contract.md`;
  `docs/contributor/testing_strategy.md`). The new padding of the jet axes is not covered by any mask. A nonzero
  there is a coefficient of the input polynomial, and it changes every jet. So:
  - P11's garbage test, read as written, would fail if it put garbage in the jet padding. The plan must either say
    the test excludes the jet padding, or mask it.
  - A backend user who packs a sample by hand (`ut3_probe` accepts a packed `ww` today and mirrors it,
    `backend/ut3_sampling.py`, lines 82 to 94) has to know the padding must be zeros. No checker says so.
  - "The engine is strict about it" does not say what is checked. I cannot tell whether it means a check of zeros
    (which is numerical and could not run under `jit`), a check of shapes, or nothing.
- *What would resolve it.* State the rule. For example: "jet padding must be zeros. It is data, not a don't-care.
  `pack_sample` writes zeros. P11's garbage test excludes the jet axes, and a separate test shows that a nonzero
  there changes the result." Or the alternative: a mask per mode and jet axis, applied once on entry, like the
  other masks. That is a new mask structure, so it should be listed as an open point if the writer prefers it.
  Then reword or delete "the engine is strict about it".

**3. P4b tells the reference module to call `affine_jet` (and `index_to_one_hot`), which its own independence test
forbids.**

- *Where.* P4b, "What breaks": "`backend/sampling_definitions.py`: three, in the definition of `z` (lines 572 to
  581). Until P4c they become the same call through `affine_jet`, and for entries at an order above 0,
  `dense_apply` at `affine_jet(index_to_one_hot(index, shape), pp)`". P4c then says "The independence test of the
  module (its forbidden names) is kept".
- *Evidence.* `tests/test_sampling_definitions.py`, `independence_violations` (lines 1064 to 1133). The reference
  may import only `probing`, `apply` and `entries`, under those names, and touch only their dense function. It may
  also import `*` from `common`. Any other import of the library is a violation (`import t3toolbox.backend.jets`,
  or `from ... import affine_jet`). `test_the_module_calls_nothing_of_the_library_but_the_dense_functions` would fail
  after P4b.
- *What would resolve it.* In P4b the reference stacks `(w, p)` into the two-slot input jet itself, in one line
  of `xnp`. For entries it already builds the unit vectors itself (`_setting`, line 233) and calls
  `apply.dense_apply` for every other output (`_dense_sample`, line 573). Say this, and say that the guard is
  unchanged. The same holds for P4c, if the reference reads `order` there: it needs its own private reading of
  `order`, not the one P4b puts in `jets`.

**4. P16's translation maps an old derivative kind at order 0 to `()`, which the old data do not fit.**

- *Where.* P16, item 4: "an integer `order=n` becomes `(n,)` (or `()` at 0), and an old sample `(ww, pp)` becomes
  `affine_jet(ww, pp)`".
- *Evidence.* The old derivative kinds carry an order axis of length 1 at order 0
  (`sampling_derivatives.t3_probe_derivatives(ww, pp, x, 0)` returns `(1, W, N_i)` per mode; my
  `/tmp/critic/old0.py`; and TARGET_DECISIONS 2.7, "Today's library ... carries an order axis of length 1"). So an
  old call `('probe_derivatives', (ww, pp), data, order=0)` has data of jet shape `(1,)` and a sample that
  `affine_jet` turns into jet shape `(2,)`. Under `()` neither fits. Under `(0,)` both do. Only the plain kinds
  (no `pp`) map to `()`.
- *What would resolve it.* "The old plain kinds give `()`. The old derivative kinds at order `n` give `(n,)`,
  `(0,)` included." The same distinction belongs in the costs of O-36. Under option (b) a user who upgrades
  `probe_derivatives(ww, pp, 0)` to `probe(affine_jet(ww, pp), order=0)` hits finding 1's misreading.

**5. The jet matrix has no input of unequal slots across the axes of one jet, and no modes of different jet
shapes, at `m = 2`.**

- *Where.* Section 6, "The jet matrix", and P5, P9, P11, P4b, P4c, which use it.
- *What is wrong.* At `m = 2` the matrix has only `affine_jet(ww, p1, p2)`, the same jet shape `(2, 2)` on every
  mode. The brief asks for "several variables at least at m = 2 with unequal orders and inputs of unequal slots".
  The orders `(1, 2)` are unequal. The inputs' slots are equal across modes and across axes. So these cases are
  never tested for the engine, the reference or the dense functions:
  - an input jet such as `(3, 2)` (a curve in one variable, a line in the other);
  - an input larger than the box on one axis and smaller on the other (the clipping of the leaves is per axis:
    `min(J_in_i, F)`);
  - entry 120's own case, a discrete mode beside continuous ones, at `m = 2` (one mode `(1, 1)`, others `(2, 2)`).
  Only `jet_contract` (P4a) tests random shapes at `m = 2`. The leaves' clipping and the uniform padding are
  per-axis rules, and only `m = 2` can show a per-axis bug.
- *What would resolve it.* Add to the matrix one case at `m = 2`, for example order `(1, 2)`, with modes of jet
  shapes `(3, 2)`, `(1, 1)` and `(2, 3)`. That covers a curve, a discrete mode and clipping on one axis only. Run it
  in P5, P9 and P11 (packed, so with per-axis padding), and in P4b and P4c.

**6. P4a's dispatch row drops the test that pins the scan's one program size, and the leaf's case changes at order
1.**

- *Where.* P4a, `tests/test_dispatch.py`: "jit `jet_contract` on a ride, a product with a leaf, ... `'scan'` at
  `m = 1` on jax compiles to one `lax.scan`".
- *Evidence.* P1's `test_jet_contract_scan_is_one_lax_scan` (`tests/test_dispatch.py`, lines 472 to 510) pins
  three things. First, the property of row 2.9: the program has the same number of lines at orders 2 and 8 under
  `'scan'` and grows under `'loop'`. Second, that a ride, an affine leaf and a readout with no third full jet have
  no scan. Third, that the eager scan body is one stable object. Section 6 says "Order 16 at `m = 1` where the scan
  is pinned (one program size)", but no slice row carries it. Under the new rule a two-slot partner at order `(1,)`
  equals the box and is scanned (the account says so), so the old assertion "an affine leaf has no scan" holds only
  from order 2 up.
- *What would resolve it.* P4a's row should keep all three assertions at `m = 1`. State that the product with a
  leaf has no scan at orders above 1 and one scan at order 1. Add that at `m = 2` both forms have no scan and the
  program grows with the box (O-38).

**7. `affine_jet` on a packed uniform array is not specified. Neither is what the uniform twins mirror once the
sample has jet axes.**

- *Where.* Section 2.4, `affine_jet`'s sketch ("`ww`: `len=d, elm_shape=W+(Ni,)`"); P11 and P12 (the twins check
  packing); 2.5 (`UniformTuckerTensorTrain.probe(ww, order=())`).
- *What is wrong.* The brief asks whether the helper is right for the packed probe array. Today `ut3_probe` takes
  a ragged `ww` or a packed `(d,) + W + (N,)` array, and mirrors it in its output (`backend/ut3_sampling.py`,
  lines 82 to 94). With input jets the packed form is `(d,) + J_in + W + (N,)` (P11). `affine_jet(packed_ww,
  packed_pp)`, read as written, iterates the first axis and returns a tuple. A user who packed gets a ragged sample
  back, and the twin then mirrors a ragged output. That may be the right rule: it is how `jets` treats a packed
  probe array under entry 115, mode by mode, giving a tuple. But the plan does not say it.
- *What would resolve it.* One sentence in 2.4: what `affine_jet` returns for a packed array (a tuple, by the rule
  of entry 115; or packed again). One in P12: the twins mirror a packed `(d,) + J_in + W + (N,)` sample. Add a test
  of both.

**8. Open points: two of the eight are in truth decided, and missing points and costs should be added.**

- **O-39 is decided** by rows 2.1 and 2.2: "Taylor coefficients replace derivatives everywhere" and "Taylor
  coefficients are the only form in the sampling, fitting and optimization layers". The input jet is part of the
  sampling layer. State it as following from 2.1 and 2.2, with the same note for a user whose curve is given by
  derivatives, and drop it from 11.2.
- **O-38 is largely decided** by entries 116 and 117: "Efficient algorithms for several variables belong to the
  performance phase", and "speed for several variables is Part 2". What is left for the maintainer is narrower,
  and the point should say so. Row 2.9's `'scan'` also had a non-speed property he asked for: "so that the compiled
  program does not grow with the order", for "other users ... who want higher order". At `m >= 2` that property is
  given up until Part 2. Put it as a confirmation of that loss, not as a fresh choice.
- **Missing points:** finding 1 (the default of `order` on the oracle and the methods) and finding 2 (the jet
  padding). Two smaller ones the account marks W and 11.2 does not list:
  - the entries oracle keeps an `order` field that must be `()`, where entry 120 says it "has no jet shape";
  - the dual-jet and derivative-data functions accept `kind='entries'` at `order=()` and return the data unchanged,
    where they could refuse it.
  Each needs one line for the maintainer, or a sentence on why it follows from a decision.
- **O-36, costs to add:**
  - under (b) the *type* of `order` decides the number of jet axes (`0` gives `m = 0`, `(0,)` gives `m = 1`), a
    weaker form of what 6.2 forbids;
  - under (b) the upgrade of an old derivative call at order 0 needs `(0,)`, not `0` (finding 4);
  - under (a) every caller's `J` becomes `(J,)` (the plan says this; keep it).

### Minor

**9. The account's statement about readouts is false at order 1.** `ACCOUNT.md` section 3 says: "In a readout P1
paired the first two full jets; the two operands with the most slots, first among ties, are the same two." For the
four `H` strings with a leaf (`JWCa,JWCi,JWCb->...`) at order `(1,)` all three operands have two slots. So the pair
is `(mu, xi)`, and `nu` is taken one coefficient at a time (scanned under `'scan'`). P1 paired `(mu, nu)` and looped
the leaf. The count is the same, 2. I checked with my own model of the new rule against P1's `n_contractions`, on
all 41 strings at orders 0 to 5 (`/tmp/critic/counts.py`): the only differences are the `V` strings with an input.
The same order-1 reversal happens in the products of the `nu` strings (`Caib,JWCi,JWCb->JWCa`): the leaf becomes
the carrier. That matters for the recount of the expected failure (row 5.5) in P4a. Say both in the account and in
P4a's xfail item.

**10. The table of declared jet shapes says the sums have none.** Section 2.3, the last row: "the blocks, the sums |
none". `r_block_sumsq` has the shape of the weight, `(d,) + F`, `F` or `()` (step 7 says so), and `y_block_sumsq`
likewise. Only `f`, `Jp_norm_sq` and the blocks have no jet axes.

**11. `jet_contract`'s semantics leave two edge cases unsaid.** One is a string with `J` in the output and no jet
operand. Under "one behaviour" it is the constant in slot 0 and zeros elsewhere; P1's code would hand `contract`
an output group no input has. The other is a readout with no jet operand at `lens.J >= 1`. Under "one behaviour"
its corner coefficient is zero unless `order` is all zeros, where P1's code returns the plain `contract`. Neither
occurs in the library. State a rule (raise, or the polynomial result) and test it. Also, "a full partner: one with
as many slots as the box" should say what happens to a partner with *more* slots than the box under `'scan'`. It is
sliced to the box first, by the rule that slots beyond the box are not read.

**12. The head paragraph contradicts N39.** Line 17: "The full jet shape is `F = ...`, written `J` in shape comments
where it is the shape of a variable's jet axes". Every signature sketch writes `F + W + ...`, and N39 proposes `F`
for the full shape and `J` for the axes in general. Make the head say what N39 says.

**13. 5.2 names the wrong slice for O-37.** "P4b waits on O-36, O-37 and O-41". P4b adds the helpers to the backend.
Their frontend reach, which O-37 is about, is P4d. Also, 5.2 does not say which slices wait on the proposed names:
P4a on N32, N33, N38, N39; P4b on N35 to N37.

**14. Leftovers of the entries kind with jets.** P16, tests added: "the stored record's shapes at `order = ()` and
`(2,)` for each kind". Entries has no `(2,)`. Write "for probe and apply; entries at `()`". P10's stack matrix
("`()`, `(2,)` and `(1, 1)`") and P11's uniform orders should also say "entries at `()` only", as section 6's
by-definition row does.

**15. Counts that are line counts.** P4a, "What breaks": "`tests/test_jet_contract.py` (53 calls of the two),
`tests/test_dispatch.py` (5), `tests/test_contractions_interpreter.py` (3), `tests/test_contractions_sharding.py`
(3), `tests/_jet_contract_cases.py` (1)". These are counts of lines that name `jet_contract(` or `GroupLengths(`,
docstrings included. An AST count of calls gives 55, 4, 2, 2 and 0 (`/tmp/critic/calls.py`). Write "lines that name
them", or give the call counts.

**16. The consumer's calls are incompletely listed.**
- `t3polynomial/fitting.py` also calls `X.probe_derivatives` at lines 597 and 598, in
  `fit_rank_continuation_adaptive`. The account lists 225, 236, 384 and 385 only.
- Section 9's table says scoring calls are "also in the scripts `d05`, `d06`, `o01`, `x02`". They are also in
  `x06`, `x07`, `s05`, `e01`, `d11` and `s00b`.
- `least_squares_problem` is also called in `d11` (frontend) and in `g01` (backend, with
  `bfit.probe_derivatives_kind`).
- `tfit.probe_derivatives_model` (a removed factory) is called in `x06` and `x07`, and has no row.
- The scripts are not part of the dry port, so this is a matter of the table's accuracy only.
- In the first row, `np.tile(weight, (d, 1)) if weight.ndim == 1 else weight` fails for `weight=None`.

**17. Stale references in the historical slice texts.** P1's test item says "every one of the 41 strings, as Part 1
writes them (section 2.4)". Section 2.4 now holds the 29. Point it at the 41 of P1's
`tests/_jet_contract_cases.py` as built, or mark the sentence historical. Section 6's doctest row lists "P2, P3, P5,
P19" and omits P4a to P4d, which rewrite doctests.

**18. P4b is "small" in 5.2.** It adds two public helpers and rebuilds three dense functions in `m` variables, with
doctests of a line, two directions and a curve. It also fixes the callers in `data_preparation`, the reference
module, and four test files. That is medium by the plan's own scale; P4c and P4d are both medium.

**19. The vocabulary differs from the record, and the amendments do not say so.** In entries 116 and 120 and the
note of 13:58, "jet shape" means the tuple of maximum orders ("what jet shape they want as output (maximum order of
the output polynomial in each direction)"; "one jet shape in place of the integer order"). The plan calls that
tuple the *jet order* and keeps *jet shape* for slot counts (`F`). The choice is good (N33), but `ACCOUNT.md`
section 7 should list the terminology among the amendments. Otherwise a reader of `TARGET_DECISIONS.md` meets
"jet shape" in two senses.

**20. Smaller amendments the account's section 7 misses.**
- The note "Groups in contraction strings" (TARGET_DECISIONS, lines 439 to 463: `A` kept, its definition, "`T` and
  `A` are not ordinary groups") should be marked superseded in place, not only through row 5.4.
- O-10's answer still says "`GroupLengths` keeps the five fields of 5.4".
- Row 3.4's "a small description (... the order)" holds for entries only as `()`.

---

## Points for the maintainer that the writer did not list

1. Whether `order` has a default on `LeastSquaresOracle`, and on the sampling methods, given that a forgotten
   `order` passes `check_batch` and fits the wrong problem at order 1 (finding 1).
2. The uniform jet padding: a zeros-contract with no mask, or a mask on the jet axes (finding 2).
3. Whether an entries oracle has an `order` field at all, or none ("has no jet shape", entry 120).
4. Whether the dual-jet and derivative-data functions accept `kind='entries'` (as a no-op at `()`) or refuse it.
5. Whether O-38 and O-39 need his answer at all (finding 8).

## What I checked, and how

- **The jet contraction** (`/tmp/critic/math1.py`). The plan's product algorithm, a loop over the partner's
  multi-indices with the carrier sliced to the box above `r` and added at offset `r`, against a brute-force
  truncated product: 200 random cases, `m = 1` to 3, operand sizes below, at and above the box, sizes of 1 included.
  All equal. The corner readout with per-axis slices and a flip equals the corner of the exact product, including
  the cases where the corner is out of reach (zero). The pairing `corner(a * flip(omega^2 b)) = sum omega^2 a b`
  holds at `m = 2` and `3`. The box ring's invariance `beta(ab, c) = beta(a, bc)` holds. The socle of the
  total-degree truncation at `m = 2`, `n = 2` has dimension 3 against the box's 1, which is why total degree has no
  corner functional.
- **The node grid** (`/tmp/critic/math2.py`). `d = 3` inputs of jet shape `(3, 2)` times a seed of order `(2, 1)`.
  `M = (9, 5)` from the plan's formula gives the box coefficients exactly. One node fewer on either axis does not.
  So the method is exact if and only if `M_j` exceeds the largest degree in `s_j` of any interpolated variable,
  which is `d (a_j - 1) + n_j` once the inputs are clipped to the box.
- **The derivative conversions and `rescale_direction`** (`/tmp/critic/math3.py`). Mixed partial derivatives of a
  product of input jets equal the Taylor coefficients times `k_1! ... k_m!`. Scaling slot `k` of the input jets
  (a curve in one variable, a line in the other) by `L_1^k_1 L_2^k_2` scales slot `k` of the output by the same
  factor.
- **The 29 strings** (`/tmp/critic/strings.py`), from P1's `STRINGS` by replacing `T` and `A` with `J` and removing
  repeats. They match the plan's table string by string: `xi` 5, `z` 3, `mu` 4, `nu` 4, `eta` 4, `V` 5, `H` 4. Of
  the 16 strings with `A`, 12 merge and 4 stay distinct. There are 12 product strings, down from 20.
- **Contraction counts** (`/tmp/critic/counts.py`). The plan's rules for carrier, pair and partners, against P1's
  `n_contractions`, on 41 strings at orders 0 to 5. The only differences are the `V` strings with an input (P1 2,
  new 1).
- **Code facts.** `jet_contract` and `GroupLengths` have no library caller. Every `dense_*` caller the plan lists in
  P4b is real (grep). `jets.py` has 11 public names and `data_preparation.py` 7. `sampling_definitions` has 37
  functions. The test files have 32, 56, 42 and 29 tests. P1 added 429 lines to `contractions.py`. The removed-name
  arithmetic is 30 + 5 + 131 = 166. The `pp` validators are called by the frontend from `sampling_derivatives`.
  The trap of finding 1 (`/tmp/critic/trap.py`). The order-0 shape of the old derivative kind (`/tmp/critic/old0.py`).

## What I could not verify

- The counts "47 of its 56 tests", "39 of 42" and "15 of 29" (which tests touch `order`, `pp` or entries), and
  "about thirty" and "eleven" references in `test_dispatch.py` and `test_jax_absent.py`. A grep of the module names
  gives 61 and 15 lines, which counts something else.
- Whether a nested scan over `m >= 2` jet axes (O-38's alternative) is feasible within the scan-body rules. Not
  needed for this review.
- On numpy, which of the 12 product strings pair the carried jet with the core first under the new carrier rule
  (P4a's recount). The carrier changes at order 1 in the `nu` strings (finding 9).
- T3Polynomial's weight shapes, read from the plan, not from its code; and its runtime behaviour (not run, as
  instructed).
- Nothing was run of the test suite or the gate.
