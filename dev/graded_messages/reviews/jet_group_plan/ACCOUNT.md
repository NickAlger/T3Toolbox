# Account of the plan change: one jet group of any number of axes

Written by the writer of the plan phase, 2026-11-17, against `graded-messages` at `ae3811d` (the library and
`tests/` as P4 left them, `5991725`). It goes with the changed `PART1_PLAN.md`. Every statement about the code was
checked against the code at that commit; the counts were made there, by the searches named beside them.

## 1. The design in one page

**One concept, the jet order.** `order` is a tuple of maximum orders, one per jet variable: `order = (n_1, ...,
n_m)`. `order = ()` (`m = 0`) is the computation without jets, today's order 0. The *full jet shape* is
`F = (n_1 + 1, ..., n_m + 1)`. Every array that carries jets has exactly `m` leading jet axes; `m = len(order)`
decides how the leading axes of an array are read, never the type of the object (6.2). Whether a bare integer is
accepted as shorthand is open (O-36).

**A jet** is an element of the polynomial ring in `s_1 ... s_m`, cut off at `s_j^(n_j + 1)` (a box, entry 116).
Its array has the coefficient of `s_1^k_1 ... s_m^k_m` at the multi-index `(k_1, ..., k_m)` of its jet axes.
**An array's jet shape means: the coefficients beyond it are zero** (entry 116). So a jet axis of size 1 is a
variable the array does not depend on, and padding with zeros never changes a jet.

**The jet group `J`** (entry 120) is the one reserved letter of the jet axes in a `jet_contract` string: a
group of `m` axes, each operand with its own sizes along them. `GroupLengths(J, W, K, C)`; `J = m`, any
nonnegative integer.

**`jet_contract(subscripts, *operands, lens, order, over_orders)`.** One behaviour: the truncated product. Each
operand is a polynomial in `s` (an operand whose term has no `J` is a constant); the result is the product,
contracted as `contract` contracts it with `J` deleted, and then

- if the output term has `J`: its coefficients at every multi-index of the box `F(order)`, so the output's jet
  shape is exactly `F(order)`;
- if the output term has no `J`: its coefficient at the corner `order` itself (`s_1^n_1 ... s_m^n_m`). This is a
  block readout: the pairing of jets with a dual jet stored turned is the corner coefficient (row 2.4).

`order` is the output's, passed at every call, with no default (entry 116). At `m = 0` (`lens.J == 0`, `order ==
()`) the call is exactly one `contract` call. The caller is responsible for the truncation: `jet_contract` treats
every operand as an exact polynomial, and cannot tell a truncated full jet of two slots from an exact line. The
engine never asks for an output box larger than a truncated operand's (section 3 of the plan's table of
variables).

**The inputs are jets** (entries 117 to 119). `ww[i]` has the shape `J_in_i + W + (N_i,)`, with `m` jet axes of
sizes of its own; `pp` leaves every signature. The line `w + s p` is the input jet of two slots, built by a helper
(proposed name `affine_jet(ww, *directions)`); a curve has more slots; several directions make a jet in several
variables. `order` is always an argument and is never inferred from the input.

**Everything of the type of `z`** (the data, the residual, the weight's blocks, the block sums) has the full jet
shape `F` where the one order axis is today. The weight is `(d,) + F` (probe) or `F` (apply). The dual jet weighs
and then flips every jet axis. The derivative conversions divide (multiply) the slot `k` by `k_1! ... k_m!`.

**The entries kind has no jets** (entry 120). Its functions take an index and no `order` or `pp`; its oracle has
`order = ()` only; its data are `W`, its weight `()`. A helper (proposed `index_to_one_hot(index, shape)`) turns an
index into one-hot vectors, so that the released `entries_derivatives` is apply at `affine_jet(one_hot, pp)`.

**Each variable of the engine has a declared jet shape**: the full one `F` for every variable but four, and for
the input `w[i]` and the leaves `xi[i]`, `xi_dot[i]`, `xi_U_dot[i]` the input's own, clipped to `F` axis by axis.
The seed and every tilde variable are full (the seed is a full dual jet).

**Today's cost where the library is today** (entry 117: "today's cost in the cases the library uses"). With one
variable, a line input and a full order, `jet_contract` makes exactly the contractions it makes today: one for a
ride, two for a product with a two-slot partner, `order + 1` (scan or loop) for two full jets, and one fold of two
jets along the jet axes for a readout. Which operand is carried and which is taken one coefficient at a time is
read off the operands' static shapes, not off the string.

## 2. What changed in the plan, section by section

- **Head.** A paragraph on the change and one on the jet order (`order` a tuple, `F`, input jets, no `pp`, entries
  without jets), used everywhere below.
- **Section 1.** The scope says jets in any number of variables for probe and apply, `J`, input jets, the two
  helpers; what Part 1 owes for several variables (interface and correctness) against Part 2 (speed). Two seams
  added: jets in several variables (fast algorithms inside `jet_contract`), total-degree data. The seam of the form
  over the orders says both forms loop for `m >= 2` (O-38).
- **Section 2.1** (modules). `contractions` (one jet group, `GroupLengths(J, W, K, C)`); `jets` (the two helpers; the
  copies of the `pp` validators go); the dense functions' signatures.
- **Section 2.2** (data). New paragraph "The jet order". The sample: `ww` input jets at every order, `index` for
  entries, no `pp`, padding of input jets on the uniform layout. Data `F + W (+ (Ni,))`, entries `W`; weight
  `(d,) + F`, `F`, `()`. The batch's field comments; the oracle's `order` a tuple, entries `()` only.
- **Section 2.3** (engine). The signature (`sample`, `order` tuple, `weight`); step 1 (an entries call with an
  order raises; held shapes by declared jet shape); step 5 (every term one `jet_contract` with the output's order);
  step 6 (the turn flips every jet axis). New table: the declared jet shape of each variable.
- **Section 2.4** (backend surface). `GroupLengths` and `jet_contract` rewritten: the one behaviour, the easy cases,
  what is checked, why the output's order is a call argument and not a field of the record, the algorithm in Part 1
  and its cost, the strings (41 to 29). The two helpers. The functions of `jets` and `data_preparation` with the jet
  order, the flip and the product of factorials; the checkers of input jets; `rescale_direction` per variable
  (O-40). The dense functions with input jets; the reference module's node grid. The oracle's and `pack_batch`'s
  `order`. The table of changed functions (no `pp`; entries without `order`); the sketches; the ambient transposes
  (O-43); the table of removed names (kept 7 to 5, removed 129 to 131).
- **Section 2.5** (frontend). The methods' forms; the check through `check_sample`; the optimizer and local-model
  examples; `t3toolbox.fitting` reaches the two helpers.
- **Section 3.** An amendment note: N16's keywords are `order=()`, N14's fields `J, W, K, C`; the N16 row itself
  says `order=()` in place of `pp=None, order=0`. New names in 11.4.
- **Section 4.** 29 strings in `tests/_jet_contract_cases.py`; memoization on the jet order and shapes; `mu_JWCa`;
  the jet order a tuple below the entry points.
- **Section 5.1, 5.2.** P4a to P4d in the overview, with the open points each waits on; the additive rule
  mentions them.
- **Section 5.3.** P1 to P3 marked as built and rebuilt. New slices P4a, P4b, P4c, P4d. P5 (the jet matrix, the
  declared jet shapes test, the identities at `()` and in two variables), P6 (the bridge through `affine_jet`), P9
  (apply with jets, entries at `()` only, the migration of entry 120 bridged), P10 (stack matrix at three orders,
  one-hot dtype), P11 (packing of input jets; scan for `m = 1` only; one program size), P12 (the AD test at
  `(1, 1)`), P13 (the conversions through `affine_jet`; the doctests), P14 (oracle order; packing), P16 (the
  translation of integer orders, `(ww, pp)` and `'entries_derivatives'`; the record and the display, O-42), P17
  (examples use `affine_jet`).
- **Section 5.4.** A row for the copies of the `pp` validators (deleted in P4d).
- **Section 6.** Rows for `jet_contract` and the rebuilds; "The jet matrix"; the existing test files sorted with
  their P4a to P4d fate.
- **Section 7.** 7.1 the checkers; 7.2 the scan only at `m = 1`, the padding of input jets; 7.3 the one-hot dtype;
  7.6 jet axes sliced; 7.7 the jet order static, a new input jet shape recompiles.
- **Section 8.** `fitting_and_optimization.md`, `batching_and_stacking.md`, `grouped_contractions.md` (the table of
  letters with `J`; box against total degree); `deferred_and_rejected.md` (three records in place of the "future
  feature" entry); release notes (the upgrade of a jet call, the removal of entries derivatives).
- **Section 9.** The dry port's two replacements through `affine_jet` and `(J,)`; a table of what the consumer's
  calls become.
- **Section 10.** 131 removed names; a grep for a keyword `pp=`.
- **Section 11.** O-27's answer amended; 11.2 holds O-36 to O-43; 11.3's rows of the dense functions, the checkers,
  `pack_batch`; new 11.4, the proposed names N31 to N39.

## 3. Checks made for this account

- `/tmp/jg/proto.py` (a script of the writer's, not in the tree): for `m = 2`, order `(2, 3)`, the corner
  coefficient of `a` times the weighted `b` flipped on both jet axes equals `sum_k omega_k a_k b_k` (the box keeps
  row 2.4's pairing); the loop over the partner's multi-indices (carrier sliced to the box above `r`, added at
  offset `r`) equals a brute-force truncated product, for a carrier `(4, 2)`, a partner `(2, 3)` and a box `(3, 3)`
  (sizes above, at and below the box). Both print `True`.
- The 29 strings: P1's `STRINGS` with `T` and `A` replaced by `J` and repeats removed (`python3` on
  `tests/_jet_contract_cases.py`): `xi` 5, `z` 3, `mu` 4, `nu` 4, `eta` 4, `V` 5, `H` 4.
- The choice of carrier and pair by shapes reproduces P1's choice by letters on the library's strings: in a product
  P1 carried the first full jet; with a full jet and a two-slot leaf the full jet has more slots, and with two full
  jets the first wins the tie. In a readout P1 paired the first two full jets; the two operands with the most slots,
  first among ties, are the same two. The one difference: the `V` strings with an input (one full jet and a two-slot
  input), where P1 took the input one coefficient at a time (two calls) and the new rule folds the pair (one call).
  At order 1 a full jet and a leaf both have two slots; the carrier is then the first of them, and under `'scan'` a
  two-slot partner equal to the box is scanned (one step), where P1 looped; the number of contractions is the same.
- Counts by search at `ae3811d`: `jet_contract`/`GroupLengths` callers (section 6 below); P3's functions are called
  only from `jets`, `data_preparation`, `t3toolbox/fitting.py` and the tests; `sampling_definitions` has no caller
  in the library; the dense functions' calls with `pp` (P4b); T3Polynomial's calls (`t3polynomial/fitting.py`
  lines 225, 236, 384, 385; the scripts `d05`, `d06`, `o01`, `x02` call `probe_derivatives`; `g06`, `g06b`,
  `g06c`, `d07` call `least_squares_problem`); nothing in `examples/` or in T3Polynomial uses entries derivatives.

## 4. The table of strings

| Family | P1 | Now | The strings with `J` |
| --- | --- | --- | --- |
| `xi` | 5 | 5 | `JWo,Cio->JWCi`, `JWo,KCio->JWKCi` (an input); `JWCo,Cio->JWCi`, `JWKCo,Cio->JWKCi`, `JWCo,KCio->JWKCi` (a seed) |
| `z` | 3 | 3 | `JWCi,Cio->JWCo`, `JWKCi,Cio->JWKCo`, `JWCi,KCio->JWKCo` |
| `mu` | 8 | 4 | `JWCa,Caib,JWCi->JWCb`, `JWKCa,Caib,JWCi->JWKCb`, `JWCa,KCaib,JWCi->JWKCb`, `JWCa,Caib,JWKCi->JWKCb` |
| `nu` | 8 | 4 | `Caib,JWCi,JWCb->JWCa`, `Caib,JWCi,JWKCb->JWKCa`, `KCaib,JWCi,JWCb->JWKCa`, `Caib,JWKCi,JWCb->JWKCa` |
| `eta` | 4 | 4 | `JWCa,Caib,JWCb->JWCi`, `JWKCa,Caib,JWCb->JWKCi`, `JWCa,KCaib,JWCb->JWKCi`, `JWCa,Caib,JWKCb->JWKCi` |
| `V` | 5 | 5 | `JWCi,JWo->Cio`, `JWKCi,JWo->KCio` (an input); `JWCi,JWCo->Cio`, `JWKCi,JWCo->KCio`, `JWCi,JWKCo->KCio` (a seed) |
| `H` | 8 | 4 | `JWCa,JWCi,JWCb->Caib`, `JWKCa,JWCi,JWCb->KCaib`, `JWCa,JWKCi,JWCb->KCaib`, `JWCa,JWCi,JWKCb->KCaib` |
| **Total** | 41 | 29 | product strings (two jet operands, `J` out): 12, from 20 |

Of the 16 strings with `A`, 12 merged (`mu`, `nu`, `H`, four each) and 4 stayed (`xi` and `V`, two each), because
there the input carries no `C` and the seed does.

## 5. The choices, each marked

Marked **D** where it follows from a named decision, **W** where the writer chose and the critic should test it,
**O** where it is open with the maintainer (section 11.2 of the plan).

1. One jet group `J`, any number of axes; `GroupLengths(J, W, K, C)`; `J` any count. **D** (entries 116, 120).
2. Box truncation; shape means zeros beyond; output order explicit, no default. **D** (entry 116).
3. The output's jet order is an argument of each call, not a field of `GroupLengths`, because it differs between the
   calls of one computation (a leaf against a sweep variable). **W** (entry 116 decides "explicit"; where it lives is
   the brief's question 1).
4. The three behaviours (ride, product, top) are one: the truncated product on the box, or its corner when the output
   has no `J`. A readout is the corner coefficient. **W**, following entry 116 and row 2.4.
5. An operand without `J` in its term is a constant; any number of jet operands is allowed in a string. **W**.
6. The truncation is the caller's; `jet_contract` cannot check it; the engine keeps it through the declared jet
   shapes. **D** in substance (entry 116: "the output's shape must not exceed..."), **W** in who enforces it.
7. What `jet_contract` checks (section 2.4 of the plan): letters, `J` leading, the record, `len(order) == lens.J`,
   ranks, jet axes of size at least 1, `over_orders`. **D** for the rank rule (row 5.4), **W** for the rest.
8. Carrier, pair and partners chosen from the operands' static jet shapes (most slots, first among ties). **W**.
9. The general algorithm: a loop over the partners' multi-indices (product) and a fold of the pair along all jet
   axes plus a loop (readout); not "along each jet axis in turn". **W** (the brief's item 6 suggested the latter;
   section 2.4 of the plan gives the reason).
10. `'scan'` scans only a partner with as many slots as the box, and only at `m = 1`; `m >= 2` loops under both
    forms. **O-38**.
11. `order` a tuple of maximum orders everywhere, default `()`; the integer shorthand open. **D** for the tuple
    (entries 117, 120), **O-36** for the shorthand.
12. `jet_contract`'s argument is called `order` (maximum orders), not a jet shape in slots. **W**, a name proposal
    (N33).
13. Each variable's declared jet shape: `F`, except `w` (the input's) and the leaves (the input's clipped to `F`).
    **D** for "the input's or the full one" (entry 117), **W** for the clipping.
14. Modes may have different input jet shapes; on the uniform layout they are padded to one. **W**, following entry
    116 (zeros beyond change nothing) and entry 120's note on discrete and continuous modes.
15. The input's slots are Taylor coefficients. **O-39** (follows from rows 2.1 and 2.2, to confirm).
16. The dual jet flips every axis; the conversions use `k_1! ... k_m!`. **D** (entry 116).
17. `rescale_direction` with one length or one per variable. **O-40**.
18. `rescale_coordinates` scales every slot of every input jet. **W** (the map is linear in the input).
19. The helpers `affine_jet(ww, *directions)` and `index_to_one_hot(index, shape)`, in `backend/jets.py`.
    **D** that they exist (entries 118, 120); names and module **W** (N35 to N37); their frontend reach **O-37**;
    the one-hot dtype **O-41**.
20. Entries: no `order`, no `pp`; the oracle's `order` must be `()`; the engine raises on an entries call with an
    order; the derivative functions return entries data unchanged at `()`. **D** (entry 120) for the first three,
    **W** for the last two.
21. The two `pp` validators that P3 copied into `jets` are deleted there in P4d (131 removed names). **W**, following
    entry 118 (`pp` leaves every signature).
22. The weight of the entries kind has the shape `()`. **D** in substance (rows 2.5 with entry 120).
23. Total-degree data go into the box with weight zero. **D** (entry 116).
24. The reference module: a grid of roots of unity, `M_j = d (a_j - 1) + n_j + 1` nodes per variable. **W**.
25. The dense functions: one mode at a time, a truncated product by explicit loops, never `jet_contract`. **W**.
26. The test reference of `jet_contract`: explicit multi-index loops, and `numpy.polynomial` at `m = 1`. **W**
    (the brief's item 7).
27. The jet matrix of section 6. **W**.
28. The expected-failure marker of row 5.5 restated on the 12 product strings, recounted by the implementer. **D**
    (entry 111) for keeping it.
29. A second Part 2 marker, `PART 2 (row 5.4, jets in several variables)`. **W** (N38), tied to O-38.
30. The display for `m >= 2`. **O-42**.
31. The ambient transposes without jets. **O-43**.
32. Four slices, P4a to P4d, in that order (P4b before P4c because the reference module calls the dense functions;
    P4d last because `data_preparation`'s doctests call the dense functions). **W**.

## 6. What is rebuilt of P1 to P3, with counts

Counts by search at `ae3811d` (`grep`, and `ast` over the module for the public functions that take `order`, `pp`
or `sample`).

| Built in | What | Rebuilt in | Count |
| --- | --- | --- | --- |
| P1 | `GroupLengths`, `jet_contract` and their private helpers (`_JET_GROUPS`, `_JetStringCase`, `_order_group`, `_jet_string_case`, `_jet_group_lens`, `_jet_scan_body`, `_jet_scan`) in `backend/contractions.py` | P4a | 2 public names; the 429 lines P1 added |
| P1 | `tests/test_jet_contract.py` (32 tests), `tests/_jet_contract_cases.py` (41 strings) | P4a | 32 tests; 41 strings to 29 |
| P1 | rows in `tests/test_contractions_interpreter.py`, `test_contractions_sharding.py`, `test_dispatch.py` | P4a | 3, 3 and 5 calls of the two names |
| P2 | `dense_probe`, `dense_apply`, `dense_entries`, and `_dense_jet` | P4b | 3 public functions; their callers with `pp`: 4 in `data_preparation`'s doctests, 3 in `sampling_definitions`, 3 in `test_jax_absent.py`, 1 in `test_data_preparation.py`, 2 in `test_jets.py`, those of `test_sampling_definitions.py`, and their own doctests |
| P2 | `backend/sampling_definitions.py`: 37 public functions, all taking `order` and a sample | P4c | 37 |
| P2 | `tests/test_sampling_definitions.py` | P4b (dense), P4c | 47 of its 56 tests build a sample, pass `order` or name entries |
| P3 | `backend/jets.py`: 11 public functions taking `order`, `pp` or a sample | P4d | 9 changed, 2 deleted (the `pp` validators) |
| P3 | `backend/data_preparation.py`: `data_block_norms`, `rescale_direction`, `rescale_coordinates` | P4d | 3 of 7 public names; `equalizing_weight`, `t3_unscale_coordinates`, `ut3_unscale_coordinates`, `check_coordinate_scales` unchanged |
| P3 | `t3toolbox/fitting.py`: its own `dual_jet_to_jet` and `rescale_coordinates`; its re-exports | P4d | 2 functions; two names added to the re-exports |
| P3 | `tests/test_jets.py` (42 tests), `tests/test_data_preparation.py` (29) | P4d | 39 and 15 touch `order`, `pp` or entries |
| P1 to P3 | rows of `tests/test_dispatch.py` and `tests/test_jax_absent.py` | P4a to P4d | about 30 and 11 references to `jets` and `data_preparation` |
| P4 | the frame checkers | nothing | they have no jets |

## 7. Amendments needed in `TARGET_DECISIONS.md` and `SAMPLE_VARIABLES.md`

Not made by the writer (the brief: Claude makes them).

`TARGET_DECISIONS.md`:
- **2.1**: "jet edge variables" stands; where it says "order axis", read "jet axes".
- **2.3**: the conversions divide (multiply) slot `k` by `k_1! ... k_m!`; `prepare_derivative_data` returns
  `k_1! ... k_m!` with no weight. The entries kind has no derivative data.
- **2.4**: the turn is a flip of every jet axis; "top coefficient" is the corner coefficient `s_1^n_1 ... s_m^n_m`;
  the box is required (total degree rejected, entry 116).
- **2.5**: the weight is `(d,) + F` (probe), `F` (apply), `()` (entries); data by total degree in the box with weight
  zero.
- **2.6**: an input jet is general (any slots, entry 119); the line is the common case, built by `affine_jet`. The
  "two-term banded form" is no longer a structure: it is a partner of two slots.
- **2.7**: order 0 is `order = ()`, `m = 0`, no jet axis ever created.
- **2.8**: probe and apply each with a jet order; entries without (entry 120).
- **2.9**: `T` becomes `J`; "affine" becomes "a partner with fewer slots than the box"; the readouts compute the
  corner; the two forms differ only where a full partner is taken one coefficient at a time at `m = 1`; for
  `m >= 2` both loop in Part 1 (O-38).
- **5.4**: the groups `J`, `W`, `K`, `C`; `J` the one jet group of any number of axes, each operand with its own
  sizes; `A` removed (its definition of 02:31 superseded: an affine variable is a jet of two slots per axis); the
  record `GroupLengths(J, W, K, C)`; the output's jet order an argument of each call; 29 strings.
- **6.2**: `order` a tuple (the jet order); the sample is `ww` (input jets) or `index`; no `pp`.
- **6.6**: `probe(ww, order=())`, returning `F + W + C + (Ni,)`; the `pp` keyword of the decided form is removed
  (entry 118 reverses it).
- **N16 note** ("The names of Part 1, and the form of the merged sampling functions"): the keywords are `order=()`;
  the entries functions have no `order`; the sketch of `t3_entries(x, index, pp=None, order=0)` is superseded.
- **O-27 / O-28 answers**: the dense functions take input jets and `order`; `dense_entries(T, index)`.
- **"A future feature: jets in several variables"**: superseded by the build; its ledger entry becomes the three
  records of the plan's section 8.
- **"Carried to Part 2"**: item 1 restated on the 12 product strings (the count of the expensive order recounted in
  P4a); a new item for the scan over several jet axes (`PART 2 (row 5.4, jets in several variables)`), if O-38 is
  answered as proposed.

`SAMPLE_VARIABLES.md`:
- **Section 4**: "at a higher order it is the truncated product over the order axis" becomes "with jets it is the
  truncated product over the jet axes (`J`)".
- **Section 6**: "top coefficient" becomes "corner coefficient"; `z`, `r`, `Jp` of jet shape `F`.
- **Section 7**: the group table: `J` (the jet axes: `F` for a full jet, the input's own sizes for `w`, the input's
  clipped to `F` for `xi`, `xi_dot`, `xi_U_dot`) in place of `T` and `A`; the shape column `J + W + [K] + C +
  (size,)`; the input `w[i]` is `J_in_i + W + (N_i,)`; the seed `F + W + [K] + C + (N_i,)`; for the entries kind no
  jet axes.
- The status note's "orders 0 and 2" of the old check stays as a record of what that script covered.

## 8. What was found contradictory or hard

- **The integer shorthand.** The proposal of 14:00 (note "The public interface is generalized from the start",
  item 1) says "an integer stays as shorthand for one variable and `0` for none". That makes `order=0` mean `()`
  while `order=n` means `(n,)`, so `0` is the one integer that does not mean `(0,)`. Entries 118 to 120 did not
  confirm the shorthand. Put as O-36.
- **The ambient transposes.** Entry 117 asks for every sampling function to take a jet order; the ambient
  transposes were decided (18:35) with no `order`. Put as O-43.
- **The brief's item 6** suggests the product "applied along each jet axis in turn". For a contraction this needs
  an intermediate carrying the remaining jet axes of both operands; the plan takes the loop over the partner's
  multi-indices, which is simpler and no costlier.
- **"Today's cost"**: met in contraction counts. Two small differences from P1, both noted in section 3: the `V`
  strings with an input make one call instead of two; at order 1 a two-slot partner is scanned under `'scan'`.
- Nothing found impossible.
