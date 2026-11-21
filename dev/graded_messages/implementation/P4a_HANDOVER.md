# P4a: `jet_contract` over one jet group `J` — handover

Slice P4a of `PART1_PLAN.md` (section 5.3; the form in section 2.4 as amended at the start of P4a; entries 116 and
120). Written by the implementer, 2026-11-19, against `graded-messages` at `96196427`. Nothing is committed. **Updated after review round 1** (`P4a_REVIEW_r1.md`, NOT YET on one should-fix test gap; the answers are in
`P4a_RESPONSE_r1.md`): two jit rows added to `test_dispatch.py`, `n_contractions` described as a regression pin, `_jet_roles`'
signature comments grouped, and the `J`-alone rule stated in `jet_contract`'s docstring. The
slice rebuilds what P1 built; nothing in the library calls `jet_contract` or `GroupLengths` (an `ast` search of
`t3toolbox/`, `docs/`, `examples/` finds no call), so only tests change beside `backend/contractions.py`.

## 1. What changed

**`t3toolbox/backend/contractions.py`.** `contract` and everything above line 433 are untouched; the module docstring's
paragraph on `jet_contract` and the imports (`math`, `numbers`) change; the `jet_contract` section is rewritten.

| Name | Kind | What |
| --- | --- | --- |
| `GroupLengths` | public | Fields `J, W, K, C` (was `T, W, K, C`). Every field a nonnegative Python `int` (no `bool`, no numpy integer), `J` any count (the bound `T <= 1` is gone) |
| `jet_contract(subscripts, *operands, lens, order, over_orders)` | public | The one behaviour of section 2.4. `lens`, `order`, `over_orders` keyword-only, none with a default |
| `_read_order(order, what)` | private | Reads a jet shape: a tuple only; entries Python ints or numpy integer scalars (`numbers.Integral`), never `bool`/`numpy.bool_`/float/negative; returned as Python ints. Called first, before any cache lookup. The one implementation that P4b's reader can reach (`jets` may import `contractions`, not the reverse) |
| `_JetString`, `_read_jet_string(subscripts)` | private, cached on the string | Validates the string from the string alone (the parser's rules; groups among `J W K C`; `J` leads its term; some operand has `J`; no input term of `J` alone) and records the terms, the output and the jet operands |
| `_JetRoles`, `_jet_roles(subscripts, jet_sizes, order)` | private, cached on `(string, jet sizes, order)` | The kind (`'ride'`, `'product'`, `'readout'`), the carrier, the pair (`carrier`, `turned`), the operands taken one multi-index at a time, the step string (`J` deleted from those), and the first full partner (`scannable`, `m = 1` only) |
| `_jet_group_lens`, `_multi_indices`, `_placed_in_box`, `_shifted_up` | private | Helpers: the `len_<G>` keywords; an operand's multi-indices clipped to the box; a part padded into the box at its offset (`xnp.pad`, jet axes only); a jet shifted up its jet axis with zeros of its dtype |
| `_jet_scan_body(step_subscripts, lens, carrier, partner)` | private, memoized | The scan body of `'scan'`, for a product and a readout alike. Keyed on every static it reads (the step string, the record, the carried and the scanned positions); closes over a string, a dict of ints and three ints |
| `_jet_scan(...)` | private | The scan: coefficient 0 by one `contract` call, then `xscan` over the rest (no scan when the partner has one slot) |

**What `jet_contract` does, in order.**
1. `lens` is a `GroupLengths` (`TypeError`); `over_orders` is `'scan'` or `'loop'` (`ValueError`), at every `m`.
2. `order` is read (`_read_order`); `len(order) == lens.J` (`ValueError`).
3. The string is read and validated from the string alone (`_read_jet_string`, cached).
4. The operand count, then every operand's rank against its term under `lens` (one message, every `m`).
5. `m = 0`: exactly one `contract` call on the string as written, with the length of every group it names
   (`len_J=0` included).
6. Every jet axis of every jet operand has at least one slot.
7. The roles, from the jet sizes clipped to the box (`_jet_roles`, cached); then:
   - **ride** (one jet operand, `J` in the output): the operand sliced to the box, one `contract`, the result padded
     to the box;
   - **product**: the carrier sliced to the box; for each combination of the looped partners' multi-indices (each
     clipped to the box) whose sum lies in the box, one `contract` of the carrier sliced to the box above the
     offset with the partners' coefficients, padded into the box at the offset and added. Under `'scan'` with
     `m = 1` the first full partner is scanned instead, the carrier starting `offset` slots up;
   - **readout**: a lone jet operand gives its corner slice, or zeros (`zeros_like` of a coefficient) where it has
     no corner slot. Otherwise, for each combination of the looped operands' multi-indices with the reduced corner
     `c >= 0`, the pair is sliced to the ranges that meet at `c` (`lo = max(0, c - size_b + 1)`,
     `hi = min(c, size_a - 1)` per axis; skipped where empty), the second flipped on every jet axis, and folded in
     one `contract`. Under `'scan'` with `m = 1` and a full operand beyond the pair, that operand is scanned (P1's
     scanned readout, the carried jet starting `n - c` slots up). If no combination reaches the corner, one
     `contract` on empty slices of the pair (an empty sum) with the looped coefficients zeroed returns zeros of the
     output's shape, dtype and backend.

The markers: `PART 2 (row 5.4, jets in several variables)` in the block comment and at the line where `scanned` is
chosen (with `m >= 2`, `scannable` is `None` and `'scan'` takes the loop); `PART 2 (row 5.5)` in the block comment
and at the product step, as before.

**Verified by script before it was asserted** (`scratchpad/implementer/check_math.py`, not in the tree): every
operand evaluated on a grid of roots of unity, a plain `contract` at each node, the coefficients of the full product
by an FFT, truncated or read at the corner. 14 strings (the library's families, a three-jet product, two-partner
products, a four-jet readout, a lone readout, scalar-core strings), `m = 1, 2, 3`, several orders each, jet sizes
the box's or random in `[1, n + 3]` per operand and axis, both forms: 896 cases, worst relative error 3.9e-15; at
`m >= 2` the two forms bit-equal; at `m = 0` one `contract` call. The jaxpr structure (one `scan` and one size at every
order under `'scan'` where a full partner is scanned; none and constant size elsewhere; none at `m = 2`, growing with
the box) was checked by a script on jax 0.10.2 before the dispatch test was written.

**`tests/_jet_contract_cases.py`** (rewritten). `STRINGS`: the 29 strings by family (xi 5, z 3, mu 4, nu 4, eta 4,
V 5, H 4). `LEAF` and `library_shapes(family, string, order)`: the jet sizes the library passes (a leaf of a line
with two slots per axis clipped to the box, a full tilde leaf, a line input riding to its own box, a seed full),
since a string no longer shows which operand is a leaf. `build(string, sizes, shape_of, rng)` takes jet sizes per
operand. `reference(string, arrays, order, shape_of)`: every stack group flattened to one axis, explicit loops over
the multi-indices of every jet operand (all slots, beyond the box too), one fixed einsum per combination,
accumulated at the sum where it lies in the box, or where it equals the corner; it imports neither `contract` nor
`jet_contract`. `n_contractions(string, sizes, order, form)`: a regression pin of the rule as built (not independent evidence; the
independent pin of the counts is P1's table in `TestStructure`).

**`tests/test_jet_contract.py`** (rebuilt), **`tests/test_contractions_interpreter.py`** (`TestJetContractDerivedStrings`
rewritten, its docstring paragraph), **`tests/test_contractions_sharding.py`** (the `jet_contract` test rewritten),
**`tests/test_dispatch.py`** (the `jet_contract` bucket rewritten).

## 2. The tests

`tests/test_jet_contract.py`: 47 test functions (46 pass, 1 is the expected failure of row 5.5), about 26 s. Numpy
only, except that `TestOrderInsideAStep` reads jax's contraction path.

| Class | Establishes |
| --- | --- |
| `TestTheStrings` (1) | 29 distinct strings, 5/3/4/4/4/5/4; groups `J W K C` only; 12 products, 9 readouts, 8 rides; `J` leads |
| `TestAgainstReference` (8) | Every string at `order = ()` over the whole stack matrix; at `(1,)` and `(3,)` with the library's shapes over the stack matrix; at `(1,)`, `(3,)`, `(1, 2)`, `(2, 0)` with random jet sizes per operand (1 to `n + 3`); at `(1, 2)` and `(2, 0)` with the library's shapes and three stack sets; `m = 3` on five strings, full and random; whitespace; beyond the 29 (a lone readout, three-jet products with two partners, readouts of four jets, with a leaf; float64 and float32, the dtype asserted); `numpy.polynomial.polymul` truncated, and its corner, at orders 0 to 9 with sizes below, at and above the box |
| `TestTheCases` (5) | NaN in every slot beyond the box: result NaN-free and bit-equal to the cut arrays (every string, library shapes, `(2,)` and `(1, 2)`); fewer slots equal the operand padded with zeros; jet axes of size 1, folded against a larger axis in readouts, against the reference and the padded operand; a box or a corner beyond every operand (exact zeros beyond reach; a corner nothing reaches gives zeros of the right shape, dtype and type, for a pair and for a lone operand); the pairing `corner(a * flip(omega^2 b)) = sum omega^2 a b` at `(3,)`, `(2, 3)`, `(1, 0)` |
| `TestTheTwoForms` (3) | `m = 1`: the 16 products and readouts with a full partner (mu, nu with a full leaf; eta; H with a full leaf), orders 1 to 16, float64 and float32: both forms against the reference and against each other. Elsewhere at order 3 (25 strings at their line or leaf shape): bit-equal. `m = 2`: bit-equal on every string and shape |
| `TestSensitivity` (1) | A 1e-6 perturbation of any slot of any operand, library shapes at `(3,)` and `(1, 2)`, both forms: caught where the slot can reach the result (derived from the definition); bit-unchanged where it cannot. 16 slots not read, as derived: the full jet's slots in the V strings with a line input that the input cannot complete to the corner |
| `TestStrictness` (4) | A stray or a missing axis on every operand at every position, at `()`, `(3,)`, `(1, 2)`: one message (4,000+ cases); a record wrong in `W`, `K`, `C`, or in `J` (ranks), or `J` against the jet shape's length; the jet shape (`3`, `[3]`, an array, a float, `True`, `numpy.bool_`, `None` raise `TypeError`; `-1`, a wrong length raise `ValueError`; `np.int64`, `np.int32`, `np.uint8` accepted, the same result, and the roles cache keyed on Python ints); a jet axis of size 0 |
| `TestStructure` (13) | `order = ()`: one `contract` call on the string as written, operands untouched, every group's length (`len_J=0`), bit-equal. `m = 1`, library shapes, orders 1, 2, 3, 6: the counts of today, one for a ride, two for a leaf of two slots, `n + 1` for two full jets and for H with a full leaf, one for V with an input (P1: two), never more than P1's. The rule as built (`n_contractions`, a regression pin) on random sizes at `(1,)`, `(3,)`, `(1, 2)`. Which partner `'scan'` scans (spied `_jet_scan`): a full one, one larger than the box, never a smaller one, never at `m = 2`, never under `'loop'`; two partners, one scanned per looped combination. No contraction names a pair of orders. float32 stays float32; numpy in numpy out; operands unchanged; no default for `over_orders`, `order`, `lens`, keyword-only; `lens` must be the record; 15 malformed strings raise at `J = 0, 1, 2` (no `J`, `J` not leading, `T`, `A`, `S`, `G`, `J` alone, the parser's); `'JWCa,JWCi,JWCb->JWCa'` is a product, against the reference; operand count |
| `TestGroupLengths` (6) | Fields `J W K C` in order; no defaults; `S=` and `T=` are `TypeError`; value equality and hash; frozen; `J` any count, the other validations; an equal record gets the same scan body, five static cells; the scan body's key holds the scanned position (a four-jet readout scanning the third or the fourth operand by their sizes, same step string, both call orders) and the carrier's |
| `TestOrderInsideAStep` (6) | Restated on the 20 product cases with the library's shapes at order 4 (8 mu and nu strings with a leaf of two slots and with a full leaf, 4 eta strings). jax's default `optimize` in `{'auto', 'optimal'}` (jax 0.10 and 0.4.30); on jax a product step pairs the coefficient with the core; readouts fold the pair first on numpy and jax (8 H cases); numpy today: 16 costly, the 4 nu strings with a leaf of two slots cheap; **the expected failure** kept |

**The recount of row 5.5** (`contractions._pairwise_path` on the step strings `_jet_roles` gives, numpy, library
shapes; a script in my scratchpad):
- **order 4: 16 of 20 costly** (every mu case, every eta case, the 4 nu strings with a full leaf, where the leaf
  ties with `nu` and, first in the string, carries); **cheap: the 4 nu strings with a leaf of two slots**
  (`'Caib,WCi,JWCb->JWCa'` and its `K` variants: the core and the coefficient are the first pair). P1's result.
- **order 1: 20 of 20 costly**: the leaf of two slots ties with `nu` and carries, so those 4 pair the carried jet
  with the core too.

The tests pin order 4; the class docstring states order 1.

`tests/test_contractions_interpreter.py`, `TestJetContractDerivedStrings`: three records (`J = 0, 1, 2`, at `()`,
`(2,)`, `(1, 2)`), the library's shapes, both forms: every string handed to `contract` matches the dialect, is the
literal with `J` deleted from zero or more input terms, comes with exactly the lengths of its groups (equal to the
record's), and gives the loop oracle's value; 24 distinct derived strings from the 29 (hand count in the test's
comment: one per mu and eta string, two per nu and H string, none from V). `HISTORICAL` unchanged (no library call
site yet; its format is P5's question).

`tests/test_contractions_sharding.py`: the 29 strings at `(2,)` with their library shapes (the 12 mu, nu, H strings in
both leaf shapes, so the scanned full partner is among them) and five strings at `(1, 2)` (products, a V and an H
readout), both forms, each `W` sub-axis sharded on 4 virtual devices: 196 compiles, zero all-gathers, about 32 s.

`tests/test_dispatch.py`: `test_jit_jet_contract` (a ride, a product with a leaf, a product of two full jets, a
readout with a leaf, a readout of three full jets; at `(3,)`, `()`, `(1, 2)`; both forms; float64 and float32; jit
and eager jax; each operand in turn a closed-over numpy array under jit; **after review round 1, two more rows at
one shape, `'loop'`, float32: a lone readout whose corner is unreachable and the pair fallback, each jitted whole and
with each operand closed over as numpy**), about 60 s. `test_jet_contract_scan_is_one_lax_scan`:
one `scan` and one size at `(2,)` and `(8,)` under `'scan'` for the full product and the readout of three, `'loop'`
growing; no scan and one size for a ride, a product with a leaf, a readout with a leaf and the V readout of two
full jets at orders 2 to 8; at `(1,)` the product with a leaf and the readout with a leaf scan once, `length=1`; at
`(1, 2)` against `(2, 3)` no scan under either form and a growing program; the eager body stable.

**Mutations** (a scratch copy of `t3toolbox/` and `tests/` under my scratchpad, each mutant of `contractions.py` run
with `python -B` and `PYTHONDONTWRITEBYTECODE=1` against `tests/test_jet_contract.py` and
`TestJetContractDerivedStrings`, plus the dispatch tests for the dispatch and memoization mutants; the file restored
after each; the copy deleted afterwards). **34 of 35 caught.** Caught:
- the carrier counted by unclipped slots; ties to the last; the pair as the first two in the string;
- the readout not flipped, scanned or not; a product part placed at offset 0; the part's pad widths swapped;
- the ride not sliced; the readout range starting at 0; the box check of a product off by one;
- the product carrier not pre-shifted, the readout's pre-shift off by one, the scan's shift 0;
- a scannable partner at any `m`; the scannable partner not clipped (caught only by `test_which_partner_scan_scans`);
- the zeros fallback missing; a lone readout without its range check; the shift's zeros in the default dtype;
- `order`: a list, a `bool`, a negative accepted, not normalized, its length unchecked;
- a string with no `J`, `J` not leading, `T` or `A` among the groups, `J` alone: accepted;
- the rank check and the size-0 check removed; `GroupLengths.J` bounded;
- dispatch read off the first operand (`test_dispatch.py`); the scan body not memoized (`test_dispatch.py` and
  `TestGroupLengths`); the scan body keyed without the scanned position (`test_the_scan_body_key_holds_...`);
- `m = 0` making two calls.

**Not caught, and equivalent:** the multi-indices of the looped operands not clipped to the box. The extra
combinations it iterates all lie beyond the box (product) or below a corner of zero (readout), and are skipped by
the check that follows before any slot is read, so the results, the calls and the reads are the same; only the
Python loop is longer. The clipping is kept for that.

## 3. The gate

The whole gate on the final tree, nothing else running:
`PATH=/home/nick/miniconda3/envs/t3toolbox/bin:$PATH dev/graded_messages/implementation/run_gate.sh --detach -n 3 -l <scratchpad>/implementer/gate_P4a`,
then `wait`. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1647 passed, 5 skipped, 1 xfailed, 1 warning, 68223 subtests passed   1338.8     10080       2230
    doctest-modules     PASS   310 passed                                                     61.4      1710      10294
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 3.0       281      11719
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               16.3       372      11634
    examples            PASS   14 examples, 0 failed                                           411       640      11384
    sphinx              PASS   0 warning/error line(s)                                      1055.0       636      11229

Against `BASELINE_laptop.md`:
- **No failure, new or old.** The suite has 15 more tests (1,632 to 1,647: `test_jet_contract.py` has 47 test
  functions where P1 had 32) and 1,980 more subtests (66,243 to 68,223).
- The same 5 skips (the slow tier), the same 1 warning (adam on a manifold), and the same 1 xfailed (the row-5.5
  marker, restated).
- The module doctests are the same 310: `GroupLengths` and `jet_contract` keep one doctest each.
- **Memory:** the suite peaked at 10.1 GB (baseline 9.8 GB), with 2.2 GB available at its lowest (baseline 2.5 GB).
  That is within the margin, but closer to it.

Before the gate, on the same code:
- the four slice files (the `jet_contract` tests of `test_dispatch.py`, about 59 s; the sharding test, 32 s;
  `test_jet_contract.py`, 26 s; `TestJetContractDerivedStrings`);
- the module doctests of `contractions.py` (3 passed).

**After review round 1** (test rows, one docstring sentence, one signature's comment layout; no full gate, as
ruled), single process: `tests/test_jet_contract.py` and `tests/test_dispatch.py -k jet_contract`: 48 passed, 1
xfailed, 13,199 subtests, 90 s; the module doctests of `contractions.py`: 3 passed. The reviewer's three host-numpy
mutants of the lone and fallback branches each fail the new rows.

Logs are under my scratchpad (`gate_P4a`; the sphinx build was deleted after reading). No process, lock or file of
the gate is left in the repository.

## 4. Departures from the plan; choices

**Departures.**
1. **One check the plan's list does not have: an input term of `J` alone raises** (from the string alone, so at
   every `m`). Taken one coefficient at a time such a term leaves `contract` an empty term, which its parser
   refuses with a message about `contract`, not about the call. No library string has one; a group of zero axes
   beside `J` (`'JW'` with `W = 0`) is the way to write a scalar polynomial, and the `polymul` test does so.
2. The sharding sweep covers all 29 strings at `m = 1` (and five at `m = 2`), a superset of "the product strings
   and one product at `m = 2`".

**Choices the plan left open, or the rulings of 2026-11-19 as carried out.**
1. A string with no `J` raises at every `m` (ruling 2); in the malformed-strings test at `J = 0, 1, 2`.
2. "Most slots" counted after clipping to the box (ruling 3); first in the string among equals. The pair of a
   readout is kept in string order: the first is the one carried under `'scan'`, the second the one flipped.
3. A product at `m = 1` with two or more partners under `'scan'`: the first full partner is scanned, the rest looped,
   the carrier starting as far up as their orders took (ruling 4). Tested against the reference (`test_beyond_the_29`)
   and structurally (`test_which_partner_scan_scans`).
4. An unreachable corner: zeros by one `contract` call, on empty slices of the pair (an empty sum) with the looped
   coefficients replaced by zeros, or on a zero coefficient for a lone operand (ruling 5). So the shape, dtype and
   backend are `contract`'s, and a NaN elsewhere cannot leak into a zero that the definition makes exact.
5. `order` (ruling 7): `_read_order`, a tuple only; `numbers.Integral` entries but not `bool` (`numpy.bool_` is not
   `Integral`), nonnegative; normalized to Python ints; read before any cache lookup. Errors: a wrong type is a
   `TypeError`, a negative entry or a wrong length a `ValueError`.
6. The scan body is memoized on `(step string, record, carried position, scanned position)`, every static it reads
   (ruling 6, item 8). The roles are cached on `(string, jet sizes, order)`, the string's reading on the string.
7. The output's padding uses `xnp.pad` on the jet axes; the start shift `concatenate` with `zeros_like` (the
   operand's dtype).
8. A scan of a partner with one slot (order `(0,)`) makes the one `contract` call and no `xscan`.
9. `test_jax_default_optimizer_is_one_read_here` accepts `('auto', 'optimal')` with a comment naming the jax of each
   (ruling 1). The behaviour tests beside it are unchanged in strictness.
10. Test helpers: the strings, the library's shapes and the reference live in `tests/_jet_contract_cases.py`;
    `SIZE` gains `c` and `j` for the strings beyond the 29. Tolerances 1e-12 (float64) and 1e-5 (float32), as P1.

## 5. Found outside the slice

1. **The floor's jax default.** jax 0.4.30 (the CI floor) has `optimize='optimal'` as `jnp.einsum`'s default, jax
   0.10.2 `'auto'` (read from the installed sources). The pin now accepts both. Whether the two jax behaviour tests
   of `TestOrderInsideAStep` hold on 0.4.30 is for CI to show.
2. `test_dispatch.py`'s `jet_contract` jit test takes about 53 s locally (60 combinations, each with one jit per
   operand position). The backlog item on trimming `test_dispatch` time may want it.
