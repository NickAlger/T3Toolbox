# P1: `jet_contract` and `GroupLengths` — handover

Slice P1 of `PART1_PLAN.md` (section 5.3; the form in section 2.4; decided rows 5.4 and 2.9, the answer to O-10).
Written by the implementer, 2026-11-17; **updated after the reviews of round 1 and round 2** (`P1_REVIEW_r1.md`,
`P1_REVIEW_r2.md`: SIGN OFF; the answers are in `P1_RESPONSE_r1.md` and `P1_RESPONSE_r2.md`). Nothing is committed. The slice adds code and changes nothing that existed: no
existing function, test or expectation was edited.

## 1. What changed

**`t3toolbox/backend/contractions.py`** (additions only; `contract` and everything under it untouched).

| Name | Kind | What |
| --- | --- | --- |
| `GroupLengths` | public, in `__all__` | `@dc.dataclass(frozen=True)`, fields `T, W, K, C` (Python integers, in that order, no defaults; no `S`: the maintainer's decision of 2026-11-17, 02:23, see choice 4). Value equality and hash from the dataclass. `__post_init__` raises `ValueError` unless every field is a nonnegative `int` (no `bool`, no numpy integer) and `T` is 0 or 1 |
| `jet_contract(subscripts, *operands, lens, over_orders)` | public, in `__all__` | `contract` with `T` and `A` multiplying as truncated polynomials. `lens` and `over_orders` are keyword-only and have no default |
| `_JetStringCase`, `_jet_string_case(subscripts)` | private | How the order groups of one string multiply, read off the string alone and cached on it: the kind (`'ride'`, `'product'`, `'top'`), the carrier, the partner, the pair, and the string `contract` is handed. (Round 1: renamed from `_JetPlan` / `_jet_plan`; "plan" is Part 2's word) |
| `_jet_group_lens(subscripts, lens)` | private | The `len_<G>` keywords for exactly the groups a string names (`A` gets `lens.T`) |
| `_jet_scan_body(subscripts, lens)` | private | The factory of the scan body of `'scan'`, `lru_cache`d on `(subscripts, lens)`; one body for a product and for a readout; it closes over a string, a dict of ints and three ints |
| `_jet_scan(...)` | private | The scan of `'scan'`: the partner's coefficient 0 by one `contract` call, then `xscan` over the others |
| `_shifted_one_slot`, `_order_group`, `_JET_GROUPS` | private | Helpers |

Also: the module docstring gains one paragraph, `__all__` two names, the imports `dataclasses` and `itertools`,
and a comment block above the new code states the three cases and the two forms.

**What `jet_contract` does, in order.**
1. `lens` must be a `GroupLengths` (`TypeError`); `over_orders` must be `'scan'` or `'loop'` (`ValueError`), at
   every order, also where it is not used.
2. The string is validated from the string alone (`_jet_string_case`, cached): the grouped-string parser's own
   rules; only the groups `T A W K C` (any other letter, `S` included, raises as an unknown group); no term with both `T` and `A`; the order group leads its term (inputs
   and output); a product with `T` in the output has exactly two operands with an order group; a product cannot
   have `A` in the output; a top coefficient needs an operand with `T`.
3. Every operand's rank is checked against its term and the record (`ValueError` naming the operand, its term and
   the record): the strictness of 5.4, with one message at every order.
4. Where there is an order axis (`lens.T == 1`), the order axes are checked, in every case, the ride included
   (round 1, finding 7): an `A` axis has 2 slots, every `T` axis the same `order+1 >= 2`. One message.
5. `lens.T == 0`, or the order axis rides: exactly one `contract` call on the string as written, with the length
   of every group it names.
6. Otherwise:
   - **product** (`T` in the output): the first full jet is the carrier and keeps `T`; the other operand is the
     partner. For an affine partner or `'loop'`: a Python loop over the partner's coefficients `r`, each
     contracted against the slice `carrier[:n-r]` and added into slots `r..` (zeros of the part's own dtype in
     front). For a full partner and `'scan'`: the partner's coefficient 0 against the whole carrier by one
     `contract` call, then `xscan` of `get_backend(True, use_jax)` over `partner[1:]`, with the carry
     `(sum, carrier shifted, every other operand)`; each step shifts the carried jet one slot and adds one
     contraction.
   - **top** (no order group in the output): two full jets (the pair) are contracted along the order axis in one
     `contract` call, one of them turned; a single full jet gives one slot; an affine operand is taken one
     coefficient at a time (two contractions, slices). **A full jet beyond the pair follows the form** (round 1,
     C2): under `'loop'` a Python loop over its coefficients with the pair sliced to `[:left+1]`; under `'scan'`
     a real scan over its coefficients with the SAME body as the product's: the first jet of the pair rides the
     carry and is shifted one slot up per step, the second is turned once, and each step contracts the two along
     the whole order axis with one coefficient of the scanned jet.

**The algebra of the scanned readout** (checked by a script against explicit loops at orders 1 to 16, largest
relative error 5.8e-16, before it was written; the two wrong variants, shifted down and not turned, were both
off by more than 1e-3). With pair `a`, `b` and scanned `c`:

    top = sum over i+j+k = order of <a[i], b[j], c[k]>
    step k:  carried[t] = a[t-k] (zero for t < k),  turned[t] = b[order-t]
             sum over t of <carried[t], turned[t], c[k]> = sum over i of <a[i], b[order-k-i], c[k]>

so the zeros the shift brings in at the bottom remove the terms with `t < k`, and what falls off the top are the
`a[i]` with `i > order-k`, which no `b` could complete. Every step has the same shapes.

No string or helper carries the layout; `use_jax` is `tree_contains_jax(operands)`; nothing is reshaped.

**`tests/_jet_contract_cases.py`** (new, test-only helper): the 41 strings by family, written for one site's arrays (the prototype's `S` deleted)
(`STRINGS`, `ALL_STRINGS`); the prototype's reference moved here and rewritten without the site axis (`make`,
`build`, `full`, `reference`, `relerr`, `spec_of`); and classifiers derived from the strings (`is_product`,
`is_full_product`, `follows_the_form`, `n_contractions`). It imports neither `contract` nor `jet_contract`.

**`tests/test_jet_contract.py`** (new). **`tests/test_contractions_interpreter.py`**,
**`tests/test_contractions_sharding.py`**, **`tests/test_dispatch.py`**: additions only (below).

## 2. The tests

`tests/test_jet_contract.py`: 32 test functions (31 pass, 1 is an expected failure, the marker of section 5),
about 14 s. Numpy only, except that `TestOrderInsideAStep` reads jax's contraction path where jax is installed.

| Class | Establishes |
| --- | --- |
| `TestTheStrings` (1) | The table is the plan's: 41 distinct strings, 5/3/8/8/4/5/8 by family, over the groups `T A W K C` and no others; 20 products, 12 of two full jets; 16 strings where a full jet is taken one coefficient at a time (the 12, and the 4 `H` readouts of three full jets) |
| `TestAgainstReference` (4) | Every string against the reference (no `contract`, no `jet_contract`): orders 0, 1, 3; `W` in `()`, `(3,)`, `(2,2)`; `K` in `()`, `(2,)`, `(2,2)` where the string has it; `C` in `()`, `(2,)`; both forms. Shape exact, value to 1e-12. Whitespace in the string. Beyond the 41: a single jet with no order group in the output; a readout with a pair, an affine operand and a further full jet, and one of four full jets (the carried jet then starts shifted by the orders already taken), against explicit loops, orders 1, 2, 4, both forms, in float64 and float32 with the dtype asserted (round 2, finding 10) |
| `TestTheTwoForms` (2) | `'scan'` against `'loop'` and each against the reference on the 16 strings that follow the form, orders 1 to 16, float64 (1e-12) and float32 (1e-5), dtype kept. On the other 25 the two forms are bit-equal |
| `TestSensitivity` (1) | A relative perturbation of 1e-6 of one order slot of one operand moves the result more than 1e4 times the tolerance of the reference test: every string, operand, slot and form. The 8 cases where a slot is not read (slots 0 and 1 of the full jet in the two `V` strings with an input) are derived from the definition and must leave the result bit-equal |
| `TestStrictness` (3) | An inserted axis at every position and a dropped axis at every position, on every operand of every string, orders 0 and 3, both forms: `ValueError` with `jet_contract`'s own message, the same at every order (3,000+ cases). A record wrong in one field raises. Order axes of a wrong length raise with one message in a product, a readout and a ride (8 cases x 2 forms) |
| `TestStructure` (10) | Exact. Order 0: one `contract` call, on the string as written, with the operands as given and the length of every group it names, bit-equal to that call. Above order 0 the number of calls. No call has two order axes on one array or a lowercase table over the orders. float32 in, float32 out. numpy in, `np.ndarray` out. Operands not changed. `over_orders` and `lens` have no default, are keyword-only, and wrong values raise at every order. Twelve malformed strings raise at `T=0` and `T=1`, among them three that name `S` (alone, after the order group, before it): refused as any unknown group is. Operand count |
| `TestOrderInsideAStep` (6) | Section 5. On jax (default optimizer recorded: `'auto'`) every product step with all slots pairs the coefficient with the core first, for the 20 product strings. A readout step with a third operand folds its two full jets first, on numpy and on jax (the 8 `H` strings). A plain passing test of what numpy does today, through the same helper as the marker (round 2, finding 11): one step with all slots per product string; the four `nu` strings with an affine leaf pair the core with the coefficient, the other 16 the carried jet with the core; the count 16 against 4 is pinned. **One expected failure**: on numpy no product step should pair the carried jet with the core first; 16 of 20 do |
| `TestGroupLengths` (5) | The four fields in order; no defaults; `S=` is a `TypeError`; value equality and hash; frozen; the validation; an equal record built anew gets the same scan body object (for a product and for a readout), a different one another, and the body's closure holds exactly five cells, each a `str`, an `int` or a `dict` of `str` to `int` |

`tests/test_contractions_interpreter.py`, class `TestJetContractDerivedStrings` (2 tests): by spying on
`contract`, over the 41 strings and every literal `jet_contract(...)` call site in the library (none yet), orders
0 and 2, both forms: every string handed to `contract` matches the dialect, is the literal with `T`/`A` deleted
from zero or more input terms (output unchanged), comes with the lengths of exactly the groups it names, each
equal to the record's, and gives the value of that file's definitional loop oracle; 26 distinct derived strings
from the 41. And every `jet_contract` call site in the library passes `lens=` and `over_orders=` (vacuous until
P5).

`tests/test_contractions_sharding.py`, `test_jet_contract_shards_free_over_every_W_sub_axis`: all 41 strings,
order 2, both forms (so the scanned readouts too), each `W` sub-axis sharded in turn on 4 virtual devices: zero
all-gathers (164 compiles, 19 s).

`tests/test_dispatch.py`, in `TestDispatch`:
- `test_jit_jet_contract`: five cases (a ride, an affine leaf, two full jets, a readout with an affine leaf, a
  readout of three full jets) x orders 3 and 0 x both forms x float64 and float32: `jax.jit` compiles and returns
  jax; eager jax in gives jax out; the dtype is kept with x64 on; the numbers are numpy's. **Then, for each
  operand position in turn, that operand a numpy array held outside the traced arguments, the others traced,
  under `jax.jit`** (round 1, finding 1): jax out, dtype kept, numpy's numbers; and the same eagerly.
- `test_jet_contract_scan_is_one_lax_scan`: for a product of two full jets and for a readout of three, the jaxpr
  of `'scan'` has exactly one `scan` and the same number of lines at orders 2 and 8; `'loop'` has none and grows.
  A ride, an affine leaf, a readout with an affine leaf and a readout of two full jets have no scan and one size
  at orders 2 and 8 under both forms. Eagerly, for both scanned cases, the first call traces the body and a second
  call (new arrays, an equal record built anew) makes one `contract` call only.

The doctests of `GroupLengths` and `jet_contract` (run, output pasted) are in the module-doctest step.

**Mutations tried by the implementer** (scratch edits, the file restored and diffed each time). Round 0: the scan
body factory not memoized; the turned slice not turned; the loop's part put in the low slots; zeros in the
default dtype; `K`'s length not passed; the rank check removed: all caught. Round 1: the reviewer's four
uncaught ones and seven new ones, in `P1_RESPONSE_r1.md`.

## 3. The gate

Run with `dev/graded_messages/implementation/run_gate.sh`, one step after another, nothing else running. The
baseline (`P0_BASELINE.md`) is all green, so any failure would be new.

**The tree after round 1 (the removal of the site group and the Part 2 markers): every step passes.** (Round 2
then changed `tests/test_jet_contract.py` only; its runs are at the end of this section.)
The whole gate, `run_gate.sh --detach -n 2 -l /tmp/gate_P1_r2`, then `wait`, copied from its summary. The six
slice files had the same checksums after the run as when it started.

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1442 passed, 5 skipped, 1 xfailed, 1 warning, 60982 subtests passed   1029.2      3639       4225
    doctest-modules     PASS   243 passed                                                     29.7      1459       6269
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 1.7       236       7422
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               13.5       454       7249
    examples            PASS   14 examples, 0 failed                                           224       361       7343
    sphinx              PASS   0 warning/error line(s)                                       202.6       690       6887

Against the baseline: 36 more tests (1,407 passed to 1,442 passed and 1 expected failure: 31 in
`test_jet_contract.py`, 2 in the interpreter file, 1 in the sharding file, 2 in `test_dispatch.py`) and 11,924
more subtests (49,058 to 60,982); the same 5 skips (the slow tier) and the same 1 intended warning; no failure,
no error. The module doctests have 2 more (241 to 243: `GroupLengths`, `jet_contract`). **`1 xfailed` is the
marker of section 5**, and the step is PASS.

Before it, the fast files of the slice on the same tree (`run_gate.sh -l /tmp/gate_P1_r2_quick quick
tests/test_jet_contract.py tests/test_contractions_interpreter.py tests/test_contractions_sharding.py
tests/test_import_graph.py tests/test_jax_absent.py`), the module doctests of `contractions.py` (3 passed) and
the two `jet_contract` tests of `test_dispatch.py` (2 passed, 164 subtests):

    step                result counts                                                      seconds   peak_MB  minavl_MB
    quick               PASS   57 passed, 1 xfailed, 15333 subtests passed                   112.9       814       6875

**After round 2** (two tests added or extended in `tests/test_jet_contract.py`, nothing else changed; no whole
gate, as instructed for a test-only change): `run_gate.sh -l /tmp/gate_P1_r3_quick quick
tests/test_jet_contract.py tests/test_contractions_interpreter.py tests/test_contractions_sharding.py`, and the
module doctests of `contractions.py` (3 passed):

    step                result counts                                                      seconds   peak_MB  minavl_MB
    quick               PASS   52 passed, 1 xfailed, 15357 subtests passed                   120.2       791       6920

So the suite now has 37 tests more than the baseline (1,443 passing and 1 expected failure), one more than the
whole gate above counted.

**Earlier runs, for the record.**
- Round 0: the whole gate passed except sphinx, which failed on one docstring of mine (an inline literal
  followed by a letter in `GroupLengths`); fixed by rewording the sentence, sphinx and the module doctests
  rerun and green (`/tmp/gate_P1`, `/tmp/gate_P1_b`, `/tmp/gate_P1_c`). The reviewer then ran the whole gate on
  that tree: all green (`P1_REVIEW_r1.md`).
- Round 1: two full runs were started and stopped by me before they finished (`run_gate.sh stop`), each because
  an instruction arrived that changed the tree (the hold on C1 with the Part 2 markers; then the removal of the
  site group). Neither had reported a failure. The run above is the third, on the final tree.

Logs: `/tmp/gate_P1_r2`, `/tmp/gate_P1_r2_quick`. No process, no lock and no file of the gate is left in the
repository (`docs/autoapi` is removed by the script).

## 4. Departures from the plan; choices the plan left open

**Departures: none in what is built.** Two things differ from the letter of the plan's test list and are
supersets: the sharding sweep covers all 41 strings, not only the product strings; the reference test includes
`K = ()` as well as non-empty `K`.

**Choices.**

1. **`GroupLengths` is a plain `@dataclass(frozen=True)`, as the plan's sketch writes it**, not a
   `ValueHashedFields` subclass. Its fields are integers, so the generated `__eq__`/`__hash__` are by value.
2. **`GroupLengths` validates its fields** (`ValueError`): nonnegative Python `int`s, `T` in `{0, 1}`.
3. **The order group must lead its term** (inputs and output), checked from the string at every order. The scan
   and the slices take the order axis at axis 0.
4. **There is no site group** (the maintainer, 2026-11-17, 02:23: "S should go"; it replaces Claude's C1 and
   the hold on it). `GroupLengths` has the four fields `T, W, K, C`; the groups of a `jet_contract` string are
   `T`, `A`, `W`, `K`, `C`; a string that names `S` raises the unknown-group error, whatever else it has;
   `test_the_site_group_is_one_more_group` is removed and three strings with `S` are among the malformed ones.
   A term is written for one site's arrays, which is what a step sees on both layouts; a comment in the code
   says that the uniform layout's site axis is not a group of these strings. This departs from the plan's
   sketch of `GroupLengths` (five fields) and from the wording of row 5.4, by the maintainer's decision; the
   plan and the row are Claude's to bring in line. The group `A` is untouched.
5. **Only the groups `T A W K C` are accepted** in a `jet_contract` string: any other group has no count in the
   record, so it could not be pinned. With `S` gone, nothing decided contradicts choice 3.
6. **The rank of every operand is checked in `jet_contract` itself**, first, with one message at every order.
7. **The order axes are checked wherever there is an order axis**, the ride included (round 1).
8. **Which jet carries.** In a product the FIRST operand with `T` keeps the order axis; in a readout the first
   two full jets are the pair and the first of them is the one shifted under `'scan'`. The choice is free: the
   value is the same either way, to rounding, and only the split of the work differs. It is the first so that it
   is one rule, read off the string. No test pins it (the values cannot see it); the code comment says so. A
   per-term choice belongs to 5.5 (Part 2).
9. **The scan's first step is made outside the scan**, by an ordinary `contract` call, which gives the carry its
   shape and dtype. One `lax.scan`, one program size at every order.
10. **Every operand but the partner rides the carry** of the scan, so the body closes over no array.
11. **A readout of three full jets follows the form** (decided by Claude in round 1, C2; the plan's sentence is
    corrected there). Beyond the 41: with more than three full jets, or an affine operand beside a third full
    jet, the first full jet beyond the pair is the one scanned and the others are taken by a Python loop over
    their coefficients, so only such strings (none in Part 1) still grow with the order under `'scan'`.
12. **The 41 strings and the reference live in a helper module**, `tests/_jet_contract_cases.py`.
13. **Test names and tolerances**: 1e-12 in float64 and 1e-5 in float32 (the prototypes' values).
14. **A list operand** is not converted: it raises `AttributeError: 'list' object has no attribute 'ndim'`,
    which is exactly what `contract` does with one (measured).

## 5. Carried to Part 2: the order of the contractions inside one step (row 5.5)

**Not fixed, by the maintainer's decision (2026-11-17, 01:51): deferred to Part 2 (row 5.5), and to be marked so
that it is not forgotten.** `contract` and `_pairwise_path` are untouched and no term is reordered.

**The finding** (Claude's; confirmed here by reading the paths). `jet_contract` leaves the order of the
contractions inside one step to `contract`. A product step contracts the carried jet with all its `order+1`
slots, the operands with no order group (a core) and one coefficient of the partner. The cheap order pairs the
coefficient with the core first (the expensive contraction, once), then meets every slot of the carried jet;
that is how the library writes it by hand today (`_eta_jets_order_step` in `backend/sampling_derivatives.py`).
- **jax.** `contract` runs one `jnp.einsum` with no `optimize` argument. In the installed jax (0.12.0.1) the
  default is `optimize='auto'`, and for the step of `'TWCa,Caib,TWCb->TWCi'` (the derived string
  `'TWCa,Caib,WCb->TWCi'`) `jnp.einsum_path` reports `[(1, 2), (0, 1)]`: the core with the coefficient first.
  For all 20 product strings the first pair is the coefficient with the core, on the steps where the carried jet
  has all its slots (the first step of `'loop'`, every step of `'scan'`). jax's path depends on the sizes: on the
  later steps of `'loop'`, where the carried slice is short, it takes another order, which is then as cheap.
- **numpy.** `contract` forces `_pairwise_path(step string)`: the pair of operands sharing the most letters.
  A product step is a three-way tie (two shared letters each), the tie goes to the first pair, so the path is
  `('einsum_path', (0, 1), (0, 1))` for every product string. For **16 of the 20** those first two operands are
  the carried jet and the core (all 8 `mu` strings, all 4 `eta` strings, the 4 `nu` strings of two full jets),
  and the expensive contraction is done on every slot at every step. For the 4 `nu` strings with an affine leaf
  the first two operands are the core and the coefficient, which is the cheap order, by the accident of the
  operand order.
- **The cost**, by the flop count of one step (Claude's example, the arithmetic repeated here): `W = 2000`,
  bonds 24, Tucker rank 12, order 4. Carried jet with the core first: `5*2000*24*12*24 + 5*2000*12*24 = 7.2e7`.
  Coefficient with the core first: `2000*24*12*24 + 5*2000*24*24 = 2.0e7`. The factor grows with the order.
  The numbers computed are the same either way. Nothing was timed.
- **The readout step** folds its two full jets first on both backends (the 8 `H` strings; numpy
  `(0, 2)` or `(0, 1)`, the pair; jax the same on the steps with all slots).

**Where the markers are** (search for `PART 2 (row 5.5)`):
1. `t3toolbox/backend/contractions.py`: one line in the block comment above the `jet_contract` code, and a
   comment at the product step in `jet_contract` saying what the order is on each backend and that the fix is
   Part 2's.
2. `tests/test_jet_contract.py`, class `TestOrderInsideAStep`:
   - `test_jax_default_optimizer_is_the_one_read_here`: the default is `'auto'` (so a change of jax's default
     is seen);
   - `test_on_jax_a_product_step_pairs_the_coefficient_with_the_core_first` (passes);
   - `test_on_numpy_a_readout_step_folds_its_two_full_jets_first` and the jax twin (pass);
   - `test_on_numpy_what_a_product_step_pairs_first_today` (passes; it pins 16 against 4 and fails at the
     same moment the marker below turns, so its counts are updated then);
   - `test_on_numpy_no_product_step_pairs_the_carried_jet_with_the_core_first`, decorated
     `@unittest.expectedFailure`: it asserts what should hold and fails today, so every run reports it as
     `1 xfailed`. Checked in a scratch process with `_pairwise_path` patched to the cheap order: the test then
     counts as an **unexpected success**, which pytest reports as a failure. Its docstring says what to do then
     (remove the decorator and the notes).
   The strings are the ones `jet_contract` derives (spied from its `contract` calls), the roles of the operands
   are read off the literal and the derived string, the numpy path is the `optimize=` handed to `np.einsum`, and
   the jax path is `jnp.einsum_path` with the default read from `jnp.einsum`'s signature.
3. This section.

**How it shows in the gate.** The suite's count line carries `1 xfailed` and the step is PASS (the gate script
copies pytest's summary line and takes PASS from pytest's exit code; an expected failure does not change the
exit code). The script was not changed. Its pytest flags are `-rfEs`, so the expected failure is counted in the
summary but not named in the log; `-rx` would name it.

## 6. Found outside the slice

1. **For P5: the `HISTORICAL` table of `tests/test_contractions_interpreter.py` cannot record 12 of the 41
   strings.** To be decided before P5 (the reviewer's point B). What the P5 implementer needs to know:
   - **What happens.** `scan_vocabulary()` collects every string literal in the library source that matches
     `^[A-Za-z]+(,[A-Za-z]+)*->[A-Za-z]*$` and has an uppercase letter, wherever it stands (an argument of
     `contract`, of `jet_contract`, or assigned to a name). So the first `jet_contract('TWCa,Caib,AWCi->TWCb',
     ...)` in the library puts that literal into the vocabulary, where it is treated as a plain `contract`
     string with `T` and `A` as two ordinary groups.
   - **The format.** `HISTORICAL` is a dict `subscripts -> str`. The value is `''` (the operand ranks determine
     every group) or ONE group letter (`contract` raises "do not determine" without `len_<that letter>` and
     succeeds with it alone). Three tests read it: `TestVocabularyCompleteness` (every scanned literal must be a
     key), `TestSupplementContract` (for a key with a letter: raises with no supplement, passes with that one;
     for `''`: passes with none), `TestVocabularyOracle` (the loop oracle with all lengths, and again with only
     the recorded one).
   - **What it cannot say.** As plain `contract` strings, 12 of the 41 are not determined by any single length.
     The smallest sufficient sets (each listed set is sufficient; found by trying every subset):

         TWKCa,Caib,AWCi->TWKCb    2: AK AT KT KW TW          Caib,AWCi,TWKCb->TWKCa    2: AK AT KT KW TW
         TWCa,KCaib,AWCi->TWKCb    2: AC AK AW CT CW KT KW TW  KCaib,AWCi,TWCb->TWKCa    2: AC AK AW CT CW KT KW TW
         TWCa,Caib,AWKCi->TWKCb    2: AK AT AW KT KW          Caib,AWKCi,TWCb->TWKCa    2: AK AT AW KT KW
         TWCi,AWo->Cio             2: AC AT CT CW TW          TWKCi,AWo->KCio           2: AT TW
         TWCa,AWCi,TWCb->Caib      2: AC AW CT CW TW          TWKCa,AWCi,TWCb->KCaib    2: AC AW CT CW TW
         TWCa,AWKCi,TWCb->KCaib    3: ACK ACT ACW AKW ATW CKT CKW KTW
         TWCa,AWCi,TWKCb->KCaib    2: AC AW CT CW TW

     Of the other 29: 15 need none, 7 need `C`, 5 need one of `C` or `K`, 2 need one of `A`, `T`, `W`. For these
     the present format works (one letter; where two letters each suffice, either can be recorded).
   - **What the format would need**, one of: (a) the value becomes a set of letters (a string of several
     letters), with `TestSupplementContract` and `TestVocabularyOracle` passing all of them and checking that
     no proper subset suffices; or (b) the literals that are first arguments of `jet_contract` are kept out of
     the `contract` vocabulary (an AST rule: skip a literal whose call is `jet_contract`), since they are never
     handed to `contract` without every length, and `TestJetContractDerivedStrings` already covers what
     `contract` receives from them, with all lengths, against the same loop oracle. Under (b) the strings
     `jet_contract` derives are not literals in the source, so nothing else needs an entry.
   - **Also to expect:** the sharding sweep `test_every_sub_axis_of_every_group_shards_free` runs over the same
     vocabulary with every length passed, so it works as it is, but compiles each new literal with `T` and `A`
     as ordinary two-axis groups: more compiles. Its `SINGLE` size table already has the letters `a i b o`.
2. **Strictness has one limit that the record cannot close**: where `W` and `K` stand together on every operand
   that has either (`'TWKCo,Cio->TWKCi'`), a record with one axis moved from `W` to `K` has the same ranks, is
   accepted, and gives the same contraction. A stray axis still raises. Pinned in
   `TestStrictness.test_a_wrong_record_raises`.
3. **A readout holds all order slots of its pair at once** (the reviewer's remark under A): the pair is
   contracted along the order axis in one `contract` call, under both forms, so that intermediate is not "one
   coefficient at a time". It is what the prototype and the plan describe; noted for Part 2's memory work.
4. **Documentation.** `docs/grouped_contractions.md`, `docs/api_reference.rst` and the changelog do not mention
   `jet_contract`; the plan puts documentation in P19, so nothing was written.
5. **`dev/graded_messages/PART1_PLAN.md` shows as modified in `git status`**: that is the coordinator's
   correction of the sentence about the readouts (section 2.4), not the implementer's.
6. **The session was interrupted once during round 0** (the server process stopped; the machine did not
   restart). Nothing was lost.
