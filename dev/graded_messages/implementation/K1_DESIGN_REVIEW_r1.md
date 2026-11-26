NOT YET

# K1 design review, round 1

Reviewer 3 (Claude), 2026-11-25, on `graded-messages` at `282b46a9`. I read `K1_DESIGN_NOTE.md` whole, `k1/prose_edits.md`
(§1 and §2.1 to §5 closely), `k1/kwonly_check.py`, the summary `k1/sites_before_k1.md`, and the code each claim names.
Before the note I added my own risks to `k1/REVIEWER_PRENOTES.md` ("Third reviewer", 3a to 3i). Methods traced while
orienting: `TuckerTensorTrain.probe` and `probe_corewise_transpose` → `jets.check_sample` → `probing.t3_probe` →
`sampling_variables` (the entry, `_check_stacks`, `_check_data_space`, the uniform packing and masking) → the data-space
steps (`sampling_steps.py:915-1099`) → `jets.data_metric`, `_check_pair`, `_omega_squared`; and, for K1, `jets.py`
`blocks_to_weight`, the derivative functions and `_mode_weights`; `backend/fitting.py` whole; `uniform_fitting.py`;
`ut3_operations.pack_vectors` / `_pack_data_space`; the `Oracle` protocol, `flat_draw`, `_require_draw`;
`make_newton_display`; `t3toolbox/fitting.py:75-110`, `:575-625`; the translation (`t3toolbox/optimizers.py:110-265`);
`tests/_fits.py`.

**Summary.** The design is right, and I found no blocking error in it. The mathematics holds under my own checks (§1).
The batch survives every attempt to break it on both environments (§2). The amended part order is sound: at no part
boundary does the tree run and give a wrong answer (§3). What keeps this at NOT YET is evidence, not design. Two of
the conversions K1 makes by hand cannot be caught by any number: a wrong `(kind, order)` moved into a batch in part 0,
and a block weight converted into a full weight in part 4. The review of those parts needs an `ast` audit that pairs
each old site with its new one (findings 1 and 2). The other findings are small.

Tools kept in `implementation/k1/reviewer_tools/`: `dense_checks.py` (my dense checks, 41 rows, all pass at
`282b46a9`), `batch_model.py` (a model of §4.6's batch, attacked; runs on both environments). Nothing is left
running.

---

## 1. The mathematics, checked independently (`reviewer_tools/dense_checks.py`)

Every reference value is an explicit `np.ndindex` loop that reads a size-1 axis at 0, with no broadcasting in the
oracle. Sizes are chosen to coincide (`d = n + 1 = |W| = |K| = |C| = N_i = 3`, and at two jet variables `F = (2, 3)`
equal to `(W, C)` in size), so that an axis placed on the wrong side cannot fail by shape. All 41 rows pass.

- **`M = gamma * b**2 * f**2` with left alignment**: probe, tuple and packed with `K`, against the loop to 1e-14. The
  packed result is the tuple's, bitwise. Apply `F+W+K+C`, entries `()`, and two jet variables all pass. The negative
  controls run and give a wrong number, as they should: numpy's right alignment for probe and for apply, a transposed
  `(2, 3)` table, and E3b (a `(n+1,)` per-slot table read per mode at `d = n + 1`).
- **Bitwise against today's metric** (`jets.data_metric` of `blocks_to_weight`), in every precision pairing
  (`r`, `b` ∈ {float64, float32}), with the cast of the absorbing rule applied before squaring:
  - a block weight alone, probe and apply;
  - a full weight alone (`W` of 1);
  - the part-2 bridge: apply's and entries' `blocks_to_weight` output read as a full weight.

  Both weights together differ by rounding only (5.7e-16). This confirms the note's check 5 and the "where the
  product is formed" rule of §3.
- **`k!`**, at `order = (3,)` and `(2, 1)`:
  - a block weight and a per-entry full weight chosen for derivatives give the same misfit as `(k! b, f)` on Taylor
    residuals (rel. 1e-12), and putting the `k!` in both weights does not;
  - today's `k!` on the placed weight equals the `k!` on the table, bitwise;
  - `prepare_derivative_data(weight=None)` on float32 data is the `k!` table in float32, bitwise (prenote 3i).
- **`<y, M y>` and the block sums with `y`'s `C` of 1** (`C = (3,)`, the full weight's `C` full):
  - per element of `C`, each element's own fit;
  - the note's recipe (broadcast to `z`'s shape, then product, then reduce) equals the loop;
  - without the broadcast the sums are `|C|` times too small;
  - today's `_y_block_metric_y` with `omega = b f` agrees to rounding.

## 2. The batch (`reviewer_tools/batch_model.py`, my own model of §4.6)

On Python 3.11 / jax 0.10.2 / numpy 2.4.6 and on the floor environment (Python 3.9.19, jax 0.4.30, numpy 1.22.4)
every check passed:

- **Old positional forms fail at once:** `(ww, y)` with a `TypeError` (`sample` missing); `(ww, y, w)`,
  `(ww, y, w, 0.5)` and the entries `(index, y, w)` with a `TypeError` on the kind; `(kind, order, ww, y, w)` with
  "takes from 4 to 5 positional".
- **The order:** a bool, a list or an entries order `(1,)` is refused; `(np.int64(2),)` is normalized to `(2,)` and
  gives the same treedef as `(2,)`.
- **`dataclasses.replace`** keeps the kind and the order, re-validates `kind='nope'`, and refuses `weight=` (an old
  field name) with a `TypeError`.
- **Copies and guards:** the frozen guard, `deepcopy`, `pickle` and `repr` work.
- **Placeholder leaves:** the unflatten with `object()` leaves passes, and so does `tree_map` to an `in_axes` spec.
- **Compilation:** a jitted function traces once for two batches of one kind and order (and a third with other
  values), and again for another order and another kind.
- **The entries block weight** may be a Python float under jit.

The design of §4.6 holds (prenote 3e: `init=False` sits on the class, the fields stay `init=True`, so `replace` passes
every field by keyword). Finding 6 is a wording point on the one old form whose message does not name the new one.

## 3. The amended parts: is any boundary silently wrong?

**No.** My reasoning, with the experiment behind each step:

- **The rule sets are nested.** The new rules for `y` and for a full weight are each a subset of the old rule "each
  axis 1 or `z`'s size". A legal full weight is applied elementwise, exactly as the old weight was. So between parts 2
  and 4 the engine either refuses a weight or reads it with the meaning it had. The same holds for `y`.
- **What the bridge `full_weight=batch.weight` meets:**
  - every `blocks_to_weight` output: probe's has an edge of 1 (refused); apply's and entries' are legal full weights
    with bitwise today's metric (`dense_checks` rows "bridge");
  - every hand-made full-form weight: the old rule is a superset, so it is legal or refused.
- **The uniform path in parts 2 and 3.** The old `pack_batch` still packs the weight by `_pack_data_space`:
  - an edge of 1 kept by it is refused by the engine (the edge must be `N`);
  - a mixed edge is broadcast to `N` over the real slots, which is the value it meant there, and the engine masks the
    padding.

  Neither is a misreading.
- **Between the parts no consumer reads the batch differently from the engine.** The oracle's old `_y_*` sums, the
  batch's old `data_metric` and `take` all read the full-form weight with the old meaning. So the display's numerator
  and denominator agree.
- **Part 4 to 5.** The old frontend re-exports (`data_metric`, `blocks_to_weight`, `prepare_derivative_data`) still
  compute their old meaning, and what they return is either refused by the new batch (probe, edge of 1) or legal with
  identical numbers (apply, entries).

The plan's worry ("an intermediate tree with the engine on two weights and the oracle on one would run and be wrong on
every weighted fit") is met by the rule itself, as the note says. Two smaller points on the parts are findings 4 and 5.

## 4. The pre-notes, point by point

Reviewer 2's (§1a to §1g):

| Pre-note | Status in the note |
|---|---|
| 1a left alignment; the coincidence tests | Handled: §3 check 2, §6.3 (sizes "distinct and all equal"), mutation "block row right-aligned" (§6.4) |
| 1a `K` in `Jp_norm_sq`, `_Jp.w_tilde` | Handled: §4.3 (`_with_K` for the full weight only), §6.3 part 2 ("the block weight with `K` ... element by element of `K`") |
| 1a `k!` with `m = 2`, aligned on the table's own `F` | Handled: §4.5 (trailing `F` of `(d,)+F` or `F`), §6.3 (`order = (2, 1)`) |
| 1a `prepare_*(None)` against the old function | Handled: check 6, battery row |
| 1b the four enforcing places agree; an edge-1 full weight refused | Handled: §4.2's table, §4.4, §4.7, §4.10; §6.2 turns the M-arc tests into refusals. See finding 5 for `pack_batch`'s message |
| 1b the modes' agreement names `np.broadcast_to` | Handled in the engine and `check_data`; not in `pack_batch` (finding 5) |
| 1b a full weight without `K` against a `K`-stacked `r` raises by rank | Handled by the rank rule (§4.4) |
| 1c the reference forms `b² f²` itself, no `jets` | Handled: §4.1, part 1 ("no import of `jets`") |
| 1c no battery of the engine against itself | Handled: the battery runs across trees, the reference is pinned by brute force in part 1 |
| 1d static aux, Python ints, placeholder leaves | Handled (§4.6, §6.3 part 0); confirmed by my experiment (§2) |
| 1d compile-once across the move | Handled (§6.3 part 0: once for one kind and order, again for another) |
| 1d positional calls fail on the kind, with a message naming the new form | Handled differently, rightly but for one form: `(ww, y)` fails on `sample` with Python's message (finding 6) |
| 1d the `ast` comparison of every rewritten call | **Missed** (finding 1); the weights' half too (finding 2) |
| 1d apply's block weight passed as a full weight is invisible | **Missed** (finding 2) |
| 1d a user batch without `take` gets a clear error | Handled (§4.11 `_require_draw`); see finding 7 for a method-shaped `n_measurements` |
| 1d the display's silent positional labels tested | Handled (D13, §6.3 part 0) |
| 1e `take` with a partly-1 `W` | Handled (§6.3 part 4: "a full weight with one `W` axis of 1") |
| 1e `<y, M y>` with `C` of 1 broadcast first | Handled (§3 check 4, §4.8, mutation in §6.4) |
| 1e `gamma` once in `f_resolution` | Handled: `_metric(..., None)` in the sums, the scale once in `_scale_tree` |
| 1f masking: `y` and full weight always, block never; finite padding until K5 | Handled (§4.2, §9) |
| 1f `pack_batch` with `pack_vectors`; modes share the pattern | Handled, with finding 5 |
| 1g removals and their callers; translation pins | Handled (§2.1's table, D8, D15; `TestTheTranslation` unchanged) |

Mine (3a to 3i):

| Pre-note | Status |
|---|---|
| 3a the precision rule over `r`, `b`, `f`; bitwise with a float64 `k!` table on float32 | Handled (§3 "Where the product is formed", §4.2 "The dtype"); confirmed in every precision pairing (§1) |
| 3b `_fits.py`'s stacked helpers must leave the block weight alone | Handled (part 4: "`shared_over` and `at_element` reshape `y` and the full weight only") |
| 3c `getattr` readers of the oracle | Handled: my scan of `getattr`/`hasattr` with `'kind'`, `'order'`, `'take'`, `'n_measurements'`, `'check_batch'`, `'weight'` in the library, tests, examples and pages finds exactly the five sites the note names. The `.kind`/`.order` reads off an oracle also match, but see finding 3 for the subclasses |
| 3d a drawn batch of another kind or order between K1 and K4 | Not stated (finding 7, for the orchestrator) |
| 3e `replace` through the hand-written `__init__`; unflatten by keyword | Handled (§4.6); confirmed (§2) |
| 3f the oracle's sums by selection while the engine multiplies | Handled and stated (§4.8 "Why selection here", §9) |
| 3g old weights with a structural axis of 1 have no single-weight image | Handled (§7.1: written out with `np.broadcast_to`, values identical) |
| 3h `gamma` twice | Handled (above) |
| 3i the `k!` default keeps its dtype | Handled (§4.5); confirmed bitwise (§1) |

## 5. Findings

### 1. Part 0's review cannot see a wrong `(kind, order)` moved into a batch (should fix)

- **Place:** §2.2 part 0 ("Every construction is rewritten"), §7.4 (the diff audit), §7.2 (`old` vs `part0`
  bitwise).
- **Evidence:**
  - Part 0 moves `kind` and `order` from about 89 `LeastSquaresOracle(` constructions into about 111
    `LeastSquaresBatch(` constructions, often on other lines or in helpers (`tests/_fits.py`).
  - A wrong `order` there is the library's own trap: input jets read at the wrong jet shape pass every check of
    shapes (`LeastSquaresOracle`'s docstring shows it, `backend/fitting.py:348-359`).
  - A test that builds both of its sides from one mis-converted batch (engine against reference, oracle against
    oracle with a held `r`, stacked against separate) then compares two wrong numbers and passes.
  - The battery covers its own cases only. §7.4's audit classifies changed lines ("a construction") but does not
    check the values moved.
- **Resolve:** in `k1/diff_audit.py`, for part 0, pair each removed `LeastSquaresOracle(kind, order)` with the batch
  constructions it served (same function, or through `Fit`/the helpers of `_fits.py`), and assert that the
  `kind`/`order` expressions moved are the same expressions. Then list every construction the pairing cannot resolve
  for the part's reviewer. The prenotes' 1d asked for this.

### 2. Part 4's review cannot see a block weight converted into a full weight (should fix)

- **Place:** §2.2 part 4, §7.1's mapping table, §7.4.
- **Evidence:**
  - For apply and entries, `blocks_to_weight(kind, order, b, y)` passed as `full_weight=` is legal and gives bitwise
    the numbers of `block_weight=b` (my rows "bridge", §1). So no assertion, no battery row and no tolerance would
    notice a test converted to the wrong one of the two.
  - The literal leftover form is caught by part 5's scan (the function is deleted), but a hand-written equivalent
    (`full_weight=b.reshape(F + (1,) * k)`, or `np.broadcast_to` of the table) is not.
  - The probe kind is safe: the edge of 1 is refused.
  - This is the orchestrator's own question for this round.
- **Resolve:** in the audit of parts 1 to 5, map each removed `blocks_to_weight(...)` call to a `block_weight=` of its
  third argument (an `ast` match), and each removed `weight=` to `block_weight=` or `full_weight=` with the reason. Any
  apply or entries site that became `full_weight=` is listed by name for the reviewer.

### 3. The test subclasses of `LeastSquaresOracle` are not in the site list (minor)

- **Place:** §5.1, §2.2 part 0, §6.1 part 0.
- **Evidence:** nine test-local subclasses pass `kind` and `order` to the base's constructor:
  - `HalfOracle` (`tests/test_dispatch.py:2178`);
  - `_NoBlockSums`, `_NoDataBlockSums` (`tests/backend/test_optimizer_display.py:333`, `:338`);
  - `_OffsetOracle`, `_NoResolutionOracle`, `_BareOracle`, `_NoKpOracle` (`tests/backend/test_optimizers.py:889`,
    `:1614`, `:1620`, `:2260`);
  - `_BlockSumApplyOracle`, `_NoKpOracle` (`tests/backend/test_curvature.py:1020`, `:1039`).

  About 15 constructions call them: `HalfOracle('apply', ())` 4 times, `_NoResolutionOracle('probe', ())`,
  `_NoKpOracle(problem.oracle.kind, problem.oracle.order)`, and `_OffsetOracle(fit.oracle.kind, fit.oracle.order,
  offset, key)`, whose own fields shift two places to the left once the base has none.

  `k1_sites.py` counts calls named `LeastSquaresOracle(` and reads of `.kind`/`.order` off a base named `*oracle`, so
  these constructions are not in §5.1's 71. They fail loudly in part 0's run, but "every construction is rewritten"
  is what part 0's review checks against.
- **Resolve:** add them to §5.1 and to the audit's classes. For `test_dispatch.py:2178-2193`, say in a line what the
  test still pins: both oracles are now fieldless, so it pins the type in `ValueHashedFields.__hash__`/`__eq__`,
  which is still its subject.

### 4. Parts 2 and 3: the expected red is larger than §5.4 says, and the summary overclaims (minor)

- **Place:** §0 item 1(b), §2.2 part 2 "Red at its end", §5.4.
- **Evidence:**
  - With no weight, `prepare_derivative_data` returns the `k!` table through `blocks_to_weight`
    (`jets.py:1174-1178`), which for probe is mode-leading with an edge of 1. So in parts 2 and 3 **every** probe
    derivative fit is refused, weighted or not: `tests/_fits.derivative_fit(..., weight=None)` and its callers, and
    every translated `probe_derivatives` call (the examples and pages run only at part 5).
  - §0 1(b) says the bridge makes "every weighted fit whose weight has a size-1 structural axis (all of
    `blocks_to_weight`'s output) fail loudly". Apply's and entries' `blocks_to_weight` outputs have no size-1
    structural axis and pass (§2.1's bullets say so correctly).
- **Resolve:** correct the parenthetical. Add "every probe derivative fit (the default `k!` weight has an edge of 1)"
  to the predicted red of parts 2 and 3, so that the part's reviewer can tell an expected red from an unexpected one.
  The list stays recorded by a run.

### 5. `pack_batch`: the modes' agreement is refused by `pack_vectors`' generic error (minor)

- **Place:** §4.10.
- **Evidence:** the plan's §2.2 has the modes' agreement "checked by the engine, the probe data checker, `pack_batch`
  and the frontend batch; the refusal names `np.broadcast_to`".
  - §4.10 refuses mode sizes that disagree and an `N` that is too small.
  - For the patterns of size-1 axes it relies on `pack_vectors`' own `ValueError` ("every vector must have the stack
    shape of the first", `ut3_operations.py:171-174`). That error is loud, but it neither names the object (`y`, the
    full weight) nor `np.broadcast_to`.
- **Resolve:** `pack_batch` checks the pattern itself before packing, with the same message as the engine. Or the note
  says why `pack_vectors`' message suffices.

### 6. `LeastSquaresBatch(ww, y)` fails with Python's "missing 'sample'" (minor)

- **Place:** §4.6, `prose_edits.md` §1.2 (its example shows exactly this call).
- **Evidence:** §2. The plan (§2.4) says the old form "fails at once, on its kind". With two positionals the kind is
  never checked: `__init__() missing 1 required positional argument: 'sample'`. The message says nothing of the new
  form, which the prenotes' 1d asked for. Three or more positionals do fail on the kind.
- **Resolve:** give `sample` a private sentinel default and check the kind first, with a message naming the new form
  (`LeastSquaresBatch(kind, order, sample, y, *, block_weight=..., full_weight=..., ...)`). Or accept Python's message
  and say so in the docstring's example.

### 7. The default draw and user draws (minor; one point for the orchestrator)

- **Place:** §4.11, `prose_edits.md` §1.8-§1.9, `backend/optimizers.py:1243`, `:1295`.
- **Evidence:**
  - (a) `_require_draw` accepts `hasattr(batch, 'n_measurements')`. A user batch written to the old protocol's shape,
    with an `n_measurements()` method, passes, and then fails inside `rng.choice` with an obscure `TypeError`.
  - (b) `flat_draw`'s new text still says a user's draw returns "a batch of the oracle". With the kind and the order
    in the batch, a user `draw=` that returns a batch of another kind or jet shape runs another objective and
    recompiles, silently. By the plan nothing compares them in the backend (§4.3, "the optimizers do not look"), and
    the frontend's comparison is K4's. So from K1 to K4 nothing compares them.
  - (c) The docstrings of `mc_sgd` (`:1243`, "which needs an oracle with `take`") and `adam` (`:1295`, "an oracle
    without `take` needs a `draw`") are not in §4.11's or §1.9's list.
- **Resolve:**
  - (a) Require an integer, e.g. `isinstance(batch.n_measurements, numbers.Integral)`, naming the property in the
    message.
  - (b) Write "a batch of the same kind and jet shape as `batch`" and say that the backend does not check.
  - (c) Add both lines to part 0's prose.
  - **For the orchestrator:** whether the frontend's `_frontend_draw`, which already wraps a user draw, compares
    `(kind, order)` from K1 (two tuples per draw), or the gap is stated in the note's §9 until K4.

### 8. The prose: what is written is true; three points (minor)

- **The split** (§10) is sound for a slice this size, and it keeps the maintainer's rule: contracts in the design
  phase; comments, shape comments and private docstrings in each part's build review, which reads them; nothing
  built before its prose is signed off. The state of the draft:
  - part 1's reference texts are not drafted;
  - `apply_data_metric`, `entries_data_metric`, the dual-jet pair and `prepare_*` are drafted as "today's text with
    the changes listed";
  - `probe_data_metric`'s packed example is a placeholder.

  Each must be in full before its part's build, as §10 says. I reviewed the texts that are written.
- **True and complete, with these fixes:**
  - §3 (the engine's `full_weight` entry) says "in ``y``'s form". The engine does not enforce that (D1 checks and
    packs `y` and the full weight each on its own), and a query with a full weight and no `y` (`Jp_norm_sq`) has no
    `y`. Write "a sequence of `d` arrays, or one packed array on the uniform layout".
  - §1.9: see finding 7(b).
  - `backend/optimizers.py:1851` and `:2098` still write the 2026.5.0 floor as `|omega y| |omega r|`. That is still
    readable as `M^(1/2)`. P19's, or one word ("the weights") now.
- **Checked as written, and right:**
  - the batch's class docstring (§1.2);
  - `check` and `take` (§1.4, §1.5);
  - the protocol (§1.8);
  - `probe_data_metric` (§2.2): every refusal in "What raises" is in §4.4's table, and E3b is shown refused;
  - `derivative_block_weight_to_taylor` (§2.4: "the weight itself is multiplied, not its square" is right);
  - the `jets` module docstring's two-weights paragraph (§2.1; its two-variable `k!` table `[[1, 1, 2], [1, 1, 2]]`
    is right);
  - `pack_batch` (§4).

### 9. A mutation that only an asymmetric block table catches (minor)

- **Place:** §6.4.
- **Evidence:** the dual jet is `flip(M r)`: the block weight must be applied before the turn, so that
  `dual[k] = b[order - k]**2 r[order - k]`; likewise the inverse divides after flipping back. A block weight applied
  after the flip is wrong only where the table is not symmetric along a jet axis. The `k!` table at `order = (1,)` is
  `[1, 1]` and an equalizing weight is often near-constant: both would hide it.
- **Resolve:** add "the block weight applied after the flip in `w_tilde_step` (and the converters)" to §6.4. Use block
  tables that are asymmetric on every jet axis in the part-2 and part-3 tests.

### 10. D4 changes the frontend's public surface (minor, for the orchestrator to log)

- **Place:** D4.
- **Evidence:** the plan renames the backend functions (§5). `t3toolbox.fitting`'s re-exports follow
  (`probe_data_metric`, `apply_data_metric`, `entries_data_metric`, `derivative_block_weight_to_taylor`,
  `prepare_probe_derivative_data`, `prepare_apply_derivative_data`), which is the plan's consequence and mirrors
  today's set exactly. I agree, but a public frontend name the plan does not list is `K_BRIEF.md` §3's case.
- **Resolve:** the orchestrator confirms and logs it. No change to the note.

## 6. The design against the plan, the decisions and the open points

**Against the plan.** I checked §4's every signature and field order against `PACKING_PLAN.md` §2.3, §2.4, §3.1, §4.3,
§4.4 and §5. The results:

- **They match:**
  - the batch's field order `kind, order, sample, y, block_weight, full_weight, minibatch_scale, drawn_indices`;
  - the three metrics' signatures;
  - the four converters and `derivative_block_weight_to_taylor`;
  - `prepare_*_derivative_data(order, y, block_weight=None, validation=None)`;
  - `pack_batch(batch, N=None)`;
  - the fieldless oracle;
  - the batch's `check`, `take`, `data_metric` and the property `n_measurements`;
  - the display reading the order off the batch and refusing a mismatched validation batch;
  - the translation's two K1 rows.
- **The structural rule** is stated the same way at the engine (§4.2's table), the batch's `check` (§4.7), the
  reference (§4.1, without `C` and `K`) and `check_data` (part 3). The metric is looser on the middle axes, by the
  plan's design (§2.3), and says so. `pack_batch` checks what it can see (the members' mode sizes against each other),
  and the engine checks the rest on the packed form. Each place has the information it needs for what the note says
  it enforces. On a uniform frame the engine sees `N`, not the real widths: that is G1's, the frontend's in K4, and
  `pack_batch`'s mode-size refusal covers the batch path in the backend.
- **Every engine call passes four positionals** (my `ast` count: 250 calls with exactly 4, none starred), so the new
  keyword after `y` moves nothing.

**The decisions.** I agree with D1 to D3 and D5 to D15. They are the implementer's to take: within the slice, or left
to K1's note by the plan (D2, D9). D4 is finding 10. D7, D8 and D15 change the plan's description of the parts. The
note puts them to the orchestrator as O4, which is right.

**O1, `flat_draw(batch, batch_size)`.**
- I recommend (a), as the note does. An unread argument is a trap. The oracle form exists only on this branch. The
  pin (`tests/backend/test_optimizers.py:1305`) is a test of the signature and moves with it.
- **What a user `draw=` receives:** `rng` only, and it returns a batch. So nothing in its signature has to agree with
  `flat_draw`'s.
- **What must agree is what it returns:** a batch of the full batch's kind and order (finding 7(b)), and under jit the
  same treedef: a drawn batch always has a scale and indices, so the full and the drawn batch compile separately, as
  today.

**O2, `check_probe_data` cannot tell `y` from a full weight.**
- **The problem is real, and it is wider than two objects.** The plan's list "y, a full weight, a residual" has three
  rules on the stack axes:
  - `y`: `W` exact, `C` 1 or full;
  - a full weight: `W` and `C` 1 or full;
  - a residual: exact.

  A residual passes `y`'s rule (exact is a special case), so two checkers per kind serve all three.
- **Why a public full-weight checker is needed at all**, when the block weight has none (plan §12.3): a full weight's
  edge widths are destroyed by packing exactly as `y`'s are (G1). A per-mode full weight with an edge of 1, packed by
  `pack_vectors`, puts its value on the edge's slot 0 and zeros elsewhere: a wrong weight, not a refusal. Only a check
  on the ragged form can catch it.
- **So I agree with (a):** `check_{probe,apply,entries}_full_weight`, eight checkers in all, decided in K2's note.
  (b) leaves a public checker looser than the engine. (c) is a switch.
- **One refinement for K2:** the full-weight checker should take `y`'s sample and `stack_shape`, as the data checker
  does, not `y` itself, so that it can be called without data.

**O3, no public full-weight checker between K1 and K3.** Accept. `batch.check` covers it, and `pack_batch` refuses the
edge mismatch. One sentence for `check_data`'s docstring (part 3): it checks `y` and refuses a full weight with a `W`
of 1, so it is not the full weight's checker.

**O4, the parts as amended.** Accept. Each amendment is checked in §3: no boundary runs and is wrong, the deletions
with their last callers keep `import t3toolbox` working (`t3toolbox/fitting.py:79-89` does re-export the four names),
and the translation in part 4 avoids a needless red between parts 4 and 5. Correct the predicted red (finding 4).

## 7. What would make this SIGN OFF

Findings 1 and 2: the part-0 and part-4 audits that pair old sites with new ones. The minor findings 3 to 10 can be
answered in the same revision, or deferred with a line each in the response.

## 8. What I ran

- `reviewer_tools/dense_checks.py` on the project environment, against the library at `282b46a9`: 41 rows, all pass.
- `reviewer_tools/batch_model.py` on the project environment and on the floor environment: every check passes on both.
- An `ast` count of `sampling_variables(` and `computed_names(` calls (code and doctest lines) in `t3toolbox/`,
  `tests/` and `examples/`: 250 calls, each with exactly 4 positionals.
- `grep` scans:
  - `getattr`/`hasattr` readers of `kind`, `order`, `take`, `n_measurements`, `check_batch` and `weight`;
  - `.kind`/`.order` reads off an oracle;
  - subclasses of `LeastSquaresOracle`;
  - the docstrings naming the oracle's `take`.
- No test of the suite was run. Nothing is left running. The repository is unchanged but for this file, the section
  "Third reviewer" of `k1/REVIEWER_PRENOTES.md`, and `k1/reviewer_tools/`.
