SIGN OFF

# K1 design review, round 2

Reviewer 3 (Claude), 2026-11-25. I read the revised note (revision 2), `K1_DESIGN_RESPONSE_r1.md`, `k1/prose_edits.md`
(revision 2: §1, §6 and §7 closely), and the orchestrator's rulings 22 to 29. I checked each round-1 finding against
the note, not against the response. Then I walked the two audits of §7.4 by hand over real test sites, and placed the
new guards of part 0 in the code. I ran two `ast` scans of `tests/` (scratch, below). I ran no tests, and nothing is
left running.

**Verdict.** Every round-1 finding is resolved in the note. The guards of part 0 sit where they should and are
number-neutral. The audits catch what they are for, with limits that the part reviews need to know (§2). The new
findings are minor (§4): the audits' blind spots, a gap in part 1's scope, and the wording of two transitional texts.
I sign off on the design. The prose of parts 2 to 5 gets its own round before each part, as ruled.

## 1. The round-1 findings, in the revised note

| # | Where it is resolved | Status |
|---|---|---|
| 1 | §7.4 (b); §2.2 part 0's review line | Resolved, with the limits of §2 below |
| 2 | §7.4 (c); `k1/full_weight_reasons.tsv`; §2.2 part 4's review line | Resolved. The reason file for every new `full_weight=` is the robust part (§2) |
| 3 | §5.1 (the nine classes, the field shift of `_OffsetOracle`, what `test_dispatch.py:2178` still pins); §2.2 part 0's count | Resolved |
| 4 | §0 1(b), §2.2 parts 2 and 3, §5.4 | Resolved; one addition in finding 13 |
| 5 | §4.10 (`pack_batch` checks the pattern itself, with the engine's message, before packing); a test and a mutation | Resolved |
| 6 | §4.6, and the class docstring's example (ruled: no sentinel) | Resolved as ruled; finding 15 is on the example's pasted line |
| 7 | §4.11 (`_require_draw` refuses a non-integer or callable `n_measurements`, with its own message); `flat_draw`'s text; `mc_sgd`'s and `adam`'s docstrings (`prose_edits.md` §1.9); the guards (ruling) | Resolved; finding 14 on one sentence |
| 8 | `prose_edits.md` §3 (the engine's `full_weight` entry states the forms the engine takes); §6 (part 1 in full); §7 (the `ω` lines, `check_data`'s O3 sentence) | Resolved |
| 9 | §6.3 (block tables drawn at random, asymmetric on every jet axis, at `order = (2,)` and `(2, 1)`); §6.4 (the flip mutations, parts 2 and 3) | Resolved |
| 10 | Ruled and logged | Closed |

## 2. The two audits, walked over real sites

### (b) The kind and the order (part 0)

**What I measured.** An `ast` scan of `tests/` (function scopes; doctests and pages not included) counted:

- 81 scopes that construct an oracle (the base or one of the nine subclasses) or a batch;
- 0 scopes with two or more distinct oracle pairs beside a batch;
- 12 scopes that construct batches with no oracle in scope (helpers and helper methods);
- 27 scopes that construct an oracle with no batch in scope (the batch comes from a helper, a `setUp`, or a `Fit`).

**The sites, walked by hand:**

- **An oracle in the test and the batch in a helper method.** `tests/test_least_squares_oracle.py:204`:
  `oracle = LeastSquaresOracle(kind, order)` and `b = self.batch(kind, order)`, where `Fixture.batch(self, kind, order)`
  (`:125`) builds `LeastSquaresBatch(s, y, ..., scale)`.
  - After part 0 the helper's construction is `LeastSquaresBatch(kind, order, ...)`, whose pair is the helper's own
    parameters.
  - Rule (b) finds no oracle in the helper's reach and lists the helper as unresolved, with its call chain. The
    removed oracle at `:204` is not itself listed: the reviewer reaches it only through that call chain. There are 27
    such oracle-side scopes, mostly in `test_least_squares_oracle.py` (9) and `backend/test_optimizer_display.py` (11).
- **An oracle at the call site, the batch from a module-level helper.** `tests/backend/test_optimizer_display.py:396-399`:
  `batch = _batch(kind, (), 40, ...)`, `val_batch = _batch(kind, (), 15, ...)`, then
  `run_display(LeastSquaresOracle(kind, ()), batch, val_batch)`. The same as the previous case: `_batch` is listed, and
  the `(kind, ())` at the two calls is compared with the removed oracle's by hand.
- **A batch rebuilt in the test beside its oracle.** `:428`:
  `LeastSquaresBatch(batch.sample, ..., omega)` and `LeastSquaresOracle(kind, order)` in one scope. Paired
  mechanically: the new pair must be `(kind, order)`.
- **A subclass.** `tests/backend/test_optimizers.py:909`, `_OffsetOracle(fit.oracle.kind, fit.oracle.order, offset, key)`.
  - The rule binds the first two arguments as the base's pair and sets `offset`, `key` aside.
  - Part 0 makes it `_OffsetOracle(offset, key)`. The scope has no batch, so there is nothing to pair and nothing to
    go wrong: the batch comes from `fit.batch`, whose `(kind, order)` were set where the `Fit` was built.
  - The audit should still list "subclass constructions whose arguments shifted" with old and new, so that a misplaced
    `offset` is read. `_OffsetOracle(offset, key)` against `_OffsetOracle(key, offset)` is invisible to every number
    except the test's own.
- **A doctest with two pairs.** `LeastSquaresOracle`'s docstring holds `oracle = ('probe', (1,))` and
  `plain = ('probe', ())`, with three batches between them. "Equals an old pair of the scope" would accept a batch
  converted to the other pair. Here the doctest's printed `f_mixed > 1.0` catches it, because a mis-converted batch
  fits exactly.

**What it does with a wrong conversion:**

- a pair that differs from every old pair in reach: refused (exit non-zero, listed);
- a pair equal to the wrong one of two old pairs in one scope: passes;
- a pair built in a helper: never compared mechanically.

**What it cannot see, for part 0's review to read by hand:**

- the helpers' call sites (about 12 helpers, with the 27 oracle-side scopes behind them);
- any scope with two distinct old pairs (none in `tests/`; the library's doctests, which their outputs cover);
- the subclasses' own arguments after the shift.

Finding 11 makes most of the hand reading mechanical.

### (c) The weights (parts 1 to 5)

**The sites, walked by hand:**

- **A table built inline, stored, and consumed through a lambda.** `tests/test_dispatch.py:1022`:
  `weight = jets.blocks_to_weight('apply', (2,), 0.5 + rng.random(3), y)`, consumed by
  `fn = lambda ...: sv.sampling_variables(..., weight=weight, ...)`.
  - The old keyword `weight=` inside the lambda pairs with the new keyword on the same callee, and is classified.
  - The removed `blocks_to_weight` asks for a `block_weight=` whose value is the old third argument,
    `0.5 + rng.random(3)`. The new code will bind that to a name first, which is new-side indirection: rule (c)
    resolves one assignment on the old side only. So the site is listed, not paired.
  - Safe either way, since a mis-conversion to `full_weight=` needs a reason line.
- **The shared harness.** `tests/_fits.py:85`:
  `omega = None if weight is None else jets.blocks_to_weight(kind, order, np.asarray(weight), y)`. The assignment is a
  conditional expression, so it is not "the one assignment". Listed, read by hand (one site).
- **A full-form weight passed through a helper parameter.** `tests/test_sampling_variables.py:81`, `:92` and
  `test_sampling_typed.py:403`: `weight=weight`, where `weight` is the helper's parameter.
  - The new `full_weight=full_weight` (or `block_weight=block_weight`) on the engine and reference calls pairs and is
    classified.
  - Whether the value is a block or a full weight is decided at the helper's callers, which often pass it
    positionally, and rule (c) counts positional weights only for `LeastSquaresBatch`. Every new `full_weight=`
    pass-through still needs a reason line, so the reason must name the callers.
- **A string key.** `tests/test_least_squares_oracle.py:473`:
  `{'weight': dc.replace(good, weight=np.ones(7))}`. The `'weight'` key is a label, but the `dc.replace(...,
  weight=...)` inside it is a call keyword and is seen. A real string key (`**{'full_weight': w}`) would not be seen.

**What (c) cannot see:**

- the truth of a reason line (the reviewer reads `full_weight_reasons.tsv`, apply and entries first);
- positional weights passed to test helpers;
- weights named by string keys;
- new-side indirection, which is only listed, not paired.

The rule that matters holds: no new `full_weight=` exists without a reason line. A block weight silently passed as a
full weight therefore needs a false reason, which a reader can see.

## 3. Part 0's new guards

**Where each sits.** The frontend entries that take a second batch are three:

- `newton_cg` (`val_batch`, `t3toolbox/optimizers.py:450`, prepared at `:839-841`);
- `mc_sgd` and `adam` (`draw=`, wrapped at `:347` and `:375`).

`gradient_descent` and `fitting.local_model` take one batch, and the translation's validation batch and `_old_draw`
reach the same three entries. So every second batch that the frontend hands the backend passes one of the guards:

- `newton_cg`'s refusal at entry (before `_setup`, so before any work);
- `_frontend_draw(draw, batch, x0)`, which now wraps on both layers and compares each drawn batch;
- the backend display's refusal when a display is built (`verbose`).

A user's own `callback=` that queries another batch is the user's code.

**What each compares.** `(getattr(b, 'kind', None), getattr(b, 'order', None))`, two static tuples. A user batch type
without them compares as `(None, None)`: refused against a library batch, accepted against another such type (D13).
The comparisons are structural and read Python values only, so they run eagerly and under a trace alike. The draw
itself is never compiled (`flat_draw`'s text).

**The battery.**
- The guards are comparisons that return the same batch object, or raise. They change no array and no call order, so
  `old` against `part0` stays bitwise.
- The one behavioural change on the ragged layer is that `_frontend_draw` now wraps a user draw there too. That adds a
  Python call per draw, not a number.
- A test that today passes a validation batch of another order on purpose would turn red; the part-0 run records it.

**The backend, deliberately unguarded.** Said in `flat_draw`'s docstring ("the backend does not check it (the
frontend optimizers do)"), which is where a backend user of the default draw reads. A backend user of `draw=` reads
`mc_sgd`'s and `adam`'s docstrings, whose new sentence states the requirement but not that it goes unchecked (finding
14).

## 4. New findings (all minor)

### 11. Audit (b): resolve one level of helper, and list subclass argument shifts (minor)

- **Place:** §7.4 (b).
- **Evidence:** §2 above.
  - 12 helper scopes are "unresolved", and the 27 scopes whose oracle disappears behind them are reached only through
    the call chain.
  - A scope with two distinct old pairs passes a batch converted to the wrong one (none in `tests/`).
  - A subclass's own arguments after the shift are set aside.
- **Resolve:**
  - When a new batch's pair is the scope's own parameter names, substitute each call site's arguments for them and
    compare with the old oracle pairs of the call site's scope. That makes the `_batch`, `Fixture.batch` and `_fits`
    cases mechanical.
  - List every removed oracle construction whose scope keeps no paired batch.
  - List as ambiguous any scope with two distinct old pairs.
  - List each subclass construction's old and new argument lists side by side.

  None of this changes the design. It changes how much part 0's reviewer reads by hand.

### 12. Part 1 must rewrite the tests that hand the reference a now-refused form (minor)

- **Place:** §2.2 part 1 ("Red at its end: none expected"), §6.1, §6.2.
- **Evidence:** `tests/test_engine_data_space.py:139-158` (`TestAgainstTheReference`) passes the reference
  (`getattr(sdef, name)(..., y=y_arg, weight=w_arg)`) two things it refuses from part 1:
  - a `y` constant in `W` (a size-1 `W`; `y_constant = j % 2 == 1`);
  - weights with a structural axis of 1 (the patterns 'per slot', 'one per mode', 'edge and first jet axis').

  §6.2 rewrites that test (y written out, the patterns as pairs) but does not say in which part, and part 1's bullet
  speaks only of passing the reference "the two weights". If the rewrite waits for part 2, part 1 is red.
- **Resolve:** part 1 takes the rewrite of every test that hands the reference an old form. The reference gets the
  written-out `y` and the `(block, full)` pair; the old engine gets the same written-out `y` and
  `blocks_to_weight(block)` or the written-out full weight. The part's run finds the list.

### 13. Parts 2 and 3: hand-built weights with a size-1 structural axis are red too, and say what part 4 rechecks (minor)

- **Place:** §2.2 part 2 "Red at its end", §5.4.
- **Evidence:**
  - Beside `blocks_to_weight`'s output, hand-built full-form weights with an edge or jet axis of 1 are refused by the
    engine from part 2. For example `tests/backend/test_optimizer_display.py:425`,
    `omega = [np.ones((1,) * m + (40, 1)) ...]` for probe: an edge of 1.
  - **Does enough stay green?** By a grep of the sites of probe derivative fits and probe weights, the red
    concentrates in the fitting and optimizer files: `test_fitting.py`, `backend/test_curvature.py`,
    `test_dispatch.py`, `test_optimizers_frontend.py`, then a few in `backend/test_optimizers.py`,
    `backend/test_uniform_curvature.py`, `backend/test_padded_start.py`, the display, the sweeps. Part 2 changes none
    of these. The engine's and the reference's own tests, which are what part 2 changes, stay green except where
    rewritten. So the list means something for part 2's subject.
  - What it hides: a test node made red by its probe case does not run its other cases. Unittest subtests are recorded
    with their labels (`SUBPASSED(...)` in the `-v` logs), so a red node's apply subtests are still visible. A test
    that stops at its first failure hides the rest until part 4.
- **Resolve:**
  - Add "hand-built probe or apply full-form weights with a structural axis of 1" to the predicted classes.
  - Record the expected red at the subtest level, from the `-v` log, each failing subtest classified by its label.
  - Say that part 4's green run is the first check of part 2's engine on those hidden cases.

### 14. The backend's `mc_sgd` and `adam`: say that a user draw goes unchecked (minor)

- **Place:** `prose_edits.md` §1.9 (the `mc_sgd` and `adam` docstrings).
- **Evidence:** "(``draw(rng)``, which returns a batch of the full batch's kind and jet shape; ...)" states the
  requirement where a backend `draw=` user reads, but not that the backend does not check it. That is said only in
  `flat_draw`, which such a user does not call.
- **Resolve:** add "(not checked here; the frontend checks it)".

### 15. The transitional texts of part 0, and one hand-written exception line (minor)

- **Place:** `prose_edits.md` §1.2, §1.4.
- **Evidence:**
  - Part 0 keeps "today's sentences on `weight`" but writes the final sentences on `y`: §1.2's bullet "Only its `C`
    axes may be 1 ... a `y` the same for every sample is not a dataset", and §1.4's "``y`` exactly ``z``'s shape but
    for ``C`` axes of 1 (`check_data`)".
  - Both are false from part 0 to part 2: `check_data` and the engine still take a size-1 axis anywhere until parts 2
    and 3. The squash ships only the final text, but each part's review reads its own docstrings as true of its code.
  - §1.2's example writes the exception line by hand ("TypeError: __init__() missing 1 required positional argument:
    'sample'"). Python 3.11 prints `LeastSquaresBatch.__init__() missing ...` (measured, round 1).
    `+IGNORE_EXCEPTION_DETAIL` makes either pass, but the house rule is a pasted line.
- **Resolve:**
  - Part 0's texts keep today's `y` sentences too; the `y` sentences change in the part that changes the rule (2 for
    the engine, 3 for `check_data`).
  - Paste the 3.11 line.

## 5. What part 0's build review must check first

In this order:

1. **The audit (b) output** (`k1/part0_audit.txt`):
   - the counts of paired and unresolved sites;
   - every unresolved helper with its call sites, against the removed oracle pairs there (or finding 11's mechanical
     version);
   - the subclass constructions' argument lists, old against new.
2. **The battery `old` against `part0`**, bitwise in every number, optimizer paths and jitted runs included. Not
   "within rounding".
3. **The real class against §4.6.** Re-run my `k1/reviewer_tools/batch_model.py` attacks against the built
   `LeastSquaresBatch`, on both environments:
   - old positional forms;
   - `replace`, `deepcopy`, `pickle`;
   - placeholder leaves;
   - one trace per kind and order;
   - an `order` given as numpy ints.
4. **The guards:**
   - `_frontend_draw` on a ragged and a uniform `x0`;
   - the frontend `newton_cg`'s refusal of a mismatched `val_batch`, before any work;
   - the display's refusal;
   - `_require_draw` with a method-shaped `n_measurements`;
   - each with its mutation.
5. **The five `getattr`/`hasattr` sites and the `.kind`/`.order` reads off an oracle**, all gone from the library (my
   scan of round 1 is the list), and the nine subclasses rewritten.
6. **The tally against `k0/k0_suite_v.log.gz`:** only the renamed `check_batch` tests move, each mapped.
7. **Part 0's docstrings are true of part 0's code** (finding 15).

## 6. What I ran

- An `ast` scan of `tests/` for oracle and batch constructions by scope (scratch, `scratchpad/reviewer3/multipair.py`):
  the counts of §2.
- A grep of the sites of probe derivative fits and probe weights per test file (for finding 13).
- Reading the cited test and library lines.

No test was run, and nothing is left running. The repository is unchanged but for this file.
