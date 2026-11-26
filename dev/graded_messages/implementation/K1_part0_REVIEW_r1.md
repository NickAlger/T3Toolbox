SIGN OFF

# K1 part 0, build review, round 1

Reviewer 3 (Claude), 2026-11-25. I judged the working tree of `k1-wip` against `HEAD` (`dfb9585f`, which differs from
K0's `282b46a9` in `dev/` only, checked with `git diff --quiet`). The code first, the handover after. Scratch:
`scratchpad/reviewer3/`. My tools: `k1/reviewer_tools/real_batch_attack.py` (new, kept), reviewer 2's
`k0/reviewer2_tools/{tally2,unitdiff}.py`, and the short scripts described below.

**Verdict: SIGN OFF.** Five minor findings, none blocking; finding 5 is resolved by my rerun.

**Summary.** Part 0 is the refactor the note designed.
- Every `(kind, order)` I read is the one its old oracle had: the 116 the audit paired, and by hand all 77 it listed.
- The battery is bitwise.
- The batch class holds on both environments.
- Each guard sits where it should and is caught by its tests, except one branch of `_require_draw` (finding 3).
- No test moved, and no library statement changed outside what part 0 means to change.

The findings are all minor: a comment column, a lost blank line before a doctest, one untested branch, one blind spot
of the audit, and the suite run, which began before two library edits (resolved by my rerun, §10).

## 1. The kind and the order: the audit, then the hand reading

**The rerun.** `diff_audit.py dfb9585f --part 0` reproduces `k1/part0_audit.txt` line for line:

- 116 PAIRED, 35 NOORCL, 16 HELPER, 11 AMBIG, 9 SUBCLS, 38 ORPHAN, 24 OREAD, 7 SCOPE, 5 TWOPRS;
- 760 LINE entries;
- RESULT: PASS.

**What it could not pair, read by hand against `HEAD`:**

- **11 AMBIG**, in the 3 scopes with several old pairs:
  - `docs/fitting_and_optimization.md`: the `apply ()` example at `:149`; the derivative-data examples at `:429`
    (`('probe', (3,))`) and `:435` (`('probe', ())`);
  - `LeastSquaresOracle`'s docstring: `batch` and `f_right` at `(1,)`; `plain` at `()`; `mixed` at `(1,)`, the data
    at `()` against an oracle at `(1,)`, which is the trap the doctest shows;
  - `newton_cg`'s docstring: four constructions, each against the oracle in its example.

  Every pair is the old oracle of its own example, and each doctest's printed output pins it.
- **16 HELPER** call sites:
  - `_batch` of the display tests: its calls at `:370-376` belong to the new test; `:636-637` give `('probe', ())`,
    the old `cls('probe', ())`;
  - `Fixture.batch` of `test_least_squares_oracle.py`: its calls at `:162`, `:191`, `:474` and `:495` are unchanged
    from `HEAD`, and they already passed `kind, order`.
- **35 NOORCL:**
  - the batch constructor's own tests (new);
  - `TestOneKindAndOrderPerFit` (new);
  - the doctests of `data_metric`, `check`, `LeastSquaresBatch` and `pack_batch`;
  - `test_data_metric.py:312-328` and `test_type_laws.py:85`, where the batch's `(kind, order)` is the one its `y`
    and `blocks_to_weight` weight were built for. Part 0's `data_metric` ignores them, but part 4's will dispatch on
    them, so they had to be right now, and they are;
  - `test_optimizers.py:1210` (`('apply', ())`, its problem is an apply fit);
  - `test_uniform_fitting.py:363` and `test_oracle_data_space.py:268`, `:283` (the kind and order of their own
    `y`).
- **9 SUBCLS:**
  - the constructions lose exactly the base's two arguments;
  - `_OffsetOracle(offset, key)` keeps its own two fields in their order (`offset: float`, then
    `start_key`);
  - `HalfOracle()` is still a different type from `LeastSquaresOracle()` (the test's subject).
- **38 ORPHAN**, removed oracles whose scope keeps no batch. Two independent checks:
  - for every test function of `test_optimizer_display.py`, the arguments of its `_batch(...)` calls equal its
    removed oracle's pair (all 12);
  - across every changed test file, a scope-by-scope `ast` scan comparing the new batch-building calls' `(kind,
    order)` with the old oracle pairs of the same scope: 58 scopes, two "unmatched". One is a new `setUp`
    (`TestOneKindAndOrderPerFit`) sharing its name with an old one; the other is `_BareBatch(b.kind, b.order, ...)`,
    a copy of the fit's own batch.

**What the audit cannot see** (finding 4): a subclass built through a variable. `test_optimizer_display.py:620`,
`cls('probe', ())` → `cls()`, is in neither SUBCLS nor ORPHAN. Read by hand: it is right.

**The examples.** `git grep` at `HEAD` finds the constructions in exactly five examples. "Seven" in the note was a
slip (the scan's "5 (5)" was right), so no example was skipped.

## 2. The battery, `old` against `part0`

- **The run.** I regenerated `old` from a `git archive` of `282b46a9` (`PYTHONPATH` set to it, run from that
  folder) and recorded `part0` on the working tree. Result: **9,156 of 9,156 arrays bitwise** (`compare` checks the
  dtype, the shape and `tobytes()`), and 3,017 of 3,017 notes equal. The 7 notes that raise are the `check_bad` rows,
  on both trees.
- **What it covers:** what §7.1 promises for part 0. That is the engine and the oracle on the three frames, at every
  weight form, on full and drawn batches; `take`, `check` and `pack_batch`; the default draw; the display; the
  frontend optimizer runs on both layers (`mc_sgd` with the default and with a user draw included); the translated old
  forms (probe, and probe derivatives); and one jitted query.
- **What it cannot catch:** the battery builds the oracle and the batch from the same case variables (`_Api.batch(kind,
  order, ...)`), so it tests the library, not the test files' conversions. Those are the audit's (§1).
- **No library mistake of part 0 that changes a number can hide from it:**
  - the kind and the order that `query` reads, `take`'s drawn batch, `pack_batch` and the translation each feed rows
    of their own;
  - a mistake that only reads the wrong batch's kind in a validation or draw path is the guards' (§4), not the
    battery's: its validation batch and user draw have the full batch's kind.

## 3. The batch class (`reviewer_tools/real_batch_attack.py`, the built class)

Each environment ran 31 checks: Python 3.11.15 / jax 0.10.2 / numpy 2.4.6, and the floor environment (3.9.19 / jax
0.4.30 / numpy 1.22.4). All pass on both. The checks:

- **The old positional forms:**
  - `(ww, y)` fails on the missing `sample`;
  - `(ww, y, w)`, `(ww, y, w, 0.5)` and the entries `(index, y, w)` fail with a `TypeError` naming the kind;
  - a fifth positional is refused.
- **Refused:** an order that is a `bool`, a list, an int or negative; an entries order of `(1,)`; an unknown kind; a
  kind of `None`.
- **The order:** `(np.int64(2),)` is normalized to Python ints, and its treedef equals that of `(2,)`.
- **Copies:**
  - `replace` keeps the kind and the order, and re-validates (`kind='nope'`, `order=[1]`);
  - the frozen guard, `deepcopy` and `pickle` work; `==` raises.
- **Rebuilding:** an unflatten with `object()` leaves works, and so does `tree_map` to an `in_axes` spec.
- **Compilation:** one trace per kind and order across three batches of one kind and order. The drawn batches trace
  once together (they have the scale and indices leaves, as today) and not again for the next draw.
- **The drawn batch:** `take` keeps the kind and the order, with scale `n/b`; `n_measurements` is an `int`.
- **The fieldless oracle:**
  - every instance is equal and of one hash, and the old form `LeastSquaresOracle('probe', (2,))` is refused;
  - a fieldless subclass differs in `==` and `hash`;
  - under `jit` with the oracle static, two base oracles trace once and the subclass once more.

I also ran on the floor `tests/test_least_squares_oracle.py`, `TestOneKindAndOrderPerFit` and `TestOracleSeam`:
33 passed.

## 4. The four guards, with my own mutations

**Where each sits** (read in the code):
- `_same_kind_and_order(batch, other, who)` (`t3toolbox/optimizers.py:117`) compares
  `(getattr(b, 'kind', None), getattr(b, 'order', None))`. It is used by:
  - `_frontend_draw(draw, batch, x0)` (`:131`), which wraps every user draw on both layers and compares before
    packing;
  - the frontend `newton_cg` (`:862`), at entry: after the option checks, before `_require_layer`, `prepare_batch`
    and `_setup`.
- The display's own refusal is at `optimizer_display.py:627-632`.
- `_require_draw` is at `backend/optimizers.py:647-665`.

All three comparisons read Python statics only: structural, and valid eagerly or under a trace. The draw itself is
never compiled.

**Mutations**, in a scratch copy, `python -B`, each run against the part's guard tests:

| mutation | result |
|---|---|
| M1 `_frontend_draw` does not compare | caught (4 failed) |
| M2 the frontend `newton_cg` does not compare its `val_batch` | caught |
| M3 the display does not refuse | caught |
| M4 `_require_draw`'s branch for a method-shaped `n_measurements` removed | caught (its own message asserted) |
| M5 `_require_draw` accepts a non-integer `n_measurements` (`or not isinstance(n, numbers.Integral)` removed) | **survived** (finding 3) |
| M6 the draw comparison on the kind only | caught (the wrong-order draw) |

**Paths around them.** A second batch enters the frontend only as:
- `newton_cg`'s `val_batch`;
- a user `draw=` of `mc_sgd` or `adam`;
- the translation's validation batch and `_old_draw`, which reach the same entries.

`gradient_descent` and `local_model` take one batch. A user's own `callback=` querying another batch is the user's
code. In the backend, the optimizers deliberately do not look. That is said where a backend user reads: `flat_draw`'s
docstring ("the backend does not check it (the frontend optimizers do)"), the protocol's text, and `mc_sgd`'s and
`adam`'s docstrings ("not checked here"). So no path through the frontend reaches the engine without a guard.

## 5. Every reader of the old oracle fields

A grep of `t3toolbox/`, `tests/`, `examples/` and `docs/` finds none left:
- `oracle*.kind|order|take|check_batch|n_measurements`;
- `getattr`/`hasattr` of `kind`, `order`, `take`, `n_measurements`, `check_batch`, `check`;
- `check_batch`;
- `n_measurements(`.

The `getattr` sites that remain are the five new ones, all on batches: the two guards, `_require_draw`, the display,
and `prepare_batch`'s `callable(getattr(batch, 'check', None))`. Besides those: `tests/test_jets.py:895` (asserting
`check_batch` is absent) and `_MethodCountBatch.n_measurements` (the test's own old-protocol batch). The protocol's
text names the batch's members. The nine subclasses are rewritten (§1).

## 6. The tally, and where each test is

`tally2.py` on K0's log against `part0_suite_v.log`, with the two roots:
- 1,804 identical tests; 78 changed; 8 added; 3 deleted, all renames or the moved constructor test (`test_check_batch`
  → `test_check`; `test_a_user_oracle_without_a_draw` → `test_a_user_batch_without_a_draw`;
  `test_order_is_required_and_entries_has_no_jets` → `TestTheBatchConstructor.test_the_kind_and_the_order`).
- Of the identical tests, two nodes differ, both explained:
  - the sharding test K0's log recorded failing before its floor fix, which passes now;
  - `test_reachable_from_the_frontend`, recorded in this log under `TestTheBatchConstructor` (the slip). It is back
    under `TestTheOracle` in the tree, and passed there in `k1/part0_lso_rerun_v.log.gz`.
- `test_optional_outputs_are_checked_where_needed` has 2 subtests more (the method-shaped batch).

**Where each test is.** An `ast` comparison of every changed test file, class by class: no test function changed
class. The only names gone or new are the renames and the new tests above.

## 7. The texts

- **The library:** `unitdiff.py` of the six modules against `HEAD`. Every changed unit is one part 0 names:
  - `backend/fitting.py`: 27 identical units; 10 changed (the docstring, the batch and oracle headers, `data_metric`,
    `query`, `_take_W`, `_take_data_space`, the registration, two field lines); 9 deleted and 10 added (the moved
    methods); the two comment banners rewritten ("the oracle's `take`" → "the batch's", and "checks, counts, draws");
  - `optimizer_display.py`: 28 identical units; 2 changed;
  - `optimizers.py`: 212 identical units; 9 changed (the protocol, four doctests, `flat_draw`, `_require_draw`,
    `mc_sgd`, `adam`).

  In the changed units, every changed line is part 0's (read in `git diff -U0`). The only text that changed without
  a change of content is `mc_sgd`'s docstring paragraph, rewrapped around its inserted sentence.
- **The interim texts are true of part 0's code** (finding 15 of the design): the batch's class docstring keeps
  today's sentences on `y` and the weight ("each axis 1 or that size"); `check` names `check_data` and `check_weight`
  with today's rule; `take` keeps today's paragraph.
- **The After-texts of `prose_edits.md` §1:** a whitespace-normalized search of each After paragraph in its file finds
  every one present, except §1.5's second paragraph, which is part 4's text by the note, so its absence is right.
- **The class docstring's exception line** is pasted from 3.11 (`LeastSquaresBatch.__init__() missing ...`).
- **The tests:** I filtered the test diff for lines that are not a construction, a moved call or a kind/order read.
  The 150 left are all inside the new or rewritten tests, continuation lines of wrapped constructions, or the
  `_fits.py` docstring. No pre-existing line was rewrapped.
- **Trailing-comment columns:** comparing every changed line that carries a trailing comment with its old
  counterpart, and then each such line's neighbours, finds one aligned pair broken (finding 1). The other moved
  columns are single comments with no aligned neighbour.

## 8. The examples

- **Which changed:** five, each by its construction lines only.
- **Shadowed names:**
  - in `fit_hilbert_from_probes_adam.py` and `_optax.py` the new local is `drawn`, used twice, which shadows nothing
    (`batch=` there is the minibatch size's keyword);
  - in `fit_hilbert_from_entries_lbfgs.py` the line `batch = ...` was already there.
- **Each rerun**, alone, on the final tree: `entries_lbfgs` 46 s, `probes_adam` 40 s, `probes_optax` 50 s,
  `tensor_newton_cg` 14 s, `varied_rank_tensor_newton_cg` 23 s. All exit 0.
- **The two pages' doctests:** each construction read against its old oracle (§1).

## 9. The `typ.Any`s

- **The precise types are true:**
  - `minibatch_scale: typ.Optional[typ.Union[float, NDArray]]`. `NDArray` is `Union[np.ndarray, jnp.ndarray]` when
    jax is present (`common.py:75-77`). A traced 0-d scale is a `jnp.ndarray` instance, so the annotation holds under
    jit. `take`'s `n / b` is a Python `float`, and `np.float64` is a `float` subclass. A user's `int` is accepted by
    the numeric tower.
  - `x0: Point` and `draw: typ.Optional[typ.Callable]` in `_setup` and `_frontend_draw`.
- **The list's reasons hold.** One could be precise today: `sample`, `y` and `weight` are, on both layers,
  `typ.Union[typ.Sequence[NDArray], NDArray, None]` (a sequence of `d` arrays ragged, one array packed or apply or
  entries, the index an array). That is the alias `PerSite` will name in K2, so deferring it there is reasonable. I
  note it for the maintainer's audit; I do not ask for it now.
- **The rest stay as listed:** `flat_draw`'s and `_require_draw`'s `batch` (duck-typed), and the oracle's `held`
  (not part 0's).

## 10. The gate

- **Why I reran the suite.** The `-v` suite log ends at 22:31:50 after 33:54, so it started at about 21:58. But
  `t3toolbox/optimizers.py` (22:09:25) and `t3toolbox/backend/fitting.py` (22:09:41) were written after it started,
  and their modules were already imported by the workers. The handover says those edits were annotations and comments
  only. I could not diff the version the run saw, so I rerun the whole suite on the final tree (below).
- **Python 3.9 syntax** of every changed `.py` file: `ast.parse(..., feature_version=(3, 9))` on the floor
  interpreter, all pass; no `X | Y`, `match` or `kw_only`.
- The examples: §8. The battery: §2. The floor tests: §3.

- **The suite, rerun by me** on the final tree (`pytest tests/ -v -rfEs -p no:cacheprovider -n 2`, project
  environment; log `scratchpad/reviewer3/suite_v.log.gz`): **1,887 passed, 5 skipped, 1 xfailed, 90,148 subtests
  passed** (32:34), 0 failed. `tally2.py` against the implementer's log: one node differs, the move of
  `test_reachable_from_the_frontend` back to `TestTheOracle`, nothing else. The one warning is K0's too.
- **The module doctests of the six changed library modules** (`--doctest-modules`, `-n 2`): 19 passed. **The two
  changed pages** (`python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE`): exit 0.
- Not rerun by me: Sphinx and the slow tier. The changes they would see are docstrings with cross-references (checked
  by the doctests and by my reading of the roles in the changed texts) and the batch every sweep builds, which my suite
  rerun and the battery cover. I take the handover's runs (Sphinx 0 warnings; slow tier 52 passed, K0's counts).

## 11. Findings

1. **A broken pair of aligned comments in a page doctest (minor).** `docs/fitting_and_optimization.md:149-150`:
   before part 0, the comments of the `oracle` and `batch` lines were both at column 73. Now `oracle`'s is at 74 and
   `batch`'s at 73. **Resolve:** one space less on `:149`.
2. **`pack_batch`'s docstring lost the blank line before its doctest (minor).** `backend/uniform_fitting.py:105-106`:
   the new sentence "``N=None`` is the largest mode size ... keeps its own width)." runs straight into `>>> import numpy
   as np`. The old text had a blank line there. Doctest still runs it, but reST reads a doctest block only as its own
   paragraph, so Sphinx renders it inside the paragraph (and `-W` is silent). **Resolve:** restore the blank line.
3. **`_require_draw`'s non-integer branch is untested (minor).** `backend/optimizers.py:662`: mutation M5 (accept any
   non-callable `n_measurements`) survives. `_BareBatch` has neither `take` nor `n_measurements`, so its refusal comes
   from the missing `take`. **Resolve:** one more case in `test_optional_outputs_are_checked_where_needed`: a batch
   with `take` whose `n_measurements` is not an integer (`None`, or `4.0`), refused with the general message.
4. **The audit does not see a subclass built through a variable (minor, for the record).**
   `tests/backend/test_optimizer_display.py:620`: `for cls in (_NoBlockSums, _NoDataBlockSums): oracle = cls('probe',
   ())` → `cls()`. It is neither SUBCLS nor ORPHAN. Read by hand: right (its batches are `_batch('probe', (), ...)`).
   **Resolve:** nothing for part 0. For later parts, the audit could treat a call of a name bound by a loop over
   subclass names as a construction, or the handover can list such sites.
5. **The suite log predates two library edits (minor; resolved by my rerun, §10).** The handover says edits after the
   suite run were rerun, but `t3toolbox/optimizers.py` and `t3toolbox/backend/fitting.py` were written while it ran,
   and no rerun is reported for them. **Resolve:** my full rerun on the final tree (§10). For later parts, stop the
   tree before the `-v` run starts, or rerun the files that import what changed.
