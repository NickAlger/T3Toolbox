# K1 part 0 handover: the kind and the order move into the batch

Implementer (the second), 2026-11-25, on the local branch `k1-wip` (from `dfb9585f`: K0 plus K1's design records),
uncommitted. Part 0 of `K1_DESIGN_NOTE.md` (§2.2 part 0, §4.6, §4.7, §4.9 to §4.11), with round 2's findings folded
in (`K1_DESIGN_RESPONSE_r2.md`) and `k1/prose_edits.md` §1's texts (the interim ones per finding 15).

**Status:** complete.

## 1. What changed

### Library (6 modules)

- `t3toolbox/backend/fitting.py`
  - `LeastSquaresBatch`: `@dataclass(frozen=True, eq=False, init=False)`, fields `kind, order, sample, y, weight,
    minibatch_scale, drawn_indices`; a hand-written `__init__` with every field after `y` keyword-only, checking the
    kind (a `str`, then one of three), the order (`contractions._read_order`: a tuple of nonnegative ints, never a
    `bool`, normalized to Python ints) and the entries kind's `()`; the methods `check(shape, stack_shape=())` (the
    oracle's `check_batch`, body unchanged but `self`), `take(indices)` (the oracle's, body unchanged), the property
    `n_measurements`, the private `_w_axes()`, `_W()`; `data_metric(r)` unchanged. The pytree: leaves `_BATCH_LEAVES =
    ('sample', 'y', 'weight', 'minibatch_scale', 'drawn_indices')` (was `_BATCH_FIELDS`), aux `(kind, order)`, rebuilt by
    keyword. The class docstring and the methods' per `prose_edits.md` §1.2 to §1.6 (interim: today's sentences on `y`
    and the weight).
  - `LeastSquaresOracle`: no fields, no `__post_init__`, no batch methods; `query` reads `batch.kind`, `batch.order`,
    `batch._W()`. Class docstring per §1.7 (its examples on batches of each order; the trap of a forgotten jet shape
    stated on the batch).
  - Module docstring per §1.1; the messages of `_take_W` name `LeastSquaresBatch.take`.
- `t3toolbox/backend/uniform_fitting.py`: `pack_batch(batch, N=None)` (the kind and the order the batch's; `N=None`
  the largest mode size of a ragged sample); the module docstring's call example.
- `t3toolbox/backend/optimizers.py`: the `Oracle` protocol's text (§1.8); `flat_draw(batch, batch_size)` (O1, ruled);
  `_require_draw(batch, draw, who)`, refusing a callable `n_measurements` with its own message and a non-integer one;
  `mc_sgd`, `adam`: their calls, `draw=` comments and docstring sentences (§1.9, with finding 14's "not checked
  here"); four doctests of the module (`local_model`, `point_objective`, `padded_start`, `newton_cg`) in the new form.
- `t3toolbox/backend/optimizer_display.py`: `make_newton_display` reads `order` off the batch and refuses a
  `val_batch` of another kind or order (`getattr`, so a user batch type without them is not refused, D13); its
  docstring (§1.10) and doctest; `_format_table`'s "(a batch that names none)".
- `t3toolbox/fitting.py`: `_is_packed(batch)`, `prepare_batch(batch, x, check=True)` (`batch.check` when callable,
  `pack_batch(batch, x.N)`); `local_model`'s call, comments and doctests; `LocalModel`'s doctest.
- `t3toolbox/optimizers.py`: `_setup(geometry, batch, x0)`; the new `_same_kind_and_order(batch, other, who)`;
  `_frontend_draw(draw, batch, x0)` wraps every user draw on both layers and compares each drawn batch's kind and order
  (the ruling on O1); `newton_cg` compares its `val_batch` at entry, before `_setup`; the translation's `_old_fit`
  builds `LeastSquaresOracle()` and `LeastSquaresBatch(kind, order, ..., weight=w)`, `_translate` reads
  `batch.n_measurements`; module docstring, parameter comments of `gradient_descent`, `mc_sgd`, `adam`, `newton_cg`,
  their docstrings (§1.11) and five doctests.

### Tests, examples, pages (28 files)

- **Scripted, logged edit by edit:** `k1/part0_rewrite.py` rewrote 71 `LeastSquaresOracle(...)` constructions to
  `LeastSquaresOracle()` and 61 `LeastSquaresBatch(` constructions whose scope had exactly one old oracle pair in reach
  (the third and later positionals became `weight=`, `minibatch_scale=`, `drawn_indices=`); it left 23 as MANUAL
  (`k1/part0_rewrite_log.txt`). `k1/part0_methods.py` moved 59 calls (`oracle.take(b, i)` to `b.take(i)`,
  `oracle.n_measurements(b)` to `b.n_measurements`, `oracle.check_batch(b, ...)` to `b.check(...)`,
  `pack_batch(k, o, b, N)` to `pack_batch(b, N)`, `flat_draw(o, b, n)` to `flat_draw(b, n)`, `prepare_batch(o, b, ...)`
  to `prepare_batch(b, ...)`; `k1/part0_methods_log.txt`). Each edit replaces the exact source segment of one call.
- **By hand:** the 23 MANUAL sites (the nine test subclasses' constructions, `_OffsetOracle(offset, key)` among them;
  the helpers `_batch` of the display tests and `Fixture.batch`; batches with no oracle in scope, each kind and order
  read off its test); `tests/test_jax_absent.py`'s script; the mocks of `check_batch` (now `LeastSquaresBatch.check`);
  `tests/_fits.py`'s reads of `fit.oracle.kind` (now `fit.batch.kind`) and its docstring; `tests/test_jets.py`'s pin of
  the checkers' one caller (now `LeastSquaresBatch.check`; it also asserts that no backend module calls a `check`);
  the translation tests' `(oracle.kind, oracle.order)` (now the batch's); the no-draw tests of both tiers (their subject
  moved from the oracle to the batch: `_BareBatch`, `_MethodCountBatch`, `_NoDrawBatch`); long lines wrapped.
- **Examples** (5 files): the construction lines only. **Pages** (2): the constructions of the doctests of
  `docs/fitting_and_optimization.md` and `docs/numerical_contracts.md`.

## 2. The tests added or rewritten, and what each establishes

New (part 0's guards and the batch's constructor, note §6.3 part 0):

- `tests/test_least_squares_oracle.py`
  - `TestTheOracle.test_value_hash_and_equality` (rewritten): the fieldless oracle, every instance equal and of one
    hash; the form before K1, `LeastSquaresOracle('probe', (2,))`, refused.
  - `TestTheBatchConstructor.test_the_kind_and_the_order` (from the oracle's `test_order_is_required_...`): an int, a
    list, a `bool` order refused (`TypeError`), a negative one (`ValueError`), entries `()` only, an unknown kind, the
    old positional forms (three positionals: `TypeError` naming the kind; two: Python's missing `sample`), a numpy-int
    order normalized to Python ints.
  - `test_the_fields_after_y_are_keywords`: a positional weight refused; `dataclasses.replace` keeps the static fields
    and re-validates.
  - `test_the_static_fields_are_aux_data`: the treedef of an `np.int64` order equals that of a Python int; an
    unflatten with `object()` leaves reads no array; a jitted function traces once for two batches of one kind and
    order, and once more for each other order and kind.
  - `TestTheBatch.test_check` (renamed from `test_check_batch`, same cases).
- `tests/test_optimizers_frontend.py::TestOneKindAndOrderPerFit`: the frontend `newton_cg` refuses a `val_batch` of
  another order or kind before `_setup` (mocked: not called) and accepts its own; `mc_sgd` with a user draw that
  returns another order or kind is refused, on a ragged and a uniform `x0`, and runs with its own.
- `tests/backend/test_optimizer_display.py::TestMakeNewtonDisplay::test_a_validation_batch_of_another_kind_or_order`.
- `tests/backend/test_optimizers.py::TestOracleSeam::test_optional_outputs_are_checked_where_needed` (rewritten): the
  default draw's requirement is on the batch (`_BareBatch`, no `take`); a batch whose `n_measurements` is a method
  (`_MethodCountBatch`) gets its own message; with a draw a batch without `take` runs; the oracle's own requirements
  (`Jp`, `r`, `Kp`) unchanged on `_BareOracle`.
- `tests/test_optimizers_frontend.py::...test_a_user_batch_without_a_draw` (was `..._user_oracle_without_a_draw`), the
  same on the frontend.

Every other changed test changes its constructions and moved calls only (the line audit, §3).

## 3. The gate

All runs on the project environment (Python 3.11, numpy 2.4.6, jax 0.10.2), `-n 2`, one at a time.

| step | command | result |
|---|---|---|
| suite (`-v`, serves the tally) | `pytest tests/ -v -rfEs -p no:cacheprovider -n 2` | **1,887 passed, 5 skipped, 1 xfailed, 90,148 subtests passed**, 0 failed (33:54); log `k1/part0_suite_v.log.gz` |
| doctest-modules | `run_gate.sh doctest-modules` | 321 passed |
| doctest-quickstart | `run_gate.sh doctest-quickstart` | 1 file, 0 failures |
| doctest-pages | `run_gate.sh doctest-pages` | 47 files, 0 failures |
| examples | `run_gate.sh examples` | 13 of 14 passed; `fit_hilbert_from_probes_adam.py` failed (my construction line named its batch `batch`, which shadowed the example's minibatch size): renamed `drawn` there and in the optax twin, both rerun alone, exit 0 |
| sphinx | `run_gate.sh sphinx` (`-W`) | 0 warnings (749 s); its output deleted |
| slow tier, serially | `run_gate.sh slow` | **52 passed, 13,076 subtests** (1,431 s), K0's counts exactly |

Lowest MemAvailable over the gate's steps: 9.9 GB (doctest-modules).

**Edits after the suite run** (each rerun): the annotation-only edits of §4b; `tests/test_least_squares_oracle.py`, where
`test_reachable_from_the_frontend` had slid from `TestTheOracle` into the new class (moved back; the file rerun: 19
passed, 290 subtests, the same counts as in the suite run; `k1/part0_lso_rerun_v.log.gz`); the two examples above;
`TestCheckBatch` in `tests/test_oracle_data_space.py` (a helper lambda I had introduced to shorten six construction
lines put them out of the audit's sight; written out again, wrapped; rerun: 1 passed, 21 subtests). The battery was
re-recorded after these edits.

**The battery** (`k1/battery.py`; note §7): `old` on a `git archive` of `282b46a9`, `part0` on this tree, re-recorded
after the edits above: **bitwise, 9,156 of 9,156 arrays and 3,017 of 3,017 notes** (`k1/battery_old_vs_part0.txt`):
every engine and oracle output of the three kinds on the three frames at every weight form (none, block, full,
both), full and drawn batches, `take`, `check`, `pack_batch`, the default draw, the display's records, the frontend
optimizer runs (`newton_cg` ragged with a block weight and with a validation batch, uniform apply; `mc_sgd` with the
default draw and with a user draw, ragged and uniform; `adam`; `gradient_descent`; the translated old forms with a
probe and a probe-derivative weight) and one jitted query.

**The audit (b)** (`k1/part0_audit.txt`, `diff_audit.py dfb9585f --part 0`): PASS. 116 batch constructions paired
mechanically with the old oracle pair of their scope (one level of helper resolved where the call site's scope has
one); listed for reading by hand: 35 constructions with no old oracle in reach (new tests, batches whose oracle came
from a fit or a helper: each reads its kind and order off its own test), 16 helper call sites whose scope holds no
old oracle (`_batch` of the display tests, `Fixture.batch`), 11 constructions in the 3 scopes with two or more old
pairs (`LeastSquaresOracle`'s and `newton_cg`'s doctests, `docs/fitting_and_optimization.md`: each read by hand, and
the doctests' outputs pin them), 9 subclass constructions with their old and new argument lists, 24 reads of
`<x>.oracle.kind/order` with their replacements, 38 removed oracle constructions (`ORPHAN`: every removed
`LeastSquaresOracle(...)` whose scope keeps no batch of its pair, mostly scopes whose batch comes from a fixture), 7
scopes on one side only (new tests; the docstrings that moved with `check_batch` to `check`).

**The line audit (a)**: no tolerance changed anywhere; the assertions that changed are those of the new and
rewritten tests of §2 and the moved calls (`oracle.n_measurements(b)` to `b.n_measurements` inside an
`assertEqual`, ...), each listed in `k1/part0_audit.txt` (`LINE` entries).

**The tally** against K0's log (`k1/tally.py k0_suite_v.log part0_suite_v.log`, `k1/tally_k0_vs_part0.txt`, taken
before the move of `test_reachable_from_the_frontend`, which then accounts for 2 of its 9 differences): the new nodes
(`TestTheBatchConstructor` x3, `TestOneKindAndOrderPerFit` x2, the display's refusal); `test_optional_outputs_are_
checked_where_needed` with 2 subtests more (the method-shaped batch); the sharding test that K0's log recorded
failing before its floor fix (455 subtests and a failure there; 228 and a pass now, as K0's handover said); the renamed
`test_check`, `test_a_user_batch_without_a_draw`, `test_the_kind_and_the_order` read under their new names. Nothing
else differs.

**The mutations** (`k1/part0_mutations.py`, a scratch copy, `python -B`; `k1/part0_mutations.txt`): all 7 caught: the
order not normalized, the display not refusing, `_frontend_draw` not comparing, the frontend `newton_cg` not
comparing, `_require_draw` accepting a method, the oracle reading a fixed kind, the pytree dropping the order.

**The kept-text check** (`k1/kept_text_check.py dfb9585f`, `k1/part0_kept_text.txt`): in the six library modules
part 0 touched, every statement whose text changed is one §1 names; every other statement, signature comments and
columns included, is unchanged (27, 28, 220, 11, 66 and 37 unchanged statements). The added and removed lists are
exactly the methods that moved from the oracle to the batch.

**The floor**: `k1/part0_floor_smoke.py` on the floor environment (Python 3.9.19, jax 0.4.30) and on 3.11: the same
results (§5).

## 4. Departures from the note, and choices

- **Five examples, not seven** (note §2.2, D14 said seven): `fit_hilbert_from_entries_lbfgs.py`,
  `fit_hilbert_from_probes_adam.py`, `fit_hilbert_from_probes_optax.py`, `fit_hilbert_tensor_newton_cg.py`,
  `fit_varied_rank_tensor_newton_cg.py`; the scan's "5 (5)" in §5.1 was right, the prose's "seven" a slip. Two of
  them build the batch on its own line now (a long line otherwise).
- **`prepare_batch` checks a batch when it has a callable `check`** (`callable(getattr(batch, 'check', None))`), where
  the oracle's `check_batch` was found by `hasattr`: a user's batch type whose `check` is not a method is not called.
- **`_frontend_draw` takes the full batch** (`_frontend_draw(draw, batch, x0)`): the comparison needs it. The
  comparison and the frontend `newton_cg`'s use one private helper, `_same_kind_and_order(batch, other, who)`.
- **`_require_draw` refuses a non-integer `n_measurements`** as well as a callable one (note §4.11): a `None` or a
  missing attribute gets the general message, a method the specific one.
- **The no-draw tests' subject moved** with the contract (from an oracle without `take` to a batch without it); the
  oracle-side checks of those tests (`Jp`, `r`, `Kp`) are unchanged.
- **The batch's `data_metric` doctest** passes `minibatch_scale=0.5` by keyword (it was the fourth positional), and the
  oracle's docstring states the trap of a forgotten jet shape on the batch (`LeastSquaresBatch('probe', (), ...)`).
- **The diff audit (b)** treats a scope present on one side only (a new test, a docstring that moved with its method)
  as listed, not as a count failure.
- **A scripted wrap of long lines went wrong and was undone**: it rewrapped pre-existing long lines of the touched test
  files; every such line was joined back against the base (`git diff` then holds only part 0's lines; the line audit
  lists every changed line that is not a construction or a moved call, for the reviewer to read).

## 4b. The `typ.Any`s left in the signatures part 0 touched (`K_BRIEF.md` §5)

| signature | `typ.Any` | why it stays | removed by |
|---|---|---|---|
| `LeastSquaresBatch` fields and `__init__`: `sample` | `typ.Any` | a per-mode object in two forms (a sequence of `d` input jets ragged, one packed array uniform) or the entries index; no single type until `PerSite` | K2 (`PerSite`), K3/K5 (the forms) |
| `LeastSquaresBatch`: `y`, `weight` | `typ.Any` | of the type of `z`: per mode in two forms for probe, one array for apply and entries | K1 part 4 (`full_weight` in the same position), K2 (`PerSite`); `block_weight` will be `typ.Optional[NDArray]` |
| `LeastSquaresBatch.data_metric(r)` | `typ.Any` in and out | `r` of the type of `z` in either form | K1 part 4 / K2 (`PerSite`) |
| `flat_draw(batch, ...)`, `_require_draw(batch, ...)` | `typ.Any` | a duck-typed user batch (the protocol asks only for `take` and `n_measurements`) | none expected (a `Protocol` type could name it; not planned) |
| `LeastSquaresOracle.query(..., held: typ.Any = None)` and its return's `held` | `typ.Any` | unchanged by part 0 (the oracle's own held record) | not part 0's |

Changed to a precise type in part 0 (a small departure from the note's text, which wrote `typ.Optional[typ.Any]`):
`minibatch_scale: typ.Optional[typ.Union[float, NDArray]]` on the field and in `__init__` (a number or a 0-d array;
traced under jit, which annotations do not see); the field comments then fall in two aligned groups (the signature
style's rule for a large spread of type lengths). In the frontend helpers part 0 touched, `x0: Point` and
`draw: typ.Optional[typ.Callable]` are annotated (`_setup`, `_frontend_draw`); their `batch` parameters, and
`prepare_batch`'s and `_same_kind_and_order`'s, stay unannotated as the module's other helpers are: a duck-typed
batch (a `LeastSquaresBatch` or a user's type, read by `getattr`). These edits are annotations and comments only,
made after the `-v` suite run had started (the run read the code before them; the gate's other steps read after).

## 5. For the orchestrator, and outside the part

- **The battery's `old` record** is kept out of the repository, as M2's was: `scratchpad/implementer2/battery/old.npz`
  and `.json` (3 MB together), regenerable from `282b46a9` in about 10 seconds (`git archive 282b46a9 t3toolbox tests`
  into an empty folder, then `PYTHONPATH=<it> python -B k1/battery.py <prefix> old` from that folder; two runs are
  bitwise identical, checked). The comparison's output is kept: `k1/battery_old_vs_part0.txt`.
- **The floor:** a smoke script of the real batch (`k1/part0_floor_smoke.py`: construction, the old positional forms,
  `replace`, `deepcopy`, `pickle`, an unflatten with placeholder leaves, a query on a drawn batch, one trace per kind
  and order under `jax.jit`, the default draw) runs on Python 3.11 / jax 0.10.2 and on the floor environment `tttt`
  (Python 3.9.19, jax 0.4.30) with the same results.
- **Sphinx and the slow tier were run** (part 0 rewrites docstrings with cross-references, and changes the batch
  every fit of the sweeps builds): Sphinx 0 warnings; the slow tier in §3.
- **For K4** (recorded, nothing to do now): `_frontend_draw` now wraps every user draw on both layers; K4 rewrites it
  and keeps the comparison. `prepare_batch` checks a batch with a callable `check`.
- **Nothing found outside the part.**
