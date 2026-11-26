# K1 design note: the weights

Implementer (the second of the arc), 2026-11-25, on `graded-messages` at `282b46a9` (K0 committed; K1's starting
tree; the scans of §1.2 were taken on the working tree just before that commit and rerun on it: identical). Plan phase: nothing under `t3toolbox/`, `tests/`, `docs/` or
`examples/` is changed. Sources: `PACKING_PLAN.md` (revision 5) §2, §3.1, §4.3, §4.4, §5, §7 "K1", §8, §12 to §14;
`PACKING_PLAN_NOTES.md` §3d, §7; `reviews/packing_plan/REVIEW_r4.md`; `DATA_METRIC_PLAN.md` §1 to §4;
`M_ARC_DECISIONS.md` (7, 10, 13, 18); `K_MEASUREMENTS.md` M2 (re-measured here); `M2_DESIGN_NOTE.md` and its review
as the model of a slice built in parts; `K_BRIEF.md`, `K_ARC_DECISIONS.md`. Working files: `implementation/k1/`
(§1.2). Scratch: `scratchpad/implementer2/`.

**Status of this note:** revision 2, after `K1_DESIGN_REVIEW_r1.md` and the orchestrator's rulings (answers in
`K1_DESIGN_RESPONSE_r1.md`): the audits of §7.4 (findings 1, 2), the guards of a drawn and a validation batch in part
0 (O1's ruling), the subclasses (§5.1), the corrected red of parts 2 and 3, `pack_batch`'s own refusal (§4.10),
`_require_draw` (§4.11), the asymmetric tables and two mutations (§6.3, §6.4), §9's additions, §10's rule. The prose
of parts 0 and 1 is in full in `k1/prose_edits.md`; parts 2 to 5 as listed changes, each completed before its part.

**Library methods traced first** (the maintainer's instruction): `TuckerTensorTrain.probe`
(`tucker_tensor_train.py:3660`) → `jets.check_sample` → `probing.t3_probe` (`backend/probing.py:44`) →
`sampling_variables.sampling_variables` (`sampling_variables.py:1678`: `_closure_facts`, the uniform packing and
masking at `:1862-1873`, `_read_setting`, `_check_stacks`, `_check_data_space`, `_run`) → the whole steps
(`sampling_steps.py:909-1099`: `_modes`, `_with_K`, `_weighted_sum`, `r_step`, `f_step`, `w_tilde_step`,
`r_block_sumsq_step`, `Jp_norm_sq_step`) → `jets.data_metric` (`jets.py:579`, `_check_pair`, `_omega_squared`);
`probe_corewise_transpose` (`:3817`, → `t3_probe_corewise_transpose`: the engine with a held `Jp`, the route whose
seed is `_Jp.w_tilde` with no weight); `t3svd` (`:4218` → `backend/t3_svd`); `__add__` (`:1949` → `t3_linalg.t3_add`).
Then, for K1, whole: `backend/fitting.py`, `backend/uniform_fitting.py`, the packers of `ut3_operations.py`
(`:128-240`), `backend/jets.py`, the weight-carrying parts of `sampling_definitions.py` (`:80-130`, `:650-830`,
`:3137-3308`), the `Oracle` protocol, `flat_draw`, `_require_draw` (`backend/optimizers.py:181-231`, `:625-654`),
`make_newton_display` (`optimizer_display.py:549-676`), `t3toolbox/fitting.py` (`:1-110`, `:575-722`),
`t3toolbox/optimizers.py` (`:1-380`), `tests/_fits.py`, and the heads of the M arc's data-space test files.

---

## 0. Summary

K1 replaces the one weight "of the type of `z`" by two named weights that multiply, `block_weight` (one number per
block: `(d,)+F`, `F`, `()`) and `full_weight` (`z`'s shape, a size of 1 only on the stack axes `W` and `C`); makes `y`
exact (only its `C` axes may be 1); splits the public metric by kind; moves `kind` and `order` from the oracle into the
batch with the batch's methods; and renames the weight-carrying helpers. It touches 17 library modules (11 in code:
`sampling_variables`, `sampling_steps`, `jets`, `sampling_definitions`, `backend/fitting`, `uniform_fitting`,
`ut3_operations`, `backend/optimizers`, `optimizer_display`, `t3toolbox/fitting`, `t3toolbox/optimizers`; 6 in
docstrings only: `probing`, `apply`, `entries`, `manifold`, `tucker_tensor_train`, `uniform_tucker_tensor_train`), 34
test files (221 test functions contain a site, §5), 7 examples (only part 0 touches them) and 4 pages with doctests.

What I found against the plan on today's tree:

1. **The plan's build order needs two amendments to be reviewable** (§2.1). (a) Removing names from `jets` in the
   `jets` part breaks `import t3toolbox` itself: `t3toolbox/fitting.py:79-89` re-exports `data_metric`,
   `blocks_to_weight`, `prepare_derivative_data` and `derivative_weight_to_taylor_weight`, so every test would turn
   red at collection and the part's expected-red list would say nothing. So each old name is deleted **with its last
   library caller**: the `jets` part adds the new functions; `pack_data`, `_pack_data_space` and `check_weight` go in
   the fitting part; `blocks_to_weight`, `data_metric`, `prepare_derivative_data` and
   `derivative_weight_to_taylor_weight` in the last part, with the frontend's re-exports. (b) The engine part
   changes the engine's keyword, which the oracle passes on **every** query (`backend/fitting.py:414`,
   `weight=batch.weight`, `None` included): without a bridge every fit of the suite is red from the engine part to the
   fitting part. One bridging line in the oracle, `full_weight=batch.weight`, keeps every fit with no weight green and
   makes every fit whose weight has a size-1 structural axis fail **loudly** on the new rule: probe's
   `blocks_to_weight` output (an edge of 1), and so every probe derivative fit, weighted or not, whose default weight is
   the `k!` table placed by `blocks_to_weight`. Apply's and entries' `blocks_to_weight` output has no size-1
   structural axis: it passes, as a full weight, with today's numbers bitwise. None runs wrong (§2.1).
   (c) The translation of the old call form builds batches, so it moves with the batch's fields (part 4), not with
   the frontend's names (part 5): otherwise 98 old-form calls are red between the two parts. The plan's order
   otherwise survives: part 0, the references, the engine, `jets`, the fitting layer, the frontend's names. Every
   part but 2 and 3 is expected to end green.
2. **`flat_draw(oracle, batch, batch_size)` is a public backend signature whose first argument becomes unread** once
   the draw calls the batch (the plan, §4.3). A test pins its argument names (`tests/backend/test_optimizers.py:1305`).
   Open point O1: I recommend `flat_draw(batch, batch_size)`.
3. **K2's `check_probe_data(value, sample, order, stack_shape=())` cannot tell `y` from a full weight** (plan §3.1:
   "y, a full weight, a residual"), but the two rules differ on `W` (§2.2: `y` exact, a full weight 1 or full). Not
   K1's to build, but K1 decides how the batch's `check` tells them apart until then (§4.7). Open point O2.
4. **Between K1 and K3 there is no public checker of a full weight** (`check_weight` goes in K1, the new checkers come
   in K2/K3): a backend user checks a batch with `batch.check(shape, stack_shape)`, which covers it. Recorded, no
   action proposed (O3).
5. **What "exact" means for the forms in K1.** The plan's §2.2 rule is about shapes; the forms (a tuple on ragged, one
   array on uniform) are K3's and K5's. K1 keeps today's acceptance of forms everywhere (a ragged `y` on a uniform
   frame is packed by the engine, now with `pack_if_ragged`; a mode-leading array on a ragged frame passes where every
   `N_i` is equal) and changes only the shape rule and the names, except where §3.3 says otherwise: the probe helpers
   renamed in K1 refuse a single array from K1. Decision D1 (§11).
6. **The arithmetic of the battery is bitwise for every single-weight case** (§3, `k1/check_math.py`, check 5): the
   new metric forms `(b*b) * r` and `(f*f) * r` exactly as the old `(omega*omega) * r` did on a `blocks_to_weight`
   weight or a full-form one. Only a call with both weights is regrouped (`b² f²` against `(b f)²`, relative 4.5e-16
   measured), and only the oracle's two data sums on a packed `y` change their summation order (one reduction over a
   padded array where the old code reduced each unpadded mode). So the battery can demand bitwise agreement almost
   everywhere, optimizer paths included (§7).

---

## 1. What was measured, and how

### 1.1 The tree

K1's starting tree, `282b46a9` (K0's commit). Counts are by the scripts of §1.2, never by hand, rerun on that commit
(`k1_sites.py` and `signatures.py` give output identical to the files kept); a `file:line` below is at `282b46a9`. The
per-test record of this tree exists: `implementation/k0/k0_suite_v.log.gz` (1,881 passed and the one floor failure
then fixed; with that file rerun, 1,882 passed, 5 skipped, 1 xfailed), so K1 needs no baseline run of its own (§7.3).

### 1.2 The tools (`implementation/k1/`)

- `k1_sites.py` (standard library only; never imports the library; 5 s): every call of a name K1 changes, every
  keyword `weight=`, every read of `.kind` / `.order` on an oracle and of `.weight`, every old-form optimizer call, in
  five places counted apart: the library's code, the library's doctests, the tests, the examples, the pages' doctests;
  plus a plain search of the pages' prose. Within a test file each site is attributed to its `def test_*`. Output on
  today's tree: `k1/sites_before_k1.md` (the summary) and `k1/sites_before_k1_full.md` (every site, `file:line`).
  The scan is re-run on each part's tree: the gate of part 5 is that no call of a removed name is left.
- `signatures.py`: the headers of every function and class K1 changes, verbatim (def line through the colon, comments
  kept). `k1/signatures_before_k1.md` is today's; the handover reruns it for the "after".
- `check_math.py`: the dense checks of §3 (numpy loops; the library imported only where a check compares with
  today's functions, marked so). All pass.

---

## 2. The parts

K1 is built in six reviewed parts on a local side branch (the orchestrator manages it; I never commit or switch
branches). Each part's handover records, by a run, the tests that fail at its end (`k1/part<j>_EXPECTED_RED.txt`, as
M2 did) and the tally against the part before. The part of the order of §7 "K1" that I keep and the two amendments:

### 2.1 Why the order is amended

- **Deletion with the last caller.** `t3toolbox/fitting.py:79-89` imports from `jets` `derivatives_to_taylor`,
  `taylor_to_derivatives`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data`, `data_metric`,
  `blocks_to_weight`, `affine_jet`, `index_to_one_hot`; `t3toolbox/__init__.py` imports `t3toolbox.fitting`. A `jets`
  part that deletes four of them makes `import t3toolbox` raise `ImportError`: the whole suite red at collection, the
  library doctests and pages too. The library callers of each removed name, measured (`k1_sites.py`):

  | removed | library callers today | it goes in part |
  |---|---|---|
  | `_pack_data_space` | the engine (`sampling_variables.py:1872-1873`), `pack_data` (`uniform_fitting.py:90`) | 4 |
  | `pack_data` | `pack_batch` (`uniform_fitting.py:116-117`) | 4 |
  | `check_weight` | `check_batch` (`backend/fitting.py:474`) | 4 |
  | `jet_to_dual_jet`, `dual_jet_to_jet` | none (tests and doctests only) | 3 |
  | `data_metric` | the steps (`sampling_steps.py:959`, `:1023`, `:1097`), `jet_to_dual_jet` (`jets.py:452`), the batch (`backend/fitting.py:116`, `:202`), the frontend re-export | 5 |
  | `blocks_to_weight` | `prepare_derivative_data` (`jets.py:1178`), the translation (`t3toolbox/optimizers.py:196`, `:216`), the frontend re-export | 5 |
  | `prepare_derivative_data`, `derivative_weight_to_taylor_weight` | the translation (`:217`), the frontend re-export | 5 |

- **The bridge in part 2.** The oracle passes `weight=batch.weight` on every query (`backend/fitting.py:414`), `None`
  included, so renaming the engine's keyword would make every fit of the suite raise a `TypeError` until part 4 (the
  measured scope: 34 test files hold a fit). In part 2 that one line becomes `full_weight=batch.weight` (and
  `block_weight=None`), with a comment that part 4 replaces it. Then:
  - an unweighted fit runs as before (bitwise: the engine's arithmetic with no weight is unchanged);
  - a probe fit whose weight is `blocks_to_weight`'s output (`(d,)+F+(1,)*len(W+C)+(1,)`: an edge of 1) is refused
    by the engine's new structural check, loudly;
  - an apply or entries fit whose weight is `blocks_to_weight`'s output (`F+(1,)*len(W+C)`, `(1,)*len(W+C)`) passes:
    its only structural axes are `F`, which it has full, and its `W`, `C` axes of 1 are legal for a full weight. It
    is then applied as a full weight constant over the samples and the stack, which is what it meant: the same
    metric entry by entry, bitwise (`check_math.py` check 5: `(f*f)*r` with `f` the reshaped blocks is today's
    `(omega*omega)*r`);
  - a fit whose weight is a full-form one obeying the new rule runs, and runs right (a full weight is applied
    elementwise, as the old weight was).

  So between parts 2 and 4 no weighted fit runs wrong: each runs right or is refused. The plan's worry ("an
  intermediate tree with the engine on two weights and the oracle on one would run and be wrong on every weighted
  fit") is met by the rule itself, which refuses the one form that would be misread.

### 2.2 The parts

**Part 0: the kind and the order move into the batch** (a refactor; no number changes; suite green at its end).

- `backend/fitting.py`: `LeastSquaresBatch` gains the static fields `kind`, `order` first, with the oracle's
  constructor checks (§4.6), a hand-written `__init__` that makes every field after `y` keyword-only (D2, §11), the
  pytree registration with `(kind, order)` as aux data, and the methods `check(shape, stack_shape=())` (today's
  `check_batch`), `take(indices)`, the property `n_measurements`, and `data_metric(r)` (still
  `jets.data_metric(r, self.weight, self.minibatch_scale)`: the weight field keeps its name, `weight`, until part 4).
  `LeastSquaresOracle` loses its fields and `__post_init__`; `query` reads `batch.kind`, `batch.order`; `_w_axes`,
  `_W` become the batch's private methods.
- `backend/uniform_fitting.py`: `pack_batch(batch, N=None)` (the kind and the order are the batch's; `N=None` the
  largest mode size of the sample). `pack_sample` and `pack_data` keep their signatures (K5 and part 4 delete them).
- `backend/optimizers.py`: the `Oracle` protocol's text (its optional members `check_batch`, `n_measurements`, `take`
  become the batch's; "the optimizers pass the batch through unread" gains the default draw's exception);
  `flat_draw(batch, batch_size)` (O1, ruled) and `_require_draw(batch, draw, who)` call the batch, and
  `_require_draw` refuses a batch whose `n_measurements` is not an integer, naming the property (an old-protocol
  user batch with an `n_measurements()` method, finding r1-7); the parameter comments of `mc_sgd` and `adam`
  (`:1233`, `:1283`) and the two sentences of their docstrings that name "an oracle with `take`" (`:1243`, `:1295`).
- `backend/optimizer_display.py`: `make_newton_display` reads `order` off `batch` (`:622`) and refuses a
  `val_batch` whose `kind` or `order` differs from `batch`'s (structural, at the call, before any run).
- `t3toolbox/fitting.py`: `_is_packed(batch)` and `prepare_batch(batch, x, check=True)` read the batch's kind and order
  and call `batch.check` (D3: the oracle argument goes from both, and from `t3toolbox/optimizers._frontend_draw`).
- `t3toolbox/optimizers.py`: `_old_fit` builds `LeastSquaresOracle()` and `LeastSquaresBatch(kind, order, sample,
  data, weight=w)`; `_translate` calls `batch.n_measurements` (`:249`); `_setup`, `mc_sgd`, `adam`, `newton_cg`
  follow `prepare_batch`'s signature; the parameter comments of `mc_sgd` and `adam` (`:331`, `:354`).
- **The guards that arrive with the move** (the orchestrator's ruling on O1, round 1): once the kind lives in the
  batch, each batch is read under its own, so a validation or drawn batch of another kind or order would be read
  silently under another objective where before it was read under the oracle's. So, in part 0, not K4:
  - the frontend `newton_cg` refuses, at entry and before any work, a `val_batch` whose `(kind, order)` differs from
    `batch`'s (`t3toolbox/optimizers.py:839-841`; the backend display refuses it too, but only when it is built);
  - `_frontend_draw(draw, batch, x0)` wraps every user `draw=` (on both layers; today it wraps only for a uniform
    `x0`) and compares the `(kind, order)` of each batch the draw returns with the full batch's, a `ValueError` naming
    both (two tuples per draw, structural); on a uniform `x0` it then packs as today;
  - in the backend the optimizers do not look (plan §4.3); `flat_draw`'s and the protocol's texts say that a user
    draw returns a batch of the same kind and order as the full batch, and that the backend does not check it.

  Both comparisons use `getattr(..., 'kind' / 'order', None)`, so a user batch type without them is not refused (D13).
  K4 rewrites `_frontend_draw` again and keeps the comparison.
- Every construction is rewritten: `LeastSquaresOracle(` 71 in the tests, 16 in the library's doctests, 2 in the
  library's code, 5 in the examples, 4 in the pages' doctests; `LeastSquaresBatch(` 75, 23, 4, 5, 4 (§5.1); and the
  constructions of the nine test-local subclasses of the oracle (§5.1, about 15). The
  tests that call `oracle.check_batch`, `oracle.take`, `oracle.n_measurements`, `flat_draw`, or read `oracle.kind`,
  `oracle.order` follow (§5.1). The structural pin `tests/test_jets.py:863-893` (the checkers called only inside
  `check_batch`) follows the method's new home and name. `tests/_fits.py`: `fit`, `derivative_fit`, `shared_over`,
  `at_element`, `uniform` read the kind and order off the batch.
- The examples change by these two lines only (seven files, §5.1); P17 still rewrites them.
- **Red at its end:** none expected. **Its review checks on its own:** the battery `old` against `part0`, bitwise in
  every number (§7); the `(kind, order)` audit (§7.4 (b): each `LeastSquaresOracle(kind, order)` removed is paired,
  by `ast`, with the batch constructions it served, the same expressions moved; the pairs it cannot resolve are
  listed for the reviewer to read); the line audit (every changed test line is a construction or a moved method
  call; no assertion and no tolerance changed, §7.4 (a)); the per-test tally against the starting tree (no test
  lost but the renamed `check_batch` tests, each mapped).

**Part 1: the independent references** (suite green at its end).

- `backend/sampling_definitions.py`: every public function's keyword `weight=None` becomes `block_weight=None,
  full_weight=None` (37 functions, the same place in the keyword-only group); `_weight_squared` becomes the product of
  the block row, left-aligned, and the full weight, both checked by the reference's own rule (§4.1), with no import of
  `jets`; `y` is checked exact (no size-1 axis: the reference has no `C`); the module docstring's paragraphs on `y` and
  the weight (`:80-117`) and the docstrings of `w_tilde`, `f`, `r_block_sumsq`, `Jp_norm_sq` with their doctests.
- `tests/oracle_sweeps/oracle.py`: additive `weighted_residual_two(kind, Sx, y, block_weight, full_weight)` and
  `weighted_two(kind, like, value, block_weight, full_weight)`, the two weights written out to `z`'s shape by
  `np.repeat` (no broadcasting), the block row placed by an explicit loop over the blocks, no library code. The old
  functions stay for the sweeps' old-vocabulary path.
- Brute-force pins of the reference (`np.ndindex` loops over the dense `z`, `Jp`): `w_tilde`, `f`, `r_block_sumsq`,
  `Jp_norm_sq`, `g` (through `<g, p> = d/dt f`, exact along one variation), `JtJp` (through `<JtJp, q> = Σ M Jp Jq`),
  with a block weight alone, a full weight alone (per entry; per slot written out on `W` and the edge; with `W` axes
  of 1), both, and neither; sizes of `F`, `W` and `N_i` distinct and all equal (where a transposed axis cannot fail by
  shape). The reference's refusals: a block weight of another shape (a `(d,)` block at `order=(n,)` with `d = n+1`,
  the coincidence of the plan's E3b, included), a full weight with a size-1 structural axis, a `y` with a size-1 axis.
- The tests that compare the engine with the reference pass the reference the two weights and the engine, still on
  one weight, `weight=jets.blocks_to_weight(...)` of the block weight (or the full weight itself): the old engine then
  pins the reference's new reading.
- **Part 1 rewrites every test that hands the reference a form it now refuses** (round 2, finding 12), among them
  `tests/test_engine_data_space.py:139-158` (`TestAgainstTheReference`: a `y` constant in `W`, weights with a
  structural axis of 1): the reference gets the written-out `y` and the `(block, full)` pair, the old engine the same
  written-out `y` and `blocks_to_weight(block)` or the written-out full weight. The part's run finds the list.
- **Red at its end:** none expected. **Review:** the pins use no library code but dense functions; the engine
  comparisons pass unchanged in their numbers.

**Part 2: the engine and its steps** (`sampling_variables`, `sampling_steps`, the private metric core of `jets`).

- The engine's arguments `block_weight`, `full_weight` in `weight`'s place (§4.2); `computed_names` follows; the
  structural check of §4.2 (`y` exact but `C`; the full weight exact but `W` and `C`; the modes agreeing; the block
  weight exact and never a sequence); on the uniform layer `y` and the full weight packed with `pack_if_ragged` (not
  `_pack_data_space`) and masked always, the block weight never; the rows of `f`, `r_block_sumsq`, `w_tilde`,
  `Jp_norm_sq` and the twin `_Jp.w_tilde` read `('r' | 'Jp', 'block_weight', 'full_weight')`; `_ARGUMENTS`,
  `_UNIFORM_MASKS` and the module docstring's table and paragraphs follow.
- The steps: `_modes` yields each mode's array with its block row and its full weight; `_with_K` places `K` in the
  full weight only (the block weight sits on the leading axes, `K` after `W`); every product of a weight is made by one
  private core of `jets`, `jets._metric(r_X, block_X, full_X, gamma)` (§4.3), which the public metrics of part 3 call
  too.
- The oracle's one bridging line (§2.1).
- The engine's tests that pass a weight move to the two weights: `tests/test_engine_data_space.py` (rewritten: its
  edge-of-1 class becomes refusals), the engine rows of `test_block_sums.py`, `test_sampling_variables.py`,
  `test_sampling_stacks.py`, `test_sampling_uniform.py`, `test_sampling_corewise.py`, `test_sampling_typed.py`,
  `test_sampling_identities.py`, `test_held_variables.py`, `test_least_squares_oracle.py`'s `computed_names` rows,
  `test_dispatch.py`'s engine rows (§6 lists what each checks against).
- **Red at its end (predicted here so that the part's reviewer can tell an expected red from an unexpected one;
  the list itself is recorded by a run):**
  - every probe fit whose weight is `blocks_to_weight`'s output (an edge of 1, refused);
  - **every probe derivative fit, weighted or not**: with no weight `prepare_derivative_data` returns the `k!` table
    placed by `blocks_to_weight` (`jets.py:1174-1178`), mode-leading with an edge of 1. So `tests/_fits.derivative_fit`
    with `kind='probe'` and its callers, `old_vocabulary('probe_derivatives', ...)`, and every translated
    `probe_derivatives` old-form call in the tests (the examples and pages are run at part 5 only);
  - every test whose `y` has a size-1 `W` or structural axis and reaches the engine;
  - hand-built probe or apply full-form weights with a structural axis of 1 (round 2, finding 13; e.g.
    `tests/backend/test_optimizer_display.py:425`, a probe weight with an edge of 1);
  - not red: apply and entries fits with a `blocks_to_weight` weight (a legal full weight, today's numbers bitwise),
    and every fit with no weight but the probe derivative ones.

  The expected red is recorded **per subtest** from the `-v` log (the `SUBPASSED(...)` / `SUBFAIL` labels), each
  failing subtest classified by its label, so that a red node's other cases stay visible. A test that stops at its
  first failure hides its other cases: **part 4's green run is the first check of part 2's engine on the cases hidden
  behind a red node**, and part 4's review says so. **Review:** the engine against the reference (part 1), uniform against ragged with
  finite junk in the padding of `y` and the full weight, the named laws (§6), the battery's engine rows bitwise (§7).

**Part 3: the public surface of `jets`** (additive, but for the dual-jet pair).

- Added: `probe_data_metric`, `apply_data_metric`, `entries_data_metric`; `probe_jet_to_dual_jet`,
  `apply_jet_to_dual_jet`, `probe_dual_jet_to_jet`, `apply_dual_jet_to_jet`; `derivative_block_weight_to_taylor`;
  `prepare_probe_derivative_data`, `prepare_apply_derivative_data` (§4.4, §4.5). Each probe helper renamed here takes
  the ragged form only and refuses a single array, naming `unpack_vectors` (§3.3 of the plan); `probe_data_metric`
  takes both forms.
- Removed: `jet_to_dual_jet`, `dual_jet_to_jet` (no library caller). Kept until their last caller goes (part 4 or
  5): `data_metric`, `blocks_to_weight`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data`,
  `check_weight`.
- `check_data` learns the exact shape of `y` (only `C` may be 1).
- The module docstring of `jets` (its walk-through of derivative data, `:67-152`, moves to the new functions).
- Tests: `tests/test_jets.py` (the derivative-data and converter classes rewritten; the rejection table of
  `check_data` re-ruled), `tests/test_data_metric.py` (rewritten for the three metrics), `tests/test_type_laws.py`,
  the `jets` rows of `test_dispatch.py` and `test_jax_absent.py`.
- **Red at its end:** part 2's list (the probe derivative fits included), plus the batch checks of tests whose `y`
  has a size-1 `W` (now refused by `check_data`). **Review:** each metric against an explicit loop over every index (no broadcasting in the oracle),
  each refusal of §4.4's table, the precision rule, the converters' laws, the derivative functions against
  factorials written out.

**Part 4: the fitting layer, and the translation that builds its batches.**

- `LeastSquaresBatch`'s fields `block_weight`, `full_weight` in `weight`'s place; `data_metric(r)` calls the metric of
  its kind; `check` adds the exact block shape and the full weight's rule (§4.7); `take` passes the block weight
  through and gathers the full weight like `y` (or keeps it when its `W` axes are 1); the oracle's call of the engine
  with both weights (the bridge goes); the oracle's two data sums as one product and one reduction on a packed `y`
  (masked by selection) and one per mode on a ragged one (§4.8); `_arrays_of`, `_y_times_metric_y`, `_y_metric_y`,
  `_y_block_metric_y` replaced by one helper; `pack_batch` packs a probe `y` and full weight with `pack_vectors` and
  refuses members whose mode sizes disagree. Deleted: `pack_data`, `ut3_operations._pack_data_space`,
  `jets.check_weight`.
- `t3toolbox/optimizers.py`'s translation (plan §4.4, K1's two rows): `_old_fit` passes the old `omega` as
  `block_weight` (plain probe `(d,)`; a derivative kind's `(d, n+1)` or `(n+1,)`) and converts derivative data with
  `prepare_{probe,apply}_derivative_data` (they exist from part 3). It is the one library builder of batches outside
  the fitting module, so it moves with the batch's fields; otherwise every old-form call (68 in the tests, 9 in the
  examples, 21 in the pages) would be red from part 4 to part 5 with nothing to learn from it.
- `tests/_fits.py`: `fit(..., block_weight=None, full_weight=None)`, `derivative_fit(..., block_weight=None)` (through
  `prepare_{probe,apply}_derivative_data`), `shared_over` and `at_element` reshape `y` and the full weight only (a
  block weight has no `C`), `uniform` calls `pack_batch(batch, ux.N)`; `old_vocabulary` keeps its 2026.5.0 weight
  and maps it to the block weight as the translation does (written separately, so the two stay independent).
- Tests: `test_least_squares_oracle.py`, `test_oracle_data_space.py` (rewritten), `test_block_sums.py` (the oracle
  half), `backend/test_uniform_fitting.py`, `backend/test_curvature.py`, `backend/test_uniform_curvature.py`,
  `backend/test_optimizers.py`, `backend/test_padded_frame.py`, `backend/test_zero_reference.py`,
  `backend/test_optimizer_display.py`, `test_sharing.py`, `test_optimizers_frontend.py` (its weighted new-form fits;
  `TestTheTranslation`'s 19 pins unchanged), `test_fitting.py`, the sweeps (`tests/oracle_sweeps/`).
- **Red at its end:** none expected (the frontend's re-exports still name the old functions, which still exist).
  **Review:** the weights audit (§7.4 (c): every `blocks_to_weight` call paired with its `block_weight=`, every
  `full_weight=` with its reason), the named oracles at the oracle (§6.3), the battery's oracle rows and the translated
  runs bitwise, `take` against index loops, `pack_batch` against loops with junk in the padding, the translation's
  pins.

**Part 5: the public names, the deletions, the pages.**

- `t3toolbox/fitting.py`: the re-exports and `__all__` under the new names (D4, §11); its module docstring's list.
- Deleted with their last callers: `jets.blocks_to_weight`, `data_metric`, `derivative_weight_to_taylor_weight`,
  `prepare_derivative_data`; `tests/oracle_sweeps/oracle.py`'s old full-form functions if no caller is left
  (`k1_sites.py` decides).
- The one-line references in the six docstring-only modules (§5.3: "pass `data_metric(r, weight,
  minibatch_scale)`" becomes the kind's metric or `batch.data_metric(r)`).
- Tests: the frontend's names (`test_jets.TestFrontendReach`), `test_dispatch.py`'s and `test_jax_absent.py`'s rows of
  the deleted functions, `test_manifold.py`, `test_aligned_padded_frame_frontend.py`; every remaining test of a
  deleted function goes with it (its subject is gone; where it was another test's oracle, that test was re-based in
  parts 1 to 4).
- Pages with doctests that pass weights: `docs/fitting_and_optimization.md:426-435`, `docs/numerical_contracts.md:92-94`
  (their output pasted from a run). The examples: nothing beyond part 0 (their weighted calls are old-form and
  translated). Narrative pages stay P19's (§9).
- **Red at its end:** none. The full gate, the slow tier serially, the battery `old` against `new`, the tally, the diff
  audit, the scan with no call of a removed name left.

### 2.3 What each review can check on its own

| Part | its review needs | the evidence it is given |
|---|---|---|
| 0 | the diff, the battery `old` vs `part0` | bitwise battery; diff audit; tally; suite green |
| 1 | the reference's diff, the pins | the pins against `np.ndindex` loops; the engine comparisons unchanged |
| 2 | parts 0 and 1 | engine vs the reference; uniform vs ragged with junk; laws; battery engine rows; expected-red list |
| 3 | part 2 (the private core) | each function against loops; refusals; converter laws; expected-red list |
| 4 | parts 2 and 3 | named oracles at the oracle; battery oracle rows and translated runs; `take` and `pack_batch` against loops; the translation's pins |
| 5 | everything | the full gate; battery `old` vs `new`; the slow tier; the scan |

---

## 3. The mathematics, checked before it is designed around

`k1/check_math.py`, plain numpy loops; every check passes on today's tree (run 2026-11-25). The library is imported
only by checks 5 and 6, which compare with today's functions.

1. **`M r = gamma * block_weight**2 * full_weight**2 * r`, entry by entry** (check 1). Probe, `d = 2`, `F = (3,)`,
   `W = (4,)`, `K = (3,)`, `C = (2,)`, `N_i = 5, 6`, a block weight `(d,)+F` and a full weight with `W` of 1, `K`
   placed (1) and `C` full: the metric equals, at every index, `gamma * b[i][k]**2 * f[i][idx with size-1 axes at 0]**2
   * r[i][idx]`, `k` the leading `F` multi-index of `idx`.
2. **"Left-aligned"** means: the block weight's axes are the leading axes of what it weighs. Against a tuple `r`, row
   `i` (shape `F`) is reshaped to `F + (1,) * (r[i].ndim - m)` and multiplies `r[i]`; against a packed `r`,
   `(d,)+F+W+[K]+C+(N,)`, the whole table `(d,)+F` is reshaped to `(d,)+F+(1,)*(r.ndim - 1 - m)`; against an apply `r`,
   `F+W+[K]+C`, the table `F` is reshaped to `F + (1,)*(r.ndim - m)`; entries, `()`, is a number. `K` sits after `W`,
   behind the leading axes, so a stack of directions needs no placement for the block weight (it does for the full
   weight: `_with_K`, which stays). Check 2: on the packed `r` with `K` the result is the tuple's mode by mode,
   **bitwise**, and the padding stays zero.
3. **The `k!` of derivative coordinates goes into the block weight only, once** (check 3, `order = (2, 1)`, a block
   weight and a full weight both chosen for derivatives, the full weight per sample and entry). A derivative residual
   is `k! r_t` slot by slot, so `(b_d f_d r_d)^2 = ((k! b_d) f_d r_t)^2`: the Taylor weights are `(k! b_d, f_d)`, equal
   misfits to 1e-13 (`2.483085e+02` both). With no weight the Taylor block weight is the `k!` table itself, `(d,)+F`
   for probe, `F` for apply (checked). This is the plan's §2.2 and §12.1; the caution for the docs stands: a user who
   computes a relative full weight from Taylor data and also passes the returned `k!` table has both.
4. **`<y, M y>` and the block sums when `y`'s `C` axes are 1** (check 4, `C = (3,)`, `y` and the full weight with `C`
   of 1, the full weight `W` of 1 too). The product `y * M y` is broadcast to `z`'s shape `F+W+C+(N_i,)` first, then
   reduced: per element of `C` it equals that element's own fit (`y` and the weights read at 0), and the block sums
   count each element of `C` once, `|C|` times one copy's. Reduced without the broadcast the block sums are `|C|` times
   too small (checked: the trap the broadcast avoids). `y` is exact on `W` now, so the only size-1 axes of `y` are `C`
   axes; a full weight's `W` axes of 1 are covered by the same broadcast.
5. **The battery's arithmetic** (check 5, against today's `jets.data_metric`): with only a block weight, `(b*b) * r`
   (the row left-aligned) is **bitwise** today's metric of `blocks_to_weight(b)`; with only a full weight, `(f*f) * r` is
   bitwise today's metric of the same full weight. With both, `b^2 f^2 r` against `(b f)^2 r` (an old weight that was the
   product) differs by rounding only (max relative 4.5e-16). So: every old single weight maps to one new weight with
   bitwise-equal metric values, and only a test that passes both weights has no bitwise counterpart on the old tree.
6. **The derivative data** (check 6): today's `prepare_derivative_data` weight is the new block weight reshaped,
   bitwise: with no weight, `k!` per mode and slot; with blocks `b_d`, `k! * b_d`.

Two consequences for the design:

- **Where the product is formed.** To keep check 5's bitwise identity, the one private core (§4.3) forms `b*b` and
  `f*f` each in the lowest floating precision of `r` and the weights (today's `_omega_squared` rule), multiplies them
  (`(b*b) * (f*f)` only when both are given), then `* r`, then the scale `gamma * (...)` last, as today.
- **The oracle's two data sums.** On a ragged `y` they stay one product and one reduction per mode (bitwise today's,
  `y` being exact now and the broadcast to `z`'s shape kept). On a packed `y` they become one product and one
  reduction on the packed array (plan §2.4, §6): the summation order changes (one reduction over `(d,)+F+W+C+(N,)`
  with zero padding against `d` reductions over unpadded modes), so `f_resolution` and `y_block_sumsq` on the uniform
  layer agree with today's to rounding, not bitwise (§7).

---

## 4. Every signature that changes, and the structural rule of each

The "before" of every header is verbatim in `k1/signatures_before_k1.md` (by `signatures.py`); the main ones are
repeated here. The "after" is in the house form (`docs/contributor/signature_style.md`). **The rule of §2.2 of the
plan, stated once and used in every row below:** a size of 1 is legal on the stack axes `W` (the samples) and `C` (the
base points) only, never on the mode, the jet slots `F` or the edge; `y` is exact on `W` (only its `C` axes may be 1);
a full weight's `W` and `C` axes are each 1 or full; never `K` in a weight or in `y` (the engine places `K`); the block
weight is exact. "Agree" (probe): the `d` arrays of `y` have one pattern of size-1 `C` axes, the `d` arrays of a full
weight one pattern of size-1 `W` and `C` axes (they may differ only in the edge). The refusals are `ValueError`s
(`TypeError` for a wrong type), each naming the object, the mode and the expected shape; the probe-mode refusals of the
agreement name `np.broadcast_to` as the fix.

### 4.1 The dense reference (`backend/sampling_definitions.py`, part 1)

Every one of the 37 public functions (`w` ... `Jp_norm_sq`) has, in its keyword-only group,

```python
        y:                typ.Optional[typ.Any] = None,      # the data, of the type of z
        weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
```

and after K1

```python
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries (); None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
```

The reference has no `C` and no `K` (its docstring: apply element by element), so its rule is: `y` exact; the full
weight's `F` and edge exact, each axis between them 1 or `W`'s size; the block weight exact. Its reading stays its own
(`_modes_of_data`, `_check_data_space`), with no import of `jets`: `_weight_squared(S, block_weight, full_weight, i,
shape)` returns `(row_i**2 reshaped left-aligned) * full_i**2` (either factor `1.0` when `None`). The block weight is
an array (a Python number for entries), never a tuple or a list (a `TypeError`, as in the engine, §4.2).

### 4.2 The engine (`backend/sampling_variables.py`, part 2)

```python
def sampling_variables(
        kind:              str,                              # 'probe' | 'apply' | 'entries'
        frame:             typ.Tuple,                        # the base point (D, U, L, R), four tuples of d arrays; or
                                                             #   the point (B, G): X
        sample:            typ.Any,                          # ww: P_in_i⊗V_i, the input jets, len=d,
                                                             #   elm_shape=F_in_i+W+(Ni,);
                                                             #   entries: index^d, shape=(d,)+W, int
        want:              typ.Sequence[str],                # variable names; the return is aligned with them

        order:             typ.Tuple[int, ...] = (),         # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        y:                 typ.Optional[typ.Any] = None,     # the data: Y*, z's shape, only its C axes may be 1:
                                                             #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        block_weight:      typ.Optional[NDArray] = None,     # one number per block, exact: probe (d,)+F; apply F;
                                                             #   entries (); None = ones
        full_weight:       typ.Optional[typ.Any] = None,     # one per measured number: z's shape, its W and C axes 1 or
                                                             #   full, never K; None = ones
        variations:        typ.Optional[typ.Tuple] = None,   # X, (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        core_directions:   typ.Optional[typ.Tuple] = None,   # X, (D_dot, L_dot), the shapes of D, L with K+C
        Vc:                typ.Optional[typ.Sequence[NDArray]] = None,   # the frame's Tucker centers: len=d,
                                                                         #   elm_shape=C+(nUi,Ni)
        sum_over_samples:  bool = True,                      # on the blocks; False: W joins their stack, W+K+C
        held:              typ.Optional[HeldVariables] = None,   # held variables; shapes checked, values trusted
) -> typ.Tuple[typ.Any, ...]:                                # aligned with want; types and shapes in the module's table
```

`computed_names` takes exactly the same arguments: `weight: typ.Optional[NDArray] = None,  # not read` becomes

```python
        block_weight:      typ.Optional[NDArray] = None,     # not read
        full_weight:       typ.Optional[typ.Any] = None,     # not read
```

Every call of the engine and of `computed_names` in the library and the tests passes exactly 4 positional arguments
(250 calls, `k1_sites.py`), so inserting a keyword after `y` moves no positional argument.

**The structural check** (`_check_data_space(setting, name, value, edges)` becomes `_check_y`, `_check_full_weight`,
`_check_block_weight`, sharing one private reader; static, so it runs under `jit`). With `body = F + W + C`:

| object | apply | entries | probe, each mode `i` | probe, across the modes |
|---|---|---|---|---|
| `y` | rank `len(F+W+C)`; `F`, `W` exact; each `C` axis 1 or `C`'s | `W` exact; `C` 1 or full | rank `len(body)+1`; `F`, `W`, the edge exact; `C` 1 or full | exactly `d` arrays; one `C` pattern |
| `full_weight` | `F` exact; each `W`, `C` axis 1 or full | each `W`, `C` axis 1 or full | `F`, the edge exact; each `W`, `C` axis 1 or full | exactly `d`; one `W` and `C` pattern |
| `block_weight` | `np.shape == F` | `np.shape == ()` | — | `np.shape == (d,) + F` |

- The edge is `N_i` for a ragged object, the padded `N` for a packed one (no edge of 1 any more).
- A block weight that is a tuple or a list raises a `TypeError` ("one array of shape (d,)+F, the same on both layers";
  a tuple would read as one more structure, and `_cast` would treat it as a sequence of arrays).
- The forms are today's (D1): on a ragged frame a probe `y` or full weight is a tuple or a list of `d` arrays, or one
  mode-leading array whose every `N_i` is that array's edge; on a uniform frame a tuple or a list (checked against
  `N_i`, then packed by `ut3_operations.pack_if_ragged(value, N)`) or one packed array (checked against `N`).
  `_pack_data_space` is no longer called (part 4 deletes it).
- **Uniform masking** (today's `:1910-1920`): `y` and the full weight are always masked (they always have the edge
  `N`; the "edge of 1" branch, `:1914-1916`, goes); the block weight is never masked (it has no edge). The mechanism
  stays the product (`_masked`) until K5 (selection is K5's, plan §3.2), so a full weight's padding must be finite in
  K1, as today.
- **The rows:** `f`, `r_block_sumsq`, `w_tilde` read `('r', 'block_weight', 'full_weight')`, `Jp_norm_sq` and the
  twin `_Jp.w_tilde` `('Jp', 'block_weight', 'full_weight')`; `_ARGUMENTS` gains both names and loses `'weight'`;
  `_UNIFORM_MASKS['full_weight'] = (('N', None),)` in `'weight'`'s place, no entry for `'block_weight'`.
- **The dtype:** both weights are counted inputs where a row reads them, as `weight` was (a float64 block weight with
  float32 cores gives float32, as today).

### 4.3 The steps (`backend/sampling_steps.py`, part 2) and the private core (`backend/jets.py`)

```python
def _metric(                                # in jets.py: the one place b**2 f**2 is formed
        r_X:      NDArray,                  # one array of r: a mode F+W+[K]+C+(Ni,); packed (d,)+F+W+[K]+C+(N,);
                                            #   apply F+W+[K]+C; entries W+[K]+C
        block_X:  typ.Optional[typ.Any],    # the block weight on r_X's leading axes: F | (d,)+F | (); None = ones
        full_X:   typ.Optional[typ.Any],    # the full weight: r_X's rank, each axis 1 or r_X's size; None = ones
        gamma:    typ.Optional[typ.Any],    # a number or a 0-d array; None = 1
        xnp,
) -> NDArray:                               # gamma * (block_X**2 * full_X**2) * r_X, r_X's shape; no checks
```

It replaces `_omega_squared` and the inner loop of `data_metric`; `_check_pair` stays the full weight's pairwise check.
The steps keep their per-mode structure in K1 (the single array operations are K5's, plan §6), with two weights:

```python
def _modes(
        kind:          str,                 # 'probe' | 'apply' | 'entries'
        value:         typ.Any,             # of the type of z: probe len=d (or packed (d,)+...), elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C; entries W+[K]+C
        block_weight:  typ.Any,             # probe (d,)+F; apply F; entries (); None = ones
        full_weight:   typ.Any,             # of the type of z with [K] placed (_with_K); None = ones
        xnp,
) -> typ.List[typ.Tuple[NDArray, typ.Any, typ.Any, int]]:   # (array, its block row, its full weight's array, edge axes)

def _with_K(
        full_weight:  typ.Any,              # of the type of z with no K: probe len=d or (d,)+F+W+C+(N,); else F+W+C or W+C
        kind:         str,
        lens:         contractions.GroupLengths,
) -> typ.Any:                               # the full weight with lens.K size-1 axes after F + W

def _weighted_sum(
        kind:          str,
        value:         typ.Any,             # of the type of z (r, or Jp with K)
        block_weight:  typ.Any,             # (d,)+F | F | (): on the leading axes, no placement; None = ones
        full_weight:   typ.Any,             # placed against value (with K where value has it); None = ones
        n_sum:         int,                 # the leading axes summed: the jet axes and W
        xnp,
) -> NDArray:                               # sum of value * M value at gamma = 1 over F, W and the edge
```

`f_step`, `r_block_sumsq_step`, `w_tilde_step` take `(r, block_weight, full_weight, *, lens, order, form, kind)`,
`Jp_norm_sq_step` takes `(Jp, block_weight, full_weight, *, lens, order, form, kind)`; their shape comments are today's
with `weight` replaced by the two rows

```python
        block_weight:  typ.Any,             # one per block: probe (d,)+F, apply F, entries (); None = ones
        full_weight:   typ.Any,             # of the type of z, no K (placed here); None = ones
```

`r_step` is unchanged in code; its comment says `y` is `z`'s shape but for `C` axes of 1.

### 4.4 The public metrics (`backend/jets.py`, part 3)

Before (`jets.py:579`):

```python
def data_metric(
        r:                typ.Any,                  # Y*, of the type of z, no K: probe len=d (or packed (d,)+...), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:           typ.Optional[typ.Any],    # omega, of the type of z: each axis 1 or r's size; None = ones
        minibatch_scale:  typ.Optional[typ.Any],    # gamma: a number or a 0-d array (traced under jit); None = 1
) -> typ.Any:                                       # M r: Y, gamma * omega**2 * r elementwise, r's structure (a tuple for a tuple or a list)
```

After (the plan's §2.3, `PerSite` written `typ.Any` until K2 adds the `TypeVar`):

```python
def probe_data_metric(
        order:            typ.Tuple[int, ...],       # the jet shape; F = tuple(n+1 for n in order)
        r:                typ.Any,                   # Y*: len=d, elm_shape=F+W+[K]+C+(Ni,); or packed (d,)+F+W+[K]+C+(N,)
        block_weight:     typ.Optional[NDArray],     # shape=(d,)+F exactly; None = ones
        full_weight:      typ.Optional[typ.Any],     # in r's form: r's rank, the mode, F and the edge exact, each
                                                     #   other axis 1 or r's size; None = ones
        minibatch_scale:  typ.Optional[typ.Any],     # gamma: a number or a 0-d array; None = 1
) -> typ.Any:                                        # M r: Y, in r's form

def apply_data_metric(
        order:            typ.Tuple[int, ...],       # the jet shape; F = tuple(n+1 for n in order)
        r:                NDArray,                   # Y*: F+W+[K]+C
        block_weight:     typ.Optional[NDArray],     # shape=F exactly; None = ones
        full_weight:      typ.Optional[NDArray],     # r's rank, F exact, each other axis 1 or r's size; None = ones
        minibatch_scale:  typ.Optional[typ.Any],     # gamma: a number or a 0-d array; None = 1
) -> NDArray:                                        # M r: Y, r's shape

def entries_data_metric(
        r:                NDArray,                   # Y*: W+[K]+C
        block_weight:     typ.Optional[typ.Any],     # shape=() exactly (a number); None = 1
        full_weight:      typ.Optional[NDArray],     # r's rank, each axis 1 or r's size; None = ones
        minibatch_scale:  typ.Optional[typ.Any],     # gamma: a number or a 0-d array; None = 1
) -> NDArray:                                        # M r: Y, r's shape
```

`entries_data_metric(r, b, f, g)` is `apply_data_metric((), r, b, f, g)` with its own name in the messages. Every
argument is required (`None` written out), as in the M arc. What each checks (structural, under `jit` too):

| | `probe_data_metric` | `apply_data_metric` | `entries_data_metric` |
|---|---|---|---|
| `order` | `contractions._read_order` | likewise | none |
| `r` | a tuple or a list of `d >= 1` arrays, each of rank `>= m + 1` with leading axes `F` exactly; or one array of rank `>= m + 2`, `d = shape[0]`, then `F` exactly | one array, rank `>= m`, leading axes `F` exactly | one array |
| `block_weight` | `np.shape == (d,) + F`, not a tuple or a list | `np.shape == F` | `np.shape == ()` |
| `full_weight` | `r`'s form (a tuple `r` takes a tuple or a list of `d`, a packed `r` one array: a mix raises); per array `r`'s rank, the mode (packed: axis 0) exact, `F` exact, the edge (last) exact, each axis between 1 or `r`'s size | `r`'s rank, `F` exact, each other axis 1 or `r`'s size | `r`'s rank, each axis 1 or `r`'s size |
| `minibatch_scale` | 0-d (a scalar or a 0-d array) | likewise | likewise |

- `r`'s own `F` is checked whatever the weights: the left-aligned block weight would otherwise broadcast a size-1
  leading axis of `r` into a larger result with no error (a shape-1 `r` slot against `F = (3,)`).
- The axes between `F` and the edge are `W`, `K`, `C`, of which the metric is not told which is which, so each may be 1
  or `r`'s size; this admits the size-1 `K` that `_with_K` inserts. Only the batch knows `y` is exact on `W` and that a
  batch has no `K` (§4.7).
- This closes today's limit (`jets.py:678-690`, a packed weight with a mode axis of 1 broadcast over the modes): the
  packed full weight's axis 0 is exact now.
- Values in a packed `r`'s padding come back multiplied by the weights: masking is the engine's and the oracle's
  business, as today (the precondition "zero padding in" of plan §1.3 for a function with no masks).

### 4.5 The dual-jet pair and the derivative data (`backend/jets.py`, part 3)

The four converters (no entries version: the entries kind has no jets; its dual jet is `entries_data_metric` at
`gamma = 1`, there being nothing to turn):

```python
def probe_jet_to_dual_jet(
        order:         typ.Tuple[int, ...],       # the jet shape; F = tuple(n+1 for n in order)
        jet:           typ.Sequence[NDArray],     # Y*: len=d, elm_shape=F+W+[K]+C+(Ni,)
        block_weight:  typ.Optional[NDArray],     # shape=(d,)+F exactly; None = ones
        full_weight:   typ.Optional[typ.Sequence[NDArray]],   # len=d, each as probe_data_metric's; None = ones
) -> typ.Tuple[NDArray, ...]:                     # Y, turned: probe_data_metric(order, jet, b, f, None), every jet
                                                  #   axis flipped; len=d

def apply_jet_to_dual_jet(
        order:         typ.Tuple[int, ...],       # the jet shape
        jet:           NDArray,                   # Y*: F+W+[K]+C
        block_weight:  typ.Optional[NDArray],     # shape=F exactly; None = ones
        full_weight:   typ.Optional[NDArray],     # as apply_data_metric's; None = ones
) -> NDArray:                                     # Y, turned

def probe_dual_jet_to_jet(
        order:         typ.Tuple[int, ...],
        dual_jet:      typ.Sequence[NDArray],     # Y, turned: len=d, elm_shape=F+W+[K]+C+(Ni,)
        block_weight:  typ.Optional[NDArray],     # shape=(d,)+F exactly; None = ones
        full_weight:   typ.Optional[typ.Sequence[NDArray]],   # len=d; None = ones
) -> typ.Tuple[NDArray, ...]:                     # Y*: every jet axis flipped back, divided by b**2 f**2; len=d

def apply_dual_jet_to_jet(
        order:         typ.Tuple[int, ...],
        dual_jet:      NDArray,                   # Y, turned: F+W+[K]+C
        block_weight:  typ.Optional[NDArray],     # shape=F exactly; None = ones
        full_weight:   typ.Optional[NDArray],     # None = ones
) -> NDArray:                                     # Y*
```

- The probe pair takes the ragged form only (a tuple or a list of `d` arrays) and refuses a single array with a
  `TypeError` naming `ut3_operations.unpack_vectors` (`backend/packing.unpack_vectors` from K2); a full weight in
  another form than the jet's raises. The divisor of the `dual_jet_to_jet` pair is `b*b * f*f` formed by `_metric`'s
  rule; a zero weight divides by zero silently, as today.
- With a stack of directions `K` in the jet, a block weight needs nothing (left-aligned) and a full weight needs its
  size-1 `K` axes (today a jet with `K` and any weight raises by rank; the new rule admits it, as the public metric
  does).

The derivative data:

```python
def derivative_block_weight_to_taylor(
        order:         typ.Tuple[int, ...],       # the jet shape; F = tuple(n+1 for n in order)
        block_weight:  NDArray,                   # a block weight chosen for derivative residuals: (d,)+F or F
) -> NDArray:                                     # the Taylor block weight: the trailing F axes times k_1! ... k_m!

def prepare_probe_derivative_data(
        order:         typ.Tuple[int, ...],                    # the jet shape; F = tuple(n+1 for n in order)
        y:             typ.Sequence[NDArray],                  # Y* as derivatives: len=d, elm_shape=F+W+C+(Ni,)
        block_weight:  typ.Optional[NDArray] = None,           # chosen for derivatives: (d,)+F; None = ones on them
        validation:    typ.Optional[typ.Sequence[NDArray]] = None,   # Y* as derivatives, y's form; or None
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                               # y: Y* in Taylor coordinates, len=d, same shapes
        NDArray,                                               # the Taylor block weight, (d,)+F: k! b, or the k! table
        typ.Optional[typ.Tuple[NDArray, ...]],                 # validation in Taylor coordinates, or None
]:

def prepare_apply_derivative_data(
        order:         typ.Tuple[int, ...],
        y:             NDArray,                                # Y* as derivatives: F+W+C
        block_weight:  typ.Optional[NDArray] = None,           # chosen for derivatives: F; None = ones on them
        validation:    typ.Optional[NDArray] = None,           # Y* as derivatives, y's form; or None
) -> typ.Tuple[
        NDArray,                                               # y in Taylor coordinates
        NDArray,                                               # the Taylor block weight, F
        typ.Optional[NDArray],                                 # validation in Taylor coordinates, or None
]:
```

- `derivative_block_weight_to_taylor`: `block_weight.ndim` is `m` or `m + 1` and its trailing `m` axes are `F`
  exactly, else a `ValueError` (with no jets, `m = 0`, it returns the table unchanged, cast). One function for both
  kinds (plan §12.2): the `k!` multiplies the trailing `F` axes either way.
- `prepare_*`: `y`'s (and `validation`'s) leading axes are `F` exactly; the block weight's shape exact; the probe one
  takes the ragged form only (a single array refused, naming `unpack_vectors`). With `block_weight=None` the returned
  block weight is the `k!` table, broadcast to `(d,)+F` for probe, in `y`'s floating dtype (today's rule). There is no
  full weight argument (plan §12.1): a full weight is not converted.

### 4.6 `LeastSquaresBatch` and `LeastSquaresOracle` (`backend/fitting.py`: part 0, then part 4)

Before: `class LeastSquaresBatch` with the fields `sample`, `y`, `weight`, `minibatch_scale`, `drawn_indices` (verbatim
in `signatures_before_k1.md`); `class LeastSquaresOracle(ValueHashedFields)` with `kind`, `order`, `__post_init__`,
`query`, `check_batch`, `_w_axes`, `_W`, `n_measurements(batch)`, `take(batch, indices)`.

After part 4 (part 0 has the same with one field `weight` in place of the two weights):

```python
@dataclass(frozen=True, eq=False, init=False)
class LeastSquaresBatch:
    kind:             str                                        # 'probe' | 'apply' | 'entries'; static (aux_data)
    order:            typ.Tuple[int, ...]                        # the jet shape, Python ints; entries (); static
    sample:           typ.Any                                    # P_in_i⊗V_i (the input jets ww) | index^d (entries)
    y:                typ.Any = None                             # Y*: probe len=d, F+W+C+(Ni,); apply F+W+C; entries W+C
    block_weight:     typ.Optional[NDArray] = None               # one per block: probe (d,)+F, apply F, entries (); None = ones
    full_weight:      typ.Any = None                             # z's shape, W and C axes 1 or full; None = ones
    minibatch_scale:  typ.Optional[typ.Any] = None               # gamma: n / batch_size on a drawn batch; None = 1
    drawn_indices:    typ.Optional[NDArray] = None               # int, (batch_size,): the draw, flat over W

    def __init__(
            self,
            kind:             str,                               # 'probe' | 'apply' | 'entries'
            order:            typ.Tuple[int, ...],               # a tuple of nonnegative ints; entries: ()
            sample:           typ.Any,
            y:                typ.Any = None,
            *,
            block_weight:     typ.Optional[NDArray] = None,
            full_weight:      typ.Any = None,
            minibatch_scale:  typ.Optional[typ.Any] = None,
            drawn_indices:    typ.Optional[NDArray] = None,
    ) -> None:

    def check(
            self,
            shape:        typ.Sequence[int],                     # the mode sizes (N0, ..., N(d-1))
            stack_shape:  typ.Tuple[int, ...] = (),              # C, the model's stack of base points; () unstacked
    ) -> None:

    def take(
            self,
            indices:  NDArray,                                   # int, (b,): flat positions over W; len(indices) static
    ) -> 'LeastSquaresBatch':                                    # the drawn batch, W flattened to (b,)

    @property
    def n_measurements(self) -> int:                             # |W|, the number of samples

    def data_metric(
            self,
            r:  typ.Any,                                         # Y*, of the type of z (no K), on this batch's samples
    ) -> typ.Any:                                                # M r: Y, in r's form


@dataclass(frozen=True, eq=False)
class LeastSquaresOracle(ValueHashedFields):
    optional_outputs: typ.ClassVar[typ.Tuple[str, ...]] = (
        'Kp', 'Jp', 'r', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution')

    def query(...)                                               # unchanged signature; reads batch.kind, batch.order
```

- **The constructor's checks** (today's `LeastSquaresOracle.__post_init__`, `backend/fitting.py:367-376`, O-44, O-46),
  on the static fields only, so they are safe when jax rebuilds the object: the kind one of the three; the order read
  by `contractions._read_order` (a tuple of nonnegative integers, never a `bool`; normalized to Python ints, the form
  that keys jit caches); the entries kind's order `()`. Nothing about the arrays: a batch is checked against a model
  by `check` (structural, called by the frontend where a batch enters), and the engine checks what it reads on every
  query. The kind is checked to be a `str` first (a `TypeError`; a packed array compared with a string by `in` would
  compare elementwise), then one of the three (a `ValueError`). So every old positional form raises at once. The
  plan's "fails on its kind" (§2.4) holds for three or more positional arguments (`LeastSquaresBatch(ww, y, w)` and
  longer: a `TypeError` on the kind); the two-argument `LeastSquaresBatch(ww, y)` fails on Python's own "missing 1
  required positional argument: 'sample'". No sentinel default is given to `sample` to change that message (the
  orchestrator's ruling, round 1: the signature stays true, and the old form exists only on this branch); the class
  docstring's example shows the message (measured on a model of the class, `k1/kwonly_check.py`, Python 3.11 and the 3.9 floor environment with
  jax 0.4.30: `replace`, `fields`, the frozen guard, the pytree round trip and `jax.jit` over it all work).
- **Keyword-only after `y`** (D2): `dataclass(kw_only=True)` and `KW_ONLY` are Python 3.10; the floor is 3.9. A
  hand-written `__init__` (with `object.__setattr__`, the frozen idiom) on a dataclass with `init=False` gives it.
  `dataclasses.replace` still works (it calls the class with every init field by keyword), and so do `fields`,
  `repr` and the frozen guard. The pytree unflatten builds by keyword:
  `lambda aux, children: LeastSquaresBatch(*aux, **dict(zip(_BATCH_LEAVES, children)))`, `aux = (kind, order)`,
  `_BATCH_LEAVES = ('sample', 'y', 'block_weight', 'full_weight', 'minibatch_scale', 'drawn_indices')`.
- **`n_measurements` is a property** (plan §14.7, both tiers): `math.prod(W)`, `W` read off the sample.
- **`data_metric(r)`**: `{probe,apply}_data_metric(self.order, r, self.block_weight, self.full_weight,
  self.minibatch_scale)`, `entries_data_metric(r, ...)` for entries (part 4; part 0 keeps `jets.data_metric(r,
  self.weight, self.minibatch_scale)`).
- `LeastSquaresOracle()` has no fields: every instance equal and of one hash (it rides as `aux_data`; one compiled
  program per kind and order, the kind and order now being the batch's aux data). A subclass is another class
  (`test_dispatch.py:2178-2184` keeps passing: `HalfOracle() != LeastSquaresOracle()`).

### 4.7 The batch's `check` (part 0, then part 4)

Part 0: today's `check_batch` body, the kind and the order the batch's own. Part 4:

- the sample: `jets.check_sample(kind, order, sample, shape)` (until K3);
- `y`: `jets.check_data(kind, order, sample, y, stack_shape)`, which from part 3 checks `y` exact on `F`, `W`, the
  edge and the mode count, each `C` axis 1 or full, and (probe) one `C` pattern across the modes;
- the full weight: the same reader with `W` added to the axes that may be 1 (a private helper of `jets`,
  `_check_of_type_of_z(..., size_one=('W', 'C'))`), and one `W` and `C` pattern across the modes;
- the block weight: `np.shape == (len(shape),) + F` (probe), `F` (apply), `()` (entries); not a tuple or a list.
- what only the batch can know beyond the metric: `y` exact on `W`; no `K` anywhere (the ranks are exact, so a `K`
  axis fails by rank).
- O2: K2's `check_probe_data(value, sample, order, stack_shape=())` must tell `y` from a full weight; until then the
  batch uses the private helper (O3: no public full-weight checker between K1 and K3).

### 4.8 The oracle's two data sums (part 4)

`_arrays_of`, `_y_times_metric_y`, `_y_metric_y`, `_y_block_metric_y` (`backend/fitting.py:150-246`) become one
private helper and two reductions:

```python
def _y_metric_y_terms(
        batch:   LeastSquaresBatch,         # its y (not None), block_weight, full_weight; its kind and order
        frame:   typ.Tuple,                 # the frame: its mode sizes, and on the uniform layer N and the shape mask
        W:       typ.Tuple[int, ...],       # the sample stack, read off the sample
        C:       typ.Tuple[int, ...],       # the stack of base points (frame_stack_shape's, today _stack_of_frame)
) -> typ.Any:                               # y * M y at gamma = 1 broadcast to z's shape: per mode (a ragged y), or one
                                            #   packed array (d,)+F+W+C+(N,) with zero padding, or one array (apply,
                                            #   entries)
```

- A packed probe `y` (one array, a uniform frame): `y` and the full weight are masked **by selection** first,
  `xnp.where(mask, y, xnp.zeros((), y.dtype))` with the host prefix mask of the frame's shape on the last axis (plan
  §2.4; the zero in the array's dtype); then one product `y * _metric(y, b, f, None)` broadcast to
  `(d,)+F+W+C+(N,)`; `<y, M y>` is one sum over every axis but `C`, `y_block_sumsq` one sum over `W`, `C` and the edge.
  No loop over the modes (plan §1.6).
- A ragged `y`: per mode, as today (the ragged layer loops over the modes), `y_i` broadcast to `F+W+C+(N_i,)`.
- apply and entries: one array, no mode, as today.
- Why selection here and the product in the engine until K5: the oracle's sums are new code in K1, and the plan rules
  selection for them (§2.4); the engine's masking is K5's to change, with its own comparison ("equal values, not equal
  bytes"). A `y` with `inf` in its packed padding then gives a finite `y_block_sumsq` from K1 on (tested, §6.3), while
  the engine's `f`, and so `f_resolution`, which reads it, wait for K5's selection (the engine's product mask makes
  `0 * inf = nan`; recorded, not promised until K5).

### 4.9 `take` (part 0, then part 4)

`take(indices)` returns `LeastSquaresBatch(self.kind, self.order, sample, y, block_weight=self.block_weight,
full_weight=full, minibatch_scale=scale, drawn_indices=indices)`: the sample gathered as today; `y` gathered (it is
exact on `W`); the full weight gathered like `y`, or kept with one axis of 1 in `W`'s place when every `W` axis is 1
(today's branch, `_take_W`); the block weight passed through (constant over the samples). M-arc decision 18's raise
(an object with fewer axes than `W` needs) stays, for `y` and the full weight.

### 4.10 `pack_batch` (`backend/uniform_fitting.py`: part 0, then part 4)

```python
def pack_batch(                                     # before
        kind:   str,                                # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],                # the jet shape
        batch:  'bfit.LeastSquaresBatch',           # ragged (or already packed: kept)
        N:      int,                                # the packed mode width
) -> 'bfit.LeastSquaresBatch':                      # the sample's vectors and the probe data packed to N

def pack_batch(                                     # after
        batch:  'bfit.LeastSquaresBatch',           # ragged (a packed member is kept until K5)
        N:      typ.Optional[int] = None,           # the packed mode width; None = the largest mode size of the sample
) -> 'bfit.LeastSquaresBatch':                      # the sample of probe and apply, a probe y and full weight packed to N
```

- The sample by `pack_sample(batch.kind, batch.order, batch.sample, N)` (until K5); a probe `y` and full weight by
  `ut3_operations.pack_vectors(value, N)` (one packer; their arrays agree in everything but the edge, §4.2); the
  entries index, apply's and entries' data, the block weight, the scale and the drawn indices as they are.
- **Refused** (part 4), each before any packing, a `ValueError` naming the object, the mode and the sizes:
  - a probe batch whose ragged members disagree in their mode sizes (`np.shape(w)[-1]` of the sample, of `y` and of
    the full weight, mode by mode);
  - a probe `y` or full weight whose modes disagree in their pattern of size-1 stack axes (everything but the edge),
    checked by `pack_batch` itself with the engine's message, which names `np.broadcast_to` as the fix (the plan's
    §2.2 lists `pack_batch` among the four places that check it; `pack_vectors`' own error, `ut3_operations.py:171-174`,
    names neither the object nor the fix);
  - an `N` below the largest mode size (today `pack_vectors` pads negatively and fails inside `np.pad`).
- `N=None` reads the largest mode size off a ragged sample's last axes. An entries batch has nothing to pad, so `N`
  is not read; a sample already packed keeps its `N` (and a ragged `y` beside it is packed to that `N`).

### 4.11 The optimizers, the display, the frontend (part 0)

```python
def flat_draw(                       # after (O1, ruled)
        batch:       typ.Any,       # the full batch: with the property n_measurements and take(indices)
        batch_size:  int,           # samples per minibatch
) -> typ.Callable:                  # draw(rng) -> a drawn batch

def _require_draw(
        batch:  typ.Any,
        draw:   typ.Optional[typ.Callable],   # the user's draw, or None (the default flat_draw)
        who:    str,
) -> None:
```

- `_require_draw` requires `callable(getattr(batch, 'take', None))` and `getattr(batch, 'n_measurements', None)` an
  integer (`numbers.Integral`), its message naming the batch's type and `draw=`; a callable `n_measurements` (a user
  batch written to the protocol before K1, a method) gets its own message: "`n_measurements` is a property of the
  batch now, an int" (finding r1-7).
- `flat_draw`'s and the protocol's texts: a user `draw` returns a batch of the same kind and jet shape as the full
  batch; the backend does not check it (the frontend does, `_frontend_draw`).
- `make_newton_display(oracle, batch, val_batch=None, ...)`: signature unchanged. `order = getattr(batch, 'order',
  None)` (a user batch without one: positional labels, as today for an oracle without one); a `val_batch` whose
  `(getattr(val_batch, 'kind', None), order)` differs from `batch`'s raises a `ValueError` at the call (the plan's
  §2.4: two quantities side by side).
- `t3toolbox/fitting.py`: `_is_packed(batch)`; `prepare_batch(batch, x, check=True)`: `batch.check(tuple(x.shape),
  stack_shape=tuple(x.stack_shape))` when the batch has `check` and is not packed; `ufit.pack_batch(batch, x.N)` for a
  uniform `x` (D3).
- `t3toolbox/optimizers.py`: `_frontend_draw(draw, batch, x0)` (the comparison of each drawn batch's kind and order,
  §2.2 part 0); the frontend `newton_cg`'s refusal of a `val_batch` of another kind or order;
  `_setup(geometry, batch, x0)` (its oracle argument is not
  read once `prepare_batch` does not take it: D3); the parameter comments of `mc_sgd` and `adam` move "(with n_measurements / take
  for the default draw)" from `oracle` to `batch`.

---

## 5. Every site, and what breaks (counts by `k1/k1_sites.py` on today's tree)

The plan's counts (`K_MEASUREMENTS.md` M2) are from `3375d213`, before K0; these replace them (in the tests:
`blocks_to_weight` 46 against the plan's 48, `data_metric` 41 against 41, `LeastSquaresBatch(` 75 against 76, the
keyword `weight=` 174 in 28 files against 206 in 29: K0 deleted the rest with the old kernels' tests). "lib" is the library's
code, "doc" its doctests, "pages" the doctests of `docs/**`; a cell is `count (files)`. The full list, `file:line` by
`file:line`, is `k1/sites_before_k1_full.md`.

### 5.1 Part 0's sites (the kind and the order into the batch)

| site | lib | doc | tests | examples | pages |
|---|---|---|---|---|---|
| `LeastSquaresOracle(` | 2 (1) | 16 (5) | 71 (14) | 5 (5) | 4 (2) |
| `LeastSquaresBatch(` | 4 (2) | 23 (6) | 75 (18) | 5 (5) | 4 (2) |
| `.check_batch(` | 1 | 2 | 6 (2) | | |
| `.take(` on an oracle | 1 | 3 | 21 (incl. `fit.oracle`) | | |
| `.n_measurements(` | 3 | | 8 (2) | | |
| `.kind` / `.order` read off an oracle | 1 / 1 | | 14 / 10 (6) | | |
| `flat_draw(` | 2 | | 2 (1), and the argument-name pin `test_optimizers.py:1305` | | |
| `make_newton_display(` | 1 | 1 | 15 (1) | | |
| `prepare_batch(` | 4 (2) | | 2 (1) | | |
| `pack_batch(` | 1 | 1 | 24 (11) | | |

- The `LeastSquaresBatch(` calls by their positional count (the old fields `sample, y, weight, minibatch_scale,
  drawn_indices`): tests 1 positional 1, 2 positional 25, 3 positional 45, 4 positional 3, 5 positional 1; library
  doctests 22 with 2 and 1 with 4; library code 1, 2, 1 with 1, 3, 5 (the pytree's unflatten, the translation's two,
  `take`). No construction uses a keyword today, so each becomes `LeastSquaresBatch(kind, order, sample, y,
  weight=...)` (part 0) and `block_weight=` or `full_weight=` (part 4), with the third and later positionals turned
  into keywords by hand (a scripted edit is checked line by line for the aligned comments: `K_BRIEF.md` §5).
- The other places that read the oracle's kind or order (the plan names them; re-measured): the display
  (`optimizer_display.py:622`), the default draw (`backend/optimizers.py:625-654`), the protocol's text (`:181-231`,
  prose), the translation (`t3toolbox/optimizers.py:198`, `:218`, `:249`), `prepare_batch` and `_is_packed`
  (`t3toolbox/fitting.py:582-608`), the comments of `mc_sgd` and `adam`. **Not named by the plan, found:**
  `t3toolbox/optimizers.py:108`, `:122` and `:841` (`_setup`, `_frontend_draw`, `newton_cg`: they pass the oracle to
  `prepare_batch`); `backend/optimizers.py:1233` and `:1283` (the `draw=` comments, `flat_draw(oracle, batch,
  batch_size)`); the module docstring of `t3toolbox/optimizers.py` (`:10`, "`oracle.check_batch` on a ragged batch");
  `t3toolbox/fitting.py:596`, `:660` (docstrings naming `oracle.check_batch`); `tests/_fits.py` (`shared_over`,
  `at_element`, `uniform` read `fit.oracle.kind` / `.order`: 4 sites); `tests/test_jets.py:863-893` (a structural pin:
  the checkers are called only inside `check_batch`); the mocks `mock.patch.object(LeastSquaresOracle, 'check_batch')`
  (`tests/test_optimizers_frontend.py:532`, `:1238`, `:1243`).
- **The test-local subclasses of the oracle** (found by the reviewer, round 1; `k1_sites.py` counted calls named
  `LeastSquaresOracle(` only): `HalfOracle` (`tests/test_dispatch.py:2178`), `_NoBlockSums`, `_NoDataBlockSums`
  (`tests/backend/test_optimizer_display.py:333`, `:338`), `_OffsetOracle`, `_NoResolutionOracle`, `_BareOracle`,
  `_NoKpOracle` (`tests/backend/test_optimizers.py:889`, `:1614`, `:1620`, `:2260`), `_BlockSumApplyOracle`,
  `_NoKpOracle` (`tests/backend/test_curvature.py:1020`, `:1039`): nine classes, about 15 constructions that pass the
  base's `kind` and `order` (`HalfOracle('apply', ())` 4 times, `_NoKpOracle(problem.oracle.kind,
  problem.oracle.order)`, `_OffsetOracle(fit.oracle.kind, fit.oracle.order, offset, key)`, whose own fields move two
  places left once the base has none). Part 0 rewrites them, and the audit of §7.4 (b) reads their constructions as
  oracle constructions. `tests/test_dispatch.py:2178-2193` keeps its subject: with both oracles fieldless it pins the
  type in `ValueHashedFields.__hash__` / `__eq__` (`common.py:678-680`, "type included: field-free siblings must not
  collide"), which is what decides that a subclass does not reuse the base's compiled program.
- The docs' prose naming the oracle's methods (`n_measurements`: 2 lines in 1 page) is P19's, as all narrative.

### 5.2 The weight's sites (parts 1 to 5)

| site | lib | doc | tests | examples | pages |
|---|---|---|---|---|---|
| `blocks_to_weight(` | 3 (2) | 7 (2) | 46 (14) | | 2 (1) |
| `jets.data_metric(` (the function) | 5 (2) | 7 (1) | 37 (4) | | |
| `.data_metric(` (the batch's method) | | 1 | 4 | | |
| `jet_to_dual_jet(` / `dual_jet_to_jet(` | | 5 / 2 | 34 / 19 (4 / 3) | | |
| `derivative_weight_to_taylor_weight(` | | 4 | 7 (3) | | |
| `prepare_derivative_data(` | 1 | 8 | 20 (5) | | 2 (2) |
| `check_weight(` / `check_data(` | 1 / 1 | 6 / 7 | 9 / 9 | | |
| `pack_data(` / `_pack_data_space(` | 2 / 3 | 3 / 0 | 4 / 0 | | |
| keyword `weight=` | 3 (3) | 15 (2) | 174 (28) | 5 (5) | 2 (1) |
| `.weight` read | 10 | 2 | 46 | | |

- **The keyword `weight=` in the tests, by what it goes to** (174): the engine (`sampling_variables` 50,
  `computed_names` 2, argument dicts `dict(..., weight=...)` 14 and helper calls built on them, about 20 more); the
  reference (`f`, `w_tilde`, ... through local lambdas); the tests' own fit helpers, whose weight is a block weight
  (`derivative_fit` 18, `_deriv_fit` 7, `_pd_fit` 8, `fit` 4, `_plain_fit` 4, `_shared_fit` 3, ...: they all end in
  `tests/_fits.py`); `dc.replace(batch, weight=...)` 6; old-form optimizer calls 11 (the translation keeps these
  working: they change only if the test's subject is the translation); and 21 `subTest(weight=...)` labels, which are
  not arguments (kept). By file, the heaviest: `backend/test_curvature.py` 30, `test_engine_data_space.py` 20,
  `test_dispatch.py` 15, `test_optimizers_frontend.py` 15, `test_sampling_definitions.py` 10.
- **The shared harness** `tests/_fits.py` speaks blocks (M-arc decision 18): `fit(..., weight=)` and
  `derivative_fit(..., weight=)` take one value per block and place it by `blocks_to_weight`. In part 4 they take
  `block_weight=` (passed straight in) and `full_weight=` (D5: the keyword is renamed, so that a test reads what it
  passes), and most fits change there and at their 30-odd call sites by a keyword rename.
- **Examples:** the 5 weighted calls are old-form (`examples/fit_hilbert_from_apply_derivatives_topt.py:164`,
  `fit_hilbert_uniform_probe_derivatives_newton_cg.py:166`, `fit_per_mode_weight_probes.py:91`,
  `fit_probe_display.py:89`, `fit_shared_factors_jetted_probes.py:129`): the translation keeps them running; no
  example changes for the weights. Seven change in part 0 (their `LeastSquaresOracle(`/`LeastSquaresBatch(` lines).
- **Pages with doctests that pass weights:** `docs/fitting_and_optimization.md:426-435` (`blocks_to_weight`,
  `prepare_derivative_data`, two batches) and `docs/numerical_contracts.md:92-94` (`prepare_derivative_data`, a batch):
  rewritten in part 5. The old-form calls with `weight=` in `fitting_and_optimization.md:79`, `:86` are translated.
  Page prose (`data_metric` 2 lines, `pack_data` 1, `weight=` 7 in 4 pages) is P19's.

### 5.3 The library's own docstrings and comments

A plain count of the lines of `t3toolbox/**.py` that name a changed name (prose, comments, doctests): `data_metric`
76 lines in 10 files, `blocks_to_weight` 39 in 5, `prepare_derivative_data` 28 in 3, `jet_to_dual_jet` 20,
`check_weight` 20, `derivative_weight_to_taylor_weight` 15, `pack_data` 15, `n_measurements` 12, `dual_jet_to_jet` 10,
`check_batch` 10. Outside `jets` and `fitting`, the transposes' docstrings name `data_metric(r, weight,
minibatch_scale)` (`tucker_tensor_train.py:3837-3839` and siblings; `manifold.py:765-772`, `:923-926`;
`backend/probing.py`, `apply.py`, `entries.py`): they become `probe_data_metric(order, r, block_weight, full_weight,
minibatch_scale)` (or `batch.data_metric(r)`) in the part that removes `data_metric` (5). §10 says how this prose is
written and reviewed.

### 5.4 What breaks, part by part (to be recorded by runs)

| after part | what fails, and why | repaired in |
|---|---|---|
| 0 | nothing (every construction rewritten in the part) | — |
| 1 | nothing (the engine comparisons pass the old engine `blocks_to_weight` of the block weight) | — |
| 2 | a probe fit with a `blocks_to_weight` weight (edge of 1, refused); every probe derivative fit, weighted or not (the default `k!` weight is placed with an edge of 1); a `y` with a size-1 `W` or structural axis reaching the engine; the translated old-form calls of those two kinds | 4 |
| 3 | 2's, and a batch check of such a `y` (`check_data` exact) | 4 |
| 4 | nothing expected (the translation moves with the batch's fields; the old names still exist for the re-exports) | — |
| 5 | nothing | — |

The tests of the M arc's size-1 axes become tests of the refusal (§6.2); no test is weakened: a test whose subject was
a size-1 structural axis keeps its inputs and asserts the refusal, or is rewritten to the shape the rule now requires
and keeps its assertion (its values bitwise, by check 5).

---

## 6. The tests, and what each is checked against

The rule of the arc: never the engine against itself, nor against a helper that calls the engine. The oracles used
below are (a) **brute force**: `np.ndindex` loops over a dense `z` or `Jp` from `to_dense()` and a hand `einsum`;
(b) **the reference** `sampling_definitions`, itself pinned against (a) in part 1; (c) **`tests/oracle_sweeps/oracle.py`**,
which writes the weights out by `np.repeat` and loops over the blocks, no library code; (d) **laws** (a zero weight
deletes, stacked equals separate, unbiasedness by exact enumeration, the jet and input laws); (e) the **tree before
K1** (the battery, §7). Each changed test keeps its assertions and tolerances (the diff audit, §7.4); where it loses
its subject it becomes a refusal test (§6.2).

### 6.1 Rewritten, by part

| part | file | what changes | checked against |
|---|---|---|---|
| 0 | every file constructing an oracle or a batch (14 + 18 files, §5.1); `test_least_squares_oracle.py` `TestTheBatch`, `TestTheOracle`; `backend/test_optimizers.py` (the draw, the pin of `flat_draw`'s names); `test_optimizers_frontend.py` (the mocks of `check_batch`); `test_jets.py:861-893` (the checkers' one caller); `backend/test_optimizer_display.py` | constructions; method calls moved; the pins follow the new homes | unchanged assertions; the battery `old` vs `part0` bitwise |
| 1 | `test_sampling_definitions.py` (its weighted tests), `test_dense_oracle.py`, the reference's doctests | the reference's two weights | (a) brute force |
| 1 | the engine-vs-reference tests (`test_sampling_variables`, `_stacks`, `_uniform`, `_corewise`, `_entries`, `_apply`, `_identities`, `test_engine_data_space.TestAgainstTheReference`, `test_block_sums` engine half) | the reference gets two weights, the old engine `weight=blocks_to_weight(b)` or the full weight | (b), pinning the reference's reading by the old engine |
| 2 | the same engine tests | the engine gets the two weights | (b); uniform against ragged; (d) |
| 2 | `test_engine_data_space.py` | rewritten: §6.2 | (b), (d), refusals |
| 2 | `test_dispatch.py`'s engine rows, `test_held_variables.py`, `test_least_squares_oracle.py`'s `computed_names` rows | the keyword | unchanged assertions |
| 3 | `test_jets.py` `TestDualJet`, `TestDerivativeData`, `TestCheckers` (the `check_data` table re-ruled; `check_weight`'s rows move to the batch's `check` in part 4); `test_data_metric.py` (rewritten for the three metrics); `test_type_laws.py` `TestConverters`; `test_dispatch.py` and `test_jax_absent.py` jets rows | the new functions | (a) explicit loops; factorials written out; the converters' laws |
| 4 | `test_least_squares_oracle.py`, `test_oracle_data_space.py` (rewritten), `test_block_sums.py` oracle half, `backend/test_uniform_fitting.py`, `backend/test_curvature.py`, `backend/test_uniform_curvature.py`, `backend/test_optimizers.py`, `backend/test_padded_frame.py`, `backend/test_zero_reference.py`, `backend/test_optimizer_display.py`, `test_sharing.py`, `test_optimizers_frontend.py` (`TestTheTranslation`'s 19 pins unchanged), `test_fitting.py`, `tests/oracle_sweeps/*`, `tests/_fits.py` | the batch's two weights; the harness's keywords; the translation | (c), (d), (e); the pins' numbers (bitwise, checks 5 and 6) |
| 5 | `test_jets.TestFrontendReach`, `test_manifold.py`, `test_aligned_padded_frame_frontend.py`, `test_dispatch.py` and `test_jax_absent.py` rows of the deleted functions | the frontend names; the deletions | unchanged assertions |

### 6.2 The M arc's tests of size-1 axes become tests of the refusal

| test (today) | what it established | after K1 |
|---|---|---|
| `test_engine_data_space.TestTheUniformEdgeOfOne` (4 tests: `y` and the weight constant over the edge; mixed edges; modes with different size-1 axes; a weight whose padding overflows when squared) | the edge-of-1 handling on the uniform layer | the first three: the refusal of an edge of 1 (`y`, full weight), and of modes that disagree (naming `np.broadcast_to`); the fourth keeps its subject and its assertions (a packed full weight whose padding holds `1e200`, whose square is `inf`): the engine masks the full weight on entry (`sampling_variables.py:1910-1918`, before any row squares it), so the result stays finite and the ragged one; it is the test that pins "the full weight is masked always" (§6.4) |
| `test_engine_data_space.TestTheCheck` (3) | the forms that pass and raise | re-ruled by §4.2's table; a `y` without `C` on a stack still raises; "an edge of 1 or N" becomes "N" |
| `test_engine_data_space.TestAgainstTheReference` | weights per entry, sample, slot, mode, on the edge and the first jet axis; `y` full or constant in `W` | the weight patterns become pairs (block, full): per mode and slot → block; per entry, per sample → full (`W` of 1 legal); "on the edge" and "on the first jet axis" (structural size-1) → written out full (values identical); `y` constant in `W` → written out full; plus the refusal of each old structural size-1 pattern, once |
| `test_oracle_data_space.TestTake.test_a_mode_leading_weight_is_drawn_after_its_mode_axis` | a `blocks_to_weight` weight drawn | a packed full weight drawn after its mode axis; the block weight passes through `take` (identity) |
| `test_oracle_data_space.TestPacking` (2) | `pack_data`'s join and edge rule | `pack_batch` with `pack_vectors`: the packed `y` and full weight against loops; members disagreeing in mode sizes refused; finite junk in the padding changes nothing |
| `test_data_metric.TestStructure`, `TestBlocksToWeight` | `data_metric`'s pairing and `blocks_to_weight`'s placement | the three metrics' table of §4.4 (each refusal, the E3b coincidence `(d,)` at `order=(d-1,)` refused); `TestBlocksToWeight` goes with the function, its bitwise bridge to the old block weighting kept as one test of `probe_data_metric` with a block weight against the formula written out |
| `test_jets.TestCheckers` `check_data` / `check_weight` rows | the old rule (each axis 1 or full) | `check_data`: `y` exact but `C`; the full weight's rows move to the batch's `check` (part 4) |
| `test_block_sums.TestSharedData` | data shared by the stack or the samples counted once per entry | shared by the stack: kept (`C` of 1); shared by the samples: a refusal of `y` with `W` of 1, and the count kept for a full weight with `W` of 1 |

### 6.3 New tests

- **Part 0:** the batch's constructor (each refusal: a non-string kind, an unknown kind, an order that is not a tuple of
  nonnegative ints or holds a `bool`, an entries order other than `()`; a numpy-integer order normalized to Python
  ints); keyword-only after `y` (a third positional raises); the pytree round trip with `(kind, order)` as aux data, and
  a jitted query that compiles once for two batches of one kind and order and again for another order; the methods
  (`take`, `check`, `n_measurements` as a property, `data_metric`) against today's oracle methods' assertions;
  `make_newton_display` refusing a validation batch of another kind or order; the frontend `newton_cg` refusing one at
  entry; `_frontend_draw` refusing a drawn batch of another kind or order, on a ragged and on a uniform `x0` (a user
  draw that returns the wrong order on purpose); `flat_draw` and `_require_draw` on the batch, and `_require_draw`
  refusing a batch whose `n_measurements` is a method.
- **Part 1:** the reference's brute-force pins (§2.2, part 1) and its refusals.
- **Part 2:** the engine's refusals (§4.2's table, each cell: a size-1 `W` in `y`; a size-1 edge, mode-count, jet axis
  in `y` or the full weight; disagreeing modes; a block weight of the wrong shape, as a tuple, with the E3b
  coincidence); **a zero block deletes the block** (`f`, `g`, `JtJp` equal those with that block's data and model
  removed: for probe, mode `i` at slot `k`, computed by the reference on the reduced problem) and **a zero full weight
  at a sample deletes the sample** (kept from M2); finite junk in the padding of a packed `y` and full weight changes
  no real entry; the block weight with `K` (left-aligned, no placement) against the reference element by element of
  `K`.
- **Part 3:** each metric against loops over every index with both weights, either, and neither (probe ragged and
  packed, apply, entries; `F`, `W`, `C`, `K` sizes distinct and equal); the precision rule over three dtypes; the
  dual-jet pair's round trip and pairing identity with both weights; `derivative_block_weight_to_taylor` against
  factorials written out (`(d,)+F` and `F`; `order = (2, 1)`); `prepare_*_derivative_data` reproducing the objective
  written on derivatives (check 3 as a test, with a full weight passed unchanged beside it); the probe helpers'
  refusal of a single array.
- **Part 4:** the batch's `check` (each part wrong, both weights); `take` against index loops (the block weight
  untouched; a full weight with one `W` axis of 1; `y` gathered); `pack_batch` refusing members of disagreeing mode
  sizes, modes of disagreeing size-1 patterns (the message naming `np.broadcast_to`) and `N` too small; `y_block_sumsq` with `inf` and `nan` in the padding of a packed `y` and full weight is
  finite and equals the ragged one (selection); the named oracles at the oracle with two weights (zero block, zero
  sample, stacked separate, unbiased by enumeration with a full weight per sample and a block weight per slot); the
  translation's 19 pins as they are, and one old-form weighted call per kind against the new form built by hand
  (`block_weight` the old `omega`).
- **Part 5:** the frontend's names (`TestFrontendReach`).
- **Block tables asymmetric on every jet axis** (finding r1-9): in the tests of parts 2 and 3 every block weight with
  jets is drawn at random (no two slots equal along any jet axis, `order = (2,)` and `(2, 1)`), never the `k!` table at
  `order = (1,)` (`[1, 1]`) or a near-constant equalizing weight, which would hide a weight applied after the flip.

### 6.4 Mutation passes (in a scratch copy of `t3toolbox/` and `tests/`, `python -B`)

By part, each with the test expected to catch it; the handover records the outcome of each, and the reviewer runs its
own.

| part | mutation | expected to fail |
|---|---|---|
| 0 | the constructor does not normalize `order` to Python ints | the aux-data / compile-once test |
| 0 | `make_newton_display` does not refuse a mismatched validation batch | its refusal test |
| 0 | `_frontend_draw` does not compare the drawn batch's kind and order (and, separately, the frontend `newton_cg` does not compare its `val_batch`'s) | the two refusal tests |
| 0 | `_require_draw` accepts a callable `n_measurements` | its refusal test |
| 2 | `_metric` applies the block weight unsquared | engine vs reference (weighted rows) |
| 2 | the block row right-aligned (on the trailing axes) | engine vs reference with `F != N_i` |
| 2 | `_with_K` skipped for the full weight | `Jp_norm_sq`, `JtJp` with `K` and a full weight (it raises by shape or differs) |
| 2 | `y` allowed a size-1 `W` | the refusal test |
| 2 | the full weight not masked on the uniform layer | the junk-in-padding test (`1e200` in the padding, squared) |
| 2 | the block weight applied after the flip in `w_tilde_step` (`flip(M r)` becomes `M flip(r)` for the block factor) | engine vs reference on `w_tilde`, `g`, `JtJp` with an asymmetric block table |
| 3 | the block weight applied after the flip in `probe_jet_to_dual_jet` (and before the flip back in `probe_dual_jet_to_jet`) | the converters against loops with an asymmetric table; the pairing identity |
| 3 | `derivative_block_weight_to_taylor` divides | the factorials test and the objective-on-derivatives test |
| 3 | `probe_data_metric` skips the check of `r`'s own `F` | the refusal of a size-1 slot axis in `r` |
| 4 | the oracle's sums without the broadcast to `z`'s shape | `y_block_sumsq` with `C` of 1 (off by `|C|`) |
| 4 | `take` gathers nothing for the full weight | `take` against index loops |
| 4 | the selection in the oracle's sums removed (the product kept) | `y_block_sumsq` with `inf` in the padding |
| 4 | `pack_batch` does not compare the mode sizes | its refusal test |
| 4 | `pack_batch` does not check the modes' size-1 pattern (left to `pack_vectors`) | the refusal test asserting the message names `np.broadcast_to` |

---

## 7. The battery and the tally

### 7.1 What is recorded, and when

`k1/battery.py` (written in the build of part 0, before any library line changes), one script run on three trees:

- `old`: K1's starting tree, `282b46a9`, a `git archive` in scratch, `python -B`;
- `part0`: the tree after part 0 (the constructions in the new form, the one field `weight`);
- `new`: the tree after part 5.

It covers, as M2's did: every public variable of the engine for the three kinds, manifold frame and corewise point,
ragged and uniform, jet shapes `()`, `(1,)`, `(2,)`, `(1, 1)`, `(2, 1)` (entries `()`), `K = ()` and `(2,)`,
`C = ()` and `(2,)` (a `y` shared over the stack), `sum_over_samples` both; the oracle's outputs (`f`, `g`, `JtJp`,
`Jp_norm_sq`, `Kp`, `r`, `Jp`, `r_block_sumsq`, `y_block_sumsq`, `f_resolution`) on full and drawn batches;
`prepare_derivative_data` (with a block weight and with none) and `derivative_weight_to_taylor_weight`;
`data_metric` / the three metrics; short weighted runs of `newton_cg`, `mc_sgd`, `adam` and `gradient_descent`, ragged
and uniform, in the new call form and in the old (translated) one; one jitted run of the engine and of a query.

The weights it uses, each in the form of its tree:

| case | `old` and `part0` | `new` |
|---|---|---|
| a block weight `b` (per mode and slot; per slot) | `weight=blocks_to_weight(kind, order, b, y)` | `block_weight=b` |
| a full weight `f` with exact structural axes (per entry; per sample with `W` full and `F`, edge written out) | `weight=f` | `full_weight=f` |
| a per-sample weight with `W` of 1 on some axes | `weight=f` | `full_weight=f` |
| an old full-form weight with a size-1 structural axis (edge of 1, jet axis of 1) | `weight=f` | `full_weight=np.broadcast_to(f, ...)` (values identical) |
| both a block and a full weight | `weight=blocks_to_weight(b) * f` (one product) | `block_weight=b, full_weight=f` |
| `y` constant over `W` (a size-1 `W`) | `y` with `W` of 1 | `y` written out on `W` |
| derivative data with blocks `b_d` | `prepare_derivative_data(kind, order, y, blocks_to_weight(b_d))` | `prepare_*_derivative_data(order, y, b_d)` |

Every stored weight-derived output is normalized to one key and one shape on both trees.

### 7.2 The expectations

- `old` vs `part0`: **bitwise**, every number, every optimizer path (part 0 moves fields and calls, not arithmetic).
- `old` vs `new`:
  - **bitwise** (dtype, shape, every bit): every engine output and every oracle output on the ragged layer, for every
    case but "both weights" (check 5: `(b*b)*r` and `(f*f)*r` are today's `(omega*omega)*r`; `w_tilde` and everything
    downstream follow); the derivative data (check 6); the translated optimizer runs (their weights go through the same
    products);
  - **to rounding** (relative 1e-13), with the reason: the "both weights" rows (`b^2 f^2` against `(b f)^2`); on the
    uniform layer `y_block_sumsq` and `f_resolution` (one reduction over the padded array, §3);
  - **the jitted runs** to 1e-12 (XLA's fusion of a changed program need not keep a reduction bitwise);
  - **the optimizer runs** in the new call form: bitwise where every case is single-weight (expected for all but the
    "both" run). Where a run is not bitwise, it is compared on what it promises: the final loss and the losses of every
    recorded iteration to 1e-10 relative, the iteration counts and the stop reason exactly, the represented tensor
    (`to_dense()`, or on a large problem its probes at 64 fixed vectors) to 1e-10 relative, never the cores (a gauge).
    A run that sits on a knife edge (an acceptance decided at rounding level) is changed, never the tolerance (plan
    §8). The maintainer: bitwise agreement of optimizer paths is welcome, not required.

### 7.3 The per-test tally

The baseline is K0's own `-v` record of this tree (`implementation/k0/k0_suite_v.log.gz`, with
`tests/test_contractions_sharding.py` rerun after its floor fix: 1,882 passed, 5 skipped, 1 xfailed). After parts 0 and 5 (and of each part, as
its expected-red record): `m0/tally.py`-style comparison, every node that disappears or appears accounted for by name
(the renamed `check_batch` tests; the deleted `blocks_to_weight` tests; the new tests).

### 7.4 The audits (`k1/diff_audit.py`, written before part 0's build)

Two conversions K1 makes by hand cannot be caught by any number (round 1, findings 1 and 2): a wrong `(kind, order)`
moved into a batch passes every check of shapes (input jets read at the wrong jet shape, the trap of
`LeastSquaresOracle`'s own docstring) and a test built from it compares two wrong numbers; and for apply and entries a
`blocks_to_weight` output passed as `full_weight=` is legal and bitwise the block weight's metric. So each part's
evidence includes an audit by construction, which pairs the old site with the new one.

**How it reads.** For every file a part changes (the tests, the examples, the library's docstrings' doctests, the
pages' doctests), the version at the part's base and at its result (`git show`), both parsed with `ast`. Sites are
grouped by **scope**: the enclosing function (for a method, `Class.method`; a doctest, its docstring; a page, the
file). Expressions are compared as `ast.unparse` text, so spacing and parentheses do not matter. The audit exits
non-zero on an unpaired or unexplained site and writes its lists to `k1/part<j>_audit.txt` for the part's reviewer.

**(a) Every part: the lines.** Every changed line classified (a construction, a keyword renamed, a method call moved,
a weight written out by `np.broadcast_to`, a refusal added, a new test), and every changed `assert*` call or tolerance
listed by name. Expected: no tolerance changed anywhere; the assertions changed only in the tests of §6.2.

**(b) Part 0: the kind and the order.**
- In the old version, every construction of `LeastSquaresOracle` or of a class that derives from it in the same file
  (found by `ast`: `class X(LeastSquaresOracle)`, `class X(bfit.LeastSquaresOracle)`; the nine of §5.1) is bound to
  the base's `(kind, order)` (positional or keyword; a subclass's own fields after them are set aside), giving a pair
  of expressions.
- In the new version, every `LeastSquaresBatch(` construction is bound to `(kind, order, sample, y, ...)`, giving its
  pair.
- In each scope: every new batch pair equals an old oracle pair of the same scope (or of the enclosing class's
  `setUp` or body, or of the module level, in that order); every old batch construction has a new one (equal counts);
  every old read `<x>.kind` / `<x>.order` with `x` an oracle is now a read off a batch, and listed.
- **One level of helper** (round 2, finding 11): when a new batch's pair is made of its scope's own parameter names
  (a helper such as `Fixture.batch(self, kind, order)`, `_batch(kind, order, ...)`, `tests/_fits.fit`), each call site
  of the helper in the same file substitutes its arguments for them, and the result is compared with the old oracle
  pairs of the call site's scope.
- **Listed, for the reviewer to read by hand**, each with its call chain: a pair that still cannot be resolved; every
  removed oracle construction whose scope keeps no paired batch; every scope with two distinct old pairs (ambiguous:
  "equals an old pair" would accept the wrong one; none in `tests/`, one in the library's doctests,
  `LeastSquaresOracle`'s, whose printed outputs cover it); and every construction of a test subclass of the oracle,
  its old and its new argument lists side by side (`_OffsetOracle(kind, order, offset, key)` becoming
  `_OffsetOracle(offset, key)`: a swapped `offset`, `key` is invisible to every number but the test's own). The
  handover gives the counts.

**(c) Parts 1 to 5: the weights.**
- Every `blocks_to_weight(k, o, b, y)` call removed is paired with a `block_weight=` in the same scope whose value is
  the old third argument `b` (directly, or through the one assignment the old code made, `w = blocks_to_weight(...)`
  then `weight=w`), on the call that consumed it.
- Every old keyword `weight=X` (any callee: the engine, the reference, a helper of the tests, `dc.replace`; a
  positional weight of `LeastSquaresBatch` counts, part 0 having made it a keyword) is paired with the new call of the
  same callee in the same scope (by the order of the calls in the scope) and classified: `block_weight=`,
  `full_weight=`, or removed (with its reason: the test's subject moved, §6.2).
- **Every new `full_weight=` site needs a line** in `k1/full_weight_reasons.tsv` (file, scope, callee, the expression,
  the reason: a weight per entry, per sample, an old full-form weight written out on a structural axis, ...), and the
  audit fails on a site with none. The apply and entries sites are marked: there a block weight passed as a full weight
  would give the same numbers, so the reason is what tells them apart. The reviewer reads the file.
- **Every new `block_weight=` site whose old value was not a `blocks_to_weight` output** is listed too, with a reason
  line in the same file: an old full-form weight turned into a block weight keeps its meaning only if it was constant
  off the blocks.
- What the audit cannot see (a weight built in a helper of another module and passed through two scopes) is listed as
  unresolved, as in (b).

---

## 8. The gate

Per part: the tests of the part's files and the expected-red list by a run (`-n 2`, one heavy run at a time); the
battery's rows for what the part changed; the scan. At part 5, the full gate (`run_gate.sh --detach -n 2`: the suite,
the module doctests, the quickstart, the pages, the examples, Sphinx with `-W`), the slow tier serially (K1 touches the
optimizers' path: `T3TOOLBOX_SLOW_TESTS=1 ... tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py`), the
battery `old` vs `new`, the tally, the diff audit, `k1_sites.py` with no call of a removed name left in `t3toolbox/`,
`tests/` or `examples/` (the pages' prose is P19's), and both CI legs after the push (the floor's Python 3.9 matters
for the batch's hand-written `__init__`, already measured on the floor environment, §4.6). Memory: MemAvailable
reported after each gate.

---

## 9. What K1 leaves to the later slices (so that nothing is half-done silently)

- **The forms** (D1): the engine still accepts a ragged `y`, full weight and sample on a uniform frame (packing them)
  and a mode-leading array on a ragged frame; `pack_batch` keeps a packed member as it is; `pack_sample`,
  `pack_if_ragged`, `is_packed` stay. K3 (ragged takes sequences only) and K5 (uniform takes packed only) change these.
- **The masking mechanism** of the engine stays the product (`_masked`) until K5; only the oracle's two data sums mask
  by selection from K1 (§4.8).
- **The per-mode loops** of the engine's data-space steps stay (one application of `_metric` per mode) until K5 (plan
  §6, its program-size gate is K5's).
- **The checkers:** `check_sample` and `check_data` stay until K3 (`check_data` exact on `y` from part 3; its
  docstring says it is `y`'s checker, not the full weight's, which `batch.check` checks: O3).
- **The public full-weight checker** (O2): K2's design note settles it, and the orchestrator takes it to the maintainer
  (public names the plan does not have; the reviewer's and my recommendation, `check_{probe,apply,entries}_full_weight`
  taking the sample and the stack, not `y`). Until then the batch's `check` uses the private helper of §4.7, whose rule
  table (§4.2's, the `W` axes added to the axes that may be 1) K2 lifts.
- **`inf` and `nan` in a packed `y`'s padding** (§4.8): from K1 the oracle's `y_block_sumsq` is finite with them (it
  masks by selection); the engine's `f`, and so `f_resolution`, are finite with them only from K5, whose selection
  replaces the engine's product mask. No docstring written in K1 promises more: the batch's, the oracle's and
  `pack_batch`'s texts say the padding of data is masked (true), never that it may hold `inf` (K5's contract,
  `PACKING_PLAN.md` §1.3).
- **The comparison of a drawn or validation batch's kind and order** is in the frontend from K1 (part 0); K4 keeps it
  when it rewrites `_frontend_draw` and adds it where a frontend batch enters. The backend optimizers never look.
- **`PerSite`**: the probe metric is annotated `typ.Any` until K2 adds the `TypeVar`.
- **The frontend batch class** (`PackedVectors` members, `pack()`, the refusal of the backend's class) is K4's; in K1
  `t3toolbox.fitting.LeastSquaresBatch` is the backend class.
- **`derivatives_to_taylor`, `taylor_to_derivatives`, `data_block_norms`, `rescale_direction`,
  `rescale_coordinates`** keep their `kind` argument until K6. `jets.py`'s module doctest and
  `tests/test_jets.py`'s `TestDerivativeData` use K1's functions beside K6's (the plan's §5, "touched twice").
- **The narrative pages** (`fitting_and_optimization.md` §4.6, the glossary, `weighting.md` if it names the fit's
  weight) are P19's; K1 changes only the pages' doctests that would fail.

---

## 10. The prose

The docstrings and comments K1 rewrites are content (`K_BRIEF.md` §5; K0's decision 12). K1 is too large for every
docstring to be written twice, so I take the orchestrator's suggestion, made precise:

- **Written in the design phase, reviewed with the note** (`k1/prose_edits.md`, before and after, as K0's): every
  **module docstring** K1 changes (`backend/jets.py`, `backend/fitting.py`, `backend/uniform_fitting.py`, the
  paragraphs of `backend/sampling_variables.py` and `backend/sampling_steps.py` on the weight and the whole steps,
  `backend/sampling_definitions.py`'s paragraphs on `y` and the weight, `t3toolbox/fitting.py`, the head of
  `t3toolbox/optimizers.py`), and the docstrings of every **public function and class K1 adds or whose contract
  changes**: the three metrics, the four converters, `derivative_block_weight_to_taylor`, the two
  `prepare_*_derivative_data`, `LeastSquaresBatch` (the class and its four methods), `LeastSquaresOracle`'s class
  docstring, the engine's parameter entries for `y`, `block_weight`, `full_weight`, `pack_batch`, `flat_draw`, the
  `Oracle` protocol's text, `make_newton_display`'s paragraph on the validation batch. The doctests in them are drafted
  with the text and **pasted from a run during the build** (a doctest's output is never written by hand; the draft
  marks each output line as "to be pasted").
- **Reviewed with each part's build:** the private helpers' docstrings, the comments, the shape comments of the
  signatures (all in §4 already), the tests' docstrings, and the one-line references in other modules' docstrings
  (the transposes' "pass `data_metric(r, weight, minibatch_scale)`", §5.3), which change by a fixed substitution
  listed in the part's handover.
- **When** (the orchestrator's rule, round 1): **a part's build starts only when its public docstrings are written in
  full and reviewed.** For the design sign-off, parts 0 and 1 are in full in `k1/prose_edits.md` (§1 and §6 there);
  parts 2 to 5 may stay "today's text with the changes listed" (§2 to §5 there), and each is completed in full and
  reviewed before its part is built (a short prose round between the parts).

---

## 11. Decisions (mine; each recorded with its reason)

- **D1. K1 changes the shape rules and the names, not the forms.** The forms (a tuple on ragged, one array on uniform)
  are K3's and K5's (plan §7); doing them here would edit the same lines twice and blur the battery. The exception is
  the plan's own: the probe helpers renamed in K1 (`probe_jet_to_dual_jet`, `probe_dual_jet_to_jet`,
  `prepare_probe_derivative_data`) take the ragged form only from K1 (plan §3.3), and `probe_data_metric` both forms.
- **D2. Keyword-only after `y`, by a hand-written `__init__`** on a frozen dataclass with `init=False` (§4.6; measured
  on 3.9 and 3.11). Two weights of different meaning, a scale and the drawn indices are four optional arguments a
  positional call can swap; for apply with `W = C = ()` a block weight and a full weight even have one shape. Cost:
  15 lines and a keyword unflatten.
- **D3. The oracle argument leaves the frontend's batch helpers** (`prepare_batch(batch, x, check=True)`,
  `_is_packed(batch)`, `_frontend_draw(draw, x0)`, `optimizers._setup`): after part 0 they would not read it. They are
  the frontend's private helpers (`prepare_batch`'s docstring: "private to the frontend"; two test calls). K4 rewrites
  them again.
- **D4. The frontend's re-exports follow the backend's names** (`t3toolbox/fitting.py:79-107`): `probe_data_metric`,
  `apply_data_metric`, `entries_data_metric`, `derivative_block_weight_to_taylor`, `prepare_probe_derivative_data`,
  `prepare_apply_derivative_data` in place of `data_metric`, `blocks_to_weight`, `derivative_weight_to_taylor_weight`,
  `prepare_derivative_data` ("a fit needs no import from `t3toolbox.backend`", its module docstring). The converters
  stay unexported there, as today.
- **D5. The test harness says which weight** (`tests/_fits.py`: `block_weight=`, `full_weight=` in place of `weight=`),
  so a converted test reads what it passes; most fits change by this rename at their call sites.
- **D6. A block weight is never a tuple or a list** (a `TypeError` in the engine, the metrics, the batch's `check`):
  it is the same object on both layers, and a sequence would be read as a ragged form by `_cast` and `_leaves`.
- **D7. The part-2 bridge** (§2.1): one line in the oracle, replaced in part 4, so that the parts between are
  reviewable; no weighted fit runs wrong in between.
- **D8. Each old name is deleted with its last library caller** (§2.1's table), not in the `jets` part: the order of
  the parts stays the plan's.
- **D9. `pack_batch` refuses members that disagree in their mode sizes from K1** (the plan's §3.1 puts the refusal on
  `pack_batch` without a slice; K1 rewrites it), and keeps a packed member as it is until K5.
- **D10. `probe_data_metric` and `apply_data_metric` check `r`'s own jet axes** against `order`, whatever the weights
  (§4.4: a size-1 slot axis of `r` would otherwise be broadcast by the left-aligned block weight).
- **D11. The converters accept a jet with `K`** when a full weight has its size-1 `K` axes (today a jet with `K` and any
  weight raises by rank): they apply the metric, whose rule admits it (plan §2.3).
- **D12. `prepare_*_derivative_data` with no weight returns the `k!` table always** (ones at `order = ()`), a block
  weight of the exact block shape, never `None`: the caller passes it straight to the batch.
- **D13. `make_newton_display` compares `(kind, order)` by `getattr(..., None)`** on both batches: a user batch type
  without them is not refused (it has no table layout to disagree on either), as the display reads `order` today. The
  frontend's comparisons of a validation batch and of each drawn batch (part 0, the ruling on O1) do the same.
- **D14. The examples change in part 0** (their two construction lines, seven files), though the examples are P17's:
  `LeastSquaresOracle('apply', order=())` raises from part 0. Nothing else in them changes.
- **D15. The translation moves in part 4, with the batch's fields** (§2.1 (c)), not in the frontend part: it is the
  one library builder of batches outside the fitting module.
- **D16. No sentinel default for `sample`** (the orchestrator's ruling on finding r1-6): the two-argument old form
  fails on Python's "missing 'sample'"; the signature stays true.

---

## 12. Open points for the orchestrator, and the rulings of round 1

**Ruled (2026-11-25, round 1):** O1 accepted, `flat_draw(batch, batch_size)`, with the guards of a drawn and a
validation batch's kind and order brought into part 0 (§2.2); O2 is K2's design note's, to be taken to the maintainer
then, the batch's `check` using the private helper meanwhile (§4.7, §9); O3 accepted, with `check_data`'s docstring
saying it is not the full weight's checker; O4 accepted; D4 (the frontend's re-exports under the new names)
confirmed and logged. The text of each point below is kept as it was put.

- **O1. `flat_draw`'s signature** (a public backend function; the plan does not name its signature). After part 0
  its `oracle` argument is unread (the draw calls `batch.n_measurements` and `batch.take`). Options: (a)
  `flat_draw(batch, batch_size)`, the unread argument removed (G3's spirit: an argument nothing reads is a trap; a
  test pins the names, `tests/backend/test_optimizers.py:1305`, and two tests call it); (b) keep
  `flat_draw(oracle, batch, batch_size)` with `oracle` documented as unread, for compatibility within the branch.
  **Recommendation: (a).** No release has the oracle form (P16 introduced it on this branch), so nothing outside the
  branch breaks.
- **O2. K2's data checker cannot tell `y` from a full weight.** The plan's §3.1 gives
  `check_probe_data(value, sample, order, stack_shape=())` for "y, a full weight, a residual", but §2.2 gives them two
  rules on `W` (`y` exact, a full weight 1 or full). Options for K2: (a) three more checkers,
  `check_{probe,apply,entries}_full_weight` (eight checkers; one rule each, G3); (b) the data checkers check the looser
  rule and only the engine and the batch's `check` enforce `y` exact on `W` (the checker then looser than the engine,
  which `PACKING_PLAN.md` §2.4 forbids for `check_data`); (c) a keyword naming the object (a switch, G3 says split).
  **Recommendation: (a)**, decided in K2's design note. K1 is not blocked: until then the batch's `check` uses a
  private helper (§4.7).
- **O3. No public checker of a full weight between K1 and K3** (`check_weight` goes in K1, the plan's §2.4). A backend
  user checks a batch with `batch.check(shape, stack_shape)`, which covers both weights. **Recommendation:** accept;
  recorded so that it is not mistaken for an omission.
- **O4. The order of K1's parts is amended** (D7, D8, D15: deletion with the last caller, one bridging line, the
  translation with the batch). The parts and their order are the plan's; what each deletes moves later, and the
  translation one part earlier. Within the slice, so mine to decide by `K_BRIEF.md`
  §3; listed here because it changes the plan's description of part 3 ("`blocks_to_weight` removed") and adds a
  temporary line to the library between parts.
