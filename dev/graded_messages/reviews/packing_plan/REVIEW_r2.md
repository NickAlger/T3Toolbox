SIGN OFF

# Review of `PACKING_PLAN.md`, revision 2

Reviewer: Claude (critic), 2026-11-24. Branch `graded-messages` at `3375d213`, with the working tree's `dev/` files
(revision 2 of the plan, 843 lines; `RESPONSE_r1.md`). The library and the tests are unchanged since round 1.

**The verdict, briefly.** No blocking finding is left. One should-fix (N1) is a small change of one rule, but the
maintainer must see it, because it departs from a ruling that §13 does not list. The minor findings are edits to the
text or to a slice's design note, and none needs another round of review. I sign off on the condition that N1 is taken
and §13 completed (N7).

## 1. What I read and ran

**Read.** Revision 2 whole; `RESPONSE_r1.md`; `PACKING_PLAN_NOTES.md` again, line by line against §12 and §13. Code
for the new design: `LeastSquaresBatch`'s registration and `_BATCH_FIELDS` (`backend/fitting.py:757-815`), `take`
(`:1190-1240`), `backend/optimizers.py` (`_prepare_jit_inputs` `:712-735`, `_jit_engaged`, `flat_draw`, the
`LocalModel` registration `:2531-2546`), `common.partition_static` / `rebuild_static` / `_is_static_leaf`
(`common.py:697-754`), the frontend `optimizers.py` (`_setup`, the `val_batch` path `:839-849`, the translation),
`ut3_sampling.ut3_probe_corewise_hessian` (its held `r`), the jets helpers' callers in the library, the current
signatures of the helpers that §5 renames, and `jets.py:678-690` (the packed weight's mode axis of 1).

**Ran** (in `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/critic/`):

| Script | What it tests | Outcome |
| --- | --- | --- |
| `e7_select_vs_multiply.py` | §1.3 and §3.2's masking by selection, against today's product, in the engine (applied to every input, so it is the stricter test): (a) clean inputs, all names, probe and apply, `order` `()` and `(2,)`, compared byte for byte; (b) `-0.0` in a padding; (c) `inf` in `y`'s padding and in a held `c`'s | (a) 104 names: equal in value everywhere; 4 differ in bytes, all at entries that are zero with a different sign (apply's `nu_tilde`, `nu_L_tilde`); (b) no difference in `r`; (c) `f` is `nan` under the product and finite (33.47) under selection; `JtJp` is non-finite under the product and finite under selection |

I also used round 1's E1 to E6 again where the revision cites them.

## 2. The findings of round 1, now

| # | Status | Note |
| --- | --- | --- |
| 1 | Resolved | K4 (the frontend) before K5 (the backend). Checked at the boundary: the frontend hands over `.array`, and the twins still accept both forms until K5. |
| 2 | Resolved | The batch holds plain arrays and has a static `widths` field. Checked (§4 below): flattening, `take`, `dataclasses.replace`, `_prepare_jit_inputs`, the backend `LocalModel`'s `partition_static` (it keeps the batch whole, so jax flattens it through its own registration), and `val_batch` through `prepare_batch`. Small leftovers: N4. |
| 3 | Resolved, with one gap | The split and the exact block weight are right. The full weight's rule reopens one silent misread the split could close: N1. |
| 4 | Resolved | Nothing in the library calls a probe helper on a packed array except the translation, the frontend's `rescale_coordinates` and docstrings, and those calls are ragged. |
| 5 | Resolved | E7: selection changes no value on clean inputs and removes E4's `nan`. |
| 6 | Resolved | |
| 7 | Resolved as a document | The question itself is §13.7 (on its fairness, §3 below). |
| 8 | Resolved | |
| 9 | Mostly resolved | What is still edited twice: N5. |
| 10 | Resolved | One wrong detail: N6. |
| 11 | Resolved | One wrong citation: N6. |
| 12 | Resolved | |
| 13 | Resolved, nearly | The K4 row is missing two items: N8. |
| 14 to 18 | Resolved | One inconsistency left from 15: N3. |
| 19 | Resolved | I accept the reason for keeping `entries_data_metric` and `entries_data_block_norms`. A user holding entries data has no `order`, and the jet functions have no entries version for a different reason. |
| 20 to 23 | Resolved | |
| 24 | Half resolved | N9. |
| C1, C3, C4 | Adopted | |
| C2 | Put to the maintainer | §3 below. |

## 3. New findings

### N1. Should fix. `probe_data_metric` keeps the mode axis of 1 that notes 3d removes (§2.3, §13)

- **What is wrong.** The full weight is checked "as a factor of an elementwise product: `r`'s rank, each axis 1 or
  `r`'s size". Against a packed `r` that includes axis 0, the mode axis. So a packed full weight built for one mode
  (mode axis 1) is broadcast over every mode, silently. This is the limit today's `data_metric` documents with a
  doctest (`jets.py:678-690`: "mode 1 weighted by mode 0's entries"). Notes 3d lists that limit as **removed** ("the
  'mode axis of 1 broadcasts unchecked' limit"), and §13 does not mention it.
  - The function now has the kind, which is what the split was made for: the plan's own criterion (§1.4, "or the
    structural check of its arguments") requires the exact check here.
  - It also breaks parametricity: the ragged branch requires exactly `d` arrays, so the two forms get different rules.
  - The engine is not affected, because it checks the full weight exactly before it calls the metric. The exposure
    is the custom-loss path the plan recommends (`probe_data_metric`, and `oracle.data_metric(batch, r)` on a batch
    built by hand that never reaches the engine).
- **What would resolve it.** For `probe_data_metric`: axis 0 of a packed full weight exactly `d`, as the ragged branch
  requires `d` arrays. The other axes keep "1 or `r`'s size". State it in §2.3, and note in §13.3 that the public
  metric's full weight keeps size-1 axes (notes 3d: "No size-1 axes"), with the mode axis as the exception.

### N2. Minor. `_as_index` at the K3 boundary (§4.2, §7 K3, K4)

- **What is wrong.** The ragged frontend's `entries` (and its corewise siblings) turns its argument into one array with
  `_as_index` (`tucker_tensor_train.py:63-68`, `:3564`). From K3 a single array is refused. So unless the ragged
  methods stop calling `_as_index` in K3, two things break there: the doctest `x.entries((1,2,3))` (`:3551`) and the
  ruled `x.entries([9, 4, 7])`. The plan deletes `_as_index` in K4, because the uniform frontend uses it until then
  (`uniform_tucker_tensor_train.py:420`). Both can be true, but the plan does not say that the ragged methods drop it
  in K3.
- **What would resolve it.** K3: the ragged methods stop calling `_as_index`, and the engine's ragged path converts
  each of the `d` members with `xnp.asarray`, so that a list of lists works. K4: the function is deleted.

### N3. Minor. Two rows of §5 disagree with the text (§5, §3.1, §3.3, §2.4)

- `check_weight` is "Removed" in K1 (§2.4), but §3.1 says the new checkers replace `check_sample`, `check_data` and
  `check_weight` "in K3", and §5's row gives "K2, K3".
- `index_to_one_hot` is "one function each, as today" in §5, but "ragged only" in §3.3. Today it takes a `(d,)+W`
  array (`jets.py:1303-1383`, and the engine's doctest at `sampling_variables.py:1843-1850`). So it changes, and no
  slice is named. It belongs in K3 (the translation's K3 row already assumes it). Its refusal names `unpack_index`
  or `tuple(index)`, not `unpack_vectors` (§3.3's wording).

### N4. Minor. The batch's `widths`: a hand-built batch and normalization (§4.3, §2.4)

- A packed batch built by hand states its widths with `dataclasses.replace(batch, widths=...)`. That skips the members'
  agreement check that `fitting.pack_batch` makes. A `y` packed from data with one coordinate missing then passes,
  which is G1's bug class (F1).
- `widths` given as a list or a numpy array makes `prepare_batch`'s comparison fail on equal values (`[4, 5, 6] !=
  (4, 5, 6)`). A numpy array cannot be hashed or compared as `aux_data`.
- **What would resolve it.** Normalize `widths` to a tuple of Python ints when the batch is built (normalization, not
  validation). Point the hand-built route to a checking constructor, or document that it is unchecked. Say what
  `prepare_batch` and `fitting.pack_batch` do for an oracle that is not a `LeastSquaresOracle` (today such a batch
  passes unchanged; `fitting.pack_batch` reads `oracle.kind` and `oracle.order`).

### N5. Minor. Code that is still edited before it is deleted (§5, "no doomed code is edited"; §7 K1)

- `pack_data` calls `_pack_data_space`, which K1 removes, but `pack_data` itself is deleted only in K5. K1 must rewrite
  it or delete it early.
- `jets.check_data` must learn the exact `y` in K1 (or stay looser than the engine until K3), and K3 replaces it.
- `jets.py`'s module doctests (`:67-152`) and `tests/test_jets.py`'s `TestDerivativeData` mix K1's renames
  (`prepare_derivative_data`, `blocks_to_weight`) with K6's (`derivatives_to_taylor`), so they are edited in both
  slices. §5's "No function, test or doctest is edited for the weights and then again for the split" is therefore not
  quite true.
- The helpers renamed in K1 (the dual-jet pair, `prepare_*_derivative_data`) are ragged only by §3.3, but no slice is
  named for their refusal of an array.
- **What would resolve it.** Name the slice for each (K1 can refuse with `isinstance`; `form_of` comes in K2), and
  soften the sentence of §5.

### N6. Minor. Three wrong details (§6, §7 K0, §4.3)

- §6: "438, 452, 491 at `d = 3, 4, 5`". E2c measured at `d = 3, 5, 8` (`REVIEW_r1.md` §1).
- K0: "`chunk_size` and `sum_over_probes` in the translation's old signatures go with it in P17". The translation has
  `chunk_size` only. `sum_over_probes` is in deleted code only (`apply.py:297`, `entries.py:255`, the old kinds at
  `backend/fitting.py:543-717`), so it goes in K0; its one page mention is P19's (`uniform_equivalence_contract.md:118`).
- §4.3 quotes the maintainer: "the uniform batch remember its shape". That phrase appears nowhere in `dev/` but the
  plan, while the notes record the opposite (3c: "the widths on `LeastSquaresBatch` was rejected as too specific").
  For a document the maintainer reviews, give the source or drop the quote.

### N7. Minor. §13 does not list three departures from the notes

1. Notes 3d: "No size-1 axes". In the public metric the full weight keeps "each axis 1 or `r`'s size" (§2.3). This is
   stated in §2.3 but not in §13.3, and with N1 the mode axis is the case that matters.
2. Notes 3d: the "mode axis of 1 broadcasts unchecked" limit is removed. The plan keeps it (N1).
3. Notes 1d and `PACKING_REFACTOR.md` §2.7: the `TypeVar` `DataSpace`. It is dropped (§3.3), which is not listed.

### N8. Minor. The translation's K4 row (§4.4)

- For a uniform `x0` the translation must also pack the old form's validation data (`val_sample`, `val_data`,
  `t3toolbox/optimizers.py:257-259`), since `prepare_batch` refuses a ragged `val_batch` too.
- It must pack at `N = x0.N` (today `prepare_batch` packs at `x.N`; `fitting.pack_batch`'s default is the largest
  width).
- The K6 row is empty: the translation calls none of K6's renamed helpers.

### N9. Minor. `ut3_unscale_coordinates` in K7's meta-test (§1.3, §3.3, §7 K7)

- It returns cores. It masks the scales, but its docstring says the supercore's rank padding passes through ("garbage
  stays garbage", `data_preparation.py:588-592`). K7's test requires every output's padding to be zero, so either K5
  makes it mask its output (it holds the masks), or it is registered with the weaker property (the real part bitwise
  equal), as the investigation's rule for operations that produce cores allows. Say which.

### N10. Minor. Wording of two rules (§1.6, §7 K2)

- §1.6: "a Python loop over the `d` slices of a packed array does not appear in library code". The unpackers
  (`unpack_vectors`, `unpack_index`, `PackedVectors.unpack`) are such loops by definition. Exempt the conversions.
- K2: `form_of` raises on "a 0-d array", but §1.1 says that nothing but the container is read. Either say that
  `form_of` also requires `ndim >= 1`, or let the 0-d case fall to the structural checks.

## 4. Challenges to rulings

**C2, as §13.7 states it.** The statement is fair. Two facts would complete it:

- **The public metric already gives `W` the meaning of size 1.** After revision 2, `probe_data_metric` and
  `apply_data_metric` accept a full weight with `W` axes of 1 (their rule is "each axis 1 or `r`'s size"). So "it
  brings back one size-1 meaning" applies to the batch and the engine only. Today a custom loss may use the cheap form
  and a fit may not.
- **A named factor for T3Polynomial's case has an edge.** A per-coordinate weight (`F + (N_i,)` per mode) is a per-mode
  object. It needs packing, masking by selection, widths and a rule for `take`: more seams than `W = 1`, which adds no
  object. The plan's recommendation (decide at P18 on the measured cost) is reasonable. The maintainer should see both
  costs side by side.

No other ruling challenged.

## 5. What is sound (checked in this round)

- **The static `widths` on the batch** is coherent:
  - it is `aux_data`, a hashable tuple, so it is part of the compilation key and constant for a problem;
  - `take` and `dataclasses.replace` keep it;
  - `_prepare_jit_inputs` maps the leaves with `jax.tree_util.tree_map`, which keeps the aux;
  - the backend `LocalModel` passes the batch whole through `partition_static` (a batch object is not a static
    leaf), so jax flattens it through its own registration and the field survives.

  Nothing in the backend optimizers rebuilds a batch positionally; only `take` does, and the plan has it keep the
  field. "The backend ignores `widths`" is accurate: the oracle masks with the frame's shape, not the batch's, and
  only `pack_batch` writes the field and `take` copies it. `None` for entries is right (the engine range-checks the
  index against the frame's shape).
- **The split of the metric.** `probe`, `apply` and `entries` with the exact block weight. The engine's internal calls
  fit: they already place the size-1 `K` axes on the full weight, and the block weight needs none. The count of 28
  is right (3 + 18 + 5 + 2).
- **Ragged only for the helpers.** Checked against every library caller and every stated workflow (the oracle, the
  display, the translation, the engine's steps): none needs a packed branch. T3Polynomial's assembly I could not
  check (§6).
- **Masking by selection.**
  - The asymmetry is coherent: data are written by users on packed arrays, while cores and samples are written by the
    library or are finite by the existing rule.
  - It changes no value on clean inputs (E7 (a)). Only the sign of some zeros differs, so K5's battery should compare
    values (`array_equal`), not bytes.
  - The cost under jit is that of the product it replaces.
- **The slice order, at every boundary** (K0 to K7), apart from N2, N3, N5 and N8:
  - K1's weights;
  - K2 additive;
  - K3's ragged change with `pack_index`;
  - K4 handing packed arrays to a backend that still accepts both;
  - K5's refusals hitting backend tests only;
  - K6 on ragged helpers, which no engine step calls;
  - K7 last.
- **Round 1's sound points still hold:** E1 (the sample's mask), the `k!` placement, the block weight against `K`,
  K0 first.
- **The other file references** of revision 2 that I checked are right:
  - `tucker_tensor_train.py:63`, `:3564`;
  - `uniform_tucker_tensor_train.py:420`, `:484-485`;
  - `backend/fitting.py:809-815`, `:1166-1190`;
  - `t3toolbox/optimizers.py:132-229`;
  - `TARGET_DECISIONS.md:344-346`;
  - `data_preparation.py:566`.

## 6. Not verified

- T3Polynomial's code (whether its assembly or `κ_auto` calls a probe helper on packed data, which would now be
  refused).
- The Sphinx build with the `TypeVar`, the 3.9 floor, CI.
- The plan's re-counted test figures for K0 (I trust M1(d) as before).
- Selection against the product under jit (E7 ran on numpy); the cost argument is by construction.
