# K1: the reviewer's first look, before the design note

Reviewer 2 (Claude), 2026-11-25. A working note, not a review. It was written from `PACKING_PLAN.md` §2, §4.3, §5 and §7
"K1", from `DATA_METRIC_PLAN.md` §1-4, and from the code at `HEAD` + K0 (the engine's data-space steps in
`sampling_steps.py`, the weight functions of `jets.py`, `backend/fitting.py`'s batch and oracle, the reference's
`_weight_squared`), before I saw K1's design note. The files the implementer is writing in this folder were not read.
No test was run.

## 1. Where K1 can go wrong

**1a. The two weights, mathematically.**
- `M = γ b² f²`: the block weight `b` is left-aligned on the leading axes, the full weight `f` is elementwise.
  - Left-aligned means: for a probe `r` packed as `(d,)+F+W+[K]+C+(N,)`, `b` of shape `(d,)+F` reshaped to
    `(d,)+F+(1,)*rest`; for a tuple `r`, row `i` (shape `F`) against `r[i]`; for apply, `F`; for entries, `()`.
  - The classic slip is numpy's right alignment. For example `b` of shape `F` against an apply `r` `F+W` with
    `F == W` in size: it broadcasts silently, onto the wrong axes. The tests need a case where the sizes of `F` and
    `W` (and `d` and `F`) coincide, so that a right-aligned or transposed placement gives a wrong number, not an
    error.
- **The engine's `K`.**
  - `_with_K` stays for the full weight only. The block weight needs no placement, since `K` sits behind `F` (the
    plan's argument). Check that the engine's `Jp` path (`Jp_norm_sq_step`, `_Jp.w_tilde`) applies `b` to an `r` that
    carries `K`, with `b` still left-aligned.
  - Test it with `K = (2,)` and the probe `d = 2 = n + 1` coincidence.
- **The `k!` of derivative coordinates** goes into the block weight only.
  - `derivative_block_weight_to_taylor(order, b)` multiplies the trailing `m` axes of `(d,)+F` or `F`.
  - With `order = ()` it is the identity, the probe `(d,)` table untouched. `F = (1,)` slots are `0!`.
  - Today's `_taylor_weight` right-aligns (a real bug class in this code, `DATA_METRIC_PLAN.md` §4); the new
    function must align on the table's own `F`, and be checked against `k! = k_1!...k_m!` written out, with `m = 2`.
- **`prepare_*_derivative_data(block_weight=None)`** returns the `k!` table (weights of 1.0 in the derivatives);
  check it against the old function on old inputs.

**1b. The rule of a size of 1 only on stack axes.**
- **`y` exact on `W`** (its `C` may be 1). The full weight is exact but for `W` and `C` axes of 1. The structural axes
  (mode, `F`, edge) are exact for both.
- The checks live in four places, which must agree: the engine; `jets.check_data` (edited in K1, replaced in K3);
  the batch's constructor and `check`; `pack_batch`.
- One table of shapes, run through all four, would show any disagreement. One rejection a test should hold: an edge
  of 1 in a full weight, legal before K1 (a weight constant over the edge) and illegal after.
- **The mode-agreement rule:** one pattern of size-1 stack axes for all `d` arrays of a probe full weight (and of
  `y`'s `C`); the refusal names `np.broadcast_to`.
- **The metric's view of the middle axes.** `probe_data_metric` sees `W`, `K` and `C` without knowing which is which,
  so each may be 1 or `r`'s. A full weight without `K` against a `K`-stacked `r` then raises by rank; check it does,
  rather than broadcasting through a coincidence.

**1c. The battery against the tree before K1.**
- Each old block weight (through `blocks_to_weight`) is passed as `block_weight`, each old full-form weight as
  `full_weight`.
- The plan's tolerances: bitwise where the arithmetic is unchanged, to rounding where `b² f²` regroups `ω²`.
- **Risk: a battery that compares the engine with itself across the change.** The independent anchor must be the
  dense reference, which today forms `ω²` itself (`sampling_definitions._weight_squared`, importing no `jets`).
  K1's step (1) changes the reference first. It must keep the reference forming `b² f²` with its own code, never
  through `jets.*_data_metric`, or a shared alignment bug cancels.

**1d. The batch (Part 0, the refactor with a bitwise battery).**
- **Static fields in a pytree.**
  - `kind` and `order` go in the aux data, normalized to Python ints, since jax hashes the aux and a numpy int in
    `order` would still hash but compare oddly across rebuilds.
  - The unflatten runs the constructor. Its checks may read only the static fields: jax rebuilds with placeholder
    leaves (`object()`, `None`, tracers) in `tree_map`, `vmap` specs and the like.
  - A test should rebuild a batch with `object()` leaves (`jax.tree_util.tree_unflatten(treedef, [object()] * n)`)
    and show it does not raise.
- **The compilation key moves from the oracle to the batch.** `LeastSquaresOracle()` has no fields. Check that the
  optimizers' compile-once tests (inner CG once per run; the exact model once across points) still hold. Two batches
  of different kinds must not share a program; that is safe now, since the kind is aux.
- **Positional calls.**
  - On Python 3.9, `kw_only` is out. If the fields after `y` are to be keyword-only, a custom `__init__` on a frozen
    dataclass (`object.__setattr__`) or a `*`-argument factory is needed. Both must survive the pytree unflatten
    and `dataclasses.replace` (which calls `__init__` with keywords).
  - An old `LeastSquaresBatch(ww, y, w, scale)` must fail at once, on its kind, and not land `y` in `order`. The
    error message should say what the new form is.
- **The rewrite of about 104 `LeastSquaresOracle(` and 76 `LeastSquaresBatch(` calls.**
  - The reviewer's check is an `ast` comparison, call by call: the old keyword or position of each argument against
    the new field it lands in. A weight that went through `blocks_to_weight(kind, order, blocks, y)` must become
    `block_weight=blocks`.
  - A wrong choice is mostly caught structurally (the full weight's edge must be exact, and `blocks_to_weight` gives
    an edge of 1), except for apply: `blocks_to_weight`'s apply output `F+(1,)*rest` is also a legal full weight,
    and gives the same numbers. Harmless numerically, but a test meant to exercise `block_weight` would exercise
    `full_weight` instead.
- **The `Oracle` protocol's changed contracts**: the default draw (`flat_draw`, `_require_draw`) now calls
  `batch.take` and `batch.n_measurements`. A user oracle with its own batch type and no `draw=` must get a clear
  error, not an `AttributeError` deep in the optimizer.
- **The display** reads `order` off the batch, and refuses a validation batch of another kind or order. The silent
  positional labels (the critic's E8b) should have a test.

**1e. `take` and the sums.**
- `take`: the block weight passes through; the full weight with every `W` axis 1 is kept; otherwise it is broadcast
  and gathered like `y`.
  - A multi-axis `W` with one axis 1 and one full is the case that splits the branches; it needs a test.
  - M-arc decision 18's raise (an object with no `W` axes) stays.
- **The oracle's `<y, M y>` and `y_block_sumsq`.** A `y` whose `C` axes are 1 is counted over `z`'s shape (13(c)).
  With a full weight whose `C` is full and `y`'s `C` 1, `y` must be broadcast before the product, or the sum
  undercounts. Check against the dense reference with `C = (2,)`.
- `f_resolution`: `γ` once.

**1f. The uniform layer in K1** (before K5's selection masking).
- `y` and the full weight are always masked on the uniform layout (edge `N`), the block weight never. Until K5 this
  is masking by a product, so the padding of the data must be finite in K1's tests. `inf` is K5's.
- `pack_batch` packs the probe `y` and full weight with `pack_vectors` (still in `ut3_operations` until K2). The
  modes must share the pattern of size-1 axes, refused otherwise.
- `pack_data` and `_pack_data_space` go. `pack_sample` stays until K5.

**1g. What K1 removes, and its callers.** `blocks_to_weight` has 3 library calls, 48 in tests, 7 library doctests and
2 page doctests. `check_weight`, `pack_data` and `_pack_data_space` go. `data_metric` is split into three functions.
The translation of the old call form passes the old `omega` matrix as `block_weight`; its 19 pre-P16 pins must stay
(bitwise or to rounding, as stated). Doctests changed in K1 need output pasted from a real run.

## 2. Checks I intend to run on the note, then on the build

- On the design: the shape tables of §1b through the four checkers, in a scratch copy, once built. A dense
  computation of `M r` with coinciding sizes (`F`, `W` and `d` equal), against each metric function. The `k!`
  function at `m = 2` against a hand table.
- On the build:
  - `unitdiff.py` of every kept function;
  - an `ast` map of every rewritten `LeastSquaresBatch(` / `LeastSquaresOracle(` call (old argument to new field);
  - the pytree unflatten with placeholder leaves; positional old-form calls;
  - mutations: a right-aligned block weight; the square dropped in one of the two weights; `take` with a
    partially-1 `W`; the `k!` on the wrong axes; `y`'s `C` broadcast skipped in `<y, M y>`;
  - the per-test tally against the tree before K1 (`tally2.py`).
