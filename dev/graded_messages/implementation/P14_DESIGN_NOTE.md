# P14 design note: the oracle and the batch

Implementer, 2026-11-21, against `graded-messages` at `d1a4335c` (P13 complete). Read-only so far. P14 is the plan's
P14 (section 5.3), with section 2.2's sketches (`LeastSquaresBatch`, the `Oracle` protocol, `LeastSquaresOracle`,
`pack_batch`) and O-7, O-8, O-18, O-21, O-29, O-30, O-44 and O-46. It is additive: the old kinds, `Problem` and the
optimizers are untouched until P16.

**`point_value_and_gradient` is P17's, not P14's.** O-50 and N47 adopt it; the plan's text places it in P17 (the
paragraph at line 2184, beside `point_objective`). I leave it there.

## 1. The oracle's operation

```python
@dataclass(frozen=True, eq=False)
class LeastSquaresOracle(ValueHashedFields):     # backend/fitting.py, beside the old kinds
    kind:   str                                  # 'probe' | 'apply' | 'entries'
    order:  Tuple[int, ...]                      # required (O-44); entries: () only (O-46), checked in __post_init__
    optional_outputs: ClassVar = ('Kp', 'Jp', 'r', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution')

    def query(self, frame, batch, want, Vc=None, variations=None, core_directions=None, held=None)
            -> (values aligned with want, held: HeldVariables)
```

`query` is one engine call: `sampling_variables(kind, frame, batch.sample, want', order, y=batch.y,
weight=batch.weight, variations=..., core_directions=..., Vc=Vc, held=held)`. Around that call:
- **Outputs the engine does not compute:**
  - `y_block_sumsq` needs no frame: the plain per-(mode, slot) sums of squares of `y` over `W`, the shape of
    `r_block_sumsq`, through `data_preparation.data_block_norms(kind, order, y) ** 2` (one reducer for both);
  - `f_resolution` is `eps (|w y| |w r| + |f|)`, with `|w r| = sqrt(2 f)` (the misfit only), `|w y|` from the same
    reducer with the weight, and `eps` from the dtype as `_resolution_eps` reads it. It needs `f`, which `want'`
    then includes.
- **The minibatch scale (O-21)** multiplies `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` and `f_resolution` by
  `batch.minibatch_scale` (`None` is 1), and nothing else. `r`, `Jp`, `r_block_sumsq` and `y_block_sumsq` stay plain.
  The scale reaches only the oracle's outputs: the regularizer's scaling and the draw that sets the scale are P16's.
- **The direction on a corewise frame.** On the point `(B, G)`, told from its structure (ragged two stores, uniform
  four elements), the oracle passes the direction it is given (`variations`) to the engine as `core_directions`. The
  engine refuses `variations` there (P8). Section 4, item 2, proposes the structure reader.
- `sum_over_samples` is always `True`: the optimizers want summed blocks.

## 2. What it holds

The hold list is fixed for Part 1: `xi`, `mu`, `nu`, `eta`, `r`. They are public names on both graphs (checked:
`_made` of all six graphs). `held` comes back as a `HeldVariables`:
- the `held` passed in;
- plus each name on the hold list that **this call computed anyway**.

No name is computed only to be held (8.3). That needs the call's closure, which today only the engine knows (section
4, item 1). Two consequences:
- `held` grows across calls at one frame: `f` holds `r`, `xi`, `mu`, `nu`, `eta`, and a later `g` reuses them.
- `held` is valid for one frame **and one batch**. A held `r` belongs to its batch's `y`, so an optimizer must not
  carry `held` across draws. The docstring says so; the optimizers already rebuild the model per draw.

## 3. The batch, `take`, `check_batch` and packing

`LeastSquaresBatch` is section 2.2's dataclass, a registered pytree with every field a leaf (`None` allowed).
- `n_measurements(batch)` is `prod(W)`, with `W` read after the `len(order)` jet axes of `sample[0]` (or after the
  mode axis of `index`). On a packed sample it is read after the leading `d`.
- `take(batch, indices)` gathers a flat `W` subset from the sample, `y` and nothing else (the weight has no `W`). It
  is traceable, with a static `len(indices)` (the existing `_flat_gather`). The jet axes come before `W`: at offset
  `m` ragged, `1 + m` packed, and probe `y` likewise. **I propose `take` sets both `drawn_indices = indices` and
  `minibatch_scale = n / len(indices)`** (times any scale already there), so a drawn batch is unbiased by
  construction. A draw (P16) then only chooses the indices.
- `check_batch(batch, shape)` calls `check_sample(kind, order, sample, shape)`, `check_data(kind, order, sample, y)`
  and `check_weight(kind, order, weight, len(shape))`. It is a checker on the ragged form: `query` does not call it
  (O-18, O-29).
- `pack_batch(kind, order, batch, N)` (`backend/uniform_fitting.py`) packs the sample with
  `pack_if_ragged(sample, N, len_J=len(order))`, which brings the jet sizes to one set across the modes (zero
  coefficients, O-45), and probe `y` with `pack_if_ragged(y, N)`. Entries' `index` and apply's `y` pass through.
  **I propose `pack_sample` and `pack_data` keep their old signatures in P14**: the old uniform kinds call them with
  the kind names `'probe_derivatives'` and others. P14 is additive, so `pack_batch` uses `pack_if_ragged` directly,
  and the two get the plan's `(kind, order, ...)` forms when the old kinds go (P16).

**Reachability:** `LeastSquaresOracle` and `LeastSquaresBatch` are re-exported from `t3toolbox.fitting` and listed in
its `__all__`. The `Oracle` protocol goes in `backend/optimizers.py` as documentation (a `typing.Protocol` with
`optional_outputs`; `check_batch` is described as optional, O-29).

## 4. What I would change (two engine helpers, both small and structural)

1. **The engine reports what it computed.** I propose a keyword `keep: Sequence[str] = ()` on
   `sampling_variables`.
   - **What it returns.** When `keep` is given, the return has one more element: a `HeldVariables` of the `held`
     passed in, plus every name in `keep` that the call's closure computed.
   - **Why the engine.** Today the closure (`_closure`) is private. The oracle could compute it a second time through
     a public `computed_names(...)`, but that repeats the engine's reading of the facts (call type, `present`,
     `held:Jp`), which is easy to get wrong. By the razor a raw-data user doing repeated calls wants the same thing.
   - **The alternative**, if you prefer the engine's return form unchanged: a public `computed_names(kind, frame,
     want, held=None, **arguments)` that returns the closure's names, and the oracle asks `want + those names`.
2. **One structure reader.** A public `sampling_variables.structure(frame) -> ('ragged' | 'uniform', 'manifold' |
   'corewise')`, the engine's own reading (layout by `is_ndarray(frame[0])`, then the count). The oracle uses it to
   pass the direction as `core_directions` on a corewise frame, and the engine uses it too: one reader, no second
   count.
3. **`take` sets the scale** (section 3).
4. **`pack_sample` and `pack_data` stay as they are until P16** (section 3).

## 5. Tests (a new `tests/test_least_squares_oracle.py`, numpy only, and `test_dispatch` rows)

- **Each output of `query`:**
  - against the engine and against `sampling_definitions`, at `()`, `(2,)` and `(1, 1)` (entries `()`);
  - on a manifold frame and a corewise point, ragged and uniform (`pack_batch` first);
  - `y_block_sumsq` and `f_resolution` by their formulas.
- **`held`:**
  - it is exactly the hold-list names this call's closure computed (none extra), merged with the `held` passed in;
  - a second call with it returns the same values;
  - asserted structurally, by the closure.
- **The direction on a corewise frame:** it is passed as `core_directions` (`JtJp`, `Jp`, `Jp_norm_sq` against the
  engine's raw route), and `Kp` holds on both call types.
- **The minibatch scale:**
  - on a small `n`, enumerating every subset of the batch size, the mean over subsets of the drawn `f` (and `g`)
    equals the full `f` exactly;
  - on a drawn batch, `r`, `Jp`, `r_block_sumsq` and `y_block_sumsq` are unscaled;
  - `f = 1/2 <r, r>`, `<Jp, r> = <p, g>` and `Jp_norm_sq = <Jp, Jp>` hold in the batch's inner product.
- **`take`:**
  - under `jax.jit` with a static size, ragged and packed, the jet axes skipped;
  - it sets `drawn_indices` and the scale.
- **`check_batch`:** passes a good batch, and raises on each wrong part (sample form, `W`, a mode size, `y`, the
  weight).
- **`pack_batch`:** then the uniform query equals the ragged one on real parts, with mixed input jet sizes.
- **The records:**
  - the batch's pytree round trip;
  - the oracle's hash and equality by value;
  - `LeastSquaresOracle('probe')` raises `TypeError` for every kind;
  - `('entries', order=(1,))` raises.
- **The bridge:** against the old derivative kinds through the conversions (`affine_jet(ww, pp)`,
  `derivative_weight_to_taylor_weight`): `f`, `g`, `JtJp` and `Kp`.
- **The docstring:** the trap of an input jet given with `order=()`, wrong against right.
- **`test_dispatch`:** `query` jitted on both layouts (jax in, jax out), with `held` carried through the program.

Gate: the full gate.
