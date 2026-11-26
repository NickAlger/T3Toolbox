# M1 design note: `data_metric`, `blocks_to_weight`, the converters, the laws, the grammar

Implementer, 2026-11-22, at `64465d8b`. Read only. Sources: plan §2, §3, §4, 9.0 to 9.3, §8; plan reviews r3
(finding 3), r4 (minor points 1 to 3) and r7 (points 2 and 4).

## 1. Signatures and homes

All new public functions live in `backend/jets.py`, are listed in its `__all__`, and are re-exported by
`t3toolbox.fitting`. Each signature is written in the new grammar (§4).

```python
def data_metric(
        r:                typ.Any,                  # Y*, of the type of z, no K: probe len=d (or packed (d,)+…), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:           typ.Optional[typ.Any],    # ω, of the type of z (each axis 1 or z's size); None = ones
        minibatch_scale:  typ.Optional[typ.Any],    # γ, a number or a 0-d array (traced under jit); None = 1
) -> typ.Any:                                       # Y: γ * ω**2 * r elementwise; r's structure (a tuple for a tuple or a list)

def blocks_to_weight(
        kind:    str,                               # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],               # the jet shape; F = (n_1 + 1, ...)
        blocks:  typ.Any,                           # one value per block: probe (d,)+F, apply F, entries ()
        y:       typ.Any,                           # Y*, of the type of z: only its rank is read (C included)
) -> NDArray:                                       # ω: probe (d,)+F+(1,)*(len W+len C)+(1,); apply F+(1,)*(len W+len C); entries (1,)*(len W+len C)
```

- **`data_metric`'s private core** `_omega_squared(r_arr, ω_arr, xnp)` forms `ω*ω` in the lower floating precision
  of the pair. It is the one place `ω²` is formed from M2 on; in M1 the old `_squared_weights` stays beside it.
- **The method** `LeastSquaresBatch.data_metric(self, r)` is `jets.data_metric(r, self.weight, self.minibatch_scale)`.
  It lives in `backend/fitting.py`, which already imports `jets`.
- **`blocks_to_weight`** places the blocks on the leading jet axes (after the mode axis for probe), with size 1 on
  `W`, `C` and the edge.
  - It reads `len W + len C` from `y`'s rank: `ndim - m - 1` per ragged probe mode, `ndim - 1 - m - 1` packed,
    `ndim - m` for apply, `ndim` for entries.
  - It returns `blocks` reshaped, in their dtype and backend.
  - Structural errors raise a `ValueError`: blocks of the wrong shape (`(d,)+F` with `d` from `y`), or a `y` of too
    low a rank.

## 2. The edge cases of `data_metric`

The pairing rules:
- **A tuple or list `r`** pairs element by element with a tuple or list weight of the same length, or with an array
  weight along its first axis (`weight.shape[0] == len(r)`).
- **An array `r`** pairs with an array weight. That includes the packed probe, `(d,)+F+W+C+(N,)`.
- **Equal rank per pair**, else a `ValueError`. So a `K`-stacked `r` raises when a weight is given.
- **`weight=None`** gives `γ r` for any `r`, `K` included (r7 point 2).
- **Mode count:** a tuple weight of another length, or an array whose first axis is not 1 or `len(r)`, raises.

The two cases the plan left open:
- **(a) A packed `r` with a ragged tuple weight: I recommend that it raises** (`ValueError`, naming `pack_batch`).
  Reading the tuple mode by mode would need the padding `N` and the masks, which belong to the packing. A mixed call
  is a layout error, not a pairing.
- **(b) A packed weight whose mode axis has size 1** broadcasts over the modes. This is documented as a limit, in the
  plan's words: the checkers take the ragged form, and `pack_batch` builds packed weights from checked ragged ones.

The scale and the precision:
- `γ` is cast to the result dtype before it multiplies. Otherwise a float64 0-d `γ` would upcast a float32 `r` under
  numpy 2's promotion rules.
- `γ = None` skips the multiplication, so M2's engine calls with `None` are `ω² · r` as computed today.
- The dtype is the lower floating precision of `r` and `ω` (`_lowest_precision_dtype`). Dispatch is inferred from
  `(r, weight, minibatch_scale)`.
- The rank checks are static, so they run under `jit`.

## 3. The law helper, `tests/_typing.py`

It works on raw arrays and is not a test file. Its pieces:
- `JetLaw(order, ell)` for `s_j -> ℓ_j s_j`, with methods:
  - `P(x, lead=0)`: slot `k` × `ℓ^k`;
  - `P_star(x, lead=0)`: × `ℓ^-k`, the law of `c`, `M r` and `ω`;
  - `P_star_turned(x, lead=0)`: × `ℓ^-(n-k)`;
  - `inputs(ww)`: the input jets on their own `F_in_i`.

  `lead` is the number of axes before the jet axes (1 for a `d`-leading probe array). The constructor rejects an
  `ℓ_j` equal to 1, repeated values, and `order` all zero (r3: the law is vacuous there).
- `InputLaw(deltas)` for `w_i -> Δ_i w_i`, with methods:
  - `inputs` (`Δ_i ·`);
  - `V_star` (`Δ_i^-1 ·` on the edge: probe `z`, `y`, `r`, `Jp`);
  - `V` (`Δ_i ·`: probe `c`, `M r`, and `ω` with a full edge);
  - `tucker` (`B_i Δ_i^-1`), `tucker_dual` (`g_B Δ_i`);
  - `entries(z, index)` (`/ ∏_i Δ_i[n_i]`);
  - `ambient(cp)`, compared as the CP tensor, not factor by factor (r4 point 1).
- `assert_law(test, f, args, moved_args, move_out, rtol)`: compares `move_out(f(*args))` with `f(*moved_args)`,
  leaf by leaf, by relative error.

Laws checked in M1 (`tests/test_type_laws.py`):
1. **`data_metric`.**
   - Jet law: `r` by `P`, `ω` by `P_star`, the output by `P_star`.
   - Input law (probe, full-edge `ω`): `r` by `V_star`, `ω` by `V`, the output by `V`.
   - `γ` is invariant under both.
2. **The backend converters** (block weights until M2).
   - `jet_to_dual_jet` maps `P` (with `ω` by `P_star`) to `P_star_turned`.
   - With `weight=None` it maps `P_star` to `P_star_turned`: r4 point 2's two readings.
   - `dual_jet_to_jet` maps the other way.
3. **`rescale_direction` is the jet law.**
   - Its `(sample, y)` equal `(inputs(ww), P(y))`.
   - `t3_probe` and `t3_apply` at the moved sample equal `P` of the original outputs.
4. **`rescale_coordinates` and `t3_unscale_coordinates` are the input law,** with `Δ` the scales.
   - The moved `(sample, y)` equal `(inputs, V_star(y))` for probe; apply `y` is unchanged.
   - `t3_probe` at `(tucker(x), moved sample)` equals `V_star` of the original.
   - `t3_unscale_coordinates(tucker(x), scales)` returns `x`.
5. **Negative controls**, so the helper is shown to catch the faults it claims to catch:
   - `ω * r` (no square) fails the jet law;
   - a missing flip fails, and so does a flip on one axis only, at `order = (1, 1)`;
   - a jet/dual-jet swap fails.

Settings for the laws:
- orders `(2,)`, `(1, 1)` and `(2, 1)`, with `ℓ = (0.7, 1.9)`;
- `Δ_i` drawn in `[0.5, 2]`;
- `rtol` 1e-12 in float64.

## 4. The grammar (`signature_style.md`, micro-grammar)

The rule:
- **Data- and parameter-space signatures** write `# <type>, len=<n>, elm_shape=<…>`: the algebraic type first, then
  `len`, then the shape. The type names the element when `len=` follows (`P*⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)`),
  and the whole object otherwise (`P*, shape=F+W+[K]+C`).
- **The vocabulary** is 9.0's: `P`, `P*`, `V_i`, `V_i*`, `X`, `X*`, `Y`, `Y*`, `E`, `E*`, `ℝ`; `*` is the algebraic
  dual, and `[G]` marks a group present only for some settings.
- **Other signatures** keep "`len` first".

One example goes in the doc, `data_metric`'s own. Only M1's new signatures use the grammar; M5 types the rest.

## 5. Tests and their oracles

- **`data_metric`.**
  - Probe (ragged with a tuple weight, ragged with a `d`-leading array weight, packed), apply and entries.
  - `F`, `W`, `C` sizes taken distinct (`3, 4, 2`) and all equal (`2, 2, 2`).
  - Weights per slot, per sample, per element, and size-1 `C`.
  - The oracle is an explicit `np.ndindex` loop: `γ ω[idx with size-1 axes at 0]² r[idx]`. It contains no
    broadcasting.
  - Also: `weight=None`; `K`-stacked `r` (raises with a weight, gives `γ r` without); edge case (a) raises; a mode
    count mismatch raises; dtype float32 with float64 `ω` and float64 0-d `γ`; one argument missing gives a
    `TypeError`.
- **`blocks_to_weight`.**
  - Expected arrays filled by loops, on ragged and packed probe `y`, with and without `C`.
  - `data_metric(r, blocks_to_weight(…), None)` equals the old block weighting `_squared_weights(…) · r` bitwise.
    This is the bridge M2 relies on.
  - Bad block shapes raise.
- **The method:** on a hand-built batch with a full-form weight and a `minibatch_scale`.
- **9.2.**
  - `t3toolbox.fitting` no longer has `jet_to_dual_jet` or `dual_jet_to_jet`.
  - The frontend `dual_jet_to_jet` and O-20's safe-mode check are deleted, with `TestFrontendDualJetToJet`.
  - The frontend uses in `test_jets`, `test_dispatch` and `test_jax_absent` move to the backend names.
  - The module text of `fitting.py` is updated.
- **Dispatch:** `test_dispatch` jits `data_metric` and `blocks_to_weight` (numpy in, numpy out, and on jax).
  `test_jax_absent` calls both.

## 6. What stays unchanged

No existing function changes behaviour. The engine, the oracle, the checkers, `prepare_derivative_data` and the
converters' code are untouched; no caller is switched.

The gate is the full suite at `-n 2 -v`, with a node-by-node tally against M0's `-v` log (scratch, at the committed
tree). The only differences allowed:
- the new tests;
- the deleted `TestFrontendDualJetToJet`;
- `test_jets` / `test_dispatch` / `test_jax_absent` nodes whose frontend calls moved, with the same subtests or
  exactly the dropped frontend ones.

Then the module doctests, the pages, the examples and Sphinx.

## 7. Decisions I would ask for

1. **`_with_K`** (plan M1 row) has no caller until M2's engine. I recommend building it in M2, where the engine's
   reference tests exercise it. Built now, it would be dead code with a unit test only.
2. **Edge case (a): raise** (recommended), not read mode by mode.
3. **`LeastSquaresBatch.data_metric` with today's block weights** raises by rank (except where block and full form
   coincide) until M2 switches the batch to the full form. This is correct, since a block weight is not a weight of
   the type of `z`, but it means the method serves only full-form batches in M1. The alternative, converting blocks
   inside the method, would bring back the reading rule the plan removed.
4. **The frontend `jet_to_dual_jet`**: I remove it from `fitting`'s namespace entirely, not merely from `__all__`, so
   "frontend-private" means not reachable.
