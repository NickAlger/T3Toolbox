# Uniform weighting layer — design note

_Started 2026-07-15. The uniform mirror of the (shipped) ragged weighted layer. Ships `UT3Weights` /
`UT3FrameWeights` (weight supercores + boolean masks) + the masked ops, so the weighted layer runs on the
jit/GPU uniform representation. **The lever: the ragged layer is the equivalence oracle** — `to_uniform →
op → to_ragged == op_ragged`. **The difficulty: the boolean masks** (garbage padding must not leak; concat/
kron go gappy)._

## 0. Resume (read first)

The ragged weighted layer is shipped: `T3Weights` (a tensor's edge weights) + `T3FrameWeights` (a metric on
a tangent's variation coordinates) + `absorb` / `weighted_norm` / `weighted_inner` / `concatenate` /
`kronecker` / `from_t3svd` / `from_t3weights`, backend (`t3_*` / `fv_*`) + frontend. This note mirrors all
of it onto the **uniform** representation (`(d,)+stack+(…)` supercores + HOST-boolean prefix edge masks;
`docs/uniform_*`). A uniform weight is "the ragged weight, padded for performance," and it **carries the
edge masks** (Nick's decision) so operations can check mask-compatibility and skip garbage.

## 1. The uniform weight objects (data = weight supercores + masks)

Mirror `UniformTuckerTensorTrain` / `UT3Frame`:

- **`UT3Weights`** (in `uniform_tucker_tensor_train.py`): `tucker_weight_supercore` `(d,)+stack+(n,)`,
  `tt_weight_supercore` `(d+1,)+stack+(r,)`, plus a `UT3Masks`-style holder — **the same two edge masks as
  the `UniformTuckerTensorTrain` it weights** (`tucker_edge_mask (d,)+stack+(n,)`, `tt_edge_mask
  (d+1,)+stack+(r,)`), because a weight's edges *are* the tensor's edges. Reuse `UT3Masks` directly.
- **`UT3FrameWeights`** (in `uniform_frame_variations_format.py`): four weight supercores
  `up/down/left/right`, each `(d,)+stack+(rank,)`, plus a `UT3VariationsMasks`-style holder (the four
  variation edge masks `nU/nD/rL/rR`) — the metric weights the variation coordinates, so it shares the
  variations' masks.

Both are frozen dataclasses; the mask holder is a `ValueHashedMasks` **static aux** (value hash/eq over
mask content), supercores are the data/leaves — the `UniformTuckerTensorTrain`↔`UT3Frame` pytree pattern
(masks fold into the compiled program as host constants; a rebuilt-identical holder is the same jit key).

## 2. Mask-safety (the whole ballgame)

Every op **masks on entry** through the layer's chokepoint (`ut3_apply_masks`-style: multiply the supercore
by the numpy edge masks, zeroing garbage), exactly as the existing uniform ops do. Then:

- **`absorb`** and **norm/inner** are *reductions/contractions*, so garbage padding **must** be zeroed
  first (a `1e6` in a padded slot would otherwise dominate). Mask on entry → garbage is inert.
- The masks are **host numpy** (`np`, never `jnp`) — a traced mask breaks the layer
  (`docs/contributor/uniform_pytree_composition.md`); mask logic (build, concat/Kronecker, `int(mask.sum())`)
  runs on the host, only the supercores flow through `xnp`.

## 3. Operations (mirror the ragged; add mask handling)

Backend `ut3_*` (base point) / `ufv_*` (tangent), frontend on the classes/`UT3Tangent`, matching the ragged
split (`t3_*`/`fv_*`):

| op | supercore action (masked) | mask action |
|---|---|---|
| `ut3_absorb_weights` | the ragged einsum, `(d,)`-leading, mask on entry (TT weights leftward, Tucker→Tucker cores) | **unchanged** (absorb is shape-preserving) |
| `ufv_absorb_weights` | absorb into variation supercores (`down`→V, `up`/`left`/`right`→H), masked | unchanged |
| `*_weighted_norm` / `_inner` | absorb → masked `corewise` stack-norm/dot (reuse the uniform corewise reductions) | unchanged |
| `*_concatenate_weights` (for `+`) | supercore **grows** to the new max rank (ranks add); concat the real weight vectors | masks **concatenate** (may go **gappy**) |
| `*_kronecker_weights` (for `⊙`) | supercore grows to the new max rank (ranks multiply); Kronecker the real weight vectors | masks **Kronecker** (gappy — strided, per `docs/uniform_masks_vs_ranks.md`) |

`concat`/`kron` are the hard ones: they change ranks, so the padded supercore re-sizes and the masks
transform (concat/Kronecker) — the `docs/uniform_masks_vs_ranks.md` `+`/`×` mask algebra, reused verbatim
(the object's add/multiply already does this; the weight is the vector version). `absorb`/`norm`/`inner`
keep the mask unchanged (shape-preserving / reducing).

## 4. Conversions + constructors

- **`to_uniform` / `to_ragged`** for both weight types (mirror `UniformTuckerTensorTrain.from_t3` /
  `.to_t3`: pad the ragged weight vectors to the max rank, build the prefix masks; and the reverse, reading
  the real prefix through the masks). These are the equivalence-test bridges.
- **`from_t3svd`** (uniform): uses the uniform `ut3_svd` singular values → a `UT3Weights` (prefix masks).
- **`from_t3weights`** (uniform `UT3FrameWeights`): the same slicing (`up=down=tucker`, `left=tt[:-1]`,
  `right=tt[1:]`) on supercores + masks — the TT-bond mask slices to the up/left/right/down variation masks.

## 5. Testing (equivalence oracle + mask-strictness — the required trio)

Per `docs/contributor/testing_strategy.md` (dense/ragged alone is **blind to too-permissive masks**):

1. **Equivalence to ragged**: `to_ragged(op_uniform(to_uniform(x), …)) == op_ragged(x, …)` on real parts,
   across structures × stacks (incl. **variable ranks per stack element** — the determinantal variety).
2. **Garbage-robustness**: write **large-finite** garbage (`1e6·(1−mask)`) into padding; assert
   `absorb`/`norm`/`inner` unchanged (the phantom-rank tripwire — NB the layer masks by *multiply*, so use
   large-finite, not NaN).
3. **Exact output masks** (derived non-circularly): `absorb`/`norm`/`inner` keep the input masks;
   `concat`/`kron` produce the **concatenated / Kronecker** masks (assert the gappy pattern, not just shape).
4. **jit** dispatch (masks stay host; a stray `jnp` on a mask → tracer → caught).

## 6. Slices

1. **S1 — `UT3Weights` + `ut3_absorb_weights` + `ut3_weighted_norm`/`_inner` + `to_uniform`/`to_ragged`.**
   The base-point layer. Equivalence + garbage + mask tests.
2. **S2 — `ut3_concatenate_weights` / `ut3_kronecker_weights`** (the gappy-mask ops) + `from_t3svd`.
3. **S3 — `UT3FrameWeights` + `ufv_absorb_weights` + uniform tangent `weighted_norm`/`inner` + `from_t3weights`.**
4. **S4 — frontend wiring** (`UT3Weights`/`UT3FrameWeights` classes, methods, `absorb_weights`,
   `UT3Tangent.weighted_norm`/`inner`/`absorb_weights`) + dispatch inference (ragged vs uniform from the arg).
5. **S5 — docs** (extend `weighting.md` with the uniform mirror; note in the uniform docs).

## 7. Watch-list (the traps)

- **Gappy masks after concat/kron** — assert the exact strided pattern, not just the rank count. The
  supercore re-size + host-mask Kronecker is the most error-prone code.
- **Mask on entry everywhere** — absorb/norm/inner are reductions; forgetting the entry-mask leaks garbage.
- **Masks stay host numpy** (`np`), supercores `xnp`; the `int(mask.sum())` rank extraction must not see a
  tracer (jit chokepoint).
- **Variable ranks per stack element** — the determinantal variety; tests must vary ranks across the stack,
  not just use one rank for the whole stack.
- The ragged oracle makes all of this checkable; lean on it hard.
