# Uniform weighting layer — design note

_Started 2026-07-15. The uniform mirror of the (shipped-to-main, unreleased) ragged weighted layer. Ships
`UT3Weights` / `UT3FrameWeights` (weight supercores + boolean masks) + the masked ops, so the weighted layer
runs on the jit/GPU uniform representation. **The lever: the ragged layer is the equivalence oracle** —
`to_uniform → op → to_ragged == op_ragged`. **The real risks** (revised post-S2): *not* the boolean masks —
their algebra is trivial and closed (§3) — but **ordering/axis** mistakes (`np.kron` vs a broadcast last-axis
outer product) and the **padding hazards** (`1/0 = inf`; a mask mismatch only uniform can hide).
**Design review 2026-07-15 (Nick) settled §8's decisions and added S0** — the ragged layer must be fixed
first, because a wrong oracle certifies a wrong mirror._

## 0. Resume (read first)

The ragged weighted layer is on `main` (NOT in `v2026.0.0` — every weighted commit postdates the tag, so it
is **unreleased and freely changeable**): `T3Weights` (a tensor's edge weights) + `T3FrameWeights` (a metric
on a tangent's variation coordinates) + `absorb` / `weighted_norm` / `weighted_inner` / `concatenate` /
`kronecker` / `from_t3svd` / `from_t3weights`, backend (`t3_*` / `fv_*`) + frontend. This note mirrors all of
it onto the **uniform** representation (`(d,)+stack+(…)` supercores + HOST-boolean prefix edge masks;
`docs/uniform_*`). A uniform weight is "the ragged weight, padded for performance," and it **carries the edge
masks** (Nick's decision) so operations can check mask-compatibility and skip garbage.

**Start at S0** (§6): the design review found the ragged frame-weight *stack model* is wrong, and the ragged
layer is this build's oracle. *(S0/S1/S2 are now done — see `dev/HANDOFF.md`. §3's "these are the hard ones"
framing was corrected post-S2: the combines are trivial; the risks are elsewhere.)*

## 1. The uniform weight objects (data = weight supercores + masks)

Mirror `UniformTuckerTensorTrain` / `UT3Frame`:

- **`UT3Weights`** (in `uniform_tucker_tensor_train.py`): `tucker_weight_supercore` `(d,)+C+(n,)`,
  `tt_weight_supercore` `(d+1,)+C+(r,)`, plus a `UT3Masks` holder — **the same two edge masks as the
  `UniformTuckerTensorTrain` it weights** (`tucker_edge_mask (d,)+C+(n,)`, `tt_edge_mask (d+1,)+C+(r,)`),
  because a weight's edges *are* the tensor's edges. Reuse `UT3Masks` directly.
  **No `shape` field** (decision §8.4): weights live only on *internal* edges — there are no physical `Nᵢ`
  legs (external weights were scoped out of the ragged design). So `UT3Weights` is a **3-field** class
  (`tucker_weight_supercore, tt_weight_supercore, masks`), unlike UT3's four, and its `.data` is
  `(tucker_weight_supercore, tt_weight_supercore, (tucker_edge_mask, tt_edge_mask))`.
- **`UT3FrameWeights`** (in `uniform_frame_variations_format.py`): four weight supercores `up/down/left/right`,
  each `(d,)+C+(rank,)`, plus a `UT3VariationsMasks`-**shaped** holder (the four variation edge masks
  `nU/nD/rL/rR`, each `len=d`) — **also at `C`** (decision §8.5). Frame-like: the weight carries the *frame's*
  stack `C`, not the variations' `K+C`. No `shape` field.

Both are frozen dataclasses; the mask holder is a `ValueHashedMasks` **static aux** (value hash/eq over mask
content), supercores are the data/leaves — the `UniformTuckerTensorTrain`↔`UT3Frame` pytree pattern (masks
fold into the compiled program as host constants; a rebuilt-identical holder is the same jit key).

## 2. Mask-safety, and the masking/weighting wall

**The wall (Nick, 2026-07-15).** Masking and weighting are *conceptually distinct* and must stay so in code,
even where the mechanics coincide — masks are boolean **structure** (static aux, non-differentiable, define
rank, value-neutral); weights are float **parameters** (traced leaves, differentiable, scale the contraction,
value-affecting). This is the same distinction `docs/contributor/uniform_rank_masks_rationale.md` invokes to
reject "a float mask doubling as edge weights." So:

> **Weighting code never calls a `*_apply_masks` / `*_make_masks` function.** Where the mechanics genuinely
> coincide, **break the shared core out into a neutral subfunction and call it from each side** — do not have
> the weighting path call the masking path.

**The wall is nearly free, because `absorb` needs no masking at all.** `absorb` is a **pointwise scale along
each edge axis**, not a reduction: output slot `i` depends only on the core's slot `i` and the weight's slot
`i`, so garbage never mixes into a real slot. It is **garbage-transparent** — garbage in the padding flows to
garbage in the padding, which the contract declares don't-care. *(This corrects the pre-review §2, which
asserted "absorb and norm/inner are reductions, so garbage padding must be zeroed first." True of norm/inner;
**false of absorb.** The claim is load-bearing — a reader would implement an unnecessary entry-mask against
it — so it is also corrected in the docs at S5.)* Precedent: `utv_to_ut3` deliberately does not mask its
inputs either (`docs/contributor/testing_strategy.md`).

So the schedule is:

| op | needs an entry mask? | why |
|---|---|---|
| `absorb` | **no** | pointwise along edge axes; garbage-transparent |
| `weighted_norm` / `weighted_inner` | yes — **but not in weighting code** | they reduce to `absorb` → the *existing* plain `ut3_norm` / `ut3_inner_product`, which mask on entry themselves. We call the **norm op**, not a masking op. |
| `reciprocal` / `sqrt` | **yes** (§8.1) | the one genuine touch point — see below |
| `concatenate` / `kronecker` | no | the masks *transform*; the supercores are copied/outer-product'd, not reduced |

**The two genuine touch points** (where the wall's "break out a subfunction" directive applies):

1. **`reciprocal` / `sqrt`** must be **mask-aware** (§8.1). `1/0 = inf` on zeroed padding, and then
   `0 × inf = nan` poisons every masked reduction downstream — the exact hazard
   `docs/uniform_equivalence_contract.md` names ("the fill must be finite"). This hits the *headline* use
   case: the Grasedyck–Kramer metric **is** `from_t3svd(x).reciprocal()`. Fix: mask on entry, then compute
   with a double-`where` so padded slots come out `0`, never `inf`/`nan` (grad-safe, canonical clean-zero
   padding preserved). No masking *function* is called — the op is a one-line elementwise `where` against the
   weight's own mask array. **This is also the argument that uniform weights must carry masks at all: a
   maskless weight cannot implement `reciprocal` correctly.**
2. **`to_uniform`** must build prefix masks from the ragged weight's ranks — which is what `ut3_make_masks`
   does. Per the wall: extract the neutral primitive (`np.arange(pad) < ranks[..., None]`) and call it from
   **both** `ut3_make_masks` and the weight's mask builder, rather than the weight layer calling
   `ut3_make_masks`.

**Masks stay host numpy** (`np`, never `jnp`) — a traced mask breaks the layer
(`docs/contributor/uniform_pytree_composition.md`); mask logic (build, concat/Kronecker, `int(mask.sum())`)
runs on the host, only the supercores flow through `xnp`.

## 3. Operations (mirror the ragged; add mask handling)

Backend `ut3_*` (base point) / `ufv_*` (tangent), frontend on the classes/`UT3Tangent`, matching the ragged
split (`t3_*`/`fv_*`):

| op | supercore action | mask action |
|---|---|---|
| `ut3_absorb_weights` | the ragged einsum, `(d,)`-leading, **no entry mask** (TT weights leftward, Tucker→Tucker cores) | **unchanged** (absorb is shape-preserving) |
| `ufv_absorb_weights` | absorb into variation supercores (`down`→V, `up`/`left`/`right`→H); the `C` weight **broadcasts over `K+C`** for free via the right-aligned `'...'` (works because `C` is innermost) | unchanged |
| `*_weighted_norm` / `_inner` | absorb → the existing plain uniform norm/inner (they mask on entry) | unchanged |
| `reciprocal` / `sqrt` | elementwise, **mask-guarded** so padding stays `0` (never `inf`/`nan`) | unchanged |
| `*_concatenate_weights` (for `+`) | supercore **grows** to the new max rank (ranks add); concat the real weight vectors | masks **concatenate** (may go **gappy**) |
| `*_kronecker_weights` (for `⊙`) | supercore grows to the new max rank (ranks multiply); Kronecker the real weight vectors | masks **Kronecker** (gappy — strided, per `docs/uniform_masks_vs_ranks.md`) |

`concat`/`kron` change ranks, so the padded supercore re-sizes and the masks transform — the
`docs/uniform_masks_vs_ranks.md` `+`/`×` mask algebra. `ut3_add` is the working precedent for the concat
side (it concatenates both rank masks on the host). `absorb`/`norm`/`inner` keep the mask unchanged.

**These are not the hard ones — the mask algebra is trivial and closed** *(corrected 2026-07-15, post-S2;
the pre-review plan billed them as the difficulty and that was wrong)*. Treat the mask as just another
weight holding 0s and 1s: the contract is `weight_AB * mask_AB == combine(weight_A * mask_A, weight_B *
mask_B)`, and it is satisfied by combining the weights and combining the masks **the same way** — because
both concatenation and the Kronecker product commute with elementwise multiply (`kron(a∘p, b∘q) =
kron(a,b) ∘ kron(p,q)`, the mixed-product property; nothing special to booleans). So each op is one
operation applied twice, with no mask cleverness at all. That closure *is* the argument for boolean masks
over integer ranks (`docs/contributor/uniform_rank_masks_rationale.md`) — an integer rank cannot even
express the strided result.

The genuine risks are elsewhere, and both are ordering/axis mistakes rather than mask logic:
1. **`kron` must be a last-axis outer product broadcasting the shared `(d,)+stack` prefix — NOT `np.kron`**,
   which would Kronecker the mode/stack axes too. The ragged build hit exactly this.
2. **A-major must agree with any core-combine that pairs with it** (in uniform there is none yet — no
   `ut3_mult`).

The output masks *are* gappy (concat leaves a hole wherever an input had rank slack; kron's real set
`{a*nB + b : mask_A[a] and mask_B[b]}` is strided over the **padded** width, so even two prefix inputs give
holes). That is a description of the result, not a difficulty: it costs the combines nothing, and only
obliges *consumers* to read the mask instead of slicing a prefix. It cannot be flattened: slot `a*nB + b`
with `b >= rank_B` holds `wA[a] * <padding>`, so a prefix mask of rank `rA*rB` would claim padding as real
data — the phantom-rank bug (killed by mutation test, S2).

**`kronecker` is unpaired in uniform, and we build it anyway** (§8.2): there is **no `ut3_mult`** — the
uniform layer has `ut3_add` but no Hadamard product. So `ut3_concatenate_weights` has a real partner
(`ut3_add`) while `ut3_kronecker_weights` is a weight-only primitive with no uniform core-combine to pair
with, and the A-major index-pairing consistency the ragged note flags as the silent-corruption trap cannot be
exercised against a uniform core-combine. It **is** still pinned by the ragged oracle
(`to_ragged(kron_uniform(…)) == kron_ragged(…)`, which fixes the gappy strided pattern), so it ships verified
but unpartnered — parity now beats an asymmetric surface later.

## 4. Conversions + constructors

- **`to_uniform` / `to_ragged`** for both weight types (mirror `UniformTuckerTensorTrain.from_t3` / `.to_t3`:
  pad the ragged weight vectors to the max rank, build the prefix masks; and the reverse, reading the real
  prefix through the masks). These are the equivalence-test bridges. **Gappy masks make `to_ragged` a
  gather, not a slice** (`docs/uniform_masks_vs_ranks.md`) — follow `ut3_to_t3`. A varying-rank stack returns
  a *tree* of ragged weights, not one stacked weight (the `ut3_to_t3` precedent).
- **`from_ut3svd`** (uniform `UT3Weights`): `ut3svd` already returns `(new_data, ss_tucker (d,)+C+(n',),
  ss_tt (d+1,)+C+(r',))` — **exactly** the `UT3Weights` supercore shapes, with `new_data`'s masks being the
  right masks for the weight. So this is a near-trivial wrap, mirroring `T3Weights.from_t3svd`.
- **`from_ut3weights`** (uniform `UT3FrameWeights`): the same slicing as ragged (`up=down=tucker`,
  `left=tt[:-1]`, `right=tt[1:]`) on supercores **and** masks — the `(d+1)` TT-bond supercore/mask slices to
  the `d`-length left/right variation families.

**Naming (proposed — flag for review).** Mirror `UT3Frame.from_t3frame` / `.to_t3frame`:
`UT3Weights.from_t3weights(W)` / `.to_t3weights()` and `UT3FrameWeights.from_t3frameweights(W)` /
`.to_t3frameweights()` for the ragged↔uniform conversions; `UT3Weights.from_ut3svd(x)` and
`UT3FrameWeights.from_ut3weights(UW)` for the constructors (the latter is the uniform twin of ragged
`T3FrameWeights.from_t3weights`). Consistent, but `UT3Weights.from_t3weights` (a conversion) and
`UT3FrameWeights.from_ut3weights` (a metric builder) read similarly — different classes disambiguate, but
worth a second look.

## 5. Testing (equivalence oracle + mask-strictness — the required trio)

Per `docs/contributor/testing_strategy.md` (dense/ragged alone is **blind to too-permissive masks**):

1. **Equivalence to ragged**: `to_ragged(op_uniform(to_uniform(x), …)) == op_ragged(x, …)` on real parts,
   across structures × stacks (incl. **variable ranks per stack element** — the determinantal variety).
2. **Garbage-robustness**: write **large-finite** garbage (`1e6·(1−mask)`) into padding; assert
   `norm`/`inner`/`reciprocal`/`sqrt` unchanged, and `absorb` unchanged **on the real parts** (absorb is
   garbage-transparent, so its padding legitimately differs — compare through `to_ragged`, not raw
   supercores). NB the layer masks by *multiply*, so use large-finite, not NaN.
3. **Exact output masks** (derived non-circularly): `absorb`/`norm`/`inner`/`reciprocal`/`sqrt` keep the input
   masks; `concat`/`kron` produce the **concatenated / Kronecker** masks (assert the gappy pattern, not just
   the rank count).
4. **jit** dispatch (masks stay host; a stray `jnp` on a mask → tracer → caught).
5. **The `reciprocal` finiteness test** (new): `from_ut3svd(x).reciprocal()` must produce finite padding, and
   `absorb`/`norm` on it must be nan-free. Known adjacent hazard, *not* fixed by the mask: a rank-deficient
   frame's SVD completion puts numerically-tiny-but-nonzero σ's in slots the mask calls **real**, so `1/tiny`
   is a legitimately huge weight. Same in ragged; the test's job is to confirm uniform is not *worse*.
6. **Stacks**: use the `_CONFIGS`-style matrix (no-stack, `C`, `K`, `K+C`, forced-larger padding, multi-axis
   `C=(2,3)`/`K=(2,3)`, varying-rank-across-`C`). For `UT3FrameWeights`, the case that matters most is the one
   the ragged suite was **missing**: a `C`-stacked weight against a `K+C`-stacked tangent (§6 S0).

## 6. Slices

**S0 — fix the ragged layer first (the oracle).** The design review found the ragged frame-weight *stack
model* is wrong (§8.5, §8.6). It must be fixed before mirroring: the ragged layer is this build's equivalence
oracle, so mirroring the bug would put it in two places and have the oracle certify it.

  - `fv_weights_consistent(variations, weights)`: keep the signature (the variations stay **blind to the
    frame** — that is the model, and `absorb_weights(variations, W)` is a standalone free function with no
    frame). Change **only** the stack test: from "the weight's stack equals the *full* variation stack" to
    "the weight's stack is the **trailing part** of the variation stack" — the same rule `check_fv_pair`
    already states for frame↔variations ("`frame.stack_shape` … the trailing (inner) part of
    `variations.stack_shape`"), with the weight playing the frame's role. Rank checks are already correct.
    **Non-breaking**: existing `K+C` weights still pass (`K_w = ()`); the `C`-only GK weight starts passing.
  - **New: the weight↔frame guard at the tangent level.** `T3Tangent.weighted_norm` / `weighted_inner` /
    `absorb_weights` require `W.stack_shape == frame.stack_shape` — the weight↔frame analog of
    `check_fv_pair`. Structural (shapes) ⇒ **hard error in both modes**, jit-safe. Proposed home:
    `check_fw_pair(frame, weights)` in `frame_variations_format.py`, beside `check_fv_pair` (frontend, like
    the same-frame guard — it needs both objects). This is what makes "the weight is frame-like"
    *enforceable* rather than merely intended. **Breaks the existing tests** (their helper builds at `K+C`
    and calls `v.weighted_norm`) — those are being rewritten anyway.
  - **Docs**: `docs/weighting.md` §Batching and the `T3FrameWeights` class docstring currently say "batches
    like `T3Variations`" — wrong. Both become: frame-like (carries `C`), *absorbed into* the variations,
    paired with them by the same trailing rule as `check_fv_pair` (cross-reference it).
  - **Tests**: fix the helper to build weights at `C`; add the missing case (`C`-only weight × `K`-stacked
    tangent: predicate `True`, norm equals a `K`-tiled reference); add a rejection case for the new
    tangent-level guard.
  - **`dev/weighted_layer_design.md`**: §4 and §6 still describe the *pre-build* design (weights absorbed
    into the frame cores `U/O/P/Q`, variations untouched). The header records the change to
    metric-on-variations, and the code + `docs/weighting.md` are correct, but the body contradicts them —
    correct it (a trap for a cold reader). Note the irony: §4b, written under the *old* model, states the
    **correct** stack rule ("weights carry `C` only — never `K`; `T3FrameWeights` mirrors `T3Frame`") — the
    implementation over-corrected at the design change, moving the *stack* to the variations when only the
    *absorption target* should have moved.

1. **S1 — `UT3Weights` + `ut3_absorb_weights` + `ut3_weighted_norm`/`_inner` + `to_uniform`/`to_ragged` +
   mask-guarded `reciprocal`/`sqrt`.** The base-point layer. Equivalence + garbage + mask tests.
2. **S2 — `ut3_concatenate_weights` / `ut3_kronecker_weights`** (the gappy-mask ops) + `from_ut3svd`.
3. **S3 — `UT3FrameWeights` + `ufv_absorb_weights` + uniform tangent `weighted_norm`/`inner` +
   `from_ut3weights`** (+ the uniform `check_fw_pair` analog). Settled by S0's model: weights at `C`, masks at
   `C`, absorb broadcasts over `K+C`.
4. **S4 — frontend wiring** (`UT3Weights`/`UT3FrameWeights` classes, methods, `absorb_weights`,
   `UT3Tangent.weighted_norm`/`inner`/`absorb_weights`). **Landed inside S1/S3** rather than as its own
   slice. ~~+ dispatch inference (ragged vs uniform from the arg)~~ — **deliberately not built**, see §8.7:
   the module *is* the dispatch.
5. **S5 — docs** (extend `weighting.md` with the uniform mirror; correct the absorb-is-a-reduction claim; note
   in the uniform docs).

## 7. Watch-list (the traps)

- **Gappy masks after concat/kron are EXPECTED, not a bug** — and the ops need no mask cleverness (combine
  the weights and the masks the same way; §3). But **assert the exact strided pattern, not the rank count**:
  a prefix mask of the right rank is the plausible-wrong answer, and it is invisible to value tests
  (phantom rank). Both such mutations were tried and killed in S2.
- **`kron` is a last-axis outer product broadcasting the shared prefix — NOT `np.kron`** (which would
  Kronecker the mode/stack axes). The ragged build hit exactly this; it is the real trap in the combines.
- **`to_ragged` through a gappy mask is a gather, not a slice.**
- **`reciprocal` → `inf` → `0 × inf = nan`** — the finite-padding rule; the headline GK path.
- **Do NOT add an entry mask to `absorb`** "for safety" — it is garbage-transparent, and the pre-review plan
  said otherwise (§2).
- **Masks stay host numpy** (`np`), supercores `xnp`; `int(mask.sum())` must not see a tracer (jit chokepoint).
- **Variable ranks per stack element** — the determinantal variety; tests must vary ranks across the stack.
- **The `C` vs `K+C` stack trap** — the one that already bit the ragged layer (S0). A weight is **frame-like**.
- The ragged oracle makes all of this checkable; lean on it hard.

## 8. Decisions (settled 2026-07-15, design review with Nick)

1. **`reciprocal`/`sqrt` are mask-aware** — mask on entry, double-`where`, padding stays finite (`0`). The
   `inf`/`nan` hazard was missed by the pre-review plan and hits the headline GK path (§2).
2. **`kronecker` ships now**, despite having no uniform Hadamard to pair with (§3).
3. **A conceptual wall between masking and weighting** — weighting code never calls masking functions; shared
   mechanics get broken out into a neutral subfunction called from each side (§2). Nearly free, because
   `absorb` needs no masking.
4. **`UT3Weights` carries no `shape` field** — weights live only on internal edges (§1).
5. **`UT3FrameWeights` is frame-like: stack `C`, masks at `C`** (§1). Verified empirically that a `C`-only
   ragged weight against a `K+C` tangent already computes the *identical* numbers as a `K`-tiled weight (max
   diff `0.0`) — the leading `'...'` lifts `C` over `K` for free because `C` is innermost. The math was
   already right; only the *checking* encoded the wrong model.
6. **The stack model (Nick's statement of it).** The frame carries `C` — a collection of different frames.
   The variations carry an arbitrary `var_stack_shape` and are **intentionally blind** to their frame. Any
   operation pairing a frame with a variation requires `var_stack_shape = something + C`, and the "something"
   is read as `K`: many frames indexed by `C`, and for each frame many variations indexed by `K`. **A weight
   acts the same way as a frame**: it carries `C`, and pairing it with a variation (for absorption) requires
   `var_stack_shape = something + C`. A weight at `K+C` is therefore not malformed — it reads as
   `C_w = K+C` (that many frames, one variation each), exactly as a `T3Variations` at `(5,3)` is ambiguous
   between `K=(5),C=(3)` and `K=(),C=(5,3)`. The frame resolves the ambiguity **at pairing time** — hence the
   new tangent-level `W.stack_shape == frame.stack_shape` guard (S0), which is the only place with enough
   information to enforce it.
   *(A rejected alternative: check the weight against the **frame** — `fv_weights_consistent(frame, weights)`.
   It invents a second pairing pattern where the library already has one, and couples the predicate to a frame
   that the standalone `absorb_weights(variations, W)` does not have.)*
7. **No ragged/uniform dispatch on the weighted free functions — the module IS the dispatch** (Nick,
   2026-07-15; drops half of S4 as originally written). The weighted surface stays **parallel and
   module-scoped**: `tucker_tensor_train.absorb_weights` and `uniform_tucker_tensor_train.absorb_weights`
   are two functions, each typed for its layer, and likewise the tangent pair in
   `frame_variations_format` / `uniform_frame_variations_format`. Nothing infers the layer from the
   argument.

   **Why.** The user controls dispatch by deciding which layer they are working in; if they need to
   switch, the conversion hooks are how (`from_t3`/`to_t3`, `from_t3weights`/`to_t3weights`,
   `from_t3frameweights`/`to_t3frameweights`, and the object conversions). This also matches the shape of
   the library everywhere else: the package root exposes ragged and uniform **side by side under distinct
   names** (`TuckerTensorTrain`/`UniformTuckerTensorTrain`, `MANIFOLD`/`UNIFORM_MANIFOLD`,
   `t3_orthogonal_representations`/`ut3_orthogonal_representations`), and every op is reached through its
   layer's module. A type-dispatching `absorb_weights` would be the only such function in the library and
   would blur a boundary the rest of the API keeps sharp.

   **The optimizers are not a counterexample.** `optimizers.py` really does
   `isinstance(x0, UniformTuckerTensorTrain)` — but `newton_cg` is a *single entry point* a user calls with
   whatever `x0` they have, so it has no module to dispatch through. `absorb_weights` does.

   **Loose end (not a blocker):** the weighted surface is not re-exported from the package root at all —
   `T3Weights` is `t3toolbox.tucker_tensor_train.T3Weights`, unlike every other frontend class. The docs
   already teach that submodule path, and the ragged layer shipped that way. Worth a look someday: the
   *classes* have distinct names (`T3Weights`/`UT3Weights`/`T3FrameWeights`/`UT3FrameWeights`) so they
   could be root-exported with no collision; only the free functions collide, which is exactly the
   collision this decision declines to resolve with dispatch.
