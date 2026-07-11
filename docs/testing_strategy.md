# Testing strategy — correctness, and the clean-padding blind spot

> The *why* behind how this library is tested. The mechanics (unittest + `subTest`, numpy-only numerical
> correctness with jax dispatch in `tests/test_dispatch.py`, doctests as reproducible examples) are in
> `CLAUDE.md` and [`doctest_style.md`](doctest_style.md). This note records the deeper reasoning — in
> particular **why dense/numerical tests alone are not enough for a representation that carries explicit
> structural metadata** (the uniform layer's rank masks), and the two tools that close the gap. The
> hardest-won lessons are from the uniform tangent layer; the principles are general.

## The model: equivalence against ground truth

Correctness is **equivalence**, checked two ways that reinforce each other:

1. **Dense ground truth.** Rebuild the represented tensor with a hand-written `np.einsum` / `.to_dense()`
   and compare residual norms (~1e-12..1e-16). This is the bedrock — it does not trust any of the library's
   own machinery.
2. **The cross-representation equivalence contract.** The uniform layer is *a faster ragged layer*:
   `to_ragged(uniform_op(to_uniform(x))) == ragged_op(x)` on the **real (masked)** content, per stack
   element (garbage in the padding is don't-care). The ragged layer is the trusted reference; every uniform
   op is verified against it. See [`uniform_equivalence_contract.md`](uniform_equivalence_contract.md).

These two together are necessary. They are **not sufficient.**

## The blind spot: too-permissive masks (phantom rank)

A uniform object is **supercores + boolean rank masks**. The masks are *structural metadata* — they name
the real rank (which stratum of the determinantal variety the point sits on). Two failure modes for a mask:

- **Too restrictive** (drops real content): the dense is *wrong* → caught by ground-truth comparison. ✓
- **Too permissive** (claims rank where the supercore is zero — *phantom rank*): the dense is *still right*
  (it kept zeros) → **invisible to every dense/numerical test.** ✗

Crucially, **all our natural test inputs have clean (zeroed) padding** — they come from
`ut3_orthogonal_representations` / `from_t3`, which canonicalize. On clean padding, a phantom-rank mask is
completely silent. This is the single most dangerous bug class here, because the wrong rank propagates: a
downstream op that *trusts* the mask (SVD truncation in `retract`, sub-block extraction in `to_t3`,
rank-controlled optimization) silently does the wrong thing.

**This is not hypothetical.** `utv_to_ut3` once built the doubled-rank tensor with `ones` boundary
masks (claiming `rR+rL` rank at a boundary whose supercore is rank-1-and-zeros). Every dense test passed at
~1e-16. Only re-deriving the construction from the paper (Appendix A.3.1, eqs 50–53) caught it; the fix is
the honest `zeros` boundaries that respect the block structure.

## The two tools that close it

### 1. Exact output-mask assertions (non-circular)

Assert the produced object's masks **exactly equal an independently-derived expected mask** — derived from
a *different source* than the implementation, so the test is not the code checking itself. Examples:

- the doubled-rank masks vs a from-the-frame-ranks construction of the prefix-pair Tucker mask and the
  `[Q | P]` honest-boundary TT mask (the paper rule), not vs the impl's own concatenation;
- `retract`'s ranks vs the **ragged** `retract`'s ranks (the ground truth — note these can drop *below* the
  frame ranks when `frame+tangent` is genuinely lower-rank, which is correct, not a bug);
- gauge output masks `==` input masks (gauge must preserve them); `project` masks `==` the frame gauge masks.

> **The tautology trap.** Corrupting an output's padding using *the object's own mask* and re-densifying
> **always passes** — `to_dense` applies that same mask, so it is testing that `to_dense` applies the mask,
> not that the mask is right. Non-circularity *requires* an independent notion of the true real-region.

### 2. Garbage-padded inputs (mask-once robustness)

The contract says padding is garbage-don't-care, so **a correct op must give the identical result on a
garbage-padded input as on a clean one** (`tests/test_uniform_manifold.py`'s `_corrupt` helper writes
large garbage into the masked-out region). This catches two things at once:

- ops that read raw padding instead of **masking-once** on entry;
- for ops that build from raw supercores (`utv_to_ut3` does **not** mask its inputs — it relies on the
  honest output mask), the garbage flows into the output's input-derived padding, so a too-permissive output
  mask there **leaks** it → the dirty dense differs from the clean one → caught.

(The phantom *boundary* mask is a constructed-zero region, not input-derived, so input-garbage alone does
not reach it — that one needs the exact-mask assertion. The two tools are complementary, not redundant.)

## The stack matrix

Batching/stacking is the most error-prone part of the library (`batching_and_stacking.md`). Every op test is
parametrized over a standard matrix (`_CONFIGS`): the no-stack case, a frame (`C`) stack, a tangent (`K`)
stack, `K+C`, **forced-larger padding** (so *every* core has a padded region, not just the sub-max-rank
ones the default tight pad leaves clean), and **multi-axis** `C=(2,3)` / `K=(2,3)` (where the negative-axis
/ `1+|K|` / suffix bookkeeping hides off-by-ones), plus a **varying-rank-across-`C`** stack (the rank-sweep
case a single ragged object cannot even represent). Each axis targets a category that has bitten us before:
stacking, masking, and TT-boundary handling.

## Why the masks earned their keep

A striking outcome of the hardening pass: once the tests were strict enough to *see* mask bugs, **the
implementation had none** — exact masks, garbage robustness, multi-axis, varying-`C`, all green, confirmed
independently by an adversarial cold-read audit. This is a point in favor of the **masks-not-integer-ranks**
design ([`uniform_masks_vs_ranks.md`](uniform_masks_vs_ranks.md),
[`uniform_rank_masks_rationale.md`](uniform_rank_masks_rationale.md)). Boolean masks compose with the
operations *algebraically* and locally: add = concat, multiply = Kronecker, the doubled-rank embedding =
concat, gauge/projection *preserve* the mask, retraction's SVD *re-derives* it. There is little integer
rank-arithmetic to get wrong, and the few non-obvious spots (the `[Q | P]` doubled-bond order, the honest
boundaries) are pinned by the exact-mask assertions above. A representation whose metadata travels through
operations by concatenation and Kronecker products is simply *harder to corrupt* than one that threads
hand-maintained integer ranks.

## The adversarial audit pattern

For subtle correctness work, a useful cross-check is an **independent cold-read audit**: a fresh agent with
no shared assumptions reads the implementation against the math and the trusted reference and tries to
*construct a passing-but-wrong case* (the verb is construct, not summarize). It is most valuable exactly
where the author's own mental model might hide a blind spot — and it is reassuring when it converges,
independently, on the same suspicious surface the author targeted (here: the garbage-in-padding coverage
gap) and finds it sound.

## Checklist for a new uniform op

When adding a uniform operation, write:
1. **dense-vs-ragged** per stack element, over the `_CONFIGS` matrix (incl. forced-pad, multi-axis,
   varying-`C`);
2. **exact output-mask** assertions against an independently-derived expectation (or, where the ground truth
   is the ragged op, against *its* ranks);
3. **garbage-padded-input** robustness (clean == dirty);
4. **jax dispatch** in `tests/test_dispatch.py` (jit the op; a stray `np.*` on a tracer raises), unless an
   in-file jit test already covers it.

Numbers 2 and 3 are the ones that distinguish a *correct* uniform object from a merely *numerically
plausible* one. Do not skip them.
