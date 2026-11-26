SIGN OFF

# Review of `K0_DESIGN_NOTE.md`, round 3 (prose only)

Reviewer: Claude, 2026-11-25. I read `K0_DESIGN_RESPONSE_r2.md`, `k0/prose_edits.md` §1 and §2 (the new section
headers part), note §1.3 and §6, and checked each against the code at `b5125f76`. I sign off with one wording edit for
the build, M1 below, whose exact text I give.

## The round-2 findings

- **S1: resolved.** `probing`'s docstring now reads "each but the ambient transpose one call of the engine", and
  names `dense_probe` on its own. This matches my `ast` scan of the kept functions: six call the engine, while
  `t3_probe_ambient_transpose` and `dense_probe` do not.
- **S2: resolved.** The three headers I named are now "cut to what stays", and two more cases have been found.
  - **`probing.py:274-276`** is kept as it is. After K0 the banner "Probing a tangent vector" directly heads
    `tv_probe`. Right.
  - **`probing.py:1236-1266`**: the banner (`:1236-1238`) stays and heads `t3_probe_corewise_hessian`. The comment
    under it (`:1239-1266`) describes the dotted kernels that go ("the kernel is mask-free", the uniform caller that
    "packs, delegates here, and re-masks"), so deleting it is right.
  - **`probing.py:672-674`**: the new banner, "Transposes of the probe; the corewise tangent", is true of the four
    functions it heads.
  - **`ut3_sampling.py:469-473`**: true. Each `ut3_*_corewise_hessian` is one engine call wanting `Kp`. The point's
    masks reach theta_hat through `D_dot` and `L_dot` on entry, and the output is masked on exit.
  - **`utv_sampling.py:444`**: "(probe, apply, entries)" is right.
  - **My own sample of the rest of the "deleted whole" list**, by the same question, from my round-2 listing of the
    first kept statement after each block. In each case the kept statement has its own header further down, so none
    of these blocks heads kept code. The response's account matches:
    - `utv_sampling.py:135-148`, `:252-262`, `:389-394`: the next kept statement is `_gauge_masks_over_Knew`, under
      `:444`;
    - `fitting.py:107-113`: `_flat_gather`, under the rewritten `:154-160`;
    - `fitting.py:177-179` to `:737-738`: `_KINDS`, under the rewritten `:745-749`;
    - `uniform_fitting.py:61-78`: `pack_sample`, under the kept packing comment at `:366-374`;
    - `uniform_fitting.py:457-462`: it explains `_fg` and the `_ptake_*` only; `uniform_minimal` has no header today;
    - `ut3_sampling.py:256-261`, `:310-315`, `:373-378`: `ut3_probe_corewise_hessian`, under `:469`;
    - `probing.py:586-595`: `tv_probe_transpose`, under the new banner `:672-674`.

## The sentence not kept (`ut3_sampling.py:124-133`)

**I agree it is no longer true.** The old sentence said the boundary-bond squash leaves junk in the padding, which the
exit mask then clears. That described the pre-engine kernel. I reran E1 in round 1 and got the same result: with the
exit mask removed, the corewise transposes' padding is exactly 0.0 for all three kinds and both values of
`sum_over_samples`, and the real block is bitwise the same. The mechanism is in the code: every operand is masked on
entry, and each sweep starts with ones on the real slots of its end bond only (`sampling_variables.py:1368-1370`). So
nothing in the engine broadcasts into the padded boundary slots.

**What remains of the block is true at K0.** The tangents and transposes are named for what they are. The raw
gradient supercores mirror the ragged return. Every input is masked on entry and every output on exit. The
equivalence contract's "padding don't-care", with consumers masking on entry, is K5's to change, not K0's. "The
masks, not the values, govern rank" still holds.

## The factoring in `fitting`'s module docstring

**It is true against the code.**
- `backend/optimizers.py:192` says the blocks come back before the projection.
- `gradient` projects the oracle's `g` (`:375`).
- `jacobian` and `gn_quadratic` project `p` before asking for `Jp` and `Jp_norm_sq` (`:380-388`).
- The Gauss-Newton `hvp` projects `p` and then `JtJp` (`:396-399`).

So `J = 𝒥∘Π` and `Jᵀr = Π∘𝒥ᵀr` hold with `Π` being the geometry's `project`, **on both geometries.**

**What `Π` is on the corewise geometry.** `CorewiseGeometryOps.project` (`backend/geometry.py:597`) and the uniform
one (`:1199`) are the identity, or the per-group mean of the tied Tucker factors when the geometry is shared
(`groups`). There is no gauge on the corewise geometry. So the docstring's "it supplies **the gauge projection** `Π`"
names the manifold's case only. The formula is right; the label is not, on corewise.

**M1. Minor; apply in the build, exact text.** In `prose_edits.md` §1, `fitting.py:1-23`, after-text, replace
"it supplies the gauge projection ``Π``, so that" with:

    it supplies the projection ``Π`` (``geometry.project``: the gauge projection on the manifold; on the corewise
    geometry the identity, or the per-group mean of tied factors), so that

and "before any gauge projection" with "before any projection".

**M2. Minor, optional.** "the Jacobian ``𝒥`` and its transpose ``𝒥ᵀ`` (``Jp``, ``g``, ``JtJp``)" would be exact as
"(``Jp = 𝒥p``, ``g = 𝒥ᵀ(M r)``, ``JtJp = 𝒥ᵀ M 𝒥p``; ``Kp`` likewise bare)".

## The build notes

Both are in note §6:
- the mutations are rerun on the built tests, with their outcomes in the handover (`:468-469`);
- `test_import_graph.py`'s module docstring gains a line (`:470`).

## For the build: what to watch

Rerun the mutations of note §6 on the built tests, not the prototypes. In the diff, check four things:
- **The deleted comment ranges** match note §1.3 exactly, and the five headers kept or rewritten are there.
- **No signature column or diagram of a kept function changes.**
- **The re-based oracles name no deleted code and no engine function where the note says dense.** This applies to the
  `test_tucker_tensor_train` forwards (`dense_*`, factorials inline) and the entries jets off the grid.
- **The hygiene negative control** asserts `'padding not zero'` with all three patches in place, and passes without
  them.

Quick scripts only this round (reading, the earlier `ast` scans, the cited lines). No test was run, and nothing is
left running.
