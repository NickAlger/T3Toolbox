# N5 handoff: the uniform alignment step

Slice N5 of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§8): `utv_align_padded_frame` in
`backend/uniform_padded_frames.py` (§3.4, §4.4, §5.3, round 7), tests T9 and T10, the T16 row, and the HISTORICAL rows.
Implementer: Claude Opus 5.5, 2026-11-09, on top of 3ae48cca (N3b committed). Everything is uncommitted; nothing
outside the repository was touched.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/uniform_padded_frames.py` | `+ utv_align_padded_frame` (public, in `__all__`, one-line docstring: it gets a twin in N8), its kernel `_utv_align_kernel`, and private helpers. Also the `UVariationsData` alias (used in the signature) and the module docstring. Imports `contractions`, plus seven private, module-level, closure-free helpers from `padded_frames` (below). T18's allowed set holds. |
| `tests/backend/_uniform_padded_frame_testing.py` | `+ scatter_variations` (the tangent at a scattered frame's positions, garbage in every junk entry). `element` takes an index tuple, so it handles C = (2, 2). |
| `tests/backend/test_uniform_padded_frame.py` | T9 (`TestStepAgainstRagged`, `TestStepStructuralErrors`), T10 (`TestStacks`: the step and the start); module docstring. |
| `tests/test_dispatch.py` | `test_uniform_align_padded_frame_dispatch`. |
| `tests/test_contractions_interpreter.py` | 14 HISTORICAL rows, all `''`: the d-twins of c1, c2, c8, c14–c18, c20–c23, c30, c31. |

## The step

- **Public function** (host work only). It validates:
  - `update`;
  - the block-layout shapes (frame, variations, masks), and the split inside the widths;
  - per element, that the A-block counts and the frame's counts are an orthogonal representation's stores with
    boundary bonds 1;
  - output ranks: both or neither, a sequence or a `(d,)+C` / `(d+1,)+C` host array, boundary 1;
  - `old ≤ out ≤ frame` per store, and no store truncated without being updated;
  - within each sharing group, equal Tucker stores (old, output, frame) and equal A-block up masks.

  It then builds the host data: the output masks `[A-mask | prefix(k¹)]` at `Ka1 = max(k¹U, k¹D)` and
  `Kb1 = max(k¹L, k¹R)`; the edge row masks (outer products of the output masks); the pin masks (the edges' A-block
  masks); the one-hot selections; and the group tables. The tables are `members` / `first` / `group_of` as int
  tuples and `valid` as a bool array, so no integer array crosses the boundary. `partition_static` splits
  `(supercores, (V, H), host)`, and the kernel is called once, eagerly.
- **`_utv_align_kernel(dynamic, skeleton, split, update)`** is jittable: every static lives in the skeleton (value
  hashed), `split` or `update`. The T16 row jits it directly with static arguments 1–3.
  - Masked once at entry through the chokepoints (`ufv_apply_frame_masks`, `ufv_apply_variations_masks`).
  - Down and up messages: one `'dC…'` contraction each, over all modes.
  - Left and right message sweeps: `xscan` with `padded_frames._left_message_step` / `_right_message_step`
    (judgment call 1), on statically sliced supercores, zero carries `C+(Kb, rT)`, `C+(Kb, Kb)`.
  - Up edges: per-mode content, gathered into groups (`Y[members]` × `valid`, members side by side), one batched
    `pad_safe_svd` over `(G,)+C`, scattered back by `group_of`. The old block comes from the group's first mode,
    so the output is tied bitwise.
  - Left and right edge sweeps: module-level bodies (judgment call 2), with zero carries `C+(Kb1, Kb)`,
    `C+(rT1, Kb)`. Each body derives the split from its operand shapes (`_new_coordinates`: `r = width −
    o.shape[-1]`), and the per-mode row masks, pins and selections ride the scan inputs as data.
  - Down edges: batched, with the sweeps' coordinate maps (zero past the ends) and `NEW` as d-twin block
    operations.
  - Every edge: `Y⊥ = Y − Ā(ĀᵀY)`; `pad_safe_svd([Ā | Y⊥ / (4‖Y⊥‖_F)], rows, [pins | ones])`, where a zero
    content stays zero (the guard); `P⁺ = U_s[:, :m] · sel[:m]` with `m = min(K, A + K1)`; and the direct movement
    `‖g_old − P⁺P⁺ᵀ g_old‖_F`. A zero-width completion skips the factorization.
  - Transport: `Y = g_old`.
  - Carriers: the masked A-blocks of `L_{d−1}` and `R_0`, zero-padded.
  - Movement: the maximum over all edges, shape `C`.
  - d = 1: the zero-length sweeps return empty outputs (`_sweep`; numpy's scan cannot stack zero steps).

### Judgment calls

1. **Reused ragged helpers.** The message sweeps reuse `padded_frames._left_message_step` and
   `_right_message_step`, and the edge bodies reuse `_new_coordinates`, `_unfold`, `_fold`, `_root` and
   `_edge_content`.
   - These are already module-level, closure-free, plain-string, and shape-driven. With statically sliced
     supercores and zero carries they are exactly the prototype's uniform bodies, so duplicating them would add
     nothing.
   - §4.2 already allows sharing private host helpers across these two modules (T18 allows the import).
   - The map sites have new d-twin functions (`_u_down_messages`, `_u_up_messages`, `_new_coordinates_d`,
     `_u_content_d`, `_u_edge_d`), as Appendix C asks.
2. **Transport without a closure.**
   - The edge bodies need `on` (updated or transported), which no operand shape carries.
   - Rather than an `lru_cache` factory (principle 4 of `scan_body_principles.md`), there are two module-level
     wrappers per sweep (`_u_left_edge_step` / `_u_left_edge_transport_step`, and the right twins) around one
     shared body with `on` as a parameter. They are closure-free with stable identities, as you asked.
3. **No `out_widths` argument.** §5.3's signature has none: the output P-block widths are the maxima of the output
   counts (§3.4), so a truncating pass narrows the P-blocks through its output ranks. T9's 'truncating', 'empty,
   truncating' and 'groups, truncating' cases cover it. Consequence: with `tucker_ranks=None` and a P-block wider
   than its counts, the output narrows to the counts' maxima. Every frame the library builds is already at that
   width.
4. **`DU = [Ā | P⁺]ᵀ g_new` (c22), as in the prototype**, not `U⁺ g_new` (c24). So c24's d-twin is not used and not
   added: 14 new d-rows, not the plan's 15.
5. **Frames compared by the plan's measure** (principal-angle sine per directed edge), in T9, T10 and T16.
   pad_safe_svd's sign gauge differs from the ragged SVD's, and from jit to eager (measured entrywise gaps of 2.0,
   subspaces 4e-15).

## Verification

- **T9 against the ragged step, through the block-layout scatter.**
  - Scatter: gappy masks in both blocks, one junk A-slot, junk in the P-blocks, N padded by one, finite garbage in
    every junk entry of the frame and the tangent.
  - Coverage: all 13 T1 cases (d = 1, 2, 3, 5, uneven, non-minimal, tangent 10x, truncating, empty, empty
    truncating, groups, groups truncating, groups spread) and the six update subsets (incl. the two colours).
  - Results: frames ≤ 1e-9 (measured ≤ 4.3e-14), movements ≤ 1e-9 (measured ≤ 2.4e-15).
  - Output masks exactly `[input A-mask | prefix(ragged output counts)]`; junk exactly 0; tied groups bitwise.
  - Garbage changes nothing (bitwise, equal layouts) on four cases.
  - The carriers come out as their masked A-blocks even when the input carrier has content in completion slots.
- **T10.**
  - The step at C = (3,), with different old, frame and output ranks per element, and at C = (2, 2): each element
    equals its own call (frames and movements ≤ 1e-14) and the ragged step (≤ 1e-9).
  - The start at C = (3,) and (2, 2) with per-element targets: each element equals `fv_padded_frame` ≤ 1e-14.
- **Structural errors, each isolated:** update; a variation shape; the split; masks that are not an orthogonal
  representation; one output sequence given; output boundary; output above the frame; output below the old block;
  truncation without update; a sharing group on an untied scatter (a valid partition).
- **T16** (`test_uniform_align_padded_frame_dispatch`, about 40 s, almost all of it eager jax's first calls):
  - numpy → numpy and jax → jax (masks host numpy both ways);
  - the public function jitted with the masks closed over;
  - the kernel jitted directly with the skeleton static (it hashes);
  - a jitted stack C = (2,);
  - every comparison ≤ 1e-12 by the frame measure and the movement, junk exactly 0;
  - a traced mask raises the actionable error.
  - Measured at d = 4: eager jax 17 s on the first call and 0.06 s on the second (the module-level scan bodies
    compile once), jit 1.8 s.
- **Mutations, 17; 16 killed after one added test:**
  - entry masks (frame, variations); the selection's pin offset; the output masks; content/transport swapped
    (both sweeps); groups' contents; the down edge's maps swapped; the row masks; the `4‖Y‖` scaling; the down
    movement; and the layout, truncation, group and fit checks;
  - "the carrier keeps its completion slots" first survived (valid inputs have zero there); the carrier test above
    now kills it;
  - "pin mask ignored" survives as an equivalent mutant: junk A-columns are exactly zero, so declaring them real
    only raises pad_safe_svd's `q`, which already covers every selected triplet.
- **HISTORICAL:** the interpreter test passes with the 14 rows; T18 holds; the module's doctests pass.

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n auto`, on the final code: **1,292 passed, 4 skipped, 48,599
subtests passed** in 10 min 45 s (N3b: 1,277 passed). The one warning is the existing `adam` one.

## Plan problems

- §4.4 says "Message carries `C+(Kb, rT)` and `C+(Kb, Kb)`". That holds because the Gram roots have `min(p, Kb) = Kb`
  columns (`p ≥ nT·rT ≥ Kb`). XiO is `C+(Ka, min(N, Ka))` and XiU `C+(Ka, min(p, Ka))`; neither is carried.
- Appendix C counts 15 new d-rows; 14 are used (judgment call 4).
- The N5 row of §8 lists no `tests/test_dispatch.py` or `_uniform_padded_frame_testing.py`, but T16 and the T9
  scatter need them.
