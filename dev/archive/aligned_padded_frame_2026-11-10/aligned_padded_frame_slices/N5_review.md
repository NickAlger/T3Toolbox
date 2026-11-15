# N5 review: the uniform alignment step

Reviewer: Claude Opus 5.5, 2026-11-09. I reviewed the uncommitted working tree on top of 3ae48cca against:
- the plan: §3.1, §3.4 (the step, the kernel and its statics, round 7), §4.1, §4.4, §5.3, Appendix C, §6 T9/T10/T16;
- the prototype `P/uniform_form2.uniform_align2`;
- `N5_handoff.md`.

## Summary

Correct, and truly uniform. My own block-layout scatter puts gappy masks in BOTH blocks and garbage in every junk entry
of the frame AND the variations. On it:
- `utv_align_padded_frame` equals the ragged step to 1.6e-14 (frames, by the subspace measure) and 1.0e-14
  (movement) on 78 generic cases out of a random 80. The cases span d = 1..4, the zero tensor, sharing, truncating
  steps, and `update` ∈ {ULRO, UL, RO, U, LRO}.
- It equals the prototype `uniform_align2` to 1.1e-14 / 6.7e-16.
- On every case: output masks exact, junk exactly 0, bitwise invariant to the garbage.
- The two remaining cases are degenerate sharing cases with no content in some edge (an information-free Tucker
  increase; a group-minimal but per-mode non-minimal old point). There the completion is arbitrary by D-12: the
  ragged step, the uniform step and the dense oracle all disagree with each other there, and every output is still a
  valid frame.
- 25 stacked `C = (2,)` cases with ranks and gaps per element: 1.3e-14 / 8.5e-15.

I have no MAJOR or MINOR findings. The NITs are below, and one plan-text note is for the maintainer.

## Style (checked first)

- **Truly uniform.**
  - The kernel has no Python loop over modes. Down and up messages, up edges (over groups), down edges and `NEW` are
    batched `'dC…'` contractions.
  - The four sweeps are `xscan` with module-level bodies on statically sliced supercores, with zero carries at the
    ends: `C+(Kb, rT)` / `C+(Kb, Kb)` for messages, `C+(Kb1, Kb)` / `C+(rT1, Kb)` for edges.
  - The bodies take the split from operand shapes (`r = width − o.shape[-1]`).
  - Reverse sweeps run on reversed inputs, with the outputs reversed back.
  - `_sweep`'s zero-length branch (d = 1) is a static shape test in the driver, not in a body.
- **Masks are the only truth.**
  - The frame and the variations are masked once at entry through the chokepoints.
  - Old and own counts are mask sums.
  - Output masks are `[A-mask as given | prefix(k¹)]` at `(Ka1, Kb1)`, the maxima of the output counts.
  - Row masks, pins and one-hot selections are host bool, built from masks and counts.
- **Host tables.**
  - `members`, `first` and `group_of` are tuples of Python ints, and `valid` is a host bool array, so no integer
    array crosses `partition_static`.
  - Inside the kernel they become `np` index arrays at trace time, as §3.4 (r7) specifies.
  - The skeleton is value-hashed (`StaticSkeleton`). T16 jits the kernel directly with `(skeleton, split, update)`
    static.
- **Closure-free.**
  - The transport choice uses two module-level wrappers per sweep around one shared body (judgment call 2), chosen
    statically by `update`. No closure reaches a scan.
  - The kernel's local `zeros = lambda` is not a scan body.
- **Contractions.**
  - `grep einsum|@|matmul|.T` on the module finds nothing.
  - Every `contract` string is in HISTORICAL: the 14 new d-twins, plus `'dCaxb,dCaub->dCxu'` and the plain strings
    from N2.
  - The scan-body sites use the plain strings and the map sites the d-twins, as Appendix C says.
- **The factorization** is `pad_safe_svd` on `[Ā | Y⊥/(4‖Y⊥‖_F)]`, with a guard so that zero content stays zero (no
  0/0). The k¹ triplets after the pins are taken by a host one-hot (`_selection`), and a zero-width completion skips
  the factorization (a static shape test).
- **Signature and docstrings.** §5.3's signature, verbatim. One-line docstring on the twinned public function.
- **No plan labels.** The grep is clean.

## Correctness

- **§4.1 items 1–10 in block layout.** Every slice is at the split. I checked each against §4.1 and the prototype:
  `H[..., Rb:, :, :]`, `O[..., Rb:, Na:, :]`, `L[..., Rb:, :, Rb:]`, `PhiU[..., Na:]`, `PhiL[..., Rb:]`,
  `PhiR[..., Rb:]`, `PhiO[..., Na:]`, and the A-blocks `[:Rb, :Na, :Rb]` padded to the output widths.
- **Edges.** The pins are the edge's output A-mask. The row masks are the outer products of the incoming slots'
  OUTPUT masks. The selection is at offset = pin count, which is `pad_safe_svd`'s pins-first order.
- **Message indexing.** The messages are indexed as in N2: the left messages are reversed back and padded with a
  zero at `d−1`; the right ones are padded with a zero at 0. The coordinate maps get a zero past each end.
- **Groups.**
  - Content: `Y[members]` × `valid`, with the members side by side, and one `pad_safe_svd` over `(G,)+C`.
  - Each member gets its group's A-block and completion, so the output is tied bitwise.
  - Under transport the members' `g_old` stack side by side. Under the TIED precondition that has the same span as
    the ragged first-mode `g_old`.
- **Carriers.** The masked A-blocks of `L_{d−1}` and `R_0`, zero-padded, so old content only. T9 includes an input
  carrier with content in its completion slots, and that content is dropped.
- **Judgment call 3: no `out_widths`. Correct.**
  - §5.3 has none, and §3.4 says the output P-widths are the maxima of the output counts.
  - A truncating step narrows the P-blocks to the target counts' maxima. My 32 truncating cases (working =
    oversampled target) came out at those widths, with exact masks.
  - For N6's whole-loop jit the carry shapes stay invariant. The start builds widths = the counts' maxima. A
    `None`-output pass keeps them. The truncating pass sets the target's maxima, and every later pass keeps those.
  - The only narrowing a caller would not ask for is a frame whose P-block is wider than its counts (e.g. a
    scattered test frame), and the return comment says so.
- **Judgment call 4.** `DU = [Ā | P⁺]ᵀ g_new` (c22) equals `U⁺ g_new` (c24) by definition. It is as the prototype
  does it, and it leaves one d-twin unused (14 rows, not 15). Accepted.
- **Judgment calls 2 and 5** accepted.

## Findings

### 1. NIT: the four transport wrappers are unannotated

`_u_left_edge_step`, `_u_left_edge_transport_step` and the right twins: `def _u_left_edge_step(carry, x):`.

**Fix.** Every signature line carries an annotation (§3.1(4)). Copy the shared body's carry and `x` annotations, or
add a one-word pointer such as `# as _u_left_edge`.

### 2. NIT, for the maintainer: the uniform module imports seven of the ragged module's math helpers

`uniform_padded_frames.py` imports `_unfold`, `_fold`, `_root`, `_new_coordinates`, `_edge_content`,
`_left_message_step` and `_right_message_step` from `padded_frames` (judgment call 1).

**The case for it.** They are module-level, closure-free and shape-driven. Used on statically sliced supercores they
ARE the uniform bodies, and duplicating them would add nothing. The gold standard's uniform layer also calls ragged
backend functions (e.g. `ut3_linalg` → `t3_linalg`).

**The catch.** §3.2 says the uniform module imports `padded_frames` "for shared host bookkeeping only". These are
private names imported across modules, so a later ragged-only change to one of them would silently change the uniform
step. T9 would catch a numerical change, but not, for example, an added Python-int assumption that breaks only under
jit; T16's jit row would catch that.

**Recommendation.** Accept, and update §3.2's wording. Optionally, add one line to the docstrings of the shared
helpers in `padded_frames.py`: "shared with the uniform step: keep it shape-driven and closure-free".

## What I ran (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2; `OPENBLAS_NUM_THREADS=1` for the scripts)

1. **The independent scatter sweep,** `n5_check.py`. It uses its own block-layout scatter of the frame and the
   variations: gappy A- and P-positions, a junk A-slot and a junk P-slot at random, N padded by 0..1, 5·N(0,1)
   garbage everywhere.
   - 80 random cases: d = 1..4; 20 zero-tensor cases; 23 shared; 32 truncating (`ULRO`, working = oversampled
     target); `update` cycling over ULRO, UL, RO, U, LRO.
   - Compared against `tv_align_padded_frame` on the ragged frame, and the dense oracle on mismatches. Also against
     `uniform_align2` at `out_widths = (Ka1, Kb1)`, which errored on 5 cases at d = 1, a case the prototype never
     supported.
   - **Generic cases (78): frames 1.6e-14, movement 1.0e-14.**
   - **Prototype: 1.1e-14 / 6.7e-16.**
   - Masks exact, junk exactly 0, garbage-invariant bitwise: **all 80**.
   - Validity on all 80, the degenerate two included: orthogonality ≤ 1.1e-14, both chains reconstruct `X_old` to
     5.2e-14 (absolute), and the old block bitwise in the read-back.
   - The two degenerate cases: trial 7 (sharing `(1,0,1,1)`, a group-minimal old point with per-mode Tucker 2 >
     rL·rR = 1) and trial 23 (sharing `(0,0,1)`, Tucker 1→2 at TT ranks 1: no down slot, so the content is
     identically zero). On both, the ragged step is itself 0.06–0.96 from the dense oracle. These are D-12's
     arbitrary completions, not a uniform defect.
2. **Stacks,** `n5_stack.py`. 25 cases with `C = (2,)`: per-element old, frame and output ranks; different gaps per
   element in one layout; UL, RO, ULRO and truncating. Per element against the ragged step: **1.3e-14 / 8.5e-15**,
   junk 0.
3. **Tests.** `pytest tests/backend/test_uniform_padded_frame.py tests/test_contractions_interpreter.py
   tests/test_import_graph.py -n 4`: **47 passed, 2,811 subtests**. That covers T8, T9, T10 and T17, the contraction
   suite with the 14 new rows, and T18.
4. **Dispatch.** `tests/test_dispatch.py -k uniform_align_padded_frame` (T16): **1 passed, 2 subtests** (42 s).
5. **Doctests** of `uniform_padded_frames`: **1 passed**.
6. **Floor env** `tttt` (py3.9, numpy 1.22.4): `test_uniform_padded_frame.py` **27 passed**.

The full suite was not run; the implementer reports 1,292 passed, 0 failed.

## Counts

MAJOR 0, MINOR 0, NIT 2.

VERDICT: APPROVE
