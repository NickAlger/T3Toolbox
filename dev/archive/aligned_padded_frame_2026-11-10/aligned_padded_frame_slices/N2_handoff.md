# N2 handoff: the ragged alignment step

Slice N2 of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§8): `tv_align_padded_frame` (§4.1, §5.1), its
HISTORICAL rows (Appendix C), the dense reference module, T1–T3 and the T16 row. Implementer: Claude Opus 5.5,
2026-11-09, on top of N1 (3f9679ef). Everything is uncommitted; nothing outside the repository was touched.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/padded_frames.py` | `tv_align_padded_frame` (§5.1 signature verbatim) and the `VariationsData` alias, plus the module-level helpers and bodies listed below. N1's `_old_frame_stores` became `_frame_stores(frame, name)`. The ranks-to-stores computation with the boundary check is factored into `_rank_stores`, which `fv_padded_frame` and the step share; `fv_padded_frame`'s behaviour and messages are unchanged. Imports `contractions` (now used). Module docstring extended. |
| `tests/test_contractions_interpreter.py` | 28 HISTORICAL rows: Appendix C's c1–c31 minus the three that exist (c19, c22, c23), each `''`, decided by hand. Every term carries one group `C` and fixed lowercase counts, so `C` is solved from any operand. |
| `tests/backend/_padded_frame_oracles.py` | **new**: the dense per-edge reference, sharing no code with the implementation. `interfaces`, `dense_frames`, `complement_in`, `top_left`, `dense_pass` (unrooted; `update` with transport; sharing groups; returns the movement too), `tangent_basis`, `dense_project`, `distances`. |
| `tests/backend/test_padded_frame.py` | T1–T3 appended (four classes); the module docstring extended. T5 untouched. |
| `tests/test_dispatch.py` | T16 row `test_align_padded_frame_dispatch`. |

### The step's code

- **Bodies are module-level and closure-free.** `_down_message`, `_up_message`, `_up_edge` and `_down_edge` are
  `xmap` bodies. `_left_message_step`, `_right_message_step`, `_left_edge_step` and `_right_edge_step` are `xscan`
  bodies.
- **Helpers:** `_unfold` / `_fold` (`_fold` takes the core's three widths as ints, round 2), `_root` (the thin SVD, `Xi = Vhᵀ·diag(s)`), `_new_coordinates` (c8 / c25 / c14 by
  axis), `_edge_content` (`Y = [g_old Φ_P + g_new | g_old Ξ]`, the members side by side), and `_edge`
  (`pinned_leading_columns`, `o`, the direct movement).
- **Sweeps.** The two message sweeps and the two edge sweeps are folds with zero-size initial carries: `C+(0, 1)` and
  `C+(0, 0)` for the messages, `C+(0, 0)` and `C+(1, 0)` for the coordinate maps. No body has a boundary branch.
  The leftward sweeps run on reversed input tuples, and their outputs are reversed back.
- **Blocks and statics.** A-blocks are pre-embedded, and the P-parts are sliced at the old widths while the inputs are
  built. The splits inside the bodies come from operand shapes (`r = X.shape[axis] − o.shape[-1]`). `k1` and the
  update flag ride the inputs as a Python int and bool, as §4.1 allows on the ragged host loop.
- **Contractions.** Every one is a named `contract` string from Appendix C. There is no raw einsum and no `@`.

## Verification

- **Against the prototype first.** On ten cases (d = 1, 2, 3, 5; truncating; non-minimal; sharing; `UL` and `RO`;
  a stack), `cycle/core_form3.align_stores` and the library agree to ≤ 7e-15 in frames and ≤ 1e-15 in movement. The
  library's `root` is the SVD where the prototype used a QR; the subspaces are the same.
- **T1** (`TestStepAgainstDense`). Each output is compared with the dense per-edge pass from the same start: the
  largest principal-angle sine per directed edge ≤ 1e-9, and the movement ≤ 1e-9. Measured worst values: frames
  3.7e-14, movement 1.3e-15, orthogonality residual 1.3e-15.
  - Cases: d = 1, 2, 3, 5; uneven increments; non-minimal ranks; a tangent part 10×; truncating; the empty old frame,
    plain and truncating; sharing groups (adjacent, all modes plus truncation, non-adjacent string labels); stacks
    C = (2,) and (2, 3) per element; transport with `update` ∈ {UL, RO, LRO, ULO, U, O}.
  - Validity: the stores; orthonormal ≤ 1e-12; the old block bitwise in the leading slots; both base-point
    reconstructions X_old ≤ 1e-12 relative; one Tucker array per group (`is`).
  - Setup check: the library's tangent projection equals `dense_project` to 1e-12.
- **T2** (`TestStepInvariances`).
  - Ungauged variations (O(1) gauge terms on every bond and leaf–core edge; the same tangent vector to 1e-12) give
    the same frames ≤ 1e-9 and the same movement.
  - Hint scales 1e-14 … 1e14 change nothing (≤ 1e-9).
  - float32 (frame and hint, after 4 informing passes; hint `N + sT`): the error is ≤ 30·u·max(1, s) with
    u = eps32/2. Measured 2.4u, 2.7u, 2.7u·max(1, s) at s = 0, 1e2, 1e4. The outputs, movement included, are
    float32.
- **T3** (`TestStepDegenerateAndStructural`).
  - Degenerate hints give valid frames: zero (plain and truncating), rank one, the information-free Tucker increase at
    d = 3, and at d = 1.
  - Structural rejections, each isolated and matched by message:
    - transport with truncation, one test per family (leave one letter out of `update`, all four truncated);
    - five bad `update` strings;
    - old above output, and output above the frame;
    - output ranks given one-sided;
    - old or output boundary ranks ≠ 1;
    - rank lengths;
    - the frame layout, and the tucker / tt variation shapes;
    - unequal group stores in the old, the output and the frame, each alone.
- **Mutation run** (scratch; the file was restored and checked): 18 mutations of the step, each turning T1–T3 red.
  - Content: drop the Ξ block, drop `g_new`.
  - Message terms: drop a message term in each of the left, right and up sweeps; reduce the down `Ξ` to `VP` alone.
  - Edge terms: drop the near term of the left, right and down edges; skip `NEW_T`.
  - Transport ignored; the movement halved; group members untied.
  - Each of the five structural checks deleted.
- **T16** (`test_align_padded_frame_dispatch`): four cases, a truncating step, a shared two-colour half (`UL`), the
  empty old frame, and a stack C = (2,) (round 2; the movement has shape C, and the frames are checked per element).
  - numpy in gives numpy out.
  - jax in jits (ranks, sharing, update static) to pure jax.
  - The movement equals numpy's to 1e-12, and the frames agree gauge-invariantly through `P_{T_F} Y` (1e-12
    relative).
  - About 30 s.
- **The contraction suite** passes with the new rows: vocabulary completeness, the definitional loop oracle,
  call-site consistency, and the sharding sweep.

## Judgment calls

| # | issue | decision | why |
|---|---|---|---|
| 1 | §6 T1 lists "d = 1" against the dense update. | The T1 d = 1 case starts at the zero tensor with target Tucker rank 1; the nonzero-old d = 1 increase is in T3's validity-only cases. A comment in the test says why. | At d = 1 the down store is one wide (`nD = min(nU, rL·rR) = 1`), so a nonzero old point's down frame spans the whole far side and the up edge's content is identically zero: its completion is arbitrary (D-12), in the library and the prototype alike. The informative d = 1 step is the cone's, at the zero tensor. This also closes the plan's "not verified at d = 1" for the step. |
| 2 | §4.1: "equal s° / s¹ Tucker stores within each group". | Also the frame's own stores `s`. | A group's completion is read from its first mode, and every member's messages need that width; unequal frame stores would fail inside a contraction instead of with a message. |
| 3 | The boundary rule. | The old and the output `tt_ranks` must have boundaries 1, as `fv_padded_frame` (N1 round 2); the frame's are checked through its layout. Both output sequences or neither (`ValueError`). | One rule for all three store sets (§4.1: "boundary bonds 1 in all three"). |
| 4 | `ragged_scan` returns `()` for an empty sweep instead of a tuple of empty tuples. | The call sites read `ys or ((),) * n`. | That keeps d = 1 (empty sweeps) branch-free in every body; the guard is about the host loop's return shape, not the mathematics. |
| 5 | The movement's type at C = (). | numpy's scalar (`np.float64`), as `fv_frame_orthogonality_residual` returns. | It is the reduction's natural output; the dispatch row accepts `np.generic`. |
| 6 | The dispatch row's eager-jax check. | Dropped for this row; the jitted check stays. | The first eager-jax call of the step costs about 49 s on jax 0.10 (each primitive compiles once per shape), while jit takes 6 s, and the module's own docstring calls jit the strong proof. **For the session:** an eager-jax user of the ragged step pays that first-call cost; the uniform step (N5) has its jit boundary for this reason. |
| 7 | The prototype's `distances` uses `‖xxᵀ − yyᵀ‖₂`, an SVD of a `prod(N)²` matrix (minutes at d = 5). | The oracle uses `‖x − y(yᵀx)‖₂`: the same number for equal widths, 1.0 otherwise. | It keeps T1 at seconds. |
| 8 | Transport at the step level. | T1 checks `update` subsets against the dense transport now; T4's loop-level two-colour and the cycle case stay in N3. | The step's `update` is N2's; the oracle needed it anyway. |
| 9 | The dense movement. | Added to the oracle: `‖T − WWᵀT‖_F` per edge with `T = Qᵀ P_old`. | The library's `g_old` is the old completion projected into the new candidate space, in that space's coordinates, so the two norms agree; this pins the movement against a dense definition, not only the frames. |

## Round 2 (review `N2_review.md`: REVISE, two MINORs; the judgment calls accepted)

| finding | fix |
|---|---|
| MINOR 1: `_fold(F, like, axis)` read an array only for its shape | `_fold(F, core_widths, axis)`; the three callers pass `A.shape[-3:]` of the arrays they use. |
| MINOR 2: the T16 row had no stack | A `C = (2,)` truncating case under jit: movement shape C, equal to numpy to 1e-12, frames per element through `P_{T_F} Y`. |
| NIT 3: the bodies' `x` comments lacked shapes | Every map and scan body's input comment now gives each element's shape (the P-part slices name the old widths cut). |
| NIT 4: the eager-jax first call | **For N8 / N9:** the reviewer measured 37.7 s for the first eager call of the step on jax arrays at d = 4, against 0.4 s on the second call and 0.024 s in numpy. The frontend twins or the user docs must say to jit the step on jax, or the twin could jit internally. N5's uniform step has its jit boundary for this reason. |

Rerun after round 2: `test_padded_frame.py`, `test_padded_start.py`, `test_import_graph.py` and
`test_contractions_interpreter.py`: 64 passed, 3,902 subtests. The padded-frame dispatch rows: 3 passed, 9 subtests.
`padded_frames`' doctests pass. The full suite was not rerun: round 2 changes one private helper's signature,
comments and one test row. The full-suite result below is round 1's.

## In the plan: wrong or ambiguous

- §6 T1's d = 1 against the dense update is degenerate unless the old point is the zero tensor (call 1).
- §4.1's group check names only s° and s¹ (call 2).
- T2's float32 bound "30·u·max(1, s)" does not define u; I took the unit roundoff eps/2. Measured about 2.5u, so the
  bound has a 10× margin either way.
- §3.1(6) and §3.5 say nothing about eager-jax cost on the ragged layer (call 6).

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n auto` (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2), the complete
`tests/` tree on the final working tree: **1,233 passed, 4 skipped, 48,195 subtests passed, 0 failed** in 20 min 45 s
on 8 workers.
- The one warning is the existing `adam`-on-MANIFOLD `RuntimeWarning` from the oracle sweep.
- N1 had 1,217 passed / 47,834 subtests; N2 adds T1–T3, the T16 row and the HISTORICAL rows.
- `padded_frames`' module doctests pass.
