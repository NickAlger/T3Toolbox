# N2 review: the ragged alignment step

Reviewer: Claude Opus 5.5, 2026-11-09. I reviewed the uncommitted working tree on top of N1 (3f9679ef) against:
- the plan `dev/aligned_padded_frame_impl_plan_2026-11-09.md`: §2.2, §3.1, §4.1, §5.1, §6 T1–T3 and T16, Appendix C,
  §12;
- the prototypes `P/core_form2.py` and `P/cycle/core_form3.py`;
- the implementer's `N2_handoff.md`.

## Summary

The step is correct and in the house style. It agrees with the prototype to 1.1e-14 (largest principal-angle sine per
directed edge) over chained passes, including truncation, groups, a stack and five `update` subsets. It also runs on
the numpy 1.22 / jax 0.4.30 floor. T1–T3 are strong: all seven of my own mutations turned them red.

There are two small items to fix in this slice. One private helper takes an array only to read its shape. The T16
row has no stacked case, although the plan's T16 asks for one and N1's rows have one.

## Style (checked first)

- **Types and shapes.**
  - Every public input and output is a basic type: tuples of arrays, int sequences, the `sharing` labels, the
    `update` string, and a movement of shape `C`.
  - The frame is read for its shapes because it is used (its cores are the pinned blocks).
  - Splits inside the bodies come from operand shapes (`r = X.shape[axis] − o.shape[-1]`, §3.1(6)).
  - `k1` and the update flag ride the scan inputs as a Python int and bool, as §4.1 allows on the ragged host loop.
  - Exception: finding 1.
- **Scans.**
  - The bodies are module-level and closure-free.
  - The two message sweeps and the two edge sweeps are `xscan` folds with zero-size initial carries (`C+(0,1)` /
    `C+(0,0)`, and `C+(0,0)` / `C+(1,0)`). No body branches on a boundary.
  - The leftward sweeps run on reversed inputs, and their outputs are reversed back.
  - The `ys or ((),)*n` guard sits in the driver, not in a body. It handles `ragged_scan`'s `()` return for an empty
    sweep, which only happens at d = 1. That is acceptable (judgment call 4).
- **Contractions.**
  - `grep -nE "einsum|@|matmul|tensordot|\.dot\("` on `padded_frames.py` finds nothing. The one `.T` is N1's host
    Haar seeds.
  - The module uses 31 distinct `contract` strings. Each one is Appendix C's string under its label (c1–c31), and
    each is in HISTORICAL. That is 28 new rows plus three that already existed (c19, c22, c23).
  - Not contractions, all allowed by Appendix C: the thin SVD in `_root`, the movement's sum of squares, slices,
    `pad_or_truncate`, concatenation, and the core-axis unfold/fold reshapes. No reshape crosses `C`.
- **Signatures and docstrings.**
  - `tv_align_padded_frame` matches §5.1 line by line.
  - The backend docstrings are one line.
  - `VariationsData` is now used, so N1's rule is satisfied.
  - `__all__` gains `tv_align_padded_frame`.
  - The imports gain `contractions`, which is inside T18's allowed set; T18 passes.

## Correctness

- **Sweep orders (§2.2).**
  - Step 1: down (map), left messages from the far end (`j = d−1..1`), right messages from the near end
    (`j = 0..d−2`), up (map).
  - Step 2: up edges per group, the left sweep `i = 0..d−2`, the right sweep `i = d−1..1`, down.
  - Every slice and term matches §4.1 items 1–9. I checked the old widths of every `H`, `O`, `L` and `R` slice and
    the `NEW_*` axes against the prototype line by line.
- **Transport.**
  - A store not in `update` uses `Y = g_old`.
  - `g_old` is orthogonal to the embedded old block: the old block is zero off its own input slots, and the
    coordinate map is the identity on them. So `(I − ĀĀᵀ)g_old = g_old`, which is the oracle's
    `top_left(Qᵀ P_old)`.
  - A truncated store outside `update` raises.
- **Movement.** The direct form `‖g_old − P⁺(P⁺ᵀ g_old)‖_F` per edge, as in §2.2 and the prototype, never
  `sqrt(k − ‖o‖²)`. The maximum is taken over edges per stack element, giving shape `C`, with one up edge per
  sharing group.
- **Carriers.** `L⁺_{d−1}` takes `L_{d−1}[:rL°, :nU°, :]` and `R⁺_0` takes `R_0[:, :nU°, :rR°]`, zero-padded:
  old content only. T1's "both chains reconstruct X_old to 1e-12" would catch any new content there.
- **Old blocks bitwise.** Up: `concat([A, P⁺ᵀ])`. TT and down: a fold of `[unfold(embed(old block)) | P⁺]`. Both
  reshape the old values without changing them, and T1 asserts it.
- **The d = 1 case** (judgment call 1) is sound. At d = 1, `nD = min(nU, 1) = 1`. A nonzero old point's down frame
  therefore spans the far side, the up edge's content is identically zero, and the increase carries no information
  (D-12). The informative d = 1 step is at the zero tensor, and T1 runs it there.

## Findings

### 1. MINOR: `_fold` takes an array only to read its shape

`t3toolbox/backend/padded_frames.py:308–316`, `_fold(F, like, axis)`.

**Problem.** `like` is read only for `like.shape[-3:]`. This is the shape-template pattern of §3.1(1) ("no object is
passed to be read for its shapes") in a private helper.

**Fix.** Pass integers, for example `core_widths: typ.Sequence[int]  # len=3, the core's (a, n, b); the entry at
`axis` is replaced by the column count`. The three callers have the arrays at hand and use them, so they pass
`A.shape[-3:]`.

### 2. MINOR: the T16 row has no stacked case

`tests/test_dispatch.py`, `test_align_padded_frame_dispatch`.

**Problem.** Plan T16 asks for rows covering "numpy/jax in → out, ragged/uniform, stacked", and N1's
`test_padded_frame_dispatch` has `C = (2,)`. The step's row has three unstacked cases. I ran the step under `jax.jit`
on a `C = (2,)` truncating case by hand: it works, the movement has shape `(2,)`, it is 2.2e-16 from numpy, every
leaf is jax, and it compiles in 4.6 s. So this is a coverage gap, not a bug.

**Fix.** Add a `C = (2,)` case with the same checks: movement shape `C`, and the frames per element through
`tangent_projection`.

### 3. NIT: the scan and map bodies' `x` comments list names without shapes

`_left_message_step`, `_right_message_step`, `_up_message`, `_left_edge_step`, `_right_edge_step`, `_down_edge` and
`_up_edge` (partly).

**Problem.** The comments say which slice each name is (`H_P`, `O_P`, `PhiP`, …) but not its shape. The exemplar
`probing.py` does both. For the P-part slices the shape is the useful information, because it says which old widths
were cut.

**Fix.** Add the shapes where cheap, for example `# (H_P, bO, bL, R, PhiO, XiO): C+(kLj, nUj, rR(j+1)) ;
C+(kLj, kDj, rR(j+1)) ; …`.

### 4. NIT, for the session: the eager-jax first-call cost

Handoff judgment call 6. I accept dropping the eager row, because the jitted row is the stronger proof. But a
ragged-layer user who calls the step eagerly on jax arrays pays a large first-call cost. I measured d = 4, `(3,4,4,3)/(1,3,4,3,1)`: **37.7 s** on
the first eager call, 0.4 s on the second, against 0.024 s in numpy. N8's twins or N9's docs should
say "jit it", or the twin could jit internally. N5's uniform step already has its jit boundary for this reason.

## What I ran (env `t3toolbox`, py3.11, numpy 2.x, jax 0.10.2, unless stated)

1. **Against the prototype.** A scratch script, `compare_proto.py`, runs `cycle/core_form3.align_stores` and the
   library step from the same `fv_padded_frame` start, with the same dense hint projected at each frame, for two
   chained passes (three for the `update` subsets). The cases: d = 2, 3, 5; d = 1 from zero; non-minimal; truncating;
   zero and truncating; spread string groups; all-mode groups with truncation; `update` ∈ {UL, RO, U, O, LR};
   `C = (2, 3)` with truncation.
   - Largest principal-angle sine over all directed edges, via the oracle's `dense_frames` / `distances`:
     **1.1e-14**. Movement: **6.7e-16**.
   - On the first pass of the d = 3 case the cores themselves agree to 4.4e-15. Later passes can differ in gauge
     inside the completion blocks with the same subspaces, which is expected: frames are compared by subspace.
2. **Tests.** `pytest tests/backend/test_padded_frame.py tests/test_contractions_interpreter.py
   tests/test_import_graph.py -n 4`: **56 passed, 2,613 subtests**.
3. **Dispatch rows.** `pytest tests/test_dispatch.py -k "align or padded_frame or empty_frame or pinned"`: **4
   passed**. The step's row takes 14.7 s plus 7.5 s of setup.
4. **Doctests.** `pytest --doctest-modules t3toolbox/backend/padded_frames.py`: **1 passed**.
5. **Floor env** `tttt` (py3.9, numpy 1.22.4, jax 0.4.30): `test_padded_frame.py` **36 passed**. The zero-size SVDs
   of the empty edges work on old numpy. The three padded-frame dispatch rows: **3 passed**.
6. **My own mutations.** A scratch copy, with T1–T3 selected by `-k Step`. All seven turned the tests red:
   - skip `NEW_L` in the left edge's `g_old`;
   - drop the `c20` subtraction in the up message;
   - `_root` without the singular values;
   - the right-edge outputs not reversed back;
   - drop the down edge's `R`-term;
   - drop the `XiR` row in the right messages;
   - a 0.1 % movement error.
7. **Stacked jit by hand.** Finding 2's check.
8. **Eager-jax timing by hand.** Finding 4's check.

The full suite was not run; the implementer reports 1,233 passed, 4 skipped, 0 failed.

## Judgment calls

All nine are accepted:
- 1 (d = 1): sound, see above.
- 2 (the frame's own group stores): a structural error with a message instead of a failure inside a contraction.
- 3 (the boundary rule for all three store sets).
- 4 (the `ys or` guard).
- 5 (`np.float64` at `C = ()`).
- 6 (the jitted row only): see finding 4.
- 7 (the oracle's distance): the same number for equal widths.
- 8 and 9.

## Counts

MAJOR 0, MINOR 2, NIT 2.

VERDICT: REVISE
