# P16 part 1, response to review round 1

Implementer, 2026-11-21, answering `P16_part1_REVIEW_r1.md` and Claude's message. Nothing is committed. The handover is
updated (sections 1 to 4).

## 1. (Should fix) The two `test_dispatch` tests of the member's contract: done

- `TestPaddedStartSeam::test_members_and_padded_start` unpacks `(frame, centers, movements)`. The timed call has
  four elements; the frame, centers and movements are compared bitwise with the untimed call, and the centers are
  asserted to be jax on jax.
- `TestAlignmentTiming::test_timing_adds_no_program` asserts three elements off and four on, compares the centers
  bitwise like the frame, and moves the movements and seconds indices by one.
- Both pass (with the part's files: 187 passed, the 5 expected failures).

## 2. (Should fix) The expected-red list: replaced by the measured one

- `P16_part1_EXPECTED_RED.txt` (63 ids, and three page lines) is the reviewer's measured 66, less the four fixed in
  this round, plus one id that finding 3's check turned red.
- **That new id is `tests/test_fitting.py::TestLocalModel::test_uniform_regularizer`.** The uniform corewise frontend
  model hands the frontend's six-element frame `(B, G, G, G, shape, masks)` to the backend `point_tangent`, which
  read `(frame[0], frame[1]) = (B, G)` by position: right by luck. It now raises. It is a frontend model, so it stays
  red until part 4.
- **How I measured it.** The check can only bite where a frame of the wrong form reaches `base_point` or
  `point_tangent`; in the tree today that is the frontend models. I reran `test_fitting.py` and `test_sharing.py`:
  red exactly on the list's ids in those two files.
- **The pages.** `docs/fitting_and_optimization.md` fails at lines 73, 75 and 1190 (corewise `adam` and `newton_cg`
  through the old `Problem`). The other corewise pages pass.
- **`TestUniformGeometryOps`: fixed here (local).** Its corewise case now compares the backend point
  `(B_sc, G_sc, shape, masks)` with the frontend frame's `(B, G)`, shape and the down and left masks, and that the
  frontend's `U` is its `G`. The project, retract and inner comparisons are unchanged and pass. Both tests pass.

## 3. (Should fix, silent wrong answer) The old four-store frame: a structural check

- **The check.** `_require_frame_form(frame, expected, who)` reads the frame with `frame_form`, the engine's one
  reader, and raises a `ValueError` naming the member, the expected form and the form received. It runs in:
  - ragged corewise `base_point`, requiring `('ragged', 'corewise')`;
  - uniform corewise `base_point`, requiring `('uniform', 'corewise')`, which means four elements and an array
    first.
- **The other members, searched:**
  - **Corewise `point_tangent` and `retract`** read the frame only through `base_point`, so they are checked.
  - **The manifold `base_point`** (ragged and uniform) reads the frame by position. Handed a uniform corewise point
    (four elements), the uniform one returned `(B, shape)` without error. It now requires the manifold form.
  - **The manifold `point_tangent`** reads the frame directly (`fv_base_point_tangent`, `ufv_base_point_tangent`; on
    a uniform point an `AttributeError` deep inside). It now gets the same check first.
  - **Not checked, on purpose:** the members that do not read the frame at all. On the corewise geometries these are
    `project`, `precompute`, the exact-mode members and `zero_point_frame`; a frame passed there cannot change a
    value. The manifold matvec members (`project`, `retract`, the lift) unpack the frame's four or six stores
    themselves, so a wrong form raises there already, and they run inside CG.
- **Tests.** `test_a_frame_of_another_form_raises` covers:
  - the old four-store frame on both corewise geometries;
  - a corewise point on both manifold geometries;
  - each raising in `base_point`, `point_tangent` and `IdentityRegularizer.gradient`;
  - with the right form, the regularizer's value equal to the cores' weight decay (relative `1e-13`) and the
    manifold `base_point` returning `(D, L)`.

  Two mutants, each removing one corewise check, are killed.

## 4. The equivalent mutant

Agreed. `x0`'s own centers at the padded frame equal the zero-padded old centers, so that mutant is equivalent and
is not counted. The handover's table now says so. The killed mutants are genuine:
- trailing padding;
- zero centers;
- the uniform masking and gather;
- the `K` reshape;
- the two shared projections;
- the two form checks.

## Runs

- **Part files, `-n 2`:** the six part files plus `TestUniformGeometryOps` and the two `test_dispatch` classes give
  187 passed and 5 failed (the expected ones), with 1829 subtests passing.
- **`test_fitting.py` and `test_sharing.py`:** red only on listed ids.
- **Doctests:** the page doctests behave as listed above, and the module doctests and `test_import_graph` pass.
