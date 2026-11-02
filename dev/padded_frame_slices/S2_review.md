# S2 review (round 1): `fv_padded_frame`, the hint-free path

Reviewer, 2026-11-02. Scope: the working-tree diff of `backend/fv_operations.py` and `tests/test_dispatch.py`
(its S2 row; the S1 row it still carries was reviewed in S1), and the new `tests/backend/test_padded_frame.py`.
Checked against plan §2.1, §2.5, §3.2, §4.2-4.3, §5.2, §6 T2 and T10 (the hint-free row), and §12.

## Overall

This is a clean, correct implementation of the hint-free construction. On both the mathematics and my probes I
found no BLOCKER, MAJOR or MINOR. The structural code path matches §4.3 line by line:

- the stores come from the shapes;
- the empty old frame stands in for `s = 0`;
- one keyed Haar factor is drawn per group and completed against `U°_g`;
- the TT and down cores take outside-first units, then the in-block completion from coordinate seeds;
- the carriers are embedded only.

The tests cover T2(a)-(f) with the plan's tolerances and use bitwise comparison only where the plan allows it.
All judgment calls are sound. I approve with NITs.

## Findings

1. **NIT: `fv_operations.py`, `fv_padded_frame` signature (the `old_frame`, `hint`, `seed` and `down` return
   lines; 123-145 characters).** These lines run past the ~120 the module otherwise keeps. The plan's §5.2 text
   is itself this long, so this is optional; you could move the long trailing comments to the docstring or wrap
   them.
2. **NIT: `_frame_stores` (validation).** It does not require the boundary bonds `rL_0, rL_d, rR_0, rR_d` to be
   1. A template with boundary 2 passes admissibility (`rL_1 ≤ rL_0·nU_0`) but has no meaning as a frame. The
   library never builds one (orthogonal representations squash the tails), so a one-line structural check is
   optional. `fv_zero_point_frame` does not make this check either.
3. **NIT: `tests/test_dispatch.py`, `test_padded_frame_dispatch`.** T10 also names "rtol `1e-5` float32" for the
   jitted constructor. Only x64 is exercised, because the module runs with x64 on. Optionally add one float32
   template through the jit with a `1e-5` comparison. Equivalently, defer it to S3's jit row, which T10 also
   covers.
4. **NIT: tests, consistency.** Determinism of two calls on identical input is asserted to `1e-15` in
   `test_tt_only_increase_inside_the_block`, but bitwise in `test_tucker_completion_is_nested` (modes 0 and 2)
   and `test_template_values_are_not_read`. Both readings are defensible. Under the plan's rule, "same computation,
   same input" is a structural fact, so bitwise is fine. Pick one if you touch the file again.
5. **For the orchestrator.**
   - (a) `tests/test_dispatch.py` carries S1's row as well. Either commit it as an S1 fixup first, or name it in
     the S2 commit message, so that `git log` does not misattribute it.
   - (b) The draft commit message has two `Co-Authored-By` trailers (Fable 5.1 and Opus 5.5). Settle it to the
     model(s) actually used, per Nick's rule.

## Checked and found correct

- **The embedding (§2.1).** `_embed` is a `pad_or_truncate` into the leading slots, so the old block is a copy:
  `U[:s] = U°`, the old columns of `L_k` (zero outside the `(sL_k, s_k)` block), the old rows of `R_k`, and the
  old down rows of `O_i`. The carriers `L_{d-1}` and `R_0` are embedded only, so they are zero on every new index.
  This is why both reconstructions equal `X_old`: every path through a new index ends at a carrier's zero.
- **Orthonormality per core.**
  - Outside units have disjoint support from the embedded block and are exact host constants.
  - The in-block columns are `pinned_completion` of the old unfolding.
  - The count bound holds: `sL_{k+1} + k_in ≤ sL_k·s_k` reduces exactly to template admissibility
    `rL_{k+1} ≤ rL_k·nU_k`, and likewise for `R` and `O`. So `s + k ≤ n` always holds in the primitive.
  - `O` is treated in its `(a, b)` bond space via `moveaxis`, and `fv_frame_orthogonality_residual` checks `D`
    over `(a, b)` (`'...iaj,...ibj->ab'`).
  - The `R` unfolding is `(n, b)` n-major as §2.5 specifies.
- **Tucker (§4.3).** `s_g = 0` is `_haar_rows` itself. For `s_g > 0`, the first `kU` rows of the same keyed stream
  are taken (nested, since `RandomState` is row-major) and completed against `U°_{min(g)}`. There is one array per
  group. `U°` is read from the group's first mode under the TIED precondition.
- **Outside-first rule and judgment call 1.** `c = 0` → the whole unfolding is outside. This matters only for the
  zero tensor's `d = 1` down core, the `(1, 1)` boundary block. Everywhere else an empty-column block already has a
  zero-size axis. It matches the plan's "at `s = 0` every coordinate is outside", and T2(a) at `d = 1` holds
  bitwise.
- **Coordinate seeds and gauge covariance (I-11, P17).** A sign gauge maps coordinate seeds to signed coordinates,
  so the spans are covariant. My fuzz used the test file's own `sign_gauge` and `projectors` over 300 random
  `d = 2..4` cases with Tucker and TT increases (29 "mixed" left cores with both outside units and in-block
  columns), two gauges each. The worst projector difference was **exactly 0**. The implementer's 0.32 probe for
  Gaussian seeds shows the test can tell the two seed rules apart.
- **Validity fuzz.** 400 random `d = 1..4` cases, minimal and non-minimal old points, increases of 0-2 per store:
  no `ValueError`. Worst orthogonality residual 1.7e-15, worst relative reconstruction 3.4e-15. A coordinate-aligned
  old core (identity Tucker and TT cores, bond 2 → 3) gives a valid frame (residual 4e-16): the E4 case completes
  validly.
- **Stacks and dispatch.** `C` is vectorized through one batched `pinned_completion`, and host constants are
  broadcast as copies (judgment call 3). The backend is inferred from `(frame, old_frame)` with no `use_jax`
  parameter, and the dtype follows the template (judgment call 4: casting the old frame is reasonable; the
  bitwise-copy claim then holds at equal dtypes, which is the only case the tests assert). `groups` and `seed`
  are static under jit.
- **Validation (judgment call 2).** The checks are structural and always on, each a `ValueError` naming the store
  and mode: consistency with neighbours, admissibility, `d`, `N`, stack, old ≤ template, group `(rank, size)` in
  both frames, and group index range. The hint is rejected with a `NotImplementedError` that names S3, before any
  other work. That is right for this slice.
- **Judgment call 6** is right. For a minimal `x_old`, `n_0 ≤ r_1` gives `sL_1 ≥ s_0`, so core 0 has no in-block
  room. A TT-only increase at bond 1 needs a non-minimal point, and §2.5's "the common case" wording is loose
  there (R5 already says core 0 has no room). The d = 4 bond-2 test is the natural in-block case.
- **Tests vs T2.**
  - (a) d = 1, 3, 4, unshared and two partitions, two seeds; coordinate cores bitwise, Haar to 1e-14, group
    identity. Stacked templates broadcast.
  - (b) all increase kinds, plus no increase and sharing.
  - (c) in-block columns equal the coordinate-seed completion to 1e-14; outside units exact in row-major order.
  - (d) nested.
  - (e) `C = (2,)`, `(2, 3)` with an in-block case.
  - (f) two in-block cases.

  Tolerances are explicit, with no `allclose` defaults. Bitwise comparisons appear only for structural facts and
  determinism.
- **Docstring and doctests.** The docstring covers §5.2's points for this slice; the hint section is a placeholder
  until S3, and the "See also" in `fv_zero_point_frame` leaves its body unchanged. The doctests are seeded and
  self-contained with real output; plan doctests (a) and (b) are present, and (c) and (d) belong to S3. Four
  doctests pass.
- **Runs.** `tests/backend/test_padded_frame.py`: 15 passed, 32 subtests. The dispatch rows (`pinned`,
  `padded_frame`) pass.
- **No behaviour change elsewhere.** The diff is additive. `fv_zero_point_frame`'s body is untouched.

VERDICT: APPROVE
