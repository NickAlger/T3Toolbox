# Critic review r2 — `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (round 1, CHANGES rows 1–30)

Scope: every change listed in CHANGES rows 1–30 and what it touches, checked against my r1 checklist. Particular
scrutiny went to D-33, D-35, D-36, D-37, the jitted kernel, the empty frames, `_misfit_gradient_at`, Appendix C and the
fourteen keywords. Claims marked **(checked)** were run at `a2e26ca8`.

Summary. The round fixes all five r1 MAJORs and nearly every MINOR, and it fixes them well: the T3-free seam through
two members, the zero-tensor residual, `stats`-only records, the kernel and closure-free bodies, and the
unverified-items table. One new MAJOR: D-37's check orthogonalizes `x` to compare structure, the exact pattern Nick
paused the arc for. Nine MINOR, six NIT.

## Status of the r1 checklist

| r1 item | status |
|---|---|
| 1 seam | resolved (D-25): members `with_frame`, `aligned_padded_frame`; uniform `x_old` full `.data`; old frame from its own masks; `from_frame` plain masks = the frame's up/left; level widths from `var_masks`. New issue: finding 1 (D-37) |
| 2 cone residual | resolved (§2.5, §3.5, T14(d)) |
| 3 `NewtonInfo` / G-bit | resolved (D-27) |
| 4 compile | resolved (D-34, T13). Clarification: finding 4 |
| 5 unverified paths | resolved (§2.3, §10, T14(e)–(g)). T14(g)'s criterion is ill-posed: finding 5 |
| 6 projected content, guard | resolved |
| 7 `pad_safe_pinned_completion` | resolved (D-30); matches `gs_complete` with the offset expressed as seed positions |
| 8 signatures | mostly resolved; §5.7 still abbreviates three: finding 7 |
| 9 `sharing` labels | resolved |
| 10 naming | resolved (`fv_` / `ufv_aligned_padded_frame`) |
| 11 uniform step takes ranks | resolved |
| 12 transport + truncate | resolved (T3) |
| 13 loop edge cases | resolved (D-36). The uniform no-op level: finding 3 |
| 14 zero old point | resolved (D-21, D-29; rank-0 construction **(checked)** for UT3 too). The backend spelling of a rank-0 `padded_from`: finding 2 |
| 15 result classes | resolved (D-28) |
| 16 keywords | resolved (D-33); see "The four calls" |
| 17 frontend validation | resolved (§3.6). One gap: finding 6 |
| 18 frontend loop placement | resolved (D-35); see "The four calls" |
| 19 widths | resolved (`ufv_frame_widths` = the superseded rule `nD = min(nU, r²)`; T17) |
| 20 group check | resolved |
| 21 per-element `oversampled_ranks` | resolved (T19) |
| 22 tests | resolved, except T14(g): finding 5 |
| 23 HISTORICAL | resolved (Appendix C) |
| 24 T5 reference | resolved (frozen in N1) |
| 25 claims, docs | resolved (the 1.4 % / 0.5 % figures **(checked)** against `cycle/README.md`) |
| 26 slices | resolved |

## The four calls

**D-35, the frontend loop per layer and no frontend `padded_start`: agree, with one correction to its reasoning.**
- The placement is cycle-free **(checked)**: `frame_variations_format` imports `tucker_tensor_train`; the uniform
  module imports `uniform_tucker_tensor_train` and `frame_variations_format`; nothing imports back. It follows the
  same-named frontend/backend pair convention (`t3_` / `ut3_orthogonal_representations` in the same two modules).
- Dropping the frontend `padded_start` is consistent with D-8: Nick enumerated the twins on 2026-11-09 and left it
  out. `newton_cg(padded_from=)` is the user path.
- The correction: "the loop twins cover it" is not true for the problem's gradient. The twins need a `project`, and
  the problem-gradient projection is the private `_misfit_gradient_at`. So a frontend user who wants a problem's
  aligned frame without running Newton goes to the backend: `least_squares_problem(...)` → `.problem`, `.x0_cores` →
  `padded_start` → rewrap.
- That is acceptable only if the backend `padded_start` is public (in `__all__`, which the plan does not say) and its
  kept doctest shows that route. State both, and correct D-35's sentence. Nick may still want the twin; flag it to
  him as the one case it serves.

**D-33, `None` defaults, and an option without its group raises: agree.** It is the house rule as
`numerical_contracts.md` words it ("an option where it would have no effect … structural": `curvature_damping` with
Gauss-Newton, `tr_radius0` with the line search). One case the plan misses is finding 6.

**D-36, the truncating pass counts against the cap: agree.**
- In the normal case it is exactly what the prototypes did: `range(max_passes)` with the truncating pass at
  `it = n_over`. Only the `n_over ≥ max_passes` edge differs, and the g06 / README pass counts are unaffected.
- Clamping `n_over` to `max_passes − 1` is preferable to raising, because the loop-level defaults are literals and
  cannot tell a set value from a default.
- The user doc should state the clamp.

**D-37, `local_model(frame=)` compares with `geom.frame(x)`: does not hold.** See finding 1.

## The requested checks

- **The jitted kernel (D-34).** The design is sound:
  - one module-level kernel;
  - `skeleton` is a `StaticSkeleton` (`ValueHashedFields`, hashed over mask bytes), with `split` and `update` as
    hashable `static_argnums`;
  - output masks are identical across passes at one structure, so the cache hits;
  - masks ride scan inputs as constants, and `pad_safe_svd` accepts traced masks.

  Two clarifications are finding 4.
- **The empty frames.** `fv_empty_frame`'s shapes equal `t3_orthogonal_representations` of the rank-0 T3 **(checked)**:
  up `C+(0, N_i)`, down/left/right `C+(b_i, 0, b_{i+1})`. `ufv_empty_frame` matches `uniform_start._old_block_layout(None)`.
  Reading `shape`, `C`, dtype and backend off `old_frame` removes every "read only when None" argument. Signature
  style: NIT 10.
- **`_misfit_gradient_at`.**
  - Module-level, data explicit through `hint = (problem, residual)`, T3-free (`with_frame`, `kind.precompute` /
    `transpose`, `geom.project` / `precompute`). It is called outside the kernel, by the host loop.
  - It reproduces `_zero_reference_gnorm`'s formula, the kind's ω² inside `transpose`.
  - Its `geom.precompute(frame)` is the shared companion at a padded frame, marked not verified (T14(f)).
- **Appendix C (checked).** All 31 strings and the 16 d-twins parse and agree with `np.einsum` under `AXIS_SIZE`
  (one `C` axis, sizes from the table). Each matches its prototype einsum letter for letter, up to renaming. The three
  "exists" claims (`Cia,Cib->Cab`, `Cia,Cab->Cib`, `Caxb,Caub->Cxu`) and the one existing d-twin
  (`dCaxb,dCaub->dCxu`) are true, so 28 + 15 = 43 new HISTORICAL rows. Every letter is in `AXIS_SIZE`.

  Ten strings pair `a` and `k` (both size 2), so the oracle cannot catch an `a`/`k` transposition by shape. 163 of the
  260 existing strings share that property, and the file's distinct-size comment governs new letters only, so this
  is not a finding.
- **The fourteen keywords.** Under D-6 (every knob a keyword, separate groups) and the rejected `PaddedFrameOptions`
  (I-3), fourteen is the minimum.
  - The one smaller spelling that keeps every knob settable is compositional: `newton_cg(start_frame=padded_start(...))`
    for the start, and `g0norm_newton=` for a custom cone. It needs a frontend `padded_start` (dropped by D-35), and it
    loses the `'cone'` label in the history.
  - So the two choices interact: dropping the frontend `padded_start` is what makes the fourteen keywords the only
    frontend route. I recommend keeping both as written.

---

## MAJOR

### 1. MAJOR — §3.5, §5.6, D-37: the `local_model(frame=)` check orthogonalizes `x` to obtain structure

**Problem.** D-37 compares `frame` with `self.geom.frame(x)` ("one orthogonalization, only when `frame` is given").
That is computing a frame only to read its shapes and masks — the pattern Nick paused the arc for (HANDOFF,
2026-11-02: "an expensive orthogonalization to obtain integers that `ranks.compute_orthogonal_representation_ranks(shape,
tucker, tt)` already gives, statically"). The cost (one SVD sweep per run) is not the issue. The rule is: structure
from integers and masks. The text nowhere says otherwise; §3.5 calls it "an exact comparison with
`self.geom.frame(x)`".

**Fix.** A third optional member, `require_frame_structure(x_cores, frame)` (the superseded plan's §3.4 name).
Absent → `local_model(frame=)` raises naming it. Integers and held masks only:

- **ragged**: `frame`'s core shapes equal the stores `compute_orthogonal_representation_ranks(shape, ranks of x)` gives
  (shape and ranks read off `x`'s core shapes), plus the stack `C`;
- **uniform**:
  - `ufv_variation_masks(frame[5])` equals `self.var_masks`, array by array;
  - the two boundary entries `frame[5][2][d]` and `frame[5][3][0]` equal `prefix_mask(1, width)`;
  - `frame[4] == self.shape`;
  - the supercore widths equal the mask widths.

Then D-37, §3.5, §5.6's comment ("structurally equal to `geom.frame(x_cores)`"), D-25 ("two members" → three),
§3.7, `padded_frame_internals.md`'s seam paragraph and T14(a)'s mutation check (it deletes the member's comparison)
follow. The frontend `start_frame` check uses the same member.

---

## MINOR

### 2. MINOR — §3.5, §3.6, D-21: a rank-0 `padded_from` has no backend spelling on the uniform layer

In the backend, `newton_cg(padded_from=None)` means "off", while `padded_start(x_old=None)` means "the zero tensor".
A frontend rank-0 `padded_from` (a zero start with an aligned frame) must reach the member as data:
- ragged: `self.frame(x_old)` works on rank 0 **(checked)**;
- uniform: `ut3_orthogonal_representations` raises on it **(checked)**.

§3.6's "the twin detects it and uses the empty frame directly" covers the frame twins, not `newton_cg`. **Fix:** the
uniform member detects a rank-0 `x_old` structurally (all up masks empty, interior bond masks empty) and builds
`ufv_empty_frame` from it; both members accept the rank-0 data. `None` stays only on `padded_start` and the members
for the cone. Add a T14 row: `newton_cg(padded_from=<rank-0>)` on both layers.

### 3. MINOR — §2.3, §3.6, §4.6: the uniform loop's "nothing to complete" return breaks its own return contract

"Both backend loops return the old frame unchanged with no movements." `ufv_aligned_padded_frame` promises a frame
"compacted to `level_widths`, prefix masks". The old frame as given may have other widths and gappy masks, and
`newton_cg(padded_from=)` at equal ranks would then fail D-10's check-and-reject. **Fix:** the uniform early return is
`ufv_compact_frame(old_frame_data, level_widths)`; add it to T6 / T12's edge cases.

### 4. MINOR — §3.4 (jit boundary), D-34: say which host data is static

`partition_static` keeps Python ints / bools and host BOOL arrays static, but traces integer numpy arrays
(`common._is_static_leaf`: "an integer numpy array … stays traced"). So the group tables (`members`, `first`,
`group_of`, int arrays in the prototype) would arrive traced. Any host-side use inside the kernel, such as
`mU[first]` in the prototype's up-edge code, then fails.

**Fix.** The public function precomputes every host-derived mask (per-group A-masks, row masks, one-hots) as bool
arrays, and passes the index tables as static tuples of ints, or only as gather indices on dynamic arrays. State it in
§3.4. Also: `_jitted` is private to `backend/optimizers.py`, which `uniform_padded_frames` must not import (§3.7).
Write a local memoized wrapper (an `lru_cache`d getter), or move `_jitted` to `common`.

### 5. MINOR — T14(g): FD along the retraction from a padded start frame is not a valid criterion

`x0` (zero-padded) is rank-deficient and not on the target-rank manifold. There the retraction "is additionally not
first-order accurate" (`uniform_equivalence_contract.md`, `numerical_contracts.md`), and the default damping gives the
new directions "no curvature at all". `Problem.local_model`'s doctest recipe holds "on a minimal, numerically
full-rank frame", so "≤ 1e-4 relative to FD" at `F` would fail or be meaningless. **Fix:** split it.
- (g1) At a numerically full-rank `x` with a gauge-rotated frame `F` of `x` (same structure, so the check accepts
  it): `local_model(x, frame=F, hessian='exact')` passes the doctest's FD criterion, and its quadratic of a matched
  direction equals the default frame's.
- (g2) At the padded `x0`: finite, and `newton_cg(hessian='exact', start_frame=F)` decreases the objective.

### 6. MINOR — §3.6, D-33: `cone_*` options under `g0norm_newton`

With `g0norm_newton` given, `'cone'` is not computed (§3.5, T14(d)), so an explicitly set `cone_*` option has no
effect. By D-33's rule that is a structural error. **Fix:** say so in §3.6 and T14(d), or state the deliberate
exception.

### 7. MINOR — §5.7: three frontend signatures still deferred to N8

`UT3Frame.padded`, the frontend `ufv_aligned_padded_frame` (whose `level_widths=None` default is a real contract) and
`UT3Tangent.align_padded` are comment one-liners "written out in N8", and the frontend `newton_cg` / `gradient_descent`
keyword blocks are summarized. The plan is what Nick signs off. **Fix:** write them out in §5.7, in the layout of the
ragged twins.

### 8. MINOR — §3.5, D-35: is `padded_start` public?

It is listed as an untwinned backend function that keeps its doctest (§3.6), and D-35 relies on it as the route.
**Fix:** add it to `backend/optimizers.py`'s `__all__`, with its signature comment and doctest showing the
`least_squares_problem` route (see "The four calls"). Keep `_misfit_gradient_at` private, or make it public as
`misfit_gradient_at` if Nick wants the problem gradient usable as a frontend `project`.

### 9. MINOR — T14(d): "equals `'zero'`" needs a tolerance

At the empty frame the start's TT and down cores equal `fv_zero_point_frame`'s exactly, but its Tucker factors only
to 1e-14 (§2.4, T5). So `'cone'` with `cone_oversampling=0, cone_max_passes=0` equals `'zero'` to rounding, not
bitwise. **Fix:** state ≤ 1e-12 relative, consistent with D-24.

---

## NIT

10. §5.1 / §5.3: `fv_empty_frame`, `fv_padded_frame`, `ufv_empty_frame` return the `FrameData` / `UFrameData` alias
    on one line. The gold standard (`t3_zeros`, `fv_zero_point_frame`) expands return tuples one element per line.
11. §3.5: `uniform_fitting._base_point_with_frame_masks` does not exist; the function is `_frame_theta`
    (`uniform_fitting.py:76`).
12. The defaults 5 / 2 / 10 / 0.05 / True are literal in seven signatures: both loops, both members, `padded_start`,
    both frontend twins. Name them once in `padded_frames.py` (e.g. `ALIGN_OVERSAMPLING`, …), referenced by the
    loops, the members and the twins. `padded_start` takes `None` and forwards only what is set, as it already does
    for `seed`.
13. T18: the package-level edge `backend.optimizer_display → from t3toolbox import corewise` makes a naive module
    scan report a cycle through `t3toolbox/__init__`. Define the graph on module files, not on the package.
14. §2.3: "With `oversampling = 0` or maximal ranks there is no oversampled phase". Add "and `oversampled_passes` is
    then ignored", since the clamp in D-36 is the only other place a set value is not honoured.
15. D-18's text still says "named contractions". It is Nick's wording; a bracketed "(grouped `contract` strings,
    §3.1(4))" avoids an implementer reading it literally.

---

## Critic's notes for round 3 (check first; then whatever the changes touch)

1. `require_frame_structure` (or an equivalent integer/mask-only check): ragged stores from ranks, uniform
   `var_masks` + boundary entries + shape + widths; D-37, §3.5, §5.5/§5.6, D-25, §3.7, T14(a) consistent.
2. Rank-0 `padded_from` through the backend `newton_cg`, both members; T14 row.
3. Uniform no-op level compacts; T6 / T12.
4. Static vs traced host tables in the kernel; the local jit wrapper.
5. T14(g) split into (g1) full rank / (g2) padded.
6. `cone_*` with `g0norm_newton`.
7. §5.7 written out.
8. `padded_start` in `__all__`; its doctest route; D-35's sentence.
9. T14(d) tolerance.
10. NITs 10–15.

VERDICT: REVISE
