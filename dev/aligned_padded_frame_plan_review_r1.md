# Critic review r1 — `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (draft 0)

Read for this review: the gold standard (`TuckerTensorTrain` / `UniformTuckerTensorTrain`: `resize`, `zeros`, `ones`,
`randn`, `stack`/`unstack`, `__add__`, `t3svd`, `share`, `continuation_ranks`, `rank_adjustment_sweep`, `from_t3`,
`reverse`, `__mul__`, `inner`/`norm`, the orthogonalizations, traced to `t3_constructors`, `t3_operations`,
`tt_operations`, `t3_linalg`, `t3_svd`, `ut3_constructors`, `ut3_linalg`, `ut3_masking`, `ut3_svd`), `CLAUDE.md`, the
uniform and contributor docs named in the brief, the model plan, the draft, the prototype README + `cycle/README.md`,
`core_form2.py`, `cycle/core_form3.py`, `uniform_form2.py`, `uniform_start.py`, `uniform_loop.py`, `ragged_loop.py`,
the library code the plan touches, `dev/HANDOFF.md`, and the superseded plan's §3.4 and review r1. Claims marked
**(checked)** were run against the library at `a2e26ca8`.

Summary. The mathematics (§2.1–§2.2, §4.1) matches `core_form2.py` / `core_form3.py` step for step, and the measured
facts in §2.3 are mostly cited correctly. The problems are in the seams: the draft drops the T3-free optimizer seam on
a misreading of D-9, computes the cone reference at the wrong residual, re-introduces a defect the superseded plan's
review already fixed (history keys on default runs), has no mechanism behind its compile budget for the uniform step,
and asserts an unverified optimizer path without a gate. Five MAJOR, twenty-one MINOR, eleven NIT.

---

## MAJOR

### 1. MAJOR — §3.2, §3.5, §5.4 (the optimizer seam): no T3-free path; D-9 misread

**Problem.** `backend/optimizers.py` is T3-free by a documented invariant (`docs/contributor/curvature_internals.md`:
"Everything T3-specific is behind those members, so `backend/optimizers.py` still imports no T3 module";
`parameters_not_closures.md` likewise). The draft puts T3-specific work in it: `padded_start` (calls the ragged/uniform
loops, `UniformManifoldGeometryOps.from_frame`, the structural reduction of `x_old`, the level widths), the `'cone'`
branch of `_resolve_g0_reference`, and `local_model(frame=)`'s store check (needs
`compute_orthogonal_representation_ranks`). §3.2 then forbids the seam: "`backend/geometry.py`: no new member (the
geometry twins are dropped, D-9)". But Nick's decision (6) of 2026-11-09 (HANDOFF) reads "**frontend twins**: drop the
three geometry twins" — the three FRONTEND twins `ManifoldGeometry` / `UniformManifoldGeometry` /
`SharedGeometry.padded_frame` of the superseded §5.8, not the backend members. The superseded plan's §3.4 ("Why
members and not a T3-specific loop: … keeps that module T3-free") was never overruled. Two concrete bugs follow:

- (a) §3.5 "the old frame = `geom.frame(x_old reduced)`" is wrong on the uniform layer:
  `UniformManifoldGeometryOps.frame` (`geometry.py:645`) orthogonalizes `(x_sc[0], x_sc[1], self.shape, self.masks)`,
  i.e. with the LEVEL's masks, so an `x_old` at its own widths fails or is read through the wrong masks.
- (b) `padded_start(problem, x0: Cores, x_old: Optional[Cores])`: on the uniform layer `Cores` is the bare supercore
  pair, which cannot carry `x_old`'s own masks; the old frame needs them (the superseded §3.4 rejected a bare pair for
  exactly this).

**Fix.** Restore optional, duck-typed backend geometry members on `ManifoldGeometryOps` and
`UniformManifoldGeometryOps` (the `zero_point_frame` precedent; absent on corewise → a structural error naming the
member), each a thin binding of the standalone functions, for example:

- `aligned_frame(x_old_data, x0_cores, transpose_at, oversampling, oversampled_passes, max_passes, tol, alternate,
  seed) -> (frame, movements)`: the old frame from `x_old`'s own data (uniform: the full `.data`, masks included), the
  target ranks from `x0` / the held masks, the per-pass projection (`self.project` on ragged; on uniform
  `from_frame(F, (F[5][0], F[5][2]), groups)` — the plain masks are the frame's own up and left masks, gappy, NOT
  prefix masks of the counts), the level widths from `self.var_masks` shapes, the loop, the compaction;
- `require_frame_structure(x0_cores, frame)` for `local_model(frame=)` (D-10).

The T3-free driver passes only T3-free callables (`transpose_at = lambda F: kind.transpose(r, sample, F,
kind.precompute(F, sample))`). Name `x_old`'s uniform type (full `.data`) in every signature that takes it. If Nick
prefers to relax the invariant, the plan must say so and amend the two docs that state it.

### 2. MAJOR — §2.5, §3.5, §5.4: the cone reference is computed at the wrong residual

**Problem.** `'cone'` is `padded_start(problem, x0, None)`, and `padded_start` computes "the residual at `x0` once".
The zero reference it parallels uses `r0 = point_forward(0) − data` (`_zero_reference_gnorm`, `optimizers.py:645`),
and so does T3Polynomial's `cone.py:80`. At level ≥ 2, `x0 ≠ 0`, so the draft reports `‖P_{T_F} 𝒥ᵀ r(x0)‖` at a
frame of the zero tensor — neither D-5's quantity nor what the stop is calibrated to. T14 ("deterministic given the
seed") cannot notice.

**Fix.** The cone path builds `r0` exactly as `_zero_reference_gnorm` does (regularizer excluded, same reason), runs
the loop at the zero tensor of `x0`'s ranks with it, and reports `‖P_{T_F} 𝒥ᵀ r0‖`. Add to T14: with
`cone_oversampling=0, cone_max_passes=0` (the start only, which at the zero tensor IS `fv_zero_point_frame`) the
`'cone'` value equals the `'zero'` value; and at a level with `x0 ≠ 0` it does not depend on `x0`'s values.

### 3. MAJOR — §3.5, §5.4, T14: new `NewtonInfo` fields break the non-breaking gate; `stats['newton']` does not exist

**Problem.** History rows record every scalar `NewtonInfo` field (`_NEWTON_SCALAR_FIELDS`, `optimizers.py:1031`), so
`start_frame`, `start_gnorm`, `padded_passes`, `cone_passes` add keys to every row of every default run. That is the
superseded review r1's M2, fixed there ("field dropped; `stats[...]` only on runs that use a start frame; G-bit
compares whole stats dicts"); it contradicts Nick's binding rule (HANDOFF: "`newton_cg` without the new keywords
bit-identical incl. `stats`") and the plan's own G-bit row. Separately, `stats['newton']` was removed in 2026.3.0
(`optimizers.py:1236`), and `start_frame` would name both a keyword (a frame) and a field (a string).

**Fix.** No `NewtonInfo` field. Put `stats['start_frame'] = 'padded' | 'given'` and the padded start's
`(gnorm, passes, movements)` in `stats` only on runs that use a start frame, the cone's pass count only with
`'cone'`; history rows carry `g0_reference='cone'` (an existing field, a new value, only when requested). Default
`set(stats)` stays the five keys pinned at `tests/backend/test_optimizers.py:477, :860`. Then say what the display
(§3.2's `optimizer_display` line) shows: nothing per iteration, or one line from `stats` after the run.

### 4. MAJOR — §3.1(6), §3.5 compile accounting, D-20, T13: no jit boundary for the uniform step; the prototype's scan bodies would recompile every pass

**Problem.** Under D-20's host loop the uniform step runs eagerly on jax arrays whenever the problem's data are jax
(`use_jit=True` converts them, and jits only the inner CG). The prototype's four scan bodies (`left_msg`, `right_msg`,
`left_edge`, `right_edge` in `uniform_form2.py`) are closures over `Na`, `Rb`, `update`, `xp`, `es`. Eager
`lax.scan` keys its cache on the body's identity, so every pass would retrace and recompile all four
(`docs/contributor/scan_body_principles.md` §"The problem"). That breaks "repeated recompiles inside a loop never";
§3.5's "all fixed, none repeated" has no mechanism behind it, and §12 has no scan-body item. T13's "loop under
`jax.jit`" contradicts D-20 (a host loop branching on the movement cannot be jitted), and no test counts compiles
across passes.

**Fix.**
- §3.1 / §12: every `xscan` / `lax.scan` body module-level and closure-free; statics recovered from carry shapes
  (`Kb1, Kb` from `o`'s shape, `Rb = rT1 − Kb1`, `Na` likewise) or from an `lru_cache` factory keyed on hashable
  statics (the `update` flag), never on arrays.
- State the jit boundary: eager with hoisted bodies, or the uniform step jitted at a module-level boundary with its
  statics (split, completion widths, `update`, the host masks / counts / one-hot selections via
  `common.partition_static`) value-hashed. Then give the count per level: the start (no scan); step programs
  {oversampled `'ULRO'`, truncating `'ULRO'`, target `'UL'`, target `'RO'`}; gradient programs at the working and
  target block widths; all of it again for `'cone'`, whose A-block widths differ (`Na = 0, Rb = 1`).
- T13: drop "loop under jit"; add "an eager-jax uniform loop of ≥ 6 passes at fixed structure compiles nothing after
  the first pass of each colour" (count with `jax.log_compiles` or a trace counter, as `test_dispatch` does for the
  damping), `d ≥ 3`.

### 5. MAJOR — §3.4 (Loop), §3.5, §10, T14: the optimizer-path gradient at padded frames is unverified and ungated

**Problem.** The prototypes verified `utv_project_ut3_onto_tangent_space` at gappy block-layout frames (README,
"The remaining uniform pieces", Results) and the ragged problem gradient on Darcy (`g06`, ragged only). Never run:

- the uniform kinds' `precompute` / `transpose` at a block-layout frame — the uniform `padded_start` path, i.e. the
  T3Polynomial GPU path;
- the tied projection through `shared_data` at a padded frame, on either layer (README: "not used"). §10 defers it to
  N8, but `padded_start` with a shared problem is N7;
- `hessian='exact'` from a supplied start frame: the lift's exactness is established for frames from
  `t3_orthogonal_representations` (curvature plan F2–F3), not for aligned ones.

The plan states all three as design without marking them unverified, and T14 tests only same-layer
self-consistency. Cross-layer starts through the optimizer are also only gauge-equivalent (superseded review r1 M3):
the old frames come from LAPACK vs `pad_safe_svd`, and the start convention is basis-dependent, so a cross-layer test
must not expect equal frames.

**Fix.** Mark the three as unverified (§2.3 facts table, §3.4, §10). Add to T14:
- (a) `padded_start` uniform vs ragged on the `probe_derivatives` kind, `d ≥ 3`, compared by gauge-invariant
  quantities (per-pass `‖P_F g‖`, spans; tight at a tight tol, loose at the default stop);
- (b) a shared problem on both layers, one pass against a dense tied-projection reference;
- (c) `newton_cg(hessian='exact', start_frame=F)`: finite, and the iteration-0 exact quadratic matches FD along the
  retraction from `F` (the recipe of `Problem.local_model`'s doctest).

---

## MINOR

### 6. MINOR — §3.4, §4.4: the uniform factorization is specified on unprojected content

The σ claims ("pins first (σ = 1), content σ ≤ 1/4") hold only if `Y ⟂ Ā`. `g_old` is orthogonal to `Ā`, but `g_new`
is not (`D = [Ā | P⁺]ᵀ g_new` carries an `Ā` part, and ungauged variations add more). The prototype projects first
(`uniform_form2.content`: `Y − Ā(ĀᵀY)`; `transported`: `g_old − Ā(Āᵀ g_old)`), as `pinned_leading_columns` does
inside on the ragged side. Followed literally, §4.4 builds wrong frames (T9 would catch it). Also: `Y = 0` (a zero
hint, D-12) needs the guarded scaling (`factor`'s double `where`), not `0/0`. **Fix:** write `Y⊥ = (I − ĀĀᵀ)Y` and the
guarded `Y⊥/(4‖Y⊥‖_F)` in §3.4, §4.4 and for transported stores.

### 7. MINOR — §5.2, §3.4: `pad_safe_pinned_completion`'s signature cannot express its two uses

The start needs per-element counts (they vary over modes and stack elements: `kU[first]`, `k_in`), and the TT / down
stores place the inside completion after the `t` outside unit vectors (`gs_complete(..., k, offset=t)`). A static
`k: int` with "`k ≤` real rows − real pins" is false for elements with fewer real rows, and there is no offset. The
real-support property also needs inputs zero on junk rows, so "junk columns anything" must become "junk entries
anything; masked at entry". **Fix:** return all `m` completed seed columns, `C+(n, m)`, with the contract "column `j`
is real-supported (bitwise zero on junk rows) when `j < n_real − s_real`, per element; later columns don't-care", and
let the caller select and place by counts (keeps the primitive a pure factorization); or add `counts` and `offset`
(HOST int, `C`). Put the real-support argument in the justification (inputs zero on junk rows, junk rows trailing ⇒
every reflector vanishes there ⇒ those columns are bitwise zero there), alongside: no content decides a rank, every
returned column is real by count, Householder is deterministic. That is what the SVD-only note's two properties ask
for.

### 8. MINOR — §1, §5, §8: signatures not in the house style, and inconsistent

- §5.3 `ufv_aligned_padded_frame` (several arguments per line, no annotations, "as the ragged loop"); §5.4
  `Problem.local_model`, `padded_start`, `newton_cg` (abbreviated); every §5.6 frontend twin (one-liners, no
  annotations or shape comments). Return tuples are not expanded one element per line (`fv_padded_frame -> Frame`,
  `utv_align_padded_frame -> Tuple[UFrame, NDArray]`; compare `fv_zero_point_frame`).
- Uniform parameter names: `frame` / `variations` in §5.3 vs `frame_data` / `variations_data` in §1; the naming rule
  is `*_data` for raw uniform tuples.
- `out_widths` is `(Ka1, Kb1)` in the step and `(WU, WD, WL, WR)` in the loop. Rename the step's
  `completion_widths` (as `ufv_padded_frame` has) and the loop's `level_widths`.
- `variations: Tuple[NDArray, NDArray, ...]`: say "variations `.data` `(V, H, shape, masks)`; only `V`, `H` read,
  masked with the frame's gauge-shifted masks".
- `seed=...` placeholders.

**Fix:** every new or changed signature in full in §5; §1 either copies them or becomes a name list pointing to §5.

### 9. MINOR — §5.1, §5.3: public backend entry points take `groups`; the gold standard takes `sharing`

`t3svd`, `ut3svd`, `ut3_rank_adjustment_sweep`, `compute_minimal_ranks`, and `oversampled_ranks` (called inside
the same loop) take `sharing: Optional[Sequence]` (`len=d` labels). Canonical `groups` appear on internal helpers and
geometry fields. **Fix:** `sharing` on the public start / step / loop of both layers, `canonical_groups` inside.

### 10. MINOR — §1, §5.1, D-22: the ragged loop's name breaks the prefix grammar

`docs/naming_conventions.md`: public backend functions carry a family prefix, and an unprefixed name means
representation-agnostic infrastructure or polymorphic ("the ragged name is the polymorphic name"). The loop is
neither (D-15). **Fix:** `fv_aligned_padded_frame` (frame in, frame out; the uniform `ufv_…` already conforms). This
settles D-22.

### 11. MINOR — §5.3, §5.6: the uniform step takes per-edge counts; the ragged step takes ranks

`utv_align_padded_frame(..., new_counts=(kU, kD, kL, kR))` makes the caller do the store arithmetic
`compute_orthogonal_representation_ranks` owns, and `UT3Tangent.align_padded(split, new_counts, ...)` hands it to
users, while the ragged twin takes `tucker_ranks, tt_ranks`. **Fix:** take target ranks (`len=d` / `d+1`, or `(d,)+C`
/ `(d+1,)+C` HOST int), default the frame's own; derive the counts as stores(target) − A-block mask sums. The same for
the frontend twin.

### 12. MINOR — §2.2, §4.1, §4.4: "a pass may not both transport and truncate" must be a structural error

The loop never does it, but both steps are public with `update` and output ranks or widths, and the README shows the
result is arbitrary ("Not matched, validity only"). **Fix:** `ValueError` when a store not in `update` has an output
count below its input count; validate `update` ⊆ `{'U', 'L', 'R', 'O'}` without repeats; add a rejection test.

### 13. MINOR — §2.3, §4.3, §5.3: loop edge cases and the movement record

- (a) `oversampled_passes ≥ max_passes`: the prototypes then run `max_passes + 1` passes (an extra truncating pass,
  `ragged_loop.py`, `uniform_loop.py`), contradicting "`max_passes` passes in all". Decide (structural error, or the
  truncating pass counts against the cap) and state it.
- (b) When working = target (`p = 0`, or ranks already maximal) the prototype skips the oversampled phase whatever
  `oversampled_passes` says; on a stack, when every element's working = target. State it.
- (c) The alternating stop counts the truncating pass as the first of its "two consecutive passes" (the prototype's
  `moves[-2:]` at `it = n_over + 1`). State the rule exactly.
- (d) The ragged loop returns a tuple of movements; the uniform one a NaN-padded `(max_passes,)+C` array (the
  prototype's is `(max_passes + 1,)+C`). Under D-20 the fixed-length array buys nothing: return
  `Tuple[NDArray, ...]` (each `C`) in both layers.
- (e) Is `max_passes = 0` legal? (It is the start itself; finding 2's test uses it.)

### 14. MINOR — D-21, §5.1, §5.3, §5.6: spelling the zero old point as `None` leaks array-free parameters everywhere

`old_frame=None` forces "read only when `old_frame` is `None`" parameters (`stack_shape`, `use_jax`, `N`) onto the
start and both loops, puts `None, None` on the step's old ranks, and leaves `T3Frame.padded(None, ...)`,
`UT3Frame.padded(None, ...)` and the frontend loop without a shape. Only the loop gets D-21's `shape=` patch.

**Recommendation (checked).** The zero tensor is a T3 at Tucker ranks 0 and TT ranks `(1, 0, …, 0, 1)`.
`TuckerTensorTrain.zeros(shape, (0,)*d, (1,)+(0,)*(d−1)+(1,), stack_shape)` constructs, and
`t3_orthogonal_representations` of it returns exactly the zero-tensor stores (up `(0, N_i)`, bonds `(1, 0, 0, 1)`),
carrying shape, stack, dtype and backend. Then `old_frame` is never `None` (it is the empty frame), the step's old
ranks are plain ints, `stack_shape` / `use_jax` / `N` disappear, and the frontend spells the zero case as a rank-0
`x_old`. Two caveats, both checked:
- `ut3_orthogonal_representations` of the rank-0 UT3 raises (`ValueError: all input arrays must have the same
  shape`), so build the uniform empty frame directly (zero-width A-block, boundary-bond masks);
- `compute_minimal_ranks` maps `(1,0,0,1)` to `(1,0,0,0)`, so the structural reduction of `x_old` must treat rank 0 as
  already minimal.

If Nick keeps `None`, then `shape=` and `stack_shape=` must also go on `T3Frame.padded` / `UT3Frame.padded`, and the
superseded J1 caveat must be stated: an array-free call uses the backend's default float, so the cone runs in float64
on float32 numpy data.

### 15. MINOR — §5.4: `PaddedStart(ValueHashedFields)` is the wrong mixin and the wrong shape

`ValueHashedFields` hashes every field by value and raises on a non-numpy array field (`common.py` `_value_key`: "aux
fields must be HOST data"). A result record holding a possibly-jax frame is not a jit aux. The gold standard returns
tuples (`t3svd -> (x, ss_tucker, ss_tt)`). **Fix:** the backend returns `(frame, gnorm, movements)`, one element per
line; the frontend twin may return a frozen `eq=False` record of frontend types (as `LeastSquaresSetup`).

### 16. MINOR — §1, §5.4: the keyword groups — count, a missing knob, `gradient_descent`, seeds

- The list has twelve keywords (`start_frame`, `padded_from`, five `padded_*`, five `cone_*`), not eleven.
- `alternate` (D-14) is settable on `padded_start` and the loops but not through `newton_cg`, against D-6 ("all
  user-settable").
- `gradient_descent(g0_reference='cone')` (§1) has no cone keywords.
- `padded_seed=None` / `cone_seed=None` must say `None = ZERO_FRAME_SEED`.

On a simpler spelling, three options:
- A frozen schedule record (`padded_schedule=`, `cone_schedule=`): four keywords in all, every knob settable, defaults
  in one place. It is the smallest, but Nick rejected exactly that (`PaddedFrameOptions`, superseded I-3: "plain
  keywords").
- One merged `align_*` group: eight keywords with `align_alternate`, but it couples the reference to the start's
  tuning, which D-7's "the value after the schedule" argues against.
- **Recommendation:** two plain groups, plus `padded_alternate` / `cone_alternate` (fourteen); give `gradient_descent`
  the cone group; record the count in D-6. Nick may still merge.

### 17. MINOR — §3.5, §5.6, T15: the frontend's validation and conversion list is missing

The superseded plan had it (`padded_structures`' checks, `_require_padded_pair`); the draft dropped it. Needed in
`optimizers.newton_cg(padded_from=)`, `optimizers.padded_start` and the twins:
- `x_old` of `x0`'s layer (uniform: the full `.data`), with the same shape, stack and padded `N`;
- `x_old`'s stores ≤ `x0`'s per edge (structural);
- safe mode: `x0.allclose(x_old)` (the zero padding), and tied factors under sharing;
- the reduction named: the optimizers' own `_ragged_minimal` / `uniform_minimal`, sharing-aware;
- `use_jit=True` converts `start_frame` / `padded_from` together with `x0` (`_prepare_jit_inputs`), else a pass mixes
  numpy and jax;
- `start_frame`, `padded_from` or `'cone'` on a corewise geometry is a structural error;
- `x0` ranks equal to `x_old` ranks: a documented no-op (or skip the loop).

### 18. MINOR — §5.6: `manifold.aligned_padded_frame` cannot reach the uniform layer without an import cycle

`uniform_manifold` imports `manifold` (`uniform_manifold.py:31`). **Fix**, keeping D-8's class dispatch on the hint:
either one twin per layer (`manifold.aligned_padded_frame`, `uniform_manifold.aligned_padded_frame`), each dispatching
on the hint's class, or a lazy import with a comment. Either way, state:
- a UT3 hint must have the frame's padded `N` (a structural error naming `from_t3(..., N=)`, worded as
  `UniformTuckerTensorTrain.__add__` does);
- a hint of the other layer is a structural error naming the conversion;
- the calling convention of a user `project`: frontend objects or raw data, and gappy block-layout frames on the
  uniform layer.

### 19. MINOR — §3.4, §5.3: the A-block widths and the level widths need integer definitions

- (a) "`Na = max(nU°, nD°)`, `Rb = max(rL°, rR°)` over edges and the stack" reads as a max of old COUNTS, but "the
  A-block is the old frame AS GIVEN (nothing moved)" needs the old frame's padded WIDTHS (`max(U°.shape[-2],
  O°.shape[-2])`, `max(L°.shape[-1], R°.shape[-1])`, as `uniform_start._old_block_layout` has it).
- (b) `out_widths` / "the level's padded widths": give the integer rule, the frame widths `ut3_orthogonal_representations`
  produces from the level point's padded `(n, r, N)`: `nU = min(n, N)`, `nD = min(nU, r²)`, `rL = rR = r` (superseded
  §3.4 table). Checked: `n = 6, r = 2 → (6, 4, 2, 2)`; `n = 5, r = 4 → (5, 5, 4, 4)`. Read it from `geom.var_masks`
  shapes inside `padded_start`; compute it from the target ranks' maxima in the frontend loop. Never by orthogonalizing
  a template (the pattern the paused arc was stopped for).

### 20. MINOR — §3.4, §4.1, §4.4: the group precondition of both steps

The uniform step writes every member's A-block from the group's first mode (`U[first_of_mode][..., :Na, :]`), and the
ragged step reads `Ā`, `g_old` from `U[i0]`, so an untied old frame is silently tied. **Fix:** a structural check that
A-block masks (ragged: stores) are equal within each group; tied values as the frontend's safe-mode precondition, as
`t3svd(sharing=)` does.

### 21. MINOR — §5.5, §5.6: `oversampled_ranks` on a varying-rank stack

`ranks.oversampled_ranks` refuses stacked arrays (`ranks.py:202`, "stacked rank arrays are not accepted (reduce per
element)"). `UniformTuckerTensorTrain.oversampled_ranks` and the uniform loop need per-element ranks (the prototype
loops `np.ndindex(*C)`). Say which: widen the unreleased function to `(d,)+C` int arrays, as `compute_minimal_ranks`
does (the superseded r8 MINOR 1 asked for this), or a host loop in the twin and the loop.

### 22. MINOR — §6: test-plan corrections

- (a) T2's float32 criterion contradicts the measurement: README "Numerics" has 3e-7 at `s = 0` (≈ 5u), and a
  two-sided "within 3×" is BLAS-fragile. Use a one-sided `≤ c·u·max(1, s)` with `c` from the measurement (e.g. 30).
- (b) T4's cycle regression needs the loop (N3, not N2). The dense reference lacks the transport pass: port
  `cycle/cycle_schemes.select_pass` beside `dense_oracle.py` for T4 / T6's two-colour passes. Name the npz's contents
  (`S`, `X_old`'s cores, the target ranks, `seed`; as in `g06c_cycle_case.npz`).
- (c) T8 "under jit bitwise equal to numpy" and T10's "bitwise" are numerical claims; D-24 allows bitwise only for
  structural facts. Use tolerances there, and keep bitwise for the structural facts (old block, junk zeros, tied
  factors).
- (d) T13: see finding 4.
- (e) T14's rejection tests must isolate each frame-mask family (up, down, left, right, with supercores byte-identical)
  and carry a mutation check (`testing_strategy.md`, "Exercising a mask check is not testing it"). D-10's check must
  compare all four frame-mask families, boundary entries included: `geom.var_masks` lacks `left[d]` and `right[0]`.
- (f) Exact output-mask assertions for `ufv_padded_frame` and `utv_align_padded_frame` (A-block masks as given,
  P-block prefix of the counts), derived independently (checklist item 2). Only T11 has them.
- (g) The garbage must be finite and written into a copy of the clean array's own layout (the `testing_strategy.md`
  precisions). Name where `uniform_testing.py`'s scatter recipe lands in `tests/`.
- (h) `d = 1`: in T1 / T9, or a structural error.
- (i) Name the test files of N4–N7.

### 23. MINOR — §3.1(4), §4.1, §8, §12: HISTORICAL lives in the tests, and "named contractions" is the wrong mechanism

`HISTORICAL` is in `tests/test_contractions_interpreter.py` (the one duty of `contractions_internals.md` §9), not in
`backend/contractions.py`. Call sites write `contract('...')` strings; the named-contraction era is over (§1 of that
doc). N2's file list and §4.1's sentence are wrong. N1 must also delete the S2/S3 strings (ten entries from 92640502;
keep S1's two) and refresh `tests/backend/test_padded_start.py`'s docstring, which promises geometry-member and
`padded_start` rows from the old plan. Fix the wording in §3.1(4), §4.1, §12 and the slice table.

### 24. MINOR — §6 T5, §8 N1: T5's reference is deleted in the same slice

N1 deletes the S2/S3 `fv_padded_frame`, and T5 compares the new one "= the shipped S2 construction (bitwise)".
**Fix:** before deleting, freeze S2's outputs on T5's cases into `tests/data/` (or keep S2's hint-free path as a
test-local reference). "Shipped" → "committed": S2 was never released. The `fv_zero_point_frame` row is unstacked
only (it raises `NotImplementedError` on a stack).

### 25. MINOR — §0, §2.3, §7: claims beyond their evidence, and docs gaps

- "Two-colour … converges monotonically (merit never decreased)" was measured on the 243-number case with `p = 0`
  (`cycle/README.md`, dense). It was not measured after the truncating pass, nor per pass on Darcy (sampled passes
  only). Say so.
- §0's "36.4 after four passes" is the all-edges (cycling) scheme; the default two-colour gives 35.9 (§2.3's own
  table).
- §7 should also cover:
  - `docs/api_reference.rst`, which enumerates the backend modules by family; add the new modules;
  - `docs/contributor/uniform_svd_prefix_orthogonalization.md`, which says "There is no QR in any orthogonalization
    path": add the one seeded completion and why it is not an orthogonalization;
  - the `'cone'` row in `fitting_and_optimization.md` §5's `g0_reference` table and in `numerical_contracts.md`;
  - D-24's "`gtol_rel=1e-4` with `'cone'`" in the user doc.

### 26. MINOR — §8: slices

- N2's T4 cycle half moves to N3 (finding 22b), and T10's loop half to N6.
- N7 needs `backend/geometry.py` under finding 1, and test files named.
- N8: list the frontend modules and test files.
- N1: add `tests/test_contractions_interpreter.py` and `tests/backend/test_padded_start.py`.

With those changes each slice is independently committable. The review stops (after N3, N6, N8, N10) are right.

---

## NIT

27. §3.2: `padded_frames.py` imports the private `_haar_rows` / `_identity_prefix` from `fv_operations`.
    `pinned_completion` with `RandomState([seed, min(g)])` seeds already is the Haar recipe (its docstring says so):
    call it and drop the private import, or make the helper public.
28. §3.3: "the backend requires `old_frame`'s stores to equal `stores(shape, old ranks)`" — `fv_padded_frame` has no
    old-rank arguments; the check is "the old frame's (down, right) widths equal those its (up, left) widths imply"
    (README finding 3).
29. §3.5: "(shared with the optimizer when old ranks are uniform)" is unclear; rewrite or delete.
30. §5.4: `PaddedStart.gnorm` duplicates iteration 0's `gnorm` when `newton_cg` uses it. Compute it for `'cone'` and
    the standalone `padded_start` only.
31. One docs sentence: `g0_reference='warm_start'` with a start frame is `‖g‖` at that frame.
32. One sentence: the alignment's projection omits the regularizer's gradient, which is harmless (`X_old` lies in
    every padded frame's tangent space, and the step uses the normal content).
33. `alternate=False` exists only for T4 and comparisons; the regression can loop the step with `update='ULRO'`
    directly. Consider dropping the mode.
34. T6's "the reference loop with the reference step" is undefined; name it, or keep only the dense loop.
35. T12's "same pass counts": choose cases whose movements are not near `tol` at the decisive pass.
36. §1 `seed=...`, §5.6 `seed=...`: write the default.
37. The plan has no "verified facts" table with script per claim, as the model plan has. §2.3's table is close; add
    the script name per row (`check_v2.py`, `check_uniform_*.py`, `cycle_schemes.py`, `g06*`), so a reviewer can rerun
    a claim.

---

## The open decisions

- **D-20 (host loops): agree.** `project` is an arbitrary callable (the problem's gradient through Python-level kind
  and geometry objects), and the stop branches on the movement. Whole-loop jit buys ~6 s of compile for a loop that
  runs once per level. The condition is finding 4: the uniform step must have a stated jit boundary or closure-free
  bodies, and a compile-count test.
- **D-21 (zero old point): the rank-0 point** (finding 14), with its two caveats handled structurally. Failing that,
  `None` plus `shape=` / `stack_shape=` on every function that can receive it, the start constructors included.
- **D-22 (naming): `fv_aligned_padded_frame`**, required by the prefix grammar (finding 10), not a matter of symmetry.
- **D-23 (modules): acceptable.** `padded_frames.py` / `uniform_padded_frames.py` mirror the gold standard's per-layer
  modules (`t3_svd` / `ut3_svd`), and Nick already chose the new ragged module (round 10). Only the private-helper
  import (NIT 27) needs care.

---

## Critic's notes for later rounds (check these first; then anything a change touches)

1. Seam: backend geometry members (or an explicit, doc-amended relaxation); `x_old` uniform type = full `.data`; the
   old frame from `x_old`'s own masks; `from_frame` plain masks = `(F[5][0], F[5][2])`; level widths from
   `geom.var_masks`.
2. Cone: `r0` at the zero tensor; the max_passes = 0 test equals `'zero'`.
3. No `NewtonInfo` field; `stats` keys only on runs that use them; G-bit = whole-dict equality; display decision.
4. Compile: closure-free module-level scan bodies or a value-hashed jit boundary; per-level program count;
   compile-count test; T13 rewritten.
5. Unverified items marked; T14 (a) cross-layer `probe_derivatives`, (b) shared, (c) exact Hessian from a start frame.
6. `(I − ĀĀᵀ)` and the zero guard in §3.4 / §4.4.
7. `pad_safe_pinned_completion` contract: per-element counts / offset, or return-all-columns; masked at entry;
   real-support argument.
8. Full house-style signatures in §5; names `frame_data` / `variations_data`, `completion_widths` / `level_widths`.
9. `sharing` labels on public backend entries.
10. `fv_aligned_padded_frame`.
11. Uniform step takes ranks.
12. Transport + truncate → `ValueError`.
13. Loop edge cases (a)–(e); tuple of movements in both layers.
14. D-21 resolution and its consequences in every start/loop/twin signature.
15. `PaddedStart` → tuple.
16. Keyword count, `*_alternate`, `gradient_descent`'s cone group, seed `None` meaning.
17. Frontend validation list.
18. Dispatcher placement (import cycle), N check, cross-layer hint error, `project` convention.
19. `Na`, `Rb` as old widths; level-width integer rule.
20. Group structural check in both steps.
21. Per-element `oversampled_ranks`.
22. T2, T4 / T6 transport reference, T8 / T10 bitwise, T14 isolation, mask assertions, finite garbage, `d = 1`, file
    names.
23. HISTORICAL location; N1 cleanup of S2/S3 strings.
24. T5 frozen reference.
25. Claims and docs list.
26. Slice table.

VERDICT: REVISE
