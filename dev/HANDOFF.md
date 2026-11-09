# T3Toolbox — current handoff

## ⚠ IN PROGRESS (2026-11-02): the padded frame / cone reference feature — read this block first

**What.** The structural, δ-free frame at a zero-padded rank increase (Tucker + TT completions from a T3 hint by a
pinned T3-SVD; seeded Haar / identity-prefix without a hint; `fv_zero_point_frame` is the s = 0 case), its subspace
iteration (`padded_start`), the optimizer seam (`Problem.local_model(frame=)`, `newton_cg(start_frame=,
padded_from=x_old, padded_passes=, padded_seed=, cone_passes=, cone_seed=)`, keyword-only, appended) and
`g0_reference='cone'`. **The plan is signed off by Nick (2026-11-02) and is the authority:**
`dev/padded_frame_cone_impl_plan_2026-11-02.md` (critic-approved after 5 review rounds + Nick's batch of changes;
reviews `dev/padded_frame_cone_plan_review_r1-5.md`; design notes `..._q_old_point.md`, `..._q_vectorize.md`).
Motivation and measurements: T3Polynomial `dev/zero_reference_cone_2026_11_02.md`.

**Binding rules (Nick).** Additive, NON-BREAKING release 2026.4.0: no default changes, `g0_reference` default stays
`'zero'`, `newton_cg` without the new keywords bit-identical incl. `stats`; the retraction and
`t3_orthogonal_representations` untouched; structural (ranks known), never a numerical rank decision, no ε content;
bitwise claims only for structural facts, tolerances + projector (gauge-invariant) checks otherwise; named
contractions via `backend/contractions.py`; no `ufv_padded_frame` twin (the uniform geometry member converts
explicitly); the optimizers unstacked, the constructor over `C`.

**Process (Nick's instruction).** Two Opus agents at MEDIUM effort: an **implementer** implements one slice of plan §8
and writes `dev/padded_frame_slices/S<n>_handoff.md`; a **reviewer** critiques it (`S<n>_review.md`,
`VERDICT: APPROVE`/`REVISE`); they iterate until the reviewer is satisfied; the orchestrating session then checks the
result (reads the diff, runs the full suite `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q -n auto`), commits
(stage by name; message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and the session line) and
starts the next slice. Review stops for Nick after S3, S7, S10, S12 (report, continue unless he objects). Unforeseen
issues: the session decides and records them in **`dev/padded_frame_cone_judgment_calls_2026_11.md`** (issue /
decision / why). Agents may be re-spawned fresh per slice with the plan as the brief if their context gets unwieldy.

**State (2026-11-02, late). IMPLEMENTATION PAUSED BY NICK at the S3 review stop — architectural style problem.**
S1–S3 are committed (b14f56eb, 2562264a, 92640502; the construction `fv_padded_frame` is algorithmically complete
and oracle-verified). S4 (the uniform geometry member `UniformManifoldGeometryOps.padded_frame`, its tests) is BUILT
but UNCOMMITTED and UNREVIEWED in the working tree (`backend/geometry.py`, `tests/backend/test_padded_frame.py`,
`tests/test_dispatch.py`, `dev/padded_frame_slices/S4_handoff.md`); do not commit it as is. Both agents are holding.

**Why paused (Nick's ruling, verbatim in spirit).** The backend implements PURE MATHEMATICAL FUNCTIONS: inputs and
outputs are immutable objects composed of basic types (arrays, tuples of arrays, integers, ...), and the code closely
follows the math. This was relaxed a little for the fitting/optimization layers only, and remains the goal even there.
`TuckerTensorTrain` / `UniformTuckerTensorTrain` and the backend functions they call are the GOLD STANDARD. The new
code violates this: `fv_padded_frame` takes a "template" FRAME only to read its shapes (an expensive orthogonalization
to obtain integers that `ranks.compute_orthogonal_representation_ranks(shape, tucker, tt)` already gives, statically);
`_frame_stores` returns a mutable dict of shapes (opaque, untyped) instead of enumerated immutable integers; the plan's
`padded_structures` orthogonalizes a zero tensor to get a template. Root cause: this "shapes-only template frame"
pattern entered with `fv_zero_point_frame` in commit 3e5bc922 (2026-10-26, exact-Hessian arc S0d); the plan copied it
as a precedent and the reviewers accepted it as such. Nick considers the problem broader than these two symptoms (a
misunderstanding of the programming style) and will go through the code himself.

**Decision (Nick, 2026-11-02): OPTION C.** Fix the construction's boundary BEFORE continuing the arc: (1) all four
agents first study the gold standard (an assortment of `TuckerTensorTrain` / `UniformTuckerTensorTrain` methods traced
to their backend functions; notes in `dev/padded_frame_slices/style_notes_*.md`) — these are the ONLY allowable
precedent; (2) the writer amends the plan, DELTA ONLY (round 9: §3.2/§3.4/§5.2/§5.4 + §12): `fv_padded_frame(shape,
tucker_ranks, tt_ranks, old_frame=None, hint=None, groups=(), seed, dtype, use_jax)`, stores via
`compute_orthogonal_representation_ranks`, no template frame, no dict helper, `padded_structures` returns integers;
the critic reviews the delta; Nick signs off; (3) one corrective slice S2′ refactors S2/S3's boundary and S4's
uncommitted member (S3's oracle tests carry over unchanged); (4) S4–S13 continue as planned. `fv_zero_point_frame` and
the codebase-wide style audit/refactor are a SEPARATE later arc (Nick's own read first; breaking release).

**Nick's round-10 decisions (2026-11-02):** padded-frame code in a NEW `backend/padded_frames.py` (all imports
downward); `compute_orthogonal_representation_ranks` stays in `ranks` (the `ranks` → `sharing` dependence is a code
smell for the drift-fix arc); no `dtype` argument anywhere in the feature; add the `with_ranks` test. Plan delta rounds
9–10 DONE and critic-approved (review_r7); Nick OK'd them with changes (typ.Union annotations; no dtype) and
raised a FURTHER issue → round 11: frontend/backend split and backend bloat. Most new backend functions need
FRONTEND TWINS (methods of the appropriate classes or standalone functions, as the gold standard does it);
detailed docstrings and doctests move from the backend to the frontend twins; most in-function argument validation
moves from the backend to the frontend (backend functions keep one-line docstrings + shape comments, as
`tucker_change_core_shapes` / `t3svd` do). Doctests that cannot be moved STAY in the backend (never delete a
backend doctest that has no frontend twin to move to). Round 11 in progress → critic → Nick's sign-off → then the corrective slice S2′ (new
`backend/padded_frames.py`; `fv_operations` back to efe5e3d; S4's uncommitted member refactored).

**2026-11-09: round 12 part 1 done; TWO decisions pending with Nick.** (1) Round 12 part 2: drop the three geometry
twins and defer `frame=` on the six fitting factories (critic r8 + session recommend both). (2) NEW, raised by Nick's
note "Completing frames by truncating A + εS, with projection-only access" (https://claude.ai/artifact/EvfyxaZKGuKUL5MdC6UfjD,
a hinted completion for tree tensor networks): the session checked it against the committed construction — it is the
SAME map (independent dense implementation: one pass agrees to 1e-13 on Tucker/left/right frames and on the down cores
under the note's rooted rule; same fixed points to 3e-15), but the plan computes it on the doubled-rank hint treated as
an opaque T3 (width-2R work), where the note's core form takes frame + variation cores and needs width-2k
factorizations and k x R messages. Measured cost of the committed hinted construction, laptop Darcy 11→16: 179 ms
(82 % einsum, 7 % SVD/QR) vs 196 ms gradient+frame and 11 ms retraction. Open design question for Nick: the hint's
TYPE (a T3, as planned, or a tangent = frame + variations, as the note) and whether to keep the committed
implementation this release. Record + scripts: `nicks_research_experiments/tensor/padded_frame_vs_tree_completion/`.
Nothing proceeds (no S2′) until Nick rules on both.

**2026-11-09, later: DIRECTION CHANGE (Nick).** The plan's hinted path will be REPLACED by the efficient core form of
the tree-completion note (frame + variation cores in, frame out; width-2k factorizations): "the R^2/k^2 cost is
significant, and it is worth doing it right now". Nick is ambivalent on rooted vs unrooted. His structural API
proposal: a loop `padded_frame(A, new_ranks, get_hint: Frame -> Variations) -> Frame` over a pure step
`construct_hinted_frame(F, H) -> Frame`, any hint tensor supported through its projection. The session's assessment
(awaiting Nick's reply): shape is right with changes — the step needs integers (old ranks, output ranks, groups) and
returns a residual; the start is the seeded hint-free completion (S2's function), not a zero pad; the loop has an
oversampled phase then a target-rank phase with a stopping residual; the frame is enough for the hint function and A
must not be passed (residual hoisted out of the loop); the callable-taking loop lives in the optimizer layer + frontend,
the backend keeps two pure functions; suggested names `fv_padded_frame` (start), `tv_align_padded_frame` (step),
`aligned_padded_frame` (loop), `project` (the callable); unrooted rule; the committed S3 construction becomes the test
ORACLE, not shipped code; prototype the T3 core form in the research repo first (sharing groups, unequal widths per
directed edge, stacks are not covered by the note). The earlier "hint is a T3" decision is superseded. Round 12's
pending twin question is moot until the new shape is settled. NOTHING is being implemented yet.

**2026-11-09, API SHAPE SETTLED WITH NICK (new hinted path).** Backend, new `backend/padded_frames.py`, pure functions:
`fv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame=None, groups=(), seed=...) -> frame` (the seeded hint-free
start; S2's function, integer structure); `tv_align_padded_frame(frame, variations, old_tucker_ranks, old_tt_ranks,
tucker_ranks=None, tt_ranks=None, groups=()) -> (frame, residual)` (ONE step, the note's core form, UNROOTED rule,
frame + variations in); `aligned_padded_frame(shape, tucker_ranks, tt_ranks, hint, project, old_frame=None, groups=(),
oversampling=5, oversampled_passes=..., max_passes=..., tol=..., seed=...) -> (frame, residuals)` (the fixed-point loop:
oversampled phase, then target ranks until the step's residual is small; lives in the FRAME module, not the optimizer
layer, because aligning a frame to a hint is useful outside optimization). The projection is a plain function plus its
data, never a closure: the loop calls `project(frame, hint)`; `hint` is opaque data handed to `project` unread. In the
BACKEND `project` is REQUIRED — no dispatch on the hint's shape (a tuple of arrays cannot be known to be a T3;
variations have the same nesting): raw users pass `tv_project_t3_onto_tangent_space` / `tv_project_dense_onto_tangent_space`
/ their own. The FRONTEND twin dispatches on the hint's class (`TuckerTensorTrain`, ndarray; anything else needs
`project`). The optimizer layer passes its own module-level projection and data (residual computed once). Backend takes
the OLD FRAME (frontend takes the tensor); the committed S3 construction becomes the test ORACLE. NEXT: prototype the
T3 step in the research repo against the oracle (open: sharing groups, unequal widths per directed edge at non-minimal
ranks, stacks), THEN a writer/critic plan for the hinted path. Nothing implemented in the library yet.

**2026-11-09: CORE-FORM PROTOTYPE DONE** (`nicks_research_experiments/tensor/aligned_padded_frame/`, README has
every number). `tv_align_padded_frame` (one step, the note's section 10, unrooted) and `aligned_padded_frame` (the
loop) for T3s, numpy/jax, stack-aware. Verified: one step equals the dense per-edge update to 3e-14 on all four stores
(d = 2..5, truncating steps, tangent part 10x); non-minimal ranks (unequal widths per directed edge), sharing groups
(one QR/SVD per group, block J), zero old point, stacks, ungauged variations all work WITHOUT changing the algorithm;
the loop equals the dense loop to 1e-14 at every pass; the free residual tracks frame movement (ratio 0.3-1.2);
jit-compiles. Findings for the plan: (1) the OLD BLOCK MUST RIDE IN THE QR (`[A | g_old | g_new]`, width r + 2k) --
the note's cheaper projected form returns non-orthonormal frames on zero / rank-one hints; (2) the step can take old
RANKS (a frame's own up/left widths reproduce its stores); (3) variations need not be gauged; (4) schedule on the note's
test family: p = 5 for 2 passes then target-rank passes gives 97 % of the converged merit in 4 passes, 99.7 % in 6
(p = 0: 76 %, 83 %); oversampling alone plateaus at 88 %; residual 0.05 ~ 99.5 %; (5) cost, numpy d = 5 k = 5: N=23
11->16 13.5 ms vs 168 ms committed; N=64 41->46 72 ms vs 22.3 s committed (retraction 636 ms). NOT prototyped: the
uniform (masked) layer, tied projections via `shared_data`, GPU, a Darcy run. NEXT (awaiting Nick): writer/critic plan
for the new hinted path (start / step / loop, `project` + `hint`), reusing S1 `pinned_completion` and S2's hint-free
start; the committed S3 hinted path becomes the test oracle only.

**2026-11-09, Nick on converting the prototype to library code:** (1) scans, not for loops; (2) named contractions,
not raw einsums; (3) shape annotations on every signature; (4) make the step POLYMORPHIC between the ragged and uniform
layers where possible (array(ragged_cores) = uniform supercore; ranks and shapes are static data); (5) his open
concern: the QR-based orthogonalization does not put the top singular vectors first. Session's finding on (5): in the
prototype the ordering is the small SVD's, not the QR's (truncating steps match the dense update); but a raw QR IS
unsafe on zero-padded data. Tested replacement (research repo `check_pad_safe.py`): ONE call of the library's
`pad_safe_svd` on `[A | Y/(4||Y||_F)]`, Y = the edge's normal content off the pinned block -- triplets come out ordered
(pinned block, sigma = 1; content by singular value, sigma <= 1/4; real-supported completions), mutually orthogonal,
pads bitwise zero; same spans as the prototype on every edge, also on interior-padded copies; valid for zero and
rank-one hints. Caveats raised for polymorphism: the old/new block boundary differs per mode, so P-parts must be
fixed-width WINDOWS at a per-mode static offset (full-width masks would lose the k-scaling); sharing groups make the
up-edge update per group. The scan/polymorphic step itself is NOT prototyped yet.

**2026-11-09, Nick's correction on the uniform layout.** The uniform layer does NOT enforce prefix form: real vs junk
slots are defined only by the host edge masks (gappy after + and x, data never compacted); junk-last is what `ut3svd`
and the SVD-based frame construction PRODUCE. Rule: a function may switch to its own junk-last layout internally and
return it, but assumptions about junk locations must stay inside functions we control. Consequences for the uniform
path of the alignment step (session's revised view, sent to Nick): (1) read inputs only through masks -- gather once at
entry (as `ut3_to_t3` does), positions are ours only after that; return junk-last with masks built from the integer
stores; (2) `pad_safe_svd` is the right factorization (pads at arbitrary positions in, junk-last triplets out); (3) by
`docs/contributor/uniform_polymorphism.md` an op that returns a frame is uniform-specific -- the shared part is the
message sweeps (polymorphic scans, mask once then mask-free), the gather / factorization / output masks are a uniform
twin; (4) mask-once keeps full padded widths (cost scales with the full rank); keeping the k-scaling needs the entry
gather to produce compact completion blocks; (5) open: how the pinned old block is named in the uniform layer (integer
old ranks + "the first r real slots in mask order", or explicit pinned masks).

**2026-11-09, uniform layer: Nick's ruling and the uniform-first prototype.** Ruling: polymorphism is not reasonable
here; a uniform version is wanted ONLY if it is truly uniform (no re-layout through ragged-like operations, no unrolled
loop over modes); otherwise the user converts UT3 -> T3, runs the ragged operation, converts back. His suggestion:
pad by ONE uniform old width and ONE uniform completion width (different r and k per edge under masks). PROTOTYPED
(`nicks_research_experiments/tensor/aligned_padded_frame/uniform_form.py`, README section "A uniform-first step"):
BLOCK LAYOUT -- every rank slot = [old block | completion block], each padded to one uniform width with its own mask;
the old block is the old frame's supercore as given (gappy allowed, nothing moved), the completion block is appended
(as under `+`); the only positional fact is the split point, one static integer for all modes. Sweeps = `lax.scan`
with a shape-uniform body; up/down edges = one batched contraction + one batched factorization over the mode axis;
factorization = `pad_safe_svd` on [old block | content/(4|content|)] with masks as data, completion triplets selected by
a static one-hot at the pin count. Verified against the ragged step on real parts with gappy masks in both blocks and
garbage in all junk: <= 1e-14 (also truncating, uneven increments, non-minimal ranks; valid on zero / rank-one hints;
old block bitwise untouched); same under `jax.jit` + real scans. First jit call: uniform 1.6-1.7 s at d = 6, 12, 24;
ragged 1.9, 3.1, 6.5 s. CPU run time at N=64, 41->46: uniform 169 ms vs ragged 47.5 ms (jax); no GPU number. NOT
prototyped for uniform: the seeded start in block layout, stack axes, sharing groups, and how the uniform geometry
consumes a block-layout frame (its masks are gappy by construction; `ufv_make_frame_masks` builds prefix masks).

**Earlier next-steps note (superseded by the decision above).** (1) The session studies the gold standard: trace ~5 methods each of
`TuckerTensorTrain` and `UniformTuckerTensorTrain` to their backend functions and characterize the style (signatures of
basic immutable types, math-following bodies, shape comments, how structure/ranks are passed as integers). (2) Then a
REPAIR PLAN with Nick: redesign the structure input of `fv_padded_frame` (integers, e.g. `(shape, tucker_ranks,
tt_ranks)` or a frozen validated `FrameStructure` of ints; dtype/backend explicit or inferred from array inputs), delete
`_frame_stores`, make `padded_structures` return integer structures, audit S2–S4 (and the plan's §3.2/§3.4/§5.2/§5.4)
against the style; decide what to do about `fv_zero_point_frame`'s same defect (public signature; non-breaking release).
Judgment calls: `dev/padded_frame_cone_judgment_calls_2026_11.md` (J-1..J-6; add J-7 for this). The plan file stays the
authority for the mathematics; its code-shape sections are under repair.

**How to resume after compaction.** Read this block, the plan's §8 slice table and the slice's §4/§5/§6 entries, the
latest `dev/padded_frame_slices/S*_handoff.md` / `S*_review.md`, the judgment-calls file, and `git log --oneline -15`;
then continue the loop at the slice named in **State**.


_Updated 2026-10-29, at the 2026.3.0 release._ The previous handoff (the exact-Hessian thread through S13, the
2026-08-22 whole-library review thread, the 2026.2 recon backlog) is archived verbatim at
`dev/archive/handoff_2026-10-29_pre-2026.3.0.md`; older ones are `dev/archive/handoff_2026-08-21_optimization_layer.md`,
`handoff_2026-08-20_pre-2026.1.0.md` and `handoff_2026-07-12_1.0_complete.md`. What each release contains is the
CHANGELOG; why it is the way it is, `docs/`. This file is where-we-are + what's next, nothing else.

## Where we are

**2026.3.0 SHIPPED to PyPI (2026-10-29)** — tag `v2026.3.0` at `ad578f6` (rc1 dry-run on TestPyPI first), published
via trusted publishing, verified from clean venvs (numpy-only: import, GN + exact fits, the jax-absent warn-and-run;
`[jax]`: the quickstart 81/81 against the installed package). The GitHub Release object is created by Nick (the session
cannot create releases). 2026.2.0 shipped 2026-08-25, 2026.1.0 2026-08-20, 2026.0.0 2026-07-13. Release prep is done
(2026-10-29): the three version sites bumped, the CHANGELOG stamped `[2026.3.0] — 2026-10-29`, the release notes'
"Upgrading from 2026.2.0" complete, the wheel smoke test passed (`twine check` PASSED; the wheel holds the 53 package
modules and nothing else; a numpy-only py3.11 venv runs the quickstart, an exact-Hessian fit to 1.5e-12 and the
jax-absent `use_jit` warning; a `[jax]` venv runs `getting_started.rst` 81/81 against the installed wheel), and the
release gates are green on the final tree: the full suite (1,170 tests / 46,364 subtests) and the serial slow tier (66 /
13,173) on py3.13, 225 module doctests, `getting_started.rst` and every doc page on all three environments below, and
`sphinx -W` (renders 2026.3.0); the floor's full suite was last run at S13 (since then the library changed only in
docstrings and annotations). What remains: the `v2026.3.0-rc1` tag → TestPyPI install check → the `v2026.3.0` tag →
approve the `pypi` environment → a clean-venv check against real PyPI → the GitHub Release from the CHANGELOG section.

**The exact-Hessian arc (2026-10-25 → 29) — DONE, shipped in 2026.3.0.** `newton_cg(..., hessian='exact')`: the
Gauss-Newton term plus the Weingarten curvature under a Steihaug–Toint trust region, for every sampling kind on every
geometry × representation, with the zero-tensor Newton reference, the operand-scaled gauge check, clean uniform
padding and recurrence jet kernels. Slices S0c–S13, the close-out review S13b, the math note (S15a, final pass S15b with
every T4S citation verified against arXiv v2), each with an implementer and an independent-reviewer record; S14 updated
T3Polynomial downstream (in its own repo). Rulings on 2026-10-29: the reorganization R **cancelled**; **I-39** (the S3
memory gate: the two-hole's TT-assembly peak 1.25–1.61× the Gauss-Newton one) accepted; **I-40** (the operand-scaled
gauge check's finite-difference limit) accepted as documented. The durable record is `docs/fitting_and_optimization.md`
§4.12, `docs/contributor/curvature_internals.md`, `docs/contributor/verification_records.md`, the ledger
`docs/contributor/deferred_and_rejected.md` and the math note `docs/exact_hessian_two_hole.tex` (+pdf). The build record
is archived: the plan `dev/archive/riemannian_newton_curvature_impl_plan_2026-10-29_shipped.md` (its design plan
`..._plan_2026-10-27_implemented.md`), the slice records and the deferred table in
`dev/archive/curvature_impl_review_2026-10-29/slices/` (`DEFERRED.md`), the experiments in `.../shared_exp/`, the S0b
downstream report in `.../s0b/`, and the cancelled R scope `dev/archive/reorg_scope_2026-10-29_cancelled.md`.

## Release gates (run all before a tag)

- **Full suite:** `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q -n auto` (per-test seeded, order-independent).
- **Slow tier — REQUIRED, run SERIALLY:** `T3TOOLBOX_SLOW_TESTS=1 PYTHONPATH=$PWD <env-python> -m pytest
  tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py -q -p no:cacheprovider` (the full optimizer matrix,
  the exact sweep's seeds × layouts × order 3, the full-length uniform exact runs).
- **Doctests + docs:** `pytest --doctest-modules t3toolbox`; `docs/getting_started.rst` and every `docs/*.md` +
  `docs/contributor/*.md` page (except `doctest_style.md`); `cd docs && sphinx-build -W -q -b html . <out>`.
- **Three local environments:** (1) the system py3.13 / numpy 2.5 / jax 0.12 — runs AHEAD of anything CI sees;
  (2) the **CI current-leg twin py3.11 / numpy 2.4 / jax 0.10.2** (what the leg's `numpy>=2` + `jax` resolve to on
  py3.11; BLAS-dependent stop reasons and step counts have differed from (1)); (3) the **floor py3.11 / numpy 1.26 /
  jax 0.4.30** — the full suite, plus `docs/chunking.md` by hand (the CI floor leg skips the doc doctests when its test
  step fails, and the measured chunk costs depend on the jax release). Venvs do not survive the container: `uv venv -p
  3.11 <dir>` then `uv pip install "numpy<2" jax==0.4.30 jaxlib==0.4.30 pytest pytest-subtests pytest-xdist` (floor) or
  `"numpy>=2,<2.5" jax==0.10.2 ...` (twin).
- **Wheel smoke** (`dev/archive/release_plan_2026-07-13.md` REL-2): build, `twine check`, the wheel's file list, a
  numpy-only venv, a `[jax]` venv running `getting_started.rst` against the installed package. A release is: bump
  `pyproject.toml` / `CITATION.cff` / the `__init__` fallback, stamp the CHANGELOG, smoke, gates, rc tag → TestPyPI →
  tag → approve `pypi` → clean-venv check → GitHub Release (REL-1/REL-3 are one-time work, done).

## Next up — the post-2026.3.0 backlog

**From the exact-Hessian arc** (each also in the ledger `docs/contributor/deferred_and_rejected.md`):
- **float32 end to end.** A float32 fit comes back float64: 38 default-dtype constructors in eight backend modules sit
  upstream of the jet kernels (which preserve float32 since 2026.3.0). A ready follow-up patch is
  `dev/archive/curvature_impl_review_2026-10-29/slices/float32_end_to_end_followup.patch` (checked 2026-10-29 over the
  six kinds × both layouts × manifold / corewise / shared × both Hessians; float64 bit for bit). Its own slice: each scan
  init must take the dtype of ALL its scan's operands, which wants a mixed-dtype jax test per site.
- **Legacy RNG** (backlog from the cancelled R): move the random constructors off the global `np.random` stream to
  explicit `Generator` / jax-key arguments (the E2 and O2 flakes were this).
- **I-36 — the Newton-schedule A/B** (Gauss-Newton vs exact as the continuation default; Nick leans exact with the warm
  start). Empirical, after implementation; the per-call override hooks exist (and T3Polynomial's per-level
  `newton_kwargs` schedule). Inputs so far: `dev/archive/curvature_impl_review_2026-10-29/shared_exp/shared_vs_unshared_exact.md`
  and `.../nonlinearity_vs_tied_factors.md` (numbers in `verification_records.md`); missing are GPU timings and the
  Hessian-apply cost against the per-iteration retraction and frame build.
- **The S9 tied-projection clip.** The uniform tied *projection* (`ufv_share_tucker_variations`, Gauss-Newton code)
  clips at the padded dims and does not mask padded `s_g` levels (17 % off ragged at a near-singular point the
  equivalence contract exempts; not robust to garbage in a hand-built companion). Needs its own bit-identity run.
- **The leftover-chunk overshoot.** On current XLA a remainder chunk is allocated on top of the full one, so
  `max_chunk_size_within` can overshoot its budget ~2×; `_wchunked_sum` doubles the peak at a near-full remainder.
- **The S7b follow-ups:** carry the operand scale through `normalized` / `sum_tangents` / `unstack_tangents`; a
  `UT3Tangent.__truediv__`; a faster `corewise_stack_norm` (safe-mode `±` is 4–6× slower since S7b; the optimizers run
  on `.data` and are unaffected).
- Smaller: `fv_share_tucker_variations`'s mixed-backend inference; optional `_blm.hvp` routing for the frontend exact
  `hessian(p)`; the `docs/conf.py` xref variant for the dead `:py:func:` links; the six
  `u_*_corewise_transpose_vs_ragged` oracle-harness checks (excluded since Phase D, repairable); `docs/pad_safe_svd.tex`
  still cites the pre-archive `dev/review_2026-08-22/` path (fix + PDF rebuild). Downstream, not ours: T3Polynomial's
  `x01` script has a pre-existing `NameError` (`K`), noted at S14.

**Carried forward:**
- The Goal-1 **`fit(...)` facade** — auto geometry / optimizer / ranks / `x0` + rank continuation and validation
  ("standard user, no fiddling"); the largest single item, deferred. The Newton-schedule hooks are its seam.
- The **Grasedyck–Kramer `SingularValueRegularizer`** and weighted `+` / `−` / scale / `⊙`; **base-point-as-tangent**
  as a public op (promote the regularizer's private `_manifold_point_tangent`); **sharded fitting**
  (`dev/sharded_fitting_plan.md`); **data-shaped residual weighting** (deferred 2026-09-03 — pick up from
  `dev/data_shaped_residual_weight_reviews.md`, not the plan's v1).
- The 2026.2 recon backlog, still open at last look (re-verify before picking one up; the archived handoff has the
  detail): frontend access to `edge_condition_numbers` / `t3_tie_tucker_factors`; uniform `share`; `UT3Weights` /
  `UT3FrameWeights` `reverse` / `stack`; `make_newton_display`'s backend uniform validation packing; the frontend-only
  `chunk_size='auto'` derivation; the `shared_geometry.py` signature pass; mechanical signature items (17 `regularizer:
  typ.Any`, the `(order+1)+W` tuple comma, `T3Weights`' `len=d+1`); `t3_stack`'s docstring; `core_shapes` vs
  `get_core_shapes`; the R2-12 / R2-13 review leftovers; a default-path doctest pass; trimming `test_dispatch` jit time.
- The **toolbox reference paper** (independent of releases): `dev/paper_scope.md`. The standalone grouped-contractions
  library: PARKED. Zenodo DOI (Nick, later); the dead `gh-pages` branch (optional). Sweep candidates: the DONE plans
  `dev/newton_display_plan.md` and `dev/per_mode_weighting_plan.md`.

## Standing open questions

- **`dev/OPEN_QUESTION_extension_surface.md`** — what extension the fitting layer is meant to support (opened
  2026-08-21; 2026.3.0 added the sixth `SamplingKind` operation and the four exact-mode geometry members to its
  surface). Standing, not a thread: it stays in `dev/` until answered.

## Standing constraints

Project-wide conventions live in **CLAUDE.md**; contributor conventions and decision records in the rendered
**Contributor guide** (`docs/contributor/`). Operational ones worth repeating: the docs build stays at **zero
warnings** (`sphinx -W` in CI); doctest outputs are **run-and-pasted, never hand-written**, and CI runs every
`docs/*.md` / `docs/contributor/*.md` page; a markdown doctest needs a **blank line before the closing fence**; a doctest
must not pin a stop reason or step count at the resolution floor (BLAS-dependent — assert a fact with a wide margin);
subTest params must be primitives under xdist.
