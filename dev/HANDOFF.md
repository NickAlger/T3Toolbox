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

**State (2026-11-10, evening). IMPLEMENTATION IN PROGRESS: N1-N3 COMMITTED (3f9679ef, 196fcfa2, 0c4d982e, +cccc6f7c the
loop's dispatch row); N4 BUILDING; plan at round 7 (97267ffd).** Nick reviewed the N3 stop (all items answered) and
approved continuing. HOW THE WORK RUNS (Nick's process): one IMPLEMENTER agent and one REVIEWER agent (Opus, extra-high
effort, persistent across slices -- continue them, do not respawn; if they are gone after a compaction, spawn fresh ones
with the same briefs: read a handful of `TuckerTensorTrain`/`UniformTuckerTensorTrain` methods and their backend first,
then `CLAUDE.md`, the signature/naming/testing contributor docs, then the plan); per slice: implementer builds and writes
`dev/aligned_padded_frame_slices/N<k>_handoff.md` -> reviewer writes `N<k>_review.md` with a verdict (style section first)
-> implementer fixes -> the session inspects the diff, runs the touched tests (full suite per slice by the implementer),
COMMITS BY NAME (never `-a`; ignore the pre-existing untracked `dev/check_t3svd_retraction_second_order.py`,
`release_notes_2026.3.0.md`; never commit `*.npz`) with `Co-Authored-By: Claude Opus 5.5` + `Claude Fable 5.1` +
the `Claude-Session:` line, pushes, starts the next slice. Review stops with Nick after N6, N8, N10 (N3 done).
QUEUE: N4 DONE (39d80022; `linalg.pad_safe_pinned_completion`, `ufv_masking.ufv_frame_widths`,
`backend/uniform_padded_frames.py` with `ufv_empty_frame` + `ufv_padded_frame` in BLOCK LAYOUT, scatter test helper
`tests/backend/_uniform_padded_frame_testing.py`, T7, T8, T16 rows) -> N3b DONE (5defef15: `use_jit` on `fv_aligned_padded_frame`, numpy inputs converted as the optimizers do, the per-pair
stop rule, NaN never stops the loop) -> N5 DONE (HEAD after N3b: `utv_align_padded_frame`; no `out_widths` -- output widths are the counts' maxima; the uniform module shares seven private math helpers with `padded_frames`, accepted, plan §3.2 to say so) -> N6 DONE (HEAD after N5: `ufv_compact_frame`, `ufv_aligned_padded_frame`; whole-loop jit under `use_jit`,
first call ~9 s at d = 6 and 24; compaction pads first when the level is wider; stacked tolerance 1e-13; `functools.partial`
binding inside the one jit accepted; the NaN-padded record cannot tell a NaN movement from an unrun pass -- N9 docs) ->
SECOND REVIEW STOP DONE (Nick 2026-11-10: items 1-4 OK; the `functools.partial` binding is accepted IN THIS INSTANCE
ONLY and establishes NO precedent -- the kernel docstring and the N9 scan-body note must say so) -> N7 DONE (commit after 9573a4e5; T14(f) was NOT a bug: `share()` gives orthonormal Tucker factors, at that
singular-value tie the two layers' orthogonal representations pick different bases, and the hint-free start reads
coordinate seeds in those bases, so the layers' STARTS differ in span (0.27-1.00 on t3svd/shared points; 1e-15 on generic
points); the aligned loop washes it out (final frames agree to 2e-5..5e-3 at the default tol, 3e-14..5e-10 at tol 1e-10);
DECIDED (Nick 2026-11-10): document in N9, NO code change -- the user doc states that the hint-free start is a function
of the old frame's representation (its bases), so representations of one point in different bases (singular-value ties,
e.g. t3svd output, shared points) give different completions, all valid starts (D-12), and the aligned loop's output is
representation-independent up to the stop tolerance (the numbers above); canonical bases at ties = a later audit item). CI: docs build and py3.11 green after 2d686e2d; the py3.9 floor job failed because
ubuntu-latest moved to 26.04 (no Python 3.9 build) -- pinned to ubuntu-24.04 in 9573a4e5. RULE: the implementer runs the
`-W` docs build (or the AutoAPI placeholder replay) and the floor env before reporting; reviews check for dead/shadowed
imports and duplicate definitions; 1e-14 cross-path tolerances are suspect; check `gh run list` after each push. MACHINE RULE: 13 GB laptop -- pytest-xdist at `-n 3` at most (`-n 8` swapped:
jax workers reach 3-4 GB each), for the implementer, the reviewer and the session. -> N7 (seam: geometry members `with_frame`, `aligned_padded_frame`, `require_frame_structure`;
`Problem.local_model(frame=)`; `_misfit_gradient_at`; public `padded_start`; 14 `newton_cg` keywords + `use_jit`
forwarded; `'cone'`) -> N8 DONE (commit after 3054a0a6; review APPROVE, 2 NITs: a decorator for the twelve `addCleanup` lines,
the wrong-N message for stacked uniform hints; T15 in ONE file `tests/test_aligned_padded_frame_frontend.py`; a
structurally-zero `padded_from` passes through uncanonicalized) -> THIRD REVIEW STOP DONE (Nick 2026-11-10: 1. eager-jax risk = DOCUMENTATION ONLY, no warning; 2-4 OK). NICK'S
N9 REQUIREMENT -- the frontend aligned-loop docstrings (both layers) get real content: the PURPOSE (bias a frame's
completion toward a hint); a BRIEF method summary (a fixed-point iteration on projections: each pass fills the missing
basis at every store from the hint's projection given the other stores' current completions; two-colour UL/RO ordering
after truncation); WHAT the iteration solves (the Galerkin-type / merit-function statement: the fixed point is a
stationary point of the captured hint energy ||Pi_T(F) g||^2 over the completions -- state precisely what holds and
demonstrate it); PROPERTIES (many local maxima; no global guarantee; representation dependence of the hint-free start,
loop output representation-independent to tol); an OVERVIEW of the parameters (oversampling + oversampled passes,
max_passes, tol, alternate = the two-colour switch, use_jit, max_passes=0); a LINK to the user doc in docs/ for details.
DOCTESTS that illustrate the properties: (a) the fixed-point / Galerkin property holds at convergence (e.g. one more
pass moves nothing; the per-store completion equals the leading directions of its content); (b) oversampling on vs off
at the SAME small pass count -- the oversampled run captures more / has less error. Doctests must be platform-stable
(structural facts or inequalities, not printed floats). FINDING FOR N9's
"jit or eager" section: each eager-jax alignment at a new structure compiles hundreds of executables and adds
8,000-9,500 process memory mappings; an eager-jax rank continuation aligning at every level could pass vm.max_map_count
(65,530) after ~7 levels and abort (`LLVM ERROR: Unable to allocate section memory!`, the failure
scan_body_principles.md records); `use_jit=True` compiles 2-4 programs per level (ragged) or 1 (uniform) -- the docs
recommend `use_jit=True` for jax inputs; optionally the loops warn once when called eagerly on jax inputs (ask Nick). -> N9 (docs incl. the "jit or eager" section; "STORE" is already defined in `docs/frame_variations.md` § "Four rank stores" (Nick
2026-11-10, fine as is) -- the feature's user doc and the `padded_frames.py` module docstring define it in one clause on first
use and link to that section, nothing more; CHANGELOG; the
math note from Nick's note; judgment calls J-7.. in `dev/padded_frame_cone_judgment_calls_2026_11.md`) -> N10
(acceptance; T3Polynomial adopts `padded_from`; review stop). DECISIONS SINCE THE PLAN'S APPROVAL (all in the plan's
§9): D-38 zero passes = the hint-free start (other schedule options then raise); D-39 jit under user control (`use_jit`);
dispatch rows required for every public backend function, host loops included; small test data as Python modules of
literals (`*.npz` are platform/version dependent), no hard rule; the skip rule compares STORES. FACTS WORTH KEEPING: the
eager first call of the step on jax arrays is op-by-op compilation (932 compiles, ~22 s at d = 4; 0.16 s after; 2 s
jitted); higher-rank Darcy (T3Polynomial `scripts/g06*`, data in `data/g0_reference/`): generic completions capture a
steady ~9-10 % of the gradient at every level, the aligned fraction falls with rank (50 % -> 23 % from 1->6 to 16->21)
but stays 2.3-4.8x the generic one; at 11->16 the aligned start with >= 10 passes cuts Newton by ~40 % and CG by ~2.4x
vs both generic starts (4 passes not enough there), where at levels 1-2 it had not. Research prototypes:
`nicks_research_experiments/tensor/aligned_padded_frame/` (README).

**QUEUED AFTER THIS ARC (Nick 2026-11-10): a unified glossary page for the library** -- one `docs/glossary.md` defining
the terms the docs use (store, frame, frame-variations format, carrier, padded frame, A-block/P-block, stack shape, sharing,
uniform masks / junk, ragged vs uniform layer, retraction, cone reference, ...), each entry one or two lines with a link to
the doc that explains it; linked from the docs index. Not part of the N1-N10 release; start it when N10 is accepted.

**Earlier (2026-11-10, later). SLICES N1-N3 COMMITTED (3f9679ef, 196fcfa2, N3 = HEAD); FIRST REVIEW STOP -- awaiting Nick.**
The ragged layer is complete: `backend/padded_frames.py` has `fv_empty_frame`, `fv_padded_frame`, `tv_align_padded_frame`,
`fv_aligned_padded_frame`; tests T1-T6, T16 rows, T18, T19; the dense per-edge reference `tests/backend/_padded_frame_oracles.py`;
the cycle case as text `tests/backend/_padded_frame_cycle_case.py`. Handoffs and reviews per slice in
`dev/aligned_padded_frame_slices/`. Items for Nick at this stop (from the N3 review): the plan's §2.3 should say "stores"
(the oversampled phase is skipped when working and target STORES match; 40,000 random structures never disagreed
with the ranks rule) and T4 / Appendix A still name an npz; small test data as a Python module of literals -- make it
the convention?; should the dispatch-test requirement exclude host loops (the loop has no T16 row; its building blocks do)?;
the eager first call of the step on jax arrays costs ~38 s (0.024 s numpy; 0.4 s after jit) -- decide before N8 how the
twins/docs handle it. Next: N4 (uniform primitives and start) once Nick has looked.

**Earlier (2026-11-10). PLAN APPROVED BY NICK ("The plan is good, and you can proceed"); IMPLEMENTATION STARTED.** Slice N1
(cleanup + the ragged start in `backend/padded_frames.py`) is with an implementer agent (Opus, extra-high effort); a
reviewer agent follows; the session checks diff + full suite, commits, starts N2. Handoffs and reviews per slice go to
`dev/aligned_padded_frame_slices/`. In parallel (research only, T3Polynomial `scripts/g06*`): the Darcy ladder
continued to higher ranks (11->16, 16->21) to see whether a generic completion captures less of the gradient near
convergence (Nick's suspicion); results feed the plan's §2.3 / §11. Nick's framing for the feature's worth: it need not
win big — independent value outside optimization, more principled, possibly more robust at higher ranks / worse
conditioning.

**Earlier (2026-11-09, end). THE PLAN WAS CRITIC-APPROVED (r6, after five writer rounds) AND NICK CONFIRMED THE FLAGGED REALIZATIONS.** Round 4-5 folded in the Newton measurement (laptop: any injected generic frame roughly halves level-2 Newton counts vs the library's padded frame; alignment passes add no further Newton savings within noise; starts decide basins) with §0 rewritten to what was measured, and `max_passes=0` = the hint-free start (D-38; other schedule options then raise).

**Earlier (2026-11-09, late). THE FRESH PLAN WAS CRITIC-APPROVED (r4, after three writer rounds).** Three realizations of his rulings are flagged at the top of the plan for confirmation: one frontend loop twin per layer (import cycle; D-35), fourteen `newton_cg` keywords (D-33), no frontend `padded_start` twin (the backend one is public). Then implementation N1–N10. Still pending input: the Newton measurement.

**Earlier (2026-11-09). THE FRESH PLAN IS DRAFTED; writer/critic rounds next.** The plan for the aligned padded frame is
`dev/aligned_padded_frame_impl_plan_2026-11-09.md` (draft 0 by the session; the old plan, its reviews r1-r8, its two
design notes and the S1-S4 slice handoffs are archived under `dev/archive/` with `_superseded` / `_2026-11` suffixes).
Process from here (Nick): a writer/critic pair of Opus 5.5 agents at EXTRA HIGH effort iterates on the draft until
they agree; then Nick and the session review; then implementation with an implementer/reviewer pair per slice (N1-N10
in the plan). Every agent first reads a handful of `TuckerTensorTrain` methods and the backend functions they trace
to (the only allowable precedent). Shipped on main: S1 (`pinned_completion`, `pinned_leading_columns`,
`oversampled_ranks`). To be DELETED by slice N1: the S2/S3 `fv_padded_frame` in `fv_operations.py` (restore efe5e3d),
its tests and dispatch rows; the uncommitted S4 working tree (`geometry.py`, test files) is to be DISCARDED
(`git checkout --`). Pending input: the Newton measurement (agent, `T3Polynomial/scripts/g06_aligned_frame.py
PART=newton`) feeds §2.3 / §11 of the plan. Mathematics and verification: Nick's note + the research prototypes
(`nicks_research_experiments/tensor/aligned_padded_frame/`, README). The decisions ledger is the plan's §9; the
history of how they were reached is below (older first at the bottom).

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

**2026-11-09, the ragged step revisited after the uniform one** (`core_form2.py`, verified equal to the first form
to 1e-14 on all cases): (1) NO BOUNDARY CASES -- the edges past the two ends are zero-width completions, their
messages zero-size arrays, every core runs the same body, both sweeps are folds with an initial carry (the scan form);
(2) the edge's content is `Y = [g_old Phi_P + g_new | g_old Xi]` and the new columns are the library's own
`linalg.pinned_leading_columns(old block, Y, k)` (S1) -- no hand-rolled QR, no matrix J; the seeded start is the
sibling primitive with seeds as content (0-40 % slower than the QR form at the tested sizes); (3) the stopping
measure is the completion's MOVEMENT `|P_old - P_new (P_new^T P_old)|_F`, which tracks the true frame movement within
15 % down to 1e-15 (use the direct form; `sqrt(k - |o|^2)` floors at 1e-8). The plan should be written from
`core_form2.py` (ragged) and `uniform_form.py` (uniform).

**2026-11-09: ALGORITHMS LOCKED IN (Nick).** Ragged = `core_form2.py` (no boundary cases, S1
`pinned_leading_columns`, movement measure); uniform = `uniform_form.py` (block layout, scans, `pad_safe_svd`). Now
under discussion: the UNIFORM GEOMETRY. Nick's lean: the geometry takes the frame's own masks. Facts found: (i) it
partly exists -- `UniformManifoldGeometryOps.from_frame(frame_data, plain_masks)` builds the variation masks from the
frame's own, the FRONTEND local model does this at every step (`fitting.py:369`), and `utv_retract` already reads its
target ranks off the frame masks; only the BACKEND loop builds one geometry `from_point(x0)` and reuses it
(`Problem.local_model` calls `self.geom.frame(x)`; `_variations` supplies `self.var_masks`). (ii) Prefix-from-ranks
sufficed so far because every frame came from the SVD-based orthogonalization of a minimal-rank point. (iii) The one
real downside: the geometry is value-hashed and part of the jit key, and every frame after the first is prefix again,
so a start frame with different masks or padded widths compiles the step programs once more for ONE Newton step
(T3Polynomial GPU runs: a level's compile is tens of seconds at rank ~20, 7-14 % of a large level). It arises only
when OLD RANKS VARY ACROSS EDGES (with uniform old ranks the block layout is already prefix, same masks). Session's
recommendation (sent): frame's-own-masks as the contract, plus the uniform loop returns its FINAL frame compacted to
junk-last at the standard padded width (one batched gather per rank axis, static indices -- not prototyped), so
nothing recompiles in practice. Awaiting Nick.

**2026-11-09, compile accounting (Nick's rules: recompile after a rank change fine; a small fixed number inside a
loop tolerable; repeated recompiles never; loop types A = Newton and B = gradient-based within rank continuation; the
alignment runs once per rank set).** Session's reading of the code (NOT measured): (1) nothing in the aligned start
recompiles repeatedly -- the geometry is value-hashed and every frame after the first comes from the SVD
orthogonalization with the same masks. (2) FOUND: `utv_retract` pads its output to the FRAME's padded widths
(`ut3_pad_ranks(new_data, frame_data[0].shape[-2], frame_data[2].shape[-1])`), so an uncompacted block-layout start
frame wider than the level's padding would inflate every array for the rest of the level and mismatch the backend
loop's geometry at step 2 => the uniform loop MUST return its final frame at the level's padded width, junk-last (one
batched gather per rank axis); with that the start frame's masks equal the level's and the frame's-own-masks contract
costs nothing. (3) Fixed extra compiles per level: the alignment step and the gradient-at-a-frame programs at the
oversampled structure (avoidable with oversampling 0) and at the block-layout target structure (shared with the
optimizer when old ranks are uniform). (4) Type A (`newton_cg`: host loop, eager local model keyed on shapes,
`_cg_core` keyed on the geometry aux): first step from the compacted frame compiles nothing new. Type B
(`mc_sgd`/`adam`: one jitted per-step kernel that builds its frame from the point): the first step from a supplied
frame is a different kernel, one extra compile per level, removable by carrying the frame in the kernel's state. (5)
Type B's hint must be ONE fixed tensor: the full batch or one frozen minibatch, never a fresh draw per pass.

**2026-11-09: NICK'S DECISIONS BEFORE THE NEW PLAN.** Scope: (1) the UNIFORM layer IS in this release; the four
unprototyped uniform pieces (seeded start in block layout, compaction at loop exit, stack axes, sharing groups) are
prototyped FIRST, by an Opus 5.5 max-effort agent in the research repo, no reviewer agent -- Nick and the session
review; (2) NEWTON ONLY gets the aligned start (gradient-based loops later); (3) the committed hinted construction is
DELETED from the library (tests use a dense per-edge reference); S1's two primitives and the hint-free start stay; the
uncommitted S4 member is discarded. API: (4) `g0_reference='cone'` keeps the FRAME-GRADIENT NORM at the aligned frame;
(5) loop defaults: oversampling 5, 2 oversampled passes, cap about 10, movement tolerance 0.05 -- ALL user-settable by
kwargs; (6) frontend twins: drop the three geometry twins, defer `frame=` on the six fitting factories; (7) uniform
geometry: CHECK-AND-REJECT a start frame whose masks differ from the level's (our loop returns matching masks).
Process: (8) MEASURE on Darcy first (laptop, ragged prototype), but do not over-weight small laptop timings; (9) a
FRESH plan, the old one archived; (10) superseded: "the hint is a T3" (now hint + project), "no uniform twin", the
fixed pass counts (old I-13: 4 gradient evaluations for the padded start, 10 for the cone reference), the rejection of
a vectorized uniform construction. Carried over: new module `backend/padded_frames.py`, no `dtype`, docstrings /
doctests / validation in the frontend twins, non-breaking additive release, the standalone math note after the code
(now based on Nick's note).

**2026-11-09: PROCESS FROM HERE (Nick).** RUNNING NOW (background agents, research repos only, nothing committed by
them): (a) an Opus max-effort agent prototyping the four remaining uniform pieces in
`nicks_research_experiments/tensor/aligned_padded_frame/` (README section "The remaining uniform pieces"); (b) an Opus
agent measuring the ragged prototype on the laptop Darcy ladder (`T3Polynomial/scripts/g06_*`, results in
`T3Polynomial/data/g0_reference/`). Review of both: Nick and the session (no reviewer agent). THEN: (1) the SESSION
writes the first draft of the FRESH plan (old plan archived); (2) a writer/critic pair of Opus 5.5 agents iterates on
it until they agree; (3) Nick and the session review; (4) implementation with the implementer/reviewer pair per slice.
Every plan and implementation agent runs at EXTRA HIGH effort and must first read a sampling of `TuckerTensorTrain`
methods and the backend functions they trace to (the only allowable precedent). The start and the cone reference each
get their own keyword group with the same defaults (session's choice; Nick may merge them).

**2026-11-09, late: TWO FINDINGS FROM THE AGENTS (both verified by the session).** (A) THE LOOP CYCLES. On Darcy level 1
(ranks 1->6, seed 0) the alignment loop never converges: period-2 cycle, consecutive frames differ by a sine of
0.09-0.13, |P_F G| alternating by 8e-7 relative; reproduced on a 243-number case by the core form, by the dense
per-edge reference, and by the session's dense version of the note's rooted variant. CAUSE (dense experiments,
`nicks_research_experiments/tensor/aligned_padded_frame/cycle/`): updating every edge from ONE projection is a
Jacobi-type simultaneous update of coupled edges and overshoots; the true stationary point (merit 0.22855) lies above
the cycle (0.22712) and is not an attractor. FIX: TWO-COLOUR updates -- after the truncating pass, odd passes update
the toward-root stores (up, left), even passes the away stores (right, down); non-updated edges are transported
(content = g_old). Converges monotonically (0 merit decreases), no parameter; on the full Darcy level-1 gradient after
p=5 for 2 passes it costs almost nothing early (35.9 vs 36.4 at pass 4, 37.0 vs 37.2 at 6) and its movement falls
(5e-4 and falling vs stuck at 1e-2); on the toy family it is as fast as all-edges. Inertia (proximal weight) slows the
real problem badly -- rejected. ALSO: on Darcy the movement reaches 0.05 only after 18-37 passes, values at 8-10
passes are ~93 % of the 40-pass value, the long-run value is seed-dependent (36.1 / 40.5 / 41.7), and at d = 24 the
defaults do not converge in 10 passes -- the cap is the effective stop. (B) UNIFORM PIECES DONE (agent; README section
"The remaining uniform pieces"; files `uniform_form2.py`, `uniform_start.py`, `uniform_loop.py`, `ragged_loop.py`,
`uniform_testing.py`, `check_uniform_{start,loop,stack,groups}.py`): start, loop + compaction, stacks, groups all
verified against the ragged code with gappy masks + garbage (<= 4e-14), whole loop under one jit (6 s, flat in d).
DEVIATION: the start that reproduces the committed ragged convention ('gs') needs a pad-safe Householder QR (the
convention is BASIS-dependent: TT/down coordinate seeds index neighbours' completion slots); the SVD-only start ('svd')
gives layer- and backend-dependent spans (loop output at the default stop differs by up to 0.19; the fixed point does
not depend on it). Also: the locked core_form2 step moves the frame on a ZERO hint (arbitrary completion of zero
content; core_form's QR form kept the old completion). PENDING: the Newton measurement (agent running, fixed pass
budgets); the agent adding the two-colour option to the uniform step/loop. DECISIONS FOR NICK: two-colour; stopping
semantics (cap as the real default); start convention (gs / svd / Gaussian seeds everywhere); tie-break by the old
completion on degenerate content; stacked stop = max over the stack.

**2026-11-09, Nick's decisions on the two findings:** (1) TWO-COLOUR updates after the truncating pass: YES. (2) Stopping:
the movement tolerance is an early exit, the pass cap (default 10) is the effective default; the cone reference is
"the value after the default schedule", deterministic given the seed. (3) Start convention: UNDER DISCUSSION (Nick asked
whether it concerns cold or warm starts; session's answer: both -- the start is F_0 of the loop in both; what is at
stake is cross-layer identity of the start frame, which needs identical BASES upstream, not just spans; options: A =
committed convention (Haar Tucker + coordinate TT/down) with a pad-safe Householder pinned completion as the uniform
twin -- equals `fv_zero_point_frame` at s = 0; B = Gaussian seeds everywhere + SVD-only + a column sign rule; C =
SVD-only with coordinate seeds -- rejected, tied clusters). (4) Degenerate content: ACCEPT arbitrary completions where
the hint carries no information (the old plan's stance). (5) Stacked loop stops on the maximum movement over the stack:
YES.

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
