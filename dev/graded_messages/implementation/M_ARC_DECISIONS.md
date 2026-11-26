# Decisions Claude made during the M slices, for discussion with the maintainer after the arc

The maintainer (2026-11-22): "Go ahead with M2 and continue through the rest of the M slices, unless a major issue
arises. Please track decisions you make and we will discuss them after the M slice arc is complete." Each entry: the
decision, the reason, where it is recorded. Larger rulings are also in `LOG.md`.

## M0 (the `J -> P` rename; committed `64465d8b`)
1. `F_in,i` is spelled `F_in_i` in code, comments and docstrings (an identifier cannot hold the comma; same width as
   `J_in_i`, so aligned columns do not move); prose writes `F_in,i`.
2. The shape rule: a concrete shape spells the sizes (`F`, `min(F_in_i, F)`); a family of shapes spells the group
   (`P + W + [K] + C`); the packed input is `(d,)+F_in+W+(N,)`.
3. The test-local jet ring `J = Jets(order)` in `test_sampling_definitions.py` renamed `ring`.
4. A static guard that no library contraction literal contains `J`, and a floor in the literal scan (a missed `'J'`
   filter would otherwise empty a test's loop silently).
5. The baseline for per-test counts is one local `-v` run of a `git archive` copy (not CI), kept gzipped in
   `implementation/m0/`.

## M1 (`data_metric`, `blocks_to_weight`, the law helper; committed `c16aa640`)
6. `_with_K` (the engine's `K`-aware weight alignment) moved from M1 to M2, where the engine first calls it.
7. `data_metric` with a packed `r` and a ragged tuple weight raises (naming `pack_batch`) rather than reading the tuple.
8. `LeastSquaresBatch.data_metric(r)` raises by rank on today's block-weight batches until M2 switches batches to the
   full form (pinned by a test), rather than converting blocks inside the method.
9. The frontend dual-jet converters leave `t3toolbox.fitting`'s namespace entirely (they are not in 2026.5.0).
10. A ragged `r` needs the exact mode count; every weight axis must be 1 or `r`'s size (so `data_metric` never grows
    `r`); the scale must be 0-d.
11. Type comments write the algebraic types in Unicode (`⊗`, `⊕`, `ℝ`, `*`), identifiers stay ASCII (`omega`,
    `gamma`); recorded in `docs/contributor/signature_style.md`.
12. The law helper is `tests/_typing_laws.py` (the plan's `_typing` shadows a CPython module).

## M2 onward
13. M2 (design, `M2_DESIGN_NOTE.md`, `M2_DESIGN_REVIEW.md`): six corrections to the plan, found by the implementer and
    verified by the reviewer, adopted. (a) The engine checks `y` and `omega` structurally against `z` (equal rank, each
    axis 1 or `z`'s size, exact mode count): without the old alignment a `y` without `C` on a stacked model is
    right-aligned and silently wrong (plan section 2 said "fails in broadcasting", true of sizes, not of rank).
    (b) The probe `y` is masked on the uniform layout whatever its edge (an edge-1 `y` otherwise put `-y` on the
    padding of `r`, and `f` summed it). (c) `f_resolution`'s `<y, M y>` and M3's `y_block_sumsq` are taken over `z`'s
    shape (`y` broadcast to `r` first; the plan's literal sum undercounted by `|W|` and raised for a fuller `omega`);
    today's display ratio on a stacked model with shared `y` is already off by `sqrt|C|`, which M3 fixes. (d) The
    checkers accept a `d`-leading array, first axis exactly `d`. (e) `check_batch` (the protocol's optional member,
    unreleased) gains `stack_shape`, always passed by the frontend. (f) The "weight's shape" wording fixed where M2
    makes it false.
14. M2's gate: the battery bitwise for `w_tilde`, `g`, `JtJp`, `Kp`, `r`, `z`, `Jp`, the block sums; to rounding for
    `f`, `Jp_norm_sq`, `f_resolution` (`r.(omega^2 r)` regroups `omega^2 (r.r)`) and for the jitted run; the short
    optimizer runs with the same iteration count and stop reason. Expected-red lists recorded by a run per part.
15. M2 part 1 changes the reference module `sampling_definitions.py` (in the package, called by nothing in the
    library): ruled in scope, since its full-form reading is part 1's subject ("no library code" meant the engine,
    `jets`, fitting and the frontend). The 20 MB pre-M2 battery record stays out of the repository (deterministic,
    regenerable from `85c6e54d`).
16. M2 part 3: on `m2-wip` the oracle's `f_resolution` is wrong between parts 3 and 4 (its old `weight**2 *
    y_block_sumsq` broadcasts into cross terms on the full-form default weight that `prepare_derivative_data` now
    returns; a factor of about 1.02 on the objective floor); left to part 4, whose first item rewrites it. The branch
    is squashed before it reaches `graded-messages`, and the float32 rounding-gate test is red meanwhile as its guard.
17. M2 implementers are rotated when their context fills (part 2 was finished by a second implementer from the
    first's handover; a third takes parts 4 and 5); the sub-arc's reviewer keeps a handoff note for a successor.
18. M2 part 4: `take` raises on a data-space object whose `W` axes are missing or of the wrong size (structural), so a
    block weight on apply cannot be silently reread as a legal per-slot weight on a drawn batch. The test helper
    `tests/_fits.py` keeps the block vocabulary, placed through `blocks_to_weight` (test code only).
19. M2 part 5: the battery's two `mc_sgd` runs' cores differed between the trees (a sign-gauge flip of the SVD
    retraction under a rounding-level change of the step, possible from any start, per the part-5 review; first
    misattributed to the zero start) while losses (4e-16), counts, stop reasons and the
    represented tensors (4e-15) agreed. Under the knife-edge ruling (change the run, not the tolerance) they start at a
    seeded random point; the pre-M2 record was re-recorded on the base tree (bitwise equal to part 1's elsewhere).
20. M3: `format_newton_iter`'s public parameter `obj_unweighted` (shipped in 2026.5.0) is removed with no shim, with
    the `(unwt ...)` figure; P19's upgrade note says "remove the fourth argument" (with no table the old positional
    call passes silently, so no error is promised).
21. M3: the display's validation cells use `val_batch`'s own weight (documented in `make_newton_display`, the frontend
    `newton_cg` and the oracle, and pinned by a test), rather than borrowing the training weight.
22. M3: the frame's stack shape `C` is read by a private helper (shared by `y_block_sumsq` and `f_resolution`). The
    reviewer proposed making it public as `frame_stack_shape` beside `frame_shape`; a new public name is the
    maintainer's, so it is kept private for now and raised here.
23. M3: `examples/fit_probe_display.py`'s `unwt` text fixed in M3; the page `docs/fitting_and_optimization.md:114`
    (the ratio sentence) goes to P19's list.
24. M4 designed without a separate design review (a scripted rename plus prose; the build is reviewed). The `c` line
    gets its type in M4 (`P*⊗V_i` probe, `P*` apply, `ℝ` entries), since "of the type of z" is wrong for `c`; the other
    signature lines wait for M5. P19's upgrade note: "`residual=` -> `c=`; positional calls unaffected; no shim".
25. M5 designed without a separate design review (comments only). Types per argument and return line only; the
    plan's function-type table goes to P19's page. The corewise point `x` is typed `X`; the manifold frame and `Vc`
    get a role only; `omega` a role only (its square is `M`'s diagonal, `omega` is not in `Y`). `P_in_i` (the input
    jets' ring) and `index^d` in comments; the glossary's `P` row gains `P_in,i` and `index`. The `T3Tangent`
    transposes return `X*` in a `T3Tangent` container, before the projection. Comment lines wrap at 120 with `#   `.
    `LocalModel`'s comments retyped (`J^T(M r)` with the scale). The engine's table gains a `Type` column (`P`, `Y*`,
    "`Y`, turned" for `w_tilde`, "`P*`, turned" for the other tilde variables, which sit on edges; `X*` for the
    blocks), pinned by a test that reads it.

## The maintainer's rulings (2026-11-24, the discussion after the arc)

Entries 1 to 6, 8, 9, 12, 14, 15, 17, 21, 23 and 13(a) to (c) were put to the maintainer as routine or right, and he
raised no objection. Entries 7, 10, 13(d), 13(e), and two unlogged choices of form (`blocks_to_weight` and
`prepare_derivative_data(weight=None)` returning one mode-leading array for a ragged probe `y`) are overtaken by the
packing rulings (`PACKING_REFACTOR.md`) and are decided there. His explicit rulings:

- **11** (Unicode in the type comments): agreed.
- **16** (a side branch knowingly wrong between parts): "it is acceptable for a side branch to be temporarily wrong
  during wip, as long as it is correct at the end."
- **18** (`take` raises on a data-space object with no `W` axes): keep the raise. What it guards is consistency, not a
  wrong number: without it `all(())` is true, a block weight `(3,)` on apply becomes `(3, 1)`, legal on the drawn
  batch, so a batch the full-batch check refuses would run under `mc_sgd` and `adam`.
- **19** (the battery's `mc_sgd` runs re-seeded): fine. "Trying to keep all optimization steps bitwise is an admirable
  goal but not strictly necessary." Rounding across machines can make optimization paths diverge to O(1) before they
  meet again at the solution. The goal is to be sure the method works; where bitwise agreement is not available,
  correctness is assessed in other ways.
- **20** (`format_newton_iter`'s released `obj_unweighted` removed with no shim): accepted.
- **22** (`frame_stack_shape`): **make it public.** A library change: carried by the packing sub-arc's plan.
- **24, 25** (M4 and M5 built without a design review): acceptable here; judgement in future. "Comments are important.
  Scripted changes of comments have caused problems before (e.g., breaking ascii diagrams, vertically misaligning
  carefully handcrafted signature type comments, etc). When in doubt, do the review."
- **25, the tangent transposes' return** ("`X*` in a `T3Tangent` container"): left as it is for now; a known pun of
  the class holding both a tangent and a cotangent.

Recorded with them, not logged at the time: the M2 CI fix moved one test bound from 9 to 17 at a new seed
(`M2_CI_FIX.md`; the new test separates its mutant by 27 against 17, the old one by 11 against 9).

**Still pending:** the audit's decisions 3 (corewise as point methods or as `COREWISE` tangent operations) and 4 (the
name "Hessian"; `t3_inner_product_corewise_hessian`), kept pending by the maintainer; the `from_t3` slice.
