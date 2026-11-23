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
