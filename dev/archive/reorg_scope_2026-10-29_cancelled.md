# Post-curvature reorganization — scope note (Nick, 2026-10-27) — **CANCELLED 2026-10-29**

> **CANCELLED (Nick, 2026-10-29):** after looking over the code during the pause, Nick concluded the organization is
> fine and the reorganization would be code churn for no good reason. **Slice R is not happening.** The note is kept as a
> record; the one item in it that is a real backlog item — moving the random constructors off the legacy global
> `np.random` stream (explicit `Generator` / jax-key arguments) — moves to the HANDOFF backlog as a standalone small
> change. The release follows the close-out below instead: a final bug/consistency review, fixes, the T3Polynomial
> update, the CHANGELOG, the tag.

> **Status: QUEUED, not started.** Nick's instruction: *do not do this while the exact-Hessian implementation is in
> flight.* **Sequencing (Nick, 2026-10-27 19:33): R goes at the VERY END of the slices, after a PAUSE** — the
> exact-Hessian arc (S10–S13: every kind exact, docs, CHANGELOG, the full gate, the `dev/` archive, no tag) completes
> first — including a PRELIMINARY math note (S15a) — and we stop there; then R, then S14 (T3Polynomial), then the
> final math-note + CHANGELOG pass for R's renames (S15b), then the release tag. 2026.3.0 is
> already a major breaking release, so this is the moment to re-organize, streamline and simplify before it ships. Planning will follow the same
> writer/critic process the curvature plan used (`dev/curvature_impl_review/`), with Nick adjudicating.

## Nick's three areas (his framing, lightly condensed)

1. **Sampling and jetted sampling.** `apply` / `entries` / `probing` each have their own module, but all their jetted
   versions live in one monster `backend/sampling_derivatives.py`. Can (or should) it be split? More generally, is
   there a cleaner organization of the sampling + jetted-sampling code?
2. **The fitting layer.** It grew organically to support the different fits we wanted; parts of the structure are
   convoluted. Every feature addition and refactor has found it confusing and thorny. Reason more generally about what
   the layer is *trying to do*, and find reorganizations that make it simpler while preserving the functionality.
3. **The optimization layer.** Two issues. (a) **Coupling:** the optimizers are general-purpose algorithms that do not
   depend on the tensor-network structure being fit — they only need gradients, Hessian-vector products, retractions,
   and so on — yet the layer is coupled to the rest of the project more than it needs to be. (b) **Typing:** manifold
   algorithms lend themselves to a type system over **manifold points, tangent vectors and covectors**, with the
   operations between them. Today this is jumbled; strict typing would prevent confused operations (e.g. using a
   gradient as a vector instead of a covector). In the gauged parameterization these spaces are isometric and
   Euclidean, so no raising/lowering is numerically needed — but it is proper to *write* the raising and lowering and
   have them be null-ops. Reference: Nick's independent optimizers project,
   <https://github.com/NickAlger/nicks-optimizers>, where strict type enforcement for manifold operations has proved
   very useful.

## Candidates to put in front of the planners (orchestrator's notes; not rulings)

- **Ragged/uniform duplication.** S8 showed the ragged T3–T3 sweep and two-hole can be made polymorphic over
  supercores; the library otherwise keeps `t3_*` / `ut3_*`, `fv_*` / `ufv_*`, `tv_*` / `utv_*` twins whose uniform
  half is "mask on entry, pack, delegate, re-mask". Worth asking how much of the twin surface can become one
  polymorphic kernel plus a thin masking wrapper (the equivalence contract already says the uniform layer is a *faster
  ragged layer*).
  - **Nick (2026-10-27): quite a bit, but not all, of the pre-existing code is already polymorphic, and the
    non-polymorphic parts fall into two classes that must be told apart.** (i) **Sweeps** were good to make
    polymorphic because the supercore layout puts the core index `d` first (`docs/uniform_supercore_layout.md`), so a
    sweep over cores is the same code on a tuple of cores and on a supercore. (ii) **Operations that are parallel
    across cores** were *intentionally* left non-polymorphic: on the uniform layer they are ONE big contraction over
    all cores at once (the `d`-prefixed grouped string), which is better than the same small contraction mapped or
    scanned over the cores — so there a polymorphic sweep-style kernel would be a regression. (iii) Elsewhere the code
    is non-polymorphic for no good reason. **The planners' first job on this item is the audit that sorts every twin
    into (i)/(ii)/(iii)** — collapse (iii), keep (ii) as deliberate (and document it where it isn't), and treat (i) as
    the already-done pattern to copy.
- **The `_from_sweep` twins** (`*_corewise_transpose` vs `*_corewise_transpose_from_sweep`, the two-holes likewise)
  and the precompute/sweep objects: the fitting layer's thorniness is partly this seam — the sampling kinds own the
  sweep, the model owns the frame, the geometry owns the base point/aux. A typed `Point`/`Tangent`/`Cotangent` layer
  (area 3b) may absorb the sweep as "what a sampling operator precomputes at a point".
- **Geometry vs kind vs model** (`backend/geometry.py`, `backend/fitting.py` `SamplingKind`, `backend/optimizers.py`
  `LocalModel`/`Problem`): the parameters-not-closures restructure (2026.2.0) fixed the hashing story; the
  *responsibilities* are still spread over three protocols plus the shared wrapper and the uniform twins. Area 2 and
  3a together.
- **Weights.** `T3Weights`/`T3FrameWeights` + uniform twins, the fitting-objective `ω[mode,order]` matrix, and the
  regularizers are three different "weight" notions; a typed covector/metric vocabulary (3b) could name them properly.
- **Constructor `use_jax` flags vs inferred dispatch** (the documented exception) and the `safety` contextvar — small,
  but a breaking release is when to settle them.
- **Random number generation — update the legacy `np.random` usage (Nick, 2026-10-27 20:10).** The library's random
  constructors (`TuckerTensorTrain.randn`, `t3_corewise_randn`, `common.randn`, the oracle sweeps' draws, …) draw from
  the legacy global `np.random` stream (`np.random.seed` / `np.random.randn`), while newer code and the tests use
  `np.random.default_rng` / `Generator` objects and jax keys. This bit us twice in the curvature arc: the plan's E2
  restart numbers were one unseeded draw, irreproducible at any commit (`check_restart_tr.py` seeded only its
  `default_rng(7)`, not the global stream `randn` uses — S10/S11 review), and the slow-tier O2 oracle sweep was flaky
  under `-n 2` for the same reason (S1/S2 review). R should move the library to explicit `Generator` / jax-key
  arguments (`rng=` on every random constructor, the canonical zero-point frame's keyed `RandomState` recipe reviewed
  for the same treatment while keeping its pinned values), with the legacy global stream used nowhere.

- **Test organization:** `tests/_jet_trs_twins.py`, the two contraction tables (interpreter + sharding) that must be
  kept in step, the oracle sweeps — whatever the module split becomes, the tests should split with it.

## Sequencing

`S10`/`S11` (in flight) → `S12` → `S13` (docs + CHANGELOG + the full gate + the `dev/` archive; **no tag**) → `S15a`
(a PRELIMINARY math note) → **PAUSE** (the curvature arc is complete; hand off) → **R: reorganization (plan →
adjudicate → slices)** → `S14` T3Polynomial → `S15b` (the math note's code references + the CHANGELOG updated for
R's renames) → **release tag**. S13's user docs will need the same pass after R; that pass is part of R's own plan.
