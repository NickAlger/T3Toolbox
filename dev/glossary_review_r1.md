# Glossary review, round 1 (critic)

Scope: `docs/glossary.md` (239 entries + symbols table) and `dev/glossary_naming_decisions.md` (16 NAMING, 21 ERROR,
389 parsed work-list rows), checked against the code at `99006763` plus the uncommitted glossary files.

**Checked against source (≥ 60 entries):** absorb, boundary bonds, central core, CP factors, determinantal variety,
edge, edge mask, edge weights, gappy mask, group ceiling, grouped T3-SVD, Hadamard product, Hilbert–Schmidt, mask
holder, matricization, minimal ranks, orthogonal core, oversample, packed vectors, pad-safe SVD, padded widths, prefix
mask, ranks, raw sweep ranks, real slot / padding, resize, share, squash tails, stratum, supercore, T3-SVD, TT core,
Tucker core, varying-rank stack, zipper; aligned padded frame, alignment pass, ambient projection, base-point tangent,
block layout, captured hint energy, carrier, compaction, completion, cone merit, COREWISE, corewise frame, empty frame,
fixed point, frame cores, gauge conditions, gauge projection, geometry aux, hint-free start, manifold dimension,
movement, oblique projection, orthogonal representation, padded frame, pinned completion, representation dependence,
retraction, same-mask precondition, schedule, seed rows, slack, store, structurally zero, tangent cone, two-colour pass,
variation masks, vector transport, zero-point frame; apply, chunk_size, containment, corewise transpose, draw, edge
variables, entries, frame sweep, Jacobian / bare Jacobian, multilinear extension, one-/two-hole sweep, sampling kind,
symmetric directional derivatives; adam, Armijo, Cauchy step, cone reference, continuation_ranks, curvature damping,
curvature term, exact Hessian, flat slot, forcing term, g0 reference, gtol_rel, history, lift, local model, mc_sgd,
newton_cg, oversampled ranks, padded start, stacked point, stats, step control, stop reason, trust region, use_jit,
eager jax, equality is explicit, import-graph test, jax-absent policy, safe mode, scan-body principles, seam, use_jax.
Everything not listed below is true of the code.

**All 21 ERROR items are real** (deciding lines: E1 no `numerically_equal` in `t3toolbox/`; E2
`backend/optimizers.py:630` vs `:1161-1163, :1208, :1328`, `optimizers.py:518`; E3 `common.jax_or_warn`; E4
`backend/ut3_operations.py:405`, `ut3_linalg.ut3_add`; E5 no "acyclic" in `tests/test_contractions_interpreter.py`;
E7 file absent; E8 `fv_base_point_tangent` puts `P_last = left[-1]` in the last slot, cores are `L0…L(d-1)`; E9
`manifold.py:126, :288`; E10 `T3Frame` docstring `U_ia V_ja = 0`; E11 `common.py:314-358`; E12, E13, E14
`common.py:753`, E15 `geometry.py:17`, E16, E17, E18, E19, E20, E21 as stated). See MINOR 5 for the one E4
replacement that over-reaches.

**`sphinx -W`:** clean (`build succeeded`, `glossary.html` written, every anchor resolved). The index toctree and README
link are right.

## MAJOR

1. **"orthogonal core" defines up/down-orthogonal the opposite way from the API's method names.** The entry says
   "down `ni × (ri·r(i+1))` rows, up (Tucker) rows". The library uses two conventions, and the glossary picks the
   frame-store one without flagging it:
   - The methods and the public docstrings call an orthonormal **Tucker** core *down*-orthogonal:
     `down_orthogonalize_tucker_cores` ("all Tucker cores down orthogonal", `tucker_tensor_train.py:3140`,
     `:3159` checks `B @ B.T == I`), `is_left_orthogonal` / `is_right_orthogonal` (`:780`, `:814`), the `t3svd`
     `assume_orthogonal` docs (`:4143`, `backend/t3_svd.py:67`), and `t3_orthogonality_residual`
     (`t3_orthogonalization.py:38`). They call a TT core that is orthonormal in its mode-leg unfolding
     *up*-orthogonal: `up_orthogonalize_tt_cores` ("all TT cores up orthogonal", `:3187`), and
     `ut3_orthogonalization.py:215`.
   - The frame docs call the down store `O`, which is orthonormal in exactly that mode-leg unfolding, "down-orthogonal"
     (`user_guide.rst:142`, `frame_variations_format.py:1372`, `backend/probing.py:647`). They also call it
     "outer-orthogonal" (`frame_variations_format.py:118`, `manifold.py:1343`). One comment calls a Tucker core
     "up-orthogonal" (`t3_orthogonalization.py:147`). `fv_conversions.py:180-184` even binds
     `up_orthogonalize_tucker_cores = t3_down_orthogonalize_tucker_cores`.

   So "down-orthogonal" means the Tucker condition in the method docs and the `O` condition in the frame docs. A
   reader who uses the glossary to read the `t3svd` docstring gets it backwards. **Fix:**
   - State the API (method) convention in the entry: a Tucker core is *down-orthogonal* (`B Bᵀ = I`) and a TT core is
     *up-orthogonal* in its mode-leg unfolding (alias: outer-orthogonal). The frame's down cores `O` have the
     up/outer property.
   - Flag the collision in the entry.
   - Add a NAMING item, N17, for Nick: retire "down-orthogonal" for `O` and pick "outer-orthogonal" or
     "up-orthogonal" for it.
   - **Hold every N4 row that rewrites "outer-orthogonal"** until N17 is decided (`t3_orthogonalization.py:106`,
     `frame_variations_format.py:118`, `:365`, `manifold.py:1343`). Replacing it with "down (`O`)" drops "orthogonal"
     and swaps the one unambiguous adjective for the ambiguous one. N4 is right for the *core symbol* (`D` → `O`) and
     the *store name*; it is wrong for the *orthogonality adjective*.
2. **The N4 row at `tucker_tensor_train.py:962` would falsify a correct comment.** The line reads
   `# keeps the outer TT bonds r0=rd=2`. "Outer" here means the boundary bonds, not the down core. Applying "down (`O`)"
   makes the comment false. Drop the row, or reword it to "boundary TT bonds".

## MINOR

1. **"group ceiling" has an index error.** It writes `n_g ≤ min(N_g, Σ min(N_g, r_{i-1}·r_i))`. With 0-based modes,
   `i = 0` gives `r_{-1}`. The glossary's own TT-core and minimal-ranks entries use `ri·r(i+1)`, and so does
   `t3svd_minimal_ranks.md` (`rL_i = r_i`, `rR_i = r_{i+1}`). Write `r_i·r_{i+1}`, and add E19-class rows for
   `sharing.md:173` and `:176`, the source of the error.
2. **"movement" says "the uniform record is NaN after the last pass"**, but the stats keys the entry names drop that
   tail. `start_movements` and `cone_movements` go through `_movement_record` (`backend/optimizers.py:866-874`).
   Only `ufv_aligned_padded_frame`'s own return is the NaN-padded `(max_passes,)+C` array. Say so.
3. **"absorb" lists `.absorb`, which does not exist.** The methods are `T3Tangent.absorb_weights` and
   `UT3Tangent.absorb_weights`.
4. **"minimal ranks" says "the uniform fitting path requires it of `x0`".** `least_squares_problem` reduces `x0`
   itself (`optimizers.py`: `uniform_minimal` always, and `_ragged_minimal` on manifold geometries), so "a frontend user
   never meets that requirement". The sentence also contradicts "No operation needs it". Suggested wording: "...; on a
   manifold geometry the fitting layer reduces `x0` to minimal ranks (the uniform path needs a minimal-rank frame)."
5. **"gappy mask" and the E4 replacement at `uniform_tucker_tensor_train.py:55` over-reach.** There is no `⊙` of two
   uniform T3s: `UniformTuckerTensorTrain.__mul__` is scalar-only and raises otherwise (`:253-262`). On the uniform
   layer, rank masks go gappy under `+` (`ut3_add`). Only the *weights'* `kronecker` Kroneckers masks. Separately,
   "nothing compacts them" is wrong: `ut3svd` re-canonicalizes and `ufv_compact_frame` compacts. Say "arithmetic
   never compacts them". The E4 replacement "gappy after `+` / `*`" should read "gappy after `+`".
6. **"store" says each store is "capped only in its own unfolding".** That is false for the ranks the orthogonal
   representation actually has. `compute_orthogonal_representation_ranks` (`ranks.py:644-674`) computes up, then
   left, then right *over the left chain* (`right_i = min(left_i, up_i·right_{i+1})`), then
   `down_i = min(up_i, left_i·right_{i+1})`. The brief also expected an entry for orthogonal-representation ranks, and
   there is none. Fold it into "store": "set by the orthogonalization sweep (up, left, right over the left chain,
   down); each core orthonormal in its own unfolding; the differences are the slack."
7. **The "sampling kind" entry misstates the protocol a user subclasses.** The six operations are `precompute`,
   `forward`, `transpose`, `point_forward`, `take` and `corewise_hessian` (`backend/fitting.py:335-380`). The `‖·‖²`
   reduction is an extra.
8. **"stratum" (and "determinantal variety") contradict themselves.** The entry says the masks name the stratum, then
   says a zero-padded point sits below its stored ranks. Use the source's own resolution
   (`uniform_ranks_and_varieties.md` "what the mask actually means"): the masks name the *nominal* stratum, and zero
   content can put the tensor lower.
9. **"frame cores" says "Down: the Tucker complement where `V` varies".** `O` is a TT-shaped core `(rL, nD, rR)`. It is
   the TT core of the Tucker-variation terms and is orthonormal in its mode-leg unfolding; `Vi` varies in the
   complement of `Ui`'s rows. The phrase is copied from the loose table in `frame_variations.md`.
10. **Coverage gaps.**
    - The T3-level orthogonal forms are missing (left-/right-orthogonal T3, `is_left_orthogonal`,
      `assume_orthogonal`, and the orthogonality center, i.e. the last or first TT core). They are needed to read
      `t3svd` and `rank_adjustment_sweep`.
    - The **shape mask** (`shape_mask`, always prefix, derived from `shape`) is missing. It is used in
      `uniform_masks_vs_ranks.md` and `uniform_ranks_and_varieties.md`.
11. **NAMING: I disagree with applying N12 (junk → padding).** All 35 occurrences of "junk" are in padded-frame or
    pinned-completion text (`uniform_padded_frames.py`, `padded_frame_internals.md`, `linalg.pad_safe_pinned_completion`,
    the two contributor lines). That is exactly where "padded" already means the zero-padded rank increase and its
    *real* new slots. The rename would create, at every site it touches, the collision that N14 removes elsewhere.
    Recommendation: skip N12. Keep "padding" canonical (rules a and b; `pad_safe_*`), and list "junk slot" in the entry
    as the word the padded-frame code uses. Making "junk slot" canonical is the other consistent choice, since it is
    the maintainer's term. Either way, there should be no bulk rename.
12. **The work list has gaps.**
    - N16 misses `backend/sampling_derivatives.py:740, 770, 1359, 1370`, which use the same words.
    - E17-class: the `T3Frame` docstring gives `tt_variations elm_shape=(rLi, nUi, rRi)` (it should be `rR(i+1)`), and
      the left/right lines write `ni` for `nUi` (`frame_variations_format.py:73-78`).
    - E1-class: `backend/optimizers.py:551` cites a nonexistent `common._jax_or_warn`.
    - N8 leaves "seeded padded frame" in five docstrings (`frame_variations_format.py:309-310`,
      `uniform_frame_variations_format.py:241, 244`, `padded_frames.py:254`, `uniform_padded_frames.py:182`). This is
      harmless, but the list should either include them or say it skips them.
13. **The E6 rows are not mechanical.** All 73 share the instruction "delete the code, keep the reason (in a runtime
    message: delete the parenthesis)", where each row needs its new text. The decisions file also counts *four*
    runtime messages, but there are seven:
    - `shared_geometry.py:186, 192, 207, 210` (R9-9)
    - `manifold.py:1249` (H3-6)
    - `backend/stacking.py:526` (R2-12)
    - `backend/linalg.py:225` (R3-6)

    Give each runtime row its reworded message. Rewording is safe: the tests use bare `assertRaises`
    (`tests/backend/test_structural_guards.py`, `tests/test_sharing.py`).

## NIT

1. "fixed point" / "captured hint energy": the saddle is *proved for matrices* (`padded_frame_cone.tex` Prop. 1). On
   T3s, only stationarity and the upward single-store rotations were measured. Add "(proved for matrices)".
2. "scan-body principles": the page says *prefer* module-level bodies, applied within reason, and allows memoized
   specializations (§4). "Are module-level and close over nothing" is stronger than that.
3. Armijo: `newton_cg` makes 40 trials, which is 39 halvings (`alpha` halves between trials, so the last trial is
   `2^-39`). `gradient_descent` makes 50 trials. "Trials" is the exact word; "halvings" matches the docstrings.
4. "gtol_rel": `gradient_descent` stops on it too.
5. Aliases that name a different thing:
   - "Riemannian gradient" on "ambient projection". It is a result, not the op, and "gauge projection" claims it too.
   - "uniform bump" on "useless-rank removal".
   - "unfolding" on "matricization", while the glossary uses "unfolding" for core reshapes.
   - "Riemannian Jacobian" on "bare Jacobian". Mark it "(older docs)".
   - "P-block" appears in two entries.
6. Private test names: the `HISTORICAL` and `_CONFIGS` entries name private test internals. Drop the names or the
   entries.
7. Symbols table: add `d` (number of modes, and the leading `d` axis or derivative prefix in local names), `m` (the
   max order in the tex note, the cone merit `m_c`, the GN model `m(p)`), and `ε` (the lift damping, the machine eps in
   `objective_floor`, Adam's eps).
8. Small wording fixes:
   - "warm start ... the `'warm_start'` reference is ‖g(x0)‖ instead" is unclear.
   - The T3-SVD aliases mix Algorithm 9 (dense) with Algorithm 10 (implicit).
   - "TT rank = bond between neighbouring TT cores" excludes `r0` and `rd`.
   - "backend = stateless functions" ignores the parameter-binding classes.
   - `t3m` only optionally truncates.
   - "q-quadratic", "Haar" and "Eisenstat–Walker" are used without definition.
9. N3: `FRAMES` is fine. Any single word next to `ORTH`/`GAUGE`/`TIED` works.
10. Decisions-file details:
    - The E8 note says `aligned_padded_frame.md` links the anchor; only `glossary.md` does.
    - The four N2 doctest rows (`fitting_and_optimization.md:52, 55, 76, 78`) cover every use of `K` in that page's
      session. They are safe, but must be applied together, and the page doctest must be re-run afterwards.
11. Backlog, not glossary: `TuckerTensorTrain.__mul__` carries an unused `use_jax=None` (`tucker_tensor_train.py:1967`).

## The three code bugs the writer reported

1. **A `'cone'` run's verbose display names no reference: REAL.** `optimizer_display._REFERENCE_NAMES` (`:121`) lacks
   `'cone'`. A run confirms it: the converged line ends "…‖g₀‖, g₀ = zero tensor)" for `'zero'` but only "…‖g₀‖)" for
   `'cone'`. The fix is cosmetic: one dict entry plus a display test.
2. **"leftward/rightward" for two left-to-right sweeps: the facts are right, but it is not a code bug.**
   `compute_mu` and `compute_tau_tilde` are both forward `xscan`s. The words name the *side of the forward variable*:
   `tau_tilde` is the adjoint of the right-side `tau`. `_tau_tilde_step`'s "adjoint rightward sweep" is even literally
   right as a travel direction, while `_mu_step`'s "leftward sweep" is not. The N16 rewording is correct and worth
   doing as a docstring clarity fix; extend it per MINOR 12.
3. **`xwhile` is not returned by `get_backend`: REAL, as a doc error (E11), not a code bug.** `get_backend` returns
   `(xnp, xmap, xscan)` (`common.py:337`). `xwhile` is separate and uses `lax.while_loop` only under `use_jit` with a
   jax state, so the eager-recompile problem on that page does not apply to it. The E11 replacement is accurate.

## Naming verdicts

- **Tucker core** (over Tucker factor): agree. The rule (a) data field is `tucker_cores`, "central core" handles the
  collision, and there should be no bulk rename.
- **padding vs junk slot**: disagree with N12 (MINOR 11).
- **edge mask** (over rank mask): agree. The `UT3Masks` fields are `tucker_edge_mask`/`tt_edge_mask`, there should be
  no bulk rename, and "rank mask" stays an alias.
- **N4**: split it as in MAJOR 1, and add N17.

Verdict: REVISE (2 MAJOR, 13 MINOR, 11 NIT)
