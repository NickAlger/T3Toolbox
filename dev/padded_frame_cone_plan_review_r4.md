# Padded frame / cone reference plan — critic review, round 4 (plan rounds 5–6)

**Plan:** `dev/padded_frame_cone_impl_plan_2026-11-02.md` (round 6, 2229 lines; CHANGES :69-78).
**Previous reviews:** `_review_r1.md` … `_r3.md`; the two design answers `_q_old_point.md` and `_q_vectorize.md`.

**New probes** (read-only, `t3toolbox` env, session scratchpad):
- `probe_seed_degeneracy2.py`: in-block coordinate seeds against library-produced old frames.
- `probe_allclose_N.py`: `UniformTuckerTensorTrain.allclose` with a different padded `N`.
- `probe_imports_r6.py`: the round-6 placement in the AST import graph.

## Overall assessment

Rounds 5 and 6 are carried through well, with one gap. The maintainer asked whether any of his calls creates a serious problem. None does:

- **I-28 revert.** Correct, provided the in-block seeds became coordinates (the writer saw this and fixed it).
- **The old point.** It removes every layout requirement and gains only two guards.
- **No uniform twin and no vectorization.** Both are consistent with the equivalence contract.
- **Plain keywords and a pure rank function.** Both are simpler.

**The gauge-covariance argument is right.** Under a sign gauge of the old frame, every slot of the hint-free start maps to a signed copy of itself. The "rare degenerate seed" case is not handled structurally. It cannot be without a value-dependent selection, and that would bring its own near-tie fragility. The probe confirms the case is rare on the old frames the library actually produces. Accepting it is right, but the claims around it need tightening.

Three MINORs keep the verdict at REVISE. Each is a few lines of text:

- **A.** The newton_cg keyword placement breaks positional calls, which contradicts the non-breaking decision.
- **B.** The `1e-4` advice brings back round 1's m7 conflation in shipped docs.
- **C.** The list of exceptions to cross-layer agreement is inconsistent, and "replaces" overstates.

## The coordinator's focus items

**1. Coordinate in-block seeds: is the hint-free start gauge-covariant in every slot?**

Yes, under the sign gauge that separates the two layers at a generic point. Slot by slot:

- **Tucker.** `pinned_completion(U°ᵀ, Gᵀ)` depends only on the projector `U°ᵀU°` and an ambient draw, so it is invariant.
- **Outside coordinates.** A sign gauge `D_a ⊗ D_n` maps each unit vector `e_(a,n)` to `±e_(a,n)`, and the selection
  "first `t` in index order" does not move.
- **In-block seeds.**
  - The same holds for the coordinate seeds. The residuals transform as `(I − P'P'ᵀ)E = (D⊗D)(I − PPᵀ)E·D_E`.
  - Householder QR with the positive-diagonal sign rule satisfies `QR(XD) = (QD, DRD)`. So the completion is `(D⊗D)·completion·D_E`: signed copies, the same spans.
- **The new slots of later cores.** These include new left-bond slots created by earlier completions. They inherit signs inductively, so the outside-coordinate argument repeats at every core. The down cores follow the same pattern.
- **Carriers.** They are embedded old carriers and transform with the old gauge.

P17 (1.4 against 0.0) and T2(f) are the right evidence and the right test.

**The degenerate case.** This is a coordinate seed inside the old columns' span when the in-block room exceeds the count.

- **Not structural.** Householder completes such a seed from noise, so the result is backend-dependent and not covariant.
- **No structural fix exists.** Pivoting the seeds by residual norm would be covariant in exact arithmetic. But it decides by value, and it is fragile exactly at the ties that coordinate-aligned cores create.
- **The probe says the case is rare.**
  - On a generic minimal point, its T3-SVD, and a retraction-like T3-SVD of a perturbed point, the first in-block seeds at every core with room have residuals of 0.05–0.8, never near zero.
  - Core 0 has no in-block room at all for a minimal `x_old`: `nU_0 ≤ r_1` forces `sL_1 = s_0`.
  - The cores L_{d−1} and R_0 are carriers and are never completed.
- **Verdict on the case.** Accepting it (R5) is right. The wording needs fixing (MINOR C).

**2. The `padded_from=x_old` flow on both layers.**

- **The old frame.** The ragged layer uses `t3_orthogonal_representations(x_old)`. The uniform layer uses `ut3_orthogonal_representations(x_old.data)`, with `x_old`'s own masks, never `self.frame`. Gappy masks are fine (T5(g)).
- **The guards.** Layer, shape, stack, and stores ≤ `x0`'s. The frontend reductions of `x0` (in `least_squares_problem`, `optimizers.py:302-314`) and of `x_old` keep `old ≤ target`, by monotonicity (3000/3000 earlier).
- **TIED.** It moves to `x_old`, and the group's old block is read from `x_old`'s frame.
- **jit.** Under `use_jit` the same conversion as `x0` (`partition_static`) applies.
- **The same-tensor check.** It is `allclose`, which takes the difference's norm through orthogonalization, at `effective_rtol`.
  - Measured earlier: 2.6e-15 and 2.7e-14 in float64 against rtol 1e-9; 2.1e-7 in float32 against the jax rtol of 1e-5.
  - Cost: three orthogonalized norms at the summed ranks, in safe mode only. Worth one docstring line.
- **One unnecessary requirement:** the backend's equal-`N` check (n2).

**3. The plain-keyword API and its mirror and forwarding tests.**

- **Names, defaults, validation.** All consistent:
  - `None` → `PADDED_START_PASSES` / `CONE_REFERENCE_PASSES` / the geometry's seed;
  - `passes ≥ 1`, `0 ≤ seed < 2**32`;
  - "without `padded_from`" and "without `'cone'`" are structural errors.
- **The mirror test** (`test_optimizers_frontend.py:448`) compares names and defaults, so the six keywords pass automatically (both default to `None`).
- **The forwarding test** (`:465`) asserts the backend kwargs dict *exactly*. That is why "forward only when set" is needed, and it is correct.
- **The new forwarding test** (T9(b)) covers the set case.
- **The gap is the keywords' position** (MINOR A).

**4. G-nonbreaking.**

- **Sound and cheap.** It runs main's `tests/test_*.py`, unmodified, against the branch. The one stated exception is T12, the vocabulary list.
- **Scope.** It omits main's `tests/backend/test_*.py`: 16 files covering the documented backend API (n3).

**5. The contract of `ranks.oversampled_ranks`.**

All of it is correct:

- **The rule.** The elementwise maximum of the input and `minimal(r + p, (1, ρ + p, …, 1), sharing)`.
- **"Minimal for a minimal input".** Monotonicity gives `minimal(r + p) ≥ minimal(r) = r`.
- **"Never below the input."** Holds by construction.
- **"`p = 0` returns the input".** Holds for any input, because `minimal(r) ≤ r`.
- **Groups and boundaries.** Groups keep one rank (`r + p` stays equal within a group, which `compute_minimal_ranks` requires); boundary bonds stay 1.
- **The working-ranks guard.** Stores ≥ target.

The one inaccuracy is an example: the frontend's `x0.oversampled_ranks(5)` exists only on the ragged class (n4).

**6. The uniform member's explicit conversion against the equivalence contract.**

The contract — `to_uniform → op → to_ragged == op_ragged` on real blocks, with garbage padding never read — is met through the member:

- T4(a): real blocks within 1e-14 on identical inputs, including `utv_to_ut3`'s gappy bond masks;
- T4(b): masks are the template's, padding is bitwise zero;
- T4(c): NaN padding;
- T4(d): per-element ranks;
- T4(e): the native old frame is gauge-equivalent.

Its "gauge-carrying operations" clause is met by T5(f) and T9(e). Two small gaps:

- the contract's "non-minimal input too" obligation has no T4 row (n5);
- three doc sentences call the uniform *padded* frame "native" (n1).

## Propagation check (rounds 5–6)

| change | API | algorithm | §3.8 | tests | docs / CHANGELOG | slices | decisions / ledger |
|---|---|---|---|---|---|---|---|
| 5: hint is a T3; `tangent_to_t3`; `K` lift gone; tail squash; `+X_old` exactness | ✓ §5.2, §5.4 | ✓ §4.4 | ✓ (hint row, caveat row) | ✓ T3(j), T4(a) | ✓ CHANGELOG backend bullet, `frame_variations` | ✓ S3, S5 | ✓ |
| 5: math note | — | — | — | ✓ `test_math_note_references.py` | ✓ §7 outline | ✓ S12 | ✓ I-36 |
| 6.1 bitwise vs tolerance | — | — | — | ✓ §6 preamble, every row | ✓ | — | ✓ I-38 |
| 6.2 revert I-28; coordinate seeds | ✓ | ✓ §2.5, §4.3 | ✓ | ✓ T2(f), T4(e), T5(f) | ✓; wording: MINOR C, n1 | ✓ S2 | ✓ I-11, I-28, ledger |
| 6.3 old point | ✓ §5.4–§5.7 | ✓ §3.4 | ✓ | ✓ T5(g), T7(b)/(g), T9(c)/(d), T13 | ✓ | ✓ S5, S7, S9 | ✓ I-20, I-30, I-34 removed, I-37, ledger |
| 6.4 acceptance | — | — | — | — | ✓ verification records | ✓ S11, "after" | ✓ I-39 |
| 6.5 `oversampled_ranks` | ✓ `ranks`, frontend, member | ✓ §3.5 | ✓ | ✓ T5(h) | ✓ CHANGELOG | ✓ S1, S5, S9 | ✓ I-29 |
| 6.6 no vectorization | — | ✓ §3.9 | — | — | ✓ ledger, internals | — | ✓ O-5 → Rejected |
| 6.7 no twin | ✓ member + recipe | ✓ §3.3, §4.8 | ✓ | ✓ T4, T10 | ✓ naming "Intended asymmetries", ledger | ✓ S4 | ✓ I-41 |
| decisions: I-3, I-32, O-1, O-9 | ✓ (MINOR A) | — | — | ✓ T7(c), T8(f), G-nonbreaking | ✓ (MINOR B) | ✓ | ✓ |

The leftover search is clean:

- `PaddedFrameOptions`, `start_options`, `cone_options`, `old_ranks`, `padded_frames`, `I-34` and `stats['start_frame']` survive only in the CHANGES history or as "removed" entries.
- `ufv_padded_frame` appears only in the "no twin" statements.
- The fv_operations placement is acyclic with the two new edges: `probe_imports_r6.py` finds no cycle, and the backend has no function-level library import.

## Findings

### MINOR A — The six `newton_cg` keywords are inserted mid-signature, which breaks positional calls

**Where.** §5.5 :1417 ("after `g0norm_cg`"); §5.7 (the frontend mirror); I-40.

**What.**
- **Neither function is keyword-only.** Backend `newton_cg` has 17 positional-or-keyword parameters and no `*`. `cg_forcing_power`, `use_jit` and `callback` follow `g0norm_cg`.
- **The frontend is the same.** It has 27, and `cg_forcing_power … tr_radius0` (10 of them) follow `g0norm_cg`.
- **The effect.** Inserting six parameters after `g0norm_cg` shifts every later position. A positional call that reaches them now binds its values to `start_frame`, `padded_from`, and so on.
- **The contradiction.** That is unlikely in practice, but it contradicts I-40's unconditional "No breaking changes". G-nonbreaking cannot see it, because no existing test calls positionally.
- **Elsewhere it is fine.** `gradient_descent` gains its two at the true end (after `g0_reference`, its last parameter), and `local_model(frame=)` is appended.

**Change.** Append the six as **keyword-only** at the end of both `newton_cg` signatures: `…, callback=None, *, start_frame=None, …`. The mirror test compares names and defaults, so it is unaffected. This also stops positional use of option keywords in future.

If Nick prefers the topical "reference block" grouping, he must accept, in the CHANGELOG, that positional calls past `g0norm_cg` change meaning. Either way it is his call, but it must be made explicitly.

### MINOR B — The `1e-4` advice brings back m7's conflation in shipped text

**Where.**
- §3.7 :901-902 ("the pre-merge `1e-3` regime");
- `fitting_and_optimization.md` §5 text :1588-1589 and the release notes :1888-1889 ("about 3× tighter than the **pre-2026.3.0** `1e-3` regime");
- O-9 :2060.

**What.**
- **The numbers are right but describe a downstream quantity.** "≈ 3e-4 of the old coordinate-aligned reference" uses the 2.3–5.4× ratio of §0's table. That reference is T3Polynomial's private `_zero_gradient_norm`.
- **The library had a different stop.** Before 2026.3.0 the library's relative stop was against the **warm start** `‖g(x0)‖` (CHANGELOG `[2026.3.0]`; round 1's m7). The plan's own §0 :153-154 still says so: "not the library's pre-2026.3.0 warm-start reference".
- **So the shipped sentences are wrong.** They tell library users their old `1e-3` regime was something it was not, and "the old coordinate-aligned reference" names something they never had.

**Change.** Keep the recommendation and drop the comparison from user-facing text. For example: "With `'cone'`, use `gtol_rel=1e-4`; the cost is in the verification records." The coordinate-aligned comparison can stay in §3.7 and O-9 if labelled as T3Polynomial's reference.

### MINOR C — The exceptions to cross-layer agreement are stated inconsistently, and "replaces" overstates

**Where.** §4.3 :1037-1039 and R5 :2041 ("…which the first hinted pass replaces"); §3.3 :690-692; R10 :2046; the `uniform_equivalence_contract.md` paragraph :1660-1667 and the fitting §4.11 paragraph :1568-1572.

**What.**
- **"Replaces" is wrong.** The first hinted pass uses the gradient *at* the hint-free start, so the start influences every pass. The iteration damps it (P13/P14); it does not remove it at four passes. So a degenerate coordinate seed can make the two layers' padded starts differ, not just their hint-free frames.
- **The exception lists disagree.**
  - §3.3 lists three exceptions: near-tie, degenerate hint, rotation in a tied cluster.
  - R10 lists the same three.
  - The shipped contract-doc and §4.11 paragraphs list only the near-tie.
  - None of them lists the degenerate coordinate seed, which is in R5 only.

**Change.**
- Write one canonical list and use it verbatim in §3.3, R10, the two shipped paragraphs and the internals page:
  - a near-tie at a cut;
  - a degenerate hint;
  - old frames that differ by more than signs (a tied or numerically deficient `x_old`);
  - an in-block coordinate seed in the old columns' span.
- Replace "which the first hinted pass replaces" with "whose effect the hinted passes damp but, at the default four, do not remove".
- Optionally cite this round's probe for the rarity: residuals 0.05–0.8 on library-produced frames; no in-block room at core 0 for a minimal `x_old`.

### NITs

- **n1** — Three sentences say the padded frame is built "natively" on each layer:
  - fitting §4.11 :1569;
  - `rank_continuation.md` :1619;
  - the contract doc :1660-1661.

  On the uniform layer only the *old* frame is native; the construction runs on unpacked real blocks (§3.3). Say "each layer builds its old frame natively".
- **n2** — The backend `padded_structures` requires equal padded `N` (§3.4 :733). Nothing in the backend needs it:
  - the old frame unpacks through masks;
  - the repack uses the template's sizes.

  Only the frontend `allclose` needs it, so keep the check there and drop it from the backend, in the spirit of the maintainer's no-unneeded-requirements rule. Related, pre-existing: `UniformTuckerTensorTrain.allclose`'s docstring says "padded widths need not match", but `probe_allclose_N.py` shows it raises on a different `N` (it works on different `n`/`r`). Fix the docstring or re-pad while here.
- **n3** — G-nonbreaking should also run main's `tests/backend/test_*.py` (16 files) unmodified. The backend is documented public API, and none of those tests pins anything the plan changes: the `g0_reference` message test matches only `'g0_reference must be'`, and no test pins an `__all__`.
- **n4** — §5.7 :1460: "`working_ranks` … e.g. `x0.oversampled_ranks(5)`" works only for a ragged `x0`; `UniformTuckerTensorTrain` has no such method. Say "ragged: `x0.oversampled_ranks(5)`; uniform: `ranks.oversampled_ranks(...)`", or add the uniform method.
- **n5** — T4(a): add a non-minimal `x_old` (or template) row. The contract demands agreement "on non-minimal-rank input too", and the backend now accepts a non-minimal `x_old`.
- **n6** — The conditional forwarding in the frontend `newton_cg` needs a one-line code comment ("forwarded only when set: G-nonbreaking"), so a later cleanup does not "simplify" it into always-forwarding and break the old forwarding test.
- **n7** — Add a caveat row to §3.8 for the in-block coordinate seed case (decides directions, never validity, backend-dependent), parallel to the hint caveat row.

## Checked and found correct (round 4)

- **Sign-gauge covariance of the hint-free start**, slot by slot (focus 1):
  - Tucker invariant;
  - coordinate completions map to signed copies;
  - Householder QR with positive-diagonal R is column-sign covariant;
  - new slots inherit signs inductively;
  - carriers follow the old gauge;
  - the hinted steps read projectors and the hint as a tensor;
  - P17's covariance plus 4-pass invariance (gradient norms equal to 4e-15).
- **Rarity of the degenerate seed case** on library-produced frames (`probe_seed_degeneracy2.py`).
- **The `padded_from=x_old` flow** on both layers:
  - the native old frame from `x_old`'s own masks;
  - the stores guard plus monotone reductions;
  - TIED on `x_old`;
  - the jit conversion;
  - the same-tensor check's route, tolerance and precision.
- **Placement.** `fv_padded_frame` in `fv_operations` with downward edges only (`t3_orthogonalization`, `tt_orthogonalization`); `geometry` gains four acyclic edges. The function-level-import gate holds today.
- **`oversampled_ranks`.** Its whole contract, including sharing, boundary bonds, `p = 0`, minimality, and the stores relation `old ≤ target ≤ working`.
- **Keywords.** The plain-keyword validation and exclusions; the mirror test's name/default comparison; why "forward only when set" is required.
- **The uniform member's conversion** against the equivalence contract (T4(a)–(e), T5(f), T9(e)), and the razor argument (the member reads only `self.groups`; the recipe in the docstring).
- **Default-path claims.** G-bit, T6(a) and T7(a) as bitwise default-path claims; T7(g) with `padded_from=x0` reducing to the library frame on each layer.
- **The ledger.** Gaussian seeds, the ragged round trip, old ranks, the options object, the twin and masked vectorization rejected; `ufv_zero_point_frame` deferred.
- **S11–S13 and the acceptance.** The laptop gate, the Runpod pair after the release, each measurement tied to a decision.
- **Still in force.** The round-1 to round-3 "checked" lists.

VERDICT: REVISE
