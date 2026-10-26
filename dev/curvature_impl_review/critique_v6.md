# Critique of `plan_v6.md` (round 6)

Critic review of v6 against `critique_v5.md`. Repos `/home/claude/t3toolbox` @ `920d776` and `/home/claude/t3polynomial`
@ `388090d`, both read-only; all runs used `python -I -B`, and no file under either repo changed (`find -newer
plan_v5.md` on the repo is empty). "SCR" means the `plan_review` scratch directory.

**Bottom line.** v6 fixes the BLOCKER, the three MAJORs and all eight MINORs from round 5, and applies the four NITs.

- I re-ran the four new checks (`c40`, `c41`, `c42`, `c31c`); all reproduce exactly.
- I independently re-derived the M6 recipe's two properties.
- I diffed the plan body against v5: all 209 changed lines belong to these fixes, and nothing unrelated moved.
- No invented names.
- Two cosmetic NITs remain, to note at implementation.
- Every open item is either a measurement (O-1…O-5) or explicitly Nick's call (O-6, the whole zero-frame mechanism,
  with its alternatives).

**VERDICT: AGREE.**

---

## 0. Verification of the round-5 findings

| # | v6 fix | verified |
|---|---|---|
| **B1** field/method clash | The mode is **derived**: a read-only `is_exact` property returns `curvature is not None`, on both the frontend and the backend `LocalModel`. `hessian(p)` stays a method; the factories keep the `hessian=` *argument*, which only decides whether a `curvature` child is built; there is no mode field or mode aux (C.9, C.13). T18 asserts "no `hessian` attribute other than the method"; T28 asserts `model.is_exact` | ✓ The clash is gone by construction. Naming follows the frontend checker rule (`naming_conventions.md:133-137`: a pure derived property); the `SharedGeometry.is_uniform` precedent exists (`shared_geometry.py:144`). Pytree: the presence of the child is static under the treedef, so compile-once is unaffected. Recorded as I-26 within I-4 |
| **M5** oracle letters | S1 extends `AXIS_SIZE` with `{l: 4, n: 6, x: 7, z: 5}` | ✓ `c40` imports the repo's own `tests/test_contractions_interpreter.py`: 9 strings would raise `KeyError` today; with the extension, 90/90 (45 + d-prefixed twins) pass `build_operands` → `loop_oracle`, the inference path and the supplement contract; the existing 123 HISTORICAL rows still pass. `c40` mutates `AXIS_SIZE` only in memory. Respelling was rejected for a reason (the letters carry meaning in the kernels' locals): acceptable |
| **M6** zero-frame recipe | Per group: `RandomState([seed, k_g])`, an `n_g × N_g` draw, sign-fixed `qr(G.T)`, `U_g = Qᵀ`. T26(d) gains independence (bitwise), nesting (to 1e-15) and a pinned small case (`allclose` 1e-12, not bitwise across BLAS). O-6 now puts the whole mechanism to Nick with three alternatives | ✓ `c41`: another mode's rank **and** size change leaves a factor bit-identical; nesting error 0.0 over r = 1…7; the reference is non-decreasing on four ladders; 4.7e1 on trailing-supported data (no collapse); 44 O2 pairs at **3.5e-12**, references 3.4e-16. My own re-derivation: nesting 0.0 at N = 6 and 1.1e-16 at N = 9 (n22); `RandomState([seed, k])` row-major draws give each group its own stream, independent of the other modes |
| **M7** T25 scope | T25 asserts only the frame claim: the first frame is the slack `t3_orthogonal_representations` frame; exact+TR from slack converges; slack and minimal starts give the same outer counts and final ratio. The GN contrast moves to the records as E11 | ✓ `c42`: first frames are the slack ones (down `[2,4,2]`; left `[1,4,9,1]` vs right `[1,4,4,1]`); slack vs minimal gives 20/20 and 34/34 outer steps, to 2.6e-9 / 2.2e-12 identical; one retraction already moves the ranks. Matches my E11 |
| m31 | bands: exact/GN ≈ 2.4–3.0× (±15 %), factored assembly 1.4–2.0× | ✓ (C.4, E, F33, X8); consistent with both runs |
| m32 | computed iff the resolved reference is `'zero'` and `g0norm_newton is None`; `NewtonInfo.g0_reference` (Newton stop) + `cg_reference`; T26(f) with a counter on `zero_point_frame` and a `g0norm_cg`-only row | ✓ (C.15 rules 2 and 4) |
| m33 | a vanishing **default** reference falls back to `'warm_start'` with a `RuntimeWarning` and records it; a stationary x0 still stops at iteration 0; only an explicit `'zero'` raises; T26(g) covers both | ✓ (C.15 rule 5). This matches the house line (numerical conditions do not hard-fail a default path) and preserves today's all-zero-data convergence |
| m34 | `zero_point_frame` optional, duck-typed (`getattr`); absent means a warm-start default; an explicit `'zero'` raises, naming the member; the four exact-mode members are checked structurally only in exact mode; a custom duck-typed geometry is tested (T26(a)); release notes updated | ✓ (C.7, C.9, C.15, G). Required for kinds (abc marker) but duck-typed for geometries is a coherent split: both keep GN-only extensions working |
| m35 | `test_optimizers.py:238` in S0d | ✓ |
| m36 | §L.3 states there is no cycle hazard; the marker's two real reasons | ✓ |
| m37 / n20 | `docs/fitting_and_optimization.md:236-240` cited; qualified class names and `:py:class:` targets everywhere both classes appear | ✓ |
| m38 | T18: a shared model's `retract` == the backend's (with `geometry_aux`) | ✓ |
| n18 | marker vs identity on a re-declared stub, stated | ✓ |
| n19 | T16 asserts only on steps with `|pred| > 1e3·res`, `C = 2`, ≥ 2 points per seed | ✓ `c31c`: the criterion keeps exactly the steps with rate ≤ 0.98 (seed 1: 0.98, 0.78; seed 2: 0.95, 0.43, 0.50) and excludes every floor step (seed 1's 1.97 at `|pred|/res` 2.9e-7; seed 2's at 2.9e2 and the rejection). Margin 2.05×; re-pinned in S5 |
| n21 | the converged-line CHANGELOG entry worded as display-only | ✓ |
| O-7 | closed: one retraction | ✓ |

## 1. Remaining NITs (implementation-time; not blocking)

- **n22 — "exactly the prefix".** C.2 and F37 say level r's factor is "exactly" the prefix of level r+1's ("error
  0.0"). That is mathematically true, but numerically it holds to rounding: I measured 1.1e-16 at N = 9, since
  LAPACK's blocked QR rounding can depend on the column count. T26(d) already asserts 1e-15, which is the right
  tolerance; just say "to rounding" in the prose.
- **n23 — `is_exact` on the backend class.** The backend `LocalModel.is_exact` is a class property, not a backend free
  function, so the `<family>_<subject>_<predicate>` grammar does not apply. It is a mode bool, not a per-stack-element
  numerical checker; the `is_uniform` precedent covers exactly that. Fine as is; a one-line note in
  `naming_conventions.md` §"Checkers" would pre-empt the question.

## 2. Sections 1–15

Unchanged from round 5 except as above.

| section | status |
|---|---|
| Razor and principles (§1) | OK; §L accepted |
| Ragged/uniform (§2) | OK: S0c hygiene, θ̂ masking, the marker check, the keyed nested zero frame |
| Safe mode (§3) | OK: every new error is structural; the vanishing default falls back |
| Batching (§4) | OK; strings verified through the repo oracle |
| Naming (§5) | OK: B1 resolved; n23 |
| Signature style (§6) | OK |
| Dispatch (§7) | OK |
| Kernels and costs (§8) | OK: bands |
| Protocols (§9) | OK: kinds required (marker), geometries duck-typed |
| Shared (§10) | OK |
| g0 interplay (§11) | OK: skip condition and records exact |
| Tests (§12) | OK: T16, T23, T25, T26 and T28 well scoped |
| Docs, routing (§13) | OK |
| Math (§14) | OK |
| Other (§15) | OK |

## Open for Nick (from §I)

- **O-6:** the canonical zero-frame mechanism behind the default reference. Recommended: the keyed nested Haar recipe.
  Alternatives: a coordinate-aligned completion (collapses on aligned data), a warm-start default with `'zero'` opt-in,
  or per-mode keys.
- **O-1…O-5:** the S0a/S0b/S3/S5 measurements (damping value, TR constants, test constants, cost and memory gates, the
  jax float32 floor bound).

## What must change for AGREE

Nothing. n22 and n23 are prose notes for implementation.

VERDICT: AGREE
