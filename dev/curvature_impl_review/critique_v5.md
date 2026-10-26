# Critique of `plan_v5.md` (round 5)

Critic review of v5 under the round-5 brief: a breaking release, the T3Polynomial update in scope, and the
conventions read as **principles**. Repos `/home/claude/t3toolbox` @ `920d776` and `/home/claude/t3polynomial` @
`388090d`, both read-only; all runs used `python -I -B`. "SCR" means the `plan_review` scratch directory.

**Bottom line.** v5 is a coherent first-principles restructure.

- All 16 requested items are present (§0.1).
- All six of my pre-checks are addressed with evidence, and P1c is fixed by construction rather than by pinning the
  tests. I confirmed this: 44/44 O2 pairs, references layer-independent to 3.9e-16, worst loss pair 2.6e-9.
- The straightforward routes are justified by the principles' intent (§L), and I accept those arguments.
- I re-ran `c29`, `c30`, `c31`, `c33`, `c33b`, `c34`, `c35c`, `c35d`, `c36`, `c37b`, `c38` and `c39`; all reproduce,
  with the caveats below.
- No invented names: every repo function, file:line and doc anchor I checked exists.

**One BLOCKER (B1).** C.13 gives frontend `fitting.LocalModel` a dataclass field `hessian` and a method `hessian(p)`.
Python cannot have both. I verified it: `M(hessian='exact').hessian(1)` raises `TypeError: 'str' object is not
callable`, and `M().hessian` returns the function, not `'gauss_newton'`. The design does not work as written, and T18 /
T28 call `hessian(p)`. The fix is a rename.

**Three MAJORs:**

| # | finding |
|---|---|
| M5 | the HISTORICAL rows for the new non-jet strings would crash the repo's contraction oracle as specified |
| M6 | the zero-frame draw recipe in the plan text is not the one that was verified, and it loses cross-level consistency |
| M7 | T25's GN contrast is attributed to the slack frame, but it is a property of the problem |

There are also seven MINORs and four NITs. The verdict is REVISE, with a short list.

---

## 0. Verification

### 0.1 The 16 requested items

| # | item | v5 | status |
|---|---|---|---|
| 1 | frontend rename `GaussNewtonModel` → `LocalModel`, a `hessian` mode field, generic `hessian(p)`/`quadratic(p)` plus `gn_*`/`curvature_*` | C.13, S0e | **present, but the field/method clash is B1** |
| 2 | `curvature_damping` as a solver parameter (newton_cg → local_model → curvature_precompute); None → the geometry default; no field; raises where it cannot apply; ε a leaf | C.7, C.9, C.13, §L.1 | ✓ |
| 3 | `cg_forcing_power = 0.5` everywhere | C.9, I-7, F29 | ✓ (n19 on T16) |
| 4 | G0P folded: `g0_reference`, the overrides' interplay, the forcing term | C.15 | ✓ (m32, m33) |
| 5 | public backend-problem constructor; `_setup` / `_geometry_ops` not load-bearing | C.16 | ✓ |
| 6 | `corewise_hessian` required, no flag or shadow; early check; `_UniformKind` raising override; T23 mutation | C.6, T23 | ✓ (§2.2) |
| 7 | uniform output-hygiene slice before the curvature; θ̂ masking kept | C.0 (S0c), F34 | ✓ |
| 8 | production kernels in house form; `_trs` only as test references; costs re-measured | C.4, E, F33 | ✓ (m31) |
| 9 | `contract()` strings for every new W/K/C kernel | D, F35 | ✓ (M5) |
| 10 | registrations under `if jax_available:`; jax-absent construction; no string annotations; explicit DAG | §M, T28, F35 | ✓ |
| 11 | `newton_steps` / `newton_accepted` / `stop_reason`; one `step` display segment | C.9, C.14, T29 | ✓ (m35) |
| 12 | explicit `step_control` / `tr_radius0` | C.9, C.13 | ✓ |
| 13 | downstream T3Polynomial section with file:line edits | §K | ✓; every line checked (§0.3) |
| 14 | sector norms deferred, with a ledger paragraph | C.14, G | ✓ |
| 15 | full `newton_cg` end-to-end on Tucker- and TT-slack frames | T25, F28 | ✓ (M7) |
| 16 | d = 1 reasoning (flat slots) | B | ✓ |

### 0.2 Re-runs

| check | plan claim | my run |
|---|---|---|
| `c37b` O2 pairs at the Haar zero reference | 44 pairs; refs 3.9e-16; losses ≤ 2.6e-9 | ✓ identical |
| `c33` / `c33b` / `c36` canonical frame validity, layer independence, shared | orth 0, base point 0, Π idempotent; ragged == packed uniform | ✓ (1.3e-16, 1.8e-16, 0) |
| `c38` collapse | LAPACK and identity-prefix 0.0 on trailing-supported data; Haar 9.5–24 | ✓ |
| `c34` strings + DAG | 25 strings == einsum; acyclic; `optimizers.py` imports `[common, regularization]` | ✓ |
| `c39` S0c | padded output up to 3.0e12; bit-identical; 0.06–0.51 % | ✓ |
| `c29` g0 interplay | zero ref 7.01e4, warm 2.07e4; COREWISE 0.0; `g0norm_newton=0.0` → ZeroDivisionError | ✓ every row |
| `c30` slack end-to-end | Tucker slack 20/78; TT slack 34/272 converged; GN 80 not converged | ✓ (M7 on the interpretation) |
| `c31` forcing power | power 1.0 quadratic; 0.5 order ≈ 1.5 | ✓ (n19) |
| `c35c` factored assembly | 2.0× faster; 3e-15 | 1.37× (small) / **1.70×** (large); 1.3e-15 / 3.0e-15 (m31) |
| `c35d` standard-form cost | 2.73× / 2.98× | **2.39×** / 2.89× (noise; m31) |

**My new checks (SCR):**

- **`check_tt_slack_gn_stall.py` (M7).** GN from the TT-slack start and from the *same tensor reduced to minimal ranks*
  produces an identical stall: 80 steps, unconverged, ratio 0.85, the same f. The stall belongs to the problem's GN
  rate, not to the slack frame.
- **`check_zero_frame_nesting.py` (M6).** I compared the two Haar recipes across a continuation ladder. Only the
  `N×N`-draw recipe that `c38` actually ran keeps one mode's factor unchanged when another mode's rank changes; the
  plan text's `N×n` sequential recipe does not.
- **Field/method clash (B1).** An inline repro, quoted above.
- **Contraction oracle (M5).** `tests/test_contractions_interpreter.py`'s `build_operands` raises `KeyError` on
  letters `x`, `l`, `z` (its `AXIS_SIZE`, `:49-50`, lacks `c`'s neighbours `l`, `n`, `x`, `z`).
- **abc marker.** It works on the house's plain-`type` dataclass hierarchy: `SamplingKind`'s metaclass is `type`; a
  subclass without the method instantiates; the inherited function carries `__isabstractmethod__` (§2.2).
- **`newton_cg` on all-zero data** (m33). Today it converges (12 steps, f = 1.3e-14) from a nonzero start. Under
  v5's C.15 rule 5 it raises.

### 0.3 Downstream (§K), checked line by line

Every §K citation is accurate at `388090d`:

- `t3polynomial/fitting.py:86-112`, `:158-164`, `:183-197`, `:316-345`, `:353`, `:364`, `:476-501`, `:513`, `:525`;
- `tests/test_fitting_selection.py:55-91`;
- `scripts/d07_curvature_dispersion.py:63-65`, `scripts/d08_profile_ops.py:113-114`, `scripts/d05_darcy_fit_assess.py:232`
  / `:237` / `:272`, `scripts/b01_backend_bench.py:275-278`;
- `scripts/x06_hessian_fd_probe.py:59-61`, `scripts/x07_hessian_free_newton.py:102-106`;
- `dev/t3toolbox-upstream-notes.md:695` / `:718` / `:788`.

The P3 collision (the `g0norm_newton=None` → warm-start idiom) is handled at all three sites and in the test. The
downstream `newton_kwargs` keys are `{chunk_size, callback}` only, so removing `**kwargs` breaks nothing downstream.

---

## 1. Razor, parameters, the principles route (§L)

- **OK, accepted.**
  - **§L.1:** the damping is an *option* of the solve, not identity of the set; it reaches compute only as the
    `LiftFactors.eps` leaf, so the aux/cache-key concern behind "parameters are fields" does not arise.
  - **§L.2:** promoting `_setup` to `optimizers.least_squares_problem` reduces unpromised surface.
  - **§L.3:** a required operation puts the contract in one place.
  - **§L.6:** no alias, per the 2026.2.0 precedent.
- **OK.** `backend/optimizers.py` stays `{common, regularization}` (+ the top-level `corewise`); `c34` confirms.
- **m36 — MINOR: the cited cycle hazard does not exist.** §L.3 says the identity check "needs `backend/optimizers.py` to
  import `backend.fitting`, its first T3-specific import". True, but there is no cycle hazard: `backend/fitting.py`
  imports only `probing`, `sampling_derivatives`, `apply`, `entries` and `common`; `uniform_fitting` imports
  `optimizers`, not the reverse. The marker route is still the better choice (§2.2), but for its own reasons.
  - **Fix:** state the reasons accurately: the invariant, plus rejecting the window stub early.

## 2. Ragged and uniform

### 2.1 S0c output hygiene

- **OK.** Measured bit-identical with negligible overhead.
- The plan keeps the `utv_*_transpose_from_sweep` seams bare by contract. Their only callers are the kinds, which now
  re-mask. Good.
- T27 freezes the pre-S0c real block, which is the right guard.

### 2.2 The abstractmethod-vs-identity check (as requested)

**Coherent.**

- `SamplingKind` is a plain-`type` frozen dataclass (`ValueHashedFields` root, no `ABCMeta`), so `@abc.abstractmethod`
  only sets `__isabstractmethod__`. Instantiation is unaffected (verified).
- The marker check is a strict superset of the identity check: it also rejects the `_UniformKind` stub, which is
  correct during the window.
- T23 asserts both forms agree on every built-in ragged kind and on a five-operation subclass.
- **n18 — NIT:** state that a third-party kind which *re-declares* a stub with the marker is rejected by design. That
  is the only case where the two forms differ outside the window.

### 2.3 The canonical zero-point frame (as requested)

- **Determinism across platforms and BLAS.** The draw is `np.random.RandomState(seed)`, whose legacy stream is frozen
  (NEP 19), so the Gaussian matrix is bit-identical across numpy 1.22↔2.x.
- The QR is LAPACK's, so Q is reproducible only up to the factorization's floating-point rounding (≈1e-15 relative)
  across BLAS builds, and the sign fix removes the sign ambiguity. That makes the reference reproducible to ~1e-15
  across machines: good enough for a relative stop, and better than today's LAPACK completion, which is also
  BLAS-dependent.
- T26(b) asserts ragged == uniform at 1e-14, which is the right tolerance; no test should assert a bit-exact reference
  across machines.
- **Sharing.** One draw per group, replicated over its modes, is tied by construction (`c33b`: 1.8e-16).
- **The uniform layer.** It unpacks the real blocks, uses the ragged construction and packs back, so the layers agree
  by construction (`c36`: 0).
- **As a library default.** A seeded-random convention is acceptable, because the alternatives are worse:
  - a coordinate-aligned completion collapses to 0 on aligned data (F32);
  - per-layer LAPACK breaks equivalence (P1c).
  But it is a genuine default-semantics decision with an O(1) effect on every manifold `newton_cg` stop, so it belongs
  to Nick. v5 lists it only as O-6 "the seed". It should list the **mechanism** (Haar draw vs any other generic
  deterministic completion) as the decision, with the evidence.

- **M6 — MAJOR: the draw recipe in the plan text is not the verified one.**
  - **The plan's text.** C.2: "Each up factor is the sign-fixed QR of an `N_i × n_i` standard-normal draw … Draws come
    in mode order, one per group … (`c38` drew `N × N` and kept `n` columns; both are Haar)".
  - **Why the difference matters.** Both are Haar *in distribution*, but the deterministic outputs differ, and only one
    is verified: `c37b` / `c38` / `c33` all run the `N×N` recipe.
  - **What the plan's recipe loses.** With an `N_i × n_i` draw from one sequential stream, the stream offset for
    mode i depends on the ranks of modes < i. So changing one mode's rank re-randomizes every later mode's factor
    (`check_zero_frame_nesting.py`: recipe A "mode-2 factor unchanged when only mode 0's rank changes: False"; recipe
    B: True).
    - Under recipe B the frames are also **nested** across levels: QR column j depends only on draw columns ≤ j. That
      is exactly the property a per-level reference in rank continuation wants: level r+1's factor extends level r's.
    - Both recipes happened to give a non-decreasing reference on my ladder, but only B gives it by construction
      (nested subspaces). A gives it by luck of the draw.
  - **Fix:**
    - Specify recipe B: per mode in a **fixed order independent of ranks**, draw `N_i × N_i` (or, for memory, the full
      `N_i` columns), sign-fixed QR, keep the first `n_i` rows of `Qᵀ`. Shared groups take the group's first mode's
      draw.
    - Better still, seed each mode's stream as `RandomState(seed + i)` (or a per-group key), so that no mode's factor
      depends on another mode's shape at all.
    - Re-run `c37b` under the exact specified recipe.
    - Add a T26 case: a mode's factor is unchanged when another mode's rank changes, and the level-r factor is a prefix
      of level r+1's.

### 2.4 θ̂ masking and window safety

- **OK.** θ̂ masking is kept as defence in depth (T17, T24). The window stub plus the marker prevent unmasked ragged
  kernels on uniform kinds (T23 mutation).

## 3. Safe mode

- **OK.** Every new error is structural, and the n16 no-effect rule is applied consistently:
  - `curvature_damping` with GN or on corewise;
  - `tr_radius0` with the line search;
  - `g0_reference='zero'` on corewise;
  - non-positive `g0norm_*`.
- **m33 — MINOR: rule 5 turns a working fit into an error.**
  - C.15 rule 5 raises `ValueError` when `zero_ref == 0`, i.e. the data gradient vanishes at the canonical frame. For
    all-zero data (`y = 0`, so `r0 = 0`) that always happens, and today such a fit converges (verified: 12 steps,
    f = 1.3e-14, from a nonzero start).
  - Raising is structural-looking but is really a numerical condition, which the house line does not enforce
    structurally.
  - **Fix:**
    - If `zero_ref == 0` *and* the warm-start gradient is also 0, x0 is stationary: stop immediately (`'gradient'`).
    - If `zero_ref == 0` but the warm-start gradient is not, fall back to `'warm_start'` with a `RuntimeWarning` that
      names the reason; `NewtonInfo.g0_reference` records it.
    - Reserve the error for an **explicit** `g0_reference='zero'` that resolves to 0.

## 4. Batching (W + K + C)

- **OK.** Grouped strings with `''` / `'C'` supplements (F35). θ̂ is K+C against C-only masks. C stays innermost.
- **M5 — MAJOR: the new rows would crash the repo's own oracle.**
  - The 12 non-jet strings use the lowercase letters `l`, `n`, `x`, `z`, absent from the oracle's `AXIS_SIZE`
    (`tests/test_contractions_interpreter.py:49-50`).
  - Adding them to `HISTORICAL` as specified makes `TestVocabularyOracle` / `TestSupplementContract` raise `KeyError`
    in `build_operands` (verified: `'Cux,KCxn->KCun'` → `KeyError 'x'`; `'KCij,Ciak,Cjal->KCkl'` → `'l'`;
    `'Czn,KCun->KCzu'` → `'z'`).
  - `c34` checked the strings with its own operand builder, not the repo's.
  - **Fix:** either
    - extend `AXIS_SIZE` with `l, n, x, z` (distinct sizes, so a transposition is caught), as an explicit S1 edit to
      the test file; or
    - respell the strings onto the existing alphabet (`a b d i j k o r s t u`, plus `c`).
    Either way, re-run `c34` through the repo's `build_operands` / `loop_oracle`, not a private builder.

## 5. Naming

- **B1 — BLOCKER: the frontend `LocalModel` cannot have both a field and a method named `hessian`.**
  - C.13 specifies "A field `hessian: str = 'gauss_newton'` (the mode, a static aux)" **and** the method
    "`hessian(p)` — the mode-dependent Hessian action".
  - In a Python class, a method of the same name replaces the dataclass default. So `LocalModel()` reads the function
    as the field default, and an instance built with `hessian='exact'` shadows the method with a string.
    Verified: `M(hessian='exact').hessian(1)` → `TypeError: 'str' object is not callable`.
  - The plan's frontend API, T18 ("`hessian(p)` / `quadratic(p)` mode-dependent") and T28 ("call its `hessian(p)`")
    are unimplementable as written.
  - The backend is unaffected: its mode is derived from `curvature is not None`, with no `hessian` field.
  - **Fix**, a Nick-level naming choice. Recommended: keep the method `hessian(p)`, mirroring the backend's
    `hvp` / `gn_hessian` family, and **name the mode field differently**. Options:
    - `hessian_kind`;
    - `mode`;
    - derive it, as the backend does: an `is_exact` property from `curvature is not None`, with the factory
      argument staying `hessian=`. This one needs no new field and mirrors the backend; it is my recommendation, by the
      plan's own "derive, don't store" rule (I-3).
    Update C.13, T18 and T28, and the `fitting.LocalModel` pytree aux (no mode field to carry, if derived).
- **OK.**
  - `curvature_damping`;
  - `g0_reference` with `'zero' | 'warm_start'`;
  - `optimizers.least_squares_problem` as a same-named frontend/backend pair (`naming_conventions.md`);
  - `fitting.LocalModel` beside the backend `LocalModel` (the root does not re-export the backend one);
  - `newton_steps` / `newton_accepted` / `stop_reason`.
- **n20 — NIT: docs cross-reference.** `docs/fitting_and_optimization.md:236-240` already says "`LocalModel` (backend)
  … and on `GaussNewtonModel`". After the rename, two classes share the bare name in prose. G should require qualified
  references (`backend.optimizers.LocalModel` / `fitting.LocalModel`) in every doc that mentions both, and Sphinx
  `:py:class:` targets that are fully qualified, since autoapi documents both.

## 6. Signature style

- **OK.** Explicit keywords replace `**kwargs`. The signatures are expanded at implementation (n5).

## 7. Dispatch, jit, compile-once

- **OK.**
  - The zero reference is host-side, eager, and once per call (`zero_point_frame` unpacks to ragged); it is never
    traced, so compile-once is unaffected.
  - `curvature_damping` reaches only the ε leaf, so no recompile (the dispatch test covers it).
  - Registrations under `if jax_available:` follow the `sharing.py:1081` pattern; T28 constructs each class.
  - Python 3.9 floor (`pyproject.toml:16`): `abc.abstractmethod` and the `if jax_available:` registrations are fine;
    nothing new needs 3.10.

## 8. Kernels and costs (the re-based cost, as requested)

- **OK in substance.** "Each dotted kernel takes the form of the GN kernel it differentiates" is the right reading of
  the house rule (P5). The `_trs` forms live only as test twins.
- **m31 — MINOR: the cost figures are run-to-run noise and should be stated as a band.**
  - My re-runs: `c35d` 2.39× / 2.89× (plan 2.73× / 2.98×); `c35c` factored assembly 1.37× / 1.70× (plan 2.0×).
  - The equalities (3.7e-16, 2.6e-15, 3.0e-15) reproduce exactly. The timings vary by ~15 %, which is fine but should
    not be quoted as point values, and the "2.0×" headline is above both of my measurements.
  - **Fix:** E should say "exact/GN ≈ 2.4–3.0× (numpy CPU, two sizes, ±15 % run to run); factored assembly
    1.4–2.0× faster". The S3 gate (≤ 3.5×) stays as stated, and is comfortable.

## 9. Protocols

- **OK.** The geometry gains `zero_point_frame` (required; `None` opts out) and the four exact members. The release
  notes list them for custom geometries.
- **m34 — MINOR: decide whether `zero_point_frame` is truly required.**
  - Making it *required* means every custom geometry breaks at the next `newton_cg` call, even with GN, because the
    default `g0_reference=None` path calls it.
  - The plan says "REQUIRED (the g0_reference default path uses it)". In a breaking release that is defensible, but
    the cheaper equivalent is a duck-typed `getattr(geom, 'zero_point_frame', None)` that falls back to
    `'warm_start'`. That keeps five-member custom geometries working, in the same spirit as the GN-only kinds.
  - **Fix:** choose explicitly, state the choice in the release notes, and cover it with a custom-geometry test.

## 10. Shared factors

- **OK.** Per-group Haar is tied by construction. The shared O2 pairs hold (`c37b` includes `shared=True`).

## 11. Regularizer, ω, g0 (the interplay, as requested)

- **OK.** The resolution order in C.15 is coherent:
  1. `g0_reference` (None → the geometry's own);
  2. the zero reference computed only when needed;
  3. the overrides win, with the chained fallback (`g0norm_newton` sets both; `g0norm_cg` only CG);
  4. `NewtonInfo.g0_reference` records `'override'`.
- With a regularizer, `g_R(0) = 0`, so only the misfit enters (T26(g)). The `c29` interplay numbers reproduce.
- **m32 — MINOR: `'override'` is ambiguous when only `g0norm_cg` is given.**
  - With `g0norm_cg` alone, the Newton stop uses the resolved reference but `NewtonInfo.g0_reference` is "the resolved
    kind". Since `NewtonInfo.g0norm` is documented as the Newton-stop reference (`optimizers.py:614`), the display's
    "g₀ = zero tensor" is correct for the stop but silent about CG.
  - Also, `g0norm_newton` given together with `g0_reference='zero'` makes the zero computation wasted work. The plan
    says it is skipped "when the overrides do not already fix both references", which is correct only if `g0norm_cg`
    is also given or inherits.
  - **Fix:** spell out the skip condition exactly (`g0norm_newton is not None`, since CG inherits it). Record
    `g0_reference` for the Newton stop and add `cg_reference` (or document that CG follows Newton unless `g0norm_cg`).
    Add a T26(e) row for "`g0norm_cg` only".

## 12. Tests

- **M7 — MAJOR: T25's GN contrast is a property of the problem, not of the frame.**
  - T25 asserts "on TT slack GN+Armijo at the same budget does not converge (the documented contrast)", and F28 /
    CHANGES present it as an I-23 finding about non-minimal frames.
  - My `check_tt_slack_gn_stall.py` runs GN from the **same tensor reduced to minimal ranks**: an identical stall
    (80 steps, unconverged, ratio 0.85, identical f = 1.335520e3). The retraction drops the slack after one step
    anyway (TT ranks `(1,9,9,1)` → `(1,4,6,1)`, and Tucker slack `(4,4,4)` → `(4,4,2)`; verified).
  - So the contrast shows that this problem has a slow GN rate. It says nothing about non-minimal frames, and pinning
    it in a test that is *about* frames makes T25 fragile: it would fail if GN happened to converge, for unrelated
    reasons.
  - **Fix:**
    - T25 asserts what I-23 is about: exact+TR from a slack start converges, the frame ranks follow
      `t3_orthogonal_representations`, and the first step's frame is the slack one.
    - Move the GN contrast to the docs or `verification_records.md` as an observation about that problem.
    - Optionally assert that exact-from-slack equals exact-from-minimal to tolerance after the first retraction.
- **n19 — NIT: T16's power-0.5 assertion needs a tighter window.** At power 0.5 my `c31` run gives
  `e_{k+1}/e_k^{1.5}` up to 0.98 inside `[1e-6, 1e-1]`, but 1.97 at the floor (seed 1) and 1.5e3 (seed 2, at
  e = 1.5e-7). With `C = 2` and the window lower bound at 1e-6, seed 1's 0.98 has margin 2×, and seed 2's step leaves
  the window just below 1e-6. Pin the window from the run, with ≥ 2× margin at both ends, as v4's `|pred| > 1e3·res`
  rule did for ρ.
- **OK.** T26–T29 are well aimed. T26(b) at 1e-14 and 1e-7 are the right tolerances (§2.3). T28 constructs rather
  than imports (P4). T23's verbatim identity assertion is the right way to keep Nick's spelling visible.

## 13. Docs, CHANGELOG, routing

- **OK.**
  - A complete CHANGELOG `[2026.3.0]` with Changed — breaking / Added / Upgrading.
  - G0P and DP archived with banners.
  - `deferred_and_rejected.md` gets the sector-norm paragraph and the rejected flag.
  - `parameters_not_closures.md` gets the "options are not parameters" paragraph.
  - The OPEN_QUESTION note says "six operations".
- **m35 — MINOR: one stats test is missing from S0d's list.**
  - `tests/backend/test_optimizers.py:238` asserts `self.assertIn('newton', stats)  # backward compat`. That is the one
    test-side consumer of `stats['newton']`, and it is not in S0d's list (which names `:262` only). It will fail on
    the rename.
  - **Fix:** list it in S0d, changed to assert the absence of `'newton'` and the presence of the three new keys (T29
    covers the contract; this is the existing site).
- **m37 — MINOR: the prose citation for the docs check.** G's doc list should cite `docs/fitting_and_optimization.md:236-240`
  explicitly, as the prose that becomes ambiguous after the rename (n20).

## 14. Math

- **OK.** The flat-slot reasoning (B) is right and matches E4/E5. Treating MANIFOLD d = 1 `None` as a cost shortcut,
  not a correctness case, is the right framing. Everything else is unchanged and verified in rounds 1–4.

## 15. Everything else

- **m38 — MINOR (O-7): the frontend retraction needs the shared companion.** One retraction for both modes is correct
  (the coordinator confirms). No finding beyond noting that the frontend `LocalModel.retract` must pass
  `geometry_aux` on shared geometries (C.13 says so), and T18 should cover a shared model's `retract` equal to the
  backend's.
- **n21 — NIT: CHANGELOG precision for the converged line.** C.14's converged-line suffix ("g₀ = zero tensor") is the
  only GN-visible text change. Its doctest at `optimizer_display.py:171` is unchanged because hand-built rows carry
  `g0_reference=None`. Good. Just ensure the CHANGELOG line names it as a display change, not a numeric one.

---

## Decisions

| # | verdict |
|---|---|
| Settled I-1…I-25 (round 5) | Recorded; not reopened. I-4 needs B1's naming fix *within* the ruling: the rename stands, and only the mode field's name changes |
| O-1…O-5 | Appropriately open (measurements) |
| O-6 | Broaden from "the seed" to "the Haar-draw mechanism and its exact recipe" (M6). Nick's call, with F30–F32 as the evidence |
| O-7 | Closed: one retraction |

## What must change for AGREE

1. **B1.** Rename the frontend `LocalModel`'s mode field, or derive it (recommended: `is_exact` from `curvature`).
   Update C.13, T18 and T28.
2. **M5.** Extend `AXIS_SIZE` (`l, n, x, z`) or respell the new strings; verify through the repo's oracle.
3. **M6.** Specify the verified, rank-independent Haar recipe (per-mode `N×N` or per-mode seeds); re-run `c37b`; add
   the nesting and independence T26 cases; widen O-6 to the mechanism.
4. **M7.** Re-scope T25 to the frame claim; move the GN contrast to the records.
5. **MINORs:**
   - m31: the cost bands;
   - m32: the reference skip condition and the CG reference record;
   - m33: no error on a vanishing *default* zero reference;
   - m34: required vs duck-typed `zero_point_frame`;
   - m35: `test_optimizers.py:238`;
   - m36: §L.3's import reasoning;
   - m37: the prose citation;
   - m38: the shared `retract` test.
   Fix each, or list it for Nick.
6. **NITs** n18–n21 can be fixed at implementation; noting them in the plan is enough.

VERDICT: REVISE
