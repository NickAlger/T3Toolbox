# S1 review (round 1): pinned primitives, `oversampled_ranks`, HISTORICAL 1-2

Reviewer, 2026-11-02. Scope: the working-tree diff (`backend/linalg.py`, `backend/ranks.py`, `test_linalg.py`,
`test_contractions_interpreter.py`, `test_dispatch.py`) plus the new `tests/backend/test_padded_start.py`, against
plan §4.0-4.1, §5.1, §5.4 (rank function), §6 T1 / T5(h) / T10 / T12, §8 S1, §12.

## Overall

The slice is solid. The mathematics matches §4.1 step for step, the code is short and in house style, and the
tests cover every T1 item. There are no BLOCKER or MAJOR findings. Two MINORs need fixing: one docstring sentence
about the rank-deficient case is wrong (a probe shows it), and the "any seeds" claim is untested on jax, the
backend S3's jitted constructor will run on. Three NITs follow.

## Findings

1. **MINOR: `t3toolbox/backend/linalg.py:950-953` (`pinned_leading_columns` docstring).** It says that when
   `Aperp` has rank `< k`, "the trailing columns are the Householder completion (`A` has no information there)".
   That is not the mechanism. The seeds are the SVD's columns, so the trailing seeds are LAPACK's left singular
   vectors for zero or roundoff-level `σ`. They are arbitrary unit vectors, generally with components in
   `span(P)`. The QR then orthogonalizes them, so the result depends on roundoff in `Aperp`, not on `Aperp`
   itself. Probe: `n=8, s=2, k=4`, `A` of rank 1 off `P`. Adding in-span content `P·X` to `A` (mathematically the
   same `Aperp`) leaves column 0 unchanged (projector distance 6e-16) but moves the trailing 3-dim subspace by
   projector distance **1.4**. The trailing columns also differ from the zero-seed Householder completion by
   O(1) (0.77). Validity is unaffected: residual ≤ 1.5e-15 in every case. This is plan R5's "LAPACK-dependent
   choice", and the plan's own §4.1 wording (line 1020) has the same inaccuracy.
   **Change wanted:** reword roughly as follows: "otherwise the trailing columns come from the SVD's trailing
   singular vectors (zero or roundoff `σ`), orthogonalized against `P` and the leading columns by the QR. That is
   a valid choice, deterministic for a given input and backend, but `A` does not determine it: roundoff-level
   changes to `A` (e.g. its content inside `span(pinned)`) can change it by O(1)." Add one line to the S1 handoff
   flagging, for S3-S5 tests, that gauge-invariant or cross-layer projector comparisons (T3(h)/(j), T5(f),
   T4(e)) must use hints with at least `k` directions per slot. Rank-deficient hints may only be asserted valid
   (T3(a)). A one-line erratum at plan line 1020 is the orchestrator's call.

2. **MINOR: `tests/test_dispatch.py:146-164`, or `test_linalg.py`.** The "valid for any seeds" claim (zero seeds,
   too few seeds, seeds in `span(pinned)`) is tested only on numpy. Judgment call 5 rightly excludes
   cross-backend *equality* for these cases, but it also drops jax *validity*. S3's jitted `fv_padded_frame` runs
   exactly this path on jax: `pinned_leading_columns` pads zero seeds whenever the hint's content has fewer than
   `k` columns. Jax's QR is a different implementation (and a different one again on GPU). My probe confirms it
   works (x64, eager and jit, `C=(2,)`, zero / few / in-span / none seeds, both primitives: worst residual
   1.5e-15; float32: 1.0e-6). **Change wanted:** add a small subtest to `test_pinned_primitives_dispatch`. Under
   jit with x64, for all-zero, too-few and in-span seeds, assert the completion residual of both primitives is
   `≤ 1e-14`. Make no comparison with numpy. This is about jax, so it is within the house "jax tests only when
   about jax" rule.

3. **NIT: `t3toolbox/backend/linalg.py:867`.** "Column ``j`` of the result is the first seed's direction ..."
   should say "seed ``j``'s direction ...". The test `test_columns_follow_the_seeds_in_order` says it correctly.

4. **NIT: `t3toolbox/backend/ranks.py:186` / docstring.** I accept judgment call 2 (keep input boundary bonds >
   1, "never below the input" literally). Worth one docstring sentence: the interior caps are computed as for
   boundary bonds 1, which is conservative. Without it, a reader of `((3, 4), (2, 2), (2, 2, 2), p=1) →
   ((3, 3), (2, 3, 2))` may wonder why the interior bond does not use the boundary 2. Optional.

5. **NIT (for the orchestrator, not the implementer): commit trailer.** Judgment call 7 names Claude Fable 5.1 per
   HANDOFF. Nick's rule is the model *actually* in use by the committing session. Confirm at commit time rather
   than copying the draft.

## Checked and found correct

- **`pinned_completion`** (§4.1 steps 1-5, §5.1). `k == 0` returns early with `result_type` dtype. The `s + k > n`,
  stack/`n` mismatch and negative/bool/non-int `k` checks are structural and come before any arithmetic. Missing
  seeds become zero columns, and a seed surplus is truncated. It does one batched reduced QR of
  `[pinned | S]`; the sign rule uses float literals cast to `Q`'s dtype, and the output is columns `s..s+k-1`.
  Mathematically the Householder `Q` is orthonormal, and its first `s` columns span `pinned` (full column rank),
  so the result is valid for every seed set with no threshold. Confirmed numerically, and the `s = 0` Haar
  identity holds (T1(b)).
- **`pinned_leading_columns`.** The projection uses the two `contract` calls with HISTORICAL `''` entries (letters
  i3 a2 b4, as §4.10). It takes the thin SVD and feeds the completion. The `m = 0` branch and the `m > n` and
  `m < k` paths (via `min(n, m)` columns plus zero padding) are right. The planted gap recovery is tested to
  `1e-12` projector distance.
- **Dispatch.** `tree_contains_jax` → `get_backend(False, ...)` matches the module's existing pattern (no `use_jax`
  parameter). Mixed numpy/jax inputs give jax, and int seeds promote to float. The new `linalg →
  contractions` import is acyclic (`contractions` imports only `common`).
- **`oversampled_ranks`.** It returns `max(input, compute_minimal_ranks(+p))`. `compute_minimal_ranks` is a sequence
  of monotone min/+/* steps, so for a minimal input the max is a no-op and the output is minimal (an idempotent
  reduction). Sharing equality holds because both arguments of the max are group-equal (the unequal input is
  rejected by `compute_minimal_ranks`). I hand-checked the `(3,10,4)` doctest values. It returns Python-int
  tuples. Validation is in judgment call 3. I accept it.
- **Tests.** T1(a)-(d) are all present, with explicit tolerances (`1e-14` orthogonality, `1e-12` projector, `1e-14`
  stack equality, `1e-5` float32), and none use `allclose` defaults. The only bitwise assertion is
  `Q == completion(P, seeds[:, :k])`. That is the same computation on identical input (a structural fact: surplus
  seeds are never read), so it is acceptable. The `k = 0` "no factorization" mock does intercept (verified: with
  `np.linalg.qr` patched, a `k = 1` call raises). T5(h) covers `p = 0`, never-below (random non-minimal inputs),
  sharing (random groups, shared-minimal) and errors. Deferring the frontend and geometry-member agreement rows
  to S5/S9 is correct (judgment call 1). T10: numpy→numpy, eager jax→jax, jit→pure jax equal to numpy at
  `1e-12`, jitted `k = 0` shape. T12: the two strings, `''`.
- **Doctests.** They are seeded and self-contained, with real output, and the bounds are wide enough for both
  numpy generations. 15 pass.
- **Runs.** `test_linalg`, `test_padded_start`, `test_ranks`, `test_contractions_interpreter`,
  `test_contractions_sharding` and the new dispatch row: 82 passed, 5,108 subtests. Doctests: 15 passed.
- **No behaviour change elsewhere.** The diff is additive apart from the `linalg` module docstring and the
  `test_linalg` module docstring. The untracked `dev/check_t3svd_retraction_second_order.py` and
  `release_notes_2026.3.0.md` are not part of this slice. Do not stage them.
- **Judgment calls 1-6.** All reasonable. Call 5 is right about equality; finding 2 asks only for validity.

VERDICT: REVISE
