# The store rename plan — critic review, round 12 (Nick's decisions)

Reviewed: draft 12 (1,064 lines), the changed parts: §1.3's D-2 helper, §3.2 (`sh`), the extended K8, K13, K14, K16,
and §8.

**Sound.**
- D-2 form (c)'s helper: every 2026.4.0 call passes a `str` and is refused; the `'O'` message is kept; the canonical
  tuple is a hashable jit static.
- D-4, D-5 (`Psi` in code), D-6 (`B` only in new text), D-9, D-10 and D-11 are applied consistently.
- K13 skips fences at R2/R3 and checks them from R4a.
- K14 names items 1–12, and K16 lists the LaTeX width forms for sign-off.

## MAJOR

**Q1. `sh` collides in the very modules that will carry it, so D-1 was decided on a false claim.** §3.2 says "`sh`
collides with nothing as an operator; the uniform tests' shape local `sh` is a NAME token". But `sh` (with `shd`) is
the **σ̂ (sigma-hat) local** of the jet kernels and their twin:
- `sampling_derivatives.py`: 19 tokens (`:1693–1694, 2159, 2167, 2201–2204, 3188–3220`);
- `apply.py:493–542`: 11 tokens (`for mu, O, sh in zip(mus, down_tt_cores, sigma_hats)`);
- `tests/_jet_trs_twins.py:213–215`: 3 tokens.

`sampling_derivatives.py` is where the 41 shift lines will read `sh^r nu`. So `sh` would mean σ̂ in the code and the
shift in the comments a few lines away. There are also the shape locals `sh` in `backend/fitting.py:162` and
`manifold.py:87`. The plan must state the collision and put it back to Nick. His options:
- (i) accept it, code against prose, with one sentence where both appear;
- (ii) rename the σ̂ locals (e.g. `s_hat`): a new listed rename, which needs his OK under §1.5 item 1 (Greek-derived
  names);
- (iii) spell the operator otherwise (e.g. `Sh`/`Shᵀ`, or `shift`/`shiftᵀ`).

**Q2. K8's new bare-`O` pattern `\bO\b(?!\()` fails on names that must stay, and K15 forbids renaming them.** That is a
gate deadlock, and an invitation to over-reach. Matches that are not frame cores:
- **The oracle module alias `O`:** `tests/backend/test_t3_orthogonalization.py:13` (`… t3_orthogonalization as O`),
  `oracle_sweeps/sweep_models.py:6`, `sweep_exact.py:29` and `sweep_optimizers.py:7` (`oracle as O`), and every
  `O.…` use.
- **The order count `O`:** `tests/test_uniform_probing.py:213–219` (`O = _ORDER + …`).
- **The interpreter flag `-O`:** `tests/test_tucker_tensor_train.py:3113, 3127`.
- **"I/O":** `optimizer_display.py:12`, `glossary.md:115`, `uniform_equivalence_contract.md:95`.
- **Author initials "O.":** "Ghattas, O." in `probing.py:94, 201, 237, …`, `tucker_tensor_train.py:291, 4691` and
  `user_guide.rst:448, 494`; "Koch, O." at `:514`; the three notes' bibliographies (`shared_t3_math.tex:766`,
  `padded_frame_cone.tex:487`, `ttm_t3m_ht_note.tex:344`).

Fix: add these to §1.5 (and `dnt.txt`) and to K8's carve-outs. Alternatively, restrict the bare-`O` pattern to
frame-core contexts (`O[`, `O.shape`, `for O in`, frame tuples, `\mathbf O`, "the O core").

VERDICT: REVISE (2 MAJOR)
