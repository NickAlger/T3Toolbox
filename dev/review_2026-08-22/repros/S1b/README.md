# S1b — the uniform frame on a numerically rank-deficient train (open; Nick working it on paper)

Run each with `PYTHONPATH=<repo>` from outside the repo.

- `s1b_cases.py`  — the six structural/numerical rank combinations; only "structurally minimal AND numerically
  deficient in both families" (the `resize` continuation warm start) loses a direction, and it is a genuine
  rank deficiency of the masked real block (one lost tangent direction), not a skewed basis.
- `s1b_hist.py`   — the same check at v2026.0.0 / v2026.1.0 / HEAD: identical; it never worked (the uniform
  frame dates from July; the April research-project code is the ragged sweep, which has no padding).
- `s1b_steps.py`, `s1b_pin2.py` — the four sweep steps by hand (all orthonormal) and the per-core masked Gram
  terms of the library frame (the last down core's mode slot 2 lives entirely in padded rR slots).
- `s1b_walk.py`   — the 3-core walkthrough: the last core's down-step unfolding (9 x 3, rows = (rL, rR) slots),
  its zero singular value, and LAPACK's completion `e_(a=0,b=2)` landing in a padded row; the ragged twin's
  3 x 3 unfolding forces the completion into the real row `a=2`.

Mechanism in one line: the SVD sorts the sigma > 0 directions ahead of the padding, which is what the prefix
masks rely on -- but the sigma = 0 block is one degenerate eigenspace that mixes the real-but-deficient
directions with the padded ones, and the uniform layer lets the completion land in the latter.
Candidate fix sketched in dev/HANDOFF.md (masked orthonormal completion after each uniform SVD step);
Nick suspects a more elegant solution -- pending.
- `s1b_t3svd.py`, `s1b_t3svd_sweep.py` — the same question for the uniform T3-SVD and the retraction: over 60
  random zero-padded continuation starts the tensor is exact in all 60 (t3svd and retract), the FRAME is
  non-orthonormal in 60/60, and the t3svd OUTPUT carries a padded completion (not left-orthogonal / masked
  block rank-deficient) in 40/60 -- the same mechanism, harmless for the tensor because a completion column
  is multiplied by sigma = 0, harmful only once a consumer reads the column itself as a direction (the frame).

## Prior art (web survey, 2026-08-23)

Nobody relies on the SVD's null-space completion to grow a basis; every rank-growing method builds the new
directions explicitly, or pads and never asks the padded slots to be orthonormal.

- **TeNPy `MPS.enlarge_chi`** (tenpy/networks/mps.py): zero-fills the `vR` leg, then for the `vL` leg draws
  RANDOM rows, Gram-Schmidt-projects them against the existing rows (`extra_B - (extra_B B2^†) B2`, warning
  if the remainder norm < 1e-12), QR-orthonormalizes the block and concatenates -- "right-canonical form,
  representing the same state, with the additional singular values being exactly zero". Its docstring adds:
  choose the extra directions sensibly, "not just into a random direction". This is the explicit masked
  completion (candidates → project → orthonormalize), inside the real dimensions by construction.
- **Rank-adaptive BUG integrator** (Ceruti, Kusch, Lubich 2022, arXiv:2104.05247, Sec. 2): augment
  `(K(t1), U0)` and take an orthonormal basis of its range "e.g. by QR decomposition" (2r columns), Galerkin
  step, SVD-truncate; the result "still holds true for a different choice of orthonormal basis" (the basis
  is gauge), and the scheme is "robust to the presence of small singular values".
- **Riemannian rank-adaptive TT completion** (Vermeylen, Vandereycken, Absil 2024, arXiv:2402.12182,
  Sec. 3.1): no zero padding at all -- the rank increase is a step along the NORMAL part of the tangent cone,
  `X'_{1:i-1} · U_i · V_{i+1} · X''_{i+2:d}` with `U_i ⟂ X_i` and `V_{i+1} ⟂ X_{i+1}` obtained by projecting the
  gradient (Prop. 3.1), and the retraction (45) appends them as new core columns/rows. I.e. the "completion"
  IS the normal-space gradient direction -- which is what a zero-padded warm start followed by one gradient
  step computes implicitly, and why that warm start works (Nick's continuation rationale, in their language).
- **DMRG subspace expansion** (Hubig, McCulloch, Schollwöck, Wolf 2015, arXiv:1501.05504, Sec. IV):
  append a residual-derived expansion term `P_i` (real, structured) to the site tensor, accept that
  "orthonormality of B_{i+1} is lost" and restore it "via singular value decomposition as usual"; contrasted
  with White's density-matrix perturbation (noise) -- structured enrichment vs jitter.
- **Vendors / frameworks**: cuSOLVER `gesvdjBatched` (≤ 32x32) and friends need equal sizes and say nothing
  about padded semantics; PyTorch's `linalg.svd` docs state that the trailing singular vectors "can be
  arbitrary bases of the corresponding subspaces" and that "different hardware and software may compute
  different singular vectors"; TensorLy `pad_tt_rank` zero-pads "without changing the reconstruction" and is
  silent on orthogonality; JAX's ragged-data guidance is pad + mask, nothing SVD-specific.
- `s1b_gsvd.py` — Nick's GSVD idea checked on the walkthrough's 9x3 unfolding: the GSVD of the pair
  `(M^T, I_real)` (computed as the CS decomposition of the stacked `[M^T; I_real]`, rank-revealing) and the
  augmented SVD of `[M | eps I_real]` give the SAME basis: the sigma > 0 vectors exactly (overlap 1 -- the
  pair `(M M^T, P_real)` commutes), the real-complement completion identical (overlap 1.000, in the real row
  `(a=2,b=0)`), the padded rows as the common null space. The augmentation is the GSVD with `B` scaled by
  `eps`, computed through one ordinary (batchable) SVD.
