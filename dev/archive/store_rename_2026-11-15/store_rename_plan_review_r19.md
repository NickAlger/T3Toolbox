# The store rename plan — critic review, round 19 (confirmation)

Reviewed: §1.2a's corewise site list, K4 and the "Critic r18" CHANGES bullet.

**r18 MINOR, resolved.**
- **`:312`, `:496` and `:3252` are listed with their rewrites.** `_curvature_oracles.py:312` and `:496` become
  `U = G`, and `:496`'s `mu O sigma^` becomes `mu U sigma^`. `sampling_derivatives.py:3252` (`# Q = O = G`) becomes
  `R = U = G`.
- **The symmetric note's sentence is listed with a full template replacement.** `symmetric_probe_derivatives.tex:556–559`
  reads "at the frame $(\mathbf B,\mathbf G,\mathbf G,\mathbf G)$ (the substitution $\mathbf B\to\mathbf D$,
  $\mathbf G_i\to\mathbf U,\mathbf L,\mathbf R$): the manifold operation requires an orthogonal frame, the corewise one
  works for any $\mathbf B$, $\mathbf G$". It replaces the old "$\mathbf U$ no longer required orthogonal" clause, and
  the R5 hand edit uses it.
- **K4 bans the single equality `\b[OPQ] = G\b`.** My own grep of `t3toolbox/`, `tests/` and `docs/` returns nine
  sites (`probing.py:1457, 1721`; `sampling_derivatives.py:2464, 2989, 3252`; `_curvature_oracles.py:312, 397, 496,
  523`), and every one is on the list.

No MAJOR or MINOR finding remains.

VERDICT: AGREE
