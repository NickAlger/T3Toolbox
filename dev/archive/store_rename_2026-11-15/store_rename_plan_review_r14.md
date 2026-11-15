# The store rename plan — critic review, round 14 (draft 15)

Reviewed: the draft-15 diff against `9363c246` (draft 14). It covers the B rows (D-6 in code), the scan's new D rows,
the shift names restored, the exact-Hessian `\mathcal{R}`, the notes' B/D classification, §1.5 items 8/11, K14 and
the smaller edits.

**Checked and sound.**
- `zB`/`zBd` are free in `t3_linalg.py`. The `BU` at 673/690 (660–760, a frame's stores) and at 837–996 (`B_z Uᵀ`, a
  T3's) are told apart by range.
- `core_dot(gA, gB, nC)` in `test_tucker_tensor_train.py` is a sibling nested function that reads no outer name, so the
  outer `gU→gB` and `u→b` do not change its meaning.
- `Uw`/`Uox` reclassified as SVD factors, matching the code.
- The D rows at `test_weighted.py:401–434` and `test_sharing.py:598–760` match the code's `zip(frame_…[1],
  frame_…[0])`.
- The restored shift names (item 8) with `sh`/`shᵀ` in comments, and `sig_hat` kept.
- The `\mathcal{R}` dispositions, and K14's rule for images that spell a listed name.

## MINOR

1. **The new B and D rows have no per-scope image-collision rule, and one site collides.** The old catch-all had "a
   per-function collision check". Its replacement (`letter_sites.md`, the B and D rows, §3.4's dispositions) says only
   that a §1.5 meaning is excluded. Running an `ast` scope scan on the B and D rows of draft 15:
   - `tests/backend/_curvature_oracles.py:237–255` (`vertical_direction`): θ's cores `U` (`:241`, listed for
     classification) would become `B` in the same function as the TT gauge matrix `B = rng.standard_normal((r, r))`.
     The docstring there says "TT gauge `G_i -> G_i B`".
   - `test_optimizers.py:274–319`, `dU→dB` against the data batch `dB`: the writer already flags this one.
   - Every other B row is clear (`test_tucker_tensor_train`'s `gB`/`b` meet only `core_dot`'s own scope).

   Add to R1's gate: every row's image is unbound in its function in the old tree (nested-def parameters excepted), or
   it takes another spelling.
2. **A core and its direction can end up on different letters in one function.** Tier C's `U`-family row gives the
   directions `D` forms in the frame, tangent, sampling and curvature modules (`Ud→Dd`, `Udot→Ddot`, `dU→dD`). Draft
   15 gives a T3's own cores `B` ("definitely should not be D", Nick). Where the cores and their directions sit
   together, the result reads `B` beside `Dd`:
   - `_curvature_oracles.py:524–525` would read `B, G = core_pair` / `Dd, Gd = core_directions`;
   - `backend/fitting.py:374–379` has `theta_hat = (Udot, Gdot)` "at `base_point` … the raw cores the directions
     vary";
   - `tv_operations.py`'s and `sharing.py`'s `Udot` (`Udot_i = (S_i^T)^{+eps} V_i`, the raw factor direction).

   State one rule, per function, and list the functions under it. By Nick's ruling, a direction of a T3's own raw cores
   takes `B` forms (`Bd`, `Bdot`, `dB`). `D` forms stay where the variable is a frame store or its variation (the
   kernels' `down_tucker_cores`, the tangent's `dD_i = <U_i, core_env_i>`). This is Nick's call. Today the plan
   decides it implicitly, module by module.

VERDICT: REVISE (0 MAJOR, 2 MINOR)
