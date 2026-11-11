# The store rename plan — critic review, round 16 (confirmation)

Reviewed: §1.2a and the draft-16 CHANGES lines.

- **r15 MINOR, resolved.**
  - `utv_corewise_retract` is under the B forms (`dU→dB`).
  - Its comments go by hand: "the point's Tucker core U" → `B`, `(U, G)` → `(B, G)`, `(dU, dG)` → `(dB, dG)`.
  - The corewise frame tuple becomes `(D, G, G, G)` by tier D.
  - The site is free in its function.
- **The re-check against the stated test is consistent.**
  - `fitting.py:351` is the kind's generic `transpose(residual, sample, frame, sweep)`, which is frame-parametrized
    (code read). It takes `D` with the other frame-generic kernels, and `corewise_hessian` (`:374, 380, 389`) keeps `B`.
  - The new "Frame-generic kernels" bullet states openly why `_deta_tilde_tangent_step`, `_tangent_z_jets_step` and
    `_deta_tilde_tangent_jets_step` take `Dd`/`dD`: their core is named for the frame's down store and serves both
    geometries.
  - It also states why their oracle twin `deta_tilde_tangent_jets_trs` takes `Bd`: only the corewise oracles call it.

## NIT

- The twin (`Bd`) and its kernel (`Dd`) now read differently in a side-by-side comparison. The reason is given, so
  this is for the reviewer to expect, not to fix.

VERDICT: AGREE
