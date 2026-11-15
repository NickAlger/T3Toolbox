# The store rename plan — critic review, round 15 (draft 16)

Reviewed: the draft-16 changes against draft 15: §1.2a (D-12), the R1 freeness check, the two dispositions, and §3.1,
§3.3, §3.4, D-6 and D-12.

**r14 MINOR 1, resolved.**
- R1 now runs an `ast` freeness check per enclosing function or doctest block, with nested-def parameters excepted.
- `vertical_direction`'s gauge matrix becomes `M`. `M` is free there, and `Bg` was avoided because `tied_lift` uses it.
- `test_derivative_kinds` uses `dB_dir`.
- I re-ran my own `ast` scope scan over §1.2a's B and D sites in `_curvature_oracles.py`, `tv_operations.py`,
  `utv_operations.py`, `sharing.py`, `test_probe_derivatives.py`, `test_curvature.py` and `test_sharing.py`. Every
  listed function exists, holds the listed tokens, and has no clash.

**r14 MINOR 2, resolved in substance.**
- One stated rule: B forms for a point's own raw cores and their raw-coordinate directions, gradients and Hessian
  blocks; D forms for a frame down core and its variations.
- §1.2a applies it function by function, code and doctests.
- `_curvature_oracles` now reads `B, G` beside `Bd, Gd`. `tv_corewise_lift` gives `Bdot`.
- The transposes' Tucker blocks (tangent `V`) give `dD`.
- The meeting point `θ = (D, L)` / `θ̂ = (Bdot, Gdot)` is stated openly for Nick (D-12).

## MINOR

1. **`utv_corewise_retract` is classified against the rule §1.2a states.** Its `dU` is, in its own words, "free core
   perturbations (dU, dG)" added to "the point's Tucker core U" (`utv_operations.py:376–396`). That is a raw
   perturbation of the point's own cores, which §1.2a lists as a `B` case ("a corewise lift, a corewise transpose or
   Hessian, a raw perturbation"). The stated reason for `dD`, that "it adds to the corewise frame's `down_sc`", holds
   for every corewise operation: the corewise frame's down slot always holds the point's own cores.
   - As written, the uniform corewise retraction would read `dD` while the corewise lift (`Bdot`), the corewise
     Hessians (`dB`) and the COREWISE doctest (`B, G = x.data`; `dB`) read `B`.
   - Classify it `dB`, with its comments' "the point's Tucker core U" → `B`. Or, if Nick prefers `D` for anything
     added in the corewise frame's slot, restate the rule so that it says so, and apply it to the lift as well.

VERDICT: REVISE (0 MAJOR, 1 MINOR)
