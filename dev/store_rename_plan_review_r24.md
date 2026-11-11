# The store rename plan — critic review, round 24 (draft 24)

Reviewed: the draft-24 diff against `61138042`, which applies Nick's ruling "shape decides the letter".

**Sound, and checked against the code.**
- An `ast` scan of all 29 renamed functions (apply, probing, sampling_derivatives, the tv/t3_linalg/utv locals,
  manifold's six transposes, the twin, and the four test modules) finds each listed old name present and no image
  already bound. `_bare_to_dense`'s `tkv`/`ttv` handles the one clash.
- The four held locals are now renamed by shape, and their shape comments support it:
  - `tv_orthogonal_gauge_projection`'s `dV` iterates `tt_variations`, so `H`;
  - `t3_linalg.py:682–694` and `utv_operations.py:600–615` have `dB` of shape `(nU, N)` new and `dG` of shape
    `(rL, nD, rR)` new;
  - `probing.py:535–545` has `B, dB` as the down core and its variation.
- The kept `D` forms are `D`-shaped: the deta kernels' `Dd` from `up_tucker_cores_dot` `(nUi, Ni)` old, so `(nD, N)`,
  and the tied `Ddot` `(n_g, N)`.
- K17's shape test is precise. `(rL…, nD…, rR…)` excludes the left `(rL, nD, rL)` and right `(rR, nD, rR)` cores, and
  `(nU…, N…)` excludes `D`'s `(nD, N)`.

## MINOR

1. **Two stale rule sentences contradict the shape rule.**
   - §1.2a still opens its site list with: "The rule for the `U` family … A variation or direction of a frame's down
     core (**a tangent's Tucker variation `V`, a transpose's Tucker block**, a kernel whose core is the frame's `D`)
     takes `D` forms. Every site below passed R1's freeness check in the draft-16 scan". Under the shape rule the first
     two now take `V`/`H`, as the lists below it say. A literal implementer meets the contradiction first. Rewrite the
     sentence: variation-shaped → `V`/`H`, and a `D`-shaped direction of a frame's down core → `Dd`/`Ddot`. Cite
     draft 24's scan.
   - K17 still says "Derivative forms (`dD…`, `Ddot`, `Dd`) are allowed by the rule". Since "no `dD` remains", say
     "`D`-shaped derivative forms (`Dd`, `Ddot`)". The shape test then needs no exception.

## NIT

- The "Kernels lettered by their keywords" bullet ends with the draft-16 re-check's test ("… or its variation → `D`
  form"). Mark it as superseded by the shape rule, or drop it.

VERDICT: REVISE (0 MAJOR, 1 MINOR)
